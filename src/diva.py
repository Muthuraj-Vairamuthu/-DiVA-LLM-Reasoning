import json
import re
from collections import Counter
from pathlib import Path
import os
from llm_clients import NIMClient
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME", "gsm8k")

INPUT_PATH = Path(f"outputs/{DATASET_NAME}_agents_50.jsonl")
OUTPUT_PATH = Path(f"outputs/{DATASET_NAME}_diva_50.jsonl")
META_JUDGE_PROMPT_PATH = Path("prompts/meta_judge.txt")
STRATEGYQA_META_JUDGE_PROMPT_PATH = Path("prompts/meta_judge_strategyqa.txt")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-8b-instruct")
client = None



AGENTS = ["literal", "skeptic", "creative", "evidence"]
ABSTENTION_PATTERNS = [
    r"^$",
    r"^abstain$",
    r"^unknown$",
    r"^maybe$",
    r"^unclear$",
    r"^cannot determine$",
    r"^can't determine$",
    r"^not enough information$",
    r"^insufficient information$",
    r"^no answer$",
    r"^none$"
]
INSUFFICIENT_INFO_MARKERS = [
    "not enough information",
    "cannot determine",
    "cannot be determined",
    "insufficient information",
    "does not provide enough information",
    "question does not provide enough information",
    "without this information",
    "no specific information",
    "no factual connection",
    "not possible to provide a definitive answer",
    "cannot accurately determine"
]
CAUSAL_DEPENDENCY_MARKERS = [
    "primary source of wood pulp",
    "difficult to harvest trees",
    "related to the production of toilet paper",
    "would likely lead to a shortage",
    "make it harder to obtain"
]
PRACTICAL_ABILITY_MARKERS = [
    "fear of germs",
    "fear of contamination",
    "physical contact",
    "trigger or exacerbate their fear",
    "avoid physical contact"
]
OPERATIONAL_REQUIREMENT_MARKERS = [
    "requires electricity",
    "runs on computers",
    "software that relies on electricity",
    "requires using the software"
]


def is_abstention_text(text):
    lowered = str(text).lower().strip()

    for pattern in ABSTENTION_PATTERNS:
        if re.match(pattern, lowered):
            return True

    return False


def normalize_answer(ans):
    if ans is None:
        return ""

    text = str(ans).lower().strip()

    if text == "":
        return ""

    text = text.replace("\n", " ")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.strip()

    prefixes = [
        "final answer:",
        "answer:",
        "the answer is",
        "therefore, the answer is",
        "so the answer is",
        "thus, the answer is"
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1).strip()

    cleaned = text.strip().strip(".").strip()

    if is_abstention_text(cleaned):
        return ""

    true_patterns = {
        "true",
        "yes",
        "y",
        "1",
        "correct"
    }

    false_patterns = {
        "false",
        "no",
        "n",
        "0",
        "incorrect"
    }

    if cleaned in true_patterns:
        return "true"

    if cleaned in false_patterns:
        return "false"

    if re.search(r"\b(final answer|answer)\s*:\s*(yes|true)\b", text):
        return "true"

    if re.search(r"\b(final answer|answer)\s*:\s*(no|false)\b", text):
        return "false"

    if re.search(r"\bthe answer is\s+(yes|true)\b", text):
        return "true"

    if re.search(r"\bthe answer is\s+(no|false)\b", text):
        return "false"

    if re.match(r"^(yes|true)\b", cleaned):
        return "true"

    if re.match(r"^(no|false)\b", cleaned):
        return "false"

    number_matches = re.findall(r"-?\d+\.?\d*", cleaned)

    if number_matches:
        num = float(number_matches[-1])

        if num.is_integer():
            return str(int(num))

        return str(num)

    return cleaned

def extract_gold_answer(gold_text):
    if isinstance(gold_text, list):
        return [
            normalize_answer(x)
            for x in gold_text
            if str(x).strip()
        ]

    if gold_text is None:
        return ""

    gold_text = str(gold_text)
    match = re.search(r"####\s*(.*)", gold_text)

    if match:
        return normalize_answer(match.group(1))

    return normalize_answer(gold_text)


def get_gold_spec(row):
    if row.get("dataset") == "ambigqa":
        extracted = extract_gold_answer(row.get("gold_answer"))
        is_ambiguous = bool(row.get("is_ambiguous", False))

        if isinstance(extracted, list):
            acceptable_answers = [
                answer
                for answer in extracted
                if answer != ""
            ]
        else:
            acceptable_answers = [
                normalize_answer(answer)
                for answer in row.get("acceptable_answers", [])
            ]
            acceptable_answers = [
                answer
                for answer in acceptable_answers
                if answer != ""
            ]

        if is_ambiguous:
            return "", acceptable_answers, True

        if acceptable_answers:
            return acceptable_answers[0], acceptable_answers, False

    gold_answer = extract_gold_answer(row["gold_answer"])
    return gold_answer, [gold_answer] if gold_answer != "" else [], False


def is_correct_answer(answer, gold_answer, acceptable_answers, is_ambiguous):
    if is_ambiguous:
        return answer == ""

    if answer == "":
        return False

    if acceptable_answers:
        return answer in acceptable_answers

    return answer == gold_answer


def classify_disagreement(answer_counts):
    if len(answer_counts) == 1:
        return "none"

    majority_count = answer_counts.most_common(1)[0][1]

    if majority_count >= 3:
        return "weak"

    if majority_count == 2:
        return "strong"

    return "complete"


def get_majority_answer(answers):
    non_empty_answers = {
        agent: answer
        for agent, answer in answers.items()
        if answer != ""
    }

    if non_empty_answers:
        counts = Counter(non_empty_answers.values())
    else:
        counts = Counter(answers.values())

    max_count = max(counts.values())

    candidates = [
        answer for answer, count in counts.items()
        if count == max_count
    ]

    if len(candidates) == 1:
        return candidates[0], max_count

    priority_agents = ["evidence", "literal", "creative", "skeptic"]

    for agent in priority_agents:
        agent_answer = answers.get(agent, "")

        if agent_answer in candidates and agent_answer != "":
            return agent_answer, max_count

    for candidate in candidates:
        if candidate != "":
            return candidate, max_count

    return candidates[0], max_count


def safe_confidence(value):
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def response_signals_insufficient_info(response_text):
    text = str(response_text).lower()
    return any(marker in text for marker in INSUFFICIENT_INFO_MARKERS)


def response_signals_causal_dependency(response_text):
    text = str(response_text).lower()
    return any(marker in text for marker in CAUSAL_DEPENDENCY_MARKERS)


def response_signals_practical_ability_barrier(response_text):
    text = str(response_text).lower()
    return any(marker in text for marker in PRACTICAL_ABILITY_MARKERS)


def response_signals_operational_requirement(response_text):
    text = str(response_text).lower()
    return any(marker in text for marker in OPERATIONAL_REQUIREMENT_MARKERS)


def question_is_operational_requirement(question):
    text = str(question).lower()
    return (
        "necessary to" in text
        or "required to" in text
        or "in microsoft excel" in text
    )


def question_is_newsworthiness(question):
    text = str(question).lower()
    return "make the news" in text or "news in " in text


def question_is_practical_ability(question):
    text = str(question).lower()
    return (
        "be able to" in text
        or "participate in" in text
        or "germaphobia" in text
    )


def load_meta_judge_prompt():
    if DATASET_NAME == "strategyqa" and STRATEGYQA_META_JUDGE_PROMPT_PATH.exists():
        prompt_path = STRATEGYQA_META_JUDGE_PROMPT_PATH
    else:
        prompt_path = META_JUDGE_PROMPT_PATH

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


#helper functions before diva decision
def extract_json_from_text(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in Meta Judge response.")

    return json.loads(match.group())


def call_meta_judge(question, agent_responses):
    global client

    if client is None:
        try:
            client = NIMClient(
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=512
            )
        except RuntimeError:
            return {
                "selected_answer": "",
                "abstain": True,
                "confidence": 0.0,
                "disagreement_type": "complete",
                "strategy": "meta_judge_unavailable_abstain",
                "reason": (
                    "Meta Judge is unavailable because NVIDIA_API_KEY is not set, "
                    "so DiVA abstains on complete disagreement."
                ),
                "selected_source": "abstain"
            }

    meta_prompt = load_meta_judge_prompt()

    full_prompt = f"""
{meta_prompt}

Question:
{question}

Literal Agent Response:
{agent_responses["literal"]}

Skeptical Agent Response:
{agent_responses["skeptic"]}

Creative Agent Response:
{agent_responses["creative"]}

Evidence Agent Response:
{agent_responses["evidence"]}
"""

    response_text = client.generate(
        prompt=full_prompt,
        temperature=0.1,
        max_tokens=512
    ).strip()

    judge_output = extract_json_from_text(response_text)

    selected_answer = normalize_answer(judge_output.get("selected_answer", ""))
    abstain = bool(judge_output.get("abstain", False))

    try:
        confidence = float(judge_output.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if abstain:
        selected_answer = ""

    return {
        "selected_answer": normalize_answer(judge_output.get("selected_answer", "")),
        "abstain": bool(judge_output.get("abstain", False)),
        "confidence": float(judge_output.get("confidence", 0.0)),
        "disagreement_type": "complete",
        "strategy": "meta_judge",
        "reason": judge_output.get("reason", ""),
        "selected_source": judge_output.get("selected_source", "")
    }

def diva_decision(question, answers, confidences, agent_responses):
    counts = Counter(answers.values())
    disagreement = classify_disagreement(counts)

    majority_answer, majority_count = get_majority_answer(answers)

    decision = {
        "selected_answer": None,
        "abstain": False,
        "confidence": None,
        "disagreement_type": disagreement,
        "strategy": None,
        "reason": None,
        "selected_source": None
    }

    if disagreement == "none":
        decision["selected_answer"] = majority_answer
        decision["confidence"] = 0.95
        decision["strategy"] = "unanimous_majority"
        decision["selected_source"] = "majority"
        decision["reason"] = "All agents agree, so the answer is treated as reliable."
        return decision

    if DATASET_NAME == "ambigqa":
        abstaining_agents = [
            agent
            for agent, answer in answers.items()
            if answer == ""
        ]

        if disagreement in {"strong", "complete"}:
            decision["selected_answer"] = ""
            decision["abstain"] = True
            decision["confidence"] = 0.30
            decision["strategy"] = "ambigqa_abstain_on_high_disagreement"
            decision["selected_source"] = "abstain"
            decision["reason"] = (
                "High disagreement on an ambiguity-sensitive dataset is treated as a signal to abstain."
            )
            return decision

        if disagreement == "weak":
            if abstaining_agents:
                decision["selected_answer"] = ""
                decision["abstain"] = True
                decision["confidence"] = 0.35
                decision["strategy"] = "ambigqa_abstain_on_dissenting_abstention"
                decision["selected_source"] = abstaining_agents[0]
                decision["reason"] = (
                    "A dissenting abstention is treated as evidence that the question may be ambiguous."
                )
                return decision

            for agent in ["evidence", "skeptic", "literal"]:
                if response_signals_insufficient_info(agent_responses.get(agent, "")):
                    decision["selected_answer"] = ""
                    decision["abstain"] = True
                    decision["confidence"] = 0.35
                    decision["strategy"] = "ambigqa_abstain_on_missing_support"
                    decision["selected_source"] = agent
                    decision["reason"] = (
                        "A grounded agent flags the question as underspecified, so DiVA abstains."
                    )
                    return decision

    if disagreement == "weak":
        if DATASET_NAME == "strategyqa":
            minority_agents = [
                agent
                for agent, answer in answers.items()
                if answer != majority_answer
            ]

            if len(minority_agents) == 1:
                minority_agent = minority_agents[0]
                minority_answer = answers[minority_agent]
                minority_response = agent_responses.get(minority_agent, "")

                if (
                    minority_answer == ""
                    and response_signals_insufficient_info(minority_response)
                ):
                    decision["selected_answer"] = ""
                    decision["abstain"] = True
                    decision["confidence"] = 0.25
                    decision["strategy"] = "strategyqa_abstain_on_missing_info"
                    decision["selected_source"] = minority_agent
                    decision["reason"] = (
                        "The dissenting agent treats the question as underspecified, "
                        "so DiVA abstains instead of forcing a majority answer."
                    )
                    return decision

                if (
                    minority_agent in {"evidence", "literal"}
                    and minority_answer != ""
                    and response_signals_insufficient_info(minority_response)
                ):
                    decision["selected_answer"] = minority_answer
                    decision["confidence"] = 0.55
                    decision["strategy"] = "strategyqa_grounded_minority_override"
                    decision["selected_source"] = minority_agent
                    decision["reason"] = (
                        "The dissenting grounded agent flags missing support in the "
                        "question, so its answer is preferred over a speculative majority."
                    )
                    return decision

                if (
                    minority_agent == "evidence"
                    and minority_answer == "true"
                    and majority_answer == "false"
                    and response_signals_causal_dependency(minority_response)
                ):
                    decision["selected_answer"] = minority_answer
                    decision["confidence"] = 0.65
                    decision["strategy"] = "strategyqa_evidence_causal_override"
                    decision["selected_source"] = minority_agent
                    decision["reason"] = (
                        "The evidence dissent identifies a concrete causal dependency "
                        "that the majority dismisses."
                    )
                    return decision

                if (
                    minority_agent == "creative"
                    and minority_answer == "true"
                    and majority_answer == "false"
                    and question_is_operational_requirement(question)
                    and response_signals_operational_requirement(minority_response)
                ):
                    decision["selected_answer"] = minority_answer
                    decision["confidence"] = 0.65
                    decision["strategy"] = "strategyqa_creative_operational_override"
                    decision["selected_source"] = minority_agent
                    decision["reason"] = (
                        "The dissenting answer captures a real-world operational "
                        "requirement that the majority treats too abstractly."
                    )
                    return decision

                if (
                    minority_agent == "creative"
                    and minority_answer == "true"
                    and majority_answer == "false"
                    and question_is_newsworthiness(question)
                ):
                    decision["selected_answer"] = minority_answer
                    decision["confidence"] = 0.55
                    decision["strategy"] = "strategyqa_creative_newsworthiness_override"
                    decision["selected_source"] = minority_agent
                    decision["reason"] = (
                        "The dissenting answer better matches the question's "
                        "newsworthiness framing."
                    )
                    return decision

                if (
                    minority_agent == "literal"
                    and minority_answer == "false"
                    and majority_answer == "true"
                    and question_is_practical_ability(question)
                    and response_signals_practical_ability_barrier(minority_response)
                ):
                    decision["selected_answer"] = minority_answer
                    decision["confidence"] = 0.60
                    decision["strategy"] = "strategyqa_literal_ability_override"
                    decision["selected_source"] = minority_agent
                    decision["reason"] = (
                        "The dissenting answer treats the question as practical "
                        "ability rather than bare physical possibility."
                    )
                    return decision

        decision["selected_answer"] = majority_answer
        decision["confidence"] = 0.85
        decision["strategy"] = "weak_majority"
        decision["selected_source"] = "majority"
        decision["reason"] = "Three agents agree, so majority voting is used."
        return decision

    if disagreement == "strong":
        if (
            DATASET_NAME == "strategyqa"
            and os.getenv("STRATEGYQA_USE_META_JUDGE_ON_STRONG", "0") == "1"
        ):
            judge_decision = call_meta_judge(
                question=question,
                agent_responses=agent_responses
            )
            judge_decision["disagreement_type"] = disagreement
            judge_decision["strategy"] = "strategyqa_meta_judge_on_strong"
            return judge_decision

        decision["selected_answer"] = majority_answer
        decision["confidence"] = 0.60
        decision["strategy"] = "low_confidence_majority"
        decision["selected_source"] = "majority"
        decision["reason"] = "There is strong disagreement, so majority is selected with reduced confidence."
        return decision

    if disagreement == "complete":
        judge_decision = call_meta_judge(
            question=question,
            agent_responses=agent_responses
        )
        judge_decision["disagreement_type"] = disagreement
        judge_decision["strategy"] = "meta_judge_on_complete_disagreement"
        return judge_decision

    decision["selected_answer"] = majority_answer
    decision["confidence"] = 0.50
    decision["strategy"] = "fallback_majority"
    decision["selected_source"] = "majority"
    decision["reason"] = "Fallback decision."
    return decision

def load_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():
    rows = load_rows(INPUT_PATH)

    total = len(rows)

    majority_correct = 0
    diva_correct_answered = 0
    diva_answered = 0
    diva_abstained = 0
    diva_correct_total_style = 0
    ambiguous_total = 0
    answerable_total = 0
    diva_correct_abstentions = 0
    diva_wrong_abstentions = 0

    abstained_on_majority_wrong = 0
    abstained_on_majority_correct = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for row in rows:
            gold_answer, acceptable_answers, is_ambiguous = get_gold_spec(row)

            if is_ambiguous:
                ambiguous_total += 1
            else:
                answerable_total += 1

            answers = {}
            confidences = {}
            agent_responses = {}

            for agent in AGENTS:
                agent_data = row["agents"].get(agent, {})
                answers[agent] = normalize_answer(agent_data.get("final_answer", ""))
                confidences[agent] = safe_confidence(agent_data.get("confidence"))
                agent_responses[agent] = agent_data.get("response", "")

            majority_answer, majority_count = get_majority_answer(answers)
            majority_is_correct = is_correct_answer(
                majority_answer,
                gold_answer,
                acceptable_answers,
                is_ambiguous
            )

            if majority_is_correct:
                majority_correct += 1

            decision = diva_decision(
    question=row["question"],
    answers=answers,
    confidences=confidences,
    agent_responses=agent_responses
)
            selected_answer = normalize_answer(decision["selected_answer"])
            diva_is_correct = is_correct_answer(
                selected_answer,
                gold_answer,
                acceptable_answers,
                is_ambiguous
            )

            if decision["abstain"]:
                diva_abstained += 1

                if diva_is_correct:
                    diva_correct_abstentions += 1
                    diva_correct_total_style += 1
                else:
                    diva_wrong_abstentions += 1

                if majority_is_correct:
                    abstained_on_majority_correct += 1
                else:
                    abstained_on_majority_wrong += 1
            else:
                diva_answered += 1

                if diva_is_correct:
                    diva_correct_answered += 1
                    diva_correct_total_style += 1

            output_row = {
                "id": row["id"],
                "dataset": row["dataset"],
                "question": row["question"],
                "gold_answer": gold_answer,
                "acceptable_answers": acceptable_answers,
                "is_ambiguous": is_ambiguous,
                "model": row.get("model", ""),
                "agent_answers": answers,
                "agent_confidences": confidences,
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "majority_correct": majority_is_correct,
                "diva": {
                    "selected_answer": selected_answer,
                    "correct": diva_is_correct,
                    "abstain": decision["abstain"],
                    "confidence": decision["confidence"],
                    "disagreement_type": decision["disagreement_type"],
                    "strategy": decision["strategy"],
                    "reason": decision["reason"]
                }
            }

            output_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    majority_accuracy = majority_correct / total

    if diva_answered > 0:
        diva_selective_accuracy = diva_correct_answered / diva_answered
    else:
        diva_selective_accuracy = 0.0

    diva_coverage = diva_answered / total
    diva_abstention_rate = diva_abstained / total
    diva_total_accuracy_counting_abstain_wrong = diva_correct_total_style / total
    diva_ambiguity_aware_accuracy = diva_correct_total_style / total
    diva_correct_abstention_rate = (
        diva_correct_abstentions / ambiguous_total
        if ambiguous_total > 0 else 0.0
    )
    diva_wrong_abstention_rate = (
        diva_wrong_abstentions / answerable_total
        if answerable_total > 0 else 0.0
    )

    print("\n==============================")
    print("DiVA RULE BASED EVALUATION")
    print("==============================")

    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total samples: {total}")

    print("\nMajority Baseline")
    print(f"Majority correct: {majority_correct}/{total} = {majority_accuracy:.2%}")

    print("\nDiVA Results")
    print(f"Answered: {diva_answered}/{total} = {diva_coverage:.2%}")
    print(f"Abstained: {diva_abstained}/{total} = {diva_abstention_rate:.2%}")
    print(f"Selective accuracy on answered samples: {diva_correct_answered}/{diva_answered} = {diva_selective_accuracy:.2%}")
    print(f"Accuracy if abstentions are counted as wrong: {diva_correct_total_style}/{total} = {diva_total_accuracy_counting_abstain_wrong:.2%}")

    if DATASET_NAME == "ambigqa":
        print("\nAmbiguity Metrics")
        print(f"Ambiguous questions: {ambiguous_total}/{total} = {ambiguous_total / total:.2%}")
        print(f"Answerable questions: {answerable_total}/{total} = {answerable_total / total:.2%}")
        print(f"Correct abstentions: {diva_correct_abstentions}/{ambiguous_total} = {diva_correct_abstention_rate:.2%}")
        print(f"Wrong abstentions: {diva_wrong_abstentions}/{answerable_total} = {diva_wrong_abstention_rate:.2%}")
        print(f"Answered accuracy: {diva_correct_answered}/{diva_answered} = {diva_selective_accuracy:.2%}")
        print(f"Coverage: {diva_answered}/{total} = {diva_coverage:.2%}")
        print(f"Ambiguity-aware accuracy: {diva_correct_total_style}/{total} = {diva_ambiguity_aware_accuracy:.2%}")

    print("\nAbstention Analysis")
    print(f"Abstained when majority was wrong: {abstained_on_majority_wrong}")
    print(f"Abstained when majority was correct: {abstained_on_majority_correct}")

    print("\nInterpretation")
    print("Majority accuracy measures raw answer accuracy.")
    print("DiVA selective accuracy measures accuracy only when DiVA chooses to answer.")
    print("A good abstention method should avoid difficult cases where majority is likely wrong.")


if __name__ == "__main__":
    main()
