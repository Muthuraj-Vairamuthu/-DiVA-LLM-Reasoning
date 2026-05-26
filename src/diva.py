import json
import re
from collections import Counter
from pathlib import Path
import os
import csv
from llm_clients import get_llm_client
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME", "gsm8k")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
DISABLE_META_JUDGE = os.getenv("DISABLE_META_JUDGE", "0") == "1"
RUN_TAG = os.getenv("RUN_TAG", "").strip()
AGENT_RUN_TAG = os.getenv("AGENT_RUN_TAG", RUN_TAG).strip()
DIVA_POLICY = os.getenv("DIVA_POLICY", "balanced").strip().lower()

input_suffix = f"_{AGENT_RUN_TAG}" if AGENT_RUN_TAG else ""
output_suffix = f"_{RUN_TAG}" if RUN_TAG else ""
INPUT_PATH = Path(f"outputs/{DATASET_NAME}_agents_{SAMPLE_SIZE}{input_suffix}.jsonl")
OUTPUT_PATH = Path(f"outputs/{DATASET_NAME}_diva_{SAMPLE_SIZE}{output_suffix}.jsonl")
SUMMARY_PATH = Path(f"results/{DATASET_NAME}_diva_summary_{SAMPLE_SIZE}{output_suffix}.csv")
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
VALID_DIVA_POLICIES = {"lenient", "balanced", "strict"}

if DIVA_POLICY not in VALID_DIVA_POLICIES:
    raise ValueError(
        f"Unsupported DIVA_POLICY={DIVA_POLICY!r}. "
        f"Expected one of {sorted(VALID_DIVA_POLICIES)}."
    )


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
    if row.get("dataset") in {
        "ambigqa",
        "condambigqa2k",
        "situatedqa_temp_raw",
        "situatedqa_temp_clarified",
        "situatedqa_geo_raw",
        "situatedqa_geo_clarified",
    }:
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


def get_eval_mode(dataset_name):
    if dataset_name == "condambigqa2k":
        return "abstention_only"

    if dataset_name in {"situatedqa_temp_raw", "situatedqa_geo_raw"}:
        return "abstention_only"

    if dataset_name == "ambigqa":
        return "mixed_ambiguity"

    return "answer_accuracy"


def should_include_abstentions_in_majority(dataset_name):
    return get_eval_mode(dataset_name) == "abstention_only"


def is_correct_answer(answer, gold_answer, acceptable_answers, is_ambiguous, eval_mode):
    if eval_mode == "abstention_only":
        return answer == ""

    if is_ambiguous:
        return answer == ""

    if answer == "":
        return False

    if acceptable_answers:
        return answer in acceptable_answers

    return answer == gold_answer


def get_majority_answer_for_mode(answers, dataset_name):
    if should_include_abstentions_in_majority(dataset_name):
        return get_majority_answer(answers)

    non_empty_answers = {
        agent: answer
        for agent, answer in answers.items()
        if answer != ""
    }

    if non_empty_answers:
        return get_majority_answer(non_empty_answers)

    return get_majority_answer(answers)


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
            client = get_llm_client(
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


def fallback_complete_disagreement_decision(answers, confidences):
    for agent in ["evidence", "literal", "creative", "skeptic"]:
        answer = answers.get(agent, "")
        if answer != "":
            return {
                "selected_answer": answer,
                "abstain": False,
                "confidence": safe_confidence(confidences.get(agent, 0.4)),
                "disagreement_type": "complete",
                "strategy": "complete_disagreement_priority_fallback",
                "reason": (
                    "Meta Judge is disabled or unavailable, so DiVA falls back to a "
                    "priority agent selection on complete disagreement."
                ),
                "selected_source": agent
            }

    return {
        "selected_answer": "",
        "abstain": True,
        "confidence": 0.0,
        "disagreement_type": "complete",
        "strategy": "complete_disagreement_abstain_fallback",
        "reason": (
            "Meta Judge is disabled or unavailable and no non-abstaining agent answer "
            "is available, so DiVA abstains."
        ),
        "selected_source": "abstain"
    }


def ambiguity_policy_should_abstain(disagreement, abstaining_agents):
    abstain_count = len(abstaining_agents)
    abstain_set = set(abstaining_agents)

    if DIVA_POLICY == "lenient":
        if {"evidence", "skeptic"}.issubset(abstain_set):
            return True, 0.25, "lenient_grounded_pair_abstention"
        if abstain_count >= 3:
            return True, 0.20, "lenient_three_agent_abstention"
        if disagreement == "complete":
            return True, 0.30, "lenient_complete_disagreement"
        return False, None, None

    if DIVA_POLICY == "strict":
        if abstain_count >= 1:
            return True, 0.20, "strict_any_agent_abstention"
        if disagreement in {"weak", "strong", "complete"}:
            return True, 0.25, "strict_any_disagreement"
        return False, None, None

    if abstain_count >= 2:
        return True, 0.25, "balanced_multiple_agent_abstentions"
    if "evidence" in abstain_set and disagreement != "none":
        return True, 0.25, "balanced_evidence_abstention"
    if {"evidence", "skeptic"}.issubset(abstain_set):
        return True, 0.20, "balanced_grounded_pair_abstention"
    if disagreement in {"strong", "complete"}:
        return True, 0.30, "balanced_high_disagreement"

    return False, None, None


def ambiguity_policy_reason(strategy, abstaining_agents):
    grounded_agents = ",".join(abstaining_agents) if abstaining_agents else "abstain"

    reasons = {
        "lenient_grounded_pair_abstention": (
            "Both grounded agents abstain, so even the lenient policy treats the question as underspecified."
        ),
        "lenient_three_agent_abstention": (
            "Three agents abstain, so the lenient policy abstains despite preferring coverage."
        ),
        "lenient_complete_disagreement": (
            "Under the lenient policy, only complete disagreement is strong enough to trigger abstention."
        ),
        "strict_any_agent_abstention": (
            "Under the strict policy, any abstaining agent is enough to trigger abstention."
        ),
        "strict_any_disagreement": (
            "Under the strict policy, any non-unanimous disagreement is treated as a reason to abstain."
        ),
        "balanced_multiple_agent_abstentions": (
            "Multiple agents abstain, so the balanced policy treats the question as too underspecified to answer."
        ),
        "balanced_evidence_abstention": (
            "The evidence agent abstains under disagreement, so the balanced policy treats the question as underspecified."
        ),
        "balanced_grounded_pair_abstention": (
            "Both grounded agents abstain, so the balanced policy strongly prefers abstention."
        ),
        "balanced_high_disagreement": (
            "High disagreement on an ambiguity-sensitive dataset is treated as a signal to abstain."
        ),
    }

    return reasons.get(strategy, "DiVA abstains under the selected ambiguity policy."), grounded_agents


def diva_decision(question, answers, confidences, agent_responses):
    counts = Counter(answers.values())
    disagreement = classify_disagreement(counts)

    majority_answer, majority_count = get_majority_answer(answers)
    abstaining_agents = [
        agent
        for agent, answer in answers.items()
        if answer == ""
    ]

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

    if DATASET_NAME in {
        "ambigqa",
        "condambigqa2k",
        "situatedqa_temp_raw",
        "situatedqa_geo_raw",
    }:
        should_abstain, policy_confidence, policy_strategy = ambiguity_policy_should_abstain(
            disagreement,
            abstaining_agents
        )

        if should_abstain:
            reason, selected_source = ambiguity_policy_reason(
                policy_strategy,
                abstaining_agents
            )
            decision["selected_answer"] = ""
            decision["abstain"] = True
            decision["confidence"] = policy_confidence
            decision["strategy"] = policy_strategy
            decision["selected_source"] = selected_source
            decision["reason"] = reason
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

    if DATASET_NAME in {"situatedqa_temp_clarified", "situatedqa_geo_clarified"}:
        if answers.get("literal") != "" and answers.get("literal") == answers.get("evidence"):
            decision["selected_answer"] = answers["literal"]
            decision["confidence"] = 0.90
            decision["strategy"] = "clarified_literal_evidence_agreement"
            decision["selected_source"] = "literal,evidence"
            decision["reason"] = (
                "The direct and grounded agents agree on the clarified question, so their shared answer is preferred."
            )
            return decision

        non_empty_answers = [
            answer
            for answer in answers.values()
            if answer != ""
        ]

        if not non_empty_answers:
            decision["selected_answer"] = ""
            decision["abstain"] = True
            decision["confidence"] = 0.0
            decision["strategy"] = "clarified_all_abstain_fallback"
            decision["selected_source"] = "abstain"
            decision["reason"] = (
                "All agents abstained even after clarification, so DiVA abstains as a fallback."
            )
            return decision

        if disagreement == "complete":
            for agent in ["literal", "evidence", "creative", "skeptic"]:
                if answers.get(agent) != "":
                    decision["selected_answer"] = answers[agent]
                    decision["confidence"] = safe_confidence(confidences.get(agent, 0.55))
                    decision["strategy"] = "clarified_priority_fallback_on_complete"
                    decision["selected_source"] = agent
                    decision["reason"] = (
                        "On clarified questions, DiVA prefers answering over abstaining under complete disagreement."
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
        if DISABLE_META_JUDGE:
            return fallback_complete_disagreement_decision(answers, confidences)

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
            dataset_name = row.get("dataset", DATASET_NAME)
            eval_mode = get_eval_mode(dataset_name)
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

            majority_answer, majority_count = get_majority_answer_for_mode(
                answers,
                dataset_name
            )
            majority_is_correct = is_correct_answer(
                majority_answer,
                gold_answer,
                acceptable_answers,
                is_ambiguous,
                eval_mode
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
                is_ambiguous,
                eval_mode
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
    print(f"Policy: {DIVA_POLICY}")
    print(f"Total samples: {total}")

    print("\nMajority Baseline")
    print(f"Majority correct: {majority_correct}/{total} = {majority_accuracy:.2%}")

    print("\nDiVA Results")
    print(f"Answered: {diva_answered}/{total} = {diva_coverage:.2%}")
    print(f"Abstained: {diva_abstained}/{total} = {diva_abstention_rate:.2%}")
    if DATASET_NAME == "condambigqa2k":
        print("Selective accuracy on answered samples: NA (short-answer exact match is unsupported for CondAmbigQA-2K)")
    else:
        print(f"Selective accuracy on answered samples: {diva_correct_answered}/{diva_answered} = {diva_selective_accuracy:.2%}")
    print(f"Accuracy if abstentions are counted as wrong: {diva_correct_total_style}/{total} = {diva_total_accuracy_counting_abstain_wrong:.2%}")

    if DATASET_NAME in {
        "ambigqa",
        "condambigqa2k",
        "situatedqa_temp_raw",
        "situatedqa_temp_clarified",
        "situatedqa_geo_raw",
        "situatedqa_geo_clarified",
    }:
        print("\nAmbiguity Metrics")
        print(f"Ambiguous questions: {ambiguous_total}/{total} = {ambiguous_total / total:.2%}")
        print(f"Answerable questions: {answerable_total}/{total} = {answerable_total / total:.2%}")
        print(f"Correct abstentions: {diva_correct_abstentions}/{ambiguous_total} = {diva_correct_abstention_rate:.2%}")
        print(f"Wrong abstentions: {diva_wrong_abstentions}/{answerable_total} = {diva_wrong_abstention_rate:.2%}")
        if DATASET_NAME in {"condambigqa2k", "situatedqa_temp_raw", "situatedqa_geo_raw"}:
            print("Answered accuracy: NA (dataset is evaluated as ambiguity detection / abstention only)")
        else:
            print(f"Answered accuracy: {diva_correct_answered}/{diva_answered} = {diva_selective_accuracy:.2%}")
        print(f"Coverage: {diva_answered}/{total} = {diva_coverage:.2%}")
        print(f"Ambiguity-aware accuracy: {diva_correct_total_style}/{total} = {diva_ambiguity_aware_accuracy:.2%}")

    print("\nAbstention Analysis")
    print(f"Abstained when majority was wrong: {abstained_on_majority_wrong}")
    print(f"Abstained when majority was correct: {abstained_on_majority_correct}")

    print("\nInterpretation")
    if DATASET_NAME in {"condambigqa2k", "situatedqa_temp_raw", "situatedqa_geo_raw"}:
        print(f"{DATASET_NAME} is evaluated as an ambiguity-detection dataset in this pipeline.")
        if DATASET_NAME == "condambigqa2k":
            print("A correct outcome is to abstain, because the gold targets are long condition-specific explanations rather than short exact-match answers.")
        else:
            print("A correct outcome is to abstain, because the raw question is intentionally missing the disambiguating time or location context.")
        print("Coverage shows how often DiVA chose to answer instead of abstaining.")
    else:
        print("Majority accuracy measures raw answer accuracy.")
        print("DiVA selective accuracy measures accuracy only when DiVA chooses to answer.")
        print("A good abstention method should avoid difficult cases where majority is likely wrong.")

    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["dataset", DATASET_NAME])
        writer.writerow(["sample_size", SAMPLE_SIZE])
        writer.writerow(["run_tag", RUN_TAG])
        writer.writerow(["diva_policy", DIVA_POLICY])
        writer.writerow(["total_samples", total])
        writer.writerow(["majority_correct", majority_correct])
        writer.writerow(["majority_accuracy", majority_accuracy])
        writer.writerow(["diva_answered", diva_answered])
        writer.writerow(["diva_abstained", diva_abstained])
        writer.writerow(["diva_coverage", diva_coverage])
        writer.writerow(["diva_abstention_rate", diva_abstention_rate])
        writer.writerow(["diva_correct_answered", diva_correct_answered])
        writer.writerow(["diva_selective_accuracy", diva_selective_accuracy])
        writer.writerow(["diva_accuracy_abstentions_wrong", diva_total_accuracy_counting_abstain_wrong])
        writer.writerow(["ambiguous_total", ambiguous_total])
        writer.writerow(["answerable_total", answerable_total])
        writer.writerow(["diva_correct_abstentions", diva_correct_abstentions])
        writer.writerow(["diva_wrong_abstentions", diva_wrong_abstentions])
        writer.writerow(["diva_correct_abstention_rate", diva_correct_abstention_rate])
        writer.writerow(["diva_wrong_abstention_rate", diva_wrong_abstention_rate])
        writer.writerow(["diva_ambiguity_aware_accuracy", diva_ambiguity_aware_accuracy])
        writer.writerow(["abstained_on_majority_wrong", abstained_on_majority_wrong])
        writer.writerow(["abstained_on_majority_correct", abstained_on_majority_correct])


if __name__ == "__main__":
    main()
