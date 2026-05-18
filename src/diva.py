import json
import re
from collections import Counter
from pathlib import Path
import os
from llm_clients import NIMClient
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = Path("outputs/gsm8k_agents_50.jsonl")
OUTPUT_PATH = Path("outputs/gsm8k_diva_50.jsonl")
META_JUDGE_PROMPT_PATH = Path("prompts/meta_judge.txt")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-8b-instruct")

client = NIMClient(
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512
)



AGENTS = ["literal", "skeptic", "creative", "evidence"]


def normalize_answer(ans):
    if ans is None:
        return ""

    ans = str(ans).lower().strip()
    ans = ans.replace("$", "").replace(",", "")

    match = re.search(r"-?\d+\.?\d*", ans)

    if match:
        num = float(match.group())

        if num.is_integer():
            return str(int(num))

        return str(num)

    return ans.strip()


def extract_gold_answer(gold_text):
    match = re.search(r"####\s*(.*)", gold_text)

    if match:
        return normalize_answer(match.group(1))

    return normalize_answer(gold_text)


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
        if answers[agent] in candidates:
            return answers[agent], max_count

    return candidates[0], max_count


def safe_confidence(value):
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


#helper functions before diva decision
def load_meta_judge_prompt():
    with open(META_JUDGE_PROMPT_PATH, "r", encoding="utf-8") as file:
        return file.read()


def extract_json_from_text(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in Meta Judge response.")

    return json.loads(match.group())


def call_meta_judge(question, agent_responses):
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

    evidence_answer = answers["evidence"]
    literal_answer = answers["literal"]
    evidence_conf = confidences["evidence"]

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
        decision["strategy"] = "majority_vote"
        decision["selected_source"] = "majority"
        decision["reason"] = "All agents agree, so the answer is treated as reliable."
        return decision

    if disagreement == "weak":
        decision["selected_answer"] = majority_answer
        decision["confidence"] = 0.85
        decision["strategy"] = "majority_vote"
        decision["selected_source"] = "majority"
        decision["reason"] = "Three agents agree, so majority voting is trusted."
        return decision

    if disagreement == "strong":
        if counts[evidence_answer] == majority_count:
            decision["selected_answer"] = evidence_answer
            decision["confidence"] = 0.75
            decision["strategy"] = "evidence_supported_majority"
            decision["selected_source"] = "evidence"
            decision["reason"] = "There is strong disagreement, but the Evidence Agent supports the majority answer."
            return decision

        if evidence_answer == literal_answer and evidence_conf >= 0.80:
            decision["selected_answer"] = evidence_answer
            decision["confidence"] = 0.70
            decision["strategy"] = "literal_evidence_agreement"
            decision["selected_source"] = "literal_evidence"
            decision["reason"] = "There is strong disagreement, but Literal and Evidence Agents agree with sufficient confidence."
            return decision

        decision["selected_answer"] = majority_answer
        decision["confidence"] = 0.60
        decision["strategy"] = "low_confidence_majority"
        decision["selected_source"] = "majority"
        decision["reason"] = "There is strong disagreement, so majority is selected with reduced confidence."
        return decision

    if disagreement == "complete":
        return call_meta_judge(
            question=question,
            agent_responses=agent_responses
        )

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

    abstained_on_majority_wrong = 0
    abstained_on_majority_correct = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for row in rows:
            gold_answer = extract_gold_answer(row["gold_answer"])

            answers = {}
            confidences = {}
            agent_responses = {}

            for agent in AGENTS:
                agent_data = row["agents"].get(agent, {})
                answers[agent] = normalize_answer(agent_data.get("final_answer", ""))
                confidences[agent] = safe_confidence(agent_data.get("confidence"))
                agent_responses[agent] = agent_data.get("response", "")

            majority_answer, majority_count = get_majority_answer(answers)
            majority_is_correct = majority_answer == gold_answer

            if majority_is_correct:
                majority_correct += 1

            decision = diva_decision(
    question=row["question"],
    answers=answers,
    confidences=confidences,
    agent_responses=agent_responses
)
            selected_answer = normalize_answer(decision["selected_answer"])
            diva_is_correct = selected_answer == gold_answer

            if decision["abstain"]:
                diva_abstained += 1

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
                "model": row.get("model", ""),
                "agent_answers": answers,
                "agent_confidences": confidences,
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "majority_correct": majority_is_correct,
                "diva": {
                    "selected_answer": selected_answer,
                    "correct": diva_is_correct if not decision["abstain"] else None,
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

    print("\nAbstention Analysis")
    print(f"Abstained when majority was wrong: {abstained_on_majority_wrong}")
    print(f"Abstained when majority was correct: {abstained_on_majority_correct}")

    print("\nInterpretation")
    print("Majority accuracy measures raw answer accuracy.")
    print("DiVA selective accuracy measures accuracy only when DiVA chooses to answer.")
    print("A good abstention method should avoid difficult cases where majority is likely wrong.")


if __name__ == "__main__":
    main()