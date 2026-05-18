import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("outputs/gsm8k_agents_50.jsonl")


def normalize_answer(ans):
    """
    Normalizes answers for comparison.

    Handles:
    $10 -> 10
    990.00 -> 990
    400 ml -> 400
    5.00 -> 5
    empty answers -> ""
    """
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
    """
    GSM8K gold answers usually end like:
    #### 72
    """
    match = re.search(r"####\s*(.*)", gold_text)

    if match:
        return normalize_answer(match.group(1))

    return normalize_answer(gold_text)


def classify_disagreement(counts):
    """
    none      = all agents same
    weak      = 3 agents agree, 1 differs
    strong    = 2 agents agree, others differ
    complete  = all agents differ
    """
    if len(counts) == 1:
        return "none"

    majority_count = counts.most_common(1)[0][1]

    if majority_count >= 3:
        return "weak"

    if majority_count == 2:
        return "strong"

    return "complete"


def main():
    total = 0
    majority_correct_count = 0

    agent_correct_counts = {
        "literal": 0,
        "skeptic": 0,
        "creative": 0,
        "evidence": 0
    }

    disagreement_counts = Counter()
    disagreement_correct_counts = Counter()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1

            gold_answer = extract_gold_answer(row["gold_answer"])

            answers = {
                agent: normalize_answer(data.get("final_answer", ""))
                for agent, data in row["agents"].items()
            }

            counts = Counter(answers.values())
            majority_answer, majority_count = counts.most_common(1)[0]

            disagreement = classify_disagreement(counts)
            disagreement_counts[disagreement] += 1

            majority_correct = majority_answer == gold_answer

            if majority_correct:
                majority_correct_count += 1
                disagreement_correct_counts[disagreement] += 1

            agent_correctness = {}

            for agent, answer in answers.items():
                is_correct = answer == gold_answer
                agent_correctness[agent] = is_correct

                if is_correct:
                    agent_correct_counts[agent] += 1

            print("\nID:", row["id"])
            print("Gold Answer:", gold_answer)
            print("Agent Answers:", answers)
            print("Agent Correctness:", agent_correctness)
            print("Majority Answer:", majority_answer)
            print("Majority Count:", majority_count)
            print("Majority Correct:", majority_correct)
            print("Disagreement:", disagreement)

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    print("Total Samples:", total)

    print("\nAgent Accuracy:")
    for agent, correct_count in agent_correct_counts.items():
        accuracy = correct_count / total
        print(f"{agent}: {correct_count}/{total} = {accuracy:.2%}")

    majority_accuracy = majority_correct_count / total
    print(f"\nMajority Accuracy: {majority_correct_count}/{total} = {majority_accuracy:.2%}")

    print("\nDisagreement Counts:")
    for dtype in ["none", "weak", "strong", "complete"]:
        print(f"{dtype}: {disagreement_counts[dtype]}")

    print("\nMajority Accuracy by Disagreement Type:")
    for dtype in ["none", "weak", "strong", "complete"]:
        count = disagreement_counts[dtype]
        correct = disagreement_correct_counts[dtype]

        if count == 0:
            print(f"{dtype}: NA")
        else:
            print(f"{dtype}: {correct}/{count} = {correct / count:.2%}")


if __name__ == "__main__":
    main()