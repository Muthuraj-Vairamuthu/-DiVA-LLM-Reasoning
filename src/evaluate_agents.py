import json
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path("outputs/gsm8k_agents_50.jsonl")

RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "gsm8k_summary.csv"
DISAGREEMENT_CSV = RESULTS_DIR / "gsm8k_disagreement_summary.csv"
AGENT_CSV = RESULTS_DIR / "gsm8k_agent_summary.csv"
FAILED_MAJORITY_JSONL = RESULTS_DIR / "gsm8k_failed_majority_cases.jsonl"
OVERCONFIDENT_ERRORS_JSONL = RESULTS_DIR / "gsm8k_overconfident_error_cases.jsonl"

AGENTS = ["literal", "skeptic", "creative", "evidence"]
PRIORITY_AGENTS = ["evidence", "literal", "creative", "skeptic"]

OVERCONFIDENCE_THRESHOLD = 0.8


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
        answer
        for answer, count in counts.items()
        if count == max_count
    ]

    if len(candidates) == 1:
        return candidates[0], max_count

    for agent in PRIORITY_AGENTS:
        if answers.get(agent) in candidates:
            return answers[agent], max_count

    return candidates[0], max_count


def safe_confidence(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def load_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = load_rows(INPUT_PATH)
    total = len(rows)

    if total == 0:
        raise ValueError("No rows found in input file.")

    agent_correct = Counter()
    agent_wrong = Counter()
    agent_overconfident_wrong = Counter()

    agent_conf_sum = defaultdict(float)
    agent_conf_count = Counter()

    majority_correct = 0
    oracle_correct = 0

    disagreement_counts = Counter()
    disagreement_majority_correct = Counter()

    agent_correct_by_disagreement = {
        agent: Counter()
        for agent in AGENTS
    }

    agent_wrong_by_disagreement = {
        agent: Counter()
        for agent in AGENTS
    }

    complete_cases = []
    failed_majority_cases = []
    overconfident_error_cases = []

    for row in rows:
        gold_answer = extract_gold_answer(row["gold_answer"])

        answers = {}
        confidences = {}

        for agent in AGENTS:
            agent_data = row["agents"].get(agent, {})
            answers[agent] = normalize_answer(agent_data.get("final_answer", ""))
            confidences[agent] = safe_confidence(agent_data.get("confidence"))

        answer_counts = Counter(answers.values())
        majority_answer, majority_count = get_majority_answer(answers)

        disagreement_type = classify_disagreement(answer_counts)
        disagreement_counts[disagreement_type] += 1

        is_majority_correct = majority_answer == gold_answer

        if is_majority_correct:
            majority_correct += 1
            disagreement_majority_correct[disagreement_type] += 1
        else:
            failed_majority_cases.append({
                "id": row["id"],
                "question": row["question"],
                "gold_answer": gold_answer,
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "disagreement_type": disagreement_type,
                "answers": answers,
                "confidences": confidences
            })

        any_agent_correct = False

        for agent in AGENTS:
            is_correct = answers[agent] == gold_answer
            confidence = confidences[agent]

            if is_correct:
                agent_correct[agent] += 1
                agent_correct_by_disagreement[agent][disagreement_type] += 1
                any_agent_correct = True
            else:
                agent_wrong[agent] += 1
                agent_wrong_by_disagreement[agent][disagreement_type] += 1

                if confidence is not None and confidence >= OVERCONFIDENCE_THRESHOLD:
                    agent_overconfident_wrong[agent] += 1
                    overconfident_error_cases.append({
                        "id": row["id"],
                        "question": row["question"],
                        "agent": agent,
                        "gold_answer": gold_answer,
                        "agent_answer": answers[agent],
                        "confidence": confidence,
                        "disagreement_type": disagreement_type
                    })

            if confidence is not None:
                agent_conf_sum[agent] += confidence
                agent_conf_count[agent] += 1

        if any_agent_correct:
            oracle_correct += 1

        if disagreement_type == "complete":
            complete_cases.append({
                "id": row["id"],
                "question": row["question"],
                "gold_answer": gold_answer,
                "answers": answers,
                "confidences": confidences
            })

    print("\n==============================")
    print("EVALUATION SUMMARY")
    print("==============================")

    print(f"Input file: {INPUT_PATH}")
    print(f"Total samples: {total}")

    print("\nAgent Accuracy")
    for agent in AGENTS:
        correct = agent_correct[agent]
        accuracy = correct / total

        avg_conf = None
        if agent_conf_count[agent] > 0:
            avg_conf = agent_conf_sum[agent] / agent_conf_count[agent]

        if avg_conf is None:
            print(f"{agent}: {correct}/{total} = {accuracy:.2%}")
        else:
            print(
                f"{agent}: {correct}/{total} = {accuracy:.2%}, "
                f"avg confidence = {avg_conf:.3f}"
            )

    print("\nOverconfident Error Rate")
    print(f"Threshold: confidence >= {OVERCONFIDENCE_THRESHOLD}")

    for agent in AGENTS:
        wrong = agent_wrong[agent]
        overconf_wrong = agent_overconfident_wrong[agent]

        if wrong == 0:
            print(f"{agent}: 0 wrong answers")
        else:
            print(
                f"{agent}: {overconf_wrong}/{wrong} = "
                f"{overconf_wrong / wrong:.2%}"
            )

    print("\nMajority Vote")
    print(
        f"majority: {majority_correct}/{total} = "
        f"{majority_correct / total:.2%}"
    )

    print("\nOracle Accuracy")
    print(
        f"oracle: {oracle_correct}/{total} = "
        f"{oracle_correct / total:.2%}"
    )
    print("Oracle means at least one agent had the correct answer.")

    print("\nDisagreement Counts")
    for dtype in ["none", "weak", "strong", "complete"]:
        count = disagreement_counts[dtype]
        print(f"{dtype}: {count}/{total} = {count / total:.2%}")

    print("\nMajority Accuracy by Disagreement Type")
    for dtype in ["none", "weak", "strong", "complete"]:
        count = disagreement_counts[dtype]
        correct = disagreement_majority_correct[dtype]

        if count == 0:
            print(f"{dtype}: NA")
        else:
            print(f"{dtype}: {correct}/{count} = {correct / count:.2%}")

    print("\nAgent Accuracy by Disagreement Type")
    for dtype in ["none", "weak", "strong", "complete"]:
        count = disagreement_counts[dtype]

        if count == 0:
            continue

        print(f"\n{dtype}:")
        for agent in AGENTS:
            correct = agent_correct_by_disagreement[agent][dtype]
            print(f"  {agent}: {correct}/{count} = {correct / count:.2%}")

    print("\nFailed Majority Cases")
    if not failed_majority_cases:
        print("None")
    else:
        for case in failed_majority_cases:
            print(case)

    print("\nComplete Disagreement Cases")
    if not complete_cases:
        print("None")
    else:
        for case in complete_cases:
            print(case)

    print("\nOverconfident Error Cases")
    if not overconfident_error_cases:
        print("None")
    else:
        for case in overconfident_error_cases:
            print(case)

    with open(AGENT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "agent",
            "correct",
            "wrong",
            "total",
            "accuracy",
            "avg_confidence",
            "overconfident_wrong",
            "overconfident_error_rate"
        ])

        for agent in AGENTS:
            correct = agent_correct[agent]
            wrong = agent_wrong[agent]
            accuracy = correct / total

            avg_conf = ""
            if agent_conf_count[agent] > 0:
                avg_conf = agent_conf_sum[agent] / agent_conf_count[agent]

            overconf_wrong = agent_overconfident_wrong[agent]

            overconf_rate = ""
            if wrong > 0:
                overconf_rate = overconf_wrong / wrong

            writer.writerow([
                agent,
                correct,
                wrong,
                total,
                accuracy,
                avg_conf,
                overconf_wrong,
                overconf_rate
            ])

    with open(DISAGREEMENT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "disagreement_type",
            "count",
            "total",
            "percentage",
            "majority_correct",
            "majority_accuracy"
        ])

        for dtype in ["none", "weak", "strong", "complete"]:
            count = disagreement_counts[dtype]
            correct = disagreement_majority_correct[dtype]
            percentage = count / total

            majority_accuracy = ""
            if count > 0:
                majority_accuracy = correct / count

            writer.writerow([
                dtype,
                count,
                total,
                percentage,
                correct,
                majority_accuracy
            ])

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        writer.writerow(["total_samples", total])
        writer.writerow(["majority_correct", majority_correct])
        writer.writerow(["majority_accuracy", majority_correct / total])
        writer.writerow(["oracle_correct", oracle_correct])
        writer.writerow(["oracle_accuracy", oracle_correct / total])
        writer.writerow(["overconfidence_threshold", OVERCONFIDENCE_THRESHOLD])

        for agent in AGENTS:
            writer.writerow([f"{agent}_correct", agent_correct[agent]])
            writer.writerow([f"{agent}_wrong", agent_wrong[agent]])
            writer.writerow([f"{agent}_accuracy", agent_correct[agent] / total])
            writer.writerow([
                f"{agent}_overconfident_wrong",
                agent_overconfident_wrong[agent]
            ])

            if agent_wrong[agent] > 0:
                writer.writerow([
                    f"{agent}_overconfident_error_rate",
                    agent_overconfident_wrong[agent] / agent_wrong[agent]
                ])

    write_jsonl(FAILED_MAJORITY_JSONL, failed_majority_cases)
    write_jsonl(OVERCONFIDENT_ERRORS_JSONL, overconfident_error_cases)

    print("\nSaved result files:")
    print(SUMMARY_CSV)
    print(DISAGREEMENT_CSV)
    print(AGENT_CSV)
    print(FAILED_MAJORITY_JSONL)
    print(OVERCONFIDENT_ERRORS_JSONL)


if __name__ == "__main__":
    main()