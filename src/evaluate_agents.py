import json
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path

import os

DATASET_NAME = os.getenv("DATASET_NAME", "gsm8k")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
RUN_TAG = os.getenv("RUN_TAG", "").strip()
output_suffix = f"_{RUN_TAG}" if RUN_TAG else ""

INPUT_PATH = Path(f"outputs/{DATASET_NAME}_agents_{SAMPLE_SIZE}{output_suffix}.jsonl")

RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / f"{DATASET_NAME}_summary_{SAMPLE_SIZE}{output_suffix}.csv"
DISAGREEMENT_CSV = RESULTS_DIR / f"{DATASET_NAME}_disagreement_summary_{SAMPLE_SIZE}{output_suffix}.csv"
AGENT_CSV = RESULTS_DIR / f"{DATASET_NAME}_agent_summary_{SAMPLE_SIZE}{output_suffix}.csv"
FAILED_MAJORITY_JSONL = RESULTS_DIR / f"{DATASET_NAME}_failed_majority_cases_{SAMPLE_SIZE}{output_suffix}.jsonl"
OVERCONFIDENT_ERRORS_JSONL = RESULTS_DIR / f"{DATASET_NAME}_overconfident_error_cases_{SAMPLE_SIZE}{output_suffix}.jsonl"

AGENTS = ["literal", "skeptic", "creative", "evidence"]
PRIORITY_AGENTS = ["evidence", "literal", "creative", "skeptic"]

OVERCONFIDENCE_THRESHOLD = 0.8
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


def is_abstention_text(text):
    lowered = str(text).lower().strip()

    for pattern in ABSTENTION_PATTERNS:
        if re.match(pattern, lowered):
            return True

    return False


def normalize_answer(ans):
    """
    Robust answer normalization for:
    1. GSM8K numeric answers
    2. StrategyQA yes/no or true/false answers
    3. Messy model outputs like:
       "Final answer: yes"
       "The answer is no"
       "0"
       "1"
       "400 ml"
       "$990.00"
    """
    if ans is None:
        return ""

    text = str(ans).lower().strip()

    if text == "":
        return ""

    text = text.replace("\n", " ")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.strip()

    # Remove common final answer prefixes
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

    # Clean surrounding punctuation
    cleaned = text.strip().strip(".").strip()

    if is_abstention_text(cleaned):
        return ""

    # Boolean normalization
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

    # Handle sentence style boolean answers
    if re.search(r"\b(final answer|answer)\s*:\s*(yes|true)\b", text):
        return "true"

    if re.search(r"\b(final answer|answer)\s*:\s*(no|false)\b", text):
        return "false"

    if re.search(r"\bthe answer is\s+(yes|true)\b", text):
        return "true"

    if re.search(r"\bthe answer is\s+(no|false)\b", text):
        return "false"

    # If the whole answer starts with yes/no, treat it as boolean
    if re.match(r"^(yes|true)\b", cleaned):
        return "true"

    if re.match(r"^(no|false)\b", cleaned):
        return "false"

    # Numeric normalization for GSM8K and numeric outputs
    number_matches = re.findall(r"-?\d+\.?\d*", cleaned)

    if number_matches:
        # Prefer the last number because models often end with the final answer
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
    agent_abstained = Counter()

    agent_conf_sum = defaultdict(float)
    agent_conf_count = Counter()

    majority_correct = 0
    oracle_correct = 0
    correct_abstentions = 0
    wrong_abstentions = 0
    ambiguous_total = 0
    answerable_total = 0

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
        dataset_name = row.get("dataset", DATASET_NAME)
        eval_mode = get_eval_mode(dataset_name)
        gold_answer, acceptable_answers, is_ambiguous = get_gold_spec(row)

        if is_ambiguous:
            ambiguous_total += 1
        else:
            answerable_total += 1

        answers = {}
        confidences = {}

        for agent in AGENTS:
            agent_data = row["agents"].get(agent, {})
            answers[agent] = normalize_answer(agent_data.get("final_answer", ""))
            confidences[agent] = safe_confidence(agent_data.get("confidence"))

        answer_counts = Counter(answers.values())
        majority_answer, majority_count = get_majority_answer_for_mode(
            answers,
            dataset_name
        )

        disagreement_type = classify_disagreement(answer_counts)
        disagreement_counts[disagreement_type] += 1

        is_majority_correct = is_correct_answer(
            majority_answer,
            gold_answer,
            acceptable_answers,
            is_ambiguous,
            eval_mode
        )

        if is_majority_correct:
            majority_correct += 1
            disagreement_majority_correct[disagreement_type] += 1
        else:
            failed_majority_cases.append({
                "id": row["id"],
                "question": row["question"],
                "gold_answer": gold_answer,
                "acceptable_answers": acceptable_answers,
                "is_ambiguous": is_ambiguous,
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "disagreement_type": disagreement_type,
                "answers": answers,
                "confidences": confidences
            })

        any_agent_correct = False

        for agent in AGENTS:
            is_correct = is_correct_answer(
                answers[agent],
                gold_answer,
                acceptable_answers,
                is_ambiguous,
                eval_mode
            )
            confidence = confidences[agent]

            if answers[agent] == "":
                agent_abstained[agent] += 1

                if eval_mode == "mixed_ambiguity" and is_ambiguous:
                    correct_abstentions += 1
                elif eval_mode == "mixed_ambiguity":
                    wrong_abstentions += 1
            

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
        answered = total - agent_abstained[agent]
        selective_accuracy = correct / answered if answered else 0.0
        abstention_rate = agent_abstained[agent] / total

        avg_conf = None
        if agent_conf_count[agent] > 0:
            avg_conf = agent_conf_sum[agent] / agent_conf_count[agent]

        if avg_conf is None:
            print(
                f"{agent}: {correct}/{total} = {accuracy:.2%}, "
                f"answered = {answered}/{total}, "
                f"selective accuracy = {selective_accuracy:.2%}, "
                f"abstention rate = {abstention_rate:.2%}"
            )
        else:
            print(
                f"{agent}: {correct}/{total} = {accuracy:.2%}, "
                f"answered = {answered}/{total}, "
                f"selective accuracy = {selective_accuracy:.2%}, "
                f"abstention rate = {abstention_rate:.2%}, "
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
    if DATASET_NAME in {"condambigqa2k", "situatedqa_temp_raw", "situatedqa_geo_raw"}:
        print("Oracle means at least one agent abstained, which is treated as the correct behavior for this ambiguity-detection setup.")
    else:
        print("Oracle means at least one agent had the correct answer.")

    if DATASET_NAME in {
        "ambigqa",
        "condambigqa2k",
        "situatedqa_temp_raw",
        "situatedqa_temp_clarified",
        "situatedqa_geo_raw",
        "situatedqa_geo_clarified",
    }:
        print("\nAmbiguity Breakdown")
        print(f"Ambiguous questions: {ambiguous_total}/{total} = {ambiguous_total / total:.2%}")
        print(f"Answerable questions: {answerable_total}/{total} = {answerable_total / total:.2%}")
        if DATASET_NAME in {"condambigqa2k", "situatedqa_temp_raw", "situatedqa_geo_raw"}:
            print("Correct abstentions: NA at the system level here; use diva.py output for final abstention metrics.")
            print("Wrong abstentions: NA because this dataset is treated as fully ambiguous in the current setup.")
        else:
            print(f"Correct abstentions: {correct_abstentions}")
            print(f"Wrong abstentions: {wrong_abstentions}")

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
            "abstained",
            "total",
            "accuracy",
            "answered",
            "selective_accuracy",
            "abstention_rate",
            "avg_confidence",
            "overconfident_wrong",
            "overconfident_error_rate"
        ])

        for agent in AGENTS:
            correct = agent_correct[agent]
            wrong = agent_wrong[agent]
            abstained = agent_abstained[agent]
            accuracy = correct / total
            answered = total - abstained
            selective_accuracy = correct / answered if answered else 0.0
            abstention_rate = abstained / total

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
                abstained,
                total,
                accuracy,
                answered,
                selective_accuracy,
                abstention_rate,
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
        writer.writerow(["evaluation_mode", get_eval_mode(DATASET_NAME)])
        writer.writerow(["majority_correct", majority_correct])
        writer.writerow(["majority_accuracy", majority_correct / total])
        writer.writerow(["oracle_correct", oracle_correct])
        writer.writerow(["oracle_accuracy", oracle_correct / total])
        writer.writerow(["overconfidence_threshold", OVERCONFIDENCE_THRESHOLD])
        if DATASET_NAME == "ambigqa":
            writer.writerow(["ambiguous_total", ambiguous_total])
            writer.writerow(["answerable_total", answerable_total])
            writer.writerow(["correct_abstentions", correct_abstentions])
            writer.writerow(["wrong_abstentions", wrong_abstentions])
        elif DATASET_NAME in {
            "condambigqa2k",
            "situatedqa_temp_raw",
            "situatedqa_temp_clarified",
            "situatedqa_geo_raw",
            "situatedqa_geo_clarified",
        }:
            writer.writerow(["ambiguous_total", ambiguous_total])
            writer.writerow(["answerable_total", answerable_total])

        for agent in AGENTS:
            writer.writerow([f"{agent}_correct", agent_correct[agent]])
            writer.writerow([f"{agent}_wrong", agent_wrong[agent]])
            writer.writerow([f"{agent}_abstained", agent_abstained[agent]])
            writer.writerow([f"{agent}_accuracy", agent_correct[agent] / total])
            writer.writerow([f"{agent}_answered", total - agent_abstained[agent]])
            if total - agent_abstained[agent] > 0:
                writer.writerow([
                    f"{agent}_selective_accuracy",
                    agent_correct[agent] / (total - agent_abstained[agent])
                ])
            writer.writerow([
                f"{agent}_abstention_rate",
                agent_abstained[agent] / total
            ])
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
