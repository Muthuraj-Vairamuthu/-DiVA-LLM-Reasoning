import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
RUN_TAG = os.getenv("RUN_TAG", "").strip()
SITUATEDQA_CONFIG = os.getenv("SITUATEDQA_CONFIG", "temp").strip().lower()

output_suffix = f"_{RUN_TAG}" if RUN_TAG else ""

RAW_DATASET_NAME = f"situatedqa_{SITUATEDQA_CONFIG}_raw"
CLARIFIED_DATASET_NAME = f"situatedqa_{SITUATEDQA_CONFIG}_clarified"

RAW_DIVA_PATH = Path(f"outputs/{RAW_DATASET_NAME}_diva_{SAMPLE_SIZE}{output_suffix}.jsonl")
CLARIFIED_DIVA_PATH = Path(f"outputs/{CLARIFIED_DATASET_NAME}_diva_{SAMPLE_SIZE}{output_suffix}.jsonl")

RAW_AGENTS_PATH = Path(f"outputs/{RAW_DATASET_NAME}_agents_{SAMPLE_SIZE}{output_suffix}.jsonl")
CLARIFIED_AGENTS_PATH = Path(f"outputs/{CLARIFIED_DATASET_NAME}_agents_{SAMPLE_SIZE}{output_suffix}.jsonl")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / f"situatedqa_{SITUATEDQA_CONFIG}_combined_summary_{SAMPLE_SIZE}{output_suffix}.csv"

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


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

    true_patterns = {"true", "yes", "y", "1", "correct"}
    false_patterns = {"false", "no", "n", "0", "incorrect"}

    if cleaned in true_patterns:
        return "true"
    if cleaned in false_patterns:
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
        return [normalize_answer(x) for x in gold_text if str(x).strip()]
    if gold_text is None:
        return ""
    return normalize_answer(gold_text)


def is_correct_for_row(answer, row):
    acceptable_answers = row.get("acceptable_answers", [])
    if row.get("is_ambiguous", False):
        return answer == ""

    if answer == "":
        return False

    if acceptable_answers:
        return answer in acceptable_answers

    gold_answer = extract_gold_answer(row.get("gold_answer"))
    if isinstance(gold_answer, list):
        return answer in gold_answer
    return answer == gold_answer


def compute_oracle_correct(agent_rows):
    correct = 0
    for row in agent_rows:
        found = False
        for agent in AGENTS:
            answer = normalize_answer(row["agents"].get(agent, {}).get("final_answer", ""))
            if row.get("is_ambiguous", False):
                if answer == "":
                    found = True
                    break
            else:
                acceptable_answers = [normalize_answer(x) for x in row.get("gold_answer", []) if str(x).strip()]
                if answer != "" and answer in acceptable_answers:
                    found = True
                    break

        if found:
            correct += 1

    return correct


def summarize_split(diva_rows, agent_rows):
    total = len(diva_rows)
    majority_correct = sum(1 for row in diva_rows if row.get("majority_correct"))
    diva_correct = sum(1 for row in diva_rows if row.get("diva", {}).get("correct"))
    diva_abstained = sum(1 for row in diva_rows if row.get("diva", {}).get("abstain"))
    oracle_correct = compute_oracle_correct(agent_rows)

    return {
        "total": total,
        "majority_correct": majority_correct,
        "diva_correct": diva_correct,
        "diva_abstained": diva_abstained,
        "oracle_correct": oracle_correct,
    }


def main():
    raw_diva_rows = load_jsonl(RAW_DIVA_PATH)
    clarified_diva_rows = load_jsonl(CLARIFIED_DIVA_PATH)
    raw_agent_rows = load_jsonl(RAW_AGENTS_PATH)
    clarified_agent_rows = load_jsonl(CLARIFIED_AGENTS_PATH)

    raw = summarize_split(raw_diva_rows, raw_agent_rows)
    clarified = summarize_split(clarified_diva_rows, clarified_agent_rows)

    total = raw["total"] + clarified["total"]
    combined_majority = raw["majority_correct"] + clarified["majority_correct"]
    combined_diva = raw["diva_correct"] + clarified["diva_correct"]
    combined_oracle = raw["oracle_correct"] + clarified["oracle_correct"]

    print("\n==============================")
    print("SITUATEDQA COMBINED EVALUATION")
    print("==============================")
    print(f"Config: {SITUATEDQA_CONFIG}")
    print(f"Sample size per split: {SAMPLE_SIZE}")
    print(f"Combined total: {total}")

    print("\nRaw split")
    print(f"Majority correct: {raw['majority_correct']}/{raw['total']} = {raw['majority_correct'] / raw['total']:.2%}")
    print(f"Oracle correct: {raw['oracle_correct']}/{raw['total']} = {raw['oracle_correct'] / raw['total']:.2%}")
    print(f"DiVA correct: {raw['diva_correct']}/{raw['total']} = {raw['diva_correct'] / raw['total']:.2%}")
    print(f"DiVA abstained: {raw['diva_abstained']}/{raw['total']} = {raw['diva_abstained'] / raw['total']:.2%}")

    print("\nClarified split")
    print(f"Majority correct: {clarified['majority_correct']}/{clarified['total']} = {clarified['majority_correct'] / clarified['total']:.2%}")
    print(f"Oracle correct: {clarified['oracle_correct']}/{clarified['total']} = {clarified['oracle_correct'] / clarified['total']:.2%}")
    print(f"DiVA correct: {clarified['diva_correct']}/{clarified['total']} = {clarified['diva_correct'] / clarified['total']:.2%}")
    print(f"DiVA abstained: {clarified['diva_abstained']}/{clarified['total']} = {clarified['diva_abstained'] / clarified['total']:.2%}")

    print("\nCombined score")
    print(f"Combined majority accuracy: {combined_majority}/{total} = {combined_majority / total:.2%}")
    print(f"Combined oracle accuracy: {combined_oracle}/{total} = {combined_oracle / total:.2%}")
    print(f"Combined DiVA accuracy: {combined_diva}/{total} = {combined_diva / total:.2%}")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["config", SITUATEDQA_CONFIG])
        writer.writerow(["sample_size_per_split", SAMPLE_SIZE])
        writer.writerow(["combined_total", total])
        writer.writerow(["raw_total", raw["total"]])
        writer.writerow(["raw_majority_correct", raw["majority_correct"]])
        writer.writerow(["raw_majority_accuracy", raw["majority_correct"] / raw["total"]])
        writer.writerow(["raw_oracle_correct", raw["oracle_correct"]])
        writer.writerow(["raw_oracle_accuracy", raw["oracle_correct"] / raw["total"]])
        writer.writerow(["raw_diva_correct", raw["diva_correct"]])
        writer.writerow(["raw_diva_accuracy", raw["diva_correct"] / raw["total"]])
        writer.writerow(["raw_diva_abstention_rate", raw["diva_abstained"] / raw["total"]])
        writer.writerow(["clarified_total", clarified["total"]])
        writer.writerow(["clarified_majority_correct", clarified["majority_correct"]])
        writer.writerow(["clarified_majority_accuracy", clarified["majority_correct"] / clarified["total"]])
        writer.writerow(["clarified_oracle_correct", clarified["oracle_correct"]])
        writer.writerow(["clarified_oracle_accuracy", clarified["oracle_correct"] / clarified["total"]])
        writer.writerow(["clarified_diva_correct", clarified["diva_correct"]])
        writer.writerow(["clarified_diva_accuracy", clarified["diva_correct"] / clarified["total"]])
        writer.writerow(["clarified_diva_abstention_rate", clarified["diva_abstained"] / clarified["total"]])
        writer.writerow(["combined_majority_correct", combined_majority])
        writer.writerow(["combined_majority_accuracy", combined_majority / total])
        writer.writerow(["combined_oracle_correct", combined_oracle])
        writer.writerow(["combined_oracle_accuracy", combined_oracle / total])
        writer.writerow(["combined_diva_correct", combined_diva])
        writer.writerow(["combined_diva_accuracy", combined_diva / total])

    print("\nSaved combined summary:")
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
