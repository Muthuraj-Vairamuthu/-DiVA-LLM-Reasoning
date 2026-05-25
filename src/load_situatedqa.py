import json
import os
from pathlib import Path

import requests


SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
SITUATEDQA_CONFIG = os.getenv("SITUATEDQA_CONFIG", "temp").strip().lower()

VALID_CONFIGS = {"temp", "geo"}

if SITUATEDQA_CONFIG not in VALID_CONFIGS:
    raise ValueError(
        f"Unsupported SITUATEDQA_CONFIG={SITUATEDQA_CONFIG!r}. "
        f"Use one of: {sorted(VALID_CONFIGS)}"
    )

BASE_URL = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data"
SOURCE_URL = f"{BASE_URL}/{SITUATEDQA_CONFIG}.train.jsonl"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_DATASET_NAME = f"situatedqa_{SITUATEDQA_CONFIG}_raw"
CLARIFIED_DATASET_NAME = f"situatedqa_{SITUATEDQA_CONFIG}_clarified"

RAW_OUTPUT_PATH = DATA_DIR / f"{RAW_DATASET_NAME}_{SAMPLE_SIZE}.jsonl"
CLARIFIED_OUTPUT_PATH = DATA_DIR / f"{CLARIFIED_DATASET_NAME}_{SAMPLE_SIZE}.jsonl"


def load_rows():
    response = requests.get(SOURCE_URL, timeout=120)
    response.raise_for_status()

    rows = []
    for line in response.text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    return rows


def normalize_answer_list(values):
    answers = []

    for value in values or []:
        text = str(value).strip()
        if text:
            answers.append(text)

    # keep order while removing duplicates
    return list(dict.fromkeys(answers))


def build_context_metadata(row):
    metadata = {}

    if row.get("location"):
        metadata["location"] = row["location"]

    if row.get("date"):
        metadata["date"] = row["date"]

    if row.get("date_type"):
        metadata["date_type"] = row["date_type"]

    return metadata


def build_samples(rows):
    raw_samples = []
    clarified_samples = []

    for i, row in enumerate(rows[:SAMPLE_SIZE]):
        answers = normalize_answer_list(row.get("answer", []))
        any_answers = normalize_answer_list(row.get("any_answer", []))

        if not answers and not any_answers:
            continue

        gold_answers = answers or any_answers
        metadata = build_context_metadata(row)

        base_fields = {
            "source_id": str(row.get("id", i)),
            "edited_question": row.get("edited_question", "").strip(),
            "context_metadata": metadata,
            "any_answer": any_answers,
        }

        raw_samples.append({
            "id": f"{RAW_DATASET_NAME}_{len(raw_samples)}",
            "dataset": RAW_DATASET_NAME,
            "question": row["question"].strip(),
            "gold_answer": gold_answers,
            "task_type": "ambiguous_qa",
            "is_ambiguous": True,
            **base_fields,
        })

        clarified_samples.append({
            "id": f"{CLARIFIED_DATASET_NAME}_{len(clarified_samples)}",
            "dataset": CLARIFIED_DATASET_NAME,
            "question": row.get("edited_question", row["question"]).strip(),
            "gold_answer": gold_answers,
            "task_type": "contextual_qa",
            "is_ambiguous": False,
            "original_question": row["question"].strip(),
            **base_fields,
        })

    return raw_samples, clarified_samples


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = load_rows()
    raw_samples, clarified_samples = build_samples(rows)

    write_jsonl(RAW_OUTPUT_PATH, raw_samples)
    write_jsonl(CLARIFIED_OUTPUT_PATH, clarified_samples)

    print(f"Downloaded {len(rows)} source rows from {SOURCE_URL}")
    print(f"Saved {len(raw_samples)} raw SituatedQA samples to {RAW_OUTPUT_PATH}")
    print(f"Saved {len(clarified_samples)} clarified SituatedQA samples to {CLARIFIED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
