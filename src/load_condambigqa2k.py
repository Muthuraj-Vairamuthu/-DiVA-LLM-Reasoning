from datasets import load_dataset
import json
import os

SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200"))
DATASET_ID = "Apocalypse-AGI-DAO/CondAmbigQA-2K"

os.makedirs("data", exist_ok=True)

dataset = load_dataset(DATASET_ID)
rows = dataset["train"]

output_path = f"data/condambigqa2k_{SAMPLE_SIZE}.jsonl"

written = 0
with open(output_path, "w", encoding="utf-8") as f:
    for row in rows:
        if written >= SAMPLE_SIZE:
            break

        question = row.get("question", "").strip()
        properties = row.get("properties", [])

        if not question or not properties:
            continue

        acceptable_answers = []
        qa_pairs = []

        for prop in properties:
            condition = str(prop.get("condition", "")).strip()
            groundtruth = str(prop.get("groundtruth", "")).strip()

            if groundtruth:
                acceptable_answers.append(groundtruth)

            if condition or groundtruth:
                qa_pairs.append({
                    "condition": condition,
                    "answer": groundtruth
                })

        acceptable_answers = list(dict.fromkeys(a for a in acceptable_answers if a))

        if not acceptable_answers:
            continue

        sample = {
            "id": f"condambigqa2k_{written}",
            "dataset": "condambigqa2k",
            "question": question,
            "gold_answer": acceptable_answers,
            "task_type": "ambiguous_qa",
            "is_ambiguous": True,
            "qa_pairs": qa_pairs
        }

        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        written += 1

print(f"Saved {written} CondAmbigQA-2K samples to {output_path}")
