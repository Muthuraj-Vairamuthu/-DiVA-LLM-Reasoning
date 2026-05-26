from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))

dataset = load_dataset("ChilleD/StrategyQA", split=f"train[:{SAMPLE_SIZE}]")

output_path = f"data/strategyqa_{SAMPLE_SIZE}.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        sample = {
            "id": f"strategyqa_{i}",
            "dataset": "strategyqa",
            "question": row["question"],
            "gold_answer": str(row["answer"]).lower(),
            "task_type": "commonsense",
            "is_ambiguous": False
        }

        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved {SAMPLE_SIZE} StrategyQA samples to {output_path}")
