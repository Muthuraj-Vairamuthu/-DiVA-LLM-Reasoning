from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("ChilleD/StrategyQA", split="train[:50]")

with open("data/strategyqa_50.jsonl", "w", encoding="utf-8") as f:
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

print("Saved 50 StrategyQA samples to data/strategyqa_50.jsonl")