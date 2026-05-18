from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("gsm8k", "main", split="train[:50]")

with open("data/gsm8k_50.jsonl", "w") as f:
    for i, row in enumerate(dataset):
        sample = {
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": row["answer"]
        }
        f.write(json.dumps(sample) + "\n")

print("done")