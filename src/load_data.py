from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))

dataset = load_dataset("openai/gsm8k", "main", split=f"train[:{SAMPLE_SIZE}]")

output_path = f"data/gsm8k_{SAMPLE_SIZE}.jsonl"

with open(output_path, "w") as f:
    for i, row in enumerate(dataset):
        sample = {
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": row["answer"]
        }
        f.write(json.dumps(sample) + "\n")

print(f"Saved {SAMPLE_SIZE} GSM8K samples to {output_path}")
