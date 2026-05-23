from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
dataset = load_dataset("sewon/ambig_qa", "light", split="train")

count = 0
ambiguous = 0
answerable = 0

output_path = f"data/ambigqa_{SAMPLE_SIZE}.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for row in dataset:
        if count >= SAMPLE_SIZE:
            break

        annotations = row["annotations"]
        ann_types = annotations.get("type", [])

        is_ambiguous = "multipleQAs" in ann_types

        answers = []

        qa_pairs = annotations.get("qaPairs", [])
        for pair in qa_pairs:
            pair_answers = pair.get("answer", [])

            for ans_group in pair_answers:
                if isinstance(ans_group, list):
                    answers.extend(ans_group)
                else:
                    answers.append(ans_group)

        if not answers:
            normal_answers = annotations.get("answer", [])
            for ans_group in normal_answers:
                if isinstance(ans_group, list):
                    answers.extend(ans_group)
                else:
                    answers.append(ans_group)

        sample = {
            "id": f"ambigqa_{count}",
            "dataset": "ambigqa",
            "question": row["question"],
            "gold_answer": answers,
            "task_type": "ambiguous_qa",
            "is_ambiguous": is_ambiguous
        }

        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        count += 1

        if is_ambiguous:
            ambiguous += 1
        else:
            answerable += 1

print(f"Saved {count} AmbigQA samples to {output_path}")
print(f"Ambiguous: {ambiguous}")
print(f"Answerable: {answerable}")
