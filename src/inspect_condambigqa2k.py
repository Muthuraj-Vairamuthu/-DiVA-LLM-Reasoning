from datasets import load_dataset
import json
import os

DATASET_NAME = "Apocalypse-AGI-DAO/CondAmbigQA-2K"

print("Loading dataset info...")
dataset = load_dataset(DATASET_NAME)

print("\nAvailable splits:")
for split_name in dataset.keys():
    print("-", split_name, "size =", len(dataset[split_name]))

first_split = list(dataset.keys())[0]
print("\nUsing split:", first_split)

row = dataset[first_split][0]

print("\nRow keys:")
print(list(row.keys()))

print("\nFirst row:")
print(json.dumps(row, indent=2, ensure_ascii=False, default=str)[:8000])