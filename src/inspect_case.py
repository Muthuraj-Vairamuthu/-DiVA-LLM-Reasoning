import json
import sys
from pathlib import Path

INPUT_PATH = Path("outputs/gsm8k_agents_50.jsonl")

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/inspect_case.py gsm8k_8")
        return

    target_id = sys.argv[1]

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row["id"] == target_id:
                print("\nID:", row["id"])
                print("\nQuestion:")
                print(row["question"])

                print("\nGold Answer:")
                print(row["gold_answer"])

                print("\nAgent Responses:")
                for agent, data in row["agents"].items():
                    print("\n====================")
                    print(agent.upper())
                    print("====================")
                    print(data["response"])

                return

    print("Case not found:", target_id)

if __name__ == "__main__":
    main()