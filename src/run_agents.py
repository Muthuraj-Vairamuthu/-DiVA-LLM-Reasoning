import json
import os
import re
import time
from pathlib import Path

from llm_clients import NIMClient

from dotenv import load_dotenv
load_dotenv()


DATA_PATH = Path("data/gsm8k_50.jsonl")
PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "gsm8k_agents_50.jsonl"

MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-8b-instruct")
NUM_SAMPLES = 50


client = NIMClient(
    model=MODEL_NAME,
    temperature=0.3,
    max_tokens=256
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def load_prompt(agent_name):
    prompt_path = PROMPT_DIR / f"{agent_name}.txt"

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_final_answer(response_text):
    match = re.search(
        r"Final Answer:\s*(.*)",
        response_text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


def extract_confidence(response_text):
    match = re.search(
        r"Confidence:\s*([0-9]*\.?[0-9]+)",
        response_text,
        re.IGNORECASE
    )

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


def extract_uncertainty(response_text):
    match = re.search(
        r"Uncertainty:\s*(none|low|medium|high)",
        response_text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    return ""


def call_agent(question, agent_name, prompt_text):
    full_prompt = f"""
{prompt_text}

Question:
{question}
"""

    response_text = client.generate(
        prompt=full_prompt,
        temperature=0.3,
        max_tokens=512
    ).strip()

    return {
        "response": response_text,
        "final_answer": extract_final_answer(response_text),
        "confidence": extract_confidence(response_text),
        "uncertainty": extract_uncertainty(response_text)
    }


def main():
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is missing. Export it before running.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    samples = load_jsonl(DATA_PATH)[:NUM_SAMPLES]

    agent_names = [
        "literal",
        "skeptic",
        "creative",
        "evidence"
    ]

    prompts = {
        agent_name: load_prompt(agent_name)
        for agent_name in agent_names
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for index, sample in enumerate(samples):
            print(f"Running sample {index + 1}/{NUM_SAMPLES}: {sample['id']}")

            result = {
                "id": sample["id"],
                "dataset": "gsm8k",
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "model": MODEL_NAME,
                "agents": {}
            }

            for agent_name in agent_names:
                print(f"Calling {agent_name} agent")

                agent_output = call_agent(
                    question=sample["question"],
                    agent_name=agent_name,
                    prompt_text=prompts[agent_name]
                )

                result["agents"][agent_name] = agent_output

                time.sleep(0.5)

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    print(f"Done. Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()