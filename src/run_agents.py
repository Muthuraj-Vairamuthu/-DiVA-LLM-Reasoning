import json
import os
import re
import time
from pathlib import Path

from llm_clients import get_llm_client

from dotenv import load_dotenv
load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME", "gsm8k")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
RUN_TAG = os.getenv("RUN_TAG", "").strip()

DATA_PATH = Path(f"data/{DATASET_NAME}_{SAMPLE_SIZE}.jsonl")
PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("outputs")
output_suffix = f"_{RUN_TAG}" if RUN_TAG else ""
OUTPUT_PATH = OUTPUT_DIR / f"{DATASET_NAME}_agents_{SAMPLE_SIZE}{output_suffix}.jsonl"

MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-8b-instruct")
NUM_SAMPLES = SAMPLE_SIZE
AGENT_SLEEP_SECONDS = float(os.getenv("AGENT_SLEEP_SECONDS", "0.5"))
RESUME_RUN = os.getenv("RESUME_RUN", "1") == "1"
AGENT_CALL_RECOVERY_RETRIES = int(os.getenv("AGENT_CALL_RECOVERY_RETRIES", "3"))
AGENT_FAILURE_COOLDOWN_SECONDS = float(os.getenv("AGENT_FAILURE_COOLDOWN_SECONDS", "45"))


client = get_llm_client(
    model=MODEL_NAME,
    temperature=0.3,
    max_tokens=256
)


def log(message):
    print(message, flush=True)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def load_completed_ids(path):
    if not path.exists():
        return set()

    completed_ids = set()

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = row.get("id")
            if sample_id:
                completed_ids.add(sample_id)

    return completed_ids


def load_prompt(agent_name):
    prompt_path = PROMPT_DIR / f"{agent_name}.txt"

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_final_answer(response_text):
    matches = re.findall(
        r"Final Answer:\s*(.*)",
        response_text,
        re.IGNORECASE
    )

    if matches:
        return matches[-1].strip()

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

def get_task_instruction(dataset_name):
    if dataset_name == "strategyqa":
        return """
Dataset specific instruction:
This is a yes/no commonsense reasoning question.

Your Final Answer must be exactly one of:
yes
no

Do not write maybe, unknown, unclear, a number, or a full sentence in the Final Answer field.
Do not put confidence or explanation inside the Final Answer field.
"""

    if dataset_name in {"ambigqa", "condambigqa2k"}:
        return """
Dataset specific instruction:
This is an ambiguity-sensitive open-domain question.

If the question is genuinely ambiguous, underspecified, or has multiple plausible answers,
you should abstain instead of guessing.

Your Final Answer must be exactly one of:
1. a short answer phrase
2. abstain

Do not write a full sentence in the Final Answer field.
Do not put confidence or explanation inside the Final Answer field.
"""

    if dataset_name == "gsm8k":
        return """
Dataset specific instruction:
This is a math word problem.

Your Final Answer must contain only the final numeric answer.
Do not include units, explanation, equations, or a full sentence in the Final Answer field.
"""

    return """
Dataset specific instruction:
Follow the answer format required by the task.
Keep the Final Answer short and exact.
"""


def call_agent(question, agent_name, prompt_text):
    task_instruction = get_task_instruction(DATASET_NAME)

    full_prompt = f"""
{prompt_text}

{task_instruction}

Question:
{question}
"""

    response_text = client.generate(
        prompt=full_prompt,
        temperature=0.3,
        max_tokens=256
    ).strip()

    return {
        "response": response_text,
        "final_answer": extract_final_answer(response_text),
        "confidence": extract_confidence(response_text),
        "uncertainty": extract_uncertainty(response_text)
    }


def call_agent_with_recovery(question, agent_name, prompt_text):
    for attempt in range(AGENT_CALL_RECOVERY_RETRIES + 1):
        try:
            return call_agent(
                question=question,
                agent_name=agent_name,
                prompt_text=prompt_text
            )
        except RuntimeError as error:
            if attempt >= AGENT_CALL_RECOVERY_RETRIES:
                raise

            wait = AGENT_FAILURE_COOLDOWN_SECONDS * (attempt + 1)
            log(
                f"{agent_name} agent failed with '{error}'. "
                f"Cooling down for {wait:.0f} seconds before retry {attempt + 1}/{AGENT_CALL_RECOVERY_RETRIES}."
            )
            time.sleep(wait)


def main():
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is missing. Export it before running.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    log(f"Starting run_agents for dataset={DATASET_NAME}, samples={NUM_SAMPLES}, model={MODEL_NAME}, run_tag={RUN_TAG or 'default'}")
    log(f"Reading data from {DATA_PATH}")
    log(f"Writing outputs to {OUTPUT_PATH}")

    samples = load_jsonl(DATA_PATH)[:NUM_SAMPLES]
    log(f"Loaded {len(samples)} samples")
    completed_ids = load_completed_ids(OUTPUT_PATH) if RESUME_RUN else set()

    if completed_ids:
        log(f"Resume enabled. Found {len(completed_ids)} completed samples in existing output.")

    remaining_samples = [
        sample
        for sample in samples
        if sample["id"] not in completed_ids
    ]

    log(f"Remaining samples to run: {len(remaining_samples)}")

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

    file_mode = "a" if completed_ids else "w"

    with open(OUTPUT_PATH, file_mode, encoding="utf-8") as output_file:
        for index, sample in enumerate(remaining_samples, start=len(completed_ids) + 1):
            log(f"Running sample {index}/{NUM_SAMPLES}: {sample['id']}")

            result = dict(sample)
            result["dataset"] = sample.get("dataset", DATASET_NAME)
            result["gold_answer"] = sample.get("answer", sample.get("gold_answer", ""))
            result["model"] = MODEL_NAME
            result["agents"] = {}

            for agent_name in agent_names:
                log(f"Calling {agent_name} agent")

                agent_output = call_agent_with_recovery(
                    question=sample["question"],
                    agent_name=agent_name,
                    prompt_text=prompts[agent_name]
                )

                result["agents"][agent_name] = agent_output

                if AGENT_SLEEP_SECONDS > 0:
                    time.sleep(AGENT_SLEEP_SECONDS)

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    log(f"Done. Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
