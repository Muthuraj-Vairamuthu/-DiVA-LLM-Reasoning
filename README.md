# DiVA: Diverse Agents for Ambiguity-Aware QA

DiVA is a multi-agent question answering pipeline that studies how **disagreement among LLM agents** can be used to decide **when to answer** and **when to abstain**.

The project uses four role-based agents:

- `literal`
- `skeptic`
- `creative`
- `evidence`

Each agent answers the same question, and DiVA aggregates their outputs into a final system decision.

## Main Idea

Instead of treating disagreement only as a voting problem, this project asks:

- When does disagreement signal that an answer is still reliable?
- When does disagreement signal that the model should abstain?

This is especially useful for:

- ambiguity-sensitive QA
- missing-context questions
- abstention-aware evaluation

## Current Project Direction

The strongest current direction is:

- **disagreement as an abstention signal under ambiguity**

The main datasets used are:

- `AmbigQA`
- `SituatedQA-temp` (raw vs clarified setup)
- `CondAmbigQA-2K` as supporting ambiguity-recognition evidence

## Repository Structure

```text
src/
  run_agents.py          # Runs all 4 agents and stores raw outputs
  diva.py                # Applies DiVA aggregation / abstention rules
  evaluate_agents.py     # Computes agent, majority, oracle, and disagreement metrics
  llm_clients.py         # NVIDIA / Groq API clients
  load_ambigqa.py
  load_condambigqa2k.py
  load_situatedqa.py
  load_strategyqa.py
  combine_situatedqa.py  # Combines raw + clarified SituatedQA results

prompts/
  literal.txt
  skeptic.txt
  creative.txt
  evidence.txt

data/                    # Processed dataset files
outputs/                 # Saved per-question agent outputs and DiVA outputs
results/                 # CSV summaries and analysis artifacts
logs/                    # Evaluation logs
```

## Installation

Create and activate a Python 3.11 virtual environment, then install dependencies:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python3.11 -m pip install --upgrade pip
.venv/bin/python3.11 -m pip install numpy pandas datasets python-dotenv requests certifi tqdm openai anthropic google-generativeai huggingface_hub
```

## Environment

Create a `.env` file with your API key:

```bash
NVIDIA_API_KEY=your_key_here
```

Optional provider switching is supported through environment variables in `src/llm_clients.py`.

## Basic Workflow

### 1. Prepare a dataset

Examples:

```bash
SAMPLE_SIZE=800 .venv/bin/python3.11 src/load_ambigqa.py
SAMPLE_SIZE=800 .venv/bin/python3.11 src/load_condambigqa2k.py
SAMPLE_SIZE=400 SITUATEDQA_CONFIG=temp .venv/bin/python3.11 src/load_situatedqa.py
```

### 2. Run agents

```bash
set -a && source .env && set +a && \
PYTHONUNBUFFERED=1 RESUME_RUN=1 AGENT_SLEEP_SECONDS=2 \
SAMPLE_SIZE=800 DATASET_NAME=ambigqa \
MODEL_NAME=meta/llama-3.1-8b-instruct \
.venv/bin/python3.11 src/run_agents.py
```

### 3. Run DiVA

```bash
PYTHONUNBUFFERED=1 DISABLE_META_JUDGE=1 \
SAMPLE_SIZE=800 DATASET_NAME=ambigqa \
MODEL_NAME=meta/llama-3.1-8b-instruct \
.venv/bin/python3.11 src/diva.py
```

### 4. Evaluate

```bash
PYTHONUNBUFFERED=1 SAMPLE_SIZE=800 DATASET_NAME=ambigqa \
MODEL_NAME=meta/llama-3.1-8b-instruct \
.venv/bin/python3.11 src/evaluate_agents.py
```

## Supported Models

The project has been run with:

- `meta/llama-3.1-8b-instruct`
- `google/gemma-2-2b-it`
- `mistralai/mistral-7b-instruct-v0.3`

Separate runs can be tagged with:

```bash
RUN_TAG=gemma2_2b
RUN_TAG=mistral7b
```

## Evaluation Concepts

### Majority

Plain majority voting over the 4 agent final answers.

### Oracle

Counts a question as correct if **at least one agent** produced the correct behavior.

### DiVA

A rule-based aggregation policy that uses:

- disagreement level
- agent abstentions
- confidence
- missing-information signals

### Disagreement Types

- `none` — all agents agree
- `weak` — 3 agree, 1 differs
- `strong` — split / no strong consensus
- `complete` — all answers differ

## Main Findings So Far

### AmbigQA

DiVA improves ambiguity-aware performance substantially over forced-answer majority voting by abstaining on risky cases.

### SituatedQA-temp

The dataset is used in two paired forms:

- `raw` — under-specified question, where abstention is often the right behavior
- `clarified` — edited question with time context added, where answering is expected

This gives a clean experimental setup for:

- missing context -> abstain
- added context -> answer

## License

This repository is for research use. Please check the original licenses of any datasets and models you use.
