# Ambiguity Dataset Plan

## Candidate Dataset

AmbigQA or a small custom ambiguity set.

## Purpose

Test whether DiVA can do more than choose among agent answers. The ambiguity phase should evaluate whether disagreement can support abstention when the question is genuinely ambiguous, underspecified, or multi-answer in nature.

## Why This Matters

1. GSM8K mainly tests numerical reasoning and aggregation.
2. StrategyQA mainly tests task-aware answer selection.
3. An ambiguity dataset is where abstention becomes a primary contribution rather than a side effect.

## What To Measure

1. Abstention rate
2. Coverage
3. Selective accuracy
4. Accuracy counting abstentions as wrong
5. Correct abstention rate
6. Wrong answer rate
7. Overconfident wrong answer rate

## What To Add In The Pipeline

1. A normalized abstention label for outputs such as `unknown`, `cannot determine`, `not enough information`, and blank answers
2. Evaluation that distinguishes correct abstentions from wrong abstentions
3. Dataset-specific answer normalization for ambiguous or multi-answer questions
4. A DiVA policy that is allowed to abstain instead of forcing a majority answer

## Research Role In The Paper

1. GSM8K shows disagreement as a reliability signal in math reasoning.
2. StrategyQA shows disagreement as a task-aware aggregation problem in commonsense reasoning.
3. AmbigQA or a custom ambiguity set shows disagreement as an abstention signal under genuine ambiguity.

## Immediate Next Step

Select the ambiguity dataset and define the exact evaluation protocol before collecting runs, so the abstention metrics are part of the design rather than added afterward.
