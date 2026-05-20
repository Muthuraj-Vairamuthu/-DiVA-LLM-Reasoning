# Cross-Dataset Pilot Comparison

## Main Accuracy Table

| Dataset | Samples | Literal | Skeptic | Creative | Evidence | Majority | Oracle | DiVA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 50 | 86% | 58% | 70% | 88% | 96% | 98% | 96% |
| StrategyQA | 50 | 74% | 62% | 64% | 76% | 72% | 88% | 86% |

## Main Interpretation

1. GSM8K and StrategyQA behave very differently under multi-agent aggregation.
2. On GSM8K, majority voting is already strong and DiVA does not improve raw accuracy.
3. On StrategyQA, majority voting is weak relative to oracle, and DiVA recovers most of that gap.
4. This supports a task-aware view of disagreement rather than a one-rule-fits-all aggregation strategy.

## Disagreement Comparison

GSM8K:

1. No disagreement: 38% of samples, majority accuracy 100%
2. Weak disagreement: 34% of samples, majority accuracy 100%
3. Strong disagreement: 24% of samples, majority accuracy 91.67%
4. Complete disagreement: 4% of samples, majority accuracy 50%

StrategyQA:

1. No disagreement: 64% of samples, majority accuracy 84.38%
2. Weak disagreement: 22% of samples, majority accuracy 27.27%
3. Strong disagreement: 14% of samples, majority accuracy 85.71%
4. Complete disagreement: 0% of samples

## Result Narrative

1. On GSM8K, disagreement behaves like a reliability signal: more disagreement generally means lower confidence in the majority answer.
2. On StrategyQA, weak disagreement is the critical failure mode: a three-agent majority is often wrong while a minority agent is correct.
3. Therefore, disagreement is useful on both datasets, but its meaning is task dependent.

## Best Current Paper Claim

Cognitive diversity does not automatically improve reasoning accuracy. Instead, disagreement patterns expose task-dependent reliability structure. On GSM8K, disagreement strength tracks reliability degradation. On StrategyQA, majority voting suppresses useful minority reasoning, and a task-aware DiVA policy improves pilot accuracy from 72% to 86% without harming GSM8K.

## Next Phase

Move from disagreement-aware aggregation to ambiguity-aware abstention by evaluating on AmbigQA or a custom ambiguity-focused dataset.
