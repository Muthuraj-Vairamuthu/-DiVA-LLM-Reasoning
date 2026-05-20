# Final Results Tables

## Table 1. Main Results

| Dataset | Samples | Literal | Skeptic | Creative | Evidence | Majority | Oracle | DiVA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 50 | 86.00% | 58.00% | 70.00% | 88.00% | 96.00% | 98.00% | 96.00% |
| StrategyQA | 50 | 74.00% | 62.00% | 64.00% | 76.00% | 72.00% | 88.00% | 86.00% |

## Table 2. AmbigQA Abstention-Aware Results

AmbigQA should not be forced into the same reporting format as GSM8K and StrategyQA because abstention is part of the target behavior.

| Dataset | Samples | Literal | Skeptic | Creative | Evidence | Majority | Oracle | DiVA Answered Accuracy | DiVA Coverage | DiVA Ambiguity-Aware Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AmbigQA | 50 | 20.00% | 18.00% | 16.00% | 18.00% | 22.00% | 24.00% | 29.17% | 48.00% | 40.00% |

## Table 3. Disagreement Distribution And Reliability

| Dataset | None | Weak | Strong | Complete | Majority Accuracy on Weak Disagreement |
|---|---:|---:|---:|---:|---:|
| GSM8K | 38.00% | 34.00% | 24.00% | 4.00% | 100.00% |
| StrategyQA | 64.00% | 22.00% | 14.00% | 0.00% | 27.27% |
| AmbigQA | 36.00% | 26.00% | 34.00% | 4.00% | 23.08% |

## Table 4. AmbigQA Abstention Metrics

| Metric | Value |
|---|---:|
| Ambiguous questions | 25/50 = 50.00% |
| Answerable questions | 25/50 = 50.00% |
| DiVA answered | 24/50 = 48.00% |
| DiVA abstained | 26/50 = 52.00% |
| Correct abstentions | 13/25 = 52.00% |
| Wrong abstentions | 13/25 = 52.00% |
| Abstained when majority was wrong | 22 |
| Abstained when majority was correct | 4 |

## Summary

1. GSM8K shows that majority voting is already strong and DiVA does not improve raw pilot accuracy.
2. StrategyQA shows the clearest method gain: a task-aware DiVA policy improves pilot accuracy from 72.00% to 86.00%.
3. AmbigQA shows that ambiguity changes the problem from answer selection to abstention, and a simple ambiguity-aware DiVA policy improves pilot ambiguity-aware accuracy from 22.00% majority accuracy to 40.00%.
