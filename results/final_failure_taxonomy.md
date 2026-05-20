# Final Failure Taxonomy

## Category 1. Minority Suppression Failures

Definition:
The correct answer is present among the agents, but simple majority voting suppresses it.

Most visible on:
StrategyQA

Evidence:
1. StrategyQA majority is 72.00% while oracle is 88.00%.
2. Weak disagreement majority accuracy is only 27.27%.
3. DiVA recovers 8 of the 14 majority-vote failures on the 50-example pilot.

Representative recovered cases:
1. `strategyqa_14`
2. `strategyqa_24`
3. `strategyqa_36`
4. `strategyqa_41`
5. `strategyqa_48`
6. `strategyqa_49`

Interpretation:
This is the strongest evidence that disagreement does not have a fixed meaning across tasks. On StrategyQA, weak disagreement is not a safe majority signal.

## Category 2. Shared Knowledge Failures

Definition:
Most or all agents fail together because they share the same factual gap, bad assumption, or mistaken world model.

Most visible on:
Remaining StrategyQA errors and much of AmbigQA

Evidence:
1. After the current StrategyQA improvements, most remaining errors are unanimous or near-unanimous failures.
2. Several AmbigQA failures involve all agents converging on a plausible but unsupported answer.

Representative cases:
1. `strategyqa_22`
2. `strategyqa_28`
3. `strategyqa_32`
4. `strategyqa_35`
5. `strategyqa_42`
6. `ambigqa_25`
7. `ambigqa_45`

Interpretation:
Better aggregation helps, but it cannot solve cases where the whole agent pool shares the same wrong belief.

## Category 3. Genuine Ambiguity And Underspecification

Definition:
The question admits multiple reasonable interpretations or valid answers, so the system should often abstain instead of forcing a single response.

Most visible on:
AmbigQA

Evidence:
1. Majority accuracy is only 22.00%.
2. Oracle accuracy is only 24.00%.
3. DiVA abstains on 52.00% of cases.
4. DiVA ambiguity-aware accuracy rises to 40.00%.

Representative cases:
1. `ambigqa_0`
2. `ambigqa_21`
3. `ambigqa_30`
4. `ambigqa_31`
5. `ambigqa_44`
6. `ambigqa_49`

Interpretation:
In this setting, disagreement is valuable mainly because it signals that answering may be unsafe, not because it identifies a reliable minority answer.

## Category 4. Overconfident Wrong Answers

Definition:
Agents produce incorrect answers with high self-reported confidence.

Visible on:
All three datasets

Evidence:
1. GSM8K overconfident error rates remain extremely high across agents.
2. StrategyQA still shows high overconfident error rates, especially for literal, creative, and evidence agents.
3. AmbigQA also shows broad high-confidence failure despite heavy uncertainty in the task itself.

Interpretation:
Self-reported confidence is not a reliable primary decision signal for DiVA.

## Main Takeaway

The project now supports a four-part failure taxonomy:

1. Minority suppression failures
2. Shared knowledge failures
3. Genuine ambiguity and underspecification
4. Overconfident wrong answers

This taxonomy helps explain why a single aggregation rule is unlikely to work well across all reasoning tasks.
