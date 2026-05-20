# StrategyQA Failure Analysis

## Majority Failures Recovered By DiVA

1. `strategyqa_13`: operational requirement framing
   Question: `Is electricity necessary to balance an account in Microsoft Excel?`
   Fix: DiVA chose the creative minority because the question depends on real-world use of Excel, not abstract arithmetic alone.

2. `strategyqa_14`: grounded minority against speculative majority
   Question: `Is a pound sterling valuable?`
   Fix: DiVA preferred the evidence minority because the majority answered from generic plausibility rather than question-grounded support.

3. `strategyqa_17`: strong disagreement resolved by StrategyQA meta judge
   Question: `Does Coast to Coast AM have more longevity than the Rush Limbaugh show?`
   Fix: DiVA used the StrategyQA-specific judge to recover the correct answer from a split case.

4. `strategyqa_24`: grounded minority against speculative majority
   Question: `Can Kit & Kaboodle hypothetically help someone past the Underworld gates?`
   Fix: DiVA rejected a majority built on invented or weakly supported backstory.

5. `strategyqa_36`: causal dependency reasoning
   Question: `Would it be hard to get toilet paper if there were no loggers?`
   Fix: DiVA preferred the evidence minority because it identified a concrete supply-chain dependency.

6. `strategyqa_41`: newsworthiness framing
   Question: `Would a broadcast from Spirit make the news in 2020?`
   Fix: DiVA preferred the creative minority because the question hinges on surprise and newsworthiness.

7. `strategyqa_48`: practical ability framing
   Question: `Would a Germaphobia be able to participate in Judo?`
   Fix: DiVA preferred the literal minority because the question is about practical ability, not bare physical possibility.

8. `strategyqa_49`: grounded minority against speculative majority
   Question: `Could Eddie Hall hypothetically deadlift the world's largest cheeseburger?`
   Fix: DiVA preferred the evidence minority because the majority depended on unsupported assumptions about size and weight.

## Remaining Errors

1. `strategyqa_22`
2. `strategyqa_28`
3. `strategyqa_32`
4. `strategyqa_35`
5. `strategyqa_38`
6. `strategyqa_42`

## Error Pattern Summary

1. Four of the six remaining failures are unanimous-majority failures.
2. This means the main bottleneck is no longer disagreement routing in many cases.
3. The remaining errors are more consistent with shared knowledge gaps, shared bad assumptions, or genuinely underspecified questions.
4. `strategyqa_38` remains a disagreement case and is the clearest target for the next StrategyQA refinement.

## Research Interpretation

The current DiVA policy solves much of the selection problem on StrategyQA. Once disagreement-aware routing is improved, the dominant remaining failure mode shifts from minority suppression to shared model failure. This transition is useful for the paper because it shows that better aggregation alone is not enough for all commonsense questions, especially when multiple agents share the same mistaken world model.
