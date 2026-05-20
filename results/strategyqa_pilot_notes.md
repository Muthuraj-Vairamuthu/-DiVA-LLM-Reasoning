# StrategyQA Pilot Notes

Dataset: StrategyQA  
Samples: 50  
Model: meta/llama-3.1-8b-instruct  
Agents: Literal, Skeptic, Creative, Evidence  

## Main Results

Literal Agent: 37/50 = 74%  
Skeptic Agent: 31/50 = 62%  
Creative Agent: 32/50 = 64%  
Evidence Agent: 38/50 = 76%  

Majority Vote: 36/50 = 72%  
Oracle Accuracy: 44/50 = 88%  
DiVA Rule Based: 43/50 = 86%  

## Disagreement Distribution

No disagreement: 32/50 = 64%  
Weak disagreement: 11/50 = 22%  
Strong disagreement: 7/50 = 14%  
Complete disagreement: 0/50 = 0%  

## Majority Accuracy by Disagreement Type

No disagreement: 27/32 = 84.38%  
Weak disagreement: 3/11 = 27.27%  
Strong disagreement: 6/7 = 85.71%  
Complete disagreement: NA  

## Overconfident Error Rate

Threshold: confidence >= 0.8  

Literal: 10/13 = 76.92%  
Skeptic: 7/19 = 36.84%  
Creative: 14/17 = 82.35%  
Evidence: 8/11 = 72.73%  

## DiVA Recovery Cases

Recovered majority-vote failures:

1. `strategyqa_13` using `strategyqa_creative_operational_override`
2. `strategyqa_14` using `strategyqa_grounded_minority_override`
3. `strategyqa_17` using `strategyqa_meta_judge_on_strong`
4. `strategyqa_24` using `strategyqa_grounded_minority_override`
5. `strategyqa_36` using `strategyqa_evidence_causal_override`
6. `strategyqa_41` using `strategyqa_creative_newsworthiness_override`
7. `strategyqa_48` using `strategyqa_literal_ability_override`
8. `strategyqa_49` using `strategyqa_grounded_minority_override`

Remaining errors:

1. `strategyqa_22` unanimous majority
2. `strategyqa_28` unanimous majority
3. `strategyqa_32` unanimous majority
4. `strategyqa_35` unanimous majority
5. `strategyqa_38` weak majority
6. `strategyqa_42` unanimous majority

## Key Takeaways

1. Majority voting is much weaker on StrategyQA than on GSM8K.
2. Weak disagreement is the main failure mode for majority voting.
3. Oracle accuracy is far above majority accuracy, so the system often contains the correct answer but selects the wrong one.
4. A task-aware DiVA policy recovers most majority-vote failures on this pilot set.
5. After improving disagreement handling, the remaining failures are mostly unanimous shared failures rather than routing failures.
6. Self reported confidence remains poorly calibrated and should not be treated as the main uncertainty signal.

## Current Research Claim

On StrategyQA, disagreement handling is not merely an uncertainty signal but an aggregation problem. Simple majority voting suppresses useful minority reasoning, especially in weak disagreement cases. A task-aware DiVA policy improves pilot accuracy from 72% to 86% by recovering grounded minority answers and better handling pragmatic question types.

## Next Step

Freeze this result as the current best StrategyQA pilot, then analyze the remaining unanimous failures and move to an ambiguity-focused dataset where abstention can be evaluated directly.
