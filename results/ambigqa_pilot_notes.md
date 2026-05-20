# AmbigQA Pilot Notes

Dataset: AmbigQA  
Samples: 50  
Composition: 25 ambiguous, 25 answerable  
Model: meta/llama-3.1-8b-instruct  
Agents: Literal, Skeptic, Creative, Evidence  

## Main Results

Literal Agent: 10/50 = 20%  
Skeptic Agent: 9/50 = 18%  
Creative Agent: 8/50 = 16%  
Evidence Agent: 9/50 = 18%  

Majority Vote: 11/50 = 22%  
Oracle Accuracy: 12/50 = 24%  
DiVA Answered Accuracy: 7/24 = 29.17%  
DiVA Coverage: 24/50 = 48%  
DiVA Abstention Rate: 26/50 = 52%  
DiVA Ambiguity-Aware Accuracy: 20/50 = 40%  

## Ambiguity Metrics

Correct abstentions: 13/25 = 52%  
Wrong abstentions: 13/25 = 52%  

## Disagreement Distribution

No disagreement: 18/50 = 36%  
Weak disagreement: 13/50 = 26%  
Strong disagreement: 17/50 = 34%  
Complete disagreement: 2/50 = 4%  

## Majority Accuracy by Disagreement Type

No disagreement: 5/18 = 27.78%  
Weak disagreement: 3/13 = 23.08%  
Strong disagreement: 2/17 = 11.76%  
Complete disagreement: 1/2 = 50%  

## Key Takeaways

1. AmbigQA is substantially harder than GSM8K and StrategyQA.
2. Raw answering performance is very low for all agents, for majority voting, and even for oracle selection.
3. DiVA responds to ambiguity by abstaining frequently rather than forcing a guess.
4. The current ambiguity-aware DiVA improves utility over raw majority when abstention is treated as a valid action.
5. However, correct and wrong abstentions are currently balanced, so abstention quality still needs improvement.

## Current Research Claim

On ambiguity-sensitive questions, disagreement is not only an aggregation signal but also an abstention signal. A simple ambiguity-aware DiVA policy increases ambiguity-aware accuracy from 22% majority accuracy to 40% by abstaining on many difficult cases, but abstention calibration remains a major open problem.

## Next Step

Improve abstention calibration on AmbigQA, especially on unanimous ambiguous cases where all agents confidently converge on a single answer despite multiple valid interpretations.
