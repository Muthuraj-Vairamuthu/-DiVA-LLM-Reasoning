# GSM8K Pilot Notes

Dataset: GSM8K  
Samples: 50  
Model: meta/llama-3.1-8b-instruct  
Agents: Literal, Skeptic, Creative, Evidence  

## Main Results

Literal Agent: 43/50 = 86%  
Skeptic Agent: 29/50 = 58%  
Creative Agent: 35/50 = 70%  
Evidence Agent: 44/50 = 88%  

Majority Vote: 48/50 = 96%  
Oracle Accuracy: 49/50 = 98%  
DiVA Rule Based: 48/50 = 96%  

## Disagreement Distribution

No disagreement: 19/50 = 38%  
Weak disagreement: 17/50 = 34%  
Strong disagreement: 12/50 = 24%  
Complete disagreement: 2/50 = 4%  

## Majority Accuracy by Disagreement Type

No disagreement: 19/19 = 100%  
Weak disagreement: 17/17 = 100%  
Strong disagreement: 11/12 = 91.67%  
Complete disagreement: 1/2 = 50%  

## Overconfident Error Rate

Threshold: confidence >= 0.8  

Literal: 7/7 = 100%  
Skeptic: 18/21 = 85.71%  
Creative: 15/15 = 100%  
Evidence: 6/6 = 100%  

## Key Takeaways

1. Majority voting is very strong on GSM8K.
2. Evidence Agent is the strongest individual agent.
3. Skeptic Agent performs poorly on straightforward math reasoning.
4. Disagreement strength correlates with reduced reliability.
5. Self reported confidence is poorly calibrated, since most wrong answers are still high confidence.
6. The main value of DiVA on GSM8K is not raw accuracy improvement, but uncertainty estimation through disagreement patterns.

## Current Research Claim

On GSM8K, cognitively diverse agents do not clearly outperform majority voting, but their disagreement patterns provide a useful reliability signal. In particular, complete disagreement is much riskier than no or weak disagreement. Self reported confidence is unreliable, making disagreement based uncertainty more useful than confidence alone.

## Next Step

Move to a commonsense reasoning dataset such as StrategyQA to test whether the same disagreement reliability pattern holds beyond math reasoning.