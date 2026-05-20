# Ambiguity Section Notes

## Role In The Paper

AmbigQA is the third phase of the story:

1. GSM8K: disagreement as a reliability signal in math reasoning
2. StrategyQA: disagreement as a task-aware aggregation problem in commonsense reasoning
3. AmbigQA: disagreement as an abstention signal under genuine ambiguity

## Main Findings

1. AmbigQA is much harder than the first two datasets.
2. Majority vote performs poorly at 22%.
3. Oracle accuracy is only 24%, so the correct answer is often not stably represented across agents.
4. DiVA abstains on 52% of cases and reaches 40% ambiguity-aware accuracy.
5. Correct abstention rate and wrong abstention rate are both 52%, showing that abstention is active but not yet well calibrated.

## Interpretation

The ambiguity dataset changes the research question. The main issue is no longer just whether DiVA can pick the right minority answer. Instead, the issue is whether DiVA can decide when not to answer at all. This makes abstention behavior central rather than optional.

## Honest Limitation

The current AmbigQA policy is only a first abstention baseline. It successfully reduces forced guessing, but it still abstains on many answerable cases and still answers many ambiguous cases with overconfidence.

## Best Paper Framing

The current pilot supports a three-part claim:

1. Disagreement strength tracks reliability on GSM8K.
2. Task-aware minority-sensitive aggregation matters on StrategyQA.
3. Ambiguity-aware abstention becomes essential on AmbigQA, where naive answering is poor and disagreement often reflects genuinely unresolved question structure.

## Immediate Follow-Up

Refine the ambiguity policy using stronger abstention triggers for unanimous ambiguous cases and more conservative answer retention for answerable cases with grounded agreement.
