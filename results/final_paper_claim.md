# Final Locked Paper Claim

## Primary Claim

Cognitive diversity does not automatically improve reasoning accuracy. Instead, disagreement patterns reveal task-dependent reliability and ambiguity signals that are hidden by simple majority voting and self-reported confidence.

## Dataset-Specific Claim

1. On GSM8K, disagreement strength behaves like a reliability signal and simple majority voting is already strong.
2. On StrategyQA, weak disagreement is especially dangerous because the correct answer often appears in the minority, and a task-aware DiVA policy improves pilot accuracy from 72.00% to 86.00% without harming GSM8K.
3. On AmbigQA, disagreement is most useful as an abstention signal: a simple ambiguity-aware DiVA policy improves pilot ambiguity-aware accuracy from 22.00% majority accuracy to 40.00%, though abstention calibration remains limited.

## Strong Short-Paper Framing

Disagreement is not a single universal signal. Its meaning depends on the task:

1. reliability signal on math reasoning,
2. minority-recovery signal on commonsense reasoning,
3. abstention signal on genuinely ambiguous questions.

## Contributions Draft

1. A four-agent cognitively diverse reasoning setup with literal, skeptic, creative, and evidence roles.
2. A task-aware DiVA aggregation policy that improves StrategyQA pilot accuracy from 72.00% to 86.00% while preserving GSM8K performance at 96.00%.
3. An ambiguity-aware abstention evaluation on AmbigQA showing that disagreement can be used to avoid unsafe answers, improving pilot ambiguity-aware accuracy from 22.00% to 40.00%.
4. A cross-dataset analysis showing that self-reported confidence is poorly calibrated and that disagreement has task-dependent meaning.

## Claims To Avoid

1. Do not claim DiVA is universally better than majority on all tasks.
2. Do not claim the current meta-judge is a reliable general recovery mechanism.
3. Do not claim self-reported confidence is a strong uncertainty estimator.
4. Do not claim AmbigQA is solved.
5. Do not overclaim from the 50-example pilot setting.

## Next Method Step

The next method step is to improve AmbigQA abstention calibration, especially for:

1. ambiguous cases where all agents still guess confidently,
2. answerable cases where DiVA abstains unnecessarily,
3. near-match alias and normalization failures that currently count as wrong.
