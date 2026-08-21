# Prior-informed expectation

Recorded on 2026-08-18 before running any rollout of this experiment.
Strongly prior-informed: the sibling experiment
`2026-08-18_pretrain_smolvla_prompt_only` already executed the identical
protocol on the identical checkpoint and scored true 1/200 (single success:
task 4 episode 4), wrong 0/200, nonsense 0/200.

This run exists to capture full rollout video for every episode of every
condition (the sibling recorded only episode 0 per task and missed its single
success on video). It is an independent execution: environment seeds and
initial states repeat exactly, but the torch RNG stream (flow-matching action
noise) does not, so outcomes are resampled.

Expected result: 0-3 successes out of 200 under `true` (the sibling's rate
resampled), 0 under both controls. Task 4 is the most likely locus but a
repeat there is not guaranteed. Mean over tasks <= 0.015.

What would be surprising: >= 5/200 under `true` (would suggest the sibling
underestimated the rate), or any control success.

Both experiments' numbers stand independently; neither overwrites the other.
