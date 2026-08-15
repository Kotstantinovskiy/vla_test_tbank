# Predictions locked before evaluation

Primary prediction: reducing `n_action_steps` will improve success by reducing
open-loop compounding error. The largest gain should occur on task 1, where the
wine bottle must be placed on the rack with centimetre-scale precision. Task 2
should improve noticeably as well. The direction should be similar for
`k ∈ {5, 10, 25}`, because this is an inference intervention applied to the same
frozen baseline weights rather than additional learning.

The expected curve is not assumed to be monotonic all the way to one action:
frequent replanning reduces stale actions, but `n_action_steps=1` may introduce
jitter and repeatedly resample flow-matching noise. The qualitative forecast is
therefore `5` or `10` best, `25` intermediate, and the checkpoint default `50`
worst on precision-sensitive tasks.

There is a plausible chance that zero-shot moves above zero under frequent
replanning. If any true-prompt zero-shot point succeeds, the predeclared
diagnostic is to evaluate wrong-task and nonsense prompts with identical
horizons, episode seeds, and initial states. If true-prompt zero-shot remains at
zero, those controls are skipped because binary success would still be
non-identifying at the floor.

Protocol interpretation is fixed in advance:

- the original Task 1 baseline curve remains frozen and is copied only as a
  reference;
- all new points use the exact existing baseline weights and change only
  `config.n_action_steps` at inference;
- the rerun at `n_action_steps=50` is an internal reproducibility anchor;
- delta versus the paired rerun at 50 estimates the inference contribution;
- no point in this experiment is presented as a new adaptation/training method.
