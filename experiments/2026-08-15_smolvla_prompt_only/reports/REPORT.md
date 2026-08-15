# Result

The standalone rerun completed with the frozen
`crislmfroes/smolvla-libero-90` checkpoint at revision `418f9d0e5b...`.
No target demonstrations were loaded, no optimizer was constructed, and the
checkpoint weights were not modified.

Logical task IDs 0/1/2 map to `libero_goal` environment IDs 0/9/3. Every
rollout now records both IDs and verifies the environment instruction before
the policy prompt is injected. This in-place rerun supersedes results produced
with the earlier incorrect 0/1/2 environment mapping.

| Task | True prompt | Wrong-task prompt | Nonsense prompt |
|---:|---:|---:|---:|
| 0 | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| 1 | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| 2 | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| Mean over tasks | 0% | 0% | 0% |

For every per-task proportion, the Wilson 95% interval is `[0.000, 0.161]`.
All conditions use the same task dynamics and episode seeds 1000–1019; only the
policy prompt changes.

## Interpretation

The result confirms the prior-informed expectation: prompt-only transfer from
this LIBERO-90 checkpoint is at the floor on the three selected held-out
`libero_goal` tasks. Equal success under the language controls does **not** by
itself establish that the policy ignores language. Since the true-prompt
condition never succeeds, binary task success has no power to distinguish
prompt sensitivity here. Action divergence, state-progress metrics, or a task
on which the checkpoint has nonzero base success would be required for that
claim.

## Artifacts

- `results/raw/{true,wrong,nonsense}.json`: all 180 episode records;
- `results/summary/metrics.csv`: nine per-task/per-condition estimates;
- `results/summary/prompt_controls.png`: prompt-control plot;
- `results/media/gifs/task_*_true.gif`: representative true-prompt rollouts;
- `results/summary/trackio_manifest.json`: logged Trackio objects and local run;
- ignored but locally retained: MP4 rollouts, runtime logs, LIBERO config, and
  `artifacts/trackio/smolvla-prompt-only.db`.
