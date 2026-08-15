# Prior-informed expectation

Recorded before this standalone rerun, but **not** presented as a blind
preregistration: an earlier, separate experiment had already evaluated the same
checkpoint and tasks.

Expected result: the frozen LIBERO-90 checkpoint will score near zero on all
three selected held-out `libero_goal` tasks under the correct prompt. Logical
IDs 0/1/2 map to environment IDs 0/9/3. Wrong-task and
nonsense prompts are also expected near zero, so this particular success metric
will probably be unable to establish whether the policy reads language.

The rerun remains useful because it makes the prompt-only point independently
reproducible: its scripts, raw rollouts, videos, summaries, GIFs, and Trackio
database are owned by this directory and do not reuse the earlier result files.
