# Prior expectation

Recorded before any Qwen3.5 progress-critic optimization steps.

The deliberately simple temporal target should be learnable: training and task-held-out
validation cross-entropy should fall during 2,000 steps. However, because supervision is
absolute normalized time on successful expert demonstrations only, the critic may learn a
mixture of visual task progress and expert-trajectory timing. No claim about checkpoint
ranking is made in this experiment.

The 50-step benchmark is an engineering gate, not a scientific result. It measures whether
Qwen3.5-4B LoRA plus a 32-bin head fits on GPU 0 without gradient checkpointing, records peak
VRAM and throughput, and projects the wall time for the declared 2,000-step run. Full training
will not be launched as part of this preparation turn.
