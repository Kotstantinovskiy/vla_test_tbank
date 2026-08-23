# Full training report

The Qwen3.5-4B progress critic completed 2,000 optimizer steps on physical GPU 0 with batch
size 4, seed 1000, frozen vision parameters, and gradient checkpointing disabled.

| Measurement | Result |
|---|---:|
| Total wall time | 2,894.9 s (48.25 min) |
| Peak reserved VRAM | 34.61 GiB |
| Validation points | 21 |
| Initial validation CE / MAE | 5.0872 / 0.3193 |
| Best validation CE | 2.1963 at step 1,300 |
| Final validation CE / MAE | 2.2159 / 0.0687 |
| Final exact-bin accuracy | 0.1992 |
| Last train loss | 1.7936 |

Checkpoints were saved at every 200 steps through `002000`. The final checkpoint contains
the LoRA adapter (`adapter_model.safetensors`, 259,794,448 bytes), adapter config, and progress
head (`progress_head.safetensors`, 164,136 bytes). Validation loss plateaus after roughly step
1,300, so downstream model selection should compare the pinned step-1,300 and final step-2,000
critics on a separate ranking evaluation rather than selecting from this validation set alone.

Checkpoint ranking and Robometer comparison remain outside this experiment's executed scope.
