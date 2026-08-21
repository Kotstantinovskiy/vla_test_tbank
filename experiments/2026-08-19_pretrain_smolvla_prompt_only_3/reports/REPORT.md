# Prompt-only v3: reproducible per-episode sampling noise

Same protocol as _2 (frozen pretrain, 10 goal tasks, true/wrong/
nonsense, seed 1000, all videos) with ONE change: the policy's flow
noise is reseeded from the episode seed (batch=1), so results are
reproducible regardless of process layout.

| condition | task | prompted with | _3 succ | 95% CI | _2 succ | success episodes (_3) |
|---|---:|---|---:|---|---:|---|
| true | 0 | `open the middle drawer of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 1 | `put the bowl on the stove` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 2 | `put the wine bottle on top of the cabine` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 3 | `open the top drawer and put the bowl ins` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 4 | `put the bowl on top of the cabinet` | 1/20 | [0.01, 0.24] | 1/20 | [0] |
| true | 5 | `push the plate to the front of the stove` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 6 | `put the cream cheese in the bowl` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 7 | `turn on the stove` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 8 | `put the bowl on the plate` | 0/20 | [0.00, 0.16] | 0/20 | — |
| true | 9 | `put the wine bottle on the rack` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 0 | `put the bowl on the stove` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 1 | `put the wine bottle on top of the cabine` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 2 | `open the top drawer and put the bowl ins` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 3 | `put the bowl on top of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 4 | `push the plate to the front of the stove` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 5 | `put the cream cheese in the bowl` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 6 | `turn on the stove` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 7 | `put the bowl on the plate` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 8 | `put the wine bottle on the rack` | 0/20 | [0.00, 0.16] | 0/20 | — |
| wrong | 9 | `open the middle drawer of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 0 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 1 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 2 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 3 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 4 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 5 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 6 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 7 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 8 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |
| nonsense | 9 | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 | — |

## Pooled per condition

- **true**: 1/200 [0.001, 0.028] (_2: 1/200)
- **wrong**: 0/200 [0.000, 0.019] (_2: 0/200)
- **nonsense**: 0/200 [0.000, 0.019] (_2: 0/200)
