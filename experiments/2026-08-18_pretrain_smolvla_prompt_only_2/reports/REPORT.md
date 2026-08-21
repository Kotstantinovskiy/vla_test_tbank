# Official-pretrain prompt-only report

The frozen official-data pretrain checkpoint (30k DDP steps from
`lerobot/smolvla_base`, seen positive control 20/20) is evaluated with
zero target demonstrations and zero optimizer steps on all ten
suite-local `libero_goal` tasks. Logical task IDs equal environment IDs.

| task | instruction | true | wrong | nonsense |
|---:|---|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 1 | `put the bowl on the stove` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 2 | `put the wine bottle on top of the cabinet` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 3 | `open the top drawer and put the bowl inside` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 4 | `put the bowl on top of the cabinet` | 1/20 (0.050) | 0/20 (0.000) | 0/20 (0.000) |
| 5 | `push the plate to the front of the stove` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 6 | `put the cream cheese in the bowl` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 7 | `turn on the stove` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 8 | `put the bowl on the plate` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |
| 9 | `put the wine bottle on the rack` | 0/20 (0.000) | 0/20 (0.000) | 0/20 (0.000) |

## Interpretation

Mean success by prompt condition: true=0.005, wrong=0.000, nonsense=0.000.

Unlike every earlier zero-shot run in this repository, the evaluation
pipeline behind these numbers has a passing seen-task positive control
(20/20), so floor results here measure generalization, not pipeline
defects.
The controls are identifiable above the floor; compare the per-task true, wrong-task, and nonsense rates rather than only their global means.
