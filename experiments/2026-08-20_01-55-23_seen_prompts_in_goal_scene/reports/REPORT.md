# seen_prompts_in_goal_scene: prompted-predicate success

Only binary success is aggregated. Every rollout video remains on disk
for manual behavior inspection. Environment and policy-noise seeds are
recorded per episode.

| label | env instruction | block | policy prompt | success | 95% CI | env success |
|---|---|---|---|---:|---|---:|
| `true_goal__host` | `open the middle drawer of the cabinet` | true_goal | `open the middle drawer of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen_prompt__seen_0` | `open the middle drawer of the cabinet` | seen_prompt | `close the top drawer of the cabinet and put the black bowl on top of it` | 7/20 | [0.18, 0.57] | 0/20 |
| `seen_prompt__seen_1` | `open the middle drawer of the cabinet` | seen_prompt | `put the black bowl in the top drawer of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen_prompt__seen_2` | `open the middle drawer of the cabinet` | seen_prompt | `open the bottom drawer of the cabinet` | 1/20 | [0.01, 0.24] | 0/20 |
| `seen_prompt__seen_3` | `open the middle drawer of the cabinet` | seen_prompt | `open the top drawer of the cabinet` | 10/20 | [0.30, 0.70] | 0/20 |
| `seen_prompt__seen_4` | `open the middle drawer of the cabinet` | seen_prompt | `open the top drawer of the cabinet and put the bowl in it` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen_prompt__seen_5` | `open the middle drawer of the cabinet` | seen_prompt | `put the black bowl on the plate` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen_prompt__seen_6` | `open the middle drawer of the cabinet` | seen_prompt | `put the black bowl on top of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen_prompt__seen_7` | `open the middle drawer of the cabinet` | seen_prompt | `turn on the stove` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen_prompt__seen_8` | `open the middle drawer of the cabinet` | seen_prompt | `put the wine bottle on the wine rack` | 0/20 | [0.00, 0.16] | 0/20 |
| `nonsense__host` | `open the middle drawer of the cabinet` | nonsense | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 |
