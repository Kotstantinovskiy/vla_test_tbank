# seen_semantic_paraphrases: prompted-predicate success

Only binary success is aggregated. Every rollout video remains on disk
for manual behavior inspection. Environment and policy-noise seeds are
recorded per episode.

| label | env instruction | block | policy prompt | success | 95% CI | env success |
|---|---|---|---|---:|---|---:|
| `exact__task_0` | `turn on the stove` | exact | `turn on the stove` | 20/20 | [0.84, 1.00] | 20/20 |
| `paraphrase__task_0` | `turn on the stove` | paraphrase | `switch the stove on` | 19/20 | [0.76, 0.99] | 19/20 |
| `exact__task_1` | `put the frying pan on the stove` | exact | `put the frying pan on the stove` | 17/20 | [0.64, 0.95] | 17/20 |
| `paraphrase__task_1` | `put the frying pan on the stove` | paraphrase | `place the frying pan on the stove` | 17/20 | [0.64, 0.95] | 17/20 |
| `exact__task_2` | `close the top drawer of the cabinet` | exact | `close the top drawer of the cabinet` | 20/20 | [0.84, 1.00] | 20/20 |
| `paraphrase__task_2` | `close the top drawer of the cabinet` | paraphrase | `shut the cabinet's top drawer` | 9/20 | [0.26, 0.66] | 9/20 |
| `exact__task_3` | `put the black bowl on top of the cabinet` | exact | `put the black bowl on top of the cabinet` | 19/20 | [0.76, 0.99] | 19/20 |
| `paraphrase__task_3` | `put the black bowl on top of the cabinet` | paraphrase | `place the black bowl atop the cabinet` | 0/20 | [0.00, 0.16] | 0/20 |
| `exact__task_4` | `put the black bowl on the plate` | exact | `put the black bowl on the plate` | 15/20 | [0.53, 0.89] | 15/20 |
| `paraphrase__task_4` | `put the black bowl on the plate` | paraphrase | `place the black bowl onto the plate` | 8/20 | [0.22, 0.61] | 8/20 |
| `exact__task_5` | `put the wine bottle on the wine rack` | exact | `put the wine bottle on the wine rack` | 8/20 | [0.22, 0.61] | 8/20 |
| `paraphrase__task_5` | `put the wine bottle on the wine rack` | paraphrase | `place the wine bottle onto the wine rack` | 3/20 | [0.05, 0.36] | 3/20 |
| `exact__task_6` | `open the top drawer of the cabinet and put the bowl in it` | exact | `open the top drawer of the cabinet and put the bowl in it` | 13/20 | [0.43, 0.82] | 13/20 |
| `paraphrase__task_6` | `open the top drawer of the cabinet and put the bowl in it` | paraphrase | `open the cabinet's top drawer and place the bowl inside it` | 0/20 | [0.00, 0.16] | 0/20 |
| `exact__task_7` | `open the microwave` | exact | `open the microwave` | 19/20 | [0.76, 0.99] | 19/20 |
| `paraphrase__task_7` | `open the microwave` | paraphrase | `open the microwave door` | 17/20 | [0.64, 0.95] | 17/20 |
| `exact__task_8` | `pick up the alphabet soup and put it in the basket` | exact | `pick up the alphabet soup and put it in the basket` | 2/20 | [0.03, 0.30] | 2/20 |
| `paraphrase__task_8` | `pick up the alphabet soup and put it in the basket` | paraphrase | `pick up the alphabet soup, then place it in the basket` | 0/20 | [0.00, 0.16] | 0/20 |
| `exact__task_9` | `pick up the book and place it in the left compartment of the caddy` | exact | `pick up the book and place it in the left compartment of the caddy` | 12/20 | [0.39, 0.78] | 12/20 |
| `paraphrase__task_9` | `pick up the book and place it in the left compartment of the caddy` | paraphrase | `pick up the book and put it into the caddy's left compartment` | 0/20 | [0.00, 0.16] | 0/20 |

## Paired comparisons

| pair | reference -> condition | delta | discordant ref/condition | McNemar p |
|---|---|---:|---|---:|
| `task_0` | 20/20 -> 19/20 | -0.05 | 1/0 | 1.0000 |
| `task_1` | 17/20 -> 17/20 | +0.00 | 0/0 | 1.0000 |
| `task_2` | 20/20 -> 9/20 | -0.55 | 11/0 | 0.0010 |
| `task_3` | 19/20 -> 0/20 | -0.95 | 19/0 | 0.0000 |
| `task_4` | 15/20 -> 8/20 | -0.35 | 9/2 | 0.0654 |
| `task_5` | 8/20 -> 3/20 | -0.25 | 6/1 | 0.1250 |
| `task_6` | 13/20 -> 0/20 | -0.65 | 13/0 | 0.0002 |
| `task_7` | 19/20 -> 17/20 | -0.10 | 2/0 | 0.5000 |
| `task_8` | 2/20 -> 0/20 | -0.10 | 2/0 | 0.5000 |
| `task_9` | 12/20 -> 0/20 | -0.60 | 12/0 | 0.0005 |
