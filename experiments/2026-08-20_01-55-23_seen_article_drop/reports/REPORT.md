# seen_article_drop: успешность по предикату промпта

Агрегируется только бинарный показатель успешности. Все видеозаписи симуляций (rollouts) сохраняются на диске для ручного анализа поведения. Сиды среды и шума политики регистрируются для каждого эпизода.

| label | инструкция среды | block | промпт политики | успешность | 95% ДИ | успешность среды |
|---|---|---|---|---:|---|---:|
| `exact__task_0` | `turn on the stove` | exact | `turn on the stove` | 20/20 | [0.84, 1.00] | 20/20 |
| `article_drop__task_0` | `turn on the stove` | article_drop | `turn on stove` | 20/20 | [0.84, 1.00] | 20/20 |
| `exact__task_1` | `put the frying pan on the stove` | exact | `put the frying pan on the stove` | 17/20 | [0.64, 0.95] | 17/20 |
| `article_drop__task_1` | `put the frying pan on the stove` | article_drop | `put frying pan on the stove` | 15/20 | [0.53, 0.89] | 15/20 |
| `exact__task_2` | `close the top drawer of the cabinet` | exact | `close the top drawer of the cabinet` | 20/20 | [0.84, 1.00] | 20/20 |
| `article_drop__task_2` | `close the top drawer of the cabinet` | article_drop | `close top drawer of the cabinet` | 20/20 | [0.84, 1.00] | 20/20 |
| `exact__task_3` | `put the black bowl on top of the cabinet` | exact | `put the black bowl on top of the cabinet` | 19/20 | [0.76, 0.99] | 19/20 |
| `article_drop__task_3` | `put the black bowl on top of the cabinet` | article_drop | `put black bowl on top of the cabinet` | 8/20 | [0.22, 0.61] | 8/20 |
| `exact__task_4` | `put the black bowl on the plate` | exact | `put the black bowl on the plate` | 15/20 | [0.53, 0.89] | 15/20 |
| `article_drop__task_4` | `put the black bowl on the plate` | article_drop | `put black bowl on the plate` | 13/20 | [0.43, 0.82] | 13/20 |
| `exact__task_5` | `put the wine bottle on the wine rack` | exact | `put the wine bottle on the wine rack` | 8/20 | [0.22, 0.61] | 8/20 |
| `article_drop__task_5` | `put the wine bottle on the wine rack` | article_drop | `put wine bottle on the wine rack` | 8/20 | [0.22, 0.61] | 8/20 |
| `exact__task_6` | `open the top drawer of the cabinet and put the bowl in it` | exact | `open the top drawer of the cabinet and put the bowl in it` | 13/20 | [0.43, 0.82] | 13/20 |
| `article_drop__task_6` | `open the top drawer of the cabinet and put the bowl in it` | article_drop | `open top drawer of the cabinet and put the bowl in it` | 11/20 | [0.34, 0.74] | 11/20 |
| `exact__task_7` | `open the microwave` | exact | `open the microwave` | 19/20 | [0.76, 0.99] | 19/20 |
| `article_drop__task_7` | `open the microwave` | article_drop | `open microwave` | 8/20 | [0.22, 0.61] | 8/20 |
| `exact__task_8` | `pick up the alphabet soup and put it in the basket` | exact | `pick up the alphabet soup and put it in the basket` | 2/20 | [0.03, 0.30] | 2/20 |
| `article_drop__task_8` | `pick up the alphabet soup and put it in the basket` | article_drop | `pick up alphabet soup and put it in the basket` | 5/20 | [0.11, 0.47] | 5/20 |
| `exact__task_9` | `pick up the book and place it in the left compartment of the caddy` | exact | `pick up the book and place it in the left compartment of the caddy` | 12/20 | [0.39, 0.78] | 12/20 |
| `article_drop__task_9` | `pick up the book and place it in the left compartment of the caddy` | article_drop | `pick up book and place it in the left compartment of the caddy` | 7/20 | [0.18, 0.57] | 7/20 |

## Парные сравнения

| пара | референс -> условие | delta | несогласованные реф/условие | McNemar p |
|---|---|---:|---|---:|
| `task_0` | 20/20 -> 20/20 | +0.00 | 0/0 | 1.0000 |
| `task_1` | 17/20 -> 15/20 | -0.10 | 3/1 | 0.6250 |
| `task_2` | 20/20 -> 20/20 | +0.00 | 0/0 | 1.0000 |
| `task_3` | 19/20 -> 8/20 | -0.55 | 11/0 | 0.0010 |
| `task_4` | 15/20 -> 13/20 | -0.10 | 3/1 | 0.6250 |
| `task_5` | 8/20 -> 8/20 | +0.00 | 3/3 | 1.0000 |
| `task_6` | 13/20 -> 11/20 | -0.10 | 5/3 | 0.7266 |
| `task_7` | 19/20 -> 8/20 | -0.55 | 11/0 | 0.0010 |
| `task_8` | 2/20 -> 5/20 | +0.15 | 2/5 | 0.4531 |
| `task_9` | 12/20 -> 7/20 | -0.25 | 8/3 | 0.2266 |
