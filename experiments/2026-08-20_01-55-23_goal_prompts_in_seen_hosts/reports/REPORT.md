# goal_prompts_in_seen_hosts: успешность по промптированному предикату (prompted-predicate)

Агрегируется только бинарный показатель успешности. Каждое видео роллаута сохраняется на диске для ручной инспекции поведения модели. Сиды окружения и шума политики (policy-noise) регистрируются для каждого эпизода.

| метка (label) | инструкция окружения (env instruction) | блок (block) | промпт политики (policy prompt) | успешность (success) | 95% ДИ (95% CI) | успешность окружения (env success) |
|---|---|---|---|---:|---|---:|
| `seen__goal_0` | `close the top drawer of the cabinet` | seen | `close the top drawer of the cabinet` | 20/20 | [0.84, 1.00] | 20/20 |
| `goal__goal_0` | `close the top drawer of the cabinet` | goal | `open the middle drawer of the cabinet` | 0/20 | [0.00, 0.16] | 19/20 |
| `nonsense__goal_0` | `close the top drawer of the cabinet` | nonsense | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen__goal_1` | `put the white bowl on top of the cabinet` | seen | `put the white bowl on top of the cabinet` | 15/20 | [0.53, 0.89] | 15/20 |
| `goal__goal_1` | `put the white bowl on top of the cabinet` | goal | `put the bowl on the stove` | 0/20 | [0.00, 0.16] | 0/20 |
| `nonsense__goal_1` | `put the white bowl on top of the cabinet` | nonsense | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 |
| `seen__goal_2` | `put the wine bottle on the wine rack` | seen | `put the wine bottle on the wine rack` | 8/20 | [0.22, 0.61] | 8/20 |
| `goal__goal_2` | `put the wine bottle on the wine rack` | goal | `put the wine bottle on top of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 |
| `nonsense__goal_2` | `put the wine bottle on the wine rack` | nonsense | `perform the dax florp twice` | 0/20 | [0.00, 0.16] | 0/20 |
