# Low-k sweep частоты перепланирования SmolVLA

Статус: **production/determinism gate пройден; inference sweep запущен на
GPU 1–3, нового обучения нет**.

Эксперимент повторяет training-протокол
`2026-08-18_pretrain_smolvla_few_shot_tune_low_k`: канонический
использует девять уже готовых адаптаций из воспроизводимого эксперимента
`2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro`: три
основные задачи `libero_goal` (task ID 0–2) × первые
`k∈{1,2,3}` официальные демонстрации. Нового обучения здесь нет. Каждый
checkpoint оценивается с одним и тем же набором весов при
`n_action_steps∈{1,10,25}` — это 27 eval-точек.

`chunk_size` остаётся 50. `n_action_steps` не меняет training loss: в
SmolVLA он задаёт длину очереди действий, исполняемых до следующего вызова
политики. Поэтому три раза переобучать одну task/k-точку не нужно.

## Что уже проверялось

В канонической official-data линии базовый checkpoint и все 30 low-k
checkpoint имели только `chunk_size=50, n_action_steps=50`. Это зафиксировано
в `artifacts/prior_action_steps_evidence.json`.

Уточнение: во всей истории репозитория утверждение «проверяли только 50»
неверно. Удалённый эксперимент 2026-08-15 проверял `1/5/10/25/50`, но он
относился к удалённой сторонней crislmfroes-линии. Его числа не считаются
результатом текущего канонического претрейна.

## Детерминизм и метрика

- каждый rollout выполняется отдельно (`batch=1`);
- `env_seed = noise_seed = 1000 + episode_index`;
- `LIBERO init_state_id = episode_index` перед каждым rollout;
- flow-sampling RNG torch/CUDA переустанавливается перед каждым эпизодом;
- все action-step условия используют одинаковые init states и seed bank;
- primary metric — только бинарный task success; двигательных proxy-метрик нет;
- сохраняются все 540 основных видео;
- Trackio получает только первый success и первый failure для каждой пары
  `(k, n_action_steps)`.

Вычислительно это тяжёлый screen: при horizon 300 верхняя оценка — 61560
вызовов политики, причём основная стоимость приходится на `n=1`. Ранние
успехи уменьшают фактическое число вызовов.

Перед fan-out обязательны реальный env smoke и production/determinism smoke.
Последний берёт готовый checkpoint `task=0, k=1`, полностью оценивает
`n=10`, выполняет prefix-прогон `n=1`, затем повторяет `n=10` в обратном
порядке эпизодов и требует одинаковых поэпизодных исходов.

## Запуск

```bash
cd experiments/2026-08-20_20-03-21_pretrain_smolvla_low_k_action_steps
scripts/prepare.sh
scripts/smoke_dataset.sh
scripts/audit.sh
scripts/smoke_env.sh
scripts/production_smoke.sh 0
scripts/run_all.sh
scripts/status.sh
```

Ручные точки:

```bash
scripts/eval_one.sh TASK_ID K N_ACTION_STEPS
```

`smoke_dataset.sh` реально создаёт все 9 filtered `LeRobotDataset` и проверяет,
что загружены именно зафиксированные episode indices. `run_all.sh` откажется
запускать fan-out без всех preflight-артефактов.
Checkpoint’ы проверяются по SHA и train-config, а eval идемпотентен: точки с
20 существующими видео пропускаются.

## Интерпретация

Старые результаты `n=50` можно показывать только как описательный ориентир:
они использовали `batch=4` и один RNG-поток на процесс. Поэтому возможное
улучшение в этом screen ещё не является чистой причинной оценкой
`n_action_steps`. Если screen выглядит лучше, следующий эксперимент должен
добавить парный `n=50`, второй training seed и бюджеты `k=5/10/25`.

Для offline-воспроизводимости canonical checkpoint используется через
локальный runtime-view, tokenizer/VLM закреплён на revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`, а LIBERO assets — на
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
