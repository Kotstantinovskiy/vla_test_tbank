# Low-k sweep частоты перепланирования SmolVLA

Статус: **код подготовлен, обучение и rollout не запускались**.

Эксперимент повторяет training-протокол
`2026-08-18_pretrain_smolvla_few_shot_tune_low_k`: канонический
official-data seen-чекпойнт независимо адаптируется к десяти задачам
`libero_goal` на первых `k∈{1,2,3}` официальных демонстрациях. Получается 30
адаптаций. Каждая адаптация затем оценивается с одним и тем же набором весов
при `n_action_steps∈{1,10,25}` — это 90 eval-точек.

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
- flow-sampling RNG torch/CUDA переустанавливается перед каждым эпизодом;
- все action-step условия используют одинаковые init states и seed bank;
- primary metric — только бинарный task success; двигательных proxy-метрик нет;
- сохраняются все 1800 основных видео;
- Trackio получает только первый success и первый failure для каждой пары
  `(k, n_action_steps)`.

Вычислительно это тяжёлый screen: при horizon 300 верхняя оценка — 205200
вызовов политики, причём основная стоимость приходится на `n=1`. Ранние
успехи уменьшают фактическое число вызовов.

Перед fan-out обязательны реальный env smoke и production/determinism smoke.
Последний обучает `task=0, k=1`, полностью оценивает `n=10`, затем повторяет
его после prefix-прогона `n=1` и требует одинаковых поэпизодных исходов.

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
scripts/train_one.sh TASK_ID K
scripts/eval_one.sh TASK_ID K N_ACTION_STEPS
```

`smoke_dataset.sh` реально создаёт все 30 filtered `LeRobotDataset` и проверяет,
что загружены именно зафиксированные episode indices. `run_all.sh` откажется
запускать fan-out без всех preflight-артефактов.
Обучение и оценка идемпотентны: готовые checkpoint и точки с 20 существующими
видео пропускаются.

## Интерпретация

Старые результаты `n=50` можно показывать только как описательный ориентир:
они использовали `batch=4` и один RNG-поток на процесс. Поэтому возможное
улучшение в этом screen ещё не является чистой причинной оценкой
`n_action_steps`. Если screen выглядит лучше, следующий эксперимент должен
добавить парный `n=50`, второй training seed и бюджеты `k=5/10/25`.
