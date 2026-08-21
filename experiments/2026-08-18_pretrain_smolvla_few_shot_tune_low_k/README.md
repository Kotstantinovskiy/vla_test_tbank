# Few-shot cost curve, низкие бюджеты (k=1/2/3)

Сиблинг `2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune` с
байт-идентичным рецептом, добавляющий точки k=1/2/3. Причина — оговорка о
потолке в задании (v15.08.2026): наивный бейзлайн насыщается уже при k=5
(0.95 на задачах 0–2), и различия методов надо искать в низкобюджетном
режиме. Точки k=5/10/25 принадлежат сиблингу и не перепрогоняются.

Дата: 2026-08-18. Кост-кривая нового претрена
(`2026-08-17_smolvla_pretrain_libero`): 10 задач `libero_goal` × бюджеты
k∈{1,2,3} = 30 независимых адаптаций, каждая стартует с базового
чекпойнта.

## Протокол

- **Обучаемое**: action expert (`lm_expert`), state/action-проекции,
  action-time MLP. **Заморожено**: vision encoder и весь VLM
  (`train_expert_only=true`); аудит имён/счётчиков параметров выполняется до
  запуска (`scripts/audit.sh` → `artifacts/trainable_parameters.json`).
- **Демо**: официальные `demo_0..demo_{k-1}` каждой задачи из in-repo
  конверсии `official/libero_goal_rot180_128` — порядок соответствует
  официальному и ассертится при построении замороженного
  `artifacts/episode_manifest.json`. Это закрывает аудит-пункт 5.
- 2000 шагов, batch 32, seed 1000, fp32, пресет SmolVLA.
- **Eval**: 20 эпизодов/точку, seed 1000, рендер 128×128; **все видео на
  диске** (600 роликов); GIF — первый успех и первый провал на каждый k
  (и они же в Trackio).
- k=0 берётся справочно из `2026-08-18_pretrain_smolvla_prompt_only`
  (0.005) с оговоркой о подмене нормализатора (см. protocol.yaml).

Предсказания до запуска: [reports/PREDICTIONS.md](reports/PREDICTIONS.md).

## Запуск

```bash
uv sync --frozen
cd experiments/2026-08-18_pretrain_smolvla_few_shot_tune_low_k
scripts/prepare.sh      # LIBERO config, манифесты, заморозка выбора демо
scripts/audit.sh        # аудит trainable/frozen параметров
scripts/smoke_env.sh    # env-проверка всех 10 задач
scripts/run_all.sh      # оркестратор: 30 train+eval jobs на GPU 0-3
scripts/status.sh
scripts/aggregate.sh
scripts/log_trackio.sh
```

Оркестратор идемпотентен: завершённые train/eval стадии распознаются и
пропускаются.

## Результаты

- `results/raw/task_*/k_*.json` — по-эпизодные исходы с путями видео.
- `results/raw/videos/task_*/k_*/eval_episode_*.mp4` — все ролики.
- `results/media/gifs/k_*_first_{success,failure}.gif`.
- `results/summary/{summary.json,cost_curve.csv,cost_curve.png}`.
- Trackio-проект `pretrain-few-shot-low-k`.
