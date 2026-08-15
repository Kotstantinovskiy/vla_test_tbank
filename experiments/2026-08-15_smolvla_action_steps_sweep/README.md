# SmolVLA `n_action_steps` sweep

Отдельная inference-only абляция на замороженных чекпойнтах эксперимента
`2026-08-15_smolvla_libero_task1`. Она проверяет, уменьшает ли более частое
перепланирование compounding error без новых демонстраций и без изменения весов.

## Что меняется

Свип фиксирован заранее: `n_action_steps ∈ {1, 5, 10, 25, 50}`. SmolVLA всё
так же предсказывает chunk из 50 действий, но выполняет только первые
`n_action_steps`, после чего строит новый chunk по свежему наблюдению. Все
значения допустимы для сохранённого `chunk_size=50`.

Проверяются четыре бюджета `k ∈ {0, 5, 10, 25}` на task IDs 0–2 из
`libero_goal`. Точка `k=0` использует закреплённую ревизию LIBERO-90, остальные
точки — ровно существующие адаптированные чекпойнты step 2000. Для каждой точки:

- 20 эпизодов, seed 1000–1019, batch size 4;
- seed сбрасывается перед каждым горизонтом, поэтому initial states парные;
- одна MP4-запись на горизонт;
- `n_action_steps=50` пересчитывается как внутренний anchor;
- старый baseline не редактируется и присутствует только как копия-reference в
  `artifacts/frozen_baseline_reference.json`.

Предсказания были записаны до rollout в [`reports/PREDICTIONS.md`](reports/PREDICTIONS.md).
Если хотя бы один zero-shot эпизод с правильным prompt успешен, launcher по
заранее заданному правилу дополнительно считает wrong-task и nonsense controls.
При полном floor они пропускаются как неидентифицируемые по binary success.

## Запуск

Из корня репозитория достаточно:

```bash
uv sync --frozen
experiments/2026-08-15_smolvla_action_steps_sweep/scripts/run_action_steps_sweep.sh 0 1 2 3
```

Запуск возобновляемый: завершённые горизонты читаются из соответствующего raw
JSON и повторно не считаются. Один отдельный job:

```bash
cd experiments/2026-08-15_smolvla_action_steps_sweep
scripts/prepare.sh
scripts/eval_job.sh adapted 1 10 0
python -m smolvla_action_steps.aggregate
scripts/log_trackio.sh
```

Для короткой технической проверки можно переопределить только runtime-параметры
(такие результаты не следует смешивать с полным протоколом):

```bash
VLA_N_EPISODES=1 VLA_ACTION_STEPS="1 50" VLA_VIDEOS=0 \
  scripts/eval_job.sh zero_shot 0 0 0
```

## Результаты и Trackio

`results/raw/` содержит rollout JSON и MP4, `results/summary/` — сводные JSON,
CSV и графики, `results/media/gifs/` — компактные GIF, `reports/REPORT.md` —
сгенерированный итог, `artifacts/trackio/` — локальную Trackio-базу.

Открыть только этот проект:

```bash
scripts/show_trackio.sh
```

Открыть общий dashboard из корня и перелистывать все эксперименты:

```bash
cd /home/nbagent174/vla_test
scripts/show_trackio.sh
```

После нового логирования общий launcher нужно перезапустить: он строит безопасный
snapshot локальных Trackio-баз и media.
