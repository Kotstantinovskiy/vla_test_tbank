# Девять обученных seen-промптов в одной goal-сцене

Статус: **код подготовлен, rollout не запускался**.

Все условия используют один BDDL host — goal task 0. У десяти native goal
задач совпадают сцена и init layout, поэтому один host исключает скрытую смену
изображения. Девять verbatim `libero_90`-строк имеют нетривиальные предикаты,
вычислимые в этой сцене. Успех определяется предикатом самой подсказанной
seen-задачи, а native host success хранится отдельно.

Исключены `close the top drawer...` и `turn off the stove` как выполненные при
reset, а также `front black bowl` как неприменимый идентификатор среди одной
миски.

```bash
cd experiments/2026-08-20_01-55-23_seen_prompts_in_goal_scene
scripts/prepare.sh
scripts/determinism_smoke.sh 0
scripts/run_all.sh
scripts/aggregate.sh
scripts/log_trackio.sh
```

Сохраняются все 220 видео. Метрики движения руки не вычисляются.
