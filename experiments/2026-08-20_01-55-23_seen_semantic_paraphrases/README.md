# Смысловые парафразы десяти seen-инструкций

Статус: **код подготовлен, rollout не запускался**.

Для десяти заранее выбранных задач `libero_90` сравниваются точная обучающая
строка и один смысловой парафраз. Парафразы сохраняют цвет/тип объекта,
отношение и составные подцели; меняется только формулировка.

Каждая пара получает одинаковые init states, env seeds и flow-noise seeds.
Эпизоды выполняются по одному (`batch=1`), seed = `1000 + episode_index`.
Первичная метрика — один и тот же native BDDL-предикат для обеих строк;
парная статистика — exact McNemar.

```bash
cd experiments/2026-08-20_01-55-23_seen_semantic_paraphrases
scripts/prepare.sh
scripts/determinism_smoke.sh 0
scripts/run_all.sh
scripts/aggregate.sh
scripts/log_trackio.sh
```

Сохраняются все 400 видео. Метрики движения руки не вычисляются.
