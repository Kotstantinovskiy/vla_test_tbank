# Goal-промпты 0–2 в совместимых seen-сценах

Статус: **код подготовлен, rollout не запускался**.

Замороженный официальный `libero_90`-претрен проверяется под первыми тремя
инструкциями `libero_goal`, но в знакомых сценах. Для каждого goal-промпта
зафиксирован один host, родной seen-контроль и nonsense-контроль. Первичная
метрика — BDDL-предикат именно подсказанной задачи; native success host-среды
сохраняется отдельно.

Для goal 1 generic `bowl` отображается на `white_bowl_1`, для goal 2 generic
`cabinet` — на `white_cabinet_1`. Эти отображения фиксируются в
`artifacts/eval_plan.json` до rollout и явно являются отклонением от точных
типов объектов goal benchmark.

## Детерминизм

Каждый эпизод выполняется отдельно (`batch=1`): env seed и torch/CUDA
flow-noise seed равны `1000 + episode_index`. `run_all.sh` откажется запускать
fan-out без прошедшего `determinism_smoke.sh`.

## Запуск

```bash
cd experiments/2026-08-20_01-55-23_goal_prompts_in_seen_hosts
scripts/prepare.sh
scripts/determinism_smoke.sh 0
scripts/run_all.sh
scripts/aggregate.sh
scripts/log_trackio.sh
```

Все 180 видео сохраняются под `results/raw/videos/`. Прокси-метрики движения
не вычисляются; поведение руки оценивается визуально по сохранённым rollout.
