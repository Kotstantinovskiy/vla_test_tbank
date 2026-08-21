# Удаление одного артикля в десяти seen-инструкциях

Статус: **код подготовлен, rollout не запускался**.

Для тех же десяти anchors, что и в semantic-paraphrase эксперименте,
сравниваются exact prompt и строка с единственным изменением: удалено первое
отдельное слово `the`. `plan.py` программно проверяет этот инвариант.

Обе строки оцениваются одним native BDDL-предикатом на одинаковых init states,
env seeds и flow-noise seeds. `batch=1`, seed = `1000 + episode_index`.

```bash
cd experiments/2026-08-20_01-55-23_seen_article_drop
scripts/prepare.sh
scripts/determinism_smoke.sh 0
scripts/run_all.sh
scripts/aggregate.sh
scripts/log_trackio.sh
```

Сохраняются все 400 видео. Прокси-метрики движения не вычисляются.
