# Marimo-анализ baseline cost curve

Производный интерактивный отчёт по результатам
`experiments/2026-08-18_analyst_curve_baseline`. Исходный эксперимент не
изменён: notebook работает с локальным snapshot компактных summary-артефактов.

## Содержимое

- `analysis.py` — Marimo notebook;
- `analysis.html` — выполненный статический HTML-snapshot (интерактивные
  controls работают при `marimo run` / `marimo edit`);
- `artifacts/combined_curve.csv` — 12 кривых на k=0/1/2/3/5/10/25;
- `artifacts/summary.json` — provenance, per-task curves и AUC;
- `artifacts/source_combined_curve.png` и `source_report.md` — исходные
  статические представления;
- `artifacts/derived_metrics.json` — ключевые производные числа notebook;
- `artifacts/MANIFEST.json` — SHA-256 snapshot-файлов.

Notebook добавляет к исходному отчёту pooled Wilson 95% CI, marginal gain на
демонстрацию, интерактивный threshold success→минимальный k, per-task heatmap,
сравнение linear/log2 AUC и аудит немонотонных участков.

## Запуск

```bash
cd vla_test

# Интерактивное редактирование.
/var/tmp/vla_tools/uv run --frozen marimo edit \
  reports/2026-08-18_analyst_curve_baseline_marimo/analysis.py

# Read-only приложение.
/var/tmp/vla_tools/uv run --frozen marimo run \
  reports/2026-08-18_analyst_curve_baseline_marimo/analysis.py

# Повторить HTML-export.
/var/tmp/vla_tools/uv run --frozen marimo export html \
  reports/2026-08-18_analyst_curve_baseline_marimo/analysis.py \
  -o reports/2026-08-18_analyst_curve_baseline_marimo/analysis.html --force
```

## Научные ограничения

- один training seed;
- 20 evaluation episodes на задачу и точку;
- k=0 и k>0 используют разные normalization statistics из-за LeRobot
  normalizer swap;
- Wilson CI в notebook pooled по задачам для обзорной визуализации и не
  заменяет парный тест по rollout outcomes;
- AUC — описательная интерполяция между измеренными бюджетами.
