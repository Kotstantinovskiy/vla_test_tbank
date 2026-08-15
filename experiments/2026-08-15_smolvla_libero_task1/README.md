# SmolVLA: cost curve на новых задачах LIBERO

Этот каталог — самодостаточный снимок Задачи 1: честный seen-чекпойнт, zero-shot с
языковыми контролями и неизменяемый naive fine-tuning baseline на 5/10/25
демонстрациях для первых трёх задач `libero_goal`.

Внутри находятся собственные `configs/`, `scripts/`, `src/`, `tests/`,
`results/`, `artifacts/` и `reports/`. Скрипты вычисляют корень эксперимента по
своему расположению, поэтому их можно вызывать из любого рабочего каталога.

## Зафиксированный выбор

Вместо повторного seen-претрейна используется открытый чекпойнт
[`crislmfroes/smolvla-libero-90`](https://huggingface.co/crislmfroes/smolvla-libero-90)
на ревизии `418f9d0e...`. Его карточка фиксирует 30k шагов на 4500 эпизодах
LIBERO-90 и базу `lerobot/smolvla_base`; `libero_goal` в списке обучения нет.
Это разрешённый условием путь и оставляет вычислительный бюджет на девять
адаптаций и статистически осмысленные роллауты.

Датасет [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero)
содержит 40 задач, а не LIBERO-90. Поэтому она используется только для
held-out демонстраций. Seen-данные и target-данные разделены. Для проверки
LIBERO-90 также зафиксирован независимый публичный LeRobot-v3 датасет
[`GT-111/libero-90-v3`](https://huggingface.co/datasets/GT-111/libero-90-v3),
но он не нужен для повторного обучения выбранного открытого чекпойнта.

Target-data pinned на `9176d427...`: это последняя ревизия до ошибочной
перезаписи episode→file metadata. Диагностика и обоснование зафиксированы до
обучения в [`reports/DEVIATIONS.md`](reports/DEVIATIONS.md).

Предсказания и протокол были закоммичены до первого policy rollout в
[`reports/PREDICTIONS.md`](reports/PREDICTIONS.md), commit `f307cf7`.

## Важная деталь разбиения

`task_id` среды локален внутри suite, а `task_index` объединённого датасета
глобален. Для target task IDs 0/1/2 глобальные индексы равны 19/11/12.
`src/vla_cost_curve/selection.py` сопоставляет эпизоды по точной инструкции и
берёт первые k в глобальном порядке; удачные или короткие траектории не
отбираются.

## Запуск

Требуются Linux, Python 3.12, CUDA и GPU от 32 GB для полного fine-tuning
(пик в этих запусках — 28.4 GB; 40 GB оставляет удобный запас). Официальная
интеграция описана в
[документации LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero).
Для воспроизводимости uv-конфигурация фиксирует проверенный стек PyTorch 2.7.1
с CUDA 12.6, на котором были получены чекпойнты.

```bash
# Один раз из корня репозитория.
uv sync --frozen

cd experiments/2026-08-15_smolvla_libero_task1

# По умолчанию используется корневая .venv;
# старый /var/tmp/vla_env остаётся fallback.
source scripts/common_env.sh

# Данные, manifest и schema-only адаптер seen-чекпойнта.
scripts/prepare.sh

# Полный протокол на 3 GPU.
scripts/run_task1.sh
```

Отдельные стадии:

```bash
scripts/eval_zero_shot.sh true 0
scripts/eval_zero_shot.sh wrong 1
scripts/eval_zero_shot.sh nonsense 2

scripts/train_naive_ft.sh 0 5 0
scripts/eval_adapted.sh 0 5 0

python -m vla_cost_curve.aggregate

# Записать training/cost curves, таблицы и rollout GIF в локальный Trackio.
scripts/log_trackio.sh
scripts/show_trackio.sh
```

Тяжёлые исходные данные и кэши лежат в `/var/tmp`. Все experiment-owned пути
находятся здесь: сырые метрики и видео — в `results/raw/`, логи — в
`results/logs/`, итоговые JSON/CSV/PNG — в `results/summary/`, manifest и веса —
в `artifacts/`, компактные GIF — в `results/media/gifs/`. Локальная база и
копии медиа Trackio находятся в игнорируемом `artifacts/trackio/`.
`artifacts/checkpoints` является локальной ссылкой на тяжёлое
хранилище `/var/tmp/vla_outputs`; путь и принадлежность артефакта эксперименту
при этом остаются однозначными. Значения внешних путей можно переопределить
переменными окружения из `scripts/common_env.sh`.
Schema-adapter меняет только имена камер `top/wrist_image` на
`image/image2`; SHA-256 весов и provenance записываются рядом с артефактом.

На H200 один 2000-step full-FT занимает примерно 35–41 минут, evaluation одной
точки на 20 эпизодах — около минуты. Четыре независимых обучения можно запускать
параллельно; полная последовательность дольше из-за девяти отдельных моделей.

## Протокол baseline

- 20 эпизодов на каждую точку, seed 1000; reset seeds одинаковы между
  языковыми условиями.
- Wrong-language: инструкция следующей target-задачи по циклу; nonsense:
  фиксированная строка.
- Каждая из девяти моделей всегда стартует с одного seen-чекпойнта.
- 2000 optimizer steps, batch 32, полный fine-tuning, без аугментаций, replay,
  PEFT и выбора лучшего checkpoint по success.
- Оценивается только финальный checkpoint.

## Trackio

`scripts/log_trackio.sh` создаёт в проекте `smolvla-libero-task1`:

- девять run'ов naive fine-tuning с кривыми loss, gradient norm, learning rate,
  throughput, memory и компонентами loss каждые 25 шагов;
- summary-run `2026-08-15-cost-curve` с per-task и mean success curves;
- таблицу baseline cost curve с Wilson 95% CI и отдельную таблицу zero-shot
  языковых контролей;
- PNG cost curve и доступные rollout GIF для k=0 и адаптированных бюджетов.

По умолчанию всё работает локально и не требует аккаунта. Для публикации в
Hugging Face Space достаточно задать переменную перед запуском:

```bash
TRACKIO_SPACE_ID="username/vla-trackio" scripts/log_trackio.sh
```

Новые adapted-rollout видео сохраняются в budget-specific каталогах, поэтому
k=5/10/25 больше не перезаписывают друг друга. В уже завершённом запуске
сохранились k=0 и последний k=25 rollout для каждой задачи.
