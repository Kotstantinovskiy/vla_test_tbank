# SmolVLA: cost curve на новых задачах LIBERO

Репозиторий воспроизводит Задачу 1: честный seen-чекпойнт, zero-shot с
языковыми контролями и неизменяемый naive fine-tuning baseline на 5/10/25
демонстрациях для первых трёх задач `libero_goal`.

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

Требуются Linux, Python 3.12, CUDA и GPU примерно от 40 GB для полного
fine-tuning. Официальная интеграция описана в
[документации LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero).

```bash
# В этом окружении зависимости уже установлены в /var/tmp/vla_env.
source scripts/common_env.sh
pip install -e '.[test]'

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
```

Тяжёлые данные и кэши лежат в `/var/tmp`, результаты — в `results/`, веса —
в `outputs/`, итоговые `summary.json`, CSV и PNG — в `reports/generated/`.
Schema-adapter меняет только имена камер `top/wrist_image` на
`image/image2`; SHA-256 весов и provenance записываются рядом с артефактом.

## Протокол baseline

- 20 эпизодов на каждую точку, seed 1000; reset seeds одинаковы между
  языковыми условиями.
- Wrong-language: инструкция следующей target-задачи по циклу; nonsense:
  фиксированная строка.
- Каждая из девяти моделей всегда стартует с одного seen-чекпойнта.
- 2000 optimizer steps, batch 32, полный fine-tuning, без аугментаций, replay,
  PEFT и выбора лучшего checkpoint по success.
- Оценивается только финальный checkpoint.
