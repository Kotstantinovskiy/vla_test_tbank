# SmolVLA на данных, сконвертированных из официальной LIBERO-репы

Дата создания: 2026-08-17. Статус: **подготовлен; обучение не запускалось.**

Тот же претрен-протокол, что и в `2026-08-17_smolvla_libero90_mirrorfix_ddp`
(30k шагов, DDP×4, expert-only, seed 1000, `lerobot/smolvla_base@c83c3163`),
но обучающие и target-данные конвертируются **в этом репозитории напрямую из
официальных HDF5** (`yifengzhu-hf/LIBERO-datasets@f13aa24`, скачанных
закреплённым корневым загрузчиком). Это устраняет в источнике все известные
дефекты сторонних конверсий:

| Проблема сторонних конверсий | Здесь |
|---|---|
| Зеркальная ориентация seen-кадров | Кадры сразу хранятся как `rot180(official)` = eval-конвенция; runtime-трансформа нет |
| Неверная метка fps=10 у target | Оба датасета объявляют честные fps=20 (= control_freq) |
| «Первые k» ≠ официальные демо, ~15% демо отсутствуют | Порядок эпизодов = официальный (файлы по имени, демо по индексу): «первые k» = demo_0..demo_{k−1}, все 50 демо на задачу |
| Домен-гэп 128→224 h264-апскейл vs eval 256 | Нативные 128×128 (AV1 crf-18, почти без потерь); eval рендерит те же 128×128 — train и eval проходят один и тот же 512-pad ресайз SmolVLA |
| Разные рецепты state | state = официальные `ee_pos+ee_ori+gripper_states` — ровно раскладка eval-процессора (`eef_pos, eef_axisangle, gripper_qpos`); actions бит-в-бит |

Подмена нормализатора при файнтюне (аудит-п.3) остаётся поведением LeRobot и
задокументирована в протоколе; но теперь seen и target идут из одного
пайплайна, так что их статистики напрямую сопоставимы.

Предсказания до запусков: [reports/PREDICTIONS.md](reports/PREDICTIONS.md).

## Пайплайн

```bash
cd /home/nbagent174/vla_test/experiments/2026-08-17_smolvla_pretrain_libero

# 1. Конверсия (идемпотентна, атомарна per-suite; libero_90 требует полной
#    загрузки официальных файлов).
scripts/convert.sh --suites libero_goal libero_90

# 2. Обязательная верификация против официальных HDF5 до обучения.
scripts/verify_conversion.sh --suites libero_goal libero_90

# 3. Подготовка (base model schema-adapter 128×128, манифесты, симлинки).
scripts/prepare.sh

# 4. Смок и обучение (НЕ запускалось).
scripts/smoke_ddp.sh
scripts/train_ddp.sh

# 5. Позитивный контроль после обучения.
scripts/eval_seen_control.sh 0
```

## Данные

- `/var/tmp/vla_libero_official_rot180/libero_90` — seen-претрен
  (`official/libero_90_rot180_128`, 90 задач × 50 демо).
- `/var/tmp/vla_libero_official_rot180/libero_goal` — target-демо для
  downstream-адаптации (`official/libero_goal_rot180_128`, 10 задач × 50
  демо, официальный порядок). Downstream-эксперименты должны брать «первые k»
  отсюда — это и есть официальные demo_0..demo_{k−1}.
- Манифесты конверсии и верификации — в `artifacts/`.

## Артефакты

- `artifacts/base_model{,_source}` — pinned base и schema-adapted view
  (128×128, hard-link весов).
- `artifacts/official_source` → официальные HDF5.
- `artifacts/dataset_seen`, `artifacts/dataset_target` → конвертированные датасеты.
- `artifacts/checkpoints` → `/var/tmp/vla_outputs/seen_libero90_official_20260817`.
- `artifacts/trackio` — Trackio-проект `smolvla-pretrain-libero`.
