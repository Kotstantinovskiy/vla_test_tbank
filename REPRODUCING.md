# Как запустить репозиторий: оценка чекпойнтов и воспроизведение экспериментов на LIBERO

## 1. Требования

- Linux x86_64, NVIDIA GPU (CUDA 12.6 драйвер), ~30 ГБ диска.
- Python 3.12 и [uv](https://docs.astral.sh/uv/). Больше ничего ставить
  руками не нужно: все версии (lerobot 0.6.1 c LIBERO-интеграцией,
  PyTorch 2.7.1+cu126, MuJoCo и т.д.) зафиксированы в `uv.lock`.

## 2. Установка окружения

```bash
git clone <repo> vla_test && cd vla_test
uv sync --frozen          # создаёт .venv строго по lock-файлу
source .venv/bin/activate
export MUJOCO_GL=egl      # headless-рендеринг (без дисплея)
```

## 3. Первый запуск LIBERO: конфиг без интерактива

При первом импорте `libero` пакет интерактивно спрашивает пути и в
headless/detached-режиме падает с `EOFError`. Сгенерируйте конфиг заранее:

```bash
export LIBERO_CONFIG_PATH=$PWD/.libero_config
python - <<'EOF'
import importlib.util, os, yaml
from pathlib import Path
root = Path(next(iter(importlib.util.find_spec("libero").submodule_search_locations)))
bench = root / "libero"
cfg = {"benchmark_root": str(bench), "bddl_files": str(bench / "bddl_files"),
       "init_states": str(bench / "init_files"), "datasets": str(root / "datasets"),
       "assets": str(bench / "assets")}
out = Path(os.environ["LIBERO_CONFIG_PATH"]); out.mkdir(parents=True, exist_ok=True)
(out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("written", out / "config.yaml")
EOF
```

Сцены/объекты (assets) lerobot скачает с Hugging Face при первом создании
среды (`lerobot/libero-assets`).

## 4. Чекпойнты

Инструкция применима к любому SmolVLA-чекпойнту в формате lerobot — папке
`pretrained_model/` (config, веса, процессоры с нормализационной
статистикой): базовому претрену, любой адаптации из
`experiments/*/artifacts/adapted_checkpoints/` (там же манифесты с путями и
SHA-256), чекпойнту из приложенного к отчёту пакета или вашему собственному,
дообученному этим же кодом. Если есть манифест — сверьте веса:

```bash
sha256sum <ckpt>/pretrained_model/model.safetensors   # должен совпасть с *.json манифестом
```

Важно про схему наблюдений: все чекпойнты этого репозитория обучены на двух
камерах 128×128 с ключами `top` (внешняя) и `wrist_image` (запястье). В
средах LIBERO камеры называются `agentview_image` /
`robot0_eye_in_hand_image`, поэтому при оценке нужен маппинг имён и размер
128 (команды ниже уже содержат это). Конвенция гриппера и ориентация кадров
согласованы со средой ещё на этапе подготовки данных — никаких
дополнительных преобразований не требуется. Для чекпойнта с другой схемой
камер поправьте маппинг/размеры под его `config.json`.

## 5. Быстрая проверка чекпойнта на любой suite (пример: libero_10)

Штатный `lerobot-eval` работает с любой папкой `pretrained_model`:

```bash
lerobot-eval \
  --policy.path=/path/to/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --env.observation_height=128 \
  --env.observation_width=128 \
  --env.camera_name_mapping='{"agentview_image":"top","robot0_eye_in_hand_image":"wrist_image"}' \
  --env.episode_length=520 \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --seed=1000 \
  --output_dir=eval_out/libero10_task0
```

- `--env.task` — любая из `libero_10`, `libero_goal`, `libero_spatial`, `libero_object`,
  `libero_90` (для goal-задач мы использовали `--env.episode_length=300`).
- `--env.task_ids` — номера задач внутри suite.

- Инференс-ручка `--policy.n_action_steps=25|35|50` (обучено с 50) задаёт,
  сколько действий из 50-шагового чанка исполнять до перепланирования.

Что ожидать: адаптированные чекпойнты (`bundle/task_T/k_K`) специализированы
под свою задачу `libero_goal`; на чужих suite (в т.ч. `libero_10`) и у них,
и у base-претрена zero-shot успех около нуля — это известное свойство
(см. prompt-only эксперименты в `README.md`). Осмысленные проверки:
бандл-чекпойнт на своей задаче (номера успехов — в `REPORT.md`
эксперимента `2026-08-23_18-20-07_pretrain_smolvla_bundle_all_k`) или
base-претрен на seen-задаче `libero_90` (позитивный контроль 20/20 на
task 0).

## 6. Точное воспроизведение нашего протокола оценки (libero_goal, задачи 0–2)

Наш протокол строже, чем дефолтный `lerobot-eval`: batch=1, пер-эпизодные
сиды среды и шума политики (`1000 + номер эпизода`), пиновка
`init_state_id = номер эпизода`, все видео на диск. Он реализован внутри
каждого эксперимента одинаково — схема универсальная:

```bash
cd experiments/<любой_эксперимент>
source scripts/common_env.sh       # выставляет пути, LIBERO_CONFIG_PATH и т.д.
scripts/prepare.sh                 # одноразовый preflight (манифесты, конфиг LIBERO)
scripts/eval_one.sh <аргументы>    # оценка одной точки
```

Аргументы `eval_one.sh` зависят от осей эксперимента (см. его README):
например, у бандла это `task_id budget n_action_steps`
(`scripts/eval_one.sh 0 3 50`), у state-noise — `task_id alpha
n_action_steps`. Результат: JSON с поэпизодными исходами в
`results/raw/...` и видео каждого эпизода рядом. Ожидается, что чекпойнт
лежит по пути из `artifacts/adapted_checkpoints/...json` (поправьте симлинк
`artifacts/checkpoints`, если веса развёрнуты в другом месте).

## 7. Полный запуск эксперимента (обучение + оценка) и свои эксперименты

Для обучения нужны данные: официальные HDF5 качаются
`scripts/download_official_libero.sh`, затем
конвертируются в LeRobot-формат экспериментом
`2026-08-17_smolvla_pretrain_libero` (см. его README; там же проверка
round-trip и конвенции гриппера).

Дальше любой эксперимент воспроизводится с нуля детерминированно (сид 1000)
одной и той же последовательностью из его README:

```bash
cd experiments/<эксперимент>
source scripts/common_env.sh && pytest -q   # тесты протокола
scripts/prepare.sh                          # preflight-артефакты
scripts/audit.sh                            # аудит обучаемых параметров
scripts/smoke_dataset.sh && scripts/smoke_env.sh
scripts/production_smoke.sh <gpu>           # determinism gate
scripts/run_all.sh                          # fan-out: тренировки + оценки + отчёт
scripts/status.sh                           # прогресс в любой момент
```

Свой эксперимент делается копией ближайшего по смыслу каталога
`experiments/YYYY-MM-DD_HH-MM-SS_имя` (правила — в корневом `README.md`:
новый timestamped-каталог, код копируется, а не импортируется, пакет
регистрируется в корневом `pyproject.toml` + `uv lock`).
