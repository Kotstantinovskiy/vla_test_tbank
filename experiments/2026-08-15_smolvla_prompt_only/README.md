# SmolVLA prompt-only на held-out LIBERO

Отдельный почти автономный эксперимент: замороженный SmolVLA, уже обученный на
`libero_90`, получает только текстовую инструкцию новой задачи. Target-демо не
загружаются, градиентных шагов нет, веса не меняются.

Это новый запуск, а не перенос zero-shot файлов из
`2026-08-15_smolvla_libero_task1`. Старый эксперимент не импортируется и не
изменяется. Этот каталог владеет собственными скриптами, исходниками,
конфигурацией, сырыми rollout-результатами, видео, GIF, таблицей, графиком и
локальной базой Trackio.

## Что измеряется

- checkpoint: `crislmfroes/smolvla-libero-90` на ревизии
  `418f9d0e5b48585bcee1e1a7d47e302629af78da`;
- held-out suite: `libero_goal`, task IDs 0–2;
- 20 эпизодов на задачу, seed 1000, одинаковые reset seeds для всех условий;
- основной режим `true`: настоящая инструкция среды;
- контроли `wrong`: инструкция следующей target-задачи и `nonsense`: фиксированный
  бессмысленный prompt;
- 0 target-демонстраций, 0 optimizer steps.

Контроли меняют только строку, которую получает policy. Динамика задачи и
начальные состояния остаются теми же. Из-за известного результата прошлого
эксперимента ожидание здесь честно помечено как prior-informed, а не как слепая
пререгистрация: [reports/PRIOR_EXPECTATION.md](reports/PRIOR_EXPECTATION.md).

## Воспроизведение

Из корня репозитория:

```bash
uv sync --frozen
cd experiments/2026-08-15_smolvla_prompt_only

# Полный свежий запуск: три prompt-условия параллельно на GPU 0/1/2,
# затем агрегация, GIF и локальный Trackio.
scripts/run_prompt_only.sh 0 1 2
```

Отдельные стадии:

```bash
source scripts/common_env.sh
scripts/prepare.sh
scripts/eval_prompt.sh true 0
scripts/eval_prompt.sh wrong 1
scripts/eval_prompt.sh nonsense 2
python -m smolvla_prompt_only.aggregate
scripts/log_trackio.sh
scripts/show_trackio.sh
```

`prepare.sh` не скачивает датасет: он только создаёт experiment-local
`artifacts/libero_config/config.yaml`, привязанный к активному корневому
`.venv`. Модель берётся из зафиксированного общего Hugging Face cache; это
неизменяемый вход, а не общий экспериментальный код.

## Где лежат результаты

```text
results/
├── raw/
│   ├── true.json
│   ├── wrong.json
│   ├── nonsense.json
│   └── videos/true/task_*/eval_episode_0.mp4
├── summary/
│   ├── summary.json
│   ├── metrics.csv
│   ├── prompt_controls.png
│   └── trackio_manifest.json
├── media/gifs/task_*_true.gif
└── logs/
```

Trackio-проект `smolvla-prompt-only` содержит per-task кривые success для трёх
prompt-условий, полную таблицу метрик с Wilson 95% CI, PNG-график и rollout GIF.
По умолчанию база локальна в `artifacts/trackio/`. При необходимости публикации:

```bash
TRACKIO_SPACE_ID="username/vla-trackio" scripts/log_trackio.sh
```
