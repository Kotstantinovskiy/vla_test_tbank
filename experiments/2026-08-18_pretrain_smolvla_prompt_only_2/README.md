# Official-pretrain SmolVLA prompt-only, прогон 2 (полное видео)

Двойник `2026-08-18_pretrain_smolvla_prompt_only` с одним отличием: на диск
пишутся **все 20 видео каждой задачи всех трёх условий** (в Trackio по-прежнему
уходит только эпизод 0 каждой задачи). Причина: в первом прогоне единственный
успех (task 4, episode 4) не попал на видео. Это независимый прогон — env-сиды
и init-состояния те же, но RNG-поток сэмплинга действий другой, поэтому исходы
пересэмплированы; результаты обоих прогонов самостоятельны и не заменяют друг
друга.

Дата: 2026-08-18. Первый оценочный эксперимент нового претрена
`2026-08-17_smolvla_pretrain_libero` (обучен на in-repo конверсии официальных
HDF5; seen-контроль 20/20).

Замороженный чекпойнт получает только текстовую инструкцию: 0 target-демо,
0 optimizer steps, веса не меняются. В отличие от всех прежних zero-shot
прогонов в репозитории, за этим пайплайном стоит пройденный позитивный
контроль, поэтому результаты на полу впервые интерпретируемы как свойство
обобщения, а не возможная поломка конвенций.

## Что измеряется

- checkpoint: `/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model`
  (symlink `artifacts/checkpoint`); SHA-256 весов фиксируется в
  `artifacts/checkpoint_manifest.json` и проверяется при агрегации;
- held-out suite: **все 10 задач** `libero_goal`; logical ID = environment ID,
  инструкции ассертятся в живой среде перед rollout;
- 20 эпизодов на задачу, seed 1000, batch 4, рендер 128×128 (из схемы
  чекпойнта — та же конвенция, что на претрене);
- условия: `true` (настоящая инструкция), `wrong` (инструкция следующей
  задачи по циклу), `nonsense` (фиксированный бессмысленный prompt).

Ожидание (prior-informed): [reports/PRIOR_EXPECTATION.md](reports/PRIOR_EXPECTATION.md).

## Воспроизведение

```bash
uv sync --frozen
cd experiments/2026-08-18_pretrain_smolvla_prompt_only_2

# Полный запуск: prepare + env smoke на всех 10 задачах, сперва `true` с
# верификацией, затем wrong/nonsense параллельно, агрегация, GIF, Trackio.
scripts/run_prompt_only.sh 0 1 2
```

Отдельные стадии:

```bash
source scripts/common_env.sh
scripts/prepare.sh
scripts/smoke_env.sh
scripts/eval_prompt.sh true 0
scripts/eval_prompt.sh wrong 1
scripts/eval_prompt.sh nonsense 2
python -m pretrain_smolvla_prompt_only_2.aggregate
scripts/log_trackio.sh
scripts/show_trackio.sh
```

## Где лежат результаты

```text
results/
├── raw/{true,wrong,nonsense}.json + videos/true/task_*/eval_episode_0.mp4
├── summary/{summary.json,metrics.csv,prompt_controls.png,trackio_manifest.json}
├── media/gifs/task_*_true.gif
└── logs/
```

Trackio-проект `pretrain-smolvla-prompt-only-2` (база в `artifacts/trackio/`).
