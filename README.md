# VLA experiments

Репозиторий организован как журнал почти автономных экспериментов. Каждый
запуск хранит свою версию кода, конфигурации, команд запуска, результатов и
артефактов. Это позволяет вернуться к старому эксперименту без зависимости от
того, как изменился код следующего.

## Реестр

| Дата | Эксперимент | Статус | Результат |
|---|---|---|---|
| 2026-08-15 | `smolvla_prompt_only` | удалён 2026-08-18 (prompt-only стороннего crislmfroes-чекпойнта: 0/60 floor; чекпойнт стёрт из HF-кэша) | — |
| 2026-08-15 | `smolvla_action_steps_sweep` | удалён 2026-08-18 (свип по n_action_steps на crislmfroes-линейке; собственных чекпойнтов не имел) | — |
| 2026-08-16 | `smolvla_libero90_full_ddp` | удалён 2026-08-18: претрен на зеркальной crislmfroes-конверсии, заменён `smolvla_pretrain_libero`; чекпойнты стёрты | — |
| 2026-08-16 | `self_smolvla_naive_learn_baseline` | удалён 2026-08-18 (наивный бейзлайн на зеркальном претрене: 0.900/0.917/0.967 на задачах 0–2; 38 ГБ чекпойнтов стёрты; заменён pretrain_smolvla_naive_baseline_few_shot_tune) | — |
| 2026-08-17 | `self_smolvla_prompt_only` | удалён 2026-08-18 (prompt-only зеркального претрена: 0/20×3 условия на задачах 0–2; orientation-диагностика: 0/20 и в pretrain-ориентации → зеркало не было причиной zero-shot нуля; не был в git — сырые данные утрачены, выжимка здесь) | — |
| 2026-08-17 | [`smolvla_pretrain_libero`](experiments/2026-08-17_smolvla_pretrain_libero/) | завершён: претрен + seen-контроль 20/20 | [отчёт](experiments/2026-08-17_smolvla_pretrain_libero/reports/REPORT.md) |
| 2026-08-18 | `pretrain_smolvla_prompt_only` | удалён 2026-08-19: zero-shot 1/200 true, 0/200 wrong и nonsense; собственных чекпойнтов не имел, raw/Trackio удалены; точная репликация сохранена в `pretrain_smolvla_prompt_only_2` | — |
| 2026-08-18 | [`pretrain_smolvla_prompt_only_2`](experiments/2026-08-18_pretrain_smolvla_prompt_only_2/) | завершён: точная репликация (1/200, та же задача/эпизод), все 600 видео на диске | [отчёт](experiments/2026-08-18_pretrain_smolvla_prompt_only_2/reports/REPORT.md) |
| 2026-08-18 | [`pretrain_smolvla_naive_baseline_few_shot_tune`](experiments/2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune/) | завершён: 30/30, кривая 0.005→0.83/0.875/0.85 (10 задач), 0.95/0.95/0.90 (задачи 0–2) | [отчёт](experiments/2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune/reports/REPORT.md) |
| 2026-08-18 | [`pretrain_smolvla_few_shot_tune_low_k`](experiments/2026-08-18_pretrain_smolvla_few_shot_tune_low_k/) | завершён: 30/30, k=1/2/3 = 0.55/0.705/0.78 (10 задач) | [отчёт](experiments/2026-08-18_pretrain_smolvla_few_shot_tune_low_k/reports/REPORT.md) |
| 2026-08-21 | [`pretrain_smolvla_naive_deterministic_repro`](experiments/2026-08-21_20-37-56_pretrain_smolvla_naive_deterministic_repro/) | выполняется повтор только assignment task IDs 0–2, k=5/10/25: train seed 1000; determinism gate task 0/k=5 прошёл (18/20 в обоих порядках) | — |
| 2026-08-21 | [`pretrain_smolvla_low_k_deterministic_repro`](experiments/2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro/) | только assignment task IDs 0–2, k=1/2/3; production gate task 0/k=1 прошёл (3/20 в обоих порядках); fanout ожидает naive | — |
| 2026-08-18 | [`analyst_curve_baseline`](experiments/2026-08-18_analyst_curve_baseline/) | завершён: общая кривая k=0..25, nAUC 0.818 (10 задач) / 0.877 (0–2), log2-nAUC 0.689 / 0.728 | [отчёт](experiments/2026-08-18_analyst_curve_baseline/reports/REPORT.md) |
| 2026-08-18 | `seen_scene_goal_prompts` | удалён 2026-08-19 (v1 языковых проб, env-метрика: trained 11–20/20, парафразы 0–2/20 McNemar p<0.05 на 4/4, cross/absent/nonsense 0/20 → правило R2; заменён v2 с предикатным успехом, который воспроизвёл его поэпизодно; в git не был — каталог перенесён в /var/tmp/vla_recovery_20260819/) | — |
| 2026-08-19 | [`seen_scene_goal_prompts_v2`](experiments/2026-08-19_seen_scene_goal_prompts_v2/) | завершён: 16/16, успех = предикат промпта; instruction following в cross 17/20 и 20/20 (≈ native rate навыка); novel-строка goal-0: 0/20 по промпту, но 15/20 отката к навыку сцены; v1 воспроизведён поэпизодно, 0 расхождений предикатов (правило R1v2) | [вердикт](experiments/2026-08-19_seen_scene_goal_prompts_v2/reports/FINDINGS.md) |
| 2026-08-19 | [`pretrain_smolvla_prompt_only_3`](experiments/2026-08-19_pretrain_smolvla_prompt_only_3/) | завершён: 30/30, воспроизводимый пер-эпизодный шум (batch=1, noise_seed=1000+e; determinism-смоук: две раскладки бит-в-бит); true 1/200 (task 4), wrong 0/200, nonsense 0/200 — канонический k=0 = 0.005, совпал с `_2` (правило R3po3) | [отчёт](experiments/2026-08-19_pretrain_smolvla_prompt_only_3/reports/REPORT.md) |
| 2026-08-19 | [`goal_scene_seen_prompts`](experiments/2026-08-19_goal_scene_seen_prompts/) | завершён: 13/13; обученные строки НЕ спасают zero-shot (seen-twin ≤ true везде, якорь «turn on the stove» 20/20 seen → 0/20 goal) — сцена связывает k=0 (R1p3 опровергнуто); поведение: вовлечённые mid-funnel провалы, nonsense в goal-сцене не простаивает; попутно найден непинуемый сэмплинг-шум политики (task 4 k=0: 1/20 vs 5/20 между RNG-потоками) | [вердикт](experiments/2026-08-19_goal_scene_seen_prompts/reports/FINDINGS.md), [harness-note](experiments/2026-08-19_goal_scene_seen_prompts/reports/HARNESS_NOTE.md) |
| 2026-08-19 | [`base_smolvla_cost_curve`](experiments/2026-08-19_base_smolvla_cost_curve/) | завершён: 60/60, аудит заморозки чист; кривая с ГОЛОГО smolvla_base почти совпала с претрен-кривой (mean-10: 0.485/0.66/0.69/0.745/0.865/0.79 vs 0.55/0.705/0.78/0.83/0.875/0.85) → ценность libero_90-претрена при k≥1 всего ~0.05–0.09 (R1bc опровергнуто, R2bc сработало); k=25-провал воспроизвёлся и на base | [вердикт](experiments/2026-08-19_base_smolvla_cost_curve/reports/FINDINGS.md) |
| 2026-08-20 01:55:23 | [`goal_prompts_in_seen_hosts`](experiments/2026-08-20_01-55-23_goal_prompts_in_seen_hosts/) | завершён: 9/9, смоук детерминизма PASSED; goal-промпты 0/20 на всех хостах (R1gh на 2 здоровых хостах; wine-хост 8/20 исключён по R4gh); откат к навыку хоста только у drawer-рамки (19/20) → «graded fallback» v2 не общий (R3gh не сработало) | [вердикт](experiments/2026-08-20_01-55-23_goal_prompts_in_seen_hosts/reports/FINDINGS.md) |
| 2026-08-20 01:55:23 | [`seen_semantic_paraphrases`](experiments/2026-08-20_01-55-23_seen_semantic_paraphrases/) | завершён: 20/20, смоук PASSED; mean Δ=−0.36, p<0.05 на 4/10 → R3sp: хрупкость зависит от типа правки — посессивы/«atop»/reorder рушат (−0.55…−0.95), глагольные замены почти нет (0…−0.35) при сохранении всех дескрипторов | [вердикт](experiments/2026-08-20_01-55-23_seen_semantic_paraphrases/reports/FINDINGS.md) |
| 2026-08-20 01:55:23 | [`seen_article_drop`](experiments/2026-08-20_01-55-23_seen_article_drop/) | завершён: 20/20, смоук PASSED; mean Δ=−0.15, p<0.05 на 2/10 → R3ad: удаление артикля в основном безвредно, но два таска рушатся на −0.55 (bowl-on-cabinet, microwave) — чувствительность к служебным словам задаче-специфична | [вердикт](experiments/2026-08-20_01-55-23_seen_article_drop/reports/FINDINGS.md) |
| 2026-08-20 01:55:23 | [`seen_prompts_in_goal_scene`](experiments/2026-08-20_01-55-23_seen_prompts_in_goal_scene/) | завершён: 11/11, смоук PASSED; pooled 18/180 — floor-предсказание ОПРОВЕРГНУТО (R2sg): «open the top drawer» 10/20 и bowl-on-cabinet 7/20 переносятся в goal-сцену → сцена не абсолютный блокер, перенос skill/init-зависим (top-vs-middle confound задокументирован) | [вердикт](experiments/2026-08-20_01-55-23_seen_prompts_in_goal_scene/reports/FINDINGS.md) |
| 2026-08-20 20:03:21 | [`pretrain_smolvla_low_k_action_steps`](experiments/2026-08-20_20-03-21_pretrain_smolvla_low_k_action_steps/) | inference-only: переиспользует 9 готовых seed=1000 low-k checkpoint’ов для assignment task ID 0–2; paired eval при n_action_steps=1/10/25, 27 точек / 540 rollout; per-episode noise/env/init-state fixation | production gate пройден; inference sweep выполняется на GPU 1–3 |
| 2026-08-22 18:10:40 | [`pretrain_smolvla_full_ft_low_k`](experiments/2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k/) | завершён: 9/9 тренировок, 18/18 оценок, determinism gate оба варианта поэпизодно; полный файнтюнинг (392.9M/450.0M обучаемых) для task ID 0–2, k=1/2/3, сиды идентичны repro; кривая n=50: 0.583/0.950/0.767 (expert-only: 0.600/0.683/0.667), n=25: 0.533/0.750/0.800; выигрыш сконцентрирован в k=2 (drawer 6/20→20/20), кривая задачи 0 немонотонна (single-seed) | [отчёт](experiments/2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k/reports/REPORT.md) |
| 2026-08-22 22:19:39 | [`pretrain_smolvla_lora_low_k`](experiments/2026-08-22_22-19-39_pretrain_smolvla_lora_low_k/) | создан, НЕ запущен: LoRA r=16/α=32 на все используемые линейные слои VLM (text 0–14, vision, connector) + полный expert/проекции через PEFT modules_to_save (lerobot native PEFT, peft 0.20.0, merge в обычный чекпойнт перед eval); task ID 0–2, k=1/2/3, eval n_action_steps=50/25; сиды идентичны expert-only/full-FT; предсказания зафиксированы | [prior](experiments/2026-08-22_22-19-39_pretrain_smolvla_lora_low_k/reports/PRIOR_EXPECTATION.md) |
| 2026-08-22 22:43:55 | [`pretrain_smolvla_state_noise_k1`](experiments/2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1/) | завершён: 12/12 тренировок и оценок, gate поэпизодно (3/20 оба порядка); гауссов шум α·ε на нормализованном state только в обучении, α∈{0, 0.01, 0.03, 0.05} × task 0–2, полный файнтюнинг k=1, eval n=50; means 0.617/0.583/0.667/0.683 — лучший α=0.05 (+0.07 к контролю), эффект несёт task 2 (15→19/20), drawer почти не двигается (3→4/20); α=0-контроль воспроизвёл full-FT k=1 в пределах ±1 эпизода | [отчёт](experiments/2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1/reports/REPORT.md) |
| 2026-08-23 14:31:07 | [`pretrain_smolvla_state_noise_k1_1k_steps`](experiments/2026-08-23_14-31-07_pretrain_smolvla_state_noise_k1_1k_steps/) | завершён: 12/12, gate поэпизодно (3/20 оба порядка); α∈{0, 0.08, 0.1, 0.2} × task 0–2 при 1000 шагов; means 0.600/0.667/0.700/0.667 — α=0.1 лучший k=1 результат линейки (+0.10 к 1k-контролю) при вдвое меньшем обучении; 1000≈2000 шагов при k=1 (контроль 0.600 vs 0.617); drawer 3→6/20 плато на всех высоких α, перегиба дозы до 0.2 нет (адаптивная сетка задекларирована) | [отчёт](experiments/2026-08-23_14-31-07_pretrain_smolvla_state_noise_k1_1k_steps/reports/REPORT.md) |
| 2026-08-23 00:18:49 | [`pretrain_smolvla_image_aug_low_k`](experiments/2026-08-23_00-18-49_pretrain_smolvla_image_aug_low_k/) | завершён: 9/9 тренировок (шаги 1000/1500/2000 при k=1/2/3), 18/18 оценок, gate поэпизодно чист; дефолтные image-аугментации lerobot + budget-зависимые шаги против full-FT (в скобках): n=50 0.583 (0.583)/0.900 (0.950)/0.783 (0.767), n=25 0.567 (0.533)/0.783 (0.750)/0.817 (0.800); k=1 не потерял от половины шагов; drawer k=2 n=25 8/20→14/20 — аугментации сгладили провал перепланирования; на k=3 (чистое сравнение) эффект ≤1 эпизода | [отчёт](experiments/2026-08-23_00-18-49_pretrain_smolvla_image_aug_low_k/reports/REPORT.md) |
| 2026-08-23 18:20:07 | [`pretrain_smolvla_bundle_all_k`](experiments/2026-08-23_18-20-07_pretrain_smolvla_bundle_all_k/) | создан, НЕ запущен: бандл всех сработавших приёмов — полный файнтюн + image-аугментации (дефолт lerobot) + шум состояния α=0.10 + шаги 1000/1500/2000 (k=1/2/≥3); полная кривая k∈{1,2,3,5,10,25} × task 0–2 (18 тренировок), eval чистый при n=50/35/25 (54 точки); слотовый оркестратор: 2 тренировки + 2 оценки на GPU одновременно, оценки стартуют по готовности чекпойнта, gate-варианты параллельно по картам; предсказания зафиксированы | [prior](experiments/2026-08-23_18-20-07_pretrain_smolvla_bundle_all_k/reports/PRIOR_EXPECTATION.md) |

## Единое uv-окружение

Корневой `pyproject.toml` объединяет экспериментальные пакеты в uv-workspace,
а `uv.lock` фиксирует одно согласованное окружение Python 3.12 для всего
репозитория. Корневой проект виртуальный: в нём нет общего изменяемого кода, он
только собирает зависимости автономных экспериментов.
Workspace ограничен Linux x86_64, поскольку текущий робототехнический стек
фиксирует проверенные PyTorch 2.7.1 и CUDA 12.6 для GPU-запусков.

```bash
# Создать или актуализировать .venv строго по lock-файлу.
uv sync --frozen

# Запустить все тесты из общего окружения.
uv run --frozen pytest

# Выполнить команду внутри конкретного эксперимента.
uv run --frozen \
  --directory experiments/2026-08-18_pretrain_smolvla_prompt_only_2 \
  pretrain-prompt-2-aggregate
```

При добавлении нового каталога `experiments/YYYY-MM-DD_HH-MM-SS_name` его `pyproject.toml`
автоматически попадает в workspace через `experiments/*`. Чтобы пакет также
устанавливался обычным `uv sync`, его имя нужно добавить в корневые
`project.dependencies` и `tool.uv.sources` с `workspace = true`.

## Общий Trackio dashboard

Каждый эксперимент хранит собственную Trackio-базу и media внутри своего
`artifacts/trackio/`. Корневой launcher создаёт согласованный снимок баз,
индексирует media через hard links и открывает один dashboard со всеми проектами:

```bash
cd /home/nbagent174/vla_test
scripts/show_trackio.sh
```

Проекты переключаются в sidebar Trackio. Индекс в `.trackio-dashboard/`
генерируется при запуске и не дублирует исходные media-файлы. Hard links нужны,
потому что Trackio намеренно не отдаёт media, если симлинк разрешается за
пределами dashboard-каталога. Старые GIF, записанные как `trackio.Video`,
автоматически получают совместимую MP4-копию только в dashboard-cache;
experiment-local база не меняется. Добавленный новый эксперимент появится после
того, как он создаст свою `artifacts/trackio/*.db`. Посмотреть найденные проекты
без запуска UI можно командой `scripts/index_trackio.sh`.

## Официальные LIBERO-датасеты

Стабильный корневой загрузчик скачивает исходные HDF5 из официального
`yifengzhu-hf/LIBERO-datasets` на закреплённой ревизии
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. Частичные загрузки продолжаются,
а готовые файлы сверяются с LFS SHA-256 из Hub-манифеста.

```bash
# Только 10 целевых задач (~5.9 GiB).
scripts/download_official_libero_goal.sh

# Только 90 seen-задач (~62.1 GiB).
scripts/download_official_libero_90.sh

# Обе suite; libero_goal скачивается первой.
scripts/download_official_libero.sh --suite all
```

По умолчанию данные и машинно-читаемые manifest/status/verification лежат в
`/var/tmp/libero_official_f13aa24`. Другой каталог задаётся через
`--root /absolute/path`; `--max-workers` управляет параллелизмом.

## Контракт структуры

```text
experiments/
└── YYYY-MM-DD_HH-MM-SS_short_name/
    ├── README.md          # постановка, протокол и точные команды
    ├── pyproject.toml     # зафиксированное окружение Python
    ├── configs/           # конфигурации только этого эксперимента
    ├── scripts/           # entrypoints только этого эксперимента
    ├── src/               # снимок экспериментального кода
    ├── tests/             # проверки этого снимка
    ├── results/
    │   ├── raw/           # сырые rollout/metrics
    │   ├── summary/       # таблицы и графики
    │   ├── media/         # GIF и другие компактные результаты
    │   └── logs/          # runtime-логи
    ├── artifacts/         # веса, manifest, schema и служебные данные
    └── reports/           # предсказания, отклонения и выводы
```

Правила для новых экспериментов:

1. Имя каталога начинается с локального timestamp создания
   в 24-часовом формате: `YYYY-MM-DD_HH-MM-SS_name`
   (`date +%Y-%m-%d_%H-%M-%S`).
2. Эксперимент не импортирует изменяемый код из другого эксперимента. Нужную
   версию кода лучше скопировать и менять локально.
3. Все пути записи ведут внутрь каталога эксперимента. Большие данные могут
   физически лежать на отдельном диске, но ссылка на них также находится в
   `artifacts/` конкретного эксперимента.
4. В общий код выносится только инфраструктура с устойчивым контрактом,
   одинаковая для нескольких уже существующих экспериментов. Сейчас это
   корневой Trackio launcher и revision-pinned загрузчик официальных LIBERO
   HDF5; экспериментальный код остаётся локальным.
5. После завершения эксперимент считается снимком: новые идеи получают новый
   каталог, а не переписывают старые результаты.

Для запуска текущего эксперимента:

```bash
cd experiments/2026-08-18_pretrain_smolvla_prompt_only_2
source scripts/common_env.sh  # автоматически выбирает корневую .venv
pytest -q
```
