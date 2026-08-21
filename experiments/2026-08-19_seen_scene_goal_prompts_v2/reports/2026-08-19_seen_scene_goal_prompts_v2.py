import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="seen_scene_goal_prompts_v2")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go

    return Path, go, json, mo


@app.cell
def _(mo):
    mo.md("""
    # Seen-сцены × goal-промпты v2 — выводы

    Замороженный претрен (`seen_libero90_official_20260817`, seen-контроль
    20/20) в **знакомых** сценах libero_90; меняется только текст
    инструкции. **Первичный успех v2 = goal-предикат *подсказанной*
    задачи**, вычисляемый на каждом шаге в живой сцене (эпизод завершается
    и по нему); предикат env-задачи — вторичная, v1-сравнимая метрика.
    16 точек × 20 эпизодов, общие сиды/init-состояния внутри env → парные
    сравнения (точный McNemar). Все 320 видео на диске.
    """)
    return


@app.cell
def _(Path, json, mo):
    # Works both from the experiment's reports/ and from a repo-root symlink.
    _nb_dir = Path(mo.notebook_dir()).resolve()
    _candidates = [
        _nb_dir.parent / "results/summary/summary.json",
        _nb_dir.parent
        / "experiments/2026-08-19_seen_scene_goal_prompts_v2/results/summary/summary.json",
    ]
    summary = json.loads(
        next(p for p in _candidates if p.is_file()).read_text()
    )
    rows = summary["rows"]
    paired = summary["paired"]
    goal_slice = summary["goal_slice"]
    return goal_slice, paired, rows


@app.cell
def _(mo):
    mo.md("""
    ## 1. Успех по всем точкам: предикат промпта vs предикат среды

    Синее — доля эпизодов, где выполнена **подсказанная** задача; серое —
    где выполнена задача, «зашитая» в среду (метрика v1). Расхождение
    этих двух столбиков и есть главный сюжет эксперимента.
    """)
    return


@app.cell
def _(go, rows):
    _order = [r["label"] for r in rows]
    _short = [
        r["label"].replace("__", "<br>").replace("_the_", "_")[:52] for r in rows
    ]
    _prompted = [r["success_rate"] for r in rows]
    _env = [r["env_task_success_rate"] for r in rows]
    _err_low = [r["success_rate"] - r["ci95_low"] for r in rows]
    _err_high = [r["ci95_high"] - r["success_rate"] for r in rows]
    _block_color = {
        "trained": "#3264a8",
        "paraphrase": "#2e9e60",
        "cross": "#d0802d",
        "goal": "#a04ac2",
        "nonsense": "#888888",
    }
    fig_all = go.Figure()
    fig_all.add_bar(
        x=_short,
        y=_prompted,
        name="предикат промпта (v2, первичный)",
        marker_color=[_block_color[r["block"]] for r in rows],
        error_y=dict(type="data", array=_err_high, arrayminus=_err_low, width=3),
        hovertext=[
            f"{r['label']}<br>env: {r['env_instruction']}<br>prompt: {r['prompt']}"
            f"<br>prompted {r['successes']}/{r['trials']}, "
            f"env {r['env_task_successes']}/{r['trials']}"
            for r in rows
        ],
        hoverinfo="text",
    )
    fig_all.add_bar(
        x=_short,
        y=_env,
        name="предикат среды (метрика v1)",
        marker_color="#c9c9c9",
        opacity=0.85,
    )
    fig_all.update_layout(
        barmode="group",
        height=520,
        margin=dict(t=60, b=140),
        title="Все 16 точек: цвет столбца = блок (Wilson 95% CI на первичной метрике)",
        yaxis=dict(title="success rate", range=[0, 1.08]),
        xaxis=dict(tickfont=dict(size=9)),
        legend=dict(orientation="h", y=1.06),
    )
    fig_all
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Cross-блок: instruction following стал числом

    Одна кухня, две обученные задачи, промпты поменяны местами. По метрике
    v1 (предикат среды) обе точки — 0/20, «обвал». По предикату
    **подсказанной** задачи — модель выполняет ровно то, что ей сказали,
    на уровне *родной* частоты навыка.
    """)
    return


@app.cell
def _(go, rows):
    _byl = {r["label"]: r for r in rows}
    _cases = [
        (
            "env «turn on the stove»<br>промпт «put the frying pan…»",
            _byl["cross__turn_on_the_stove"],
            _byl["trained__put_the_frying_pan_on_the_stove"],
        ),
        (
            "env «put the frying pan…»<br>промпт «turn on the stove»",
            _byl["cross__put_the_frying_pan_on_the_stove"],
            _byl["trained__turn_on_the_stove"],
        ),
    ]
    fig_cross = go.Figure()
    fig_cross.add_bar(
        x=[c[0] for c in _cases],
        y=[c[1]["env_task_success_rate"] for c in _cases],
        name="метрика v1: предикат среды",
        marker_color="#c9c9c9",
        text=[f"{c[1]['env_task_successes']}/20" for c in _cases],
        textposition="outside",
    )
    fig_cross.add_bar(
        x=[c[0] for c in _cases],
        y=[c[1]["success_rate"] for c in _cases],
        name="v2: предикат подсказанной задачи",
        marker_color="#d0802d",
        error_y=dict(
            type="data",
            array=[c[1]["ci95_high"] - c[1]["success_rate"] for c in _cases],
            arrayminus=[c[1]["success_rate"] - c[1]["ci95_low"] for c in _cases],
        ),
        text=[f"{c[1]['successes']}/20" for c in _cases],
        textposition="outside",
    )
    fig_cross.add_bar(
        x=[c[0] for c in _cases],
        y=[c[2]["success_rate"] for c in _cases],
        name="native rate навыка (trained в его env)",
        marker_color="#3264a8",
        opacity=0.6,
        text=[f"{c[2]['successes']}/20" for c in _cases],
        textposition="outside",
    )
    fig_cross.update_layout(
        barmode="group",
        height=440,
        title="Cross 2×2: «обвал в 0» был свойством метрики, а не политики",
        yaxis=dict(title="success rate", range=[0, 1.15]),
        legend=dict(orientation="h", y=1.08),
    )
    fig_cross
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Срез всех 10 goal-промптов в seen-сценах

    Для каждого промпта libero_goal — одна seen-точка, где его предикат
    вычислим. Четыре промпта (goal 1/2/5/6) невычислимы **ни в одной** из
    90 сцен претрейна: нужные объекты не сосуществуют (например, миска и
    плита) — прогонов нет, зафиксированы как skipped.
    """)
    return


@app.cell
def _(go, goal_slice):
    _labels = [f"goal {it['goal_id']}<br>{it['prompt'][:38]}" for it in goal_slice]
    _vals = [it.get("success_rate") if it["status"] != "skipped" else 0 for it in goal_slice]
    _colors = {
        "verbatim_trained": "#3264a8",
        "paraphrase_of_trained": "#2e9e60",
        "novel_string": "#a04ac2",
    }
    _bar_colors = [
        "#dddddd" if it["status"] == "skipped" else _colors[it["relationship"]]
        for it in goal_slice
    ]
    _texts = [
        "skipped:<br>нет сцены"
        if it["status"] == "skipped"
        else f"{it['successes']}/{it['trials']}"
        for it in goal_slice
    ]
    fig_slice = go.Figure(
        go.Bar(
            x=_labels,
            y=_vals,
            marker_color=_bar_colors,
            text=_texts,
            textposition="outside",
            hovertext=[
                f"status: {it['status']}"
                + (f"<br>host env: {it.get('env_instruction')}" if it["status"] != "skipped" else "")
                + (f"<br>relationship: {it.get('relationship')}" if it["status"] != "skipped" else "")
                for it in goal_slice
            ],
            hoverinfo="text",
        )
    )
    fig_slice.update_layout(
        height=460,
        title=(
            "Goal-промпты в seen-сценах, успех = предикат промпта "
            "(синий=verbatim-trained, зелёный=парафраза, фиолетовый=новая строка)"
        ),
        yaxis=dict(title="prompted success rate", range=[0, 1.15]),
        xaxis=dict(tickfont=dict(size=9)),
        margin=dict(b=120),
    )
    fig_slice
    return


@app.cell
def _(mo, paired):
    _table_lines = [
        "## 4. Парные сравнения с trained той же среды (первичная метрика)",
        "",
        "| env | блок | trained → условие | Δ | McNemar p |",
        "|---|---|---|---:|---:|",
    ] + [
        f"| `{p['env_instruction'][:44]}` | {p['block']} | {p['trained']} → "
        f"{p['condition']} | {p['delta']:+.2f} | {p['mcnemar_p']:.4g} |"
        for p in paired
    ]
    mo.md("\n".join(_table_lines))
    return


@app.cell
def _(mo, rows):
    _viol = sum(r["consistency_violations"] for r in rows)
    mo.md(
        f"""
        ## Выводы (против предрегистрированных правил)

        1. **R1v2 сработало на максимуме: языковой селектор исправен.**
           Cross prompted-success — 17/20 и 20/20 — на уровне native rate
           навыков (16/20 и 20/20). «Обвал cross в 0» из v1 был свойством
           метрики; язык каузально и чисто выбирает навык при одинаковой
           картинке.
        2. **Хрупкость к форме строки подтверждена** (вердикт v1 R2 в силе):
           честные goal-парафразы дают 0–2/20 при McNemar p<0.05 на всех
           четырёх парах — выпадение одного слова ломает выполнение.
        3. **R3v2 не сработало: новая строка не запускает новый навык** —
           goal-0 «open the middle drawer…» 0/20 по своему предикату. Но
           деградация ступенчатая: под этой родственной строкой модель в
           15/20 выполнила *обученный навык сцены* (close top drawer), тогда
           как под бессмыслицей — 0/20. Иерархия: точная строка → навык;
           родственная новая → деградированный навык сцены; nonsense → ничего.
        4. **Целостность измерения (R4v2 чист)**: {_viol} расхождений
           внешнего вычисления предикатов с `env.check_success()`; все 14
           общих с v1 точек воспроизведены **поэпизодно** точно.

        ### Что это значит для Задачи 2

        Экшн-эксперт здоров, селектор здоров — сломано ровно одно звено:
        **обобщение строка→навык**. Главный рычаг — hindsight-relabeling /
        аугментация инструкций (многие формулировки → один навык), с
        mixed-batch, чтобы не разрушить рабочий селектор. Предупреждение из
        среза: для goal-задач 1/2/5/6 retrieval из seen-данных может дать
        только примитивы движений — end-to-end демонстраций с нужной парой
        объектов в претрейне не существует.
        """
    )
    return


if __name__ == "__main__":
    app.run()
