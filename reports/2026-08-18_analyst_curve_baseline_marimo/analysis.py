import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="LIBERO naive-baseline cost curve")


@app.cell
def _():
    import csv
    import json
    import math
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    return Path, go, json, math, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Наивный SmolVLA baseline: интерактивный разбор cost curve

    Производный анализ результатов `analyst_curve_baseline` без новых
    rollout’ов и обучения. Одна expert-only схема, 2 000 optimizer steps,
    seed 1000 и первые официальные `demo_0..demo_{k-1}` измерены при
    **k = 0/1/2/3/5/10/25** на десяти задачах LIBERO-Goal.

    Главный вопрос: **где именно кривая сдвигается влево и есть ли ещё
    измеримый режим после насыщения к k=5?**
    """)
    return


@app.cell
def _(Path, json, mo, pd):
    report_dir = Path(mo.notebook_dir()).resolve()
    artifact_dir = report_dir / "artifacts"
    source_summary = json.loads((artifact_dir / "summary.json").read_text())
    curves_df = pd.read_csv(artifact_dir / "combined_curve.csv")
    budgets = [int(value) for value in source_summary["budgets"]]
    task_instructions = {
        0: "open the middle drawer of the cabinet",
        1: "put the bowl on the stove",
        2: "put the wine bottle on top of the cabinet",
        3: "open the top drawer and put the bowl inside",
        4: "put the bowl on top of the cabinet",
        5: "push the plate to the front of the stove",
        6: "put the cream cheese in the bowl",
        7: "turn on the stove",
        8: "put the bowl on the plate",
        9: "put the wine bottle on the rack",
    }
    return artifact_dir, budgets, curves_df, source_summary, task_instructions


@app.cell
def _(artifact_dir, mo, source_summary):
    mo.callout(
        mo.md(
            f"""
            **Зафиксированный snapshot.** Notebook читает только локальные
            `{artifact_dir.name}/combined_curve.csv` и `summary.json`.
            Исходный checkpoint: `{source_summary['checkpoint']}`.
            Исходный эксперимент не импортируется и не изменяется.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    curve_selector = mo.ui.multiselect(
        options=[
            "mean_all_10",
            "mean_tasks_0_2",
            *[f"task_{task_id}" for task_id in range(10)],
        ],
        value=["mean_all_10", "mean_tasks_0_2"],
        label="Кривые",
    )
    x_scale = mo.ui.dropdown(
        options=["linear k", "log2(1+k)"],
        value="linear k",
        label="Шкала бюджета",
    )
    threshold_slider = mo.ui.slider(
        start=0.5,
        stop=1.0,
        step=0.05,
        value=0.8,
        label="Целевой success",
        show_value=True,
    )
    mo.hstack([curve_selector, x_scale, threshold_slider], widths="equal")
    return curve_selector, threshold_slider, x_scale


@app.cell
def _(math):
    def wilson_interval(successes: int, trials: int, z: float = 1.95996398454):
        if trials <= 0:
            return (0.0, 0.0)
        rate = successes / trials
        denom = 1 + z**2 / trials
        center = (rate + z**2 / (2 * trials)) / denom
        spread = (
            z
            * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
            / denom
        )
        return max(0.0, center - spread), min(1.0, center + spread)

    return (wilson_interval,)


@app.cell
def _(
    budgets,
    curve_selector,
    curves_df,
    go,
    math,
    task_instructions,
    wilson_interval,
    x_scale,
):
    _labels = {
        "mean_all_10": "среднее: все 10 задач",
        "mean_tasks_0_2": "среднее: задачи 0–2",
        **{
            f"task_{task_id}": f"task {task_id}: {instruction}"
            for task_id, instruction in task_instructions.items()
        },
    }
    _colors = {
        "mean_all_10": "#275dad",
        "mean_tasks_0_2": "#d97706",
    }
    _x = (
        budgets
        if x_scale.value == "linear k"
        else [math.log2(1 + budget) for budget in budgets]
    )
    _figure = go.Figure()
    for _index, _series in enumerate(curve_selector.value):
        _row = curves_df.loc[curves_df["series"] == _series].iloc[0]
        _values = [float(_row[f"k={budget}"]) for budget in budgets]
        _trials = 200 if _series == "mean_all_10" else 60 if _series == "mean_tasks_0_2" else 20
        _intervals = [
            wilson_interval(round(value * _trials), _trials) for value in _values
        ]
        _figure.add_scatter(
            x=_x,
            y=_values,
            mode="lines+markers",
            name=_labels[_series],
            line=dict(width=4 if _series.startswith("mean_") else 2),
            marker=dict(size=9 if _series.startswith("mean_") else 7),
            marker_color=_colors.get(_series),
            error_y=dict(
                type="data",
                array=[high - value for value, (_, high) in zip(_values, _intervals)],
                arrayminus=[value - low for value, (low, _) in zip(_values, _intervals)],
                width=3,
                thickness=1.2,
            ),
            customdata=budgets,
            hovertemplate="k=%{customdata}<br>success=%{y:.3f}<extra>%{fullData.name}</extra>",
        )
    _figure.update_layout(
        height=560,
        title="Cost curve с pooled Wilson 95% CI",
        xaxis=dict(
            title="число демонстраций k",
            tickmode="array",
            tickvals=_x,
            ticktext=budgets,
        ),
        yaxis=dict(title="success rate", range=[-0.04, 1.08]),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=90),
        hovermode="x unified",
    )
    _figure
    return


@app.cell
def _(budgets, curves_df, mo, pd, wilson_interval):
    _headline_rows = []
    for _series, _trials in (("mean_all_10", 200), ("mean_tasks_0_2", 60)):
        _row = curves_df.loc[curves_df["series"] == _series].iloc[0]
        for _budget in budgets:
            _rate = float(_row[f"k={_budget}"])
            _successes = round(_rate * _trials)
            _low, _high = wilson_interval(_successes, _trials)
            _headline_rows.append(
                {
                    "series": _series,
                    "k": _budget,
                    "success": _rate,
                    "successes/trials": f"{_successes}/{_trials}",
                    "Wilson 95% CI": f"[{_low:.3f}, {_high:.3f}]",
                }
            )
    _headline_df = pd.DataFrame(_headline_rows)
    mo.vstack(
        [
            mo.md(
                """
                ## 1. Headline: потолок виден уже при k=5

                Для обязательных задач 0–2 средний success достигает **0.95
                при k=5**, остаётся 0.95 при k=10 и 0.90 при k=25. Интервалы
                перекрываются: падение при k=25 нельзя трактовать как
                доказанный вред дополнительных демонстраций. Оно показывает,
                что при 20 эпизодах на задачу и одном training seed область
                k≥5 уже плохо различима.
                """
            ),
            mo.ui.table(_headline_df, selection=None),
        ]
    )
    return


@app.cell
def _(budgets, curves_df, pd):
    _efficiency_rows = []
    for _series in ("mean_all_10", "mean_tasks_0_2"):
        _row = curves_df.loc[curves_df["series"] == _series].iloc[0]
        for _left, _right in zip(budgets, budgets[1:]):
            _gain = float(_row[f"k={_right}"]) - float(_row[f"k={_left}"])
            _efficiency_rows.append(
                {
                    "series": _series,
                    "segment": f"{_left}→{_right}",
                    "delta_success": _gain,
                    "gain_per_added_demo": _gain / (_right - _left),
                }
            )
    efficiency_df = pd.DataFrame(_efficiency_rows)
    return (efficiency_df,)


@app.cell
def _(efficiency_df, go, mo):
    _efficiency_figure = go.Figure()
    for _series, _color in (
        ("mean_all_10", "#275dad"),
        ("mean_tasks_0_2", "#d97706"),
    ):
        _part = efficiency_df[efficiency_df["series"] == _series]
        _efficiency_figure.add_bar(
            x=_part["segment"],
            y=_part["gain_per_added_demo"],
            name=_series,
            marker_color=_color,
            text=[f"{value:+.3f}" for value in _part["gain_per_added_demo"]],
            textposition="outside",
        )
    _efficiency_figure.add_hline(y=0, line_color="#555")
    _efficiency_figure.update_layout(
        barmode="group",
        height=460,
        title="Предельная отдача: Δ success на одну добавленную демонстрацию",
        xaxis_title="интервал бюджета",
        yaxis_title="Δ success / demo",
        legend=dict(orientation="h", y=1.08),
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Где находится «левый сдвиг»

                Самая дорогая единица информации — **первая демонстрация**:
                +0.545 success на всех задачах и +0.567 на задачах 0–2.
                После k=5 средняя отдача близка к нулю. Следовательно, для
                сравнения методов наиболее доказательны k=1–2, как и требует
                ceiling-clause задания.
                """
            ),
            _efficiency_figure,
            mo.ui.table(efficiency_df.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(budgets, curves_df, pd, task_instructions, threshold_slider):
    _threshold_rows = []
    for _task_id, _instruction in task_instructions.items():
        _row = curves_df.loc[curves_df["series"] == f"task_{_task_id}"].iloc[0]
        _eligible = [
            budget
            for budget in budgets
            if float(_row[f"k={budget}"]) >= threshold_slider.value
        ]
        _threshold_rows.append(
            {
                "task_id": _task_id,
                "instruction": _instruction,
                "first_measured_k": min(_eligible) if _eligible else None,
                "max_observed_success": max(
                    float(_row[f"k={budget}"]) for budget in budgets
                ),
            }
        )
    threshold_df = pd.DataFrame(_threshold_rows)
    return (threshold_df,)


@app.cell
def _(go, mo, threshold_df, threshold_slider):
    _threshold_plot_values = [
        27 if value != value else value for value in threshold_df["first_measured_k"]
    ]
    _threshold_text = [
        ">25" if value != value else f"k={int(value)}"
        for value in threshold_df["first_measured_k"]
    ]
    _threshold_figure = go.Figure(
        go.Bar(
            x=[f"task {task_id}" for task_id in threshold_df["task_id"]],
            y=_threshold_plot_values,
            text=_threshold_text,
            textposition="outside",
            marker_color=[
                "#d14343" if value == 27 else "#2f855a"
                for value in _threshold_plot_values
            ],
            hovertext=threshold_df["instruction"],
            hoverinfo="text+y",
        )
    )
    _threshold_figure.update_layout(
        height=430,
        title=f"Первый измеренный k с success ≥ {threshold_slider.value:.2f}",
        xaxis_title="задача",
        yaxis=dict(title="минимальный измеренный k", tickvals=[0, 1, 2, 3, 5, 10, 25, 27], ticktext=[0, 1, 2, 3, 5, 10, 25, ">25"]),
        margin=dict(t=70),
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 3. Не одна кривая, а разные режимы сложности

                Выберите порог сверху. При success≥0.8 задачи 1 и 7 требуют
                лишь одну демонстрацию, task 2 — две, task 0 — пять, task 5 —
                десять, а task 3 не достигает порога даже при k=25. Средняя
                кривая скрывает эту неоднородность.
                """
            ),
            _threshold_figure,
            mo.ui.table(threshold_df, selection=None),
        ]
    )
    return


@app.cell
def _(budgets, curves_df, go, mo, task_instructions):
    _task_rows = [
        curves_df.loc[curves_df["series"] == f"task_{task_id}"].iloc[0]
        for task_id in task_instructions
    ]
    _heat_values = [
        [float(row[f"k={budget}"]) for budget in budgets] for row in _task_rows
    ]
    _heatmap = go.Figure(
        go.Heatmap(
            z=_heat_values,
            x=[f"k={budget}" for budget in budgets],
            y=[f"task {task_id}" for task_id in task_instructions],
            text=[[f"{value:.2f}" for value in row] for row in _heat_values],
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="success"),
            hovertext=[
                [instruction] * len(budgets)
                for instruction in task_instructions.values()
            ],
        )
    )
    _heatmap.update_layout(
        height=560,
        title="Per-task cost curves: каждая клетка = 20 rollout’ов",
        xaxis_title="бюджет демонстраций",
        yaxis_title="LIBERO-Goal task",
    )
    mo.vstack([mo.md("## 4. Карта неоднородности"), _heatmap])
    return


@app.cell
def _(curves_df, go, mo):
    _auc = curves_df[
        ["series", "auc_normalized", "auc_log2_normalized"]
    ].copy()
    _auc_figure = go.Figure()
    _auc_figure.add_bar(
        y=_auc["series"],
        x=_auc["auc_normalized"],
        orientation="h",
        name="linear nAUC",
        marker_color="#7796c7",
    )
    _auc_figure.add_bar(
        y=_auc["series"],
        x=_auc["auc_log2_normalized"],
        orientation="h",
        name="log2(1+k) nAUC",
        marker_color="#e29a3b",
    )
    _auc_figure.update_layout(
        barmode="group",
        height=570,
        title="AUC: линейная шкала vs акцент на дешёвых k",
        xaxis=dict(title="normalized AUC", range=[0, 1]),
        legend=dict(orientation="h", y=1.07),
        margin=dict(l=180),
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 5. Один скаляр — только с оговоркой о шкале

                Линейный nAUC (`0.818` по десяти задачам) сильно весит широкий
                интервал k=10→25. Log2-nAUC (`0.689`) сильнее весит k=0→3 и
                поэтому лучше соответствует цели «тот же success за меньше
                демо». Для сравнения методов нужно заранее фиксировать шкалу,
                иначе headline меняется вместе с предпочтением аналитика.
                """
            ),
            _auc_figure,
            mo.ui.table(_auc.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(budgets, curves_df, mo, pd, task_instructions):
    _violations = []
    for _task_id, _instruction in task_instructions.items():
        _row = curves_df.loc[curves_df["series"] == f"task_{_task_id}"].iloc[0]
        for _left, _right in zip(budgets, budgets[1:]):
            _before = float(_row[f"k={_left}"])
            _after = float(_row[f"k={_right}"])
            if _after < _before:
                _violations.append(
                    {
                        "task": _task_id,
                        "instruction": _instruction,
                        "segment": f"{_left}→{_right}",
                        "before": _before,
                        "after": _after,
                        "delta": _after - _before,
                    }
                )
    _violation_df = pd.DataFrame(_violations).sort_values("delta")
    mo.vstack(
        [
            mo.md(
                f"""
                ## 6. Почему кривая не обязана быть монотонной

                Найдено **{len(_violation_df)}** локальных снижений success при
                росте k. Это не доказывает отрицательный эффект данных:
                каждую клетку оценивали на 20 эпизодах, обучение имеет один
                seed, а поэпизодных парных исходов в snapshot нет. Правильная
                формулировка — «насыщение плюс измерительная/optimization
                variance», пока нет второго training seed и парного теста.
                """
            ),
            mo.ui.table(_violation_df.round(3), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Итоговый аналитический вердикт

    1. **Ceiling-clause активирован.** На обязательных tasks 0–2 baseline
       уже 0.95 при k=5; k=5/10/25 не дают убедительного пространства для
       победы. Основное сравнение метода должно включать k=1–2.
    2. **Первое демо создаёт большую часть левого сдвига**, но это среднее:
       task 0 требует k≈5, а task 3 остаётся тяжёлой при k=25.
    3. **Рекомендуемый headline — log2-nAUC**, дополненный per-task
       threshold-таблицей. Один linear nAUC скрывает именно дешёвую часть
       кривой, ради которой поставлено задание.
    4. **Числа пока single-seed.** Для заявлений о превосходстве метода
       нужны минимум два training seed. Кроме того, k=0 использует
       pretrain-нормализацию, а k>0 — target-нормализацию после LeRobot
       normalizer swap; скачок 0→1 нельзя целиком приписать демонстрации.
    5. **Источник k=0 удалён 2026-08-19**, но его точная репликация
       `pretrain_smolvla_prompt_only_2` сохранена: тот же единственный
       успех на той же задаче и эпизоде. Этот report хранит собственный
       immutable snapshot агрегированных чисел.
    """)
    return


if __name__ == "__main__":
    app.run()
