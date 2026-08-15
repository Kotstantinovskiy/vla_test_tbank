from pathlib import Path

from vla_cost_curve.trackio_report import (
    cost_curve_points,
    cost_curve_table,
    language_control_table,
    parse_training_log,
)


def _summary():
    metric = {
        "successes": 1,
        "trials": 2,
        "success_rate": 0.5,
        "ci95_low": 0.1,
        "ci95_high": 0.9,
    }
    return {
        "tasks": {
            "0": {
                "instruction": "do the thing",
                "k0": {condition: metric for condition in ("true", "wrong", "nonsense")},
                "adapted": {str(k): metric for k in (5, 10, 25)},
            }
        },
        "mean_cost_curve": {str(k): 0.5 for k in (0, 5, 10, 25)},
    }


def test_parse_training_log_uses_exact_periodic_steps(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "INFO x ot_train.py:641 step:25 loss:0.5 grdn:1.2 lr:1e-4 smp/s:30\n"
        "INFO x ot_train.py:641 step:50 loss:0.4 grdn:1.0 lr:9e-5 smp/s:31\n"
    )

    points = parse_training_log(log, log_frequency=25)

    assert [point["step"] for point in points] == [25, 50]
    assert points[-1]["train/loss"] == 0.4
    assert points[-1]["throughput/samples_per_second"] == 31


def test_cost_curve_and_tables_cover_all_budgets_and_controls():
    summary = _summary()

    assert [point["step"] for point in cost_curve_points(summary)] == [0, 5, 10, 25]
    assert len(cost_curve_table(summary)[1]) == 4
    assert len(language_control_table(summary)[1]) == 3
