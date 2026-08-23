from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .constants import ARTIFACTS_DIR, MANIFEST_PATH, RAW_DIR, SUMMARY_DIR
from .utils import atomic_json, atomic_jsonl, load_config, sha256_file, utc_now


FORBIDDEN_KEYS = {"success", "successes", "success_rate", "reward", "outcome", "max_reward", "sum_reward"}


def build_manifest(config: dict) -> list[dict]:
    scope = config["scope"]
    tasks = {int(key): value for key, value in config["tasks"].items()}
    bundle_root = Path(config["sources"]["bundle_experiment"])
    pretrain_root = Path(config["sources"]["pretrain_experiment"])
    rows: list[dict] = []
    for task_id in scope["task_ids"]:
        candidates: list[tuple[str, int | None, Path]] = [
            (
                "pretrain",
                None,
                pretrain_root / "results" / "raw" / "videos" / "true" / f"task_{task_id}",
            )
        ]
        candidates.extend(
            (
                f"bundle_k_{budget}",
                int(budget),
                bundle_root
                / "results"
                / "raw"
                / "videos"
                / f"task_{task_id}"
                / f"k_{budget}"
                / "n_50",
            )
            for budget in scope["budgets"]
        )
        for candidate, budget, video_dir in candidates:
            for episode_ix in range(scope["episodes_per_candidate"]):
                video_path = (video_dir / f"eval_episode_{episode_ix}.mp4").resolve()
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise FileNotFoundError(video_path)
                rows.append(
                    {
                        "video_id": f"task_{task_id}__{candidate}__episode_{episode_ix:02d}",
                        "task_id": int(task_id),
                        "instruction": tasks[int(task_id)],
                        "candidate": candidate,
                        "demo_budget": budget,
                        "episode_ix": episode_ix,
                        "n_action_steps": 50,
                        "video_path": str(video_path),
                        "video_bytes": video_path.stat().st_size,
                    }
                )
    expected = int(scope["expected_videos"])
    if len(rows) != expected or len({row["video_id"] for row in rows}) != expected:
        raise AssertionError(f"expected {expected} unique videos, got {len(rows)}")
    if any(FORBIDDEN_KEYS.intersection(row) for row in rows):
        raise AssertionError("blind manifest leaked a target label")
    if {row["n_action_steps"] for row in rows} != {50}:
        raise AssertionError("manifest includes a non-50 action schedule")
    return rows


def prepare() -> None:
    config = load_config()
    for directory in (ARTIFACTS_DIR, RAW_DIR, SUMMARY_DIR, RAW_DIR.parent / "logs", RAW_DIR.parent / "media"):
        directory.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(config)
    atomic_jsonl(MANIFEST_PATH, rows)
    manifest = {
        "created_at": utc_now(),
        "entries": len(rows),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "n_action_steps": 50,
        "contains_target_labels": False,
        "task_counts": {
            str(task): sum(row["task_id"] == task for row in rows)
            for task in config["scope"]["task_ids"]
        },
    }
    atomic_json(ARTIFACTS_DIR / "manifest_summary.json", manifest)
    policy_config_path = Path(config["sources"]["pretrain_checkpoint"]) / "config.json"
    policy_config = json.loads(policy_config_path.read_text())
    schedule_audit = {
        "pretrain_policy_config": str(policy_config_path),
        "pretrain_chunk_size": int(policy_config["chunk_size"]),
        "pretrain_n_action_steps": int(policy_config["n_action_steps"]),
        "bundle_video_path_component": "n_50",
        "manifest_n_action_steps": sorted({row["n_action_steps"] for row in rows}),
    }
    if schedule_audit["pretrain_n_action_steps"] != 50 or schedule_audit["manifest_n_action_steps"] != [50]:
        raise AssertionError("source schedule is not n_action_steps=50")
    atomic_json(ARTIFACTS_DIR / "source_schedule_audit.json", schedule_audit)
    for name, target in {
        "source_bundle_experiment": config["sources"]["bundle_experiment"],
        "source_pretrain_experiment": config["sources"]["pretrain_experiment"],
        "source_pretrain_checkpoint": config["sources"]["pretrain_checkpoint"],
        "own_critic_checkpoint": config["own_critic"]["checkpoint"],
        "robometer_checkpoint": config["robometer"]["local_snapshot"],
        "robometer_base_processor": config["robometer"]["base_processor_snapshot"],
    }.items():
        link = ARTIFACTS_DIR / name
        if not link.exists() and not link.is_symlink():
            os.symlink(target, link)
    print(f"prepared {len(rows)} blind video records: {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    prepare()


if __name__ == "__main__":
    main()
