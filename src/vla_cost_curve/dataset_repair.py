from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def download_candidate_shards(
    root: str | Path, repo_id: str, revision: str, episodes: list[int]
) -> list[int]:
    """Download the bounded physical shard range containing selected episodes."""

    root = Path(root)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path, columns=["episode_index", "data/file_index"])
    rows = {
        int(episode): int(file_index)
        for episode, file_index in zip(
            table["episode_index"].to_pylist(),
            table["data/file_index"].to_pylist(),
            strict=True,
        )
    }
    reported = [rows[episode] for episode in episodes]
    # The upstream metadata rewrite shifted only shard boundaries. Expanding by
    # one shard on both sides covers every selected physical episode while
    # avoiding a download of all 377 ~100 MB data files.
    candidate_ids = list(range(max(0, min(reported) - 1), max(reported) + 2))
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=[f"data/chunk-000/file-{i:03d}.parquet" for i in candidate_ids],
    )
    return candidate_ids


def repair_selected_metadata(root: str | Path, episodes: list[int]) -> dict[str, Any]:
    """Repair shard mapping and frame bounds used by LeRobot's sampler.

    The pinned LIBERO revision stores correct global indices in each episode's
    ``stats/index/min|max`` fields, while ``dataset_from_index`` and
    ``dataset_to_index`` are relative to the data parquet shard.  LeRobot 0.6.1
    expects the latter pair to be global.  Only selected target episodes are
    changed, and the original metadata file is retained next to the repaired
    copy for auditability.
    """

    root = Path(root)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    backup = path.with_suffix(".parquet.upstream")
    audit_path = root / "meta/FRAME_BOUNDS_REPAIR.json"
    table = pq.read_table(path)
    upstream = pq.read_table(backup) if backup.exists() else table
    episode_ids = table["episode_index"].to_pylist()
    row_for_episode = {int(episode): row for row, episode in enumerate(episode_ids)}
    from_values = table["dataset_from_index"].to_pylist()
    to_values = table["dataset_to_index"].to_pylist()
    file_values = table["data/file_index"].to_pylist()
    stats_min = table["stats/index/min"].to_pylist()
    stats_max = table["stats/index/max"].to_pylist()

    physical_file_for_episode: dict[int, int] = {}
    for data_path in sorted((root / "data/chunk-000").glob("file-*.parquet")):
        file_index = int(data_path.stem.rsplit("-", 1)[1])
        for episode in set(pq.read_table(data_path, columns=["episode_index"])["episode_index"].to_pylist()):
            if episode in physical_file_for_episode:
                raise ValueError(f"Episode {episode} appears in multiple physical shards")
            physical_file_for_episode[int(episode)] = file_index

    repairs: list[dict[str, int]] = []
    for episode in sorted(set(episodes)):
        row = row_for_episode[episode]
        if episode not in physical_file_for_episode:
            raise FileNotFoundError(f"No downloaded physical shard contains episode {episode}")
        corrected_from = int(stats_min[row][0])
        corrected_to = int(stats_max[row][0]) + 1
        if corrected_to - corrected_from != int(table["length"][row].as_py()):
            raise ValueError(f"Global bounds disagree with length for episode {episode}")
        corrected_file = physical_file_for_episode[episode]
        upstream_from = int(upstream["dataset_from_index"][row].as_py())
        upstream_to = int(upstream["dataset_to_index"][row].as_py())
        upstream_file = int(upstream["data/file_index"][row].as_py())
        repairs.append(
            {
                "episode_index": episode,
                "old_file": upstream_file,
                "new_file": corrected_file,
                "old_from": upstream_from,
                "old_to": upstream_to,
                "new_from": corrected_from,
                "new_to": corrected_to,
            }
        )
        from_values[row] = corrected_from
        to_values[row] = corrected_to
        file_values[row] = corrected_file

    if not backup.exists():
        shutil.copy2(path, backup)
    table = table.set_column(
        table.schema.get_field_index("dataset_from_index"),
        "dataset_from_index",
        pa.array(from_values, type=pa.int64()),
    )
    table = table.set_column(
        table.schema.get_field_index("dataset_to_index"),
        "dataset_to_index",
        pa.array(to_values, type=pa.int64()),
    )
    table = table.set_column(
        table.schema.get_field_index("data/file_index"),
        "data/file_index",
        pa.array(file_values, type=pa.int64()),
    )
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)

    audit = {
        "reason": "metadata shard mapping and frame bounds disagree with physical parquet files",
        "bounds_source_of_truth": "stats/index/min and stats/index/max",
        "shard_source_of_truth": "physical episode_index column",
        "selected_episode_count": len(set(episodes)),
        "changed_bounds_count": sum(
            (r["old_from"], r["old_to"]) != (r["new_from"], r["new_to"]) for r in repairs
        ),
        "changed_file_count": sum(r["old_file"] != r["new_file"] for r in repairs),
        "repairs": repairs,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


# Backward-compatible name used by the first preparation implementation.
repair_selected_frame_bounds = repair_selected_metadata
