from __future__ import annotations

"""Convert official LIBERO HDF5 suites into LeRobot v3 datasets.

Contract (see constants.py): frames stored as rot180(official) at native
128x128, actions bit-exact float32, state = official ee_pos + ee_ori +
gripper_states (the eval-processor layout), fps=20, episodes in official
order (files sorted by name, demos by numeric index).  The conversion is
atomic per suite: it writes into ``<root>.building`` and renames on success,
so an existing final root is trusted and skipped.
"""

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import (
    CAMERA_KEYS,
    CONVERTED_ROOT,
    DEMOS_PER_TASK,
    EXPECTED_SEEN_TASKS,
    EXPECTED_TARGET_TASKS,
    FPS,
    IMAGE_SIZE,
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    OFFICIAL_ROOT,
    SEEN_REPO_ID,
    SEEN_SUITE,
    TARGET_REPO_ID,
    TARGET_SUITE,
    experiment_root,
)

SUITES = {
    SEEN_SUITE: {"repo_id": SEEN_REPO_ID, "expected_tasks": EXPECTED_SEEN_TASKS},
    TARGET_SUITE: {"repo_id": TARGET_REPO_ID, "expected_tasks": EXPECTED_TARGET_TASKS},
}


def natural_demo_index(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"Cannot parse demo index from {name!r}")
    return int(match.group(1))


def task_from_file(path: Path, suite: str) -> str:
    name = path.stem.removesuffix("_demo")
    if suite == SEEN_SUITE:
        name = re.sub(r"^[A-Z_]+\d+_", "", name)
    return name.replace("_", " ")


def rot180(frames: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frames[:, ::-1, ::-1])


def episode_arrays(demo: h5py.Group) -> dict[str, np.ndarray]:
    obs = demo["obs"]
    actions = np.asarray(demo["actions"], dtype=np.float32)
    state = np.concatenate(
        [
            np.asarray(obs["ee_pos"], dtype=np.float32),
            np.asarray(obs["ee_ori"], dtype=np.float32),
            np.asarray(obs["gripper_states"], dtype=np.float32),
        ],
        axis=-1,
    )
    cameras = {
        camera_key: rot180(np.asarray(obs[official_name]))
        for camera_key, official_name in CAMERA_KEYS.items()
    }
    lengths = {len(actions), len(state), *(len(value) for value in cameras.values())}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent stream lengths in demo: {lengths}")
    for camera_key, value in cameras.items():
        if value.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
            raise ValueError(
                f"Unexpected {camera_key} shape/dtype: {value.shape} {value.dtype}"
            )
    # Sanity: official ee_states must equal concat(ee_pos, ee_ori).
    ee_states = np.asarray(obs["ee_states"], dtype=np.float32)
    if not np.array_equal(ee_states, state[:, :6]):
        raise ValueError("Official ee_states != concat(ee_pos, ee_ori); recipe invalid")
    return {"actions": actions, "state": state, **cameras}


def dataset_features() -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        camera_key: {
            "dtype": "video",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        }
        for camera_key in CAMERA_KEYS
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    }
    features["action"] = {"dtype": "float32", "shape": (7,), "names": None}
    return features


def convert_suite(suite: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .constants import VIDEO_CRF

    spec = SUITES[suite]
    final_root = output_root / suite
    manifest_path = final_root / "conversion_manifest.json"
    if final_root.exists():
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text())
        raise FileExistsError(
            f"{final_root} exists without a conversion manifest; move it aside "
            "and rerun (refusing to overwrite)"
        )

    files = sorted((source_root / suite).glob("*_demo.hdf5"))
    if len(files) != spec["expected_tasks"]:
        raise FileNotFoundError(
            f"{suite}: expected {spec['expected_tasks']} official files, found "
            f"{len(files)} in {source_root / suite} (download incomplete?)"
        )

    building_root = final_root.with_name(final_root.name + ".building")
    if building_root.exists():
        shutil.rmtree(building_root)

    dataset = LeRobotDataset.create(
        repo_id=spec["repo_id"],
        fps=FPS,
        features=dataset_features(),
        root=building_root,
        robot_type="panda",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=8,
        rgb_encoder=RGBEncoderConfig(crf=VIDEO_CRF),
    )
    episodes: list[dict[str, Any]] = []
    for path in files:
        task = task_from_file(path, suite)
        with h5py.File(path, "r") as handle:
            demos = sorted(handle["data"].keys(), key=natural_demo_index)
            for demo_name in demos:
                arrays = episode_arrays(handle[f"data/{demo_name}"])
                for index in range(len(arrays["actions"])):
                    frame = {
                        "task": task,
                        "action": arrays["actions"][index],
                        "observation.state": arrays["state"][index],
                    }
                    for camera_key in CAMERA_KEYS:
                        frame[camera_key] = arrays[camera_key][index]
                    dataset.add_frame(frame)
                dataset.save_episode()
                episodes.append(
                    {
                        "episode_index": len(episodes),
                        "official_file": path.name,
                        "official_demo": demo_name,
                        "official_demo_index": natural_demo_index(demo_name),
                        "task": task,
                        "length": int(len(arrays["actions"])),
                    }
                )
    dataset.finalize()

    manifest = {
        "suite": suite,
        "repo_id": spec["repo_id"],
        "source_repo": OFFICIAL_REPO,
        "source_revision": OFFICIAL_REVISION,
        "source_root": str(source_root / suite),
        "converted_at": datetime.now(UTC).isoformat(),
        "orientation": "rot180(official) == eval convention",
        "image_size": IMAGE_SIZE,
        "fps": FPS,
        "video_crf": VIDEO_CRF,
        "state_recipe": "official ee_pos(3) + ee_ori(3) + gripper_states(2), float32",
        "action_recipe": "official actions, float32, bit-exact",
        "episode_order": "files sorted by name, demos by numeric index (official order)",
        "total_tasks": len(files),
        "total_episodes": len(episodes),
        "total_frames": int(sum(item["length"] for item in episodes)),
        "demos_per_task_expected": DEMOS_PER_TASK,
        "episodes": episodes,
    }
    (building_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    building_root.rename(final_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites", nargs="+", choices=sorted(SUITES), default=[TARGET_SUITE, SEEN_SUITE]
    )
    parser.add_argument("--source-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=CONVERTED_ROOT)
    args = parser.parse_args()

    results = {}
    for suite in args.suites:
        manifest = convert_suite(suite, args.source_root, args.output_root)
        results[suite] = {
            key: manifest[key]
            for key in ("repo_id", "total_tasks", "total_episodes", "total_frames")
        }
        summary_path = (
            experiment_root() / "artifacts" / f"conversion_{suite}.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, indent=2)
            + "\n"
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
