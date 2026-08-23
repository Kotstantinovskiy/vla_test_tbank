from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .constants import ARTIFACTS_DIR, CONFIG_PATH, RESULTS_DIR
from .data import (
    VideoProgressDataset,
    build_validation_specs,
    load_episode_records,
    split_records_by_task,
)
from .prepare import prepare
from .utils import atomic_json, load_config, now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    prepare(args.config)
    data_config = config["data"]
    seed = int(config["experiment"]["seed"])
    records = load_episode_records(data_config["root"], data_config["camera"])
    train, validation, _ = split_records_by_task(
        records, float(data_config["validation_task_fraction"]), seed
    )
    train_dataset = VideoProgressDataset(
        train,
        fps=int(data_config["fps"]),
        num_bins=int(config["model"]["num_progress_bins"]),
        max_frames=int(config["model"]["max_frames"]),
        backend=data_config["video_backend"],
        seed=seed,
        samples_per_epoch=8,
    )
    validation_specs = build_validation_specs(
        validation, list(data_config["validation_bins"]), 2, seed
    )
    validation_dataset = VideoProgressDataset(
        validation,
        fps=int(data_config["fps"]),
        num_bins=int(config["model"]["num_progress_bins"]),
        max_frames=int(config["model"]["max_frames"]),
        backend=data_config["video_backend"],
        seed=seed,
        fixed_specs=validation_specs,
    )
    samples = [("train", train_dataset[0]), ("train", train_dataset[1])]
    samples.extend(("validation", validation_dataset[index]) for index in range(len(validation_dataset)))
    figure, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))
    for row_index, (split, sample) in enumerate(samples):
        for column_index, image in enumerate(sample["images"]):
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].axis("off")
            axes[row_index, column_index].set_title(
                f"{split} e{sample['episode_index']} f{sample['frame_indices'][column_index]}"
            )
        axes[row_index, 0].set_ylabel(
            f"bin={sample['target_bin']}\np={sample['target_progress']:.3f}", rotation=0, labelpad=40
        )
    figure.tight_layout()
    output = RESULTS_DIR / "media/data_smoke.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    result = {
        "created_at": now_iso(),
        "state": "passed",
        "samples": [
            {
                "split": split,
                "episode_index": sample["episode_index"],
                "task": sample["task"],
                "target_bin": sample["target_bin"],
                "target_progress": sample["target_progress"],
                "endpoint": sample["endpoint"],
                "frame_indices": sample["frame_indices"],
                "frame_size": list(sample["images"][0].size),
            }
            for split, sample in samples
        ],
        "contact_sheet": str(output),
        "split_manifest": str(ARTIFACTS_DIR / "dataset_split.json"),
    }
    atomic_json(ARTIFACTS_DIR / "data_smoke.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
