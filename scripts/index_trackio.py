from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = Path(
    os.environ.get("TRACKIO_DASHBOARD_DIR", REPO_ROOT / ".trackio-dashboard")
).resolve()


def hardlink_tree(source: Path, destination: Path) -> int:
    if destination.is_symlink():
        destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if destination_file.exists() or destination_file.is_symlink():
            try:
                if source_file.samefile(destination_file):
                    count += 1
                    continue
            except FileNotFoundError:
                pass
            destination_file.unlink()
        os.link(source_file, destination_file)
        count += 1
    return count


def snapshot_database(source: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _video_dicts(value: Any):
    if isinstance(value, dict):
        if value.get("_type") == "trackio.video":
            yield value
        for nested in value.values():
            yield from _video_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _video_dicts(nested)


def gif_to_mp4(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def make_videos_browser_compatible(database: Path, media_root: Path) -> int:
    connection = sqlite3.connect(database)
    converted = 0
    try:
        rows = connection.execute("SELECT id, metrics FROM metrics").fetchall()
        for row_id, metrics_blob in rows:
            metrics = json.loads(metrics_blob)
            changed = False
            for video in _video_dicts(metrics):
                relative = Path(video.get("file_path", ""))
                if relative.suffix.lower() != ".gif":
                    continue
                source = media_root / relative
                if not source.is_file():
                    continue
                destination_relative = relative.with_suffix(".mp4")
                destination = media_root / destination_relative
                if (
                    not destination.is_file()
                    or destination.stat().st_mtime < source.stat().st_mtime
                ):
                    gif_to_mp4(source, destination)
                video["file_path"] = destination_relative.as_posix()
                changed = True
                converted += 1
            if changed:
                encoded = json.dumps(metrics, separators=(",", ":")).encode()
                connection.execute(
                    "UPDATE metrics SET metrics = ? WHERE id = ?", (encoded, row_id)
                )
        connection.commit()
    finally:
        connection.close()
    return converted


def main() -> None:
    databases = sorted(REPO_ROOT.glob("experiments/*/artifacts/trackio/*.db"))
    if not databases:
        raise SystemExit("No experiment-local Trackio databases found.")

    DASHBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    media_root = DASHBOARD_ROOT / "media"
    media_root.mkdir(exist_ok=True)
    projects: list[str] = []
    media_files = 0
    converted_videos = 0
    seen_projects: set[str] = set()

    for source_database in databases:
        project = source_database.stem
        if project in seen_projects:
            raise RuntimeError(
                f"Duplicate Trackio project {project!r}; project names must be unique"
            )
        seen_projects.add(project)
        projects.append(project)

        destination_database = DASHBOARD_ROOT / source_database.name
        snapshot_database(source_database, destination_database)
        destination_lock = DASHBOARD_ROOT / f"{project}.lock"
        if destination_lock.is_symlink():
            destination_lock.unlink()
        destination_lock.touch(exist_ok=True)

        source_media = source_database.parent / "media" / project
        if source_media.is_dir():
            media_files += hardlink_tree(source_media, media_root / project)
        converted_videos += make_videos_browser_compatible(
            destination_database, media_root
        )

    print(
        json.dumps(
            {
                "projects": projects,
                "dashboard_dir": str(DASHBOARD_ROOT),
                "hardlinked_media_files": media_files,
                "gif_videos_converted_to_mp4": converted_videos,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
