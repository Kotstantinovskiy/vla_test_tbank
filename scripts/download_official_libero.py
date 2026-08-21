#!/usr/bin/env python3
"""Download revision-pinned official LIBERO HDF5 demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile, snapshot_download


OFFICIAL_REPO = "yifengzhu-hf/LIBERO-datasets"
OFFICIAL_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
DEFAULT_ROOT = Path("/var/tmp/libero_official_f13aa24")
SUITE_FILE_COUNTS = {"libero_90": 90, "libero_goal": 10}
ALL_SUITES = ("libero_goal", "libero_90")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def selected_suites(value: str) -> tuple[str, ...]:
    return ALL_SUITES if value == "all" else (value,)


def remote_manifest(suites: tuple[str, ...]) -> dict[str, Any]:
    api = HfApi()
    suite_manifests: dict[str, Any] = {}
    for suite in suites:
        files = [
            item
            for item in api.list_repo_tree(
                OFFICIAL_REPO,
                path_in_repo=suite,
                recursive=True,
                expand=False,
                revision=OFFICIAL_REVISION,
                repo_type="dataset",
            )
            if isinstance(item, RepoFile) and item.path.endswith(".hdf5")
        ]
        expected_count = SUITE_FILE_COUNTS[suite]
        if len(files) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} official {suite} files, got {len(files)}"
            )
        suite_manifests[suite] = {
            "file_count": len(files),
            "total_bytes": sum(item.size for item in files),
            "files": [
                {
                    "path": item.path,
                    "size": item.size,
                    "lfs_sha256": item.lfs.sha256 if item.lfs else None,
                    "blob_id": item.blob_id,
                }
                for item in sorted(files, key=lambda item: item.path)
            ],
        }
    return {
        "repo_id": OFFICIAL_REPO,
        "revision": OFFICIAL_REVISION,
        "captured_at": utc_now(),
        "suites": suite_manifests,
    }


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for suite in manifest["suites"].values():
        for expected in suite["files"]:
            path = root / expected["path"]
            actual_size = path.stat().st_size if path.is_file() else None
            actual_hash = sha256(path) if actual_size == expected["size"] else None
            checks.append(
                {
                    **expected,
                    "exists": path.is_file(),
                    "actual_size": actual_size,
                    "actual_sha256": actual_hash,
                    "size_match": actual_size == expected["size"],
                    "sha256_match": actual_hash == expected["lfs_sha256"],
                }
            )
    return {
        "verified_at": utc_now(),
        "root": str(root),
        "files": checks,
        "complete": bool(checks)
        and all(item["size_match"] and item["sha256_match"] for item in checks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official raw LIBERO HDF5 data from a pinned Hugging Face "
            "revision. Existing partial downloads are resumed."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("all", "libero_90", "libero_goal"),
        default="all",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--verification-out", type=Path)
    parser.add_argument("--status-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = selected_suites(args.suite)
    label = args.suite
    root = args.root.expanduser().resolve()
    manifest_path = args.manifest_out or root / f"official_source_manifest_{label}.json"
    verification_path = (
        args.verification_out or root / f"download_verification_{label}.json"
    )
    status_path = args.status_out or root / f"download_status_{label}.json"

    root.mkdir(parents=True, exist_ok=True)
    manifest = remote_manifest(suites)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        status_path,
        {
            "stage": "download",
            "state": "running",
            "started_at": utc_now(),
            "repo_id": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "suites": list(suites),
            "root": str(root),
        },
    )

    try:
        resolved = str(root)
        for suite in suites:
            write_json_atomic(
                status_path,
                {
                    "stage": "download",
                    "state": "running",
                    "active_suite": suite,
                    "repo_id": OFFICIAL_REPO,
                    "revision": OFFICIAL_REVISION,
                    "suites": list(suites),
                    "root": str(root),
                },
            )
            resolved = snapshot_download(
                repo_id=OFFICIAL_REPO,
                repo_type="dataset",
                revision=OFFICIAL_REVISION,
                local_dir=root,
                allow_patterns=f"{suite}/*.hdf5",
                max_workers=args.max_workers,
            )

        verification: dict[str, Any]
        if args.skip_hash:
            verification = {
                "complete": None,
                "skipped": True,
                "root": str(root),
                "suites": list(suites),
            }
        else:
            verification = verify(root, manifest)
        write_json_atomic(verification_path, verification)
        if verification.get("complete") is False:
            raise RuntimeError("Official download failed SHA-256 verification")

        write_json_atomic(
            status_path,
            {
                "stage": "download",
                "state": "completed",
                "finished_at": utc_now(),
                "repo_id": OFFICIAL_REPO,
                "revision": OFFICIAL_REVISION,
                "suites": list(suites),
                "resolved_root": resolved,
                "verified": verification.get("complete"),
            },
        )
        print(
            json.dumps(
                {
                    "root": resolved,
                    "suites": list(suites),
                    "verified": verification.get("complete"),
                    "manifest": str(manifest_path),
                    "verification": str(verification_path),
                },
                indent=2,
            )
        )
    except BaseException as error:
        write_json_atomic(
            status_path,
            {
                "stage": "download",
                "state": "failed",
                "finished_at": utc_now(),
                "repo_id": OFFICIAL_REPO,
                "revision": OFFICIAL_REVISION,
                "suites": list(suites),
                "root": str(root),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
