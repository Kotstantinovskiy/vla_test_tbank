from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the locked language-control gate")
    parser.add_argument("--summary", type=Path, default=Path("results/summary/summary.json"))
    parser.add_argument(
        "--decision", type=Path, default=Path("results/summary/control_decision.json")
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    required = bool(summary["zero_shot_any_true_success"])
    decision = {
        "run_language_controls": required,
        "rule": "run iff any true-prompt zero-shot success is nonzero",
        "reason": (
            "At least one true-prompt zero-shot episode succeeded."
            if required
            else "All true-prompt zero-shot points stayed at the binary-success floor."
        ),
    }
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(json.dumps(decision, indent=2) + "\n")
    print("run" if required else "skip")


if __name__ == "__main__":
    main()
