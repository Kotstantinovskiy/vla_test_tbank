#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

while :; do
  state="$($VLA_PYTHON -c 'import json; print(json.load(open("results/status.json"))["state"])')"
  case "$state" in
    completed)
      scripts/summarize.sh
      "$VLA_REPO_ROOT/scripts/index_trackio.sh"
      exit 0
      ;;
    failed)
      echo "Training failed; final summary was not generated." >&2
      exit 1
      ;;
    starting|running)
      sleep 30
      ;;
    *)
      echo "Unexpected training state: $state" >&2
      exit 1
      ;;
  esac
done
