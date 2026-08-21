#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
exec trackio show --host "${TRACKIO_HOST:-0.0.0.0}" --project "$TRACKIO_PROJECT" "$@"
