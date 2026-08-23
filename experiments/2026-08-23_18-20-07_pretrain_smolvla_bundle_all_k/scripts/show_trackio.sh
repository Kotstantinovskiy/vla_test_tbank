#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
exec trackio show --project "${TRACKIO_PROJECT:-pretrain-smolvla-bundle-all-k}" "$@"
