#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$EXPERIMENT_ROOT"
bonus-ranking-run
