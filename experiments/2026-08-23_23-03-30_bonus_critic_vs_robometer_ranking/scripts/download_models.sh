#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --exclude 'model*.safetensors' \
  --exclude model.safetensors.index.json \
  --cache-dir /var/tmp/vla_hf/hub
hf download aliangdw/Robometer-4B-LIBERO \
  --revision 637fa8ecb7fb872cb5783c19d0825a08dc20fc8c \
  --cache-dir /var/tmp/vla_hf/hub
