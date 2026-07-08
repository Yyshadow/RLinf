#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RLINF_RUN_MODE=smoke
bash "${SCRIPT_DIR}/train_lwd_qam_cloud.sh"
