#!/bin/bash

# Run LWD critic training.
# Usage: bash examples/lwd/run_lwd_critic.sh [CONFIG_NAME] [EXTRA_ARGS...]

export SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SRC_FILE="${SCRIPT_DIR}/train_lwd_critic.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_PATH}:$PYTHONPATH"

source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="robotwin_lwd_critic"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true

LOG_DIR="${REPO_PATH}/logs/lwd_critic/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
LOG_FILE="${LOG_DIR}/run_lwd_critic.log"
mkdir -p "${LOG_DIR}"

HYDRA_ARGS=("runner.logger.log_path=${LOG_DIR}")
CMD_BASE="python ${SRC_FILE} --config-path ${SCRIPT_DIR}/config/ --config-name ${CONFIG_NAME}"
echo "${CMD_BASE} ${HYDRA_ARGS[*]} $*" > "${LOG_FILE}"
${CMD_BASE} "${HYDRA_ARGS[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
