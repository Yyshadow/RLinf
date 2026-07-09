#!/usr/bin/env bash

set -euo pipefail

MODE="${RLINF_RUN_MODE:-train}"

CLOUD_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122"
ENV_PREFIX="${CLOUD_ROOT}/Miniforge/envs/rlinf_lwd"

source "${CLOUD_ROOT}/Miniforge/bin/activate" "${ENV_PREFIX}"

export REPO_PATH="${REPO_PATH:-${CLOUD_ROOT}/RLinf}"
export EMBODIED_PATH="${REPO_PATH}/examples/sft"
export RLINF_PI05_DATA_ROOT="${RLINF_PI05_DATA_ROOT:-${CLOUD_ROOT}/datasets/rl_data/robotwin_aloha_pi05_quick}"
export RLINF_PI05_MODEL_PATH="${RLINF_PI05_MODEL_PATH:-${CLOUD_ROOT}/weights/rlinf_pi05_pytorch/pi05_base_hammer50}"
export RLINF_PI05_NORM_STATS_PATH="${RLINF_PI05_NORM_STATS_PATH:-${CLOUD_ROOT}/datasets/rl_data/robotwin_aloha_lwd_split/pi05_norm_stats.json}"
export RLINF_PI05_LOG_ROOT="${RLINF_PI05_LOG_ROOT:-${CLOUD_ROOT}/checkpoints/rlinf_pi05_sft}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CLOUD_ROOT}/weights}"
export TORCH_HOME="${TORCH_HOME:-${CLOUD_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${CLOUD_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

case "${MODE}" in
    smoke)
        EXPERIMENT_NAME="pi05_hammer50_overfit50_allstats_smoke10"
        MAX_STEPS=10
        SAVE_INTERVAL=10
        MICRO_BATCH_SIZE=1
        GLOBAL_BATCH_SIZE=8
        AUTO_RESUME=0
        ;;
    train)
        EXPERIMENT_NAME="pi05_hammer50_overfit50_10k_allstats_v1"
        MAX_STEPS=10000
        SAVE_INTERVAL=500
        MICRO_BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=64
        AUTO_RESUME=1
        ;;
    *)
        echo "Usage: set RLINF_RUN_MODE to smoke or train" >&2
        exit 2
        ;;
esac

find_latest_checkpoint() {
    local ckpt_root="$1"
    local latest=""
    local latest_step=-1

    shopt -s nullglob
    for ckpt in "${ckpt_root}"/global_step_*; do
        [ -d "${ckpt}" ] || continue
        local step="${ckpt##*global_step_}"
        [[ "${step}" =~ ^[0-9]+$ ]] || continue
        [ -f "${ckpt}/actor/dcp_checkpoint/.metadata" ] || continue
        [ -f "${ckpt}/actor/model_state_dict/full_weights.pt" ] || continue
        if (( step > latest_step )); then
            latest_step="${step}"
            latest="${ckpt}"
        fi
    done
    shopt -u nullglob

    if [ -n "${latest}" ]; then
        echo "${latest}"
    fi
    return 0
}

require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

cd "${REPO_PATH}"

echo "Using python: $(command -v python)"
echo "Script mode: ${MODE}"
echo "SFT data: ${RLINF_PI05_DATA_ROOT}/beat_block_hammer_success_50_train"
echo "SFT norm stats: ${RLINF_PI05_NORM_STATS_PATH}"

ray stop -f || true

TOKENIZER_PATH="${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
require_path "${TOKENIZER_PATH}" "OpenPI tokenizer cache"
require_path "${RLINF_PI05_DATA_ROOT}/beat_block_hammer_success_50_train" "50-success SFT dataset"
require_path "${RLINF_PI05_NORM_STATS_PATH}" "shared pi0.5 norm stats"

python -c "import openpi, lerobot; from rlinf.models.embodiment.openpi import get_model; print('pi05 sft imports ok')"

HYDRA_ARGS=(
    "--config-name" "robotwin_sft_openpi_pi05_hammer50_allstats_cloud"
    "runner.max_steps=${MAX_STEPS}"
    "runner.save_interval=${SAVE_INTERVAL}"
    "runner.logger.experiment_name=${EXPERIMENT_NAME}"
    "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
    "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
)

if [ "${AUTO_RESUME}" = "1" ] && [ "${RLINF_FORCE_RESTART:-0}" != "1" ]; then
    CKPT_ROOT="${RLINF_PI05_LOG_ROOT}/${EXPERIMENT_NAME}/checkpoints"
    LATEST_CKPT="$(find_latest_checkpoint "${CKPT_ROOT}")"
    if [ -n "${LATEST_CKPT}" ]; then
        echo "Auto resume from ${LATEST_CKPT}"
        HYDRA_ARGS+=("runner.resume_dir=${LATEST_CKPT}")
    else
        echo "No complete checkpoint found under ${CKPT_ROOT}; start from scratch"
    fi
fi

python examples/sft/train_vla_sft.py "${HYDRA_ARGS[@]}"
