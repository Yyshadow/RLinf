#!/usr/bin/env bash

set -euo pipefail

MODE="${RLINF_RUN_MODE:-train}"

CLOUD_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122"
ENV_PREFIX="${CLOUD_ROOT}/Miniforge/envs/rlinf_lwd"

source "${CLOUD_ROOT}/Miniforge/bin/activate" "${ENV_PREFIX}"

export REPO_PATH="${REPO_PATH:-${CLOUD_ROOT}/RLinf}"
export RLINF_LWD_DATA_ROOT="${RLINF_LWD_DATA_ROOT:-${CLOUD_ROOT}/datasets/rl_data/robotwin_aloha_lwd_split}"
export RLINF_QAM_LOG_ROOT="${RLINF_QAM_LOG_ROOT:-${CLOUD_ROOT}/checkpoints/rlinf_lwd_qam}"
export RLINF_QAM_ACTOR_MODEL_PATH="${RLINF_QAM_ACTOR_MODEL_PATH:-${CLOUD_ROOT}/checkpoints/rlinf_pi05_sft_10000/pi05_hammer50_overfit50_10k_v1/checkpoints/global_step_10000}"
export RLINF_QAM_REFERENCE_MODEL_PATH="${RLINF_QAM_REFERENCE_MODEL_PATH:-${RLINF_QAM_ACTOR_MODEL_PATH}}"
export RLINF_QAM_CRITIC_MODEL_PATH="${RLINF_QAM_CRITIC_MODEL_PATH:-${CLOUD_ROOT}/checkpoints/rlinf_lwd/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000/actor}"
export RLINF_PI05_NORM_STATS_PATH="${RLINF_PI05_NORM_STATS_PATH:-${CLOUD_ROOT}/datasets/rl_data/robotwin_aloha_pi05_quick/beat_block_hammer_success_50_train/norm_stats.json}"
export RLINF_SIGLIP_PATH="${RLINF_SIGLIP_PATH:-${CLOUD_ROOT}/weights/pretrained/siglip2-so400m-patch14-224}"
export RLINF_GEMMA3_PATH="${RLINF_GEMMA3_PATH:-${CLOUD_ROOT}/weights/pretrained/gemma-3-270m}"
export RLINF_TOKENIZER_PATH="${RLINF_TOKENIZER_PATH:-${RLINF_GEMMA3_PATH}}"
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
        CONFIG_NAME="robotwin_beat_block_hammer_lwd_qam_openpi_pi05_smoke"
        EXPERIMENT_NAME="robotwin_beat_block_hammer_lwd_qam_openpi_pi05_strict_smoke"
        MAX_STEPS=10
        SAVE_INTERVAL=10
        AUTO_RESUME=0
        ;;
    train)
        CONFIG_NAME="robotwin_beat_block_hammer_lwd_qam_openpi_pi05"
        EXPERIMENT_NAME="robotwin_beat_block_hammer_lwd_qam_openpi_pi05_strict"
        MAX_STEPS=500
        SAVE_INTERVAL=100
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
echo "QAM actor: ${RLINF_QAM_ACTOR_MODEL_PATH}"
echo "QAM reference: ${RLINF_QAM_REFERENCE_MODEL_PATH}"
echo "QAM critic: ${RLINF_QAM_CRITIC_MODEL_PATH}"
echo "QAM data root: ${RLINF_LWD_DATA_ROOT}"

ray stop -f || true

TOKENIZER_PATH="${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
require_path "${TOKENIZER_PATH}" "OpenPI tokenizer cache"
require_path "${RLINF_QAM_ACTOR_MODEL_PATH}/actor/model_state_dict/full_weights.pt" "QAM actor full weights"
require_path "${RLINF_QAM_REFERENCE_MODEL_PATH}/actor/model_state_dict/full_weights.pt" "QAM reference full weights"
require_path "${RLINF_QAM_CRITIC_MODEL_PATH}/model_state_dict/full_weights.pt" "QAM critic full weights"
require_path "${RLINF_LWD_DATA_ROOT}/pi05_norm_stats.json" "LWD QAM norm stats"
require_path "${RLINF_PI05_NORM_STATS_PATH}" "OpenPI norm stats"

python -c "from rlinf.algorithms.lwd import qam_vector_field_loss; from rlinf.workers.sft.fsdp_lwd_qam_worker import FSDPLWDQAMWorker; print('lwd qam imports ok')"

HYDRA_ARGS=(
    "--config-name" "${CONFIG_NAME}"
    "runner.max_steps=${MAX_STEPS}"
    "runner.save_interval=${SAVE_INTERVAL}"
    "runner.logger.experiment_name=${EXPERIMENT_NAME}"
)

if [ "${AUTO_RESUME}" = "1" ] && [ "${RLINF_FORCE_RESTART:-0}" != "1" ]; then
    CKPT_ROOT="${RLINF_QAM_LOG_ROOT}/${EXPERIMENT_NAME}/checkpoints"
    LATEST_CKPT="$(find_latest_checkpoint "${CKPT_ROOT}")"
    if [ -n "${LATEST_CKPT}" ]; then
        echo "Auto resume from ${LATEST_CKPT}"
        HYDRA_ARGS+=("runner.resume_dir=${LATEST_CKPT}")
    else
        echo "No complete checkpoint found under ${CKPT_ROOT}; start from scratch"
    fi
fi

python examples/lwd/train_lwd_qam.py "${HYDRA_ARGS[@]}"
