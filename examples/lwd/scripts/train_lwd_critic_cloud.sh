#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-train}"

CLOUD_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122"
CONDA_ROOT="${RLINF_CONDA_ROOT:-${CLOUD_ROOT}/Miniforge}"
CONDA_ENV="${RLINF_CONDA_ENV:-rlinf_lwd}"

source "${CONDA_ROOT}/bin/activate"
conda activate "${CONDA_ENV}"

export REPO_PATH="${REPO_PATH:-${CLOUD_ROOT}/RLinf}"
export RLINF_LWD_DATA_ROOT="${RLINF_LWD_DATA_ROOT:-${CLOUD_ROOT}/datasets/rl_data/robotwin_aloha_lwd_split}"
export RLINF_LWD_LOG_ROOT="${RLINF_LWD_LOG_ROOT:-${CLOUD_ROOT}/checkpoints/rlinf_lwd}"
export RLINF_SIGLIP_PATH="${RLINF_SIGLIP_PATH:-${CLOUD_ROOT}/weights/pretrained/siglip2-so400m-patch14-224}"
export RLINF_GEMMA3_PATH="${RLINF_GEMMA3_PATH:-${CLOUD_ROOT}/weights/pretrained/gemma-3-270m}"
export RLINF_TOKENIZER_PATH="${RLINF_TOKENIZER_PATH:-${RLINF_GEMMA3_PATH}}"
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
        CONFIG_NAME="robotwin_lwd_critic_cloud_beat_block_smoke"
        EXPERIMENT_NAME="robotwin_lwd_critic_smoke_8a100"
        MAX_STEPS=20
        SAVE_INTERVAL=20
        AUTO_RESUME=0
        ;;
    train)
        CONFIG_NAME="robotwin_lwd_critic_cloud_beat_block"
        EXPERIMENT_NAME="robotwin_lwd_critic_train_8a100"
        MAX_STEPS=8000
        SAVE_INTERVAL=1000
        AUTO_RESUME=1
        ;;
    *)
        echo "Usage: $0 [smoke|train]" >&2
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
        [ -f "${ckpt}/actor/target_model.pt" ] || continue
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

cd "${REPO_PATH}"

ray stop -f || true

python -c "from transformers import AutoTokenizer; from rlinf.data.datasets.lwd.chunk_dataset import LWDChunkDataset; from rlinf.models.embodiment.lwd_critic.lwd_critic_model import LWDCriticModel; print('lwd imports ok')"

HYDRA_ARGS=(
    "--config-name" "${CONFIG_NAME}"
    "runner.max_steps=${MAX_STEPS}"
    "runner.save_interval=${SAVE_INTERVAL}"
    "runner.logger.experiment_name=${EXPERIMENT_NAME}"
)

if [ "${AUTO_RESUME}" = "1" ] && [ "${RLINF_FORCE_RESTART:-0}" != "1" ]; then
    CKPT_ROOT="${RLINF_LWD_LOG_ROOT}/${EXPERIMENT_NAME}/checkpoints"
    LATEST_CKPT="$(find_latest_checkpoint "${CKPT_ROOT}")"
    if [ -n "${LATEST_CKPT}" ]; then
        echo "Auto resume from ${LATEST_CKPT}"
        HYDRA_ARGS+=("runner.resume_dir=${LATEST_CKPT}")
    else
        echo "No complete checkpoint found under ${CKPT_ROOT}; start from scratch"
    fi
fi

python examples/lwd/train_lwd_critic.py "${HYDRA_ARGS[@]}"
