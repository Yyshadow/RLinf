#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/data/wam_codebase/RLinf
PYTHON=${REPO_ROOT}/.venv-openpi/bin/python
COLLECT=${REPO_ROOT}/toolkits/robotwin/collect_dense_lerobot_aloha.py
DATA_ROOT=${REPO_ROOT}/datasets/robotwin_aloha
LOG_ROOT=${REPO_ROOT}/logs/robotwin_collect_520
TARGET_EPISODES=520

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export HF_HOME=${REPO_ROOT}/.cache/hf
export HF_DATASETS_CACHE=${REPO_ROOT}/.cache/hf_datasets
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "${LOG_ROOT}"

run_step() {
  local name=$1
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

collect_task() {
  local task=$1
  local seed=$2
  local success_dir="${DATA_ROOT}/${task}_success_${TARGET_EPISODES}ep_dense"
  local failed_dir="${DATA_ROOT}/${task}_failed_${TARGET_EPISODES}ep_dense"
  local nearmiss_dir="${DATA_ROOT}/${task}_nearmiss_${TARGET_EPISODES}ep_dense"

  run_step "${task}_success" \
    "${PYTHON}" "${COLLECT}" \
      --mode success \
      --task "${task}" \
      --task-config demo_30 \
      --output "${success_dir}" \
      --num-episodes "${TARGET_EPISODES}" \
      --max-attempts 6000 \
      --seed-start "${seed}" \
      --planner-backend mplib \
      --save-freq 15 \
      --overwrite

  run_step "${task}_failed" \
    "${PYTHON}" "${COLLECT}" \
      --mode replay_failure \
      --failure-kind failed \
      --task "${task}" \
      --task-config demo_30 \
      --expert-root "${success_dir}" \
      --output "${failed_dir}" \
      --num-episodes "${TARGET_EPISODES}" \
      --max-attempts 8000 \
      --seed-start "$((seed + 100000))" \
      --planner-backend mplib \
      --perturb-scale 0.12 \
      --perturb-modes joint_bias,smooth_noise,action_lag,early_release,gripper_delay \
      --overwrite

  run_step "${task}_nearmiss" \
    "${PYTHON}" "${COLLECT}" \
      --mode replay_failure \
      --failure-kind nearmiss \
      --task "${task}" \
      --task-config demo_30 \
      --expert-root "${success_dir}" \
      --output "${nearmiss_dir}" \
      --num-episodes "${TARGET_EPISODES}" \
      --max-attempts 10000 \
      --seed-start "$((seed + 200000))" \
      --planner-backend mplib \
      --perturb-scale 0.075 \
      --perturb-modes joint_bias,smooth_noise,action_lag,early_release,gripper_delay \
      --overwrite
}

collect_task beat_block_hammer 600000
collect_task place_container_plate 700000
collect_task place_empty_cup 800000
collect_task place_shoe 900000
