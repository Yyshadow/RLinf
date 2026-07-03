#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/data/wam_codebase/RLinf
PYTHON=${REPO_ROOT}/.venv-openpi/bin/python
COLLECT=${REPO_ROOT}/toolkits/robotwin/collect_dense_lerobot_aloha.py
DATA_ROOT=${REPO_ROOT}/datasets/robotwin_aloha
LOG_ROOT=${REPO_ROOT}/logs/robotwin_collect_520_resume
TARGET_EPISODES=520
RUN_ID=$(date '+%Y%m%d_%H%M%S')
LOCK_FILE=${DATA_ROOT}/.dense_collection.lock
NICE_LEVEL=${NICE_LEVEL:-10}
IONICE_CLASS=${IONICE_CLASS:-2}
IONICE_LEVEL=${IONICE_LEVEL:-7}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib
export HF_HOME=${REPO_ROOT}/.cache/hf
export HF_DATASETS_CACHE=${REPO_ROOT}/.cache/hf_datasets
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

mkdir -p "${LOG_ROOT}" "${DATA_ROOT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another RobotWin dense collection is already running; lock=${LOCK_FILE}"
  exit 1
fi

run_low_priority() {
  local prefix=()
  if command -v nice >/dev/null 2>&1; then
    prefix+=(nice -n "${NICE_LEVEL}")
  fi
  if command -v ionice >/dev/null 2>&1; then
    prefix+=(ionice -c "${IONICE_CLASS}" -n "${IONICE_LEVEL}")
  fi
  "${prefix[@]}" "$@"
}

episode_count() {
  local root=$1
  local meta="${root}/meta/robotwin_episode_metadata.jsonl"
  if [[ -f "${meta}" ]]; then
    wc -l < "${meta}"
  else
    echo 0
  fi
}

run_step() {
  local name=$1
  shift
  echo "[$(date '+%F %T')] START ${name}"
  run_low_priority "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

collect_success() {
  local task=$1
  local seed=$2
  local output="${DATA_ROOT}/${task}_success_${TARGET_EPISODES}ep_dense"
  local done_count
  done_count=$(episode_count "${output}")

  if (( done_count >= TARGET_EPISODES )); then
    echo "[skip] ${task}_success already has ${done_count}/${TARGET_EPISODES}: ${output}"
    SUCCESS_ROOT="${output}"
    return
  fi

  if [[ -d "${output}" ]]; then
    output="${output}_retry_${RUN_ID}"
  fi

  run_step "${task}_success" \
    "${PYTHON}" "${COLLECT}" \
      --mode success \
      --task "${task}" \
      --task-config demo_30 \
      --output "${output}" \
      --num-episodes "${TARGET_EPISODES}" \
      --max-attempts 6000 \
      --seed-start "${seed}" \
      --planner-backend mplib \
      --save-freq 15

  SUCCESS_ROOT="${output}"
}

collect_replay() {
  local task=$1
  local kind=$2
  local expert_root=$3
  local seed=$4
  local scale=$5
  local episodes=$6
  local output=$7
  local done_count
  done_count=$(episode_count "${output}")

  if (( done_count >= episodes )); then
    echo "[skip] ${task}_${kind} already has ${done_count}/${episodes}: ${output}"
    return
  fi

  if [[ -d "${output}" ]]; then
    output="${output}_retry_${RUN_ID}"
  fi

  run_step "${task}_${kind}" \
    "${PYTHON}" "${COLLECT}" \
      --mode replay_failure \
      --failure-kind "${kind}" \
      --task "${task}" \
      --task-config demo_30 \
      --expert-root "${expert_root}" \
      --output "${output}" \
      --num-episodes "${episodes}" \
      --max-attempts 10000 \
      --seed-start "${seed}" \
      --planner-backend mplib \
      --perturb-scale "${scale}" \
      --perturb-modes joint_bias,smooth_noise,action_lag,early_release,gripper_delay
}

complete_existing_beat_block_hammer_failed() {
  local task=beat_block_hammer
  local success_root="${DATA_ROOT}/${task}_success_${TARGET_EPISODES}ep_dense"
  local failed_root="${DATA_ROOT}/${task}_failed_${TARGET_EPISODES}ep_dense"
  local merged_failed_root="${failed_root}_merged"
  local done_count
  done_count=$(episode_count "${merged_failed_root}")

  if (( done_count >= TARGET_EPISODES )); then
    echo "[skip] ${task}_failed already has ${done_count}/${TARGET_EPISODES}: ${merged_failed_root}"
    return
  fi

  done_count=$(episode_count "${failed_root}")

  if (( done_count >= TARGET_EPISODES )); then
    echo "[skip] ${task}_failed already has ${done_count}/${TARGET_EPISODES}: ${failed_root}"
    return
  fi

  local remaining=$((TARGET_EPISODES - done_count))
  local output="${DATA_ROOT}/${task}_failed_${TARGET_EPISODES}ep_dense_part2_from${done_count}"
  collect_replay "${task}" failed "${success_root}" 760000 0.12 "${remaining}" "${output}"
}

collect_task() {
  local task=$1
  local seed=$2

  collect_success "${task}" "${seed}"
  collect_replay "${task}" failed "${SUCCESS_ROOT}" "$((seed + 100000))" 0.12 "${TARGET_EPISODES}" \
    "${DATA_ROOT}/${task}_failed_${TARGET_EPISODES}ep_dense"
  collect_replay "${task}" nearmiss "${SUCCESS_ROOT}" "$((seed + 200000))" 0.075 "${TARGET_EPISODES}" \
    "${DATA_ROOT}/${task}_nearmiss_${TARGET_EPISODES}ep_dense"
}

complete_existing_beat_block_hammer_failed
collect_replay beat_block_hammer nearmiss "${DATA_ROOT}/beat_block_hammer_success_${TARGET_EPISODES}ep_dense" 800000 0.075 "${TARGET_EPISODES}" \
  "${DATA_ROOT}/beat_block_hammer_nearmiss_${TARGET_EPISODES}ep_dense"

collect_task place_container_plate 700000
collect_task place_empty_cup 800000
collect_task place_shoe 900000
