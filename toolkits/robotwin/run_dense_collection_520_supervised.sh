#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=/data/wam_codebase/RLinf
PYTHON=${REPO_ROOT}/.venv-openpi/bin/python
COLLECT=${REPO_ROOT}/toolkits/robotwin/collect_dense_lerobot_aloha.py
DATA_ROOT=${REPO_ROOT}/datasets/robotwin_aloha
LOG_ROOT=${REPO_ROOT}/logs/robotwin_collect_520_supervised
SCRATCH_ROOT=${REPO_ROOT}/datasets/_scratch_robotwin_collect_520
INCOMPLETE_ROOT=${REPO_ROOT}/datasets/_incomplete_robotwin_collect_520
TARGET_EPISODES=520
RUN_ID=$(date '+%Y%m%d_%H%M%S')
LOCK_FILE=${DATA_ROOT}/.dense_collection.lock

MAX_RETRIES=${MAX_RETRIES:-3}
TARGET_TIMEOUT=${TARGET_TIMEOUT:-18h}
GPU_START_MAX_MIB=${GPU_START_MAX_MIB:-6000}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-300}
CPU_LIST=${CPU_LIST:-8-31}
NICE_LEVEL=${NICE_LEVEL:-15}
IONICE_CLASS=${IONICE_CLASS:-3}
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
export LEROBOT_IMAGE_WRITER_THREADS=${LEROBOT_IMAGE_WRITER_THREADS:-2}

mkdir -p "${LOG_ROOT}" "${DATA_ROOT}" "${SCRATCH_ROOT}/${RUN_ID}" "${INCOMPLETE_ROOT}"
MASTER_LOG=${LOG_ROOT}/master_${RUN_ID}.log

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another RoboTwin dense collection is already running; lock=${LOCK_FILE}"
  exit 1
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
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

parquet_count() {
  local root=$1
  if [[ -d "${root}/data" ]]; then
    find "${root}/data" -path '*/episode_*.parquet' -type f | wc -l
  else
    echo 0
  fi
}

dataset_complete() {
  local root=$1
  [[ -d "${root}" ]] || return 1

  local meta_count
  local data_count
  meta_count=$(episode_count "${root}")
  data_count=$(parquet_count "${root}")

  (( meta_count >= TARGET_EPISODES && data_count >= TARGET_EPISODES ))
}

find_complete_root() {
  local final_root=$1
  local candidate
  for candidate in "${final_root}" "${final_root}_merged"; do
    if dataset_complete "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

wait_for_gpu_headroom() {
  local used_mib
  while command -v nvidia-smi >/dev/null 2>&1; do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
    if [[ -z "${used_mib}" || "${used_mib}" -le "${GPU_START_MAX_MIB}" ]]; then
      return 0
    fi
    log "GPU memory is ${used_mib}MiB; waiting for <= ${GPU_START_MAX_MIB}MiB before starting next dataset"
    sleep "${GPU_WAIT_SECONDS}"
  done
}

run_low_priority() {
  local prefix=()
  if command -v taskset >/dev/null 2>&1; then
    prefix+=(taskset -c "${CPU_LIST}")
  fi
  if command -v nice >/dev/null 2>&1; then
    prefix+=(nice -n "${NICE_LEVEL}")
  fi
  if command -v ionice >/dev/null 2>&1; then
    if [[ "${IONICE_CLASS}" == "3" ]]; then
      prefix+=(ionice -c 3)
    else
      prefix+=(ionice -c "${IONICE_CLASS}" -n "${IONICE_LEVEL}")
    fi
  fi
  if command -v timeout >/dev/null 2>&1; then
    prefix+=(timeout -k 2m "${TARGET_TIMEOUT}")
  fi
  "${prefix[@]}" "$@"
}

promote_output() {
  local tmp_output=$1
  local final_output=$2
  local final_name
  final_name=$(basename "${final_output}")

  if dataset_complete "${final_output}"; then
    log "[skip-promote] ${final_output} is already complete; keeping ${tmp_output}"
    return 0
  fi

  if [[ -e "${final_output}" ]]; then
    local archived="${INCOMPLETE_ROOT}/${final_name}_incomplete_${RUN_ID}"
    log "[archive] moving incomplete ${final_output} -> ${archived}"
    mv "${final_output}" "${archived}"
  fi

  mv "${tmp_output}" "${final_output}"
  log "[ready] ${final_output} ($(episode_count "${final_output}") metadata rows, $(parquet_count "${final_output}") parquet files)"
}

collect_with_retries() {
  local name=$1
  local final_output=$2
  shift 2

  local complete_root
  if complete_root=$(find_complete_root "${final_output}"); then
    log "[skip] ${name} already complete: ${complete_root}"
    return 0
  fi

  local final_name
  final_name=$(basename "${final_output}")
  local attempt
  for (( attempt = 1; attempt <= MAX_RETRIES; attempt++ )); do
    wait_for_gpu_headroom

    local attempt_dir="${SCRATCH_ROOT}/${RUN_ID}/${name}_try${attempt}"
    local tmp_output="${attempt_dir}/${final_name}"
    local step_log="${LOG_ROOT}/${name}_${RUN_ID}_try${attempt}.log"
    mkdir -p "${attempt_dir}"

    log "[start] ${name} try ${attempt}/${MAX_RETRIES}: ${tmp_output}"
    if run_low_priority "${PYTHON}" "${COLLECT}" "$@" --output "${tmp_output}" --num-episodes "${TARGET_EPISODES}" \
      2>&1 | tee -a "${step_log}"; then
      if dataset_complete "${tmp_output}"; then
        promote_output "${tmp_output}" "${final_output}"
        return 0
      fi
      log "[retry] ${name} try ${attempt} ended but output is incomplete: $(episode_count "${tmp_output}") metadata, $(parquet_count "${tmp_output}") parquet"
    else
      local rc=$?
      log "[retry] ${name} try ${attempt} failed with exit code ${rc}; partial output kept at ${attempt_dir}"
    fi
  done

  log "[failed] ${name} did not complete after ${MAX_RETRIES} attempt(s)"
  return 1
}

collect_success() {
  local task=$1
  local seed=$2
  local output="${DATA_ROOT}/${task}_success_${TARGET_EPISODES}ep_dense"

  collect_with_retries "${task}_success" "${output}" \
    --mode success \
    --task "${task}" \
    --task-config demo_30 \
    --max-attempts 6000 \
    --seed-start "${seed}" \
    --planner-backend mplib \
    --save-freq 15
}

collect_replay() {
  local task=$1
  local kind=$2
  local expert_root=$3
  local seed=$4
  local scale=$5
  local output="${DATA_ROOT}/${task}_${kind}_${TARGET_EPISODES}ep_dense"

  collect_with_retries "${task}_${kind}" "${output}" \
    --mode replay_failure \
    --failure-kind "${kind}" \
    --task "${task}" \
    --task-config demo_30 \
    --expert-root "${expert_root}" \
    --max-attempts 10000 \
    --seed-start "${seed}" \
    --planner-backend mplib \
    --perturb-scale "${scale}" \
    --perturb-modes joint_bias,smooth_noise,action_lag,early_release,gripper_delay
}

collect_task() {
  local task=$1
  local seed=$2
  local success_output="${DATA_ROOT}/${task}_success_${TARGET_EPISODES}ep_dense"
  local success_root
  local task_failed=0

  collect_success "${task}" "${seed}" || return 1
  success_root=$(find_complete_root "${success_output}") || {
    log "[failed] no complete success dataset available for ${task}; skipping replay datasets"
    return 1
  }

  collect_replay "${task}" failed "${success_root}" "$((seed + 100000))" 0.12 || task_failed=1
  collect_replay "${task}" nearmiss "${success_root}" "$((seed + 200000))" 0.075 || task_failed=1
  return "${task_failed}"
}

main() {
  log "supervised RoboTwin dense collection started: run_id=${RUN_ID}"
  log "limits: CPU_LIST=${CPU_LIST}, NICE_LEVEL=${NICE_LEVEL}, IONICE_CLASS=${IONICE_CLASS}, TARGET_TIMEOUT=${TARGET_TIMEOUT}, MAX_RETRIES=${MAX_RETRIES}"

  local failures=0
  collect_task beat_block_hammer 600000 || failures=$((failures + 1))
  collect_task place_container_plate 700000 || failures=$((failures + 1))
  collect_task place_empty_cup 800000 || failures=$((failures + 1))
  collect_task place_shoe 900000 || failures=$((failures + 1))

  if (( failures > 0 )); then
    log "supervised collection finished with ${failures} failed task group(s)"
    exit 1
  fi

  log "supervised collection finished successfully"
}

main "$@"
