#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/data/wam_codebase/RLinf
LOG_ROOT=${REPO_ROOT}/logs/robotwin_collect_520_resume
RUN_SCRIPT=${REPO_ROOT}/toolkits/robotwin/run_dense_collection_520_resume.sh

mkdir -p "${LOG_ROOT}"
nohup /bin/bash "${RUN_SCRIPT}" > "${LOG_ROOT}/master.log" 2>&1 < /dev/null &
echo $! > "${LOG_ROOT}/pid.txt"
echo "started pid=$(cat "${LOG_ROOT}/pid.txt")"
