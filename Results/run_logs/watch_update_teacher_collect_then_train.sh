#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
LOG_DIR="${ROOT}/Results/run_logs"
COLLECT_STATUS="${LOG_DIR}/plotly_update_teacher_collect.latest.status"
WATCH_LOG="${LOG_DIR}/plotly_update_teacher_pipeline_watcher.log"
TRAIN_SESSION="${TRAIN_SESSION:-update_teacher_train_gpu3}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-3}"
TRAIN_RL_NPROCS="${TRAIN_RL_NPROCS:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"

touch "${WATCH_LOG}"
exec >>"${WATCH_LOG}" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher started"

while true; do
  if [[ -f "${COLLECT_STATUS}" ]]; then
    if grep -q '^status=finished$' "${COLLECT_STATUS}"; then
      TEACHER_DATA_PATH="$(grep '^teacher_data_path=' "${COLLECT_STATUS}" | tail -1 | cut -d= -f2-)"
      if [[ -z "${TEACHER_DATA_PATH}" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] collect finished but teacher_data_path missing"
        exit 1
      fi
      if [[ "${TEACHER_DATA_PATH}" != /* ]]; then
        TEACHER_DATA_PATH="$(realpath -m "${ROOT}/Table2Charts/${TEACHER_DATA_PATH}")"
      fi

      if tmux ls 2>/dev/null | grep -q "^${TRAIN_SESSION}:"; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train session ${TRAIN_SESSION} already exists, exiting watcher"
        exit 0
      fi

      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] collect finished, launching train with data path ${TEACHER_DATA_PATH}"
      tmux new-session -d -s "${TRAIN_SESSION}" /bin/bash -lc \
        "cd ${ROOT} && GPU_IDS=${TRAIN_GPU_IDS} RL_NPROCS=${TRAIN_RL_NPROCS} TEACHER_DATA_PATH=${TEACHER_DATA_PATH} TEACHER_TRAIN_STEPS_PER_EPOCH=${TRAIN_STEPS} EPOCHS=${TRAIN_EPOCHS} MEMORY_SAMPLE_SIZE=${TRAIN_BATCH_SIZE} ${ROOT}/Results/run_logs/run_plotly_update_teacher_train.sh"
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train launched in tmux session ${TRAIN_SESSION}"
      exit 0
    fi

    if grep -q '^status=failed$' "${COLLECT_STATUS}"; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] collect failed, watcher exiting"
      exit 1
    fi
  fi

  sleep 30
done
