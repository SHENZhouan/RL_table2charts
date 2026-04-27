#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_update_teacher_train_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_update_teacher_train_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_update_teacher_train_${RUN_ID}.results"
MARKER_FILE="${LOG_DIR}/plotly_update_teacher_train_${RUN_ID}.marker"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-5}"
RL_NPROCS="${RL_NPROCS:-1}"
MASTER_PORT="${MASTER_PORT:-29682}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="${CORPUS_PATH:-../Data/PlotlyTable2Charts}"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"
TEACHER_DATA_PATH="${TEACHER_DATA_PATH:-../Results/teacher_data/latest}"

UPDATE_TEACHER_EXACT="${UPDATE_TEACHER_EXACT:-0.95}"
UPDATE_TEACHER_DEFAULT="${UPDATE_TEACHER_DEFAULT:-0.05}"
UPDATE_TEACHER_SAME_TOKEN="${UPDATE_TEACHER_SAME_TOKEN:-0.10}"
UPDATE_TEACHER_FIELD="${UPDATE_TEACHER_FIELD:-0.15}"
UPDATE_TEACHER_SAME_FIELD_TYPE="${UPDATE_TEACHER_SAME_FIELD_TYPE:-0.35}"
UPDATE_TEACHER_POSITIVE_THRESHOLD="${UPDATE_TEACHER_POSITIVE_THRESHOLD:-0.5}"
TEACHER_TRAIN_STEPS_PER_EPOCH="${TEACHER_TRAIN_STEPS_PER_EPOCH:-1000}"

EPOCHS="${EPOCHS:-1}"
MEMORY_SAMPLE_SIZE="${MEMORY_SAMPLE_SIZE:-64}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${MARKER_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_update_teacher_train.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_update_teacher_train.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_update_teacher_train.latest.results"

if [[ "${TEACHER_DATA_PATH}" != /* ]]; then
  TEACHER_DATA_PATH="$(realpath -m "${CODE_DIR}/${TEACHER_DATA_PATH}")"
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() { echo "$1" | tee -a "${STATUS_FILE}"; }
write_result() { echo "$1" | tee -a "${RESULT_FILE}"; }
append_results() { echo "$1" | tee -a "${RESULTS_MD}"; }

on_error() {
  local exit_code=$?
  write_status "status=failed"
  write_status "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_status "failed_line=${BASH_LINENO[0]}"
  write_status "exit_code=${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

write_status "run_id=${RUN_ID}"
write_status "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "gpu_ids=${GPU_IDS}"
write_status "teacher_data_path=${TEACHER_DATA_PATH}"
write_status "teacher_train_steps_per_epoch=${TEACHER_TRAIN_STEPS_PER_EPOCH}"
write_status "status=update_teacher_train_running"

test -d "${TEACHER_DATA_PATH}"
test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

echo "[Teacher:train] train student from teacher supervision shards"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT}" \
  --module reinforce.update_teacher_learn_dist \
  --teacher_stage=train \
  --teacher_data_path="${TEACHER_DATA_PATH}" \
  --teacher_train_steps_per_epoch="${TEACHER_TRAIN_STEPS_PER_EPOCH}" \
  --corpus_path="${CORPUS_PATH}" \
  --model_size=small \
  --model_name=cp \
  --features=all-fast \
  --negative_weight=0.8 \
  --search_limits=e50-b4-na \
  --epochs="${EPOCHS}" \
  --model_save_path="${MODEL_SAVE_PATH}" \
  -p "${SFT_CKPT}" \
  --summary_path="${SUMMARY_PATH}" \
  --search_type="${CHART_TYPE}" \
  --input_type="${CHART_TYPE}" \
  --previous_type="${CHART_TYPE}" \
  --lang=en \
  --queue_mode=local \
  --log_freq_batch=100 \
  --memory_sample_size="${MEMORY_SAMPLE_SIZE}" \
  --update_teacher_exact="${UPDATE_TEACHER_EXACT}" \
  --update_teacher_default="${UPDATE_TEACHER_DEFAULT}" \
  --update_teacher_same_token="${UPDATE_TEACHER_SAME_TOKEN}" \
  --update_teacher_field="${UPDATE_TEACHER_FIELD}" \
  --update_teacher_same_field_type="${UPDATE_TEACHER_SAME_FIELD_TYPE}" \
  --update_teacher_positive_threshold="${UPDATE_TEACHER_POSITIVE_THRESHOLD}"

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
