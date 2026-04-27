#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_update_teacher_collect_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_update_teacher_collect_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_update_teacher_collect_${RUN_ID}.results"
MARKER_FILE="${LOG_DIR}/plotly_update_teacher_collect_${RUN_ID}.marker"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-5}"
RL_NPROCS="${RL_NPROCS:-1}"
MASTER_PORT="${MASTER_PORT:-29681}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="${CORPUS_PATH:-../Data/PlotlyTable2Charts}"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"
TEACHER_DATA_PATH="${TEACHER_DATA_PATH:-../Results/teacher_data/plotly_teacher_collect_${RUN_ID}}"

UPDATE_TEACHER_EXACT="${UPDATE_TEACHER_EXACT:-0.95}"
UPDATE_TEACHER_DEFAULT="${UPDATE_TEACHER_DEFAULT:-0.05}"
UPDATE_TEACHER_SAME_TOKEN="${UPDATE_TEACHER_SAME_TOKEN:-0.10}"
UPDATE_TEACHER_FIELD="${UPDATE_TEACHER_FIELD:-0.15}"
UPDATE_TEACHER_SAME_FIELD_TYPE="${UPDATE_TEACHER_SAME_FIELD_TYPE:-0.35}"
UPDATE_TEACHER_POSITIVE_THRESHOLD="${UPDATE_TEACHER_POSITIVE_THRESHOLD:-0.5}"
TEACHER_POLICY_EPSILON_START="${TEACHER_POLICY_EPSILON_START:-0.2}"
TEACHER_POLICY_EPSILON_END="${TEACHER_POLICY_EPSILON_END:-0.02}"
TEACHER_POLICY_EPSILON_DECAY="${TEACHER_POLICY_EPSILON_DECAY:-0.8}"
TEACHER_POLICY_EXPLORE_TOP_M="${TEACHER_POLICY_EXPLORE_TOP_M:-5}"
TEACHER_POLICY_SEED="${TEACHER_POLICY_SEED:-20260427}"
TEACHER_COLLECT_RATIO="${TEACHER_COLLECT_RATIO:-0.1}"

EPOCHS="${EPOCHS:-1}"
MAX_TABLES="${MAX_TABLES:-64}"
MIN_MEMORY="${MIN_MEMORY:-1000}"
MEMORY_SAMPLE_SIZE="${MEMORY_SAMPLE_SIZE:-64}"
MEMORY_SAMPLE_ROUNDS="${MEMORY_SAMPLE_ROUNDS:-2}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${MARKER_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_update_teacher_collect.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_update_teacher_collect.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_update_teacher_collect.latest.results"

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
  append_results ""
  append_results "### Plotly Update Teacher Collect failed"
  append_results ""
  append_results "- failed_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  append_results "- exit_code: ${exit_code}"
  append_results "- log_file: ${LOG_FILE}"
  exit "${exit_code}"
}
trap on_error ERR

write_status "run_id=${RUN_ID}"
write_status "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "gpu_ids=${GPU_IDS}"
write_status "rl_nprocs=${RL_NPROCS}"
write_status "master_port=${MASTER_PORT}"
write_status "sft_ckpt=${SFT_CKPT}"
write_status "teacher_data_path=${TEACHER_DATA_PATH}"
write_status "teacher_collect_ratio=${TEACHER_COLLECT_RATIO}"
write_status "epochs=${EPOCHS}"
write_status "log_file=${LOG_FILE}"
write_status "status=update_teacher_collect_running"

write_result "run_id=${RUN_ID}"
write_result "teacher_data_path=${TEACHER_DATA_PATH}"
write_result "teacher_collect_ratio=${TEACHER_COLLECT_RATIO}"
write_result "gpu_ids=${GPU_IDS}"

append_results ""
append_results "## Plotly Update Teacher Collect ${RUN_ID}"
append_results ""
append_results "- started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- gpu_ids: ${GPU_IDS}"
append_results "- teacher_data_path: ${TEACHER_DATA_PATH}"
append_results "- teacher_collect_ratio: ${TEACHER_COLLECT_RATIO}"
append_results "- log_file: ${LOG_FILE}"

test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

echo "[Teacher:collect] search on subset and save teacher supervision shards"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT}" \
  --module reinforce.update_teacher_learn_dist \
  --teacher_stage=collect \
  --teacher_data_path="${TEACHER_DATA_PATH}" \
  --teacher_collect_ratio="${TEACHER_COLLECT_RATIO}" \
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
  --log_freq_agent=500 \
  --log_freq_batch=100 \
  --max_tables="${MAX_TABLES}" \
  --min_memory="${MIN_MEMORY}" \
  --memory_sample_size="${MEMORY_SAMPLE_SIZE}" \
  --memory_sample_rounds="${MEMORY_SAMPLE_ROUNDS}" \
  --update_teacher_exact="${UPDATE_TEACHER_EXACT}" \
  --update_teacher_default="${UPDATE_TEACHER_DEFAULT}" \
  --update_teacher_same_token="${UPDATE_TEACHER_SAME_TOKEN}" \
  --update_teacher_field="${UPDATE_TEACHER_FIELD}" \
  --update_teacher_same_field_type="${UPDATE_TEACHER_SAME_FIELD_TYPE}" \
  --update_teacher_positive_threshold="${UPDATE_TEACHER_POSITIVE_THRESHOLD}" \
  --teacher_policy_epsilon_start="${TEACHER_POLICY_EPSILON_START}" \
  --teacher_policy_epsilon_end="${TEACHER_POLICY_EPSILON_END}" \
  --teacher_policy_epsilon_decay="${TEACHER_POLICY_EPSILON_DECAY}" \
  --teacher_policy_explore_top_m="${TEACHER_POLICY_EXPLORE_TOP_M}" \
  --teacher_policy_seed="${TEACHER_POLICY_SEED}"

write_status "teacher_data_path=${TEACHER_DATA_PATH}"
write_result "teacher_data_path=${TEACHER_DATA_PATH}"
test -d "${TEACHER_DATA_PATH}"

append_results ""
append_results "### Update Teacher Collect"
append_results ""
append_results "- finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- teacher_data_path: ${TEACHER_DATA_PATH}"

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
