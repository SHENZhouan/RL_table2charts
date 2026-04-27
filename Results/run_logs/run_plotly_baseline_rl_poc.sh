#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_baseline_rl_poc_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_baseline_rl_poc_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_baseline_rl_poc_${RUN_ID}.results"
MARKER_FILE="${LOG_DIR}/plotly_baseline_rl_poc_${RUN_ID}.marker"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-4}"
RL_NPROCS="${RL_NPROCS:-1}"
MASTER_PORT="${MASTER_PORT:-29685}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
EVAL_NPROCS="${EVAL_NPROCS:-1}"
NAGENTS="${NAGENTS:-32}"
NTHREADS="${NTHREADS:-4}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="${CORPUS_PATH:-../Data/PlotlyTable2Charts}"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"

TRAIN_TABLE_LIMIT="${TRAIN_TABLE_LIMIT:-128}"
VALID_TABLE_LIMIT="${VALID_TABLE_LIMIT:-32}"
MAX_EVAL_TABLES="${MAX_EVAL_TABLES:-32}"
TRAIN_TABLE_OFFSET="${TRAIN_TABLE_OFFSET:-0}"
VALID_TABLE_OFFSET="${VALID_TABLE_OFFSET:-0}"
EVAL_TABLE_OFFSET="${EVAL_TABLE_OFFSET:-0}"
TRAIN_TABLE_STRIDE="${TRAIN_TABLE_STRIDE:-1}"
VALID_TABLE_STRIDE="${VALID_TABLE_STRIDE:-1}"
EVAL_TABLE_STRIDE="${EVAL_TABLE_STRIDE:-1}"

EPOCHS="${EPOCHS:-1}"
MAX_TABLES="${MAX_TABLES:-48}"
MIN_MEMORY="${MIN_MEMORY:-512}"
MEMORY_SAMPLE_SIZE="${MEMORY_SAMPLE_SIZE:-48}"
MEMORY_SAMPLE_ROUNDS="${MEMORY_SAMPLE_ROUNDS:-1}"
SEARCH_LIMITS="${SEARCH_LIMITS:-e50-b4-na}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${MARKER_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_baseline_rl_poc.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_baseline_rl_poc.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_baseline_rl_poc.latest.results"

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
write_status "rl_nprocs=${RL_NPROCS}"
write_status "master_port=${MASTER_PORT}"
write_status "train_table_limit=${TRAIN_TABLE_LIMIT}"
write_status "valid_table_limit=${VALID_TABLE_LIMIT}"
write_status "train_table_offset=${TRAIN_TABLE_OFFSET}"
write_status "valid_table_offset=${VALID_TABLE_OFFSET}"
write_status "train_table_stride=${TRAIN_TABLE_STRIDE}"
write_status "valid_table_stride=${VALID_TABLE_STRIDE}"
write_status "max_eval_tables=${MAX_EVAL_TABLES}"
write_status "eval_table_offset=${EVAL_TABLE_OFFSET}"
write_status "eval_table_stride=${EVAL_TABLE_STRIDE}"
write_status "log_file=${LOG_FILE}"

write_result "run_id=${RUN_ID}"
write_result "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "gpu_ids=${GPU_IDS}"
write_result "train_table_limit=${TRAIN_TABLE_LIMIT}"
write_result "valid_table_limit=${VALID_TABLE_LIMIT}"
write_result "max_eval_tables=${MAX_EVAL_TABLES}"

append_results ""
append_results "## Plotly Baseline RL PoC ${RUN_ID}"
append_results ""
append_results "- started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- goal: standard RL baseline on the same subset as MC-light"
append_results "- gpu_ids: ${GPU_IDS}"
append_results "- train_table_limit: ${TRAIN_TABLE_LIMIT}"
append_results "- valid_table_limit: ${VALID_TABLE_LIMIT}"
append_results "- max_eval_tables: ${MAX_EVAL_TABLES}"

test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

write_status "status=baseline_rl_running"
echo "[RL:baseline] standard RL on the same subset used by MC-light PoC"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT}" \
  script.py \
  --corpus_path="${CORPUS_PATH}" \
  --model_size=small \
  --model_name=cp \
  --features=all-fast \
  --negative_weight=0.8 \
  --search_limits="${SEARCH_LIMITS}" \
  --epochs="${EPOCHS}" \
  -m "${MODEL_SAVE_PATH}" \
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
  --train_table_limit="${TRAIN_TABLE_LIMIT}" \
  --valid_table_limit="${VALID_TABLE_LIMIT}" \
  --train_table_offset="${TRAIN_TABLE_OFFSET}" \
  --valid_table_offset="${VALID_TABLE_OFFSET}" \
  --train_table_stride="${TRAIN_TABLE_STRIDE}" \
  --valid_table_stride="${VALID_TABLE_STRIDE}" \
  --min_memory="${MIN_MEMORY}" \
  --memory_sample_size="${MEMORY_SAMPLE_SIZE}" \
  --memory_sample_rounds="${MEMORY_SAMPLE_ROUNDS}"

RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name '*2el192fd128.128GRUh-allCharts-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
write_status "rl_dir=${RL_DIR}"
write_status "rl_ckpt=${RL_CKPT}"
write_result "rl_dir=${RL_DIR}"
write_result "rl_ckpt=${RL_CKPT}"
test -f "${RL_CKPT}"

RUN_ID="${RUN_ID}" \
GPU_IDS="${GPU_IDS}" \
EVAL_SPLIT="${EVAL_SPLIT}" \
EVAL_NPROCS="${EVAL_NPROCS}" \
NAGENTS="${NAGENTS}" \
NTHREADS="${NTHREADS}" \
MAX_EVAL_TABLES="${MAX_EVAL_TABLES}" \
EVAL_TABLE_OFFSET="${EVAL_TABLE_OFFSET}" \
EVAL_TABLE_STRIDE="${EVAL_TABLE_STRIDE}" \
MODEL_DIR="${RL_DIR}" \
CORPUS_PATH="${ROOT}/Data/PlotlyTable2Charts" \
SEARCH_LIMITS="${SEARCH_LIMITS}" \
"${ROOT}/Results/run_logs/run_plotly_update_teacher_eval_only.sh"

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
