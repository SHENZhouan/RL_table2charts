#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_update_UCB_eval_only_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_update_UCB_eval_only_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_update_UCB_eval_only_${RUN_ID}.results"
PID_FILE="${LOG_DIR}/plotly_update_UCB_eval_only_${RUN_ID}.pid"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-0}"
EVAL_NPROCS="${EVAL_NPROCS:-1}"
NAGENTS="${NAGENTS:-64}"
NTHREADS="${NTHREADS:-5}"
UCB_EXPLORATION="${UCB_EXPLORATION:-0.5}"
CORPUS_PATH="${CORPUS_PATH:-../Data/PlotlyTable2Charts}"
CHART_TYPE="${CHART_TYPE:-allCharts}"
SEARCH_LIMITS="${SEARCH_LIMITS:-e50-b4-na}"
MODEL_DIR="${MODEL_DIR:-${1:-}}"
MODEL_FILE="${MODEL_FILE:-states_ep0.pt}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_update_UCB_eval_only.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_update_UCB_eval_only.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_update_UCB_eval_only.latest.results"
printf '%s\n' "$$" > "${PID_FILE}"
ln -sfn "${PID_FILE}" "${LOG_DIR}/plotly_update_UCB_eval_only.latest.pid"

exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
  echo "$1" | tee -a "${STATUS_FILE}"
}

write_result() {
  echo "$1" | tee -a "${RESULT_FILE}"
}

append_results() {
  echo "$1" | tee -a "${RESULTS_MD}"
}

on_error() {
  local exit_code=$?
  write_status "status=failed"
  write_status "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_status "failed_line=${BASH_LINENO[0]}"
  write_status "exit_code=${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

if [[ -z "${MODEL_DIR}" ]]; then
  MODEL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -name '*-UCB-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi

MODEL_CKPT="${MODEL_DIR}/${MODEL_FILE}"
EVAL_SUBDIR="evaluations/test-update-UCB-plotly-small-${RUN_ID}"
EVAL_DIR="${MODEL_DIR}/${EVAL_SUBDIR}"

write_status "run_id=${RUN_ID}"
write_status "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "pid=$$"
write_status "gpu_ids=${GPU_IDS}"
write_status "eval_nprocs=${EVAL_NPROCS}"
write_status "ucb_exploration=${UCB_EXPLORATION}"
write_status "model_dir=${MODEL_DIR}"
write_status "model_ckpt=${MODEL_CKPT}"
write_status "log_file=${LOG_FILE}"
write_status "result_file=${RESULT_FILE}"
write_status "status=eval_running"

write_result "run_id=${RUN_ID}"
write_result "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "gpu_ids=${GPU_IDS}"
write_result "eval_nprocs=${EVAL_NPROCS}"
write_result "ucb_exploration=${UCB_EXPLORATION}"
write_result "model_dir=${MODEL_DIR}"
write_result "model_ckpt=${MODEL_CKPT}"

test -d "${MODEL_DIR}"
test -f "${MODEL_CKPT}"

cd "${CODE_DIR}"

echo "[EVAL:update_UCB] Plotly test split, single-GPU reusable UCB search evaluation"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" update_UCB_test_agent_mp.py \
  -m "${MODEL_DIR}" \
  -f "${MODEL_FILE}" \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path "${EVAL_SUBDIR}" \
  --search_type "${CHART_TYPE}" \
  --input_type "${CHART_TYPE}" \
  --previous_type "${CHART_TYPE}" \
  --nprocs "${EVAL_NPROCS}" \
  --nagents "${NAGENTS}" \
  --nthreads "${NTHREADS}" \
  --search_limits "${SEARCH_LIMITS}" \
  --corpus_path "${CORPUS_PATH}" \
  --lang en \
  --ucb_exploration "${UCB_EXPLORATION}" \
  --limit_search_group

write_result "evaluation_log_dir=${EVAL_DIR}"
grep -E "Complete recall info|Complete R@|R@1=|R@3=|R@5=|R@10=" "${LOG_FILE}" | tail -40 | tee -a "${RESULT_FILE}" || true

append_results ""
append_results "### Update UCB Eval Only ${RUN_ID}"
append_results ""
append_results "- finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- model_dir: ${MODEL_DIR}"
append_results "- model_ckpt: ${MODEL_CKPT}"
append_results "- eval_log_dir: ${EVAL_DIR}"
append_results "- log_file: ${LOG_FILE}"
append_results ""
append_results '```text'
grep -E "Complete recall info|Complete R@|R@1=|R@3=|R@5=|R@10=" "${LOG_FILE}" | tail -80 | tee -a "${RESULTS_MD}" || true
append_results '```'

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
