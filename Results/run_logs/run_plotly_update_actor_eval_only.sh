#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-${ROOT}/Table2Charts}"
PY="${PY:-${ROOT}/.venv/bin/python}"
MODEL_DIR="${1:?usage: $0 MODEL_DIR [RUN_ID]}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS="${GPU_IDS:-3,4,5,6}"

LOG_FILE="${ROOT}/Results/run_logs/plotly_update_actor_eval_only_${RUN_ID}.log"
STATUS_FILE="${ROOT}/Results/run_logs/plotly_update_actor_eval_only_${RUN_ID}.status"
RESULT_FILE="${ROOT}/Results/run_logs/plotly_update_actor_eval_only_${RUN_ID}.results"
RESULTS_MD="${ROOT}/results.md"
EVAL_DIR="${MODEL_DIR}/evaluations/test-update_actor-actor-critic-plotly-small-${RUN_ID}"

write_status() {
  printf '%s\n' "$1" >> "${STATUS_FILE}"
}

on_error() {
  local exit_code=$?
  write_status "status=failed"
  write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_status "exit_code=${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

: > "${STATUS_FILE}"
write_status "run_id=${RUN_ID}"
write_status "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "gpu_ids=${GPU_IDS}"
write_status "model_dir=${MODEL_DIR}"
write_status "log_file=${LOG_FILE}"
write_status "result_file=${RESULT_FILE}"
write_status "status=eval_running"

cd "${CODE_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" update_actor_test_agent_mp.py \
  -m "${MODEL_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path "evaluations/test-update_actor-actor-critic-plotly-small-${RUN_ID}" \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 4 \
  --nagents 64 \
  --nthreads 5 \
  --search_limits e50-b4-na \
  --corpus_path ../Data/PlotlyTable2Charts \
  --lang en \
  --limit_search_group > "${LOG_FILE}" 2>&1

write_status "status=completed"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'model_dir=%s\n' "${MODEL_DIR}"
  printf 'evaluation_log_dir=%s\n' "${EVAL_DIR}"
} > "${RESULT_FILE}"

{
  printf '\n'
  printf '### Update Actor eval rerun %s\n' "${RUN_ID}"
  printf '\n'
  printf -- '- finished_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf -- '- model_dir: %s\n' "${MODEL_DIR}"
  printf -- '- eval_log_dir: %s\n' "${EVAL_DIR}"
  printf -- '- log_file: %s\n' "${LOG_FILE}"
  printf '\n```text\n'
  grep -E "SUMMARY:|Complete recall info|Complete R@|R@1=|R@3=|R@5=|R@10=" "${LOG_FILE}" | tail -80 || true
  printf '```\n'
} >> "${RESULTS_MD}"
