#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-${ROOT}/Table2Charts}"
PY="${PY:-${ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_updated_policy_rl_eval_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_updated_policy_rl_eval_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_updated_policy_rl_eval_${RUN_ID}.results"
MARKER_FILE="${LOG_DIR}/plotly_updated_policy_rl_eval_${RUN_ID}.marker"
GPU_IDS="${GPU_IDS:-3,5,6,7}"
RL_NPROCS="${RL_NPROCS:-4}"
EVAL_NPROCS="${EVAL_NPROCS:-4}"
MASTER_PORT="${MASTER_PORT:-29626}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="../Data/PlotlyTable2Charts"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"
POLICY_EPSILON_START="${POLICY_EPSILON_START:-0.2}"
POLICY_EPSILON_END="${POLICY_EPSILON_END:-0.02}"
POLICY_EPSILON_DECAY="${POLICY_EPSILON_DECAY:-0.8}"
POLICY_EXPLORE_TOP_M="${POLICY_EXPLORE_TOP_M:-5}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${MARKER_FILE}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_updated_policy_rl_eval.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_updated_policy_rl_eval.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_updated_policy_rl_eval.latest.results"

exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
  echo "$1" | tee -a "${STATUS_FILE}"
}

write_result() {
  echo "$1" | tee -a "${RESULT_FILE}"
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

write_status "run_id=${RUN_ID}"
write_status "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "gpu_ids=${GPU_IDS}"
write_status "rl_nprocs=${RL_NPROCS}"
write_status "eval_nprocs=${EVAL_NPROCS}"
write_status "master_port=${MASTER_PORT}"
write_status "sft_ckpt=${SFT_CKPT}"
write_status "policy_epsilon_start=${POLICY_EPSILON_START}"
write_status "policy_epsilon_end=${POLICY_EPSILON_END}"
write_status "policy_epsilon_decay=${POLICY_EPSILON_DECAY}"
write_status "policy_explore_top_m=${POLICY_EXPLORE_TOP_M}"
write_status "log_file=${LOG_FILE}"
write_status "status_file=${STATUS_FILE}"
write_status "result_file=${RESULT_FILE}"

write_result "run_id=${RUN_ID}"
write_result "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "gpu_ids=${GPU_IDS}"
write_result "sft_ckpt=${SFT_CKPT}"
write_result "policy=epsilon_top_m"
write_result "policy_epsilon_start=${POLICY_EPSILON_START}"
write_result "policy_epsilon_end=${POLICY_EPSILON_END}"
write_result "policy_epsilon_decay=${POLICY_EPSILON_DECAY}"
write_result "policy_explore_top_m=${POLICY_EXPLORE_TOP_M}"

test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

write_status "status=updated_policy_rl_running"
echo "[RL:updated_policy] resume from existing SFT checkpoint, full Plotly corpus, 1 epoch"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT}" \
  --module reinforce.updated_policy_learn_dist \
  --corpus_path="${CORPUS_PATH}" \
  --model_size=small \
  --model_name=cp \
  --features=all-fast \
  --negative_weight=0.8 \
  --search_limits=e50-b4-na \
  --epochs=1 \
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
  --max_tables=64 \
  --min_memory=1000 \
  --memory_sample_size=64 \
  --memory_sample_rounds=2 \
  --policy_epsilon_start="${POLICY_EPSILON_START}" \
  --policy_epsilon_end="${POLICY_EPSILON_END}" \
  --policy_epsilon_decay="${POLICY_EPSILON_DECAY}" \
  --policy_explore_top_m="${POLICY_EXPLORE_TOP_M}"

RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name '*2el192fd128.128GRUh-allCharts-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
write_status "rl_dir=${RL_DIR}"
write_status "rl_ckpt=${RL_CKPT}"
write_result "rl_dir=${RL_DIR}"
write_result "rl_ckpt=${RL_CKPT}"
test -f "${RL_CKPT}"

write_status "status=updated_policy_rl_eval_running"
echo "[EVAL:updated_policy_RL] Plotly test split, deterministic greedy evaluation"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
  -m "${RL_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path "evaluations/test-updated-policy-rl-plotly-small-${RUN_ID}" \
  --search_type "${CHART_TYPE}" \
  --input_type "${CHART_TYPE}" \
  --previous_type "${CHART_TYPE}" \
  --nprocs "${EVAL_NPROCS}" \
  --nagents 64 \
  --nthreads 5 \
  --search_limits e50-b4-na \
  --corpus_path "${CORPUS_PATH}" \
  --lang en \
  --limit_search_group

write_result "evaluation_log_dir=${RL_DIR}/evaluations/test-updated-policy-rl-plotly-small-${RUN_ID}"
grep -E "Complete recall info|Complete R@|R@1=|R@3=|R@5=|R@10=" "${LOG_FILE}" | tail -40 | tee -a "${RESULT_FILE}" || true

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
