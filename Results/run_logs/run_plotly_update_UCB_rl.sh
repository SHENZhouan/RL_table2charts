#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_update_UCB_rl_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_update_UCB_rl_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_update_UCB_rl_${RUN_ID}.results"
MARKER_FILE="${LOG_DIR}/plotly_update_UCB_rl_${RUN_ID}.marker"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-3,4,5,6}"
RL_NPROCS="${RL_NPROCS:-4}"
MASTER_PORT="${MASTER_PORT:-29661}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="../Data/PlotlyTable2Charts"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"

UCB_EXPLORATION="${UCB_EXPLORATION:-0.5}"
EPOCHS="${EPOCHS:-1}"
MAX_TABLES="${MAX_TABLES:-64}"
MIN_MEMORY="${MIN_MEMORY:-1000}"
MEMORY_SAMPLE_SIZE="${MEMORY_SAMPLE_SIZE:-64}"
MEMORY_SAMPLE_ROUNDS="${MEMORY_SAMPLE_ROUNDS:-2}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${MARKER_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_update_UCB_rl.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_update_UCB_rl.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_update_UCB_rl.latest.results"

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
  append_results ""
  append_results "### Plotly Update UCB failed"
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
write_status "ucb_exploration=${UCB_EXPLORATION}"
write_status "epochs=${EPOCHS}"
write_status "log_file=${LOG_FILE}"
write_status "status_file=${STATUS_FILE}"
write_status "result_file=${RESULT_FILE}"

write_result "run_id=${RUN_ID}"
write_result "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "gpu_ids=${GPU_IDS}"
write_result "sft_ckpt=${SFT_CKPT}"
write_result "ucb_exploration=${UCB_EXPLORATION}"
write_result "epochs=${EPOCHS}"

append_results ""
append_results "## Plotly Update UCB ${RUN_ID}"
append_results ""
append_results "- started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- gpu_ids: ${GPU_IDS}"
append_results "- sft_ckpt: ${SFT_CKPT}"
append_results "- ucb_exploration: ${UCB_EXPLORATION}"
append_results "- epochs: ${EPOCHS}"
append_results "- log_file: ${LOG_FILE}"

test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

write_status "status=update_UCB_rl_running"
echo "[RL:update_UCB] resume from SFT checkpoint, UCB over critic scores"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT}" \
  --module reinforce.update_UCB_learn_dist \
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
  --ucb_exploration="${UCB_EXPLORATION}" \
  --ucb_eval_exploration

RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name '*-UCB-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
write_status "rl_dir=${RL_DIR}"
write_status "rl_ckpt=${RL_CKPT}"
write_result "rl_dir=${RL_DIR}"
write_result "rl_ckpt=${RL_CKPT}"
test -f "${RL_CKPT}"

append_results ""
append_results "### Update UCB RL"
append_results ""
append_results "- finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- rl_dir: ${RL_DIR}"
append_results ""
append_results '```text'
grep -E "SUMMARY:|R@1=|R@3=|R@5=|R@10=" "${LOG_FILE}" | tail -80 | tee -a "${RESULTS_MD}" | tee -a "${RESULT_FILE}" || true
append_results '```'

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_result "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
