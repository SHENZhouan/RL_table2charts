#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_update_MC_light_rl_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_update_MC_light_rl_${RUN_ID}.status"
RESULT_FILE="${LOG_DIR}/plotly_update_MC_light_rl_${RUN_ID}.results"
MARKER_FILE="${LOG_DIR}/plotly_update_MC_light_rl_${RUN_ID}.marker"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-5}"
RL_NPROCS="${RL_NPROCS:-1}"
MASTER_PORT="${MASTER_PORT:-29674}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="${CORPUS_PATH:-../Data/PlotlyTable2Charts}"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"

MC_TOP_K="${MC_TOP_K:-2}"
MC_ROLLOUT_DEPTH="${MC_ROLLOUT_DEPTH:-1}"
MC_NUM_ROLLOUTS="${MC_NUM_ROLLOUTS:-1}"
MC_DISCOUNT="${MC_DISCOUNT:-0.9}"
MC_ROLLOUT_WEIGHT="${MC_ROLLOUT_WEIGHT:-0.35}"
EPOCHS="${EPOCHS:-1}"
MAX_TABLES="${MAX_TABLES:-64}"
TRAIN_TABLE_LIMIT="${TRAIN_TABLE_LIMIT:-256}"
VALID_TABLE_LIMIT="${VALID_TABLE_LIMIT:-64}"
TRAIN_TABLE_OFFSET="${TRAIN_TABLE_OFFSET:-0}"
VALID_TABLE_OFFSET="${VALID_TABLE_OFFSET:-0}"
TRAIN_TABLE_STRIDE="${TRAIN_TABLE_STRIDE:-1}"
VALID_TABLE_STRIDE="${VALID_TABLE_STRIDE:-1}"
MIN_MEMORY="${MIN_MEMORY:-1000}"
MEMORY_SAMPLE_SIZE="${MEMORY_SAMPLE_SIZE:-64}"
MEMORY_SAMPLE_ROUNDS="${MEMORY_SAMPLE_ROUNDS:-2}"
AGENT_WORKERS="${AGENT_WORKERS:-0}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULT_FILE}" "${MARKER_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_update_MC_light_rl.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_update_MC_light_rl.latest.status"
ln -sfn "${RESULT_FILE}" "${LOG_DIR}/plotly_update_MC_light_rl.latest.results"

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
write_status "mc_top_k=${MC_TOP_K}"
write_status "mc_rollout_depth=${MC_ROLLOUT_DEPTH}"
write_status "mc_num_rollouts=${MC_NUM_ROLLOUTS}"
write_status "mc_rollout_weight=${MC_ROLLOUT_WEIGHT}"
write_status "train_table_limit=${TRAIN_TABLE_LIMIT}"
write_status "valid_table_limit=${VALID_TABLE_LIMIT}"
write_status "train_table_offset=${TRAIN_TABLE_OFFSET}"
write_status "valid_table_offset=${VALID_TABLE_OFFSET}"
write_status "train_table_stride=${TRAIN_TABLE_STRIDE}"
write_status "valid_table_stride=${VALID_TABLE_STRIDE}"
write_status "agent_workers=${AGENT_WORKERS}"
write_status "log_file=${LOG_FILE}"
write_status "status=update_MC_light_rl_running"

test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

echo "[RL:update_MC_light] lighter top-k + shallow MC rollout over critic scores"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT}" \
  --module reinforce.update_MC_light_learn_dist \
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
  --train_table_limit="${TRAIN_TABLE_LIMIT}" \
  --valid_table_limit="${VALID_TABLE_LIMIT}" \
  --train_table_offset="${TRAIN_TABLE_OFFSET}" \
  --valid_table_offset="${VALID_TABLE_OFFSET}" \
  --train_table_stride="${TRAIN_TABLE_STRIDE}" \
  --valid_table_stride="${VALID_TABLE_STRIDE}" \
  --min_memory="${MIN_MEMORY}" \
  --memory_sample_size="${MEMORY_SAMPLE_SIZE}" \
  --memory_sample_rounds="${MEMORY_SAMPLE_ROUNDS}" \
  --mc_top_k="${MC_TOP_K}" \
  --mc_rollout_depth="${MC_ROLLOUT_DEPTH}" \
  --mc_num_rollouts="${MC_NUM_ROLLOUTS}" \
  --mc_discount="${MC_DISCOUNT}" \
  --mc_rollout_weight="${MC_ROLLOUT_WEIGHT}" \
  --agent_workers="${AGENT_WORKERS}"

RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name '*-MC-light-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
write_status "rl_dir=${RL_DIR}"
write_status "rl_ckpt=${RL_CKPT}"
write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
