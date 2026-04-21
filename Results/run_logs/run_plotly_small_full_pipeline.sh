#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_small_full_pipeline_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_small_full_pipeline_${RUN_ID}.status"
GPU_IDS="${GPU_IDS:-5}"
MASTER_PORT="${MASTER_PORT:-29516}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="../Data/PlotlyTable2Charts"
CHART_TYPE="allCharts"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
  echo "$1" | tee -a "${STATUS_FILE}"
}

write_status "run_id=${RUN_ID}"
write_status "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "gpu_ids=${GPU_IDS}"
write_status "log_file=${LOG_FILE}"

cd "${CODE_DIR}"

write_status "status=sft_running"
echo "[SFT] small model, full Plotly corpus, 1 epoch, conservative batch"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" MASTER_PORT="${MASTER_PORT}" "${PY}" -m nn_train.pretrain \
  --model_name=cp \
  --model_size=small \
  --features=all-fast \
  --train_batch_size=32 \
  --valid_batch_size=32 \
  --log_freq=200 \
  --negative_weight=0.8 \
  --search_type="${CHART_TYPE}" \
  --input_type="${CHART_TYPE}" \
  --previous_type="${CHART_TYPE}" \
  --model_save_path="${MODEL_SAVE_PATH}" \
  --summary_path="${SUMMARY_PATH}" \
  --corpus_path="${CORPUS_PATH}" \
  --lang=en \
  --epochs=1 \
  --nprocs=1 \
  --num_workers=0

SFT_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -name '*2el192fd128.128GRUh-allCharts' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
SFT_CKPT="${SFT_DIR}/states_ep0.pt"
write_status "sft_dir=${SFT_DIR}"
write_status "sft_ckpt=${SFT_CKPT}"
test -f "${SFT_CKPT}"

write_status "status=rl_running"
echo "[RL] small model initialized from full Plotly SFT, 1 epoch, local queue"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  script.py \
  --corpus_path="${CORPUS_PATH}" \
  --model_size=small \
  --model_name=cp \
  --features=all-fast \
  --negative_weight=0.8 \
  --search_limits=e50-b4-na \
  --epochs=1 \
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
  --max_tables=64 \
  --min_memory=1000 \
  --memory_sample_size=64 \
  --memory_sample_rounds=2

RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -name '*2el192fd128.128GRUh-allCharts-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
write_status "rl_dir=${RL_DIR}"
write_status "rl_ckpt=${RL_CKPT}"
test -f "${RL_CKPT}"

write_status "status=rl_eval_running"
echo "[EVAL:RL] Plotly test split, light search"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
  -m "${RL_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/test-rl-fast \
  --search_type "${CHART_TYPE}" \
  --input_type "${CHART_TYPE}" \
  --previous_type "${CHART_TYPE}" \
  --nprocs 1 \
  --nagents 32 \
  --nthreads 5 \
  --search_limits e50-b4-na \
  --corpus_path "${CORPUS_PATH}" \
  --lang en \
  --limit_search_group

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
