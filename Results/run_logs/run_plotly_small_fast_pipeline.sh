#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-${ROOT}/Table2Charts}"
PY="${PY:-${ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_small_fast_pipeline_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_small_fast_pipeline_${RUN_ID}.status"
GPU_IDS="${GPU_IDS:-5}"
MASTER_PORT="${MASTER_PORT:-29515}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="../Data/PlotlyTable2Charts"
CHART_TYPE="allCharts"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "run_id=${RUN_ID}" | tee -a "${STATUS_FILE}"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${STATUS_FILE}"
echo "gpu_ids=${GPU_IDS}" | tee -a "${STATUS_FILE}"
echo "log_file=${LOG_FILE}" | tee -a "${STATUS_FILE}"
echo "status=sft_running" | tee -a "${STATUS_FILE}"

cd "${CODE_DIR}"

echo "[SFT] small model, full Plotly corpus, 1 epoch"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m nn_train.pretrain \
  --model_name=cp \
  --model_size=small \
  --features=all-fast \
  --train_batch_size=64 \
  --valid_batch_size=64 \
  --log_freq=500 \
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
  --num_workers=2

SFT_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -name '*2el192fd128.128GRUh-allCharts' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
SFT_CKPT="${SFT_DIR}/states_ep0.pt"
echo "sft_dir=${SFT_DIR}" | tee -a "${STATUS_FILE}"
echo "sft_ckpt=${SFT_CKPT}" | tee -a "${STATUS_FILE}"
test -f "${SFT_CKPT}"

echo "status=sft_eval_running" | tee -a "${STATUS_FILE}"
echo "[EVAL:SFT] test split, light search"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
  -m "${SFT_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/test-sft-fast \
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

echo "status=rl_running" | tee -a "${STATUS_FILE}"
echo "[RL] small model initialized from SFT, full Plotly corpus, 1 epoch, local queue"
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
echo "rl_dir=${RL_DIR}" | tee -a "${STATUS_FILE}"
echo "rl_ckpt=${RL_CKPT}" | tee -a "${STATUS_FILE}"
test -f "${RL_CKPT}"

echo "status=rl_eval_running" | tee -a "${STATUS_FILE}"
echo "[EVAL:RL] test split, light search"
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

echo "status=finished" | tee -a "${STATUS_FILE}"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${STATUS_FILE}"
