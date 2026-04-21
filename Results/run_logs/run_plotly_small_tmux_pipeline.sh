#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_small_tmux_pipeline_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_small_tmux_pipeline_${RUN_ID}.status"
MARKER_FILE="${LOG_DIR}/plotly_small_tmux_pipeline_${RUN_ID}.marker"
GPU_IDS="${GPU_IDS:-3,4,5,6}"
SFT_NPROCS="${SFT_NPROCS:-4}"
EVAL_NPROCS="${EVAL_NPROCS:-4}"
MASTER_PORT="${MASTER_PORT:-29616}"
RUN_SFT_EVAL="${RUN_SFT_EVAL:-0}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="../Data/PlotlyTable2Charts"
CHART_TYPE="allCharts"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${MARKER_FILE}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_small_tmux_pipeline.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_small_tmux_pipeline.latest.status"

exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
  echo "$1" | tee -a "${STATUS_FILE}"
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
write_status "sft_nprocs=${SFT_NPROCS}"
write_status "eval_nprocs=${EVAL_NPROCS}"
write_status "run_sft_eval=${RUN_SFT_EVAL}"
write_status "master_port=${MASTER_PORT}"
write_status "log_file=${LOG_FILE}"
write_status "status_file=${STATUS_FILE}"

cd "${CODE_DIR}"

write_status "status=sft_running"
echo "[SFT] small model, full Plotly corpus, 1 epoch, ${SFT_NPROCS} GPU(s)"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" MASTER_ADDR=localhost MASTER_PORT="${MASTER_PORT}" "${PY}" -m nn_train.pretrain \
  --model_name=cp \
  --model_size=small \
  --features=all-fast \
  --train_batch_size=64 \
  --valid_batch_size=64 \
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
  --nprocs="${SFT_NPROCS}" \
  --num_workers=2

SFT_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name '*2el192fd128.128GRUh-allCharts' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
SFT_CKPT="${SFT_DIR}/states_ep0.pt"
write_status "sft_dir=${SFT_DIR}"
write_status "sft_ckpt=${SFT_CKPT}"
test -f "${SFT_CKPT}"

if [[ "${RUN_SFT_EVAL}" == "1" ]]; then
  write_status "status=sft_eval_running"
  echo "[EVAL:SFT] Plotly test split, parallel light search"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
    -m "${SFT_DIR}" \
    -f states_ep0.pt \
    --model_name cp \
    --model_size small \
    --features all-fast \
    --log_save_path evaluations/test-sft-plotly-small-${RUN_ID} \
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
else
  write_status "status=sft_eval_skipped"
  echo "[EVAL:SFT] skipped to reduce total runtime"
fi

write_status "status=rl_running"
echo "[RL] small model initialized from SFT, full Plotly corpus, 1 epoch, local queue"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${SFT_NPROCS}" \
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

RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name '*2el192fd128.128GRUh-allCharts-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
write_status "rl_dir=${RL_DIR}"
write_status "rl_ckpt=${RL_CKPT}"
test -f "${RL_CKPT}"

write_status "status=rl_eval_running"
echo "[EVAL:RL] Plotly test split, parallel light search"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
  -m "${RL_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/test-rl-plotly-small-${RUN_ID} \
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

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
