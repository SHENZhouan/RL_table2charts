#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
LOG_DIR="${ROOT}/Results/run_logs"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_MD="${ROOT}/results.md"

GPU_IDS="${GPU_IDS:-5}"
RL_NPROCS="${RL_NPROCS:-1}"
MASTER_PORT="${MASTER_PORT:-29684}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
EVAL_NPROCS="${EVAL_NPROCS:-1}"
NAGENTS="${NAGENTS:-32}"
NTHREADS="${NTHREADS:-4}"

TRAIN_TABLE_LIMIT="${TRAIN_TABLE_LIMIT:-192}"
VALID_TABLE_LIMIT="${VALID_TABLE_LIMIT:-48}"
MAX_EVAL_TABLES="${MAX_EVAL_TABLES:-48}"
TRAIN_TABLE_OFFSET="${TRAIN_TABLE_OFFSET:-0}"
VALID_TABLE_OFFSET="${VALID_TABLE_OFFSET:-0}"
EVAL_TABLE_OFFSET="${EVAL_TABLE_OFFSET:-0}"
TRAIN_TABLE_STRIDE="${TRAIN_TABLE_STRIDE:-1}"
VALID_TABLE_STRIDE="${VALID_TABLE_STRIDE:-1}"
EVAL_TABLE_STRIDE="${EVAL_TABLE_STRIDE:-1}"

SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"

mkdir -p "${LOG_DIR}"
touch "${RESULTS_MD}"

{
  printf '\n'
  printf '## Plotly Update MC Light PoC %s\n' "${RUN_ID}"
  printf '\n'
  printf -- '- started_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf -- '- goal: 3-hour proof-of-concept subset run\n'
  printf -- '- gpu_ids: %s\n' "${GPU_IDS}"
  printf -- '- train_table_limit: %s\n' "${TRAIN_TABLE_LIMIT}"
  printf -- '- valid_table_limit: %s\n' "${VALID_TABLE_LIMIT}"
  printf -- '- max_eval_tables: %s\n' "${MAX_EVAL_TABLES}"
} >> "${RESULTS_MD}"

RUN_ID="${RUN_ID}" \
GPU_IDS="${GPU_IDS}" \
RL_NPROCS="${RL_NPROCS}" \
MASTER_PORT="${MASTER_PORT}" \
SFT_CKPT="${SFT_CKPT}" \
MC_TOP_K="${MC_TOP_K:-2}" \
MC_ROLLOUT_DEPTH="${MC_ROLLOUT_DEPTH:-1}" \
MC_NUM_ROLLOUTS="${MC_NUM_ROLLOUTS:-1}" \
MC_DISCOUNT="${MC_DISCOUNT:-0.9}" \
MC_ROLLOUT_WEIGHT="${MC_ROLLOUT_WEIGHT:-0.35}" \
EPOCHS="${EPOCHS:-1}" \
MAX_TABLES="${MAX_TABLES:-48}" \
TRAIN_TABLE_LIMIT="${TRAIN_TABLE_LIMIT}" \
VALID_TABLE_LIMIT="${VALID_TABLE_LIMIT}" \
TRAIN_TABLE_OFFSET="${TRAIN_TABLE_OFFSET}" \
VALID_TABLE_OFFSET="${VALID_TABLE_OFFSET}" \
TRAIN_TABLE_STRIDE="${TRAIN_TABLE_STRIDE}" \
VALID_TABLE_STRIDE="${VALID_TABLE_STRIDE}" \
MIN_MEMORY="${MIN_MEMORY:-512}" \
MEMORY_SAMPLE_SIZE="${MEMORY_SAMPLE_SIZE:-48}" \
MEMORY_SAMPLE_ROUNDS="${MEMORY_SAMPLE_ROUNDS:-1}" \
AGENT_WORKERS="${AGENT_WORKERS:-0}" \
"${ROOT}/Results/run_logs/run_plotly_update_MC_light_rl.sh"

MODEL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -name '*-MC-light-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"

RUN_ID="${RUN_ID}" \
GPU_IDS="${GPU_IDS}" \
EVAL_SPLIT="${EVAL_SPLIT}" \
EVAL_NPROCS="${EVAL_NPROCS}" \
NAGENTS="${NAGENTS}" \
NTHREADS="${NTHREADS}" \
MAX_EVAL_TABLES="${MAX_EVAL_TABLES}" \
EVAL_TABLE_OFFSET="${EVAL_TABLE_OFFSET}" \
EVAL_TABLE_STRIDE="${EVAL_TABLE_STRIDE}" \
MODEL_DIR="${MODEL_DIR}" \
"${ROOT}/Results/run_logs/run_plotly_update_teacher_eval_only.sh"
