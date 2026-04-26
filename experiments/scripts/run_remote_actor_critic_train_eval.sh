#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-${ROOT}/Table2Charts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS_PATH="${CORPUS_PATH:-${ROOT}/Data/PlotlyTable2Charts}"
MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-${ROOT}/Results/Models}"
SUMMARY_PATH="${SUMMARY_PATH:-${ROOT}/Results/summary}"
GPU_IDS="${GPU_IDS:-0}"
RL_NPROCS="${RL_NPROCS:-1}"
EVAL_NPROCS="${EVAL_NPROCS:-1}"
MASTER_PORT="${MASTER_PORT:-29653}"
DRY_RUN="${DRY_RUN:-0}"
SFT_CKPT="${SFT_CKPT:?SFT_CKPT must point to a completed SFT states_ep0.pt on the remote host}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_LOG_ROOT="${ROOT}/experiments/results/raw_logs"
RUN_LOG="${RAW_LOG_ROOT}/actor_critic_train_eval_${RUN_ID}.log"
RESULT_FILE="${RAW_LOG_ROOT}/actor_critic_train_eval_${RUN_ID}.results"
MARKER_FILE="${RAW_LOG_ROOT}/actor_critic_train_eval_${RUN_ID}.marker"
TRAIN_CONFIG="experiments/configs/actor_critic_train_eval.json"
DISCOVER_GLOB="*update_actor_new-*allCharts-actor-policy-RL"

EVAL_CONFIGS=(
  "experiments/configs/actor_critic_actor_score.json"
  "experiments/configs/actor_critic_critic_score.json"
  "experiments/configs/actor_critic_blend_score.json"
)

mkdir -p "${RAW_LOG_ROOT}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "run_id=${RUN_ID}"
echo "root=${ROOT}"
echo "code_dir=${CODE_DIR}"
echo "python_bin=${PYTHON_BIN}"
echo "corpus_path=${CORPUS_PATH}"
echo "sft_ckpt=${SFT_CKPT}"
echo "model_save_path=${MODEL_SAVE_PATH}"
echo "summary_path=${SUMMARY_PATH}"
echo "gpu_ids=${GPU_IDS}"
echo "rl_nprocs=${RL_NPROCS}"
echo "eval_nprocs=${EVAL_NPROCS}"
echo "master_port=${MASTER_PORT}"
echo "dry_run=${DRY_RUN}"
echo "run_log=${RUN_LOG}"
echo "result_file=${RESULT_FILE}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  ROOT="${ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CORPUS_PATH="${CORPUS_PATH}" \
  SFT_CKPT="${SFT_CKPT}" \
  MODEL_SAVE_PATH="${MODEL_SAVE_PATH}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  GPU_IDS="${GPU_IDS}" \
  RL_NPROCS="${RL_NPROCS}" \
  EVAL_NPROCS="${EVAL_NPROCS}" \
  MASTER_PORT="${MASTER_PORT}" \
  "${PYTHON_BIN}" experiments/scripts/run_experiments.py --config "${TRAIN_CONFIG}" --dry-run

  echo "post_train_discovery: find ${MODEL_SAVE_PATH} -maxdepth 1 -type d -newer ${MARKER_FILE} -name '${DISCOVER_GLOB}'"
  echo "post_train_checkpoint: <discovered_actor_critic_dir>/states_ep0.pt"
  for config in "${EVAL_CONFIGS[@]}"; do
    echo
    echo "=== planned eval ${config} ==="
    ROOT="${ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    CORPUS_PATH="${CORPUS_PATH}" \
    ACTOR_CRITIC_CKPT="${MODEL_SAVE_PATH}/<discovered_actor_critic_dir>/states_ep0.pt" \
    GPU_IDS="${GPU_IDS}" \
    EVAL_NPROCS="${EVAL_NPROCS}" \
    "${PYTHON_BIN}" experiments/scripts/run_experiments.py --config "${config}" --dry-run
  done
  exit 0
fi

test -d "${CORPUS_PATH}"
test -f "${SFT_CKPT}"
mkdir -p "${MODEL_SAVE_PATH}" "${SUMMARY_PATH}"
touch "${MARKER_FILE}"

ROOT="${ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
CORPUS_PATH="${CORPUS_PATH}" \
SFT_CKPT="${SFT_CKPT}" \
MODEL_SAVE_PATH="${MODEL_SAVE_PATH}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
GPU_IDS="${GPU_IDS}" \
RL_NPROCS="${RL_NPROCS}" \
EVAL_NPROCS="${EVAL_NPROCS}" \
MASTER_PORT="${MASTER_PORT}" \
CONFIG_PATH="${TRAIN_CONFIG}" \
"${PYTHON_BIN}" - <<'PY'
import os
import subprocess
from pathlib import Path

from experiments.scripts.run_experiments import load_json, resolve_defaults, build_command_plan, quote_command

config_path = Path(os.environ["CONFIG_PATH"]).resolve()
config = load_json(config_path)
config.setdefault("runtime", {})
config["runtime"]["rl_nprocs"] = int(os.environ["RL_NPROCS"])
config["runtime"]["eval_nprocs"] = int(os.environ["EVAL_NPROCS"])
resolved = resolve_defaults(config)
plan = build_command_plan(config, resolved)
if plan["status"] != "runnable":
    raise SystemExit(f"{config['name']} is not runnable: {plan['todo']}")

train_command = None
for command in plan["commands"]:
    if isinstance(command, list) and command and not str(command[0]).startswith("#"):
        train_command = command
        break
if train_command is None:
    raise SystemExit(f"No runnable training command found for {config['name']}")

print(f"resolved_train_command={quote_command(train_command)}")
subprocess.run(train_command, check=True, cwd=resolved["code_dir"])
PY

RL_DIR="$(find "${MODEL_SAVE_PATH}" -maxdepth 1 -type d -newer "${MARKER_FILE}" -name "${DISCOVER_GLOB}" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
RL_CKPT="${RL_DIR}/states_ep0.pt"
test -n "${RL_DIR}"
test -f "${RL_CKPT}"
echo "discovered_actor_critic_dir=${RL_DIR}"
echo "discovered_actor_critic_ckpt=${RL_CKPT}"

{
  echo "run_id=${RUN_ID}"
  echo "model_dir=${RL_DIR}"
  echo "checkpoint=${RL_CKPT}"
} >> "${RESULT_FILE}"

for config in "${EVAL_CONFIGS[@]}"; do
  CONFIG_NAME="$(basename "${config}" .json)"
  echo
  echo "=== running eval ${config} ==="
  ROOT="${ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CORPUS_PATH="${CORPUS_PATH}" \
  ACTOR_CRITIC_CKPT="${RL_CKPT}" \
  MODEL_SAVE_PATH="${MODEL_SAVE_PATH}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_NPROCS="${EVAL_NPROCS}" \
  "${PYTHON_BIN}" experiments/scripts/run_experiments.py --config "${config}"
  {
    echo "eval_config=${CONFIG_NAME}"
    echo "eval_log_dir=${RL_DIR}/$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_dir"])' "${config}")"
  } >> "${RESULT_FILE}"
done

{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
} >> "${RESULT_FILE}"
