#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-${ROOT}/Table2Charts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS_PATH="${CORPUS_PATH:-${ROOT}/Data/PlotlyTable2Charts}"
MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-${ROOT}/Results/Models}"
SUMMARY_PATH="${SUMMARY_PATH:-${ROOT}/Results/summary}"
GPU_IDS="${GPU_IDS:-0,1}"
RL_NPROCS="${RL_NPROCS:-2}"
EVAL_NPROCS="${EVAL_NPROCS:-2}"
MASTER_PORT="${MASTER_PORT:-29649}"
DRY_RUN="${DRY_RUN:-0}"
SFT_CKPT="${SFT_CKPT:?SFT_CKPT must point to a completed SFT states_ep0.pt on the remote host}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_LOG_ROOT="${ROOT}/experiments/results/raw_logs"
RUN_LOG="${RAW_LOG_ROOT}/reward_interaction_${RUN_ID}.log"

CONFIGS=(
  "experiments/configs/reward_current_greedy.json"
  "experiments/configs/reward_current_epsilon.json"
)

# Optional sensitivity run; do not include by default.
# OPTIONAL_CONFIG="experiments/configs/reward_conservative_greedy.json"

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

cd "${ROOT}"

for config in "${CONFIGS[@]}"; do
  echo
  echo "=== running ${config} ==="
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
  "${PYTHON_BIN}" experiments/scripts/run_experiments.py --config "${config}" --dry-run

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

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
  CONFIG_PATH="${config}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import subprocess
from pathlib import Path

from experiments.scripts.run_experiments import load_json, resolve_defaults, build_command_plan, quote_command

config_path = Path(os.environ["CONFIG_PATH"]).resolve()
config = load_json(config_path)
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
print("todo=post-training evaluation is intentionally deferred; discover the produced RL model dir before running test_agent_mp.py")
PY
done
