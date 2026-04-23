#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/ssd/shenzhouan/Table2Charts"
CODE_DIR="${ROOT}/Table2Charts"
PY="${ROOT}/.venv/bin/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/Results/run_logs"
LOG_FILE="${LOG_DIR}/plotly_update_reward_then_combo_${RUN_ID}.log"
STATUS_FILE="${LOG_DIR}/plotly_update_reward_then_combo_${RUN_ID}.status"
RESULTS_MD="${ROOT}/results.md"
GPU_IDS="${GPU_IDS:-3,4,5,6}"
RL_NPROCS="${RL_NPROCS:-4}"
EVAL_NPROCS="${EVAL_NPROCS:-4}"
MASTER_PORT_REWARD="${MASTER_PORT_REWARD:-29627}"
MASTER_PORT_COMBO="${MASTER_PORT_COMBO:-29628}"

MODEL_SAVE_PATH="../Results/Models"
SUMMARY_PATH="../Results/summary"
CORPUS_PATH="../Data/PlotlyTable2Charts"
CHART_TYPE="allCharts"
SFT_CKPT="${SFT_CKPT:-${ROOT}/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt}"

UPDATE_REWARD_EXACT="${UPDATE_REWARD_EXACT:-0.95}"
UPDATE_REWARD_DEFAULT="${UPDATE_REWARD_DEFAULT:-0.05}"
UPDATE_REWARD_SAME_TOKEN="${UPDATE_REWARD_SAME_TOKEN:-0.10}"
UPDATE_REWARD_FIELD="${UPDATE_REWARD_FIELD:-0.15}"
UPDATE_REWARD_SAME_FIELD_TYPE="${UPDATE_REWARD_SAME_FIELD_TYPE:-0.35}"
UPDATE_REWARD_POSITIVE_THRESHOLD="${UPDATE_REWARD_POSITIVE_THRESHOLD:-0.5}"

POLICY_EPSILON_START="${POLICY_EPSILON_START:-0.2}"
POLICY_EPSILON_END="${POLICY_EPSILON_END:-0.02}"
POLICY_EPSILON_DECAY="${POLICY_EPSILON_DECAY:-0.8}"
POLICY_EXPLORE_TOP_M="${POLICY_EXPLORE_TOP_M:-5}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}" "${STATUS_FILE}" "${RESULTS_MD}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/plotly_update_reward_then_combo.latest.log"
ln -sfn "${STATUS_FILE}" "${LOG_DIR}/plotly_update_reward_then_combo.latest.status"

exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
  echo "$1" | tee -a "${STATUS_FILE}"
}

append_results() {
  echo "$1" | tee -a "${RESULTS_MD}"
}

append_result_block() {
  local title="$1"
  local phase_log="$2"
  local eval_dir="$3"

  append_results ""
  append_results "### ${title}"
  append_results ""
  append_results "- finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  append_results "- eval_log_dir: ${eval_dir}"
  append_results ""
  append_results '```text'
  grep -E "Complete recall info|Complete R@|R@1=|R@3=|R@5=|R@10=" "${phase_log}" | tail -60 | tee -a "${RESULTS_MD}" || true
  append_results '```'
}

on_error() {
  local exit_code=$?
  write_status "status=failed"
  write_status "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_status "failed_line=${BASH_LINENO[0]}"
  write_status "exit_code=${exit_code}"
  append_results ""
  append_results "### update_reward pipeline failed"
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
write_status "eval_nprocs=${EVAL_NPROCS}"
write_status "sft_ckpt=${SFT_CKPT}"
write_status "log_file=${LOG_FILE}"
write_status "status_file=${STATUS_FILE}"
write_status "results_md=${RESULTS_MD}"

append_results ""
append_results "## Plotly Update Reward Experiments ${RUN_ID}"
append_results ""
append_results "- started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results "- gpu_ids: ${GPU_IDS}"
append_results "- sft_ckpt: ${SFT_CKPT}"
append_results "- reward: exact=${UPDATE_REWARD_EXACT}, default=${UPDATE_REWARD_DEFAULT}, same_token=${UPDATE_REWARD_SAME_TOKEN}, field=${UPDATE_REWARD_FIELD}, same_field_type=${UPDATE_REWARD_SAME_FIELD_TYPE}"
append_results "- combo_policy: epsilon_start=${POLICY_EPSILON_START}, epsilon_end=${POLICY_EPSILON_END}, epsilon_decay=${POLICY_EPSILON_DECAY}, explore_top_m=${POLICY_EXPLORE_TOP_M}"
append_results "- log_file: ${LOG_FILE}"

test -f "${SFT_CKPT}"
cd "${CODE_DIR}"

REWARD_MARKER="${LOG_DIR}/plotly_update_reward_only_${RUN_ID}.marker"
touch "${REWARD_MARKER}"
write_status "status=update_reward_only_rl_running"
echo "[RL:update_reward_only] existing SFT checkpoint, dense reward, original greedy policy"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT_REWARD}" \
  --module reinforce.update_reward_learn_dist \
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
  --update_reward_exact="${UPDATE_REWARD_EXACT}" \
  --update_reward_default="${UPDATE_REWARD_DEFAULT}" \
  --update_reward_same_token="${UPDATE_REWARD_SAME_TOKEN}" \
  --update_reward_field="${UPDATE_REWARD_FIELD}" \
  --update_reward_same_field_type="${UPDATE_REWARD_SAME_FIELD_TYPE}" \
  --update_reward_positive_threshold="${UPDATE_REWARD_POSITIVE_THRESHOLD}"

REWARD_RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${REWARD_MARKER}" -name '*2el192fd128.128GRUh-allCharts-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
REWARD_RL_CKPT="${REWARD_RL_DIR}/states_ep0.pt"
write_status "update_reward_only_rl_dir=${REWARD_RL_DIR}"
write_status "update_reward_only_rl_ckpt=${REWARD_RL_CKPT}"
test -f "${REWARD_RL_CKPT}"

write_status "status=update_reward_only_eval_running"
echo "[EVAL:update_reward_only] deterministic original greedy search"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
  -m "${REWARD_RL_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path "evaluations/test-update-reward-only-plotly-small-${RUN_ID}" \
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

append_result_block "Update Reward Only, Original Greedy Policy" "${LOG_FILE}" "${REWARD_RL_DIR}/evaluations/test-update-reward-only-plotly-small-${RUN_ID}"

COMBO_MARKER="${LOG_DIR}/plotly_update_reward_policy_${RUN_ID}.marker"
touch "${COMBO_MARKER}"
write_status "status=update_reward_policy_rl_running"
echo "[RL:update_reward_policy] existing SFT checkpoint, dense reward, epsilon top-M policy update"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -m torch.distributed.launch \
  --nproc_per_node="${RL_NPROCS}" \
  --master_port="${MASTER_PORT_COMBO}" \
  --module reinforce.update_reward_policy_learn_dist \
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
  --update_reward_exact="${UPDATE_REWARD_EXACT}" \
  --update_reward_default="${UPDATE_REWARD_DEFAULT}" \
  --update_reward_same_token="${UPDATE_REWARD_SAME_TOKEN}" \
  --update_reward_field="${UPDATE_REWARD_FIELD}" \
  --update_reward_same_field_type="${UPDATE_REWARD_SAME_FIELD_TYPE}" \
  --update_reward_positive_threshold="${UPDATE_REWARD_POSITIVE_THRESHOLD}" \
  --policy_epsilon_start="${POLICY_EPSILON_START}" \
  --policy_epsilon_end="${POLICY_EPSILON_END}" \
  --policy_epsilon_decay="${POLICY_EPSILON_DECAY}" \
  --policy_explore_top_m="${POLICY_EXPLORE_TOP_M}"

COMBO_RL_DIR="$(find "${ROOT}/Results/Models" -maxdepth 1 -type d -newer "${COMBO_MARKER}" -name '*2el192fd128.128GRUh-allCharts-RL' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
COMBO_RL_CKPT="${COMBO_RL_DIR}/states_ep0.pt"
write_status "update_reward_policy_rl_dir=${COMBO_RL_DIR}"
write_status "update_reward_policy_rl_ckpt=${COMBO_RL_CKPT}"
test -f "${COMBO_RL_CKPT}"

write_status "status=update_reward_policy_eval_running"
echo "[EVAL:update_reward_policy] deterministic original greedy search"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" test_agent_mp.py \
  -m "${COMBO_RL_DIR}" \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path "evaluations/test-update-reward-policy-plotly-small-${RUN_ID}" \
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

append_result_block "Update Reward + Updated Policy, Greedy Evaluation" "${LOG_FILE}" "${COMBO_RL_DIR}/evaluations/test-update-reward-policy-plotly-small-${RUN_ID}"

write_status "status=finished"
write_status "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
append_results ""
append_results "- all_finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
