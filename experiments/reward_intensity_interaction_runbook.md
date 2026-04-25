# Reward Intensity Interaction Runbook

## Scope

This runbook prepares the reward-intensity × sampling interaction stage on the Plotly-only Table2Charts setup.

Fixed experiment settings:

- `CORPUS_PATH=$ROOT/Data/PlotlyTable2Charts`
- `SFT_CKPT=$ROOT/Results/Models/sft_states_ep0.pt`
- `search_limits=e50-b4-na`
- `model_size=small`
- `features=all-fast`
- `search_type=input_type=previous_type=allCharts`
- `limit_search_group=true`
- current remote machine target: `2 GPUs`
- `RL_NPROCS=2`
- `EVAL_NPROCS=2`

## Why The Old 2x2 Was Too Coarse

The previous coarse framing mostly tested reward on/off against sampling on/off. That was useful as a first pass, but it does not isolate whether the interaction appears only when dense reward becomes stronger.

The new question is more specific:

- do epsilon exploration and denser reward shaping interfere because both encourage near-positive or intermediate actions?

That requires comparing multiple reward intensities, not just hard vs one soft setting.

## Reward Intensity × Sampling Matrix

Core matrix:

1. `hard reward + greedy`
2. `hard reward + epsilon=0.20`
3. `conservative soft reward + greedy`
4. `conservative soft reward + epsilon=0.20`
5. `current soft reward + greedy`
6. `current soft reward + epsilon=0.20`

Optional extra matrix:

7. `aggressive soft reward + greedy`
8. `aggressive soft reward + epsilon=0.20`

## Existing Relevant Results

Completed:

- hard reward + epsilon=0.20:
  [final_eval_epsilon_sweep_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_epsilon_sweep_20260425.csv)

Represented by config but not automatically counted as completed on the regenerated corpus:

- hard reward + greedy:
  `experiments/configs/baseline_rl_greedy_train_eval.json`

This should be treated as **needs to run** unless a matching regenerated-corpus final evaluation result is present.

The reward helper had the same historical pipeline gap as the epsilon helper: older training runs could stop after RL training and leave only training-time `EP-0 test/valid SUMMARY` lines. Those summaries are provisional only. Formal report metrics must come from `test_agent_mp.py` `[test-summary]` logs.

Formal final-eval file now exists for the four completed soft-reward runs:

- [final_eval_reward_intensity_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_reward_intensity_20260425.csv)

Completed on regenerated corpus with formal final eval:

- `reward_conservative_greedy`
- `reward_conservative_epsilon`
- `reward_current_greedy`
- `reward_current_epsilon`

Still needs to run:

- `hard + greedy` unless a regenerated-corpus final eval artifact is available
- `reward_aggressive_greedy`
- `reward_aggressive_epsilon`

Historical next-run list before the formal eval catch-up was:

- `reward_conservative_greedy`
- `reward_conservative_epsilon`
- `reward_current_greedy`
- `reward_current_epsilon`

Optional runs:

- `reward_aggressive_greedy`
- `reward_aggressive_epsilon`

## Runtime Override Rule

Configs may still contain older default process counts, but the runner now resolves runtime values from environment variables first:

- `RL_NPROCS` overrides `config["runtime"]["rl_nprocs"]`
- `EVAL_NPROCS` overrides `config["runtime"]["eval_nprocs"]`
- `MASTER_PORT` overrides `config["runtime"]["master_port"]`

For the current 2-GPU server, always export:

```bash
RL_NPROCS=2
EVAL_NPROCS=2
```

## Dry-Run

Preview the reward-intensity stage:

```bash
ROOT="$PWD" \
PYTHON_BIN=python \
CORPUS_PATH="$PWD/Data/PlotlyTable2Charts" \
SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt" \
MODEL_SAVE_PATH="$PWD/Results/Models" \
SUMMARY_PATH="$PWD/Results/summary" \
GPU_IDS=0,1 \
RL_NPROCS=2 \
EVAL_NPROCS=2 \
MASTER_PORT=29649 \
DRY_RUN=1 \
bash experiments/scripts/run_remote_reward_intensity_sweep.sh
```

## Default Remote Launch

When ready to run on the current 2-GPU server:

```bash
ROOT="$PWD" \
PYTHON_BIN=python \
CORPUS_PATH="$PWD/Data/PlotlyTable2Charts" \
SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt" \
MODEL_SAVE_PATH="$PWD/Results/Models" \
SUMMARY_PATH="$PWD/Results/summary" \
GPU_IDS=0,1 \
RL_NPROCS=2 \
EVAL_NPROCS=2 \
MASTER_PORT=29649 \
bash experiments/scripts/run_remote_reward_intensity_sweep.sh
```

For `tmux`:

```bash
tmux new-session -d -s reward_intensity \
  "cd /path/to/RL_table2charts && \
   ROOT=\$PWD \
   PYTHON_BIN=python \
   CORPUS_PATH=\$PWD/Data/PlotlyTable2Charts \
   SFT_CKPT=\$PWD/Results/Models/sft_states_ep0.pt \
   MODEL_SAVE_PATH=\$PWD/Results/Models \
   SUMMARY_PATH=\$PWD/Results/summary \
   GPU_IDS=0,1 \
   RL_NPROCS=2 \
   EVAL_NPROCS=2 \
   MASTER_PORT=29649 \
   bash experiments/scripts/run_remote_reward_intensity_sweep.sh"
```

In current helper behavior, one config is only considered fully completed after:

1. RL training finishes
2. the fresh RL model directory is discovered
3. `test_agent_mp.py` runs on the test split
4. the helper records the model dir and eval log dir

## Files Produced

Use these result tables for this stage:

- [reward_intensity_model_dirs_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/reward_intensity_model_dirs_20260425.csv)
- [final_eval_reward_intensity_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_reward_intensity_20260425.csv)

`reward_intensity_model_dirs_20260425.csv` is for discovered RL output directories.

`final_eval_reward_intensity_20260425.csv` is only for metrics produced by `test_agent_mp.py` final evaluation. Do not backfill this file from training-time `test/valid SUMMARY` lines.

The extractor interface is shared with epsilon:

```bash
python experiments/scripts/extract_test_summary.py \
  --family reward_intensity \
  --output experiments/results/final_eval_reward_intensity_20260425.csv \
  --overwrite
```

## Common Failure Modes

- missing processed corpus under `Data/PlotlyTable2Charts`
- `RL_NPROCS` not matching visible GPU count
- reusing an occupied `MASTER_PORT`
- on lower-memory GPUs such as `2080`, reduce concurrency or training-state size if you hit OOM:
  - first try `RL_NPROCS=2` with `GPU_IDS=0,1`
  - if that still fails, lower `max_tables` and `memory_sample_size`, or fall back to `RL_NPROCS=1`
- forgetting that `hard reward + epsilon=0.20` is already available from the epsilon sweep and rerunning it unnecessarily
- assuming `hard reward + greedy` is completed just because the config exists; that still requires a regenerated-corpus final eval artifact
- treating training-time `EP-0 test/valid SUMMARY` as if it were the formal final test-set result
