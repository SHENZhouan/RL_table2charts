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
- current remote machine target: `2x4090`
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

Next runs:

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

For the current 2x4090 server, always export:

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

When ready to run on the 2x4090 server:

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

## Files Produced

Use these result tables for this stage:

- [reward_intensity_model_dirs_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/reward_intensity_model_dirs_20260425.csv)
- [final_eval_reward_intensity_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_reward_intensity_20260425.csv)

Fill these after the actual remote runs complete.

## Common Failure Modes

- missing processed corpus under `Data/PlotlyTable2Charts`
- `RL_NPROCS` not matching visible GPU count
- reusing an occupied `MASTER_PORT`
- forgetting that `hard reward + epsilon=0.20` is already available from the epsilon sweep and rerunning it unnecessarily
- assuming `hard reward + greedy` is completed just because the config exists; that still requires a regenerated-corpus final eval artifact
