# Reward Interaction Runbook

## Scope

This runbook prepares the dense reward plus epsilon interaction stage on the Plotly-only Table2Charts setup.

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

## 2x2 Design

The main interaction design is:

1. `hard reward + greedy`
2. `hard reward + epsilon=0.20`
3. `current soft reward + greedy`
4. `current soft reward + epsilon=0.20`

Interpretation for this repo:

- `hard reward + greedy`:
  baseline greedy run, already covered by the existing baseline/SFT comparison setup
- `hard reward + epsilon=0.20`:
  already available from the current epsilon sweep at `epsilon=0.20`
- `current soft reward + greedy`:
  `experiments/configs/reward_current_greedy.json`
- `current soft reward + epsilon=0.20`:
  `experiments/configs/reward_current_epsilon.json`

Optional comparison:

- `reward_conservative_greedy.json` is a conservative soft-reward variant for sensitivity analysis, but it is not part of the default 2x2 run.

## Existing Relevant Results

Already available without new training:

- hard reward + epsilon=0.20:
  [final_eval_epsilon_sweep_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_epsilon_sweep_20260425.csv)

The dense-reward interaction stage therefore only needs to add:

- current soft reward + greedy
- current soft reward + epsilon=0.20

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

Preview the two reward interaction runs:

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
bash experiments/scripts/run_remote_reward_interaction.sh
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
bash experiments/scripts/run_remote_reward_interaction.sh
```

For `tmux`:

```bash
tmux new-session -d -s reward_interaction \
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
   bash experiments/scripts/run_remote_reward_interaction.sh"
```

## Files Produced

Use these result tables for this stage:

- [reward_interaction_model_dirs_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/reward_interaction_model_dirs_20260425.csv)
- [final_eval_reward_interaction_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_reward_interaction_20260425.csv)

Fill these after the actual remote runs complete.

## Common Failure Modes

- missing processed corpus under `Data/PlotlyTable2Charts`
- `RL_NPROCS` not matching visible GPU count
- reusing an occupied `MASTER_PORT`
- forgetting that `hard reward + epsilon=0.20` is already available from the epsilon sweep and rerunning it unnecessarily
