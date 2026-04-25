# Epsilon Sweep Runbook

## Scope

This runbook documents the updated-policy epsilon-greedy sweep that was run on the Plotly-only Table2Charts setup.

Sweep settings:

- method: `updated_policy`
- search limits: `e50-b4-na`
- epsilon start: `{0.05, 0.10, 0.20, 0.30}`
- epsilon end: `0.02`
- epsilon decay: `0.8`
- top-M: `5`
- model init: `Results/Models/sft_states_ep0.pt`
- epochs: `1`

## What Was Trained

The successful sweep used the processed corpus at:

```text
Data/PlotlyTable2Charts
```

The training logs show:

- `load_at_most=None`
- `num_train_analysis=None`
- `train_ratio=0.7`
- `valid_ratio=0.1`
- `test_ratio=0.2`
- `Total schemas is 72939`
- `Total tables is 69037`

That means this was not a smoke-subset run. It used the full processed Plotly corpus available under `Data/PlotlyTable2Charts`, then split it by the repo's default ratio:

- 70% train
- 10% valid
- 20% test

So the RL training stage used the full training split of the processed corpus, not a manually capped subset. It still trained for only one epoch.

## Observed Runtime

From `experiments/results/raw_logs/epsilon_sweep_20260425T024412Z.log`:

- `epsilon=0.05`: about 16.3 min
- `epsilon=0.10`: about 16.1 min
- `epsilon=0.20`: about 16.1 min
- `epsilon=0.30`: about 16.0 min

Practical estimate:

- one epsilon run: about 16 min
- four-run sequential sweep: about 65 min

This estimate covers the RL train+valid stage recorded in the sweep log. Final evaluation is extra.

## Smoke Test Command

This was the single-GPU remote smoke-test shape used before the full sweep:

```bash
cd ~/RL_table2charts

export ROOT="$PWD"
export PYTHON_BIN=python
export CORPUS_PATH="$PWD/Data/PlotlyTable2Charts"
export SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt"
export MODEL_SAVE_PATH="$PWD/Results/Models"
export SUMMARY_PATH="$PWD/Results/summary"
export GPU_IDS=0
export MASTER_PORT=29531

cd Table2Charts

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port 29531 \
  --module reinforce.updated_policy_learn_dist \
  --corpus_path "$CORPUS_PATH" \
  --model_size=small \
  --model_name=cp \
  --features=all-fast \
  --negative_weight=0.8 \
  --search_limits=e50-b4-na \
  --epochs=1 \
  --model_save_path "$MODEL_SAVE_PATH" \
  -p "$SFT_CKPT" \
  --summary_path "$SUMMARY_PATH" \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --lang=en \
  --queue_mode=local \
  --log_freq_agent=500 \
  --log_freq_batch=100 \
  --max_tables=64 \
  --min_memory=1000 \
  --memory_sample_size=64 \
  --memory_sample_rounds=2 \
  --policy_epsilon_start=0.05 \
  --policy_epsilon_end=0.02 \
  --policy_epsilon_decay=0.8 \
  --policy_explore_top_m=5
```

Use this only to verify startup and data/model wiring. It is not the recommended way to launch the full sweep.

## Recommended Sweep Entry Point

The tracked helper script is:

```text
experiments/scripts/run_remote_epsilon_sweep.sh
```

Use it with environment variables instead of editing machine-specific paths into the script.

Example remote launch:

```bash
cd /path/to/RL_table2charts

ROOT="$PWD" \
PYTHON_BIN=python \
CORPUS_PATH="$PWD/Data/PlotlyTable2Charts" \
SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt" \
MODEL_SAVE_PATH="$PWD/Results/Models" \
SUMMARY_PATH="$PWD/Results/summary" \
GPU_IDS=0,1 \
RL_NPROCS=2 \
EVAL_NPROCS=2 \
bash experiments/scripts/run_remote_epsilon_sweep.sh
```

For a `tmux` launch:

```bash
tmux new-session -d -s epsilon_sweep \
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
   bash experiments/scripts/run_remote_epsilon_sweep.sh"
```

## Variables You Must Adjust Per Machine

Do not hard-code these into the script for one machine:

- `ROOT`
- `PYTHON_BIN`
- `CORPUS_PATH`
- `SFT_CKPT`
- `MODEL_SAVE_PATH`
- `SUMMARY_PATH`
- `GPU_IDS`
- `RL_NPROCS`
- `EVAL_NPROCS`
- `MASTER_PORT` if you run a one-off direct launch

Rules:

- `GPU_IDS` must match the actual visible GPU list
- `RL_NPROCS` must match the number of visible GPUs used for training
- do not set `nproc_per_node=4` when only 2 GPUs are visible
- keep `queue_mode=local` for single-node smoke tests unless you intentionally use RabbitMQ

## Current Result Artifacts

Training model directories:

- [epsilon_sweep_model_dirs_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/epsilon_sweep_model_dirs_20260425.csv)

Final evaluation summary table:

- [final_eval_epsilon_sweep_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_epsilon_sweep_20260425.csv)

Main sweep raw log:

- [epsilon_sweep_20260425T024412Z.log](/home/lyl610/RL_table2charts/experiments/results/raw_logs/epsilon_sweep_20260425T024412Z.log)

## Current Final Eval Table

From `final_eval_epsilon_sweep_20260425.csv`:

- `epsilon=0.20` gives the best `R@1`, `R@3`, `R@5`, and `R@10`
- `epsilon=0.10` gives the best `R@20` and `recall_all`

This is the table currently stored in the repo, but the CSV should still be treated as a normalized summary. If exact recomputation is needed later, preserve the underlying final-eval raw logs alongside the CSV.

## Common Failure Modes

### Missing processed corpus

If training fails with missing `index/schema_ids.json`, the issue is not the runner. It means `CORPUS_PATH` does not point to a processed Table2Charts corpus.

### Duplicate GPU / NCCL invalid usage

If logs show duplicate GPU detection, `GPU_IDS` and `RL_NPROCS` do not match the actual visible devices.

### Script permissions

If the helper says permission denied, run:

```bash
bash experiments/scripts/run_remote_epsilon_sweep.sh
```

or add execute permission once and keep it tracked intentionally.

### Dry-run vs real run

To preview commands without training:

```bash
DRY_RUN=1 bash experiments/scripts/run_remote_epsilon_sweep.sh
```

## Recommendation for TA Reproduction

For TA-facing reproduction, keep this split:

- `experiments/README.md`: high-level map
- `experiments/epsilon_sweep_runbook.md`: concrete operational steps
- `notes/experiment_log.md`: what was actually run and what happened

That keeps the pipeline understandable without mixing historical notes into the main instructions.
