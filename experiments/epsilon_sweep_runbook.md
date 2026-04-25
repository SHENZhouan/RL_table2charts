# Epsilon Sweep Runbook

## Summary

This document is the detailed operator record for the updated-policy epsilon sweep on the regenerated PlotlyTable2Charts corpus.

The key correction is procedural: the original experiment scaffold could finish RL training without running the formal post-training `test_agent_mp.py` evaluation. That meant training-time `EP-0 test/valid SUMMARY` lines could be mistaken for final report metrics. The corrected pipeline defines completion as:

1. RL training finishes
2. the fresh RL model directory is discovered
3. `test_agent_mp.py` runs on the test split
4. the resulting `[test-summary]` log is recorded and extracted into CSV

Report metrics must come from the `[test-summary]` logs written by `test_agent_mp.py`.

## Experiment Matrix

Sweep settings:

- method: `updated_policy`
- search limits: `e50-b4-na`
- epsilon start: `{0.05, 0.10, 0.20, 0.30}`
- epsilon end: `0.02`
- epsilon decay: `0.8`
- top-M: `5`
- model init: `Results/Models/sft_states_ep0.pt`
- epochs: `1`
- model size: `small`
- features: `all-fast`
- search/input/previous type: `allCharts`

The successful sweep used the processed corpus at:

```text
Data/PlotlyTable2Charts
```

Training logs showed:

- `load_at_most=None`
- `num_train_analysis=None`
- `train_ratio=0.7`
- `valid_ratio=0.1`
- `test_ratio=0.2`
- `Total schemas is 72939`
- `Total tables is 69037`

This was a full processed-corpus run, not a smoke subset.

## Scripts Involved

Training/eval entrypoints:

- `experiments/scripts/run_experiments.py`
- `experiments/scripts/run_remote_epsilon_sweep.sh`
- `Table2Charts/reinforce/updated_policy_learn_dist.py`
- `Table2Charts/test_agent_mp.py`
- `experiments/scripts/extract_test_summary.py`

Roles:

- `run_experiments.py` generates the training command shape
- `run_remote_epsilon_sweep.sh` is the helper that runs training, discovers the new RL model dir, and launches `test_agent_mp.py`
- `updated_policy_learn_dist.py` performs RL training
- `test_agent_mp.py` writes the formal `[test-summary]...log` final-eval artifact under the trained RL model directory
- `extract_test_summary.py` normalizes those `[test-summary]` logs into CSV rows

## Actual Artifacts Produced

### RL model directories

Tracked in:

- [epsilon_sweep_model_dirs_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/epsilon_sweep_model_dirs_20260425.csv)

Authoritative model-dir mapping:

- `epsilon_eps005_top5` -> `Results/Models/20260425104442-2el192fd128.128GRUh-allCharts-RL`
- `epsilon_eps010_top5` -> `Results/Models/20260425110039-2el192fd128.128GRUh-allCharts-RL`
- `epsilon_eps020_top5` -> `Results/Models/20260425111651-2el192fd128.128GRUh-allCharts-RL`
- `epsilon_eps030_top5` -> `Results/Models/20260425113303-2el192fd128.128GRUh-allCharts-RL`

There is also an older duplicate `eps005` model directory:

- `Results/Models/20260425104458-2el192fd128.128GRUh-allCharts-RL`

Do not use that duplicate as the authoritative `eps005` result source unless you intentionally re-verify it. The tracked mapping file points to `20260425104442...`, which is the authoritative row for the current epsilon sweep.

### Formal final-eval summary logs

These logs are the formal `test_agent_mp.py` outputs and are the provenance source for report metrics:

- `Results/Models/20260425104442-2el192fd128.128GRUh-allCharts-RL/evaluations/test-epsilon-eps005-top5-20260425/[test-summary]20260425T1821.log`
- `Results/Models/20260425110039-2el192fd128.128GRUh-allCharts-RL/evaluations/test-epsilon-eps010-top5-20260425/[test-summary]20260425T1826.log`
- `Results/Models/20260425111651-2el192fd128.128GRUh-allCharts-RL/evaluations/test-epsilon-eps020-top5-20260425/[test-summary]20260425T1832.log`
- `Results/Models/20260425113303-2el192fd128.128GRUh-allCharts-RL/evaluations/test-epsilon-eps030-top5-20260425/[test-summary]20260425T1838.log`

These files are JSON-style merged summaries written by `Table2Charts/test_agent_mp.py`.

## Runtime And Launch Shapes

### Historical smoke test

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

### Correct full helper entrypoint

Use the helper script:

```text
experiments/scripts/run_remote_epsilon_sweep.sh
```

Current helper behavior in non-dry-run mode:

1. create a marker file before training
2. launch RL training
3. discover the newly produced RL model directory
4. assert `states_ep0.pt` exists
5. run `test_agent_mp.py`
6. record the config name, model dir, checkpoint, eval log dir, and completion time in a sidecar `.results` file

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

### Machine-specific variables

Do not hard-code these into the helper:

- `ROOT`
- `PYTHON_BIN`
- `CORPUS_PATH`
- `SFT_CKPT`
- `MODEL_SAVE_PATH`
- `SUMMARY_PATH`
- `GPU_IDS`
- `RL_NPROCS`
- `EVAL_NPROCS`
- `MASTER_PORT`

Rules:

- `GPU_IDS` must match the actual visible GPUs
- `RL_NPROCS` must match the number of visible GPUs used for training
- `EVAL_NPROCS` must match the intended evaluation parallelism
- keep `queue_mode=local` for these single-node remote runs unless you intentionally switch to a queue-backed setup

## Extracting Formal Final-Eval Metrics

The extractor is:

- [extract_test_summary.py](/home/lyl610/RL_table2charts/experiments/scripts/extract_test_summary.py)

By default it reads `experiments/results/epsilon_sweep_model_dirs_20260425.csv`, discovers one `[test-summary]` log under each authoritative model directory, and writes normalized rows.

Preview rows to stdout:

```bash
python experiments/scripts/extract_test_summary.py
```

Overwrite the normalized epsilon final-eval CSV:

```bash
python experiments/scripts/extract_test_summary.py \
  --output experiments/results/final_eval_epsilon_sweep_20260425.csv \
  --overwrite
```

Explicit-path mode is also supported if you want to pass `[test-summary]` files directly:

```bash
python experiments/scripts/extract_test_summary.py \
  Results/Models/20260425104442-2el192fd128.128GRUh-allCharts-RL/evaluations/test-epsilon-eps005-top5-20260425/[test-summary]20260425T1821.log
```

The extractor only supports formal `test_agent_mp.py` outputs. It must not be pointed at training logs or `EP-0 test/valid SUMMARY` text.

## Current Result Files

Training model directories:

- [epsilon_sweep_model_dirs_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/epsilon_sweep_model_dirs_20260425.csv)

Normalized final-eval table:

- [final_eval_epsilon_sweep_20260425.csv](/home/lyl610/RL_table2charts/experiments/results/final_eval_epsilon_sweep_20260425.csv)

Main sweep raw log:

- [epsilon_sweep_20260425T024412Z.log](/home/lyl610/RL_table2charts/experiments/results/raw_logs/epsilon_sweep_20260425T024412Z.log)

## Metric Provenance Warning

Do not use training-time `EP-0 test/valid SUMMARY` lines as report metrics.

Those summaries are useful for monitoring training, but they are not the formal post-training test-set evaluation. Formal report metrics must come from:

1. `[test-summary]...log` files under `Results/Models/.../evaluations/...`
2. the normalized CSV generated from those logs

## Observed Runtime

From `experiments/results/raw_logs/epsilon_sweep_20260425T024412Z.log`:

- `epsilon=0.05`: about 16.3 min
- `epsilon=0.10`: about 16.1 min
- `epsilon=0.20`: about 16.1 min
- `epsilon=0.30`: about 16.0 min

Practical estimate:

- one epsilon run: about 16 min of RL training
- four-run sequential sweep: about 65 min of RL training
- formal `test_agent_mp.py` evaluation is extra wall-clock time beyond that

## Common Failure Modes

### Missing processed corpus

If training fails with missing `index/schema_ids.json`, `CORPUS_PATH` does not point to a processed Table2Charts corpus.

### Duplicate GPU / NCCL invalid usage

If logs show duplicate GPU detection, `GPU_IDS` and `RL_NPROCS` do not match the actual visible devices.

### Script permissions

If the helper says permission denied, run:

```bash
bash experiments/scripts/run_remote_epsilon_sweep.sh
```

or add execute permission once and keep that change intentional.

### Dry-run confusion

To preview commands without training or test:

```bash
DRY_RUN=1 bash experiments/scripts/run_remote_epsilon_sweep.sh
```

### Empty or truncated summary log

If `extract_test_summary.py` fails on an empty or truncated `[test-summary]` file, rerun only the final-eval step for that specific RL model directory instead of retraining the whole sweep.
