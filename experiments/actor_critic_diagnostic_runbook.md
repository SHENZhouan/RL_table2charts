# Actor-Critic Diagnostic Runbook

This document explains how to run the actor-critic scoring diagnostic experiment when the local machine does not already contain the trained actor-critic checkpoint.

The key point is:

- you do **not** need to retrain the model;
- you do **not** have to move the checkpoint into a hard-coded `/ssd/...` path;
- you only need a valid actor-critic checkpoint file such as `states_ep0.pt`, then point the eval commands at it.

## What This Experiment Is

This experiment evaluates the **same trained actor-critic checkpoint** in three eval-only score modes:

- `actor`
- `critic`
- `blend`

The model weights do not change. Only the inference-time ranking score changes.

That means this experiment is much cheaper than re-training and is suitable as a diagnostic comparison.

## Required Files

You need:

1. This repository.
2. The Plotly dataset.
3. One trained actor-critic checkpoint file, typically:

```text
states_ep0.pt
```

4. The checkpoint's parent model directory.

`update_actor_test_agent_mp.py` expects:

- `-m MODEL_DIR`
- `-f CHECKPOINT_FILENAME`

So if the file is:

```text
/path/to/actor_critic_run/states_ep0.pt
```

then:

- `MODEL_DIR=/path/to/actor_critic_run`
- `CHECKPOINT_FILENAME=states_ep0.pt`

## Recommended Ways To Provide The Checkpoint

There are two supported ways.

### Option A: Do Not Move The Checkpoint

This is the safest option.

If the checkpoint already exists somewhere on the remote server, keep it there and export:

```bash
ACTOR_CRITIC_CKPT=/actual/path/to/states_ep0.pt
```

The experiment runner will use that path directly.

This is the preferred method because it avoids duplicating large files and avoids fake assumptions about machine-specific folder layouts.

### Option B: Copy Or Move The Checkpoint Into A Project-Local Results Folder

If your teammate wants everything under this repository, they can place the checkpoint under a project-local model directory, for example:

```text
Results/Models/actor_critic_checkpoint/states_ep0.pt
```

Then export:

```bash
ACTOR_CRITIC_CKPT=$PWD/Results/Models/actor_critic_checkpoint/states_ep0.pt
```

This is fine, but it is not required.

## How To Check Whether The Checkpoint Exists

From the repository root:

```bash
find "$PWD" -name states_ep0.pt
```

If the checkpoint is somewhere else on the machine:

```bash
find /path/to/search/root -name states_ep0.pt | grep update_actor
```

If you already know the exact path, verify it directly:

```bash
test -f /actual/path/to/states_ep0.pt && echo ok
```

## Required Environment Variables

From the repository root:

```bash
export ROOT="$PWD"
export PYTHON_BIN=python
export CORPUS_PATH="$ROOT/Data/PlotlyTable2Charts"
export ACTOR_CRITIC_CKPT="/actual/path/to/states_ep0.pt"
```

If your Python is in a virtual environment, use that instead:

```bash
export PYTHON_BIN="$ROOT/.venv/bin/python"
```

If you use Conda:

```bash
conda activate t2c
export PYTHON_BIN=python
```

## Step 1: Dry-Run The Three Diagnostics

These commands do not run evaluation. They only print the exact command plan.

### Actor

```bash
$PYTHON_BIN experiments/scripts/run_experiments.py \
  --config experiments/configs/actor_critic_actor_score.json \
  --dry-run
```

### Critic

```bash
$PYTHON_BIN experiments/scripts/run_experiments.py \
  --config experiments/configs/actor_critic_critic_score.json \
  --dry-run
```

### Blend

```bash
$PYTHON_BIN experiments/scripts/run_experiments.py \
  --config experiments/configs/actor_critic_blend_score.json \
  --dry-run
```

If the runner says `planned_not_implemented`, then one of these is wrong:

- `ACTOR_CRITIC_CKPT` was not exported
- the checkpoint path does not exist
- the config was changed unexpectedly

## Step 2: Run A Small Local Smoke Test

Use this only for correctness checking. Do not use these settings for final throughput numbers.

Recommended single-GPU local settings:

- `nprocs=1`
- `nagents=8`
- `nthreads=2`

### Actor Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" Table2Charts/update_actor_test_agent_mp.py \
  -m "$(dirname "$ACTOR_CRITIC_CKPT")" \
  -f "$(basename "$ACTOR_CRITIC_CKPT")" \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/local_smoke_actor \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 1 \
  --nagents 8 \
  --nthreads 2 \
  --search_limits e50-b4-na \
  --corpus_path "$CORPUS_PATH" \
  --lang en \
  --limit_search_group \
  --score_mode actor
```

### Critic Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" Table2Charts/update_actor_test_agent_mp.py \
  -m "$(dirname "$ACTOR_CRITIC_CKPT")" \
  -f "$(basename "$ACTOR_CRITIC_CKPT")" \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/local_smoke_critic \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 1 \
  --nagents 8 \
  --nthreads 2 \
  --search_limits e50-b4-na \
  --corpus_path "$CORPUS_PATH" \
  --lang en \
  --limit_search_group \
  --score_mode critic \
  --critic_score_weight 1.0
```

### Blend Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" Table2Charts/update_actor_test_agent_mp.py \
  -m "$(dirname "$ACTOR_CRITIC_CKPT")" \
  -f "$(basename "$ACTOR_CRITIC_CKPT")" \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/local_smoke_blend \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 1 \
  --nagents 8 \
  --nthreads 2 \
  --search_limits e50-b4-na \
  --corpus_path "$CORPUS_PATH" \
  --lang en \
  --limit_search_group \
  --score_mode blend \
  --critic_score_weight 0.5
```

## Step 3: Run The Full Diagnostic On The Remote Server

After the smoke test is stable, use the runner dry-run output as the source of truth and run one mode at a time on the remote GPU host.

The current config names map to the metrics experiment names exactly:

- `actor_critic_actor_score`
- `actor_critic_critic_score`
- `actor_critic_blend_score`

Keep those names unchanged when logging results so the final comparison table stays clean.

## Common Failure Cases

### 1. `planned_not_implemented` In Dry-Run

Cause:

- `ACTOR_CRITIC_CKPT` is missing

Fix:

```bash
export ACTOR_CRITIC_CKPT=/actual/path/to/states_ep0.pt
```

### 2. Checkpoint Load Failure

Cause:

- wrong `MODEL_DIR`
- wrong filename
- wrong checkpoint type

Fix:

Check:

```bash
dirname "$ACTOR_CRITIC_CKPT"
basename "$ACTOR_CRITIC_CKPT"
test -f "$ACTOR_CRITIC_CKPT" && echo ok
```

### 3. Local GPU Memory Pressure

Cause:

- too many eval processes on one GPU

Fix:

Reduce:

- `--nprocs`
- `--nagents`
- `--nthreads`

Use the smoke-test settings first.

## Recommendation

For this experiment, do not invent a fake project-local checkpoint path unless you need one for organization.

The simplest reliable workflow is:

1. find the real `states_ep0.pt`
2. export `ACTOR_CRITIC_CKPT`
3. run dry-run
4. run small smoke test
5. run the full eval remotely
