# Actor-Critic Train + Diagnostic Runbook

This runbook explains how to train the actor-critic model locally or on a GPU
host, then evaluate the same checkpoint in three score modes:

- `actor`
- `critic`
- `blend`

The training code is the `update_actor_new` actor-critic chain from
`origin/actor_try`, adapted to the current `experiments/` runner format.

## Current Local State

This repository now contains the actor-critic training scripts, but this local
machine may still not contain the large runtime inputs:

- Plotly corpus: default `Data/PlotlyTable2Charts`
- SFT warm-start checkpoint: a completed `states_ep0.pt`

Before a real run, provide them on the machine that will run the experiment.

## Required Environment

From the repository root:

```bash
export ROOT="$PWD"
export PYTHON_BIN=python
export CORPUS_PATH="$ROOT/Data/PlotlyTable2Charts"
export SFT_CKPT="/actual/path/to/sft/states_ep0.pt"
export MODEL_SAVE_PATH="$ROOT/Results/Models"
export SUMMARY_PATH="$ROOT/Results/summary"
export GPU_IDS=0
export RL_NPROCS=1
export EVAL_NPROCS=1
export MASTER_PORT=29653
```

If using a virtualenv or conda env, set `PYTHON_BIN` to that Python.

Check required inputs:

```bash
test -d "$CORPUS_PATH" && echo corpus-ok
test -f "$SFT_CKPT" && echo sft-ok
```

## Dry Run

Dry-run prints the training command, the checkpoint discovery rule, and the
three planned diagnostic eval commands. It does not require the placeholder SFT
path to exist.

```bash
SFT_CKPT=/path/to/states_ep0.pt DRY_RUN=1 \
  experiments/scripts/run_remote_actor_critic_train_eval.sh
```

You can also inspect only the train command:

```bash
SFT_CKPT=/path/to/states_ep0.pt \
  "$PYTHON_BIN" experiments/scripts/run_experiments.py \
  --config experiments/configs/actor_critic_train_eval.json \
  --dry-run
```

## Small Smoke Run

Use one GPU first:

```bash
GPU_IDS=0 RL_NPROCS=1 EVAL_NPROCS=1 \
  experiments/scripts/run_remote_actor_critic_train_eval.sh
```

The helper will:

1. train with `reinforce.update_actor_new_learn_dist`;
2. find the newest model directory matching
   `*update_actor_new-*allCharts-actor-policy-RL`;
3. verify `<model_dir>/states_ep0.pt`;
4. export `ACTOR_CRITIC_CKPT` to that checkpoint;
5. run:
   - `experiments/configs/actor_critic_actor_score.json`
   - `experiments/configs/actor_critic_critic_score.json`
   - `experiments/configs/actor_critic_blend_score.json`

Main logs are written under:

```text
experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.log
experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.results
```

Each eval summary is written under the discovered model directory, using the
configured eval log subdirectory.

## Full Remote Run

On the GPU host, choose GPU and process counts for the machine:

```bash
export GPU_IDS=0,1,2,3
export RL_NPROCS=4
export EVAL_NPROCS=4
export SFT_CKPT=/actual/path/to/sft/states_ep0.pt
export CORPUS_PATH=/actual/path/to/Data/PlotlyTable2Charts

experiments/scripts/run_remote_actor_critic_train_eval.sh
```

The produced actor-critic checkpoint can be reused later:

```bash
export ACTOR_CRITIC_CKPT=/actual/path/to/update_actor_new_run/states_ep0.pt
```

Then run any diagnostic directly:

```bash
"$PYTHON_BIN" experiments/scripts/run_experiments.py \
  --config experiments/configs/actor_critic_actor_score.json
```

## Result Collection

After the eval summaries exist, use the existing result tools:

```bash
"$PYTHON_BIN" experiments/scripts/collect_results.py
"$PYTHON_BIN" experiments/scripts/summarize_results.py
```

Keep the diagnostic config names unchanged:

- `actor_critic_actor_score`
- `actor_critic_critic_score`
- `actor_critic_blend_score`

That keeps the final metrics table comparable with the existing experiment
rows.

## Common Failures

### `SFT_CKPT must point to...`

`SFT_CKPT` was not exported. Set it to a real SFT `states_ep0.pt`.

### `test -d "$CORPUS_PATH"` Fails

The Plotly corpus is missing or the path is wrong. Set `CORPUS_PATH` to the
actual corpus directory on the run host.

### Checkpoint Discovery Fails

The helper looks for:

```text
*update_actor_new-*allCharts-actor-policy-RL/states_ep0.pt
```

If training completed but discovery failed, inspect `MODEL_SAVE_PATH` and the
main run log for the saved checkpoint path.

### GPU Memory Pressure

Reduce:

- `RL_NPROCS`
- `EVAL_NPROCS`

For smoke tests, keep both at `1`.
