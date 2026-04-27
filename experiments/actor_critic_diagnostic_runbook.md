# Actor-Critic Train + Diagnostic Runbook

This runbook explains how to train the actor-critic model locally or on a GPU
host, then evaluate the same checkpoint in three score modes:

- `actor`
- `critic`
- `blend`

The training code is the `update_actor_new` actor-critic chain from
`origin/actor_try`, adapted to the current `experiments/` runner format.

## 2026-04-27 Fix Notes and Insights

The first remote actor-critic run was abnormally slow. The log kept advancing
from `B492000` to `B495000` after more than 13 hours, while GPU memory stayed
under 1GB. That behavior was not normal training progress through new Plotly
tables. It pointed to a training-loop bug rather than a GPU bottleneck.

Root cause:

- `update_actor_new_learn_dist.py` originally filled the active table queue
  while `student.n_tables() < max_tables`.
- `UpdateActorNewStudent.n_tables()` returned `remaining + finished`.
- After the first batch of tables finished, `finished` stayed high, so the
  loop stopped admitting new tables.
- The process then kept sampling from the replay memory of the first batch,
  which produced endless-looking `EP-0 train Bxxxxx` logs without completing
  the epoch over the intended corpus.

Fix implemented:

- `iteration()` now admits new tables while
  `student.agents.remaining() < args.max_tables`.
- The epoch stops only when the distributed finished-table count reaches
  `n_tables`.
- Memory readiness uses distributed `dist_min(...)` before sampling, matching
  the existing RL training loops.
- Progress logs now include:

```text
handled=<global_added>/<n_tables> finished=<global_finished>/<n_tables> active=<local_active>
```

Expected healthy actor-critic logs should periodically show `handled` and
`finished` increasing. If logs only show `EP-0 train Bxxxxx` for a long time
without progress lines, the run is not trustworthy.

The already-running old actor-critic job should be stopped and rerun with this
fix. Existing reward, epsilon, and hard-greedy experiments do not need retrain;
they only need eval-only reruns if per-table recommendation JSON is required.

## Single-GPU Remote Guidance

For a one-GPU rented host, keep both process counts at `1`:

```bash
export GPU_IDS=0
export RL_NPROCS=1
export EVAL_NPROCS=1
```

Do not run another GPU-heavy training job beside actor-critic unless there is
clear spare memory and the second job is easy to restart. Eval-only jobs are
safer than training jobs, but if an OOM kills the process, the current eval
must be rerun from the last complete checkpoint. Finished checkpoints are not
invalidated by an eval OOM.

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

It also writes per-table recommendation JSON for each eval mode under that
eval's `recommendations/` directory, and records `summary_log` plus
`recommend_log_dir` in the `.results` file.

Main logs are written under:

```text
experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.log
experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.results
```

Each eval summary is written under the discovered model directory, using the
configured eval log subdirectory.

During smoke, confirm both types of progress:

```bash
grep -E "handled=.*finished=.*active=" experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.log
grep -E "discovered_actor_critic_ckpt|recommend_log_dir" experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.results
```

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

After the eval summaries exist, extract the actor-critic final eval CSV from
the helper `.results` file:

```bash
python experiments/scripts/extract_actor_critic_eval.py --overwrite
```

By default this uses the newest:

```text
experiments/results/raw_logs/actor_critic_train_eval_*.results
```

and writes:

```text
experiments/results/final_eval_actor_critic_<RUN_ID>.csv
```

For an explicit run:

```bash
python experiments/scripts/extract_actor_critic_eval.py \
  --results experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.results \
  --output experiments/results/final_eval_actor_critic_<RUN_ID>.csv \
  --overwrite
```

Then use the existing result tools as needed:

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

## Recommendation Diversity Backfill

Old `[test-summary]` logs are enough for recall and aggregate search metrics,
but they are not enough to recover chart-type diversity or unique
recommendation counts. Those metrics require per-table `ranked_recommend`
JSON, so existing metrics rows must be evaluated again without retraining.

The eval-only backfill is strictly aligned to `metrics.csv`:

- each row reads `model_dir=...` and `log_path=...` from the row's `notes`;
- checkpoint is fixed to `<model_dir>/states_ep0.pt`;
- no latest-checkpoint discovery or substitution is allowed;
- missing checkpoints are recorded as `missing_checkpoint` and skipped.

Run a dry-run or first-row smoke:

```bash
DRY_RUN=1 LIMIT=1 \
  experiments/scripts/run_metrics_recommendation_eval.sh
```

Run the full eval-only backfill on the remote machine:

```bash
ROOT="$PWD" \
PYTHON_BIN=python \
CORPUS_PATH="$PWD/Data/PlotlyTable2Charts" \
GPU_IDS=0 \
EVAL_NPROCS=1 \
experiments/scripts/run_metrics_recommendation_eval.sh
```

The manifest is written to:

```text
experiments/results/recommendation_eval_manifest.csv
```

Then extract diversity and unique recommendation metrics:

```bash
python experiments/scripts/extract_recommendation_diversity.py \
  --manifest experiments/results/recommendation_eval_manifest.csv \
  --output experiments/results/recommendation_diversity.csv \
  --overwrite
```

Definitions:

- `chart_type` is the recommendation JSON `ANA` field.
- `unique_recommendation` is canonical JSON after removing only the top-level
  `score` field, so score-only differences do not create new unique
  recommendations.
- `table_count` should match the number of per-table recommendation JSON files.

Actor-critic evals produced by `run_remote_actor_critic_train_eval.sh` can also
be included directly from the `.results` file:

```bash
python experiments/scripts/extract_recommendation_diversity.py \
  --actor-critic-results experiments/results/raw_logs/actor_critic_train_eval_<RUN_ID>.results \
  --output experiments/results/actor_critic_recommendation_diversity_<RUN_ID>.csv \
  --overwrite
```

## Aggregate Search Metrics Insight

`experiments/results/search_aggregate_metrics.csv` is sourced only from the
exact `log_path=...` entries in `metrics.csv`. It reflects search behavior,
not recommendation diversity:

- `expanded_states` describes how much search work was actually expanded.
- `reached_states` describes how many states were reached before filtering.
- `dropped_states` and `drop_rate` describe how much search was discarded.
- `complete_states`, `complete_reached`, and `complete_targets` describe how
  many complete chart candidates were generated or reached.
- `search_efficiency = complete_states / expanded_states` measures candidate
  completion yield per expanded state.

These fields help explain whether a method changes the search frontier,
pruning/drop behavior, or candidate completion volume. They cannot recover
chart-type diversity or unique recommendation counts; those require the
recommendation backfill above.

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
