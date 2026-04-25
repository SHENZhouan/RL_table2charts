# Experiments Scaffold

## Purpose

This directory provides a reproducible experiment-management layer for the Plotly-only Table2Charts adaptation.

It is intended to:

- document experiment families and assumptions;
- store machine-independent config files;
- generate dry-run command plans before remote execution;
- normalize result extraction into one CSV;
- support lightweight notebook analysis without embedding training code inline.

It is not intended to replace the existing training or evaluation code in `Table2Charts/` during this first step.

## Structure

- `configs/`: JSON configs for baseline, updated-policy, reward, and actor-critic diagnostics
- `scripts/run_experiments.py`: dry-run-first orchestration layer
- `scripts/collect_results.py`: normalize existing results into `metrics.csv`
- `results/metrics.csv`: normalized experiment table
- `results/raw_logs/`: optional future output location for runner-managed logs

## Baseline Split

Two baselines are tracked separately on purpose:

- `baseline_sft_greedy_eval.json`: SFT checkpoint plus original greedy evaluation only
- `baseline_rl_greedy_train_eval.json`: RL fine-tuned original greedy training plus evaluation

The first one is the fair comparison baseline for most planned studies.

## Environment Variables

The runner resolves missing path and runtime values from environment variables when possible:

- `ROOT`
- `CORPUS_PATH`
- `SFT_CKPT`
- `MODEL_SAVE_PATH`
- `SUMMARY_PATH`
- `GPU_IDS`
- `PYTHON_BIN`

This keeps configs portable across WSL local editing and remote GPU execution.

## Epsilon Sweep

To preview the full epsilon sweep locally without running training:

```bash
python experiments/scripts/run_experiments.py --config-dir experiments/configs --only epsilon_ --dry-run
```

Recommended remote environment variables:

- `ROOT`
- `PYTHON_BIN`
- `CORPUS_PATH`
- `SFT_CKPT`
- `MODEL_SAVE_PATH`
- `SUMMARY_PATH`
- `GPU_IDS`
- `RL_NPROCS`
- `EVAL_NPROCS`

To launch the sequential remote sweep inside `tmux`:

```bash
tmux new-session -d -s epsilon_sweep \
  "cd /path/to/RL_table2charts && \
   ROOT=\$PWD \
   PYTHON_BIN=python \
   CORPUS_PATH=\$PWD/Data/PlotlyTable2Charts \
   SFT_CKPT=/actual/path/to/states_ep0.pt \
   MODEL_SAVE_PATH=\$PWD/Results/Models \
   SUMMARY_PATH=\$PWD/Results/summary \
   GPU_IDS=0,1,2,3 \
   RL_NPROCS=4 \
   EVAL_NPROCS=4 \
   experiments/scripts/run_remote_epsilon_sweep.sh"
```

For a remote dry-run using the helper:

```bash
DRY_RUN=1 experiments/scripts/run_remote_epsilon_sweep.sh
```

The current `metrics.csv` contains historical updated-policy baseline context, not the new epsilon sweep runs.

For the concrete remote workflow, observed runtime, smoke-test command, and machine-specific variables, see:

- [epsilon_sweep_runbook.md](/home/lyl610/RL_table2charts/experiments/epsilon_sweep_runbook.md)
- [reward_interaction_runbook.md](/home/lyl610/RL_table2charts/experiments/reward_interaction_runbook.md)

Do not launch full training from the notebook.

## Planned-But-Not-Implemented Handling

Some planned experiment families are not yet supported by the codebase, such as Boltzmann or UCB exploration and possibly critic-only or blended actor-critic evaluation modes.

For these cases the scaffold should:

- keep the config file;
- generate a dry-run record;
- mark the command as `planned_not_implemented`;
- attach a TODO instead of fabricating runnable support.

## Requirements Note

Do not generate the final dependency file here yet. When the environment is stable, export it separately as `project_requirements_groupID.txt`.
