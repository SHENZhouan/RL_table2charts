# Experiments Scaffold

## Purpose

This directory provides a reproducible experiment-management layer for the Plotly-only Table2Charts adaptation.

It is intended to:

- document experiment families and assumptions;
- store machine-independent config files;
- generate dry-run command plans before remote execution;
- normalize result extraction into one CSV;
- support lightweight notebook analysis without embedding training code inline.

It does not replace the existing training or evaluation code in `Table2Charts/`. The helpers orchestrate those entrypoints so one command can run RL training and the follow-up `test_agent_mp.py` final evaluation.

## Structure

- `configs/`: JSON configs for baseline, updated-policy, reward, and actor-critic diagnostics
- `scripts/run_experiments.py`: dry-run-first orchestration layer
- `scripts/collect_results.py`: normalize existing results into `metrics.csv`
- `scripts/extract_test_summary.py`: normalize tracked `test_agent_mp.py` final-eval logs into family-specific CSVs
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

In non-dry-run mode, the helper now treats a config as completed only after:

1. RL training finishes
2. the new RL model directory is discovered
3. `test_agent_mp.py` runs on the test split
4. the helper records the model dir and eval log dir in its sidecar results file

Do not treat training-time `test/valid SUMMARY` lines as formal report metrics. Final report metrics must come from the post-training `test_agent_mp.py` evaluation.

The current `metrics.csv` contains normalized formal experiment rows, not raw training-time summaries.

For the concrete remote workflow, observed runtime, smoke-test command, and machine-specific variables, see:

- [epsilon_sweep_runbook.md](/home/lyl610/RL_table2charts/experiments/epsilon_sweep_runbook.md)
- [reward_intensity_interaction_runbook.md](/home/lyl610/RL_table2charts/experiments/reward_intensity_interaction_runbook.md)

The epsilon runbook is also the source of truth for:

- authoritative RL model-dir mappings for the 2026-04-25 sweep;
- formal `[test-summary]` final-eval log locations;
- `experiments/scripts/extract_test_summary.py`, which now provides one interface for both:
  - `--family epsilon_sweep`
  - `--family reward_intensity`

For reward reruns, the extractor reads a dedicated rerun manifest so report-facing CSVs are rebuilt from authoritative rerun logs only, without mixing old and rerun `[test-summary]` files from the same model directory.

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
