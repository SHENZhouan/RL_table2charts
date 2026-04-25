# Experiment Log

## 2026-04-24

- Added the first reproducible experiment-management scaffold on `yilin/experiment-pipeline`, including project context docs, experiment configs, dry-run runner, normalized results collection, notebook analysis entrypoint, and experiment raw-log gitignore rules.
- Replaced the metrics summary helper's pandas dependency with standard-library CSV parsing so the scaffold works in the local `t2c` environment without extra dataframe compatibility issues.
- Fixed the `results.md` collector to split multi-block sections into separate normalized rows, recover missing updated-policy `top_m`, and refresh `metrics.csv` cleanly under `--overwrite` without leaving stale parser artifacts.
- Added eval-only actor/critic/blend score-mode support to `update_actor_test_agent_mp.py`, made all three actor-critic diagnostic configs runnable in dry-run, and recorded reduced-concurrency local smoke-test commands for a single-GPU RTX 4060 setup.
- Normalized the epsilon sweep scaffold for remote execution by fixing config endpoints, clarifying updated-policy train/eval TODOs, and adding a tmux-safe sequential remote sweep helper.
- Added a local WSL runbook that pins `SFT_CKPT=$PWD/Results/Models/sft_states_ep0.pt` for consistent dry-run usage.

### Local Smoke-Test Commands (not executed)

```bash
CUDA_VISIBLE_DEVICES=0 python update_actor_test_agent_mp.py \
  -m ${ROOT}/Results/Models/20260423200116-update_actor-2el192fd128.128GRUh-allCharts-actor-critic-RL \
  -f states_ep0.pt \
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
  --corpus_path /path/to/Data/PlotlyTable2Charts \
  --lang en \
  --limit_search_group \
  --score_mode actor

CUDA_VISIBLE_DEVICES=0 python update_actor_test_agent_mp.py \
  -m ${ROOT}/Results/Models/20260423200116-update_actor-2el192fd128.128GRUh-allCharts-actor-critic-RL \
  -f states_ep0.pt \
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
  --corpus_path /path/to/Data/PlotlyTable2Charts \
  --lang en \
  --limit_search_group \
  --score_mode critic \
  --critic_score_weight 1.0

CUDA_VISIBLE_DEVICES=0 python update_actor_test_agent_mp.py \
  -m ${ROOT}/Results/Models/20260423200116-update_actor-2el192fd128.128GRUh-allCharts-actor-critic-RL \
  -f states_ep0.pt \
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
  --corpus_path /path/to/Data/PlotlyTable2Charts \
  --lang en \
  --limit_search_group \
  --score_mode blend \
  --critic_score_weight 0.5
```

## 2026-04-25

- Fast-forward synced local `yilin/experiment-pipeline` to `1716132` (`Add missing Plotly preprocessing helpers`) after the remote preprocessing work landed upstream.
- Confirmed the remote setup needed preprocessed Plotly corpus artifacts that are not stored in Git. The remote host was updated with raw Plotly input plus preprocessing code, then used to build a smoke-test corpus and unblock RL startup.
- Ran a single-GPU smoke test on a remote 2080 against `reinforce.updated_policy_learn_dist` with `queue_mode=local` and the updated-policy epsilon settings. The command successfully started training initialization; it was manually stopped after confirming startup behavior.
- Decision: keep syncing code changes through Git, but do not sync generated `Data/PlotlyTable2Charts_smoke/` corpus artifacts or other large preprocessing outputs. Those remain local/remote runtime data, not source-controlled assets.
- Completed the remote epsilon sweep and final evaluation on the processed full Plotly corpus. Each one-epoch updated-policy training run took about 16 minutes wall-clock, so the 4-run sequential sweep took about 65 minutes before final eval.
- Recorded the produced model directories in `experiments/results/epsilon_sweep_model_dirs_20260425.csv` and the final evaluation summary in `experiments/results/final_eval_epsilon_sweep_20260425.csv`.
- Confirmed from the training logs that the successful sweep used `Data/PlotlyTable2Charts` with `load_at_most=None`, `num_train_analysis=None`, and the default `train/valid/test = 0.7/0.1/0.2` split. This was not a smoke-subset training run.
- Moved the previously parsed teammate/historical rows out of `experiments/results/metrics.csv` into `experiments/results/metrics_historical.csv`, and repopulated the main `metrics.csv` with the 2026-04-25 epsilon sweep final-eval results only so the main table matches the current formal analysis set.
- Prepared the next-stage reward-intensity × sampling scaffold for the current 2-GPU server: added conservative/aggressive reward configs, locked runtime env overrides in `run_experiments.py`, replaced the coarse reward-interaction helper/runbook with an intensity-focused version, and added header-only result CSV templates. No reward training was run in this step.
- Aligned `reward_conservative_greedy.json` with the intended conservative reward definition (`0.95/0.05/0.07/0.10/0.20`) and updated the reward-intensity runbook to describe the current target as a generic 2-GPU server, with a note for running on 2x2080 if memory becomes tight.
- After syncing the latest remote logs, confirmed that the real reward-intensity training run `reward_intensity_20260425T054952Z.log` completed the four core configs in order: `reward_conservative_greedy`, `reward_conservative_epsilon`, `reward_current_greedy`, and `reward_current_epsilon`. Recorded the produced model directories in `experiments/results/reward_intensity_model_dirs_20260425.csv` and prepared final-eval command templates for the four checkpoints.
- Fixed the experiment helper contract so one command now means train plus formal test-set evaluation: `run_remote_epsilon_sweep.sh` and `run_remote_reward_intensity_sweep.sh` now create per-config markers, discover the fresh RL model dir, run `test_agent_mp.py`, and record the discovered model/eval paths in sidecar results files. Updated the README and runbooks to distinguish training completion from final-eval completion and to treat training-time `test/valid SUMMARY` lines as provisional only.
- Narrowed `.gitignore` so only formal `Results/Models/*/evaluations/test-*/[test-summary]*.log` artifacts can be tracked from RL model directories, while checkpoints remain ignored. Added `experiments/scripts/extract_test_summary.py` to normalize epsilon final-eval `[test-summary]` logs into CSV rows, and expanded `experiments/epsilon_sweep_runbook.md` into the detailed operator record for the corrected train-plus-final-eval pipeline.
- Unified final-eval extraction for both experiment families behind `experiments/scripts/extract_test_summary.py` using `--family epsilon_sweep` and `--family reward_intensity`. Rebuilt `final_eval_epsilon_sweep_20260425.csv` from tracked `[test-summary]` logs, filled `final_eval_reward_intensity_20260425.csv` from the four completed reward runs, and updated the reward runbook to mark training-time summaries as provisional only.
- Tightened reward result provenance: added `reward_intensity_rerun_manifest_20260425.csv` so reward extraction can target authoritative rerun logs deterministically, archived the older pre-helper-fix reward final-eval rows into `final_eval_reward_intensity_pre_helper_20260425.csv`, and updated the reward runbook to treat the current helper-managed full rerun as the report source of truth.

### Remote Updated-Policy Smoke-Test Command

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
