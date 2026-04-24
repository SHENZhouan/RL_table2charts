# Experiment Log

## 2026-04-24

- Added the first reproducible experiment-management scaffold on `yilin/experiment-pipeline`, including project context docs, experiment configs, dry-run runner, normalized results collection, notebook analysis entrypoint, and experiment raw-log gitignore rules.
- Replaced the metrics summary helper's pandas dependency with standard-library CSV parsing so the scaffold works in the local `t2c` environment without extra dataframe compatibility issues.
- Fixed the `results.md` collector to split multi-block sections into separate normalized rows, recover missing updated-policy `top_m`, and refresh `metrics.csv` cleanly under `--overwrite` without leaving stale parser artifacts.
- Added eval-only actor/critic/blend score-mode support to `update_actor_test_agent_mp.py`, made all three actor-critic diagnostic configs runnable in dry-run, and recorded reduced-concurrency local smoke-test commands for a single-GPU RTX 4060 setup.

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
