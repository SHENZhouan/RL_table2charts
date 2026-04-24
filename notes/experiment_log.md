# Experiment Log

## 2026-04-24

- Added the first reproducible experiment-management scaffold on `yilin/experiment-pipeline`, including project context docs, experiment configs, dry-run runner, normalized results collection, notebook analysis entrypoint, and experiment raw-log gitignore rules.
- Replaced the metrics summary helper's pandas dependency with standard-library CSV parsing so the scaffold works in the local `t2c` environment without extra dataframe compatibility issues.
- Fixed the `results.md` collector to split multi-block sections into separate normalized rows, recover missing updated-policy `top_m`, and refresh `metrics.csv` cleanly under `--overwrite` without leaving stale parser artifacts.
