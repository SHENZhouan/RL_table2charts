# Experiment Log

## 2026-04-23

- Step 1 (`checkpoint/configurable-entrypoint-defaults`): made Python experiment entrypoint defaults portable by replacing old `/storage` and `/home/exp` defaults with repo-relative paths, added CPU fallback for `test_agent_mp.py` smoke tests, and exposed `single_inference.py --output_path`.
