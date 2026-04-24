# Local WSL Runbook (No Heavy Training)

This note is for local WSL/VS Code usage only. It focuses on dry-run and lightweight command planning.

## Local SFT Checkpoint Path

This repo includes a local SFT checkpoint at:

```text
Results/Models/sft_states_ep0.pt
```

For local dry-run commands, always set:

```bash
export SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt"
```

## Local Epsilon Sweep Dry-Run

This prints the four epsilon sweep training commands without executing them:

```bash
ROOT="$PWD" \
PYTHON_BIN=python \
CORPUS_PATH="$PWD/Data/PlotlyTable2Charts" \
SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt" \
python experiments/scripts/run_experiments.py \
  --config-dir experiments/configs \
  --only epsilon_ \
  --dry-run
```

To dry-run a single epsilon config:

```bash
ROOT="$PWD" \
PYTHON_BIN=python \
CORPUS_PATH="$PWD/Data/PlotlyTable2Charts" \
SFT_CKPT="$PWD/Results/Models/sft_states_ep0.pt" \
python experiments/scripts/run_experiments.py \
  --config experiments/configs/epsilon_eps020_top5.json \
  --dry-run
```

## Notes

- Do not run full training locally from WSL unless explicitly intended.
- If `Data/PlotlyTable2Charts` does not exist locally, either sync the dataset or set `CORPUS_PATH` to a valid location.
