#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export ROOT
export PYTHON_BIN
export CODE_DIR="${CODE_DIR:-${ROOT}/Table2Charts}"
export CORPUS_PATH="${CORPUS_PATH:-${ROOT}/Data/PlotlyTable2Charts}"
export GPU_IDS="${GPU_IDS:-0}"
export EVAL_NPROCS="${EVAL_NPROCS:-1}"
export METRICS_CSV="${METRICS_CSV:-${ROOT}/experiments/results/metrics.csv}"
export RECOMMEND_ROOT="${RECOMMEND_ROOT:-${ROOT}/experiments/results/recommendations}"
export MANIFEST="${MANIFEST:-${ROOT}/experiments/results/recommendation_eval_manifest.csv}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export DRY_RUN="${DRY_RUN:-0}"
export LIMIT="${LIMIT:-0}"
export ONLY_RUN_ID="${ONLY_RUN_ID:-}"

"${PYTHON_BIN}" - <<'PY'
import csv
import os
import re
import subprocess
from pathlib import Path

root = Path(os.environ["ROOT"]).resolve()
code_dir = Path(os.environ["CODE_DIR"]).resolve()
corpus_path = Path(os.environ["CORPUS_PATH"]).resolve()
gpu_ids = os.environ["GPU_IDS"]
eval_nprocs = os.environ["EVAL_NPROCS"]
python_bin = os.environ.get("PYTHON_BIN", "python")
metrics_csv = Path(os.environ["METRICS_CSV"]).resolve()
recommend_root = Path(os.environ["RECOMMEND_ROOT"]).resolve()
manifest = Path(os.environ["MANIFEST"]).resolve()
run_id = os.environ["RUN_ID"]
dry_run = os.environ["DRY_RUN"] == "1"
limit = int(os.environ.get("LIMIT") or 0)
only_run_id = os.environ.get("ONLY_RUN_ID", "")

fields = [
    "run_id", "method", "reward_mode", "sampling_strategy", "epsilon_start",
    "score_mode", "critic_score_weight", "model_dir", "checkpoint", "source_log_path",
    "summary_log", "recommend_log_dir", "status", "notes",
]

def note_value(notes: str, key: str) -> str:
    match = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", notes or "")
    return match.group(1).strip() if match else ""

def normalize(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        return (root / path).resolve()
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "Results" in parts:
        idx = parts.index("Results")
        candidate = root.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate.resolve()
    return path

def repo_relative(path: Path | str) -> str:
    if not path:
        return ""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)

def safe_label(row: dict) -> str:
    raw = "_".join([
        row.get("run_id", ""),
        row.get("method", ""),
        row.get("sampling_strategy", ""),
        row.get("epsilon_start", ""),
        row.get("score_mode", ""),
    ])
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")

def base_command(program: str, model_dir: Path, eval_log_dir: str, recommend_log_dir: Path) -> list[str]:
    return [
        python_bin,
        program,
        "-m", str(model_dir),
        "-f", "states_ep0.pt",
        "--model_name", "cp",
        "--model_size", "small",
        "--features", "all-fast",
        "--log_save_path", eval_log_dir,
        "--search_type", "allCharts",
        "--input_type", "allCharts",
        "--previous_type", "allCharts",
        "--nprocs", str(eval_nprocs),
        "--nagents", "64",
        "--nthreads", "5",
        "--search_limits", "e50-b4-na",
        "--corpus_path", str(corpus_path),
        "--lang", "en",
        "--limit_search_group",
        "--recommend_log_path", str(recommend_log_dir),
    ]

def run_eval(row: dict) -> dict:
    notes = row.get("notes", "")
    model_dir_raw = note_value(notes, "model_dir")
    source_log_raw = note_value(notes, "log_path")
    model_dir = normalize(model_dir_raw) if model_dir_raw else Path("")
    source_log = normalize(source_log_raw) if source_log_raw else Path("")
    checkpoint = model_dir / "states_ep0.pt" if model_dir_raw else Path("")
    label = safe_label(row)
    eval_log_dir = f"evaluations/recommendation-{label}-{run_id}"
    recommend_log_dir = recommend_root / f"{label}-{run_id}"

    status = "ok"
    summary_log = Path("")
    out_notes = ""

    if not model_dir_raw or not source_log_raw:
        status = "missing_model_or_log_path_in_metrics"
    elif not checkpoint.exists():
        status = "missing_checkpoint"
    elif not source_log.exists():
        status = "missing_source_log"
    elif dry_run:
        status = "dry_run"
        print(f"dry_run: run_id={row.get('run_id')} checkpoint={checkpoint} recommend_log_dir={recommend_log_dir}")
    else:
        recommend_log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        if row.get("method") == "actor_critic":
            score_mode = row.get("score_mode") or ""
            if not score_mode:
                status = "missing_actor_critic_score_mode"
            else:
                command = base_command("update_actor_test_agent_mp.py", model_dir, eval_log_dir, recommend_log_dir)
                command.extend([
                    "--score_mode", score_mode,
                    "--critic_score_weight", row.get("critic_score_weight") or "0.5",
                ])
                subprocess.run(command, check=True, cwd=code_dir, env=env)
        else:
            command = base_command("test_agent_mp.py", model_dir, eval_log_dir, recommend_log_dir)
            subprocess.run(command, check=True, cwd=code_dir, env=env)

        if status == "ok":
            matches = sorted((model_dir / eval_log_dir).glob("[[]test-summary]*.log"))
            if not matches:
                status = "missing_new_summary_log"
            elif not recommend_log_dir.exists():
                status = "missing_recommend_log_dir"
            else:
                summary_log = matches[-1]

    return {
        "run_id": row.get("run_id", ""),
        "method": row.get("method", ""),
        "reward_mode": row.get("reward_mode", ""),
        "sampling_strategy": row.get("sampling_strategy", ""),
        "epsilon_start": row.get("epsilon_start", ""),
        "score_mode": row.get("score_mode", ""),
        "critic_score_weight": row.get("critic_score_weight", ""),
        "model_dir": repo_relative(model_dir),
        "checkpoint": repo_relative(checkpoint),
        "source_log_path": repo_relative(source_log),
        "summary_log": repo_relative(summary_log),
        "recommend_log_dir": repo_relative(recommend_log_dir),
        "status": status,
        "notes": out_notes,
    }

print(f"root={root}")
print(f"code_dir={code_dir}")
print(f"corpus_path={corpus_path}")
print(f"metrics_csv={metrics_csv}")
print(f"recommend_root={recommend_root}")
print(f"manifest={manifest}")
print(f"run_id={run_id}")
print(f"dry_run={dry_run}")
print(f"limit={limit}")
print(f"only_run_id={only_run_id}")

if not metrics_csv.exists():
    raise FileNotFoundError(metrics_csv)
if not dry_run and not corpus_path.exists():
    raise FileNotFoundError(corpus_path)

recommend_root.mkdir(parents=True, exist_ok=True)
manifest.parent.mkdir(parents=True, exist_ok=True)

rows = []
with metrics_csv.open("r", encoding="utf-8", newline="") as f:
    for metrics_row in csv.DictReader(f):
        if only_run_id and metrics_row.get("run_id") != only_run_id:
            continue
        if limit and len(rows) >= limit:
            break
        rows.append(run_eval(metrics_row))

with manifest.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

ok_count = sum(1 for row in rows if row["status"] == "ok")
print(f"wrote {len(rows)} rows to {manifest} ({ok_count} ok)")
PY
