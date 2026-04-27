import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.extract_test_summary import parse_summary_file, repo_relative
DEFAULT_EPSILON_CSV = REPO_ROOT / "experiments" / "results" / "final_eval_epsilon_sweep_20260425.csv"
DEFAULT_REWARD_CSV = REPO_ROOT / "experiments" / "results" / "final_eval_reward_intensity_20260425.csv"
DEFAULT_METRICS_CSV = REPO_ROOT / "experiments" / "results" / "metrics.csv"
DEFAULT_HARD_RESULTS_GLOB = "hard_greedy_*.results"

METRICS_FIELDS = [
    "run_id",
    "method",
    "reward_mode",
    "sampling_strategy",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "top_m",
    "temperature",
    "ucb_c",
    "exact_reward",
    "default_reward",
    "same_token_reward",
    "field_reward",
    "same_field_type_reward",
    "score_mode",
    "critic_score_weight",
    "checkpoint",
    "search_limits",
    "R@1",
    "R@3",
    "R@5",
    "R@10",
    "R@20",
    "first_rank",
    "reached",
    "targets",
    "recall_all",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote authoritative epsilon, reward, and hard-greedy final-eval results into metrics.csv."
    )
    parser.add_argument("--epsilon-csv", default=str(DEFAULT_EPSILON_CSV))
    parser.add_argument("--reward-csv", default=str(DEFAULT_REWARD_CSV))
    parser.add_argument("--hard-results", help="Path to hard_greedy_*.results. Defaults to latest matching file.")
    parser.add_argument("--hard-summary-log", help="Optional explicit hard-greedy [test-summary] log path.")
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the metrics CSV.")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_hard_results() -> Path:
    matches = sorted((REPO_ROOT / "experiments" / "results" / "raw_logs").glob(DEFAULT_HARD_RESULTS_GLOB))
    if not matches:
        raise FileNotFoundError("No hard_greedy_*.results files found under experiments/results/raw_logs")
    return matches[-1].resolve()


def parse_results_file(path: Path) -> Dict[str, str]:
    block: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, sep, value = stripped.partition("=")
        if sep:
            block[key] = value
    if not block:
        raise ValueError(f"Results file is empty or malformed: {path}")
    return block


def metrics_row_template() -> Dict[str, str]:
    return {field: "" for field in METRICS_FIELDS}


def hard_run_id_from_results(path: Path) -> str:
    stem = path.stem
    prefix = "hard_greedy_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def promote_epsilon_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    promoted: List[Dict[str, str]] = []
    for row in rows:
        out = metrics_row_template()
        epsilon = row["epsilon_start"]
        run_id_map = {
            "0.05": "20260425T182100Z",
            "0.10": "20260425T182600Z",
            "0.20": "20260425T183200Z",
            "0.30": "20260425T183800Z",
        }
        out.update(
            {
                "run_id": run_id_map.get(epsilon, ""),
                "method": "updated_policy",
                "reward_mode": "baseline",
                "sampling_strategy": "epsilon_top_m",
                "epsilon_start": epsilon,
                "epsilon_end": "0.02",
                "epsilon_decay": "0.8",
                "top_m": "5",
                "checkpoint": "Results/Models/sft_states_ep0.pt",
                "search_limits": "e50-b4-na",
                "R@1": row["R@1"],
                "R@3": row["R@3"],
                "R@5": row["R@5"],
                "R@10": row["R@10"],
                "R@20": row["R@20"],
                "first_rank": row["first_rank"],
                "recall_all": row["recall_all"],
                "notes": (
                    f"2026-04-25 regenerated PlotlyTable2Charts formal final eval extracted from tracked [test-summary] log; "
                    f"model_dir={row['model_dir']}; log_path={row['log_path']}; t_cnt={row['t_cnt']}"
                ),
            }
        )
        promoted.append(out)
    return promoted


def reward_params(experiment: str) -> Dict[str, str]:
    mapping = {
        "reward_conservative_greedy": {
            "method": "update_reward",
            "sampling_strategy": "greedy",
            "epsilon_start": "",
            "epsilon_end": "",
            "epsilon_decay": "",
            "top_m": "",
            "exact_reward": "0.95",
            "default_reward": "0.05",
            "same_token_reward": "0.07",
            "field_reward": "0.1",
            "same_field_type_reward": "0.2",
        },
        "reward_conservative_epsilon": {
            "method": "update_reward_policy",
            "sampling_strategy": "epsilon_top_m",
            "epsilon_start": "0.2",
            "epsilon_end": "0.02",
            "epsilon_decay": "0.8",
            "top_m": "5",
            "exact_reward": "0.95",
            "default_reward": "0.05",
            "same_token_reward": "0.07",
            "field_reward": "0.1",
            "same_field_type_reward": "0.2",
        },
        "reward_current_greedy": {
            "method": "update_reward",
            "sampling_strategy": "greedy",
            "epsilon_start": "",
            "epsilon_end": "",
            "epsilon_decay": "",
            "top_m": "",
            "exact_reward": "0.95",
            "default_reward": "0.05",
            "same_token_reward": "0.1",
            "field_reward": "0.15",
            "same_field_type_reward": "0.35",
        },
        "reward_current_epsilon": {
            "method": "update_reward_policy",
            "sampling_strategy": "epsilon_top_m",
            "epsilon_start": "0.2",
            "epsilon_end": "0.02",
            "epsilon_decay": "0.8",
            "top_m": "5",
            "exact_reward": "0.95",
            "default_reward": "0.05",
            "same_token_reward": "0.1",
            "field_reward": "0.15",
            "same_field_type_reward": "0.35",
        },
        "reward_aggressive_greedy": {
            "method": "update_reward",
            "sampling_strategy": "greedy",
            "epsilon_start": "",
            "epsilon_end": "",
            "epsilon_decay": "",
            "top_m": "",
            "exact_reward": "0.95",
            "default_reward": "0.05",
            "same_token_reward": "0.15",
            "field_reward": "0.25",
            "same_field_type_reward": "0.5",
        },
        "reward_aggressive_epsilon": {
            "method": "update_reward_policy",
            "sampling_strategy": "epsilon_top_m",
            "epsilon_start": "0.2",
            "epsilon_end": "0.02",
            "epsilon_decay": "0.8",
            "top_m": "5",
            "exact_reward": "0.95",
            "default_reward": "0.05",
            "same_token_reward": "0.15",
            "field_reward": "0.25",
            "same_field_type_reward": "0.5",
        },
    }
    if experiment not in mapping:
        raise KeyError(f"Unknown reward experiment: {experiment}")
    return mapping[experiment]


def reward_run_id(log_path: str) -> str:
    name = Path(log_path).name
    if "20260425T2331" in name:
        return "20260425T233100Z"
    if "20260426T0017" in name:
        return "20260426T001700Z"
    if "20260426T0104" in name:
        return "20260426T010400Z"
    if "20260426T0150" in name:
        return "20260426T015000Z"
    if "20260426T0238" in name:
        return "20260426T023800Z"
    if "20260426T0325" in name:
        return "20260426T032500Z"
    return ""


def promote_reward_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    promoted: List[Dict[str, str]] = []
    for row in rows:
        params = reward_params(row["experiment"])
        out = metrics_row_template()
        out.update(
            {
                "run_id": reward_run_id(row["log_path"]),
                "method": params["method"],
                "reward_mode": row["reward_mode"],
                "sampling_strategy": params["sampling_strategy"],
                "epsilon_start": params["epsilon_start"],
                "epsilon_end": params["epsilon_end"],
                "epsilon_decay": params["epsilon_decay"],
                "top_m": params["top_m"],
                "exact_reward": params["exact_reward"],
                "default_reward": params["default_reward"],
                "same_token_reward": params["same_token_reward"],
                "field_reward": params["field_reward"],
                "same_field_type_reward": params["same_field_type_reward"],
                "checkpoint": "Results/Models/sft_states_ep0.pt",
                "search_limits": "e50-b4-na",
                "R@1": row["R@1"],
                "R@3": row["R@3"],
                "R@5": row["R@5"],
                "R@10": row["R@10"],
                "R@20": row["R@20"],
                "first_rank": row["first_rank"],
                "recall_all": row["recall_all"],
                "notes": (
                    f"2026-04-25 authoritative helper-managed full rerun formal final eval extracted from tracked [test-summary] log; "
                    f"model_dir={row['model_dir']}; log_path={row['log_path']}; t_cnt={row['t_cnt']}"
                ),
            }
        )
        promoted.append(out)
    return promoted


def promote_hard_greedy_row(results_path: Path, summary_log_path: Optional[Path]) -> Dict[str, str]:
    block = parse_results_file(results_path)
    summary_path = summary_log_path.resolve() if summary_log_path else resolve_path(block["summary_log"])
    summary = parse_summary_file(summary_path)
    model_dir = repo_relative(resolve_path(block["model_dir"]))
    out = metrics_row_template()
    out.update(
        {
            "run_id": hard_run_id_from_results(results_path),
            "method": "baseline",
            "reward_mode": "baseline",
            "sampling_strategy": "greedy",
            "checkpoint": "Results/Models/sft_states_ep0.pt",
            "search_limits": "e50-b4-na",
            "R@1": summary["R@1"],
            "R@3": summary["R@3"],
            "R@5": summary["R@5"],
            "R@10": summary["R@10"],
            "R@20": summary["R@20"],
            "first_rank": summary["first_rank"],
            "recall_all": summary["recall_all"],
            "notes": (
                f"2026-04-26 regenerated PlotlyTable2Charts hard reward greedy formal final eval extracted from tracked [test-summary] log; "
                f"model_dir={model_dir}; log_path={repo_relative(summary_path)}; t_cnt={summary['t_cnt']}"
            ),
        }
    )
    return out


def to_csv_text(rows: List[Dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=METRICS_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_metrics(path: Path, rows: List[Dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing metrics CSV without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    epsilon_rows = load_csv(resolve_path(args.epsilon_csv))
    reward_rows = load_csv(resolve_path(args.reward_csv))
    hard_results = resolve_path(args.hard_results) if args.hard_results else latest_hard_results()
    hard_summary = resolve_path(args.hard_summary_log) if args.hard_summary_log else None

    rows = []
    rows.extend(promote_epsilon_rows(epsilon_rows))
    rows.extend(promote_reward_rows(reward_rows))
    rows.append(promote_hard_greedy_row(hard_results, hard_summary))
    rows.sort(key=lambda row: row["run_id"])

    metrics_path = resolve_path(args.metrics_csv)
    write_metrics(metrics_path, rows, overwrite=args.overwrite)
    sys.stdout.write(to_csv_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
