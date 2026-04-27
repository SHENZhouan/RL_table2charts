import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_CSV = REPO_ROOT / "experiments" / "results" / "metrics.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "results" / "search_aggregate_metrics.csv"

IDENTITY_FIELDS = [
    "run_id",
    "method",
    "reward_mode",
    "sampling_strategy",
    "epsilon_start",
    "score_mode",
    "search_limits",
]

AGGREGATE_FIELDS = [
    "t_cnt",
    "process_time",
    "perf_time",
    "reached_states",
    "expanded_states",
    "cut_states",
    "dropped_states",
    "complete_states",
    "complete_reached",
    "complete_targets",
    "complete_first_rank",
    "search_efficiency",
    "drop_rate",
]

OUTPUT_FIELDS = IDENTITY_FIELDS + ["log_path", "status"] + AGGREGATE_FIELDS + ["notes"]

LOG_PATH_RE = re.compile(r"(?:^|;\s*)log_path=([^;]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aggregate search metrics from the exact [test-summary] logs "
            "declared in experiments/results/metrics.csv notes."
        )
    )
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_metrics_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def log_path_from_notes(notes: str) -> Optional[str]:
    match = LOG_PATH_RE.search(notes or "")
    if not match:
        return None
    return match.group(1).strip()


def extract_all_section(text: str, source: Path) -> Dict:
    marker = text.rfind("[all ")
    if marker == -1:
        raise ValueError(f"missing [all N] section: {source}")
    brace = text.find("{", marker)
    if brace == -1:
        raise ValueError(f"missing JSON payload after [all N] section: {source}")
    return json.loads(text[brace:].strip())


def as_str(value) -> str:
    return "" if value is None else str(value)


def safe_ratio(numerator, denominator) -> str:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return ""
    if denominator_f == 0:
        return ""
    return str(numerator_f / denominator_f)


def parse_summary(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"summary log is empty: {path}")
    merged = extract_all_section(text, path)
    complete = merged.get("evaluation", {}).get("stages", {}).get("complete", {})

    row = {
        "t_cnt": as_str(merged.get("t_cnt")),
        "process_time": as_str(merged.get("process_time")),
        "perf_time": as_str(merged.get("perf_time")),
        "reached_states": as_str(merged.get("reached_states")),
        "expanded_states": as_str(merged.get("expanded_states")),
        "cut_states": as_str(merged.get("cut_states")),
        "dropped_states": as_str(merged.get("dropped_states")),
        "complete_states": as_str(merged.get("complete_states")),
        "complete_reached": as_str(complete.get("reached")),
        "complete_targets": as_str(complete.get("targets")),
        "complete_first_rank": as_str(complete.get("first_rank")),
    }
    row["search_efficiency"] = safe_ratio(row["complete_states"], row["expanded_states"])
    row["drop_rate"] = safe_ratio(row["dropped_states"], row["reached_states"])
    return row


def empty_aggregate() -> Dict[str, str]:
    return {field: "" for field in AGGREGATE_FIELDS}


def build_output_row(metrics_row: Dict[str, str]) -> Dict[str, str]:
    out = {field: metrics_row.get(field, "") for field in IDENTITY_FIELDS}
    out["notes"] = metrics_row.get("notes", "")
    out.update(empty_aggregate())

    raw_log_path = log_path_from_notes(metrics_row.get("notes", ""))
    if not raw_log_path:
        out["log_path"] = ""
        out["status"] = "missing_log_path_in_metrics_notes"
        print(f"warning: run_id={out['run_id']} has no log_path in notes", file=sys.stderr)
        return out

    out["log_path"] = raw_log_path
    summary_path = resolve_path(raw_log_path)
    if not summary_path.exists():
        out["status"] = "summary_log_not_found"
        print(f"warning: summary log not found for run_id={out['run_id']}: {summary_path}", file=sys.stderr)
        return out

    try:
        out.update(parse_summary(summary_path))
    except Exception as exc:  # Keep row alignment with metrics.csv even on malformed logs.
        out["status"] = f"parse_error:{type(exc).__name__}"
        print(f"warning: failed to parse run_id={out['run_id']} log={summary_path}: {exc}", file=sys.stderr)
        return out

    out["status"] = "ok"
    out["log_path"] = repo_relative(summary_path)
    return out


def write_rows(rows: List[Dict[str, str]], output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_path}")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metrics_csv = resolve_path(args.metrics_csv)
    output_path = resolve_path(args.output)

    metrics_rows = read_metrics_rows(metrics_csv)
    output_rows = [build_output_row(row) for row in metrics_rows]
    write_rows(output_rows, output_path, args.overwrite)

    ok_count = sum(1 for row in output_rows if row["status"] == "ok")
    print(f"wrote {len(output_rows)} rows to {output_path} ({ok_count} ok, {len(output_rows) - ok_count} warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
