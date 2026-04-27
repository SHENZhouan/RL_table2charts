import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "results" / "recommendation_eval_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "results" / "recommendation_diversity.csv"

OUTPUT_FIELDS = [
    "run_id",
    "method",
    "reward_mode",
    "sampling_strategy",
    "epsilon_start",
    "score_mode",
    "checkpoint",
    "source_log_path",
    "summary_log",
    "recommend_log_dir",
    "status",
    "table_count",
    "total_recommendations",
    "unique_recommendations",
    "unique_recommendation_rate",
    "chart_type_count",
    "chart_type_entropy",
    "chart_type_distribution",
    "avg_recommendations_per_table",
    "avg_unique_recommendations_per_table",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract chart-type diversity and unique recommendation metrics from per-table recommendation logs."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--actor-critic-results", action="append", default=[])
    parser.add_argument("--recommend-log-dir", action="append", default=[])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path | str) -> str:
    if not path:
        return ""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_actor_critic_results(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    common: Dict[str, str] = {}
    current: Dict[str, str] = {}

    def flush_current() -> None:
        if current.get("recommend_log_dir"):
            row = dict(common)
            row.update(current)
            row.setdefault("method", "actor_critic")
            row.setdefault("run_id", common.get("run_id", path.stem))
            row.setdefault("checkpoint", common.get("checkpoint", ""))
            row.setdefault("status", "ok")
            rows.append(row)

    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if not sep:
            continue
        if key == "eval_config":
            flush_current()
            current = {"eval_config": value}
            continue
        if key in {"run_id", "model_dir", "checkpoint"} and not current:
            common[key] = value
        else:
            current[key] = value
    flush_current()
    return rows


def canonical_recommendation(recommendation: Dict) -> str:
    comparable = {key: value for key, value in recommendation.items() if key != "score"}
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def entropy(counter: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = count / total
        value -= p * math.log2(p)
    return value


def analyze_recommend_dir(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*.json"))
    chart_types: Counter = Counter()
    unique_keys = set()
    total_recommendations = 0
    per_table_total: List[int] = []
    per_table_unique: List[int] = []

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        ranked = payload.get("ranked_recommend", [])
        table_unique = set()
        for recommendation in ranked:
            if not isinstance(recommendation, dict):
                continue
            chart_type = recommendation.get("ANA", "")
            if chart_type:
                chart_types[chart_type] += 1
            key = canonical_recommendation(recommendation)
            unique_keys.add(key)
            table_unique.add(key)
            total_recommendations += 1
        per_table_total.append(len(ranked))
        per_table_unique.append(len(table_unique))

    unique_count = len(unique_keys)
    table_count = len(files)
    return {
        "table_count": str(table_count),
        "total_recommendations": str(total_recommendations),
        "unique_recommendations": str(unique_count),
        "unique_recommendation_rate": "" if total_recommendations == 0 else str(unique_count / total_recommendations),
        "chart_type_count": str(len(chart_types)),
        "chart_type_entropy": str(entropy(chart_types, total_recommendations)),
        "chart_type_distribution": json.dumps(dict(sorted(chart_types.items())), sort_keys=True),
        "avg_recommendations_per_table": "" if table_count == 0 else str(sum(per_table_total) / table_count),
        "avg_unique_recommendations_per_table": "" if table_count == 0 else str(sum(per_table_unique) / table_count),
    }


def output_row_from_manifest(row: Dict[str, str]) -> Dict[str, str]:
    out = {field: row.get(field, "") for field in OUTPUT_FIELDS}
    recommend_log_dir = resolve_path(row.get("recommend_log_dir", ""))
    out["recommend_log_dir"] = repo_relative(recommend_log_dir)

    if row.get("status") != "ok":
        out["status"] = row.get("status", "skipped")
        return out

    try:
        out.update(analyze_recommend_dir(recommend_log_dir))
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = f"parse_error:{type(exc).__name__}"
        out["notes"] = str(exc)
    return out


def explicit_dir_row(path: Path) -> Dict[str, str]:
    row = {field: "" for field in OUTPUT_FIELDS}
    row["recommend_log_dir"] = repo_relative(path)
    row["status"] = "ok"
    try:
        row.update(analyze_recommend_dir(path))
    except Exception as exc:
        row["status"] = f"parse_error:{type(exc).__name__}"
        row["notes"] = str(exc)
    return row


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
    rows: List[Dict[str, str]] = []

    manifest_rows = read_manifest(resolve_path(args.manifest))
    rows.extend(output_row_from_manifest(row) for row in manifest_rows)

    for raw_path in args.actor_critic_results:
        for row in parse_actor_critic_results(resolve_path(raw_path)):
            rows.append(output_row_from_manifest(row))

    for raw_dir in args.recommend_log_dir:
        rows.append(explicit_dir_row(resolve_path(raw_dir)))

    output_path = resolve_path(args.output)
    write_rows(rows, output_path, args.overwrite)
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"wrote {len(rows)} rows to {output_path} ({ok_count} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
