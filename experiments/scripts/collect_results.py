import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "experiments" / "results" / "metrics.csv"
METRIC_FIELDS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Collect normalized experiment metrics")
    parser.add_argument("--results-md", default=str(REPO_ROOT / "results.md"))
    parser.add_argument("--log-file", help="Optional future direct log-file parser input")
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def default_row() -> Dict[str, str]:
    return {field: "" for field in METRIC_FIELDS}


def extract_float(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def infer_method(text: str) -> str:
    lowered = text.lower()
    if "actor" in lowered and "critic" in lowered:
        return "actor_critic"
    if "update reward + updated policy" in lowered or "combo_policy" in lowered:
        return "update_reward_policy"
    if "updated policy" in lowered or "policy=epsilon_top_m" in lowered:
        return "updated_policy"
    if "update reward" in lowered or "reward:" in lowered:
        return "update_reward"
    if "baseline" in lowered or "greedy" in lowered:
        return "baseline"
    return ""


def infer_sampling(text: str) -> str:
    lowered = text.lower()
    if "epsilon_top_m" in lowered or "explore_top_m" in lowered or "combo_policy" in lowered:
        return "epsilon_top_m"
    if "greedy" in lowered:
        return "greedy"
    return ""


def infer_reward_mode(text: str) -> str:
    lowered = text.lower()
    if "reward:" in lowered or "dense reward" in lowered:
        return "soft_reward"
    return "baseline"


def infer_score_mode(text: str) -> str:
    lowered = text.lower()
    if "actor policy scores" in lowered or "update actor eval rerun" in lowered:
        return "actor"
    if "critic" in lowered and "weight" in lowered:
        return "blend"
    return ""


def parse_complete_recall(text: str) -> Optional[Dict]:
    match = re.search(r"Complete recall info:\s*(\{.*\})", text)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None


def parse_results_md(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## ", text)
    rows = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        row = default_row()
        row["notes"] = "parsed from results.md"
        row["method"] = infer_method(section)
        row["reward_mode"] = infer_reward_mode(section)
        row["sampling_strategy"] = infer_sampling(section)
        row["score_mode"] = infer_score_mode(section)

        title = section.splitlines()[0].strip()
        run_id_match = re.search(r"(\d{8}T\d{6}Z)", title)
        if run_id_match:
            row["run_id"] = run_id_match.group(1)

        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith("- sft_ckpt:"):
                row["checkpoint"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- policy_epsilon_start:"):
                row["epsilon_start"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- policy_epsilon_end:"):
                row["epsilon_end"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- policy_epsilon_decay:"):
                row["epsilon_decay"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- combo_policy:"):
                row["epsilon_start"] = extract_float(r"epsilon_start=([0-9.]+)", stripped) or row["epsilon_start"]
                row["epsilon_end"] = extract_float(r"epsilon_end=([0-9.]+)", stripped) or row["epsilon_end"]
                row["epsilon_decay"] = extract_float(r"epsilon_decay=([0-9.]+)", stripped) or row["epsilon_decay"]
                row["top_m"] = extract_float(r"explore_top_m=([0-9.]+)", stripped) or row["top_m"]
            elif stripped.startswith("- actor_loss_weight:"):
                row["notes"] += "; actor_loss_weight present"
            elif stripped.startswith("- critic_score_weight:"):
                row["critic_score_weight"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- reward:"):
                row["exact_reward"] = extract_float(r"exact=([0-9.]+)", stripped) or ""
                row["default_reward"] = extract_float(r"default=([0-9.]+)", stripped) or ""
                row["same_token_reward"] = extract_float(r"same_token=([0-9.]+)", stripped) or ""
                row["field_reward"] = extract_float(r"field=([0-9.]+)", stripped) or ""
                row["same_field_type_reward"] = extract_float(r"same_field_type=([0-9.]+)", stripped) or ""
            elif stripped.startswith("- eval_log_dir:") and not row["checkpoint"]:
                row["notes"] += f"; eval_log_dir={stripped.split(':', 1)[1].strip()}"

        complete = parse_complete_recall(section)
        if complete:
            recall = complete.get("recall", {})
            row["R@1"] = str(recall.get("@01", ""))
            row["R@3"] = str(recall.get("@03", ""))
            row["R@5"] = str(recall.get("@05", ""))
            row["R@10"] = str(recall.get("@10", ""))
            row["R@20"] = str(recall.get("@20", ""))
            row["recall_all"] = str(recall.get("all", ""))
            row["first_rank"] = str(complete.get("first_rank", ""))
            row["reached"] = str(complete.get("reached", ""))
            row["targets"] = str(complete.get("targets", ""))
        else:
            row["notes"] += "; missing Complete recall info"

        row["search_limits"] = "e50-b4-na"
        rows.append(row)

    return rows


def dedup_key(row: Dict[str, str]) -> tuple:
    return (
        row["run_id"],
        row["method"],
        row["sampling_strategy"],
        row["reward_mode"],
        row["score_mode"],
    )


def load_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(existing: List[Dict[str, str]], new_rows: List[Dict[str, str]], overwrite: bool) -> List[Dict[str, str]]:
    merged = list(existing)
    existing_index = {dedup_key(row): idx for idx, row in enumerate(merged)}

    for row in new_rows:
        key = dedup_key(row)
        if key in existing_index:
            if overwrite:
                merged[existing_index[key]] = row
            continue
        existing_index[key] = len(merged)
        merged.append(row)
    return merged


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_csv).resolve()
    ensure_metrics_header(metrics_path)

    new_rows = []
    if args.results_md:
        results_path = Path(args.results_md).resolve()
        if results_path.exists():
            new_rows.extend(parse_results_md(results_path))
    if args.log_file:
        new_rows.append(
            {
                **default_row(),
                "notes": f"TODO: direct log parsing not implemented yet for {args.log_file}",
            }
        )

    existing_rows = load_existing_rows(metrics_path)
    merged_rows = merge_rows(existing_rows, new_rows, overwrite=args.overwrite)
    write_rows(metrics_path, merged_rows)
    print(f"rows_in_file={len(merged_rows)} new_rows_seen={len(new_rows)} overwrite={args.overwrite}")


if __name__ == "__main__":
    main()
