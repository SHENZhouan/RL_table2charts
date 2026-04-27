import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    if "update reward + updated policy" in lowered:
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
    if "epsilon_top_m" in lowered or "explore_top_m" in lowered:
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
    matches = re.findall(r"Complete recall info:\s*(\{.*\})", text)
    if not matches:
        return None
    try:
        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return None


def extract_first_float(text: str, patterns: List[str]) -> str:
    for pattern in patterns:
        value = extract_float(pattern, text)
        if value is not None:
            return value
    return ""


def parse_metadata_lines(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        key, sep, value = stripped[2:].partition(":")
        if not sep:
            continue
        metadata[key.strip()] = value.strip()
    return metadata


def split_subsections(section: str) -> Tuple[str, List[Tuple[str, str]]]:
    lines = section.splitlines()
    if not lines:
        return "", []

    title = lines[0].strip()
    rest = "\n".join(lines[1:])
    parts = re.split(r"(?m)^### ", rest)
    shared_body = parts[0]
    subsections: List[Tuple[str, str]] = []
    for part in parts[1:]:
        block = part.strip()
        if not block:
            continue
        block_lines = block.splitlines()
        subsections.append((block_lines[0].strip(), "\n".join(block_lines[1:])))
    return shared_body, subsections


def classify_block(section_title: str, subsection_title: str, shared_body: str, block_body: str) -> Tuple[str, str, str]:
    title_text = f"{section_title}\n{subsection_title}".strip()
    lowered = title_text.lower()
    if "update reward + updated policy" in lowered:
        return "update_reward_policy", "soft_reward", "epsilon_top_m"
    if "update reward only" in lowered:
        return "update_reward", "soft_reward", "greedy"
    if "updated policy" in lowered:
        return "updated_policy", "baseline", "epsilon_top_m"
    if "actor" in lowered and "critic" in lowered:
        return "actor_critic", "baseline", ""
    return infer_method(title_text), infer_reward_mode(f"{shared_body}\n{block_body}"), infer_sampling(title_text)


def populate_common_fields(row: Dict[str, str], section_title: str, shared_meta: Dict[str, str], block_title: str, block_body: str) -> None:
    title_for_run = f"{section_title}\n{block_title}"
    run_id_match = re.search(r"(\d{8}T\d{6}Z)", title_for_run)
    if run_id_match:
        row["run_id"] = run_id_match.group(1)
    row["notes"] = "parsed from results.md"

    if "sft_ckpt" in shared_meta:
        row["checkpoint"] = shared_meta["sft_ckpt"]

    block_meta = parse_metadata_lines(block_body)
    if "model_dir" in block_meta:
        row["checkpoint"] = block_meta["model_dir"]
    elif "rl_dir" in shared_meta:
        row["notes"] += f"; rl_dir={shared_meta['rl_dir']}"
    elif "eval_log_dir" in block_meta and not row["checkpoint"]:
        row["notes"] += f"; eval_log_dir={block_meta['eval_log_dir']}"

    if "critic_score_weight" in shared_meta:
        row["critic_score_weight"] = shared_meta["critic_score_weight"]
    if "actor_loss_weight" in shared_meta:
        row["notes"] += "; actor_loss_weight present"

    complete = parse_complete_recall(block_body)
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


def populate_policy_fields(row: Dict[str, str], shared_meta: Dict[str, str], block_title: str) -> None:
    title_lower = block_title.lower()
    if row["method"] == "updated_policy":
        row["epsilon_start"] = shared_meta.get("policy_epsilon_start", "")
        row["epsilon_end"] = shared_meta.get("policy_epsilon_end", "")
        row["epsilon_decay"] = shared_meta.get("policy_epsilon_decay", "")
        row["top_m"] = shared_meta.get("policy_explore_top_m", "")
        return

    if row["method"] == "update_reward_policy":
        combo = shared_meta.get("combo_policy", "")
        row["epsilon_start"] = extract_first_float(combo, [r"epsilon_start=([0-9.]+)"])
        row["epsilon_end"] = extract_first_float(combo, [r"epsilon_end=([0-9.]+)"])
        row["epsilon_decay"] = extract_first_float(combo, [r"epsilon_decay=([0-9.]+)"])
        row["top_m"] = extract_first_float(combo, [r"explore_top_m=([0-9.]+)"])
        return

    if row["sampling_strategy"] == "greedy" or "original greedy policy" in title_lower:
        row["epsilon_start"] = ""
        row["epsilon_end"] = ""
        row["epsilon_decay"] = ""
        row["top_m"] = ""


def populate_reward_fields(row: Dict[str, str], shared_meta: Dict[str, str]) -> None:
    if row["reward_mode"] != "soft_reward":
        return
    reward_spec = shared_meta.get("reward", "")
    row["exact_reward"] = extract_first_float(reward_spec, [r"exact=([0-9.]+)"])
    row["default_reward"] = extract_first_float(reward_spec, [r"default=([0-9.]+)"])
    row["same_token_reward"] = extract_first_float(reward_spec, [r"same_token=([0-9.]+)"])
    row["field_reward"] = extract_first_float(reward_spec, [r"field=([0-9.]+)"])
    row["same_field_type_reward"] = extract_first_float(reward_spec, [r"same_field_type=([0-9.]+)"])


def build_row(section_title: str, shared_body: str, block_title: str, block_body: str) -> Dict[str, str]:
    row = default_row()
    shared_meta = parse_metadata_lines(shared_body)
    row["method"], row["reward_mode"], row["sampling_strategy"] = classify_block(
        section_title, block_title, shared_body, block_body
    )
    row["score_mode"] = infer_score_mode(f"{section_title}\n{block_title}\n{shared_body}\n{block_body}")
    populate_common_fields(row, section_title, shared_meta, block_title, block_body)
    populate_policy_fields(row, shared_meta, block_title)
    populate_reward_fields(row, shared_meta)
    return row


def parse_results_md(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## ", text)
    rows = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        section_title = section.splitlines()[0].strip()
        shared_body, subsections = split_subsections(section)
        if subsections:
            for block_title, block_body in subsections:
                rows.append(build_row(section_title, shared_body, block_title, block_body))
        else:
            rows.append(build_row(section_title, shared_body, "", shared_body))

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


def is_results_md_row(row: Dict[str, str]) -> bool:
    return (row.get("notes") or "").startswith("parsed from results.md")


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
    if args.overwrite and args.results_md:
        existing_rows = [row for row in existing_rows if not is_results_md_row(row)]
    merged_rows = merge_rows(existing_rows, new_rows, overwrite=args.overwrite)
    write_rows(metrics_path, merged_rows)
    print(f"rows_in_file={len(merged_rows)} new_rows_seen={len(new_rows)} overwrite={args.overwrite}")


if __name__ == "__main__":
    main()
