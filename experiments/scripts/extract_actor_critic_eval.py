import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_LOG_ROOT = REPO_ROOT / "experiments" / "results" / "raw_logs"
RESULTS_GLOB = "actor_critic_train_eval_*.results"

OUTPUT_FIELDS = [
    "run_id",
    "method",
    "reward_mode",
    "sampling_strategy",
    "score_mode",
    "critic_score_weight",
    "model_dir",
    "checkpoint",
    "eval_config",
    "search_limits",
    "R@1",
    "R@3",
    "R@5",
    "R@10",
    "R@20",
    "recall_all",
    "first_rank",
    "reached",
    "targets",
    "t_cnt",
    "log_path",
    "recommend_log_dir",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract actor-critic final-eval CSV rows from helper .results and [test-summary] logs."
    )
    parser.add_argument(
        "--results",
        help="Actor-critic helper .results file. Defaults to the newest actor_critic_train_eval_*.results.",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to experiments/results/final_eval_actor_critic_<RUN_ID>.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "Results" in parts:
        idx = parts.index("Results")
        candidate = REPO_ROOT.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate.resolve()
    if "experiments" in parts:
        idx = parts.index("experiments")
        candidate = REPO_ROOT.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate.resolve()
    return path


def repo_relative(path: Path | str) -> str:
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latest_results_file() -> Path:
    matches = sorted(RAW_LOG_ROOT.glob(RESULTS_GLOB), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No {RESULTS_GLOB} files found under {RAW_LOG_ROOT}")
    return matches[0]


def run_id_from_results_path(path: Path) -> str:
    match = re.search(r"actor_critic_train_eval_(.+)\.results$", path.name)
    return match.group(1) if match else path.stem


def parse_results_file(path: Path) -> List[Dict[str, str]]:
    common: Dict[str, str] = {"run_id": run_id_from_results_path(path)}
    blocks: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    def flush_current() -> None:
        if current.get("summary_log"):
            block = dict(common)
            block.update(current)
            blocks.append(block)

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in {"run_id", "model_dir", "checkpoint"} and not current:
            common[key] = value
            continue
        if key == "eval_config":
            flush_current()
            current = {"eval_config": value}
            continue
        if current:
            current[key] = value
        else:
            common[key] = value
    flush_current()

    if not blocks:
        raise ValueError(f"No eval blocks with summary_log found in {path}")
    return blocks


def extract_all_section(text: str, source: Path) -> Dict:
    marker = text.rfind("[all ")
    if marker == -1:
        raise ValueError(f"Missing [all N] section in {source}")
    brace = text.find("{", marker)
    if brace == -1:
        raise ValueError(f"Missing JSON payload after [all N] section in {source}")
    return json.loads(text[brace:].strip())


def parse_summary_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Summary log is empty: {path}")
    merged = extract_all_section(text, path)
    try:
        complete = merged["evaluation"]["stages"]["complete"]
        recall = complete["recall"]
    except KeyError as exc:
        raise ValueError(f"Missing expected evaluation keys in {path}: {exc}") from exc

    return {
        "R@1": str(recall.get("@01", "")),
        "R@3": str(recall.get("@03", "")),
        "R@5": str(recall.get("@05", "")),
        "R@10": str(recall.get("@10", "")),
        "R@20": str(recall.get("@20", "")),
        "recall_all": str(recall.get("all", "")),
        "first_rank": str(complete.get("first_rank", "")),
        "reached": str(complete.get("reached", "")),
        "targets": str(complete.get("targets", "")),
        "t_cnt": str(merged.get("t_cnt", "")),
    }


def config_path(eval_config: str) -> Path:
    name = Path(eval_config).stem
    return REPO_ROOT / "experiments" / "configs" / f"{name}.json"


def load_config(eval_config: str) -> Dict:
    path = config_path(eval_config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def value_from_config(config: Dict, section: str, key: str) -> str:
    value = config.get(section, {}).get(key, "")
    return "" if value is None else str(value)


def build_row(block: Dict[str, str]) -> Dict[str, str]:
    summary_path = resolve_path(block["summary_log"])
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary log does not exist: {summary_path}")

    config = load_config(block.get("eval_config", ""))
    summary = parse_summary_file(summary_path)

    score_mode = block.get("score_mode") or value_from_config(config, "actor_critic", "score_mode")
    critic_score_weight = block.get("critic_score_weight") or value_from_config(
        config, "actor_critic", "critic_score_weight"
    )

    row = {field: "" for field in OUTPUT_FIELDS}
    row.update(
        {
            "run_id": block.get("run_id", ""),
            "method": "actor_critic",
            "reward_mode": value_from_config(config, "reward", "mode") or "baseline",
            "sampling_strategy": value_from_config(config, "sampling", "strategy") or "greedy",
            "score_mode": score_mode,
            "critic_score_weight": critic_score_weight,
            "model_dir": repo_relative(resolve_path(block.get("model_dir", ""))) if block.get("model_dir") else "",
            "checkpoint": repo_relative(resolve_path(block.get("checkpoint", ""))) if block.get("checkpoint") else "",
            "eval_config": block.get("eval_config", ""),
            "search_limits": value_from_config(config, "search", "search_limits") or "e50-b4-na",
            "log_path": repo_relative(summary_path),
            "recommend_log_dir": repo_relative(resolve_path(block.get("recommend_log_dir", "")))
            if block.get("recommend_log_dir")
            else "",
            "notes": "actor-critic final eval extracted from helper .results and [test-summary]",
        }
    )
    row.update(summary)
    return row


def default_output_path(results_path: Path, blocks: List[Dict[str, str]]) -> Path:
    run_id = blocks[0].get("run_id") or run_id_from_results_path(results_path)
    return REPO_ROOT / "experiments" / "results" / f"final_eval_actor_critic_{run_id}.csv"


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
    results_path = resolve_path(args.results) if args.results else latest_results_file()
    blocks = parse_results_file(results_path)
    rows = [build_row(block) for block in blocks]
    output_path = resolve_path(args.output) if args.output else default_output_path(results_path, blocks)
    write_rows(rows, output_path, args.overwrite)
    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
