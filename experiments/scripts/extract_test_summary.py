import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

FAMILY_CONFIG = {
    "epsilon_sweep": {
        "default_model_dirs_csv": REPO_ROOT / "experiments" / "results" / "epsilon_sweep_model_dirs_20260425.csv",
        "default_output": REPO_ROOT / "experiments" / "results" / "final_eval_epsilon_sweep_20260425.csv",
        "output_fields": [
            "experiment",
            "epsilon_start",
            "model_dir",
            "R@1",
            "R@3",
            "R@5",
            "R@10",
            "R@20",
            "recall_all",
            "first_rank",
            "t_cnt",
            "log_path",
            "notes",
        ],
        "passthrough_fields": ["epsilon_start", "model_dir"],
    },
    "reward_intensity": {
        "default_model_dirs_csv": REPO_ROOT / "experiments" / "results" / "reward_intensity_model_dirs_20260425.csv",
        "default_output": REPO_ROOT / "experiments" / "results" / "final_eval_reward_intensity_20260425.csv",
        "output_fields": [
            "experiment",
            "reward_mode",
            "sampling_strategy",
            "epsilon_start",
            "model_dir",
            "R@1",
            "R@3",
            "R@5",
            "R@10",
            "R@20",
            "recall_all",
            "first_rank",
            "t_cnt",
            "log_path",
            "notes",
        ],
        "passthrough_fields": ["reward_mode", "sampling_strategy", "epsilon_start", "model_dir"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized final-eval rows from test_agent_mp.py [test-summary] logs."
    )
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_CONFIG.keys()),
        default="epsilon_sweep",
        help="Experiment family schema and default file set to use.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional explicit [test-summary] log paths. If omitted, discover logs from --model-dirs-csv.",
    )
    parser.add_argument(
        "--model-dirs-csv",
        help="Optional mapping CSV. Defaults depend on --family.",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV output path. If omitted, rows are printed to stdout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output instead of refusing when the file already exists.",
    )
    parser.add_argument(
        "--note",
        default="formal test_agent_mp.py final eval extracted from tracked [test-summary] log",
        help="Notes text appended to extracted rows.",
    )
    return parser.parse_args()


def family_defaults(family: str) -> Dict:
    return FAMILY_CONFIG[family]


def load_model_dir_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def normalize_model_dir(model_dir_value: str) -> Path:
    model_dir = Path(model_dir_value)
    if not model_dir.is_absolute():
        return (REPO_ROOT / model_dir).resolve()
    if model_dir.exists():
        return model_dir.resolve()
    parts = model_dir.parts
    if "Results" in parts:
        idx = parts.index("Results")
        candidate = REPO_ROOT.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate.resolve()
    fallback = REPO_ROOT / "Results" / "Models" / model_dir.name
    if fallback.exists():
        return fallback.resolve()
    return model_dir.resolve()


def discover_summary_log(row: Dict[str, str]) -> Path:
    model_dir = normalize_model_dir(row["model_dir"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    matches = sorted(model_dir.glob("evaluations/test-*/[[]test-summary]*.log"))
    if not matches:
        raise FileNotFoundError(f"No [test-summary] logs found under {model_dir}/evaluations")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one [test-summary] log under {model_dir}/evaluations, found {len(matches)}: "
            + ", ".join(str(match) for match in matches)
        )
    return matches[0]


def extract_all_section(text: str, source: Path) -> Dict:
    marker = text.rfind("[all ")
    if marker == -1:
        raise ValueError(f"Missing [all N] section in {source}")
    brace = text.find("{", marker)
    if brace == -1:
        raise ValueError(f"Missing JSON payload after [all N] section in {source}")
    payload = text[brace:].strip()
    if not payload:
        raise ValueError(f"Empty JSON payload in {source}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload in {source}: {exc}") from exc


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
        "R@1": str(recall["@01"]),
        "R@3": str(recall["@03"]),
        "R@5": str(recall["@05"]),
        "R@10": str(recall["@10"]),
        "R@20": str(recall["@20"]),
        "recall_all": str(recall["all"]),
        "first_rank": str(complete["first_rank"]),
        "t_cnt": str(merged["t_cnt"]),
    }


def base_experiment_name(config_value: str) -> str:
    return Path(config_value).stem


def combine_notes(*parts: str) -> str:
    values = [part.strip() for part in parts if part and part.strip()]
    return "; ".join(values)


def rows_from_explicit_paths(paths: Sequence[str], family: str, note: str) -> List[Dict[str, str]]:
    defaults = family_defaults(family)
    rows: List[Dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Summary log does not exist: {path}")
        summary = parse_summary_file(path)
        model_dir = path.parents[2]
        row = {field: "" for field in defaults["output_fields"]}
        row["experiment"] = path.parent.name.removeprefix("test-")
        row["model_dir"] = repo_relative(model_dir)
        row["log_path"] = repo_relative(path)
        row["notes"] = note
        row.update(summary)
        rows.append(row)
    return rows


def rows_from_model_dirs(mapping_rows: Iterable[Dict[str, str]], family: str, note: str) -> List[Dict[str, str]]:
    defaults = family_defaults(family)
    rows: List[Dict[str, str]] = []
    for mapping in mapping_rows:
        log_path = discover_summary_log(mapping)
        summary = parse_summary_file(log_path)
        row = {field: "" for field in defaults["output_fields"]}
        row["experiment"] = base_experiment_name(mapping["config"])
        for field in defaults["passthrough_fields"]:
            if field == "model_dir":
                row[field] = repo_relative(normalize_model_dir(mapping[field]))
            elif field in mapping:
                row[field] = mapping[field]
        row.update(summary)
        row["log_path"] = repo_relative(log_path)
        row["notes"] = combine_notes(mapping.get("notes", ""), note)
        rows.append(row)
    return rows


def write_rows(rows: List[Dict[str, str]], output_fields: Sequence[str], output_path: Path | None, overwrite: bool) -> None:
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_path}")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    defaults = family_defaults(args.family)
    model_dirs_csv = Path(args.model_dirs_csv) if args.model_dirs_csv else defaults["default_model_dirs_csv"]

    if args.paths:
        rows = rows_from_explicit_paths(args.paths, args.family, args.note)
    else:
        mapping_rows = load_model_dir_rows(model_dirs_csv)
        rows = rows_from_model_dirs(mapping_rows, args.family, args.note)

    output_path = Path(args.output) if args.output else None
    write_rows(rows, defaults["output_fields"], output_path, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
