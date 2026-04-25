import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIRS = REPO_ROOT / "experiments" / "results" / "epsilon_sweep_model_dirs_20260425.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "results" / "final_eval_epsilon_sweep_20260425.csv"

OUTPUT_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized final-eval rows from test_agent_mp.py [test-summary] logs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional explicit [test-summary] log paths. If omitted, discover logs from --model-dirs-csv.",
    )
    parser.add_argument(
        "--model-dirs-csv",
        default=str(DEFAULT_MODEL_DIRS),
        help="CSV mapping of epsilon configs to model directories used for log discovery.",
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
        help="Notes text to attach to every extracted row.",
    )
    return parser.parse_args()


def load_model_dir_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def model_dir_path(model_dir_value: str) -> Path:
    model_dir = Path(model_dir_value)
    if model_dir.is_absolute():
        return model_dir
    return REPO_ROOT / model_dir


def discover_summary_log(row: Dict[str, str]) -> Path:
    model_dir = model_dir_path(row["model_dir"])
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


def parse_summary_file(path: Path) -> Dict:
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


def derive_experiment_from_path(path: Path) -> str:
    test_dir = path.parent.name
    if test_dir.startswith("test-"):
        return test_dir[len("test-") :]
    return test_dir


def rows_from_explicit_paths(paths: Sequence[str], note: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Summary log does not exist: {path}")
        summary = parse_summary_file(path)
        model_dir = path.parents[2]
        experiment = derive_experiment_from_path(path)
        rows.append(
            {
                "experiment": experiment,
                "epsilon_start": "",
                "model_dir": repo_relative(model_dir),
                **summary,
                "log_path": repo_relative(path),
                "notes": note,
            }
        )
    return rows


def rows_from_model_dirs(mapping_rows: Iterable[Dict[str, str]], note: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for mapping in mapping_rows:
        log_path = discover_summary_log(mapping)
        summary = parse_summary_file(log_path)
        config_path = Path(mapping["config"])
        rows.append(
            {
                "experiment": config_path.stem,
                "epsilon_start": mapping["epsilon_start"],
                "model_dir": mapping["model_dir"],
                **summary,
                "log_path": repo_relative(log_path),
                "notes": note,
            }
        )
    return rows


def epsilon_sort_key(row: Dict[str, str]) -> float:
    try:
        return float(row["epsilon_start"])
    except ValueError:
        return float("inf")


def write_rows(rows: List[Dict[str, str]], output_path: Optional[Path], overwrite: bool) -> None:
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_path}")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.paths:
        rows = rows_from_explicit_paths(args.paths, args.note)
    else:
        mapping_rows = load_model_dir_rows(Path(args.model_dirs_csv))
        rows = rows_from_model_dirs(mapping_rows, args.note)

    rows.sort(key=epsilon_sort_key)
    output_path = Path(args.output) if args.output else None
    write_rows(rows, output_path, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
