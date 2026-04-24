import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "experiments" / "results" / "metrics.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize normalized experiment metrics")
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS))
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_csv).resolve()
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        print("metrics.csv contains only the header.")
        return

    columns = [
        column
        for column in [
            "method",
            "sampling_strategy",
            "reward_mode",
            "score_mode",
            "R@1",
            "R@3",
            "R@5",
            "R@10",
            "R@20",
            "recall_all",
        ]
        if column in reader.fieldnames
    ]
    widths = {
        column: max(len(column), *(len((row.get(column) or "").strip()) for row in rows))
        for column in columns
    }
    header = " ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print(" ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" ".join((row.get(column) or "").strip().ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
