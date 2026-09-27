"""Summarize a YOLO training run from its results.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def load_rows(results_csv: Path) -> list[dict[str, float]]:
    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {key.strip(): float(value) for key, value in row.items()}
            for row in reader
        ]


def format_row(row: dict[str, float]) -> str:
    values = "  ".join(
        f"{name}={row[column]:.3f}" for name, column in METRICS.items()
    )
    return f"epoch {int(row['epoch']):>4}  {values}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print final and best-epoch metrics from a YOLO results.csv."
    )
    parser.add_argument("results_csv", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.results_csv}")

    best = max(rows, key=lambda row: row[METRICS["mAP50-95"]])

    print(f"Run: {args.results_csv.parent}")
    print(f"Epochs logged: {len(rows)}")
    print(f"Final  {format_row(rows[-1])}")
    print(f"Best   {format_row(best)}  (selected by mAP50-95)")


if __name__ == "__main__":
    main()
