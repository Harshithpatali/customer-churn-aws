from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the churn data pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/Telco-Customer-Churn.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model_ready.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_quality.json"))
    args = parser.parse_args()

    df = build_dataset(args.input, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
