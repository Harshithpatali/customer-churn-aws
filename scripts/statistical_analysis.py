from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.statistics import run_statistical_analysis

DATA = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    print(run_statistical_analysis(df, REPORTS))


if __name__ == "__main__":
    main()
