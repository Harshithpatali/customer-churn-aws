from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA = ROOT / "data/raw/Telco-Customer-Churn.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download the official IBM CSV before training")
    args = parser.parse_args()
    if args.download and not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading IBM dataset -> {DATA}")
        urllib.request.urlretrieve(URL, DATA)
    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}. Download it from {URL} or run with --download.")
    from src.pipeline import train
    report = train(DATA, ROOT / "models")
    print("Training complete")
    print(report)

if __name__ == "__main__":
    main()
