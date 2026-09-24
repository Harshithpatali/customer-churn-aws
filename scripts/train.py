from __future__ import annotations

import argparse
import sys
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

DATA = (
    ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
    )

    args = parser.parse_args()

    if args.download:

        DATA.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Downloading Telco dataset..."
        )

        urllib.request.urlretrieve(
            URL,
            DATA,
        )

    if not DATA.exists():

        raise SystemExit(
            f"Dataset not found: {DATA}"
        )

    from src.pipeline import (
        build_dataset,
        train,
    )

    from src.statistics import (
        run_statistical_analysis,
    )

    print(
        "\n"
        "====================================\n"
        " CUSTOMER CHURN DATA SCIENCE PIPELINE\n"
        "====================================\n"
    )

    print(
        "[1/3] Training ML models..."
    )

    report = train(
        DATA,
        ROOT / "models",
        ROOT / "reports",
    )

    print(
        "[2/3] Running statistical analysis..."
    )

    dataset = build_dataset(
        DATA
    )

    statistical = (
        run_statistical_analysis(
            dataset,
            ROOT / "reports",
        )
    )

    report[
        "statistical_model"
    ] = statistical

    (
        ROOT
        / "models"
        / "metrics.json"
    ).write_text(
        __import__("json").dumps(
            report,
            indent=2,
        )
    )

    print(
        "[3/3] Training complete."
    )

    print(
        "\nModel performance:"
    )

    print(
        report["test"]
    )

    print(
        "\nCross-validation:"
    )

    print(
        report["cv"]
    )

    print(
        "\nStatistical summary:"
    )

    print(
        statistical["summary"]
    )


if __name__ == "__main__":
    main()