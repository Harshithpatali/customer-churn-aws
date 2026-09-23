from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.data_pipeline.pipeline import build_dataset
from src.statistics import run_statistics
from src.pipeline import train


if __name__ == "__main__":
    raw = ROOT / "data/raw/Telco-Customer-Churn.csv"

    print("[1/3] Data quality + feature pipeline")
    df = build_dataset(raw, ROOT / "reports/data_quality.json")
    processed = ROOT / "data/processed/model_ready.csv"
    processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed, index=False)
    print(f"saved {processed} shape={df.shape}")

    print("[2/3] Statistical analysis")
    run_statistics(raw, ROOT / "reports")

    print("[3/3] ML training")
    report = train(raw, ROOT / "models")
    print(f"ROC-AUC={report['test']['roc_auc']:.4f}")
    print(f"PR-AUC={report['test']['pr_auc']:.4f}")
    print(f"threshold={report['threshold_selection']['threshold']:.2f}")

    print("Production training pipeline complete.")
