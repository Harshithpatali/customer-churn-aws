from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.statistics import run_statistics
from src.pipeline import train

if __name__ == "__main__":
    data = ROOT / "data/raw/Telco-Customer-Churn.csv"
    print("[1/2] Statistical analysis")
    run_statistics(data, ROOT / "reports")
    print("[2/2] ML training")
    train(data, ROOT / "models", ROOT / "reports")
    print("Pipeline complete")
