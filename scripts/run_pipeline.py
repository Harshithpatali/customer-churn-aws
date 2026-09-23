from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from services.data_pipeline.pipeline import build_dataset
from src.statistics import run_statistics
from src.pipeline import train

if __name__=="__main__":
    data=ROOT/"data/raw/Telco-Customer-Churn.csv"
    print("[1/3] Data pipeline")
    processed=ROOT/"data/processed/model_ready.csv"
    processed.parent.mkdir(parents=True,exist_ok=True)
    df=build_dataset(data,ROOT/"reports/data_quality.json")
    df.to_csv(processed,index=False)
    print(f"processed shape={df.shape}")
    print("[2/3] Statistical analysis")
    run_statistics(data,ROOT/"reports")
    print("[3/3] ML training")
    print(train(data,ROOT/"models",ROOT/"reports"))
