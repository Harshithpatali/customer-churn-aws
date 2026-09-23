from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from services.data_pipeline.pipeline import build_dataset

if __name__=="__main__":
    output=ROOT/"data/processed/model_ready.csv"; output.parent.mkdir(parents=True,exist_ok=True)
    df=build_dataset(ROOT/"data/raw/Telco-Customer-Churn.csv",ROOT/"reports/data_quality.json")
    df.to_csv(output,index=False); print(f"saved {output} shape={df.shape}")
