from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.statistics import run_statistics

if __name__=="__main__":
    print(run_statistics(ROOT/"data/raw/Telco-Customer-Churn.csv",ROOT/"reports"))
