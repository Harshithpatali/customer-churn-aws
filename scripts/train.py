from __future__ import annotations
import argparse
import sys
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA=ROOT/"data/raw/Telco-Customer-Churn.csv"

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--download",action="store_true"); args=parser.parse_args()
    if args.download:
        DATA.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve(URL,DATA)
    if not DATA.exists(): raise SystemExit(f"Dataset not found: {DATA}")
    from src.pipeline import train
    print(train(DATA,ROOT/"models",ROOT/"reports"))

if __name__=="__main__": main()
