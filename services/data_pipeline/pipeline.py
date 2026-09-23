from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data_quality import save_profile, validate_schema

def load_raw(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path); validate_schema(df); return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy(); out.columns=[c.strip() for c in out.columns]
    out["TotalCharges"]=pd.to_numeric(out["TotalCharges"],errors="coerce")
    out=out.drop_duplicates(subset=["customerID"]).reset_index(drop=True)
    out["Churn"]=out["Churn"].map({"Yes":1,"No":0}).astype(int)
    return out

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy(); out["TotalCharges"]=pd.to_numeric(out["TotalCharges"],errors="coerce")
    yes_cols=["PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    out["service_count"]=(out[yes_cols]=="Yes").sum(axis=1)
    out["tenure_group"]=pd.cut(out["tenure"],bins=[-1,6,12,24,48,72],labels=["0-6","7-12","13-24","25-48","49+"])
    out["monthly_to_tenure_value"]=out["MonthlyCharges"]/(out["tenure"]+1)
    out["avg_monthly_revenue"]=out["TotalCharges"]/out["tenure"].replace(0,1)
    out["contract_risk_flag"]=(out["Contract"]=="Month-to-month").astype(int)
    out["fiber_risk_flag"]=(out["InternetService"]=="Fiber optic").astype(int)
    out["support_gap_flag"]=((out["InternetService"]!="No")&(out["TechSupport"]=="No")).astype(int)
    return out

def build_dataset(raw_path: Path, report_path: Path|None=None) -> pd.DataFrame:
    raw=load_raw(raw_path)
    if report_path: save_profile(raw,report_path)
    return engineer(clean(raw))
