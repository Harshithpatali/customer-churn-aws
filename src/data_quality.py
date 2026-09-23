from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd

EXPECTED_COLUMNS = {"customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"}

def profile(df: pd.DataFrame) -> dict[str, Any]:
    missing=df.isna().sum()
    return {"rows":int(len(df)),"columns":int(df.shape[1]),"duplicate_rows":int(df.duplicated().sum()),"missing_cells":int(missing.sum()),"missing_by_column":{k:int(v) for k,v in missing[missing>0].items()},"unique_customers":int(df["customerID"].nunique()) if "customerID" in df else None,"target_distribution":df["Churn"].value_counts(dropna=False).to_dict() if "Churn" in df else {}}

def validate_schema(df: pd.DataFrame) -> None:
    missing=EXPECTED_COLUMNS-set(df.columns)
    if missing: raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df["customerID"].duplicated().any(): raise ValueError("customerID must be unique at ingestion time")
    if not set(df["Churn"].dropna().unique()).issubset({"Yes","No"}): raise ValueError("Churn must contain only Yes/No values")

def save_profile(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    result=profile(df); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)); return result
