from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import shap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import add_features

if __name__=="__main__":
    artifact=joblib.load(ROOT/"models/churn_pipeline.joblib")
    raw=pd.read_csv(ROOT/"data/raw/Telco-Customer-Churn.csv")
    sample=add_features(raw).drop(columns=["Churn","customerID"],errors="ignore").sample(min(600,len(raw)),random_state=42)
    pipe=artifact["pipeline"]
    transformed=pipe.named_steps["preprocess"].transform(sample)
    names=pipe.named_steps["preprocess"].get_feature_names_out()
    explainer=shap.TreeExplainer(pipe.named_steps["model"])
    values=np.asarray(explainer.shap_values(transformed))
    if values.ndim==3: values=values[1]
    values=values if values.ndim==2 else values.reshape(1,-1)
    output=pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
    output.to_csv(ROOT/"reports/shap_global_importance.csv",index=False)
    print(output.head(20).to_string(index=False))
