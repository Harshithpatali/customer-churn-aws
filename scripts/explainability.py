from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import add_features


if __name__ == "__main__":
    artifact = joblib.load(ROOT / "models/churn_pipeline.joblib")
    raw = pd.read_csv(ROOT / "data/raw/Telco-Customer-Churn.csv")
    df = add_features(raw)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore").sample(
        min(600, len(df)), random_state=42
    )
    pipe = artifact["pipeline"]
    X_t = pipe.named_steps["preprocess"].transform(X)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    values = np.asarray(explainer.shap_values(X_t))
    values = values[1] if values.ndim == 3 else values
    importance = np.abs(values).mean(axis=0)
    out = pd.DataFrame(
        {"feature": names, "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    out.to_csv(ROOT / "reports/shap_global_importance.csv", index=False)
    print(out.head(20).to_string(index=False))
