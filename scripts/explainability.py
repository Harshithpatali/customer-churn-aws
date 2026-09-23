from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODEL_PATH, DATA_PATH
from src.pipeline import add_features


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    df = add_features(pd.read_csv(DATA_PATH)).drop(columns=["Churn"], errors="ignore")
    df = df.drop(columns=["customerID"], errors="ignore")
    pipe = artifact["pipeline"]
    x = pipe.named_steps["preprocess"].transform(df)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]
    sample = x[: min(2000, len(x))]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample))
    if values.ndim == 3:
        values = values[1]
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    out = ROOT / "reports" / "shap_global_importance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
