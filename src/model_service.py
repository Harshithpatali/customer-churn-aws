from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODEL_PATH

_ARTIFACT = None

def load_artifact(path: Path = MODEL_PATH):
    global _ARTIFACT
    if _ARTIFACT is None:
        _ARTIFACT = joblib.load(path)
    return _ARTIFACT


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def recommendation(row: dict, probability: float) -> str:
    if probability >= 0.70:
        if row.get("Contract") == "Month-to-month":
            return "Prioritize retention outreach and offer a contract-stability incentive."
        if row.get("TechSupport") == "No" and row.get("InternetService") != "No":
            return "Prioritize service-support outreach and a technical assistance offer."
        return "Prioritize proactive retention outreach and investigate billing/service friction."
    if probability >= 0.40:
        return "Place in a monitored retention segment and review service engagement."
    return "No immediate intervention; continue normal lifecycle engagement."


def predict_one(payload: dict):
    artifact = load_artifact()
    df = pd.DataFrame([payload])
    from .pipeline import add_features
    df = add_features(df)
    p = float(artifact["pipeline"].predict_proba(df)[:, 1][0])
    return {"churn_probability": p, "risk_band": risk_band(p), "prediction": int(p >= artifact["threshold"]), "threshold": artifact["threshold"], "recommendation": recommendation(payload, p)}


def explain_one(payload: dict, top_n: int = 10):
    artifact = load_artifact()
    from .pipeline import add_features
    df = add_features(pd.DataFrame([payload]))
    pipe = artifact["pipeline"]
    X_t = pipe.named_steps["preprocess"].transform(df)
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    values = explainer.shap_values(X_t)
    if isinstance(values, list):
        values = values[1]
    vals = np.asarray(values)[0]
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    return [{"feature": str(feature_names[i]), "shap_value": float(vals[i]), "direction": "increases_risk" if vals[i] > 0 else "decreases_risk"} for i in order]
