from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODEL_PATH

_ARTIFACT=None
_EXPLAINER=None

def load_artifact(path:Path=MODEL_PATH):
    global _ARTIFACT
    if _ARTIFACT is None:
        _ARTIFACT=joblib.load(path)
    return _ARTIFACT

def risk_band(p:float,threshold:float=0.2)->str:
    if p>=0.70: return "High"
    if p>=threshold: return "Medium"
    return "Low"

def recommendation(row:dict,probability:float)->str:
    if probability>=0.70:
        if row.get("Contract")=="Month-to-month": return "Prioritize retention outreach and offer a contract-stability incentive."
        if row.get("TechSupport")=="No" and row.get("InternetService")!="No": return "Prioritize service-support outreach and a technical assistance offer."
        return "Prioritize proactive retention outreach and investigate billing or service friction."
    if probability>=0.40: return "Place in a monitored retention segment and review service engagement."
    return "No immediate intervention; continue normal lifecycle engagement."

def _prepare(payload):
    from .pipeline import add_features
    return add_features(pd.DataFrame([payload]))

def predict_one(payload:dict):
    artifact=load_artifact(); df=_prepare(payload); p=float(artifact["pipeline"].predict_proba(df)[:,1][0]); threshold=float(artifact["threshold"])
    return {"churn_probability":p,"risk_band":risk_band(p,threshold),"prediction":int(p>=threshold),"threshold":threshold,"estimated_monthly_revenue_at_risk":round(p*float(payload.get("MonthlyCharges",0)),2),"recommendation":recommendation(payload,p)}

def explain_one(payload:dict,top_n:int=10):
    global _EXPLAINER
    artifact=load_artifact(); pipe=artifact["pipeline"]; transformed=pipe.named_steps["preprocess"].transform(_prepare(payload)); names=pipe.named_steps["preprocess"].get_feature_names_out()
    if _EXPLAINER is None: _EXPLAINER=shap.TreeExplainer(pipe.named_steps["model"])
    values=np.asarray(_EXPLAINER.shap_values(transformed))
    if values.ndim==3: values=values[1]
    values=values[0]; order=np.argsort(np.abs(values))[::-1][:top_n]
    return [{"feature":str(names[i]),"shap_value":float(values[i]),"direction":"increases_risk" if values[i]>0 else "decreases_risk","magnitude":float(abs(values[i]))} for i in order]
