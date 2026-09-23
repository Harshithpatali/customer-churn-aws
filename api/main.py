from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import METRICS_PATH, MODEL_PATH
from src.model_service import explain_one, predict_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

app = FastAPI(title="Customer Churn Intelligence API", version="1.0.0", description="XGBoost churn prediction, SHAP explanations and retention intelligence.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Training metrics not found. Run scripts/train.py first.")
    return json.loads(METRICS_PATH.read_text())

@app.post("/predict")
def predict(payload: dict):
    try:
        return predict_one(payload)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(400, str(exc)) from exc

@app.post("/explain")
def explain(payload: dict):
    try:
        result = predict_one(payload)
        result["explanations"] = explain_one(payload)
        return result
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(400, str(exc)) from exc

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        rows = []
        for record in df.to_dict(orient="records"):
            rows.append({**record, **predict_one(record)})
        return {"rows": rows, "count": len(rows)}
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(400, str(exc)) from exc
