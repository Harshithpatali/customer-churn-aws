from __future__ import annotations

import io
import json
import logging
import time

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import METRICS_PATH, MODEL_PATH
from src.model_service import explain_one, predict_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

app = FastAPI(
    title="Customer Churn Intelligence Platform",
    version="2.0.0",
    description="Production-style churn prediction, statistical intelligence and SHAP explainability API.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "churn-api", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/ready")
def ready():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model artifact unavailable")
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Training metrics not found")
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
        df = pd.read_csv(io.BytesIO(await file.read()))
        started = time.perf_counter()
        rows = [{**r, **predict_one(r)} for r in df.to_dict(orient="records")]
        return {
            "rows": rows,
            "count": len(rows),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(400, str(exc)) from exc
