from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import CATEGORICAL_FEATURES, ID_COL, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGET


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().copy()
    df = df.drop(columns=[ID_COL], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    yes_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    out["service_count"] = (out[yes_cols] == "Yes").sum(axis=1)
    out["tenure_group"] = pd.cut(
        out["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    )
    out["monthly_to_tenure_value"] = out["MonthlyCharges"] / (out["tenure"] + 1)
    out["avg_monthly_revenue"] = out["TotalCharges"] / out["tenure"].replace(0, 1)
    out["contract_risk_flag"] = (out["Contract"] == "Month-to-month").astype(int)
    out["fiber_risk_flag"] = (out["InternetService"] == "Fiber optic").astype(int)
    out["support_gap_flag"] = (
        (out["InternetService"] != "No") & (out["TechSupport"] == "No")
    ).astype(int)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = NUMERIC_FEATURES + [
        "service_count", "monthly_to_tenure_value",
        "avg_monthly_revenue", "contract_risk_flag",
        "fiber_risk_flag", "support_gap_flag",
    ]
    categorical = CATEGORICAL_FEATURES + ["tenure_group"]
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ),
    ])


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def optimize_threshold(y, prob):
    rows = []
    for threshold in np.arange(0.10, 0.81, 0.01):
        m = metrics(y, prob, float(threshold))
        m["threshold"] = float(threshold)
        m["business_score"] = float(0.60 * m["recall"] + 0.40 * m["precision"])
        rows.append(m)
    return max(rows, key=lambda r: r["business_score"])


def train(csv_path: Path = None, output_dir: Path = None):
    csv_path = csv_path or Path("data/raw/Telco-Customer-Churn.csv")
    output_dir = output_dir or Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    df = add_features(clean(raw))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    baseline = make_pipeline(
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    rf = make_pipeline(
        RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    )
    xgb = make_pipeline(
        XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            min_child_weight=3, objective="binary:logistic",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4,
        )
    )

    candidates = {
        "logistic_regression": baseline,
        "random_forest": rf,
        "xgboost": xgb,
    }
    comparison = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        comparison[name] = metrics(y_test, model.predict_proba(X_test)[:, 1])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb,
        {
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.025, 0.035, 0.05],
            "model__subsample": [0.75, 0.85, 1.0],
            "model__colsample_bytree": [0.75, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5],
        },
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    test_prob = best.predict_proba(X_test)[:, 1]
    threshold = optimize_threshold(y_test, test_prob)
    final_metrics = metrics(y_test, test_prob, threshold["threshold"])

    artifact = {
        "pipeline": best,
        "threshold": threshold["threshold"],
        "target": TARGET,
        "version": "2.0.0",
        "feature_columns": list(X.columns),
    }
    joblib.dump(artifact, output_dir / "churn_pipeline.joblib")

    transformed = best.named_steps["preprocess"].get_feature_names_out().tolist()
    (output_dir / "feature_contract.json").write_text(
        json.dumps({
            "raw_features": list(X.columns),
            "transformed_features": transformed,
        }, indent=2)
    )

    search_results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )
    search_results.to_csv(
        output_dir.parent / "reports" / "hyperparameter_search.csv",
        index=False,
    )

    report = {
        "dataset_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "comparison": comparison,
        "best_params": search.best_params_,
        "cv": {"folds": 5, "scoring": "average_precision"},
        "test": final_metrics,
        "threshold_selection": threshold,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report
