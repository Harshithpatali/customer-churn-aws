from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODEL_PATH
from .pipeline import add_features


_ARTIFACT = None
_EXPLAINER = None


def load_artifact(
    path: Path = MODEL_PATH,
):

    global _ARTIFACT

    if _ARTIFACT is None:
        _ARTIFACT = joblib.load(path)

    return _ARTIFACT


def risk_band(
    probability: float,
    threshold: float = 0.2,
) -> str:

    if probability >= 0.70:
        return "High"

    if probability >= threshold:
        return "Medium"

    return "Low"


def recommendation(
    row: dict,
    probability: float,
) -> str:

    if probability >= 0.70:

        if row.get(
            "Contract"
        ) == "Month-to-month":

            return (
                "High-risk customer. "
                "Prioritize retention outreach "
                "and a contract-stability incentive."
            )

        if (
            row.get("TechSupport") == "No"
            and row.get("InternetService") != "No"
        ):

            return (
                "High-risk customer. "
                "Prioritize service-support "
                "outreach and technical assistance."
            )

        return (
            "High-risk customer. "
            "Investigate billing, service "
            "engagement and retention friction."
        )

    if probability >= 0.40:

        return (
            "Moderate-risk customer. "
            "Place into a monitored retention "
            "segment and review engagement."
        )

    return (
        "Low-risk customer. "
        "Continue normal lifecycle engagement."
    )


def _prepare(
    payload: dict,
) -> pd.DataFrame:

    frame = pd.DataFrame(
        [payload]
    )

    return add_features(frame)


def predict_one(
    payload: dict,
) -> dict:

    artifact = load_artifact()

    prepared = _prepare(
        payload
    )

    probability = float(
        artifact["pipeline"]
        .predict_proba(prepared)[:, 1][0]
    )

    threshold = float(
        artifact["threshold"]
    )

    return {
        "churn_probability":
            probability,
        "risk_band":
            risk_band(
                probability,
                threshold,
            ),
        "prediction":
            int(
                probability >= threshold
            ),
        "threshold":
            threshold,
        "estimated_monthly_revenue_at_risk":
            round(
                probability
                * float(
                    payload.get(
                        "MonthlyCharges",
                        0,
                    )
                ),
                2,
            ),
        "recommendation":
            recommendation(
                payload,
                probability,
            ),
        "model_version":
            artifact.get(
                "version",
                "unknown",
            ),
    }


def explain_one(
    payload: dict,
    top_n: int = 12,
) -> list[dict]:

    global _EXPLAINER

    artifact = load_artifact()

    pipeline = artifact[
        "pipeline"
    ]

    prepared = _prepare(
        payload
    )

    transformed = (
        pipeline
        .named_steps["preprocess"]
        .transform(prepared)
    )

    feature_names = (
        pipeline
        .named_steps["preprocess"]
        .get_feature_names_out()
    )

    model = (
        pipeline
        .named_steps["model"]
    )

    if _EXPLAINER is None:
        _EXPLAINER = (
            shap.TreeExplainer(
                model
            )
        )

    values = np.asarray(
        _EXPLAINER.shap_values(
            transformed
        )
    )

    if values.ndim == 3:
        values = values[1]

    values = values[0]

    order = (
        np.argsort(
            np.abs(values)
        )[::-1][:top_n]
    )

    result = []

    for index in order:

        value = float(
            values[index]
        )

        result.append(
            {
                "feature":
                    str(
                        feature_names[index]
                    ),
                "shap_value":
                    value,
                "direction":
                    (
                        "increases_risk"
                        if value > 0
                        else "decreases_risk"
                    ),
                "magnitude":
                    abs(value),
            }
        )

    return result