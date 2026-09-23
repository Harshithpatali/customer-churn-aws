from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu

TARGET = "Churn"


def _effect_size_chi2(stat: float, n: int, r: int, k: int) -> float:
    return float(np.sqrt(stat / max(n * max(min(r - 1, k - 1), 1), 1)))


def categorical_tests(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    categorical = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in {"customerID", target}
    ]
    for feature in categorical:
        table = pd.crosstab(df[feature], df[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        stat, p_value, dof, _ = chi2_contingency(table)
        rows.append({
            "feature": feature,
            "test": "chi_square",
            "statistic": float(stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "cramers_v": _effect_size_chi2(stat, len(df), table.shape[0], table.shape[1]),
            "significant_at_0_05": bool(p_value < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_value")


def numeric_tests(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    for feature in numeric:
        a = df.loc[df[target] == 0, feature].dropna()
        b = df.loc[df[target] == 1, feature].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "feature": feature,
            "test": "mann_whitney_u",
            "statistic": float(stat),
            "p_value": float(p_value),
            "non_churn_median": float(a.median()),
            "churn_median": float(b.median()),
            "significant_at_0_05": bool(p_value < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_value")


def logistic_inference(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    work = df.copy()
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [
        c for c in [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ] if c in work
    ]
    x = pd.get_dummies(work[numeric + categorical], columns=categorical, drop_first=True, dtype=float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True)).fillna(0)
    y = work[target].astype(int)
    x = sm.add_constant(x, has_constant="add")
    model = sm.Logit(y, x).fit(disp=False, maxiter=300)
    conf = model.conf_int()
    return pd.DataFrame({
        "feature": model.params.index,
        "coefficient": model.params.values,
        "odds_ratio": np.exp(model.params.values),
        "p_value": model.pvalues.values,
        "ci_lower": np.exp(conf[0].values),
        "ci_upper": np.exp(conf[1].values),
    }).sort_values("p_value")


def run_statistical_analysis(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cat = categorical_tests(df)
    num = numeric_tests(df)
    odds = logistic_inference(df)
    cat.to_csv(output_dir / "categorical_significance.csv", index=False)
    num.to_csv(output_dir / "numeric_significance.csv", index=False)
    odds.to_csv(output_dir / "statistical_logit_odds_ratios.csv", index=False)
    summary = {
        "sample_size": int(len(df)),
        "churn_rate": float(df[TARGET].mean()),
        "categorical_tests": int(len(cat)),
        "numeric_tests": int(len(num)),
        "significant_categorical": int(cat["significant_at_0_05"].sum()) if not cat.empty else 0,
        "significant_numeric": int(num["significant_at_0_05"].sum()) if not num.empty else 0,
        "logit_aic": float(sm.Logit(
            df[TARGET].astype(int),
            sm.add_constant(
                pd.get_dummies(
                    df.drop(columns=["customerID", TARGET]),
                    drop_first=True,
                    dtype=float
                ).replace([np.inf, -np.inf], np.nan).fillna(0),
                has_constant="add"
            )
        ).fit(disp=False, maxiter=300).aic),
    }
    (output_dir / "statistical_summary.json").write_text(__import__("json").dumps(summary, indent=2))
    return summary
