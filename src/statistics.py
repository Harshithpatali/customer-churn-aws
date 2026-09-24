from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency,
    mannwhitneyu,
)


TARGET = "Churn"


def cramers_v(
    statistic: float,
    n: int,
    rows: int,
    columns: int,
) -> float:

    denominator = max(
        n
        * max(
            min(
                rows - 1,
                columns - 1,
            ),
            1,
        ),
        1,
    )

    return float(
        np.sqrt(
            statistic
            / denominator
        )
    )


def categorical_tests(
    df: pd.DataFrame,
    target: str = TARGET,
) -> pd.DataFrame:

    rows = []

    categorical = [
        column
        for column in df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns
        if column not in {
            "customerID",
            target,
        }
    ]

    for feature in categorical:

        table = pd.crosstab(
            df[feature],
            df[target],
        )

        if (
            table.shape[0] < 2
            or table.shape[1] < 2
        ):
            continue

        statistic, p_value, dof, _ = (
            chi2_contingency(table)
        )

        rows.append(
            {
                "feature": feature,
                "test": "Chi-square",
                "statistic":
                    float(statistic),
                "p_value":
                    float(p_value),
                "degrees_of_freedom":
                    int(dof),
                "cramers_v":
                    cramers_v(
                        statistic,
                        len(df),
                        table.shape[0],
                        table.shape[1],
                    ),
                "significant_at_0_05":
                    bool(
                        p_value < 0.05
                    ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "p_value"
        )
        .reset_index(
            drop=True
        )
    )


def numeric_tests(
    df: pd.DataFrame,
    target: str = TARGET,
) -> pd.DataFrame:

    rows = []

    numeric = [
        column
        for column in df.select_dtypes(
            include=np.number
        ).columns
        if column != target
    ]

    for feature in numeric:

        non_churn = (
            df.loc[
                df[target] == 0,
                feature,
            ]
            .dropna()
        )

        churn = (
            df.loc[
                df[target] == 1,
                feature,
            ]
            .dropna()
        )

        if (
            len(non_churn) == 0
            or len(churn) == 0
        ):
            continue

        statistic, p_value = (
            mannwhitneyu(
                non_churn,
                churn,
                alternative="two-sided",
            )
        )

        rows.append(
            {
                "feature": feature,
                "test":
                    "Mann-Whitney U",
                "statistic":
                    float(statistic),
                "p_value":
                    float(p_value),
                "non_churn_median":
                    float(
                        non_churn.median()
                    ),
                "churn_median":
                    float(
                        churn.median()
                    ),
                "median_difference":
                    float(
                        churn.median()
                        - non_churn.median()
                    ),
                "significant_at_0_05":
                    bool(
                        p_value < 0.05
                    ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "p_value"
        )
        .reset_index(
            drop=True
        )
    )


def prepare_logistic_data(
    df: pd.DataFrame,
    target: str = TARGET,
):

    numeric = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical = [
        column
        for column in [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
        if column in df.columns
    ]

    X = pd.get_dummies(
        df[
            numeric + categorical
        ],
        columns=categorical,
        drop_first=True,
        dtype=float,
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        X.median()
    )

    X = X.fillna(0)

    # Remove zero-variance columns.
    variance = X.var()

    X = X.loc[
        :,
        variance > 1e-12,
    ]

    # Remove duplicate columns.
    X = X.T.drop_duplicates().T

    # Remove columns that are perfectly
    # correlated with another column.
    correlation = X.corr().abs()

    upper = correlation.where(
        np.triu(
            np.ones(
                correlation.shape,
                dtype=bool,
            ),
            k=1,
        )
    )

    to_drop = [
        column
        for column in upper.columns
        if any(
            upper[column] > 0.999999
        )
    ]

    if to_drop:
        X = X.drop(
            columns=to_drop
        )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    y = df[target].astype(int)

    return X, y


def logistic_inference(
    df: pd.DataFrame,
    target: str = TARGET,
) -> pd.DataFrame:

    X, y = prepare_logistic_data(
        df,
        target,
    )

    try:

        model = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(),
        ).fit(
            disp=False
        )

        confidence = (
            model.conf_int()
        )

        result = pd.DataFrame(
            {
                "feature":
                    model.params.index,

                "coefficient":
                    model.params.values,

                "odds_ratio":
                    np.exp(
                        model.params.values
                    ),

                "p_value":
                    model.pvalues.values,

                "ci_lower":
                    np.exp(
                        confidence[0].values
                    ),

                "ci_upper":
                    np.exp(
                        confidence[1].values
                    ),
            }
        )

        result[
            "significant_at_0_05"
        ] = (
            result["p_value"]
            < 0.05
        )

        return (
            result
            .sort_values(
                "p_value"
            )
            .reset_index(
                drop=True
            )
        )

    except Exception as exc:

        print(
            "WARNING: Classical GLM "
            f"inference failed: {exc}"
        )

        # Regularized GLM is used only as
        # a numerical fallback. It provides
        # stable coefficients but not valid
        # classical p-values/confidence
        # intervals.
        model = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(),
        ).fit_regularized(
            alpha=0.01,
            L1_wt=0.0,
        )

        coefficients = model.params

        result = pd.DataFrame(
            {
                "feature":
                    X.columns,

                "coefficient":
                    coefficients,

                "odds_ratio":
                    np.exp(
                        coefficients
                    ),

                "p_value":
                    np.nan,

                "ci_lower":
                    np.nan,

                "ci_upper":
                    np.nan,

                "significant_at_0_05":
                    False,
            }
        )

        return result


def model_aic(
    df: pd.DataFrame,
) -> float:

    X, y = prepare_logistic_data(
        df
    )

    try:

        model = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(),
        ).fit()

        return float(
            model.aic
        )

    except Exception:

        return float("nan")


def run_statistical_analysis(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    categorical = categorical_tests(
        df
    )

    numeric = numeric_tests(
        df
    )

    odds_ratios = logistic_inference(
        df
    )

    categorical.to_csv(
        output_dir
        / "categorical_significance.csv",
        index=False,
    )

    numeric.to_csv(
        output_dir
        / "numeric_significance.csv",
        index=False,
    )

    odds_ratios.to_csv(
        output_dir
        / "statistical_logit_odds_ratios.csv",
        index=False,
    )

    summary = {
        "sample_size":
            int(len(df)),

        "churn_rate":
            float(
                df[TARGET].mean()
            ),

        "categorical_tests":
            int(
                len(categorical)
            ),

        "numeric_tests":
            int(
                len(numeric)
            ),

        "significant_categorical":
            int(
                categorical[
                    "significant_at_0_05"
                ].sum()
            )
            if not categorical.empty
            else 0,

        "significant_numeric":
            int(
                numeric[
                    "significant_at_0_05"
                ].sum()
            )
            if not numeric.empty
            else 0,

        "logit_significant_terms":
            int(
                odds_ratios[
                    "significant_at_0_05"
                ].sum()
            )
            if not odds_ratios.empty
            else 0,

        "logit_aic":
            model_aic(df),
    }

    summary_path = (
        output_dir
        / "statistical_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return {
        "summary":
            summary,

        "categorical_tests":
            categorical.to_dict(
                orient="records"
            ),

        "numeric_tests":
            numeric.to_dict(
                orient="records"
            ),

        "logistic_inference":
            odds_ratios.to_dict(
                orient="records"
            ),
    }