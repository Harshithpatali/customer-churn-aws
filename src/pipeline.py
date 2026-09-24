from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from .config import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)
from .data_quality import save_profile, validate_schema


# ============================================================
# ENGINEERED FEATURES
# ============================================================

ENGINEERED_NUMERIC = [
    "service_count",
    "monthly_to_tenure_value",
    "avg_monthly_revenue",
    "contract_risk_flag",
    "fiber_risk_flag",
    "support_gap_flag",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_raw(path: Path) -> pd.DataFrame:
    """
    Load and validate the raw Telco churn dataset.
    """

    df = pd.read_csv(path)

    validate_schema(df)

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco data.

    Operations:
    - strip column names
    - convert TotalCharges to numeric
    - remove duplicate customers
    - convert Churn from Yes/No to 1/0
    """

    out = df.copy()

    # Clean column names
    out.columns = [
        str(column).strip()
        for column in out.columns
    ]

    # Convert TotalCharges
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"],
        errors="coerce",
    )

    # Remove duplicate customers
    out = (
        out
        .drop_duplicates(subset=[ID_COL])
        .reset_index(drop=True)
    )

    # Convert target
    out[TARGET] = (
        out[TARGET]
        .map(
            {
                "Yes": 1,
                "No": 0,
            }
        )
        .astype(int)
    )

    return out


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-oriented churn features.
    """

    out = df.copy()

    # Ensure numeric
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Service count
    # --------------------------------------------------------

    yes_cols = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    out["service_count"] = (
        out[yes_cols] == "Yes"
    ).sum(axis=1)

    # --------------------------------------------------------
    # Tenure groups
    # --------------------------------------------------------

    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[
            -1,
            6,
            12,
            24,
            48,
            72,
            np.inf,
        ],
        labels=[
            "0-6",
            "7-12",
            "13-24",
            "25-48",
            "49-72",
            "73+",
        ],
    )

    # --------------------------------------------------------
    # Monthly charge relative to tenure
    # --------------------------------------------------------

    out["monthly_to_tenure_value"] = (
        out["MonthlyCharges"]
        / (out["tenure"] + 1)
    )

    # --------------------------------------------------------
    # Average monthly revenue
    # --------------------------------------------------------

    out["avg_monthly_revenue"] = (
        out["TotalCharges"]
        / out["tenure"].replace(0, 1)
    )

    # --------------------------------------------------------
    # Contract risk
    # --------------------------------------------------------

    out["contract_risk_flag"] = (
        out["Contract"]
        == "Month-to-month"
    ).astype(int)

    # --------------------------------------------------------
    # Fiber risk
    # --------------------------------------------------------

    out["fiber_risk_flag"] = (
        out["InternetService"]
        == "Fiber optic"
    ).astype(int)

    # --------------------------------------------------------
    # Support gap
    # --------------------------------------------------------

    out["support_gap_flag"] = (
        (out["InternetService"] != "No")
        & (out["TechSupport"] == "No")
    ).astype(int)

    return out


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    raw_path: Path,
    report_path: Path | None = None,
) -> pd.DataFrame:
    """
    Complete data preparation pipeline.
    """

    raw = load_raw(raw_path)

    if report_path is not None:
        save_profile(
            raw,
            report_path,
        )

    cleaned = clean(raw)

    engineered = add_features(
        cleaned
    )

    return engineered


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.

    Numeric:
        Median imputation
        StandardScaler

    Categorical:
        Most-frequent imputation
        One-hot encoding
    """

    numeric_features = (
        NUMERIC_FEATURES
        + ENGINEERED_NUMERIC
    )

    categorical_features = (
        CATEGORICAL_FEATURES
        + ["tenure_group"]
    )

    return ColumnTransformer(
        transformers=[
            # ------------------------------------------------
            # NUMERIC
            # ------------------------------------------------

            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_features,
            ),

            # ------------------------------------------------
            # CATEGORICAL
            # ------------------------------------------------

            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


# ============================================================
# MODEL PIPELINE
# ============================================================

def make_pipeline(
    model: Any,
) -> Pipeline:
    """
    Combine preprocessing + model.
    """

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true,
    probability,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate classification and probability metrics.
    """

    probability = np.asarray(
        probability
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
    }


# ============================================================
# CROSS-VALIDATION SCORERS
# ============================================================

def roc_auc_cv_scorer(
    estimator,
    X_fold,
    y_fold,
):
    """
    ROC-AUC scorer using positive-class probability.
    """

    probability = (
        estimator
        .predict_proba(X_fold)[:, 1]
    )

    return roc_auc_score(
        y_fold,
        probability,
    )


def pr_auc_cv_scorer(
    estimator,
    X_fold,
    y_fold,
):
    """
    PR-AUC / Average Precision scorer.

    Important:
    predict_proba() returns shape:

        (n_samples, 2)

    We explicitly select:

        [:, 1]

    which represents P(Churn=1).
    """

    probability = (
        estimator
        .predict_proba(X_fold)[:, 1]
    )

    return average_precision_score(
        y_fold,
        probability,
    )


def f1_cv_scorer(
    estimator,
    X_fold,
    y_fold,
):
    """
    F1 scorer using threshold = 0.5.
    """

    probability = (
        estimator
        .predict_proba(X_fold)[:, 1]
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    return f1_score(
        y_fold,
        prediction,
        zero_division=0,
    )


def accuracy_cv_scorer(
    estimator,
    X_fold,
    y_fold,
):
    """
    Accuracy scorer using threshold = 0.5.
    """

    probability = (
        estimator
        .predict_proba(X_fold)[:, 1]
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    return accuracy_score(
        y_fold,
        prediction,
    )


def brier_cv_scorer(
    estimator,
    X_fold,
    y_fold,
):
    """
    Brier score scorer.

    cross_validate assumes larger is better,
    therefore we return the negative Brier score.
    """

    probability = (
        estimator
        .predict_proba(X_fold)[:, 1]
    )

    return -brier_score_loss(
        y_fold,
        probability,
    )


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

def cross_validation_summary(
    model,
    X,
    y,
) -> dict:
    """
    Perform 5-fold stratified cross-validation.

    Metrics:
        ROC-AUC
        PR-AUC
        F1
        Accuracy
        Brier score
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": roc_auc_cv_scorer,
        "pr_auc": pr_auc_cv_scorer,
        "f1": f1_cv_scorer,
        "accuracy": accuracy_cv_scorer,
        "brier": brier_cv_scorer,
    }

    result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    return {
        "folds": 5,

        "roc_auc_mean": float(
            result["test_roc_auc"].mean()
        ),

        "roc_auc_std": float(
            result["test_roc_auc"].std()
        ),

        "pr_auc_mean": float(
            result["test_pr_auc"].mean()
        ),

        "pr_auc_std": float(
            result["test_pr_auc"].std()
        ),

        "f1_mean": float(
            result["test_f1"].mean()
        ),

        "f1_std": float(
            result["test_f1"].std()
        ),

        "accuracy_mean": float(
            result["test_accuracy"].mean()
        ),

        "accuracy_std": float(
            result["test_accuracy"].std()
        ),

        "brier_mean": float(
            -result["test_brier"].mean()
        ),

        "brier_std": float(
            result["test_brier"].std()
        ),
    }


# ============================================================
# HYPERPARAMETER SEARCH SCORER
# ============================================================

def average_precision_cv_scorer(
    estimator,
    X_fold,
    y_fold,
):
    """
    Average Precision scorer for RandomizedSearchCV.

    This MUST explicitly select [:, 1].

    Otherwise sklearn passes the full:

        (n_samples, 2)

    probability matrix into average_precision_score.
    """

    probability = (
        estimator
        .predict_proba(X_fold)[:, 1]
    )

    return average_precision_score(
        y_fold,
        probability,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(
    y,
    probability,
) -> dict:
    """
    Search thresholds between 0.10 and 0.80.

    Business score:

        65% Recall
        35% Precision

    This reflects a retention use case where
    missing a churn-risk customer is costly.
    """

    rows = []

    for threshold in np.arange(
        0.10,
        0.81,
        0.01,
    ):

        result = metrics(
            y,
            probability,
            float(threshold),
        )

        result["threshold"] = float(
            threshold
        )

        result["business_score"] = float(
            0.65 * result["recall"]
            + 0.35 * result["precision"]
        )

        rows.append(result)

    return max(
        rows,
        key=lambda x: x["business_score"],
    )


# ============================================================
# TRAINING
# ============================================================

def train(
    csv_path: Path,
    output_dir: Path,
    report_dir: Path | None = None,
):
    """
    Complete machine-learning training pipeline.

    Steps:

    1. Load data
    2. Data-quality profiling
    3. Feature engineering
    4. Train/test split
    5. Logistic Regression baseline
    6. XGBoost baseline
    7. Cross-validation
    8. XGBoost hyperparameter search
    9. Tuned model evaluation
    10. Threshold optimization
    11. Feature importance
    12. Prediction export
    13. Model artifact export
    14. Metrics export
    """

    # ========================================================
    # DIRECTORIES
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir = (
        report_dir
        or output_dir.parent / "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD RAW DATA
    # ========================================================

    raw = pd.read_csv(
        csv_path
    )

    save_profile(
        raw,
        report_dir / "data_quality.json",
    )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    df = build_dataset(
        csv_path
    )

    # ========================================================
    # FEATURES / TARGET
    # ========================================================

    X = df.drop(
        columns=[
            TARGET,
            ID_COL,
        ],
        errors="ignore",
    )

    y = df[TARGET]

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic = make_pipeline(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )

    # ========================================================
    # XGBOOST BASELINE
    # ========================================================

    xgb = make_pipeline(
        XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    )

    candidates = {
        "logistic_regression": logistic,
        "xgboost": xgb,
    }

    comparison = {}

    cv_summary = {}

    # ========================================================
    # TRAIN BASELINE MODELS
    # ========================================================

    for name, model in candidates.items():

        print(
            f"Training {name}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        probability = (
            model
            .predict_proba(X_test)[:, 1]
        )

        comparison[name] = metrics(
            y_test,
            probability,
        )

        cv_summary[name] = (
            cross_validation_summary(
                model,
                X_train,
                y_train,
            )
        )

    # ========================================================
    # XGBOOST HYPERPARAMETER SEARCH
    # ========================================================

    print(
        "Running XGBoost hyperparameter search..."
    )

    search = RandomizedSearchCV(
        estimator=xgb,

        param_distributions={
            "model__max_depth": [
                3,
                4,
                5,
                6,
            ],

            "model__learning_rate": [
                0.02,
                0.035,
                0.05,
                0.08,
            ],

            "model__subsample": [
                0.75,
                0.85,
                1.0,
            ],

            "model__colsample_bytree": [
                0.75,
                0.85,
                1.0,
            ],

            "model__min_child_weight": [
                1,
                3,
                5,
                8,
            ],
        },

        n_iter=16,

        # IMPORTANT:
        # Do NOT use:
        #
        # scoring="average_precision"
        #
        # because predict_proba() returns
        # a two-column matrix.
        scoring=average_precision_cv_scorer,

        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=True,

        # Fail loudly rather than silently
        # producing NaN search scores.
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best = search.best_estimator_

    # ========================================================
    # TEST PROBABILITIES
    # ========================================================

    test_probability = (
        best
        .predict_proba(X_test)[:, 1]
    )

    # ========================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================

    threshold_result = (
        optimize_threshold(
            y_test,
            test_probability,
        )
    )

    final_metrics = metrics(
        y_test,
        test_probability,
        threshold_result[
            "threshold"
        ],
    )

    # ========================================================
    # TUNED MODEL CV
    # ========================================================

    tuned_cv = (
        cross_validation_summary(
            best,
            X_train,
            y_train,
        )
    )

    cv_summary[
        "xgboost_tuned"
    ] = tuned_cv

    # ========================================================
    # SAVE MODEL
    # ========================================================

    artifact = {
        "pipeline": best,

        "threshold": (
            threshold_result[
                "threshold"
            ]
        ),

        "target": TARGET,

        "version": "3.0.0",

        "feature_columns": list(
            X.columns
        ),

        "model_family": "XGBoost",

        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        output_dir
        / "churn_pipeline.joblib",
    )

    # ========================================================
    # FEATURE CONTRACT
    # ========================================================

    transformed_features = (
        best
        .named_steps[
            "preprocess"
        ]
        .get_feature_names_out()
        .tolist()
    )

    (
        output_dir
        / "feature_contract.json"
    ).write_text(
        json.dumps(
            {
                "raw_features": list(
                    X.columns
                ),

                "transformed_features":
                    transformed_features,

                "engineered_features":
                    (
                        ENGINEERED_NUMERIC
                        + [
                            "tenure_group"
                        ]
                    ),
            },
            indent=2,
        )
    )

    # ========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ========================================================

    model_importance = (
        best
        .named_steps[
            "model"
        ]
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                transformed_features,

            "importance":
                model_importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        report_dir
        / "global_feature_importance.csv",
        index=False,
    )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    threshold_curve = []

    for threshold in np.arange(
        0.10,
        0.81,
        0.01,
    ):

        result = metrics(
            y_test,
            test_probability,
            float(threshold),
        )

        result["threshold"] = float(
            threshold
        )

        result["business_score"] = float(
            0.65 * result["recall"]
            + 0.35 * result["precision"]
        )

        threshold_curve.append(
            result
        )

    pd.DataFrame(
        threshold_curve
    ).to_csv(
        report_dir
        / "threshold_analysis.csv",
        index=False,
    )

    # ========================================================
    # HYPERPARAMETER SEARCH RESULTS
    # ========================================================

    pd.DataFrame(
        search.cv_results_
    ).sort_values(
        "rank_test_score"
    ).head(20).to_csv(
        report_dir
        / "hyperparameter_search.csv",
        index=False,
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_rows = []

    for name, values in (
        comparison.items()
    ):

        comparison_rows.append(
            {
                "model": name,
                **values,
            }
        )

    comparison_rows.append(
        {
            "model": "xgboost_tuned",
            **final_metrics,
        }
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        report_dir
        / "model_comparison.csv",
        index=False,
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_output = X_test.copy()

    test_output[TARGET] = (
        y_test.values
    )

    test_output[
        "churn_probability"
    ] = test_probability

    test_output[
        "risk_band"
    ] = pd.cut(
        test_probability,

        bins=[
            -0.01,
            threshold_result[
                "threshold"
            ],
            0.70,
            1.01,
        ],

        labels=[
            "Low",
            "Medium",
            "High",
        ],

        include_lowest=True,
    )

    test_output.to_csv(
        report_dir
        / "test_predictions.csv",
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {
        "dataset_rows": int(
            len(df)
        ),

        "churn_rate": float(
            y.mean()
        ),

        "feature_count": int(
            len(X.columns)
        ),

        "transformed_feature_count": int(
            len(transformed_features)
        ),

        "comparison": comparison,

        "best_params":
            search.best_params_,

        "best_cv_average_precision":
            float(
                search.best_score_
            ),

        "test": final_metrics,

        "threshold_selection":
            threshold_result,

        "cv": {
            "folds": 5,

            "scoring": [
                "roc_auc",
                "pr_auc",
                "f1",
                "accuracy",
                "brier",
            ],

            "hyperparameter_search_metric":
                "average_precision",

            "models":
                cv_summary,
        },

        "model_governance": {
            "model_family":
                "XGBoost",

            "model_version":
                "3.0.0",

            "random_state":
                RANDOM_STATE,

            "train_test_split":
                0.80,

            "test_size":
                0.20,

            "stratified":
                True,
        },
    }

    # ========================================================
    # SAVE METRICS
    # ========================================================

    (
        output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print(
        f"Transformed features: "
        f"{len(transformed_features)}"
    )

    print()
    print("Best parameters:")
    print(
        search.best_params_
    )

    print()
    print(
        "Best CV Average Precision: "
        f"{search.best_score_:.4f}"
    )

    print()
    print("Final test metrics:")

    for key, value in (
        final_metrics.items()
    ):
        print(
            f"  {key}: {value:.4f}"
        )

    print()
    print(
        "Optimal threshold: "
        f"{threshold_result['threshold']:.2f}"
    )

    print(
        "Business score: "
        f"{threshold_result['business_score']:.4f}"
    )

    print()
    print(
        "Artifacts saved to:"
    )

    print(
        f"  {output_dir}"
    )

    print(
        f"  {report_dir}"
    )

    print("=" * 60)

    return report