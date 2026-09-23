import pandas as pd

from services.data_pipeline.pipeline import clean, engineer


def test_clean_converts_target_and_deduplicates():
    df = pd.DataFrame({
        "customerID": ["a", "a"],
        "Churn": ["Yes", "Yes"],
        "TotalCharges": ["10", "10"],
    })
    out = clean(df)
    assert len(out) == 1
    assert out["Churn"].iloc[0] == 1
    assert out["TotalCharges"].iloc[0] == 10


def test_engineer_adds_business_features():
    df = pd.DataFrame({
        "customerID": ["a"],
        "Churn": [1],
        "tenure": [12],
        "MonthlyCharges": [100.0],
        "TotalCharges": [1200.0],
        "PhoneService": ["Yes"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
        "InternetService": ["Fiber optic"],
        "Contract": ["Month-to-month"],
    })
    out = engineer(df)
    expected = [
        "service_count", "tenure_group", "monthly_to_tenure_value",
        "avg_monthly_revenue", "contract_risk_flag", "fiber_risk_flag",
        "support_gap_flag",
    ]
    assert all(column in out.columns for column in expected)
