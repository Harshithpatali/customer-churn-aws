from __future__ import annotations

import os
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

st.set_page_config(page_title="Churn Intelligence", page_icon="📊", layout="wide")
st.title("Customer Churn Intelligence")
st.caption("IBM Telco Customer Churn • XGBoost • SHAP • FastAPI • AWS-ready")

try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT).json()
    st.success(f"API: {health['status']} | model: {'ready' if health.get('model_loaded') else 'missing'}")
except Exception as exc:
    st.error(f"Backend unavailable: {exc}")

page = st.sidebar.radio("Workspace", ["Executive Overview", "Customer Prediction", "Explainability", "Batch Prediction"])

if page == "Executive Overview":
    try:
        m = requests.get(f"{BACKEND_URL}/metrics", timeout=TIMEOUT).json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{m['dataset_rows']:,}")
        c2.metric("Churn rate", f"{m['churn_rate']:.1%}")
        c3.metric("ROC-AUC", f"{m['test']['roc_auc']:.3f}")
        c4.metric("Recall", f"{m['test']['recall']:.1%}")
        comp = pd.DataFrame(m["comparison"]).T.reset_index(names="model")
        st.subheader("Model comparison")
        st.plotly_chart(px.bar(comp, x="model", y=["roc_auc", "pr_auc"], barmode="group"), use_container_width=True)
        st.json(m["test"])
    except Exception as exc:
        st.warning(str(exc))

elif page == "Customer Prediction":
    with st.form("prediction"):
        cols = st.columns(3)
        gender = cols[0].selectbox("Gender", ["Female", "Male"])
        senior = cols[1].selectbox("Senior Citizen", [0, 1])
        tenure = cols[2].number_input("Tenure (months)", 0, 100, 12)
        partner = cols[0].selectbox("Partner", ["Yes", "No"])
        dependents = cols[1].selectbox("Dependents", ["Yes", "No"])
        contract = cols[2].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = cols[0].selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        monthly = cols[1].number_input("Monthly Charges", 0.0, 300.0, 70.0)
        total = cols[2].number_input("Total Charges", 0.0, 20000.0, 800.0)
        phone = cols[0].selectbox("Phone Service", ["Yes", "No"])
        multi = cols[1].selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        paperless = cols[2].selectbox("Paperless Billing", ["Yes", "No"])
        payment = cols[0].selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        security = cols[1].selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = cols[2].selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = cols[0].selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = cols[1].selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = cols[2].selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = cols[0].selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        submit = st.form_submit_button("Predict churn")
    if submit:
        payload = locals().copy()
        payload = {k: payload[k] for k in ["gender","senior","tenure","partner","dependents","contract","internet","phone","multi","paperless","payment","security","backup","device","support","tv","movies","monthly","total"]}
        payload = {"SeniorCitizen": payload.pop("senior"), "tenure": payload.pop("tenure"), "MonthlyCharges": payload.pop("monthly"), "TotalCharges": payload.pop("total"), "gender": payload.pop("gender"), "Partner": payload.pop("partner"), "Dependents": payload.pop("dependents"), "PhoneService": payload.pop("phone"), "MultipleLines": payload.pop("multi"), "InternetService": payload.pop("internet"), "OnlineSecurity": payload.pop("security"), "OnlineBackup": payload.pop("backup"), "DeviceProtection": payload.pop("device"), "TechSupport": payload.pop("support"), "StreamingTV": payload.pop("tv"), "StreamingMovies": payload.pop("movies"), "Contract": payload.pop("contract"), "PaperlessBilling": payload.pop("paperless"), "PaymentMethod": payload.pop("payment")}
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        result = r.json()
        st.metric("Churn probability", f"{result['churn_probability']:.1%}")
        st.metric("Risk band", result["risk_band"])
        st.info(result["recommendation"])

elif page == "Explainability":
    st.info("Use the same customer form from Prediction to request local SHAP explanations. For API clients, POST the customer JSON to /explain.")
    st.code("POST /explain\n{ ... customer fields ... }")

else:
    upload = st.file_uploader("Upload a CSV with model input columns", type="csv")
    if upload:
        r = requests.post(f"{BACKEND_URL}/batch-predict", files={"file": (upload.name, upload.getvalue(), "text/csv")}, timeout=TIMEOUT)
        r.raise_for_status()
        out = pd.DataFrame(r.json()["rows"])
        st.dataframe(out, use_container_width=True)
        st.download_button("Download predictions", out.to_csv(index=False), "churn_predictions.csv", "text/csv")
