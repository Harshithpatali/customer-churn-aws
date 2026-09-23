from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .hero {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg,#111827,#334155);
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 { font-size: 2.2rem; margin-bottom: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api(method, path, **kwargs):
    response = requests.request(
        method, f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kwargs
    )
    response.raise_for_status()
    return response.json()


try:
    health = api("GET", "/health")
    api_ok = True
except Exception as exc:
    health = {}
    api_ok = False
    st.error(f"Backend unavailable: {exc}")

st.markdown(
    '<div class="hero"><h1>Customer Churn Intelligence</h1>'
    "<div>Decision platform • XGBoost • SHAP • Statistical inference • FastAPI • AWS ECS</div></div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Workspace",
    [
        "Executive Overview",
        "Customer 360 Prediction",
        "SHAP Explainability",
        "Batch Scoring",
        "Model Governance",
    ],
)
st.sidebar.caption(f"API: {'ONLINE' if api_ok else 'OFFLINE'}")

if page == "Executive Overview":
    m = api("GET", "/metrics")
    t = m["test"]
    a, b, c, d, e = st.columns(5)
    a.metric("Customers", f"{m['dataset_rows']:,}")
    b.metric("Churn rate", f"{m['churn_rate']:.1%}")
    c.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    d.metric("PR-AUC", f"{t['pr_auc']:.3f}")
    e.metric("Recall", f"{t['recall']:.1%}")
    comp = pd.DataFrame(m["comparison"]).T.reset_index(names="model")
    st.subheader("Model benchmark")
    st.plotly_chart(
        px.bar(comp, x="model", y=["roc_auc", "pr_auc"], barmode="group"),
        use_container_width=True,
    )
    left, right = st.columns(2)
    left.subheader("Threshold strategy")
    left.write({
        "selected_threshold": m["threshold_selection"]["threshold"],
        "business_score": m["threshold_selection"]["business_score"],
    })
    right.subheader("Model governance")
    right.write({"CV": m.get("cv", {}), "model": "XGBoost"})

elif page in ["Customer 360 Prediction", "SHAP Explainability"]:
    st.subheader("Customer profile")
    with st.form("customer"):
        c = st.columns(3)
        gender = c[0].selectbox("Gender", ["Female", "Male"])
        senior = c[1].selectbox("Senior Citizen", [0, 1])
        tenure = c[2].number_input("Tenure (months)", 0, 100, 12)
        partner = c[0].selectbox("Partner", ["Yes", "No"])
        dependents = c[1].selectbox("Dependents", ["Yes", "No"])
        contract = c[2].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = c[0].selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        monthly = c[1].number_input("Monthly Charges", 0.0, 300.0, 70.0)
        total = c[2].number_input("Total Charges", 0.0, 20000.0, 800.0)
        phone = c[0].selectbox("Phone Service", ["Yes", "No"])
        multi = c[1].selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        paperless = c[2].selectbox("Paperless Billing", ["Yes", "No"])
        payment = c[0].selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        security = c[1].selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = c[2].selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = c[0].selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = c[1].selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = c[2].selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = c[0].selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        submit = st.form_submit_button("Analyze customer", type="primary")

    if submit:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        result = api(
            "POST",
            "/explain" if page == "SHAP Explainability" else "/predict",
            json=payload,
        )
        a, b, c = st.columns(3)
        a.metric("Churn probability", f"{result['churn_probability']:.1%}")
        b.metric("Risk band", result["risk_band"])
        c.metric(
            "Monthly revenue at risk",
            f"{result['estimated_monthly_revenue_at_risk']:,.2f}",
        )
        st.info(result["recommendation"])
        if "explanations" in result:
            ex = pd.DataFrame(result["explanations"])
            st.subheader("Local SHAP drivers")
            st.plotly_chart(
                px.bar(
                    ex.sort_values("shap_value"),
                    x="shap_value",
                    y="feature",
                    orientation="h",
                ),
                use_container_width=True,
            )

elif page == "Batch Scoring":
    st.subheader("Batch scoring")
    upload = st.file_uploader("Upload customer CSV", type="csv")
    if upload:
        out = api(
            "POST",
            "/batch-predict",
            files={"file": (upload.name, upload.getvalue(), "text/csv")},
        )
        data = pd.DataFrame(out["rows"])
        st.metric("Customers scored", out["count"])
        st.dataframe(data, use_container_width=True)
        st.download_button(
            "Download scored CSV",
            data.to_csv(index=False),
            "churn_scored.csv",
            "text/csv",
        )

else:
    m = api("GET", "/metrics")
    st.subheader("Model governance")
    st.json({
        "version": "2.0.0",
        "family": "XGBoost",
        "CV": m.get("cv", {}),
        "best_params": m.get("best_params", {}),
        "threshold": m["threshold_selection"]["threshold"],
        "test_metrics": m["test"],
    })
    st.caption("Metrics are generated from a fixed stratified holdout; rerun the pipeline to reproduce the artifact.")
