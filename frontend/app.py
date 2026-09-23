from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

BACKEND_URL=os.getenv("BACKEND_URL","http://localhost:8000").rstrip("/")
TIMEOUT=int(os.getenv("API_TIMEOUT","30"))

st.set_page_config(page_title="Churn Intelligence",page_icon="📊",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1450px}
.hero{padding:30px;border-radius:22px;background:linear-gradient(135deg,#111827,#334155);color:white;margin-bottom:24px}
.hero h1{font-size:2.4rem;margin:0 0 6px}
.hero p{margin:0;color:#cbd5e1}
[data-testid="stMetric"]{border:1px solid #e2e8f0;border-radius:16px;padding:12px;background:#ffffff}
</style>
""",unsafe_allow_html=True)

def api(method,path,**kwargs):
    response=requests.request(method,f"{BACKEND_URL}{path}",timeout=TIMEOUT,**kwargs)
    response.raise_for_status()
    return response.json()

try:
    api("GET","/health")
    api_online=True
except Exception as exc:
    api_online=False
    st.error(f"Backend unavailable: {exc}")

st.markdown('<div class="hero"><h1>Customer Churn Intelligence</h1><p>Explainable ML • Statistical analysis • Retention intelligence • AWS ECS</p></div>',unsafe_allow_html=True)

page=st.sidebar.radio("Workspace",["Executive Overview","Customer 360","SHAP Explainability","Batch Scoring","Model Governance"])
st.sidebar.caption("API: "+("ONLINE" if api_online else "OFFLINE"))

if page=="Executive Overview":
    m=api("GET","/metrics"); t=m["test"]; cols=st.columns(5)
    cols[0].metric("Customers",f"{m['dataset_rows']:,}")
    cols[1].metric("Churn rate",f"{m['churn_rate']:.1%}")
    cols[2].metric("ROC-AUC",f"{t['roc_auc']:.3f}")
    cols[3].metric("PR-AUC",f"{t['pr_auc']:.3f}")
    cols[4].metric("Recall",f"{t['recall']:.1%}")
    st.subheader("Model benchmark")
    comparison=pd.DataFrame(m["comparison"]).T.reset_index(names="model")
    st.plotly_chart(px.bar(comparison,x="model",y=["roc_auc","pr_auc"],barmode="group",template="plotly_white"),use_container_width=True)
    left,right=st.columns(2)
    left.subheader("Business threshold"); left.json({"threshold":m["threshold_selection"]["threshold"],"business_score":m["threshold_selection"]["business_score"]})
    right.subheader("Model governance"); right.json({"model":"XGBoost","CV folds":m["cv"]["folds"],"scoring":m["cv"]["scoring"]})

elif page in ["Customer 360","SHAP Explainability"]:
    st.subheader("Customer profile")
    with st.form("customer"):
        c=st.columns(3)
        gender=c[0].selectbox("Gender",["Female","Male"]); senior=c[1].selectbox("Senior Citizen",[0,1]); tenure=c[2].number_input("Tenure (months)",0,100,12)
        partner=c[0].selectbox("Partner",["Yes","No"]); dependents=c[1].selectbox("Dependents",["Yes","No"]); contract=c[2].selectbox("Contract",["Month-to-month","One year","Two year"])
        internet=c[0].selectbox("Internet Service",["DSL","Fiber optic","No"]); monthly=c[1].number_input("Monthly Charges",0.0,300.0,70.0); total=c[2].number_input("Total Charges",0.0,20000.0,800.0)
        phone=c[0].selectbox("Phone Service",["Yes","No"]); multi=c[1].selectbox("Multiple Lines",["Yes","No","No phone service"]); paperless=c[2].selectbox("Paperless Billing",["Yes","No"])
        payment=c[0].selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]); security=c[1].selectbox("Online Security",["Yes","No","No internet service"]); backup=c[2].selectbox("Online Backup",["Yes","No","No internet service"])
        device=c[0].selectbox("Device Protection",["Yes","No","No internet service"]); support=c[1].selectbox("Tech Support",["Yes","No","No internet service"]); tv=c[2].selectbox("Streaming TV",["Yes","No","No internet service"]); movies=c[0].selectbox("Streaming Movies",["Yes","No","No internet service"])
        submit=st.form_submit_button("Analyze customer",type="primary")
    if submit:
        payload={"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,"PhoneService":phone,"MultipleLines":multi,"InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}
        endpoint="/explain" if page=="SHAP Explainability" else "/predict"
        result=api("POST",endpoint,json=payload)
        a,b,c=st.columns(3); a.metric("Churn probability",f"{result['churn_probability']:.1%}"); b.metric("Risk band",result["risk_band"]); c.metric("Revenue at risk",f"USD {result['estimated_monthly_revenue_at_risk']:,.2f}")
        st.info(result["recommendation"])
        if "explanations" in result:
            ex=pd.DataFrame(result["explanations"]).sort_values("shap_value")
            st.subheader("Local SHAP drivers")
            st.plotly_chart(px.bar(ex,x="shap_value",y="feature",orientation="h",template="plotly_white"),use_container_width=True)

elif page=="Batch Scoring":
    st.subheader("Batch scoring")
    upload=st.file_uploader("Upload customer CSV",type="csv")
    if upload:
        result=api("POST","/batch-predict",files={"file":(upload.name,upload.getvalue(),"text/csv")})
        data=pd.DataFrame(result["rows"]); st.metric("Customers scored",result["count"]); st.dataframe(data,use_container_width=True)
        st.download_button("Download scored CSV",data=data.to_csv(index=False),file_name="churn_scored.csv",mime="text/csv")

else:
    m=api("GET","/metrics")
    st.subheader("Model governance")
    st.json({"model_version":"2.0.0","model_family":"XGBoost","cross_validation":m["cv"],"best_parameters":m["best_params"],"selected_threshold":m["threshold_selection"]["threshold"],"test_metrics":m["test"]})
