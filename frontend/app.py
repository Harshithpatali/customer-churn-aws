from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")

TIMEOUT = int(
    os.getenv(
        "API_TIMEOUT",
        "30",
    )
)


st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>

/* ============================================================
   BASE
   ============================================================ */

#MainMenu,
footer,
header [data-testid="stToolbar"] {
    visibility: hidden;
}

html, body, [class*="css"] {
    font-feature-settings: "cv11", "ss01";
    -webkit-font-smoothing: antialiased;
}

.block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* ============================================================
   THEME TOKENS
   ============================================================ */

:root {
    --card-bg: #ffffff;
    --card-bg-soft: #f8fafc;
    --card-border: #e2e8f0;
    --card-border-strong: #cbd5e1;
    --card-text: #0f172a;
    --card-muted: #64748b;
    --card-shadow: 0 1px 2px rgba(15,23,42,.04),
                   0 8px 24px rgba(15,23,42,.06);
    --card-shadow-hover: 0 2px 4px rgba(15,23,42,.06),
                         0 14px 34px rgba(15,23,42,.10);
    --accent: #6366f1;
    --accent-soft: rgba(99,102,241,.10);
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: #1a1f2e;
        --card-bg-soft: #232838;
        --card-border: #2d3446;
        --card-border-strong: #3a4257;
        --card-text: #f1f5f9;
        --card-muted: #94a3b8;
        --card-shadow: 0 1px 2px rgba(0,0,0,.25),
                       0 8px 24px rgba(0,0,0,.35);
        --card-shadow-hover: 0 2px 4px rgba(0,0,0,.35),
                             0 14px 34px rgba(0,0,0,.45);
    }
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    padding: 38px 42px;
    border-radius: 26px;
    margin-bottom: 28px;
    background:
        radial-gradient(120% 140% at 0% 0%, rgba(99,102,241,.35) 0%, transparent 55%),
        radial-gradient(120% 140% at 100% 100%, rgba(56,189,248,.20) 0%, transparent 55%),
        linear-gradient(135deg, #0b1220 0%, #1e293b 55%, #334155 100%);
    box-shadow:
        0 24px 60px rgba(15,23,42,.20),
        inset 0 1px 0 rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.05);
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(120% 120% at 20% 0%, black 0%, transparent 70%);
    -webkit-mask-image: radial-gradient(120% 120% at 20% 0%, black 0%, transparent 70%);
    pointer-events: none;
}

.hero h1 {
    position: relative;
    z-index: 1;
    color: #ffffff;
    font-size: clamp(1.9rem, 3vw, 2.7rem);
    font-weight: 750;
    margin: 0;
    letter-spacing: -1.1px;
    line-height: 1.1;
}

.hero p {
    position: relative;
    z-index: 1;
    color: #cbd5e1;
    margin-top: 10px;
    font-size: 1.02rem;
    letter-spacing: .2px;
}

/* ============================================================
   TYPOGRAPHY
   ============================================================ */

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.4px;
    margin-top: 14px;
    margin-bottom: 14px;
    color: var(--card-text);
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 20px;
    border-radius: 3px;
    background: linear-gradient(180deg, #6366f1, #38bdf8);
}

h3 {
    letter-spacing: -0.3px;
    font-weight: 680;
}

/* ============================================================
   METRIC CARDS — theme-safe
   ============================================================ */

[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 18px 20px !important;
    box-shadow: var(--card-shadow) !important;
    transition: transform .18s ease, box-shadow .18s ease,
                border-color .18s ease;
    color: var(--card-text) !important;
    overflow: hidden;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover) !important;
    border-color: var(--card-border-strong) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] * {
    color: var(--card-text) !important;
    opacity: 1 !important;
    font-weight: 700 !important;
    letter-spacing: -0.6px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: var(--card-muted) !important;
    opacity: 1 !important;
    font-weight: 550 !important;
    font-size: .85rem !important;
    text-transform: uppercase;
    letter-spacing: .6px;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    opacity: 1 !important;
}

.metric-card {
    padding: 18px 20px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    color: var(--card-text);
    box-shadow: var(--card-shadow);
}

/* ============================================================
   RISK BANNERS — theme-safe
   ============================================================ */

.risk-high,
.risk-medium,
.risk-low {
    padding: 16px 20px;
    border-radius: var(--radius-md);
    color: var(--card-text);
    font-size: .98rem;
    line-height: 1.55;
    box-shadow: var(--card-shadow);
}

.risk-high strong,
.risk-medium strong,
.risk-low strong {
    color: var(--card-text);
    font-weight: 700;
}

.risk-high {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-left: 4px solid #f43f5e;
}

.risk-medium {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
}

.risk-low {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    border-right: 1px solid var(--card-border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] .stRadio > label {
    font-weight: 700;
    letter-spacing: .3px;
    text-transform: uppercase;
    font-size: .78rem;
    color: var(--card-muted);
}

[data-testid="stSidebar"] .stRadio label {
    border-radius: var(--radius-sm);
    padding: 6px 10px;
    transition: background .15s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--accent-soft);
}

/* ============================================================
   WIDGETS
   ============================================================ */

[data-testid="stFileUploader"] {
    border-radius: var(--radius-lg);
}

[data-testid="stFileUploader"] section {
    border-radius: var(--radius-lg);
    border: 1.5px dashed var(--card-border-strong);
    transition: border-color .18s ease, background .18s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
}

div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: var(--radius-sm);
    font-weight: 650;
    letter-spacing: .2px;
    transition: transform .12s ease, box-shadow .18s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(15,23,42,.12);
}

div[data-testid="stForm"] {
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 20px 22px;
    background: var(--card-bg);
    box-shadow: var(--card-shadow);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
}

/* ============================================================
   TABS, DATAFRAMES, ALERTS
   ============================================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--card-border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    padding: 10px 16px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-soft);
}

[data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--card-border);
}

[data-testid="stAlert"] {
    border-radius: var(--radius-md);
}

[data-testid="stPlotlyChart"] {
    border-radius: var(--radius-lg);
}

html {
    scroll-behavior: smooth;
}

</style>
""",
    unsafe_allow_html=True,
)


def api(
    method: str,
    path: str,
    **kwargs,
):

    response = requests.request(
        method,
        f"{BACKEND_URL}{path}",
        timeout=TIMEOUT,
        **kwargs,
    )

    response.raise_for_status()

    return response.json()


def cv_info(metrics):

    cv = metrics.get(
        "cv",
        {},
    )

    models = cv.get(
        "models",
        {},
    )

    tuned = models.get(
        "xgboost_tuned",
        {},
    )

    return tuned


def render_risk(
    probability,
    band,
):

    css = {
        "High":
            "risk-high",
        "Medium":
            "risk-medium",
        "Low":
            "risk-low",
    }.get(
        band,
        "risk-low",
    )

    st.markdown(
        f"""
        <div class="{css}">
            <strong>{band} churn risk</strong><br>
            Estimated probability:
            <strong>{probability:.1%}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:

    health = api(
        "GET",
        "/health",
    )

    api_online = True

except Exception as exc:

    api_online = False

    st.error(
        f"Backend unavailable: {exc}"
    )


st.markdown(
    """
<div class="hero">

<h1>Customer Churn Intelligence</h1>

<p>
Statistical inference • Predictive modeling •
Explainable ML • Retention analytics
</p>

</div>
""",
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    "## Intelligence Workspace"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Statistical Evidence",
        "Customer 360",
        "SHAP Explainability",
        "Batch Scoring",
        "Model Governance",
    ],
)


if api_online:

    st.sidebar.success(
        "API ONLINE"
    )

else:

    st.sidebar.error(
        "API OFFLINE"
    )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    metrics = api(
        "GET",
        "/metrics",
    )

    test = metrics["test"]

    columns = st.columns(6)

    columns[0].metric(
        "Customers",
        f"{metrics['dataset_rows']:,}",
    )

    columns[1].metric(
        "Churn rate",
        f"{metrics['churn_rate']:.1%}",
    )

    columns[2].metric(
        "ROC-AUC",
        f"{test['roc_auc']:.3f}",
    )

    columns[3].metric(
        "PR-AUC",
        f"{test['pr_auc']:.3f}",
    )

    columns[4].metric(
        "Recall",
        f"{test['recall']:.1%}",
    )

    columns[5].metric(
        "Brier score",
        f"{test.get('brier_score', 0):.3f}",
    )

    st.markdown(
        '<div class="section-title">Model benchmark</div>',
        unsafe_allow_html=True,
    )

    comparison = (
        pd.DataFrame(
            metrics["comparison"]
        )
        .T
        .reset_index(
            names="model"
        )
    )

    chart = px.bar(
        comparison,
        x="model",
        y=[
            "roc_auc",
            "pr_auc",
        ],
        barmode="group",
        template="plotly_white",
        labels={
            "value":
                "Score",
            "model":
                "Model",
        },
    )

    chart.update_layout(
        legend_title="Metric",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### Cross-validation"
        )

        tuned = cv_info(
            metrics
        )

        st.metric(
            "5-fold CV ROC-AUC",
            (
                f"{tuned.get('roc_auc_mean', 0):.3f}"
                f" ± "
                f"{tuned.get('roc_auc_std', 0):.3f}"
            ),
        )

        st.metric(
            "5-fold CV PR-AUC",
            (
                f"{tuned.get('pr_auc_mean', 0):.3f}"
                f" ± "
                f"{tuned.get('pr_auc_std', 0):.3f}"
            ),
        )

    with right:

        st.markdown(
            "### Decision threshold"
        )

        threshold = metrics[
            "threshold_selection"
        ]

        st.metric(
            "Selected threshold",
            f"{threshold['threshold']:.2f}",
        )

        st.metric(
            "Business score",
            f"{threshold['business_score']:.3f}",
        )

    st.markdown(
        "### Risk distribution"
    )

    st.info(
        "Risk bands are generated from the model probability "
        "and the selected business threshold."
    )


# ============================================================
# STATISTICAL EVIDENCE
# ============================================================

elif page == "Statistical Evidence":

    metrics = api(
        "GET",
        "/metrics",
    )

    statistical = metrics.get(
        "statistical_model",
        {},
    )

    summary = statistical.get(
        "summary",
        {},
    )

    st.markdown(
        '<div class="section-title">Statistical evidence</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Sample size",
        f"{summary.get('sample_size', metrics['dataset_rows']):,}",
    )

    b.metric(
        "Categorical tests",
        summary.get(
            "categorical_tests",
            0,
        ),
    )

    c.metric(
        "Numeric tests",
        summary.get(
            "numeric_tests",
            0,
        ),
    )

    d.metric(
        "Significant logit terms",
        summary.get(
            "logit_significant_terms",
            0,
        ),
    )

    st.caption(
        "Statistical significance is reported at α = 0.05. "
        "Effect sizes should be considered alongside p-values."
    )

    categorical = pd.DataFrame(
        statistical.get(
            "categorical_tests",
            [],
        )
    )

    numeric = pd.DataFrame(
        statistical.get(
            "numeric_tests",
            [],
        )
    )

    odds = pd.DataFrame(
        statistical.get(
            "logistic_inference",
            [],
        )
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Categorical associations",
            "Numeric differences",
            "Logistic inference",
        ]
    )

    with tab1:

        if not categorical.empty:

            st.dataframe(
                categorical,
                use_container_width=True,
                hide_index=True,
            )

            chart = px.bar(
                categorical.head(10),
                x="cramers_v",
                y="feature",
                orientation="h",
                template="plotly_white",
                title="Effect size — Cramér's V",
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

    with tab2:

        if not numeric.empty:

            st.dataframe(
                numeric,
                use_container_width=True,
                hide_index=True,
            )

            chart = px.bar(
                numeric.head(10),
                x="median_difference",
                y="feature",
                orientation="h",
                template="plotly_white",
                title="Median difference between churn groups",
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

    with tab3:

        if not odds.empty:

            display = odds.copy()

            display[
                "odds_ratio"
            ] = display[
                "odds_ratio"
            ].round(3)

            display[
                "ci_lower"
            ] = display[
                "ci_lower"
            ].round(3)

            display[
                "ci_upper"
            ] = display[
                "ci_upper"
            ].round(3)

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Odds ratios above 1 indicate higher modeled odds "
                "of churn relative to the reference category; "
                "values below 1 indicate lower modeled odds."
            )


# ============================================================
# CUSTOMER 360
# ============================================================

elif page == "Customer 360":

    st.markdown(
        '<div class="section-title">Customer 360</div>',
        unsafe_allow_html=True,
    )

    with st.form(
        "customer_form"
    ):

        c1, c2, c3 = st.columns(3)

        gender = c1.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior = c2.selectbox(
            "Senior Citizen",
            [0, 1],
        )

        tenure = c3.number_input(
            "Tenure (months)",
            0,
            100,
            12,
        )

        partner = c1.selectbox(
            "Partner",
            ["Yes", "No"],
        )

        dependents = c2.selectbox(
            "Dependents",
            ["Yes", "No"],
        )

        contract = c3.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )

        internet = c1.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
        )

        monthly = c2.number_input(
            "Monthly Charges",
            0.0,
            300.0,
            70.0,
        )

        total = c3.number_input(
            "Total Charges",
            0.0,
            20000.0,
            800.0,
        )

        phone = c1.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        multiple = c2.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service",
            ],
        )

        paperless = c3.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment = c1.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        security = c2.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service",
            ],
        )

        backup = c3.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service",
            ],
        )

        device = c1.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service",
            ],
        )

        support = c2.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service",
            ],
        )

        tv = c3.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service",
            ],
        )

        movies = c1.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service",
            ],
        )

        submit = st.form_submit_button(
            "Run churn analysis",
            type="primary",
            use_container_width=True,
        )

    if submit:

        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
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

        try:

            result = api(
                "POST",
                "/predict",
                json=payload,
            )

            a, b, c = st.columns(3)

            a.metric(
                "Churn probability",
                f"{result['churn_probability']:.1%}",
            )

            b.metric(
                "Risk band",
                result["risk_band"],
            )

            c.metric(
                "Revenue at risk",
                (
                    f"USD "
                    f"{result['estimated_monthly_revenue_at_risk']:,.2f}"
                ),
            )

            render_risk(
                result[
                    "churn_probability"
                ],
                result[
                    "risk_band"
                ],
            )

            st.markdown(
                "### Retention intelligence"
            )

            st.info(
                result[
                    "recommendation"
                ]
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# ============================================================
# SHAP
# ============================================================

elif page == "SHAP Explainability":

    st.markdown(
        '<div class="section-title">Local SHAP Explainability</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Positive SHAP values increase the model's churn score; "
        "negative values decrease it."
    )

    with st.form(
        "shap_form"
    ):

        st.info(
            "Use the same customer profile fields as Customer 360."
        )

        c1, c2, c3 = st.columns(3)

        gender = c1.selectbox(
            "Gender",
            ["Female", "Male"],
            key="shap_gender",
        )

        senior = c2.selectbox(
            "Senior Citizen",
            [0, 1],
            key="shap_senior",
        )

        tenure = c3.number_input(
            "Tenure",
            0,
            100,
            12,
            key="shap_tenure",
        )

        contract = c1.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            key="shap_contract",
        )

        internet = c2.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
            key="shap_internet",
        )

        monthly = c3.number_input(
            "Monthly Charges",
            0.0,
            300.0,
            70.0,
            key="shap_monthly",
        )

        total = c1.number_input(
            "Total Charges",
            0.0,
            20000.0,
            800.0,
            key="shap_total",
        )

        support = c2.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service",
            ],
            key="shap_support",
        )

        partner = c3.selectbox(
            "Partner",
            ["Yes", "No"],
            key="shap_partner",
        )

        analyze = st.form_submit_button(
            "Explain prediction",
            type="primary",
            use_container_width=True,
        )

    if analyze:

        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": "No",
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        try:

            result = api(
                "POST",
                "/explain",
                json=payload,
            )

            a, b = st.columns(2)

            a.metric(
                "Churn probability",
                f"{result['churn_probability']:.1%}",
            )

            b.metric(
                "Risk",
                result["risk_band"],
            )

            explanations = pd.DataFrame(
                result[
                    "explanations"
                ]
            )

            explanations = (
                explanations
                .sort_values(
                    "shap_value"
                )
            )

            chart = px.bar(
                explanations,
                x="shap_value",
                y="feature",
                orientation="h",
                template="plotly_white",
                title="Local SHAP contribution",
            )

            chart.add_vline(
                x=0,
                line_width=1,
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

            st.dataframe(
                explanations,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:

            st.error(
                f"Explainability failed: {exc}"
            )


# ============================================================
# BATCH
# ============================================================

elif page == "Batch Scoring":

    st.markdown(
        '<div class="section-title">Batch Scoring</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload a CSV containing customer feature columns."
    )

    upload = st.file_uploader(
        "Customer dataset",
        type=["csv"],
    )

    if upload:

        try:

            result = api(
                "POST",
                "/batch-predict",
                files={
                    "file": (
                        upload.name,
                        upload.getvalue(),
                        "text/csv",
                    )
                },
            )

            data = pd.DataFrame(
                result["rows"]
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Customers scored",
                f"{result['count']:,}",
            )

            c2.metric(
                "Processing time",
                f"{result['latency_ms']:.0f} ms",
            )

            c3.metric(
                "High-risk customers",
                int(
                    (
                        data["risk_band"]
                        == "High"
                    ).sum()
                ),
            )

            st.dataframe(
                data,
                use_container_width=True,
                height=500,
            )

            st.download_button(
                "Download scored dataset",
                data=data.to_csv(
                    index=False
                ),
                file_name=(
                    "churn_scored.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Batch scoring failed: {exc}"
            )


# ============================================================
# GOVERNANCE
# ============================================================

else:

    metrics = api(
        "GET",
        "/metrics",
    )

    st.markdown(
        '<div class="section-title">Model Governance</div>',
        unsafe_allow_html=True,
    )

    governance = metrics.get("model_governance", {})
    cv = metrics.get("cv", {})
    cv_models = cv.get("models", {})
    threshold_data = metrics.get("threshold_selection", {})
    best_params = metrics.get("best_params", {})
    test_data = metrics.get("test", {})

    metric_labels = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "brier_score": "Brier Score",
    }

    # ---------------------------------------------------------
    # KPI ROW
    # ---------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Model version",
        governance.get("model_version", "3.0.0"),
    )

    c2.metric(
        "Model family",
        governance.get("model_family", "XGBoost"),
    )

    c3.metric(
        "CV folds",
        cv.get("folds", 5),
    )

    c4.metric(
        "Random state",
        governance.get("random_state", 42),
    )

    st.markdown(
        "<div style='height:14px'></div>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # TEST-SET PERFORMANCE
    # ---------------------------------------------------------

    st.markdown("### Test-set performance")

    if test_data:

        ordered = [k for k in metric_labels if k in test_data]

        for i in range(0, len(ordered), 4):

            chunk = ordered[i:i + 4]
            cols = st.columns(4)

            for j, key in enumerate(chunk):

                cols[j].metric(
                    metric_labels[key],
                    f"{test_data[key]:.4f}",
                )

        plot_metrics = [k for k in ordered if k != "brier_score"]

        bar_df = pd.DataFrame({
            "Metric": [metric_labels[k] for k in plot_metrics],
            "Value": [test_data[k] for k in plot_metrics],
        })

        fig = px.bar(
            bar_df,
            x="Metric",
            y="Value",
            text="Value",
            color="Metric",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            marker_line_width=0,
        )

        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 1.1],
            yaxis_title="Score",
            xaxis_title="",
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # HYPERPARAMETERS
    # ---------------------------------------------------------

    st.markdown("### Selected hyperparameters")

    param_labels = {
        "model__subsample": "Subsample",
        "model__min_child_weight": "Min Child Weight",
        "model__max_depth": "Max Depth",
        "model__learning_rate": "Learning Rate",
        "model__colsample_bytree": "Colsample by Tree",
        "model__n_estimators": "N Estimators",
        "model__gamma": "Gamma",
        "model__reg_alpha": "Reg Alpha",
        "model__reg_lambda": "Reg Lambda",
    }

    if best_params:

        items = list(best_params.items())

        for i in range(0, len(items), 4):

            chunk = items[i:i + 4]
            cols = st.columns(4)

            for j, (key, val) in enumerate(chunk):

                label = param_labels.get(
                    key,
                    key.replace("model__", "").replace("_", " ").title(),
                )

                display = (
                    f"{val:g}"
                    if isinstance(val, (int, float))
                    else str(val)
                )

                cols[j].metric(label, display)

    else:

        st.info(
            "No hyperparameters were reported by the backend."
        )

    # ---------------------------------------------------------
    # THRESHOLD POLICY
    # ---------------------------------------------------------

    st.markdown("### Threshold policy")

    if threshold_data:

        threshold_val = threshold_data.get("threshold", 0)
        business = threshold_data.get("business_score", 0)

        t1, t2, t3, t4 = st.columns(4)

        t1.metric(
            "Selected threshold",
            f"{threshold_val:.2f}",
        )

        t2.metric(
            "Business score",
            f"{business:.4f}",
        )

        t3.metric(
            "Recall",
            f"{threshold_data.get('recall', 0):.2%}",
        )

        t4.metric(
            "Precision",
            f"{threshold_data.get('precision', 0):.2%}",
        )

        profile_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]

        profile_present = [
            k for k in profile_keys if k in threshold_data
        ]

        if profile_present:

            left, right = st.columns(2)

            profile_df = pd.DataFrame({
                "Metric": [metric_labels[k] for k in profile_present],
                "Value": [threshold_data[k] for k in profile_present],
            })

            with left:

                fig = px.bar(
                    profile_df,
                    x="Metric",
                    y="Value",
                    text="Value",
                    color="Metric",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                    marker_line_width=0,
                )

                fig.update_layout(
                    showlegend=False,
                    yaxis_range=[0, 1.1],
                    yaxis_title="Score",
                    xaxis_title="",
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                    title="Threshold policy metric profile",
                )

                st.plotly_chart(fig, use_container_width=True)

            with right:

                fig = px.pie(
                    profile_df,
                    names="Metric",
                    values="Value",
                    hole=0.55,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )

                fig.update_traces(
                    textposition="outside",
                    textinfo="percent+label",
                    marker_line_width=2,
                    marker_line_color="white",
                )

                fig.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                    title="Relative composition of policy metrics",
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # CROSS-VALIDATION
    # ---------------------------------------------------------

    st.markdown("### Cross-validation")

    if cv_models:

        rows = []

        for model_name, mdata in cv_models.items():

            rows.append({
                "Model": model_name.replace("_", " ").title(),
                "ROC-AUC": mdata.get("roc_auc_mean", 0),
                "PR-AUC": mdata.get("pr_auc_mean", 0),
                "F1": mdata.get("f1_mean", 0),
                "Accuracy": mdata.get("accuracy_mean", 0),
                "Brier": mdata.get("brier_mean", 0),
            })

        cv_df = pd.DataFrame(rows)

        st.dataframe(
            cv_df.style.format({
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "F1": "{:.4f}",
                "Accuracy": "{:.4f}",
                "Brier": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        melted = cv_df.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score",
        )

        melted_plot = melted[melted["Metric"] != "Brier"]

        fig = px.bar(
            melted_plot,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            marker_line_width=0,
        )

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_range=[0, 1.1],
            legend_title="",
            title="Model comparison across CV metrics",
        )

        st.plotly_chart(fig, use_container_width=True)

        radar_metrics = ["ROC-AUC", "PR-AUC", "F1", "Accuracy"]

        fig = go.Figure()

        for _, row in cv_df.iterrows():

            values = [row[m] for m in radar_metrics]
            values += values[:1]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + radar_metrics[:1],
                fill="toself",
                name=row["Model"],
                opacity=0.55,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                ),
            ),
            showlegend=True,
            template="plotly_white",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40),
            title="Multi-metric model profile",
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # CV CONFIGURATION
    # ---------------------------------------------------------

    st.markdown("### CV configuration")

    cfg1, cfg2, cfg3 = st.columns(3)

    cfg1.metric(
        "Folds",
        cv.get("folds", 5),
    )

    search_metric = cv.get("hyperparameter_search_metric") or "—"

    cfg2.metric(
        "Search metric",
        search_metric.replace("_", " ").title(),
    )

    scoring = cv.get("scoring", [])

    cfg3.metric(
        "Scoring metrics",
        len(scoring),
    )

    if scoring:

        st.caption(
            "Cross-validation scoring: "
            + "  ·  ".join(
                m.replace("_", " ").upper()
                for m in scoring
            )
        )