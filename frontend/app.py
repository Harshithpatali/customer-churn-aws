from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Customer Churn Intelligence — "
            "Statistical inference · Predictive modeling · "
            "Explainable ML · Retention analytics"
        ),
    },
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
<style>

/* ------------------------------------------------------------
   SAFE CHROME HIDING (Preserves sidebar toggle)
   ------------------------------------------------------------ */

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ------------------------------------------------------------
   SIDEBAR SAFETY NET (Forces sidebar to stay visible)
   ------------------------------------------------------------ */

section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    min-width: 300px !important;
    max-width: 400px !important;
    transform: none !important;
    left: 0 !important;
    z-index: 9999 !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
}

/* ------------------------------------------------------------
   DESIGN TOKENS
   ------------------------------------------------------------ */

:root {
    --bg-0: #f6f8fc;
    --bg-1: #ffffff;
    --bg-2: #f1f5f9;

    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --surface-inset: #f1f5f9;

    --border: #e2e8f0;
    --border-strong: #cbd5e1;

    --text: #0b1220;
    --text-soft: #334155;
    --muted: #64748b;
    --muted-2: #94a3b8;

    --brand: #6366f1;
    --brand-2: #8b5cf6;
    --brand-3: #38bdf8;
    --brand-soft: rgba(99,102,241,.10);
    --brand-softer: rgba(99,102,241,.06);

    --success: #10b981;
    --warning: #f59e0b;
    --danger:  #f43f5e;
    --info:    #0ea5e9;

    --radius-xl: 22px;
    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-sm: 9px;

    --shadow-1: 0 1px 2px rgba(15,23,42,.05),
                0 6px 20px rgba(15,23,42,.06);
    --shadow-2: 0 2px 4px rgba(15,23,42,.06),
                0 16px 40px rgba(15,23,42,.10);
    --shadow-inset:
        inset 0 1px 0 rgba(255,255,255,.7),
        inset 0 -1px 0 rgba(15,23,42,.04);

    --mono: ui-monospace, SFMono-Regular, "JetBrains Mono",
            "Fira Code", Menlo, Consolas, monospace;
    --sans: "Inter", ui-sans-serif, system-ui, -apple-system,
            "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-0: #0b0f1a;
        --bg-1: #0f1524;
        --bg-2: #131a2b;

        --surface: #141b2d;
        --surface-soft: #1a2236;
        --surface-inset: #0f1524;

        --border: #26304a;
        --border-strong: #364264;

        --text: #f1f5f9;
        --text-soft: #cbd5e1;
        --muted: #94a3b8;
        --muted-2: #64748b;

        --brand-soft: rgba(99,102,241,.18);
        --brand-softer: rgba(99,102,241,.10);

        --shadow-1: 0 1px 2px rgba(0,0,0,.35),
                    0 6px 20px rgba(0,0,0,.35);
        --shadow-2: 0 2px 4px rgba(0,0,0,.45),
                    0 16px 40px rgba(0,0,0,.55);
        --shadow-inset:
            inset 0 1px 0 rgba(255,255,255,.04),
            inset 0 -1px 0 rgba(0,0,0,.35);
    }
}

/* ------------------------------------------------------------
   BASE
   ------------------------------------------------------------ */

html, body, [class*="css"], .stApp {
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    font-feature-settings: "cv11", "ss01", "tnum";
}

.stApp {
    background:
        radial-gradient(1200px 600px at -10% -10%,
            rgba(99,102,241,.06), transparent 60%),
        radial-gradient(1000px 500px at 110% 0%,
            rgba(56,189,248,.06), transparent 55%),
        var(--bg-0);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1500px;
}

html { scroll-behavior: smooth; }

/* Custom scrollbars */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 10px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
*::-webkit-scrollbar-thumb:hover {
    background: var(--brand);
    background-clip: padding-box;
    border: 2px solid transparent;
}

/* ------------------------------------------------------------
   HERO
   ------------------------------------------------------------ */

.hero {
    position: relative;
    overflow: hidden;
    padding: 44px 48px;
    border-radius: 28px;
    margin-bottom: 26px;
    background:
        radial-gradient(120% 160% at 0% 0%,
            rgba(99,102,241,.45) 0%, transparent 55%),
        radial-gradient(120% 160% at 100% 100%,
            rgba(56,189,248,.28) 0%, transparent 55%),
        linear-gradient(135deg, #0b1220 0%, #1e293b 55%, #334155 100%);
    box-shadow:
        0 30px 70px rgba(15,23,42,.28),
        inset 0 1px 0 rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.06);
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(120% 120% at 20% 0%, black 0%, transparent 72%);
    -webkit-mask-image: radial-gradient(120% 120% at 20% 0%, black 0%, transparent 72%);
    pointer-events: none;
}

.hero-eyebrow {
    position: relative; z-index: 1;
    display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--mono);
    font-size: .72rem;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: #a5b4fc;
    background: rgba(99,102,241,.16);
    border: 1px solid rgba(165,180,252,.28);
    padding: 6px 12px;
    border-radius: 999px;
    margin-bottom: 16px;
}

.hero-eyebrow::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 12px #38bdf8;
}

.hero h1 {
    position: relative; z-index: 1;
    color: #ffffff;
    font-size: clamp(1.9rem, 3vw, 2.75rem);
    font-weight: 780;
    margin: 0;
    letter-spacing: -1.2px;
    line-height: 1.08;
}

.hero p {
    position: relative; z-index: 1;
    color: #cbd5e1;
    margin-top: 12px;
    margin-bottom: 0;
    font-size: 1.02rem;
    letter-spacing: .2px;
    max-width: 780px;
}

.hero-chips {
    position: relative; z-index: 1;
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-top: 22px;
}
.hero-chip {
    font-family: var(--mono);
    font-size: .72rem;
    letter-spacing: .08em;
    color: #e2e8f0;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12);
    padding: 5px 10px;
    border-radius: 8px;
}

/* ------------------------------------------------------------
   SECTION TITLES
   ------------------------------------------------------------ */

.section-title {
    font-size: 1.28rem;
    font-weight: 720;
    letter-spacing: -0.4px;
    margin: 18px 0 14px 0;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::before {
    content: "";
    display: inline-block;
    width: 4px; height: 20px;
    border-radius: 3px;
    background: linear-gradient(180deg, var(--brand), var(--brand-3));
    box-shadow: 0 0 12px rgba(99,102,241,.35);
}

h3 {
    letter-spacing: -0.3px;
    font-weight: 700;
    color: var(--text);
}

/* ------------------------------------------------------------
   METRIC CARDS (st.metric)
   ------------------------------------------------------------ */

[data-testid="stMetric"] {
    background:
        linear-gradient(180deg,
            color-mix(in srgb, var(--surface) 100%, transparent),
            color-mix(in srgb, var(--surface-soft) 100%, transparent));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
    box-shadow: var(--shadow-1), var(--shadow-inset);
    transition: transform .18s ease, box-shadow .18s ease,
                border-color .18s ease;
    color: var(--text);
    overflow: hidden;
    position: relative;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        var(--brand), var(--brand-3), transparent);
    opacity: .55;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-2);
    border-color: var(--border-strong);
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] * {
    color: var(--text) !important;
    opacity: 1 !important;
    font-weight: 760 !important;
    letter-spacing: -0.7px;
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    opacity: 1 !important;
    font-weight: 620 !important;
    font-size: .74rem !important;
    text-transform: uppercase;
    letter-spacing: .14em;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    opacity: 1 !important;
    font-family: var(--mono);
}

/* ------------------------------------------------------------
   RISK BANNERS
   ------------------------------------------------------------ */

.risk-high,
.risk-medium,
.risk-low {
    padding: 16px 20px;
    border-radius: var(--radius-md);
    color: var(--text);
    font-size: .98rem;
    line-height: 1.55;
    box-shadow: var(--shadow-1);
    margin-top: 12px;
}

.risk-high strong,
.risk-medium strong,
.risk-low strong {
    color: var(--text);
    font-weight: 750;
    letter-spacing: -0.2px;
}

.risk-high {
    background: linear-gradient(180deg, #fff1f2, #ffe4e6);
    border: 1px solid #fecdd3;
    border-left: 4px solid var(--danger);
}
.risk-medium {
    background: linear-gradient(180deg, #fffbeb, #fef3c7);
    border: 1px solid #fde68a;
    border-left: 4px solid var(--warning);
}
.risk-low {
    background: linear-gradient(180deg, #f0fdf4, #dcfce7);
    border: 1px solid #bbf7d0;
    border-left: 4px solid var(--success);
}

@media (prefers-color-scheme: dark) {
    .risk-high   { background: rgba(244,63,94,.12);  border-color: rgba(244,63,94,.35); }
    .risk-medium { background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.35); }
    .risk-low    { background: rgba(16,185,129,.12); border-color: rgba(16,185,129,.35); }
}

/* ------------------------------------------------------------
   SIDEBAR
   ------------------------------------------------------------ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,
            color-mix(in srgb, var(--surface) 96%, transparent),
            color-mix(in srgb, var(--surface-soft) 96%, transparent));
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

[data-testid="stSidebar"] h2 {
    font-size: 1.05rem;
    letter-spacing: -0.3px;
    font-weight: 760;
    margin-top: 4px;
    margin-bottom: 4px;
    color: var(--text);
}

[data-testid="stSidebar"] hr {
    margin: 12px 0;
    border-color: var(--border);
}

[data-testid="stSidebar"] .stRadio > label {
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
    font-size: .68rem;
    color: var(--muted);
}

[data-testid="stSidebar"] .stRadio label {
    border-radius: var(--radius-sm);
    padding: 8px 10px;
    transition: background .15s ease, color .15s ease;
    font-weight: 550;
    color: var(--text-soft);
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--brand-softer);
    color: var(--text);
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div,
[data-testid="stSidebar"] .stRadio input:checked + div {
    color: var(--brand);
    font-weight: 700;
}

/* Sidebar brand block */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface-soft);
    margin-bottom: 6px;
}
.sb-brand-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--brand), var(--brand-3));
    box-shadow: 0 0 12px rgba(99,102,241,.55);
}
.sb-brand-text {
    font-family: var(--mono);
    font-size: .74rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ------------------------------------------------------------
   WIDGETS
   ------------------------------------------------------------ */

[data-testid="stFileUploader"] { border-radius: var(--radius-lg); }

[data-testid="stFileUploader"] section {
    border-radius: var(--radius-lg);
    border: 1.5px dashed var(--border-strong);
    background: var(--surface-soft);
    transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--brand);
    background: var(--brand-softer);
}

div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: var(--radius-sm);
    font-weight: 680;
    letter-spacing: .02em;
    border: 1px solid var(--border);
    transition: transform .12s ease, box-shadow .18s ease,
                border-color .18s ease, background .18s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px);
    border-color: var(--brand);
    box-shadow: 0 10px 24px rgba(99,102,241,.20);
}

div[data-testid="stForm"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 22px 24px;
    background: var(--surface);
    box-shadow: var(--shadow-1), var(--shadow-inset);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    background: var(--surface-soft) !important;
    border-color: var(--border) !important;
}

/* ------------------------------------------------------------
   TABS / DATAFRAMES / ALERTS / CHARTS
   ------------------------------------------------------------ */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    padding: 10px 16px;
    font-weight: 650;
    color: var(--muted);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--brand-soft);
    color: var(--brand);
}

[data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-1);
}

[data-testid="stAlert"] {
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
}

[data-testid="stPlotlyChart"] {
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    padding: 6px;
    background: var(--surface);
    box-shadow: var(--shadow-1);
}

/* Captions */
[data-testid="stCaptionContainer"],
.stCaption, small {
    color: var(--muted) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
}

/* Metric row spacing under headers */
.metric-row {
    margin-top: 6px;
    margin-bottom: 4px;
}

/* Small helper card */
.mini-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    box-shadow: var(--shadow-1);
    color: var(--text);
    font-size: .92rem;
    line-height: 1.5;
}
.mini-card .k {
    font-family: var(--mono);
    color: var(--muted);
    font-size: .72rem;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.mini-card .v {
    font-family: var(--mono);
    color: var(--text);
    font-weight: 700;
    font-size: 1rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS (unchanged contract)
# ============================================================

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
    cv = metrics.get("cv", {})
    models = cv.get("models", {})
    tuned = models.get("xgboost_tuned", {})
    return tuned


def render_risk(probability, band):
    css = {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(band, "risk-low")

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


def format_dataframe(df, format_dict):
    """Apply formatting to a DataFrame safely."""
    if df.empty:
        return df
    active_fmt = {k: v for k, v in format_dict.items() if k in df.columns}
    return df.style.format(active_fmt, na_rep="")


# ============================================================
# HEALTH CHECK
# ============================================================

try:
    health = api("GET", "/health")
    api_online = True
except Exception as exc:
    api_online = False
    st.error(f"Backend unavailable: {exc}")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-eyebrow">ML Console · v3.0</div>
    <h1>Customer Churn Intelligence</h1>
    <p>
        Statistical inference · Predictive modeling ·
        Explainable ML · Retention analytics
    </p>
    <div class="hero-chips">
        <span class="hero-chip">XGBoost</span>
        <span class="hero-chip">SHAP</span>
        <span class="hero-chip">Cross-Validation</span>
        <span class="hero-chip">Threshold Tuning</span>
        <span class="hero-chip">Batch Scoring</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR NAV
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-dot"></div>
            <div class="sb-brand-text">Churn Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Intelligence Workspace")

    page = st.radio(
        "Navigate",
        [
            "Executive Overview",
            "Statistical Evidence",
            "Customer 360",
            "SHAP Explainability",
            "Batch Scoring",
            "Model Governance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if api_online:
        st.success("API ONLINE")
    else:
        st.error("API OFFLINE")

    # Removed the backend URL caption as requested


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    metrics = api("GET", "/metrics")

    test = metrics["test"]

    st.markdown(
        '<div class="section-title">Key performance indicators</div>',
        unsafe_allow_html=True,
    )

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
        pd.DataFrame(metrics["comparison"])
        .T
        .reset_index(names="model")
    )

    chart = px.bar(
        comparison,
        x="model",
        y=["roc_auc", "pr_auc"],
        barmode="group",
        template="plotly_white",
        labels={"value": "Score", "model": "Model"},
        color_discrete_sequence=["#6366f1", "#38bdf8"],
    )

    chart.update_layout(
        legend_title="Metric",
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(chart, use_container_width=True)

    left, right = st.columns(2)

    with left:

        st.markdown("### Cross-validation")

        tuned = cv_info(metrics)

        st.metric(
            "5-fold CV ROC-AUC",
            f"{tuned.get('roc_auc_mean', 0):.3f}"
            f" ± "
            f"{tuned.get('roc_auc_std', 0):.3f}",
        )
        st.metric(
            "5-fold CV PR-AUC",
            f"{tuned.get('pr_auc_mean', 0):.3f}"
            f" ± "
            f"{tuned.get('pr_auc_std', 0):.3f}",
        )

    with right:

        st.markdown("### Decision threshold")

        threshold = metrics["threshold_selection"]

        st.metric(
            "Selected threshold",
            f"{threshold['threshold']:.2f}",
        )
        st.metric(
            "Business score",
            f"{threshold['business_score']:.3f}",
        )

    st.markdown("### Risk distribution")

    st.info(
        "Risk bands are generated from the model probability "
        "and the selected business threshold."
    )


# ============================================================
# STATISTICAL EVIDENCE
# ============================================================

elif page == "Statistical Evidence":

    metrics = api("GET", "/metrics")

    statistical = metrics.get("statistical_model", {})
    summary = statistical.get("summary", {})

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
        summary.get("categorical_tests", 0),
    )
    c.metric(
        "Numeric tests",
        summary.get("numeric_tests", 0),
    )
    d.metric(
        "Significant logit terms",
        summary.get("logit_significant_terms", 0),
    )

    st.caption(
        "Statistical significance is reported at α = 0.05. "
        "Effect sizes should be considered alongside p-values."
    )

    categorical = pd.DataFrame(
        statistical.get("categorical_tests", [])
    )
    numeric = pd.DataFrame(
        statistical.get("numeric_tests", [])
    )
    odds = pd.DataFrame(
        statistical.get("logistic_inference", [])
    )

    # Formatting dictionary for statistical tables
    format_dict = {
        "p_value": "{:.2e}",
        "statistic": "{:.3f}",
        "cramers_v": "{:.3f}",
        "median_difference": "{:.3f}",
        "odds_ratio": "{:.3f}",
        "ci_lower": "{:.3f}",
        "ci_upper": "{:.3f}",
        "coef": "{:.3f}",
        "std_err": "{:.3f}",
    }

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
                format_dataframe(categorical, format_dict),
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
                color="cramers_v",
                color_continuous_scale="Viridis",
            )
            chart.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )

            st.plotly_chart(chart, use_container_width=True)

    with tab2:

        if not numeric.empty:

            st.dataframe(
                format_dataframe(numeric, format_dict),
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
                color="median_difference",
                color_continuous_scale="RdBu_r",
            )
            chart.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )

            st.plotly_chart(chart, use_container_width=True)

    with tab3:

        if not odds.empty:

            st.dataframe(
                format_dataframe(odds, format_dict),
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

    with st.form("customer_form"):

        c1, c2, c3 = st.columns(3)

        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior Citizen", [0, 1])
        tenure = c3.number_input("Tenure (months)", 0, 100, 12)

        partner = c1.selectbox("Partner", ["Yes", "No"])
        dependents = c2.selectbox("Dependents", ["Yes", "No"])
        contract = c3.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        internet = c1.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        monthly = c2.number_input(
            "Monthly Charges", 0.0, 300.0, 70.0
        )
        total = c3.number_input(
            "Total Charges", 0.0, 20000.0, 800.0
        )

        phone = c1.selectbox("Phone Service", ["Yes", "No"])
        multiple = c2.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )
        paperless = c3.selectbox(
            "Paperless Billing", ["Yes", "No"]
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
            ["Yes", "No", "No internet service"],
        )
        backup = c3.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )

        device = c1.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )
        support = c2.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )
        tv = c3.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )

        movies = c1.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
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

            result = api("POST", "/predict", json=payload)

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
                f"USD {result['estimated_monthly_revenue_at_risk']:,.2f}",
            )

            render_risk(
                result["churn_probability"],
                result["risk_band"],
            )

            st.markdown("### Retention intelligence")
            st.info(result["recommendation"])

        except Exception as exc:

            st.error(f"Prediction failed: {exc}")


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

    with st.form("shap_form"):

        st.info(
            "Use the same customer profile fields as Customer 360."
        )

        c1, c2, c3 = st.columns(3)

        gender = c1.selectbox(
            "Gender", ["Female", "Male"], key="shap_gender"
        )
        senior = c2.selectbox(
            "Senior Citizen", [0, 1], key="shap_senior"
        )
        tenure = c3.number_input(
            "Tenure", 0, 100, 12, key="shap_tenure"
        )

        contract = c1.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            key="shap_contract",
        )
        internet = c2.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key="shap_internet",
        )
        monthly = c3.number_input(
            "Monthly Charges", 0.0, 300.0, 70.0,
            key="shap_monthly",
        )

        total = c1.number_input(
            "Total Charges", 0.0, 20000.0, 800.0,
            key="shap_total",
        )
        support = c2.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
            key="shap_support",
        )
        partner = c3.selectbox(
            "Partner", ["Yes", "No"], key="shap_partner"
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

            result = api("POST", "/explain", json=payload)

            a, b = st.columns(2)

            a.metric(
                "Churn probability",
                f"{result['churn_probability']:.1%}",
            )
            b.metric("Risk", result["risk_band"])

            explanations = pd.DataFrame(result["explanations"])
            explanations = explanations.sort_values("shap_value")

            chart = px.bar(
                explanations,
                x="shap_value",
                y="feature",
                orientation="h",
                template="plotly_white",
                title="Local SHAP contribution",
                color="shap_value",
                color_continuous_scale="RdBu_r",
            )

            chart.add_vline(x=0, line_width=1)

            chart.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )

            st.plotly_chart(chart, use_container_width=True)

            st.dataframe(
                format_dataframe(explanations, {"shap_value": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:

            st.error(f"Explainability failed: {exc}")


# ============================================================
# BATCH
# ============================================================

elif page == "Batch Scoring":

    st.markdown(
        '<div class="section-title">Batch Scoring</div>',
        unsafe_allow_html=True,
    )

    st.info("Upload a CSV containing customer feature columns.")

    upload = st.file_uploader("Customer dataset", type=["csv"])

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

            data = pd.DataFrame(result["rows"])

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
                int((data["risk_band"] == "High").sum()),
            )

            st.dataframe(
                data,
                use_container_width=True,
                height=500,
            )

            st.download_button(
                "Download scored dataset",
                data=data.to_csv(index=False),
                file_name="churn_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(f"Batch scoring failed: {exc}")


# ============================================================
# GOVERNANCE
# ============================================================

else:

    metrics = api("GET", "/metrics")

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
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
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

        st.info("No hyperparameters were reported by the backend.")

    # ---------------------------------------------------------
    # THRESHOLD POLICY
    # ---------------------------------------------------------

    st.markdown("### Threshold policy")

    if threshold_data:

        threshold_val = threshold_data.get("threshold", 0)
        business = threshold_data.get("business_score", 0)

        t1, t2, t3, t4 = st.columns(4)

        t1.metric("Selected threshold", f"{threshold_val:.2f}")
        t2.metric("Business score", f"{business:.4f}")
        t3.metric(
            "Recall",
            f"{threshold_data.get('recall', 0):.2%}",
        )
        t4.metric(
            "Precision",
            f"{threshold_data.get('precision', 0):.2%}",
        )

        profile_keys = ["accuracy", "precision", "recall", "f1"]
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
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
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
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
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            showlegend=True,
            template="plotly_white",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40),
            title="Multi-metric model profile",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # CV CONFIGURATION
    # ---------------------------------------------------------

    st.markdown("### CV configuration")

    cfg1, cfg2, cfg3 = st.columns(3)

    cfg1.metric("Folds", cv.get("folds", 5))

    search_metric = cv.get("hyperparameter_search_metric") or "—"

    cfg2.metric(
        "Search metric",
        search_metric.replace("_", " ").title(),
    )

    scoring = cv.get("scoring", [])

    cfg3.metric("Scoring metrics", len(scoring))

    if scoring:

        st.caption(
            "Cross-validation scoring: "
            + "  ·  ".join(
                m.replace("_", " ").upper()
                for m in scoring
            )
        )
