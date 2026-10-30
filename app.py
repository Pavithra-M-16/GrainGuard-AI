from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GrainGuard AI",
    page_icon="🌾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

SENSOR_FILE = BASE_DIR / "data" / "processed" / "sensor_readings_cleaned.csv"
MODEL_FILE = BASE_DIR / "model" / "grain_risk_model.pkl"


# --------------------------------------------------
# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --forest: #123b2b;
        --forest-2: #1d5a3d;
        --leaf: #3f8b5b;
        --sage: #e8f0e8;
        --cream: #f5f7f2;
        --ink: #1c3327;
        --muted: #718176;
        --line: #dce7dc;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(99,145,104,.10), transparent 34%),
            linear-gradient(180deg, #f4f7f1 0%, #edf3ed 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    [data-testid="stHeader"] {
        background: rgba(244,247,241,.82);
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 38px;
        border-radius: 24px;
        background:
            radial-gradient(circle at 88% 12%, rgba(157,196,135,.30), transparent 24%),
            linear-gradient(115deg, #0d3023 0%, #174a32 55%, #286b45 100%);
        color: #fff;
        margin: 4px 0 24px 0;
        box-shadow: 0 16px 38px rgba(15,58,36,.18);
        border: 1px solid rgba(255,255,255,.12);
    }
    .hero:after {
        content: "🌾";
        position: absolute;
        right: 5%;
        top: 5%;
        font-size: 5.5rem;
        opacity: .16;
        transform: rotate(-12deg);
        pointer-events: none;
    }
    .hero h1 {
        color: #fff;
        margin: 0 0 9px 0;
        font-size: 2.55rem;
        font-weight: 800;
        letter-spacing: -1.1px;
    }
    .hero p {
        color: #dcebe0;
        font-size: 1.04rem;
        line-height: 1.65;
        margin: 0;
        max-width: 760px;
    }
    h1, h2, h3 {
        color: var(--forest);
        letter-spacing: -.45px;
        font-weight: 750;
    }
    h2 { margin-top: 1.1rem; margin-bottom: .8rem; }
    h3 { margin-top: .8rem; }
    p, label, [data-testid="stMarkdownContainer"] {
        line-height: 1.55;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f9fcf8 100%);
        border: 1px solid var(--line);
        padding: 19px 21px;
        border-radius: 18px;
        box-shadow: 0 7px 20px rgba(24,61,39,.055);
        transition: transform .18s ease, box-shadow .18s ease;
        min-height: 112px;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px rgba(24,61,39,.10);
    }
    div[data-testid="stMetricLabel"] {
        color: #68796d;
        font-weight: 650;
        font-size: .88rem;
    }
    div[data-testid="stMetricValue"] {
        color: var(--forest);
        font-weight: 800;
        letter-spacing: -.7px;
    }
    div[data-testid="stSegmentedControl"] {
        background: #e5eee5;
        border: 1px solid #d4e2d5;
        border-radius: 15px;
        padding: 5px;
        margin: 5px 0 18px 0;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 11px !important;
        font-weight: 650;
        min-height: 2.55rem;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] [data-selected="true"],
    div[data-testid="stSegmentedControl"] button[data-active="true"] {
        background: var(--forest-2) !important;
        color: #fff !important;
        border-color: var(--forest-2) !important;
        box-shadow: 0 3px 8px rgba(17,65,39,.18) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p,
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p {
        color: #fff !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stTextInput"] > div {
        border-radius: 11px;
        border-color: #d6e2d7;
        background: #fff;
    }
    .stButton > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(110deg, #1b593a, #2b754c);
        color: #fff;
        border: 1px solid #1b593a;
        border-radius: 12px;
        padding: .68rem 1.15rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(24,83,50,.12);
        transition: all .18s ease;
    }
    .stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #123d2a;
        color: #fff;
        border-color: #123d2a;
        transform: translateY(-1px);
    }
    div[data-testid="stAlert"] {
        border-radius: 14px;
        border-width: 1px;
        padding-top: .75rem;
        padding-bottom: .75rem;
    }
    div[data-testid="stAlert"] p {
        font-size: .94rem;
        line-height: 1.55;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 15px;
        overflow: hidden;
        background: #fff;
    }
    div[data-testid="stPlotlyChart"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 9px;
        box-shadow: 0 6px 18px rgba(25,62,40,.045);
    }
    hr {
        border-color: #d8e4d9;
        margin-top: 1.45rem;
        margin-bottom: 1.45rem;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    [data-testid="stFileUploader"] section {
        background: #fbfdfb;
        border: 1.5px dashed #a9c5ae;
        border-radius: 16px;
    }
    @media (max-width: 768px) {
        .block-container { padding: .8rem 1rem 2rem 1rem; }
        .hero { padding: 24px 21px; border-radius: 18px; }
        .hero h1 { font-size: 1.9rem; }
        .hero:after { font-size: 3.5rem; right: 3%; }
        div[data-testid="stMetric"] { padding: 14px; }
    }
    
    /* GrainGuard dark sidebar product shell */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c2d21 0%, #123d2b 58%, #0b291e 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.15rem;
    }
    [data-testid="stSidebar"] * {
        color: #eaf3eb;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.055);
        border-radius: 10px;
        padding: .48rem .62rem;
        margin: .16rem 0;
        transition: all .16s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(137,185,135,.17);
        border-color: rgba(174,213,166,.25);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, #397c4d, #2a6540);
        border-color: rgba(204,235,193,.25);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-weight: 650;
        font-size: .91rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 5px 22px 5px;
        border-bottom: 1px solid rgba(255,255,255,.13);
        margin-bottom: 18px;
    }
    .sidebar-logo {
        width: 39px; height: 39px; display: grid; place-items: center;
        background: rgba(193,222,151,.13); border: 1px solid rgba(193,222,151,.28);
        border-radius: 12px; font-size: 1.35rem;
    }
    .sidebar-name { font-size: 1.05rem; font-weight: 800; color: #fff; }
    .sidebar-tag { font-size: .57rem; letter-spacing: 1.1px; color: #a9c9ad; margin-top: 2px; }
    .sidebar-section-label {
        color: #8eaf95; font-size: .65rem; letter-spacing: 1.6px;
        font-weight: 800; margin: 14px 0 9px 7px;
    }
    .sidebar-assistant {
        margin-top: 25px; padding: 15px 13px; border-radius: 15px;
        background: linear-gradient(145deg, rgba(87,139,86,.30), rgba(255,255,255,.055));
        border: 1px solid rgba(208,233,193,.16);
    }
    .assistant-eyebrow { font-size: .62rem; color: #b7d79f; letter-spacing: 1px; font-weight: 800; }
    .assistant-title { color: #fff; font-size: .96rem; font-weight: 750; margin: 9px 0 5px; }
    .assistant-copy { color: #c4d8c8; font-size: .76rem; line-height: 1.5; }
    .sidebar-foot { color: #779a80; font-size: .59rem; letter-spacing: 1px; margin: 24px 5px 4px; }
    .product-header {
        display: flex; align-items: center; gap: 13px;
        padding: 13px 18px; margin: 0 0 18px 0;
        background: rgba(255,255,255,.78); border: 1px solid #dce7dc;
        border-radius: 16px; box-shadow: 0 5px 16px rgba(20,61,38,.045);
    }
    .brand-mark {
        width: 43px; height: 43px; border-radius: 13px;
        display: grid; place-items: center; font-size: 1.45rem;
        background: #e6f0e4; border: 1px solid #d4e4d3;
    }
    .brand-title { color: #123b2b; font-size: 1.3rem; font-weight: 850; line-height: 1.1; }
    .brand-title span { color: #4b8a53; }
    .brand-subtitle { color: #78907c; font-size: .59rem; letter-spacing: 1.15px; margin-top: 4px; font-weight: 750; }
    .header-status {
        margin-left: auto; color: #647b68; font-size: .72rem;
        border: 1px solid #dce8dc; border-radius: 20px; padding: 7px 11px;
        background: #f8fbf7; white-space: nowrap;
    }
    .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #5a9b61; margin-right: 6px; }
    .block-container { max-width: 1600px; padding-top: 1rem; }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] [data-selected="true"] {
        background: #1d5a3d !important; color: #fff !important;
        border-color: #1d5a3d !important;
    }
    @media (max-width: 768px) {
        .header-status { display: none; }
        .brand-title { font-size: 1.08rem; }
        .brand-subtitle { font-size: .5rem; }
    }

    
    .overview-hero {
        position: relative; overflow: hidden;
        padding: 28px 31px 25px; margin: 7px 0 22px;
        border-radius: 21px;
        background: radial-gradient(circle at 87% 18%, rgba(185,214,126,.28), transparent 23%),
                    linear-gradient(112deg, #103626 0%, #1b5839 58%, #397b4a 100%);
        box-shadow: 0 13px 30px rgba(17,63,39,.16);
        border: 1px solid rgba(255,255,255,.12);
    }
    .overview-hero:after {
        content: "🌱"; position: absolute; right: 5%; top: 7%;
        font-size: 5.3rem; opacity: .18; transform: rotate(-14deg);
    }
    .overview-kicker { color:#b9d9ad; font-size:.66rem; letter-spacing:1.7px; font-weight:850; margin-bottom:10px; }
    .overview-title { color:#fff; font-size:2rem; line-height:1.18; font-weight:850; letter-spacing:-.8px; margin-bottom:8px; }
    .overview-subtitle { color:#e0eee2; font-size:.94rem; max-width:760px; line-height:1.55; position:relative; z-index:1; }
    .overview-tags { display:flex; flex-wrap:wrap; gap:8px; margin-top:17px; position:relative; z-index:1; }
    .overview-tags span { color:#eaf4e9; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.16); border-radius:30px; padding:6px 10px; font-size:.7rem; font-weight:650; }
    [data-testid="stSidebar"] input[type="radio"] { accent-color:#b7d79c !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:linear-gradient(90deg,#397c4d,#2a6540) !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color:#fff !important; }

    
    .overview-hero {
        position:relative; overflow:hidden; padding:30px 32px 27px; margin:8px 0 23px;
        border-radius:22px;
        background:radial-gradient(circle at 88% 12%,rgba(180,211,132,.25),transparent 24%),
                   linear-gradient(112deg,#0e3324 0%,#185237 58%,#39794a 100%);
        box-shadow:0 14px 32px rgba(14,57,34,.17);
        border:1px solid rgba(255,255,255,.13);
    }
    .overview-hero:after { content:"🌾"; position:absolute; right:5%; top:6%; font-size:5.5rem; opacity:.17; }
    .overview-kicker { color:#b8d7a7; font-size:.66rem; letter-spacing:1.65px; font-weight:850; margin-bottom:11px; }
    .overview-title { color:#fff; font-size:2rem; line-height:1.2; font-weight:850; letter-spacing:-.8px; margin-bottom:8px; }
    .overview-subtitle { color:#e0eee2; font-size:.94rem; max-width:760px; line-height:1.6; position:relative; z-index:1; }
    .overview-tags { display:flex; flex-wrap:wrap; gap:8px; margin-top:17px; position:relative; z-index:1; }
    .overview-tags span { color:#edf5e9; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.16); border-radius:30px; padding:6px 10px; font-size:.7rem; font-weight:650; }
    .sensor-status-card {
        background:linear-gradient(145deg,#fff,#f7fbf6); border:1px solid #dce8dc;
        border-radius:14px; padding:13px 14px; margin-bottom:10px;
        box-shadow:0 5px 14px rgba(22,61,39,.045);
    }
    .sensor-status-top { display:flex; justify-content:space-between; align-items:center; gap:8px; }
    .sensor-name { color:#1b4630; font-size:.9rem; font-weight:800; }
    .sensor-pill { border-radius:20px; padding:4px 8px; font-size:.63rem; font-weight:750; }
    .sensor-values { display:flex; gap:35px; margin-top:12px; }
    .sensor-values b { display:block; color:#173d2b; font-size:1.16rem; }
    .sensor-values small { display:block; color:#829087; font-size:.68rem; margin-top:1px; }
    .sensor-timestamp { color:#89958c; font-size:.65rem; margin-top:9px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] input { display:none !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display:flex; align-items:center; min-height:42px; padding:7px 12px;
        border-radius:11px; border:1px solid transparent; background:transparent;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:linear-gradient(90deg,#397c4d,#2a6540) !important;
        border-color:rgba(210,235,200,.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { color:#eaf3eb !important; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#fff,#f8fbf7); }

    </style>
    """,
    unsafe_allow_html=True,
)


# 3. LOAD SENSOR DATA
# --------------------------------------------------

@st.cache_data
def load_sensor_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["Timestamp"])

    required_columns = [
        "Timestamp",
        "Sensor",
        "Temperature",
        "Humidity",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Sensor data is missing columns: {missing_columns}"
        )

    data["Temperature"] = pd.to_numeric(
        data["Temperature"], errors="coerce"
    )

    data["Humidity"] = pd.to_numeric(
        data["Humidity"], errors="coerce"
    )

    data = data.dropna(subset=required_columns)
    data = data.sort_values("Timestamp")

    return data


@st.cache_data(show_spinner="Analysing historical sensor patterns...")
def detect_environmental_anomalies(dataframe):
    """Fit Isolation Forest on a bounded sample and score all valid readings."""
    from sklearn.ensemble import IsolationForest

    features = dataframe[["Temperature", "Humidity"]].astype(float)
    if len(features) < 20:
        result = dataframe.copy()
        result["Anomaly"] = "Insufficient data"
        result["Anomaly Score"] = 0.0
        return result

    fit_data = features
    if len(fit_data) > 20000:
        fit_data = fit_data.sample(n=20000, random_state=42)

    detector = IsolationForest(
        n_estimators=120,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    detector.fit(fit_data)
    scores = detector.decision_function(features)
    predictions = detector.predict(features)

    result = dataframe.copy()
    result["Anomaly"] = ["Unusual" if value == -1 else "Normal" for value in predictions]
    result["Anomaly Score"] = scores.round(5)
    return result


@st.cache_resource
def load_risk_model(file_path):
    return joblib.load(file_path)


# --------------------------------------------------
# 4. APP SHELL: SIDEBAR NAVIGATION + PRODUCT HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="product-header">
      <div class="brand-mark">🌾</div>
      <div>
        <div class="brand-title">GrainGuard <span>AI</span></div>
        <div class="brand-subtitle">INTELLIGENT GRAIN STORAGE & QUALITY MONITORING</div>
      </div>
      <div class="header-status"><span class="status-dot"></span> Prototype workspace</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo">🌾</div>
          <div><div class="sidebar-name">GrainGuard AI</div>
          <div class="sidebar-tag">SMART GRAIN INTELLIGENCE</div></div>
        </div>
        <div class="sidebar-section-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Environmental Intelligence",
            "Risk Assessment",
            "Image Inspection",
            "AI Assistant",
            "Reports",
            "About",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )
    st.markdown(
        """
        <div class="sidebar-assistant">
          <div class="assistant-eyebrow">✦ GRAINGUARD ASSISTANT</div>
          <div class="assistant-title">Need a storage insight?</div>
          <div class="assistant-copy">Ask about sensor history, unusual readings or preliminary risk results.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open AI Assistant  →", use_container_width=True):
        st.session_state["main_navigation"] = "AI Assistant"
        st.rerun()
    st.markdown(
        '<div class="sidebar-foot">HISTORICAL DATA • PROTOTYPE BUILD</div>',
        unsafe_allow_html=True,
    )

st.caption("Historical demonstration data • Not a live monitoring system")


# --------------------------------------------------
# 6. DASHBOARD
# --------------------------------------------------

if page == "Overview":

    if not SENSOR_FILE.exists():
        st.error("Sensor data file not found: data/processed/sensor_readings_cleaned.csv")
        st.stop()

    try:
        sensor_data = load_sensor_data(str(SENSOR_FILE))
    except Exception as error:
        st.error(f"Unable to load sensor data: {error}")
        st.stop()

    if sensor_data.empty:
        st.warning("No valid sensor readings are available.")
        st.stop()

    latest_by_sensor = (
        sensor_data.sort_values("Timestamp")
        .groupby("Sensor", as_index=False)
        .tail(1)
        .sort_values("Sensor")
        .copy()
    )

    # Dashboard uses the current demonstration review limits.
    temperature_limit = 30.0
    humidity_limit = 70.0

    temperature_alerts = sensor_data["Temperature"] >= temperature_limit
    humidity_alerts = sensor_data["Humidity"] >= humidity_limit

    latest_by_sensor["Needs Review"] = (
        (latest_by_sensor["Temperature"] >= temperature_limit)
        | (latest_by_sensor["Humidity"] >= humidity_limit)
    )
    sensors_needing_review = int(latest_by_sensor["Needs Review"].sum())

    st.markdown(
        """
        <div class="overview-hero">
          <div class="overview-kicker">GRAINGUARD INTELLIGENCE OVERVIEW</div>
          <div class="overview-title">Welcome back to GrainGuard AI 🌾</div>
          <div class="overview-subtitle">
            Your grain-storage monitoring workspace. Track historical environmental
            conditions, review unusual patterns and assess preliminary grain risk.
          </div>
          <div class="overview-tags">
            <span>🌾 Grain quality intelligence</span>
            <span>◉ Historical sensor monitoring</span>
            <span>✦ AI-assisted analysis</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Storage health at a glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Historical readings", f"{len(sensor_data):,}", help="Number of valid records in the historical sensor dataset.")
    k2.metric("Sensors monitored", f"{sensor_data['Sensor'].nunique():,}")
    k3.metric("Sensors needing review", f"{sensors_needing_review:,}", help="Latest reading per sensor compared with the configurable demonstration limits.")
    k4.metric("Historical threshold alerts", f"{int((temperature_alerts | humidity_alerts).sum()):,}")

    st.caption(
        "Historical demonstration dataset • Review limits are configurable examples, "
        "not official grain-storage or aflatoxin safety thresholds."
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("### Environmental trend")
        trend_data = sensor_data.copy()
        trend_data["Date"] = trend_data["Timestamp"].dt.date
        daily = (
            trend_data.groupby("Date", as_index=False)
            .agg(Temperature=("Temperature", "mean"), Humidity=("Humidity", "mean"))
            .tail(45)
        )
        import plotly.graph_objects as go
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Temperature"], mode="lines+markers",
            name="Avg temperature (°C)", line=dict(color="#397b55", width=3),
            marker=dict(size=5)
        ))
        trend_fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Humidity"], mode="lines",
            name="Avg humidity (%)", yaxis="y2",
            line=dict(color="#d4a64c", width=2.5, dash="dot")
        ))
        trend_fig.update_layout(
            template="plotly_white", height=315, margin=dict(l=12,r=12,t=22,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fbfdfb",
            legend=dict(orientation="h", y=1.12, x=0),
            xaxis_title="", yaxis_title="Temperature (°C)",
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
            font=dict(color="#53685a")
        )
        st.plotly_chart(trend_fig, use_container_width=True)

    with right:
        st.markdown("### Sensor status")
        for _, row in latest_by_sensor.iterrows():
            needs_review = bool(row["Needs Review"])
            status_text = "Review needed" if needs_review else "Within review limits"
            status_color = "#c77a32" if needs_review else "#4c9560"
            st.markdown(
                f"""
                <div class="sensor-status-card">
                  <div class="sensor-status-top">
                    <span class="sensor-name">◉ {row['Sensor']}</span>
                    <span class="sensor-pill" style="color:{status_color};background:{status_color}18">{status_text}</span>
                  </div>
                  <div class="sensor-values">
                    <span><b>{row['Temperature']:.1f}°C</b><small>Temperature</small></span>
                    <span><b>{row['Humidity']:.1f}%</b><small>Humidity</small></span>
                  </div>
                  <div class="sensor-timestamp">Latest record: {row['Timestamp']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Storage review")
    review_col, task_col = st.columns([1, 1])

    with review_col:
        if sensors_needing_review:
            st.warning(
                f"{sensors_needing_review} sensor(s) have a latest historical "
                "reading at or above a configured review limit."
            )
        else:
            st.success("All sensors' latest historical readings are below the configured review limits.")
        st.markdown(
            """
            **Suggested review actions**
            - Check sensor placement and calibration.
            - Review repeated temperature or humidity changes.
            - Inspect grain condition and storage ventilation.
            - Arrange laboratory testing if aflatoxin is suspected.
            """
        )

    with task_col:
        st.markdown(
            """
            **Quick access**
            - **Environmental Intelligence:** identify unusual historical sensor patterns.
            - **Risk Assessment:** run the saved preliminary survey-based model.
            - **Image Inspection:** upload a grain photograph for visual review.
            - **AI Assistant:** ask questions about the available historical data.
            """
        )
        st.download_button(
            "↓ Download historical sensor data",
            data=sensor_data.to_csv(index=False).encode("utf-8"),
            file_name="grainguard_sensor_readings.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("View latest sensor records"):
        st.dataframe(latest_by_sensor, use_container_width=True, hide_index=True)

    with st.expander("Configure environmental review limits"):
        st.caption(
            "The dashboard defaults are 30°C and 70% RH. These are prototype review "
            "settings, not official safety thresholds. For detailed alert filtering, "
            "use the Environmental Intelligence page."
        )
        st.write("Temperature review limit: 30°C")
        st.write("Humidity review limit: 70% RH")

    st.info(
        "This dashboard summarizes historical sensor records. It is not live monitoring, "
        "and environmental readings do not confirm fungal growth or aflatoxin contamination."
    )


# --------------------------------------------------
# 7. ENVIRONMENTAL INTELLIGENCE
# --------------------------------------------------

elif page == "Environmental Intelligence":

    st.markdown("## Environmental Intelligence")
    st.write(
        "Machine-learning analysis of historical temperature and humidity "
        "readings to flag patterns that differ from the dataset's usual behaviour."
    )

    if not SENSOR_FILE.exists():
        st.error("Sensor data file not found: data/processed/sensor_readings_cleaned.csv")
        st.stop()

    try:
        env_data = load_sensor_data(str(SENSOR_FILE))
        if env_data.empty:
            st.warning("No valid sensor readings are available.")
            st.stop()

        with st.spinner("Running anomaly detection on historical readings..."):
            scored_data = detect_environmental_anomalies(env_data)

    except Exception as error:
        st.error(f"Environmental analysis could not be completed: {error}")
        st.stop()

    unusual = scored_data[scored_data["Anomaly"] == "Unusual"].copy()
    normal_count = int((scored_data["Anomaly"] == "Normal").sum())
    unusual_count = len(unusual)
    anomaly_rate = unusual_count / max(len(scored_data), 1) * 100

    st.caption(
        "Isolation Forest is an unsupervised anomaly-detection method. "
        "An unusual reading is a review flag, not proof of spoilage, mould, "
        "aflatoxin, or unsafe grain."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Readings analysed", f"{len(scored_data):,}")
    m2.metric("Normal patterns", f"{normal_count:,}")
    m3.metric("Unusual patterns", f"{unusual_count:,}")
    m4.metric("Unusual readings", f"{anomaly_rate:.2f}%")

    if unusual_count:
        st.warning(
            f"Review suggested: the model flagged {unusual_count:,} historical readings "
            "as unusual compared with the overall temperature-humidity pattern."
        )
    else:
        st.success("No unusual readings were flagged in this analysis.")

    st.markdown("### Sensor-wise anomaly summary")
    sensor_summary = (
        scored_data.groupby("Sensor")
        .agg(
            Total_Readings=("Anomaly", "size"),
            Unusual_Readings=("Anomaly", lambda values: int((values == "Unusual").sum())),
            Mean_Temperature=("Temperature", "mean"),
            Mean_Humidity=("Humidity", "mean")
        )
        .reset_index()
    )
    sensor_summary["Unusual_%"] = (
        sensor_summary["Unusual_Readings"] / sensor_summary["Total_Readings"] * 100
    ).round(2)
    sensor_summary["Mean_Temperature"] = sensor_summary["Mean_Temperature"].round(2)
    sensor_summary["Mean_Humidity"] = sensor_summary["Mean_Humidity"].round(2)
    sensor_summary = sensor_summary.rename(columns={
        "Total_Readings": "Total Readings",
        "Unusual_Readings": "Unusual Readings",
        "Mean_Temperature": "Mean Temperature (°C)",
        "Mean_Humidity": "Mean Humidity (%)",
        "Unusual_%": "Unusual (%)"
    })
    st.dataframe(sensor_summary, use_container_width=True, hide_index=True)

    st.markdown("### Temperature and humidity pattern")
    plot_data = scored_data
    if len(plot_data) > 12000:
        plot_data = plot_data.sample(n=12000, random_state=7)

    import plotly.express as px
    scatter = px.scatter(
        plot_data,
        x="Temperature",
        y="Humidity",
        color="Anomaly",
        hover_data=["Sensor", "Timestamp", "Anomaly Score"],
        color_discrete_map={"Normal": "#3f8b5b", "Unusual": "#d97736"},
        opacity=0.65,
        title="Historical readings: normal vs unusual patterns"
    )
    scatter.update_layout(
        template="plotly_white",
        height=440,
        paper_bgcolor="white",
        plot_bgcolor="#fbfdfb",
        legend_title_text="Model flag"
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Unusual readings for review")
    if unusual.empty:
        st.info("No unusual readings to display.")
    else:
        display_unusual = unusual.sort_values("Anomaly Score").head(500)
        st.dataframe(
            display_unusual[
                ["Timestamp", "Sensor", "Temperature", "Humidity", "Anomaly Score"]
            ],
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "Download unusual-reading report (CSV)",
            data=unusual.to_csv(index=False).encode("utf-8"),
            file_name="grainguard_environmental_anomalies.csv",
            mime="text/csv"
        )

    st.info(
        "Interpretation: the model learns common combinations of temperature and humidity "
        "from this historical dataset. Review flagged readings in context and verify sensor "
        "calibration. This is not a grain-specific spoilage or aflatoxin classifier."
    )


# --------------------------------------------------
# 8. RISK ASSESSMENT
# --------------------------------------------------

elif page == "Risk Assessment":

    st.markdown("## Grain Aflatoxin Risk Assessment")

    st.info(
    "Uses historical survey and storage data only—not the sensor readings."
)

    st.warning(
    "Preliminary model output only. It does not confirm contamination or safety."
)

    if not MODEL_FILE.exists():
        st.error(
            "The risk model file was not found at "
            "model/grain_risk_model.pkl."
        )
        st.stop()

    with st.form("risk_assessment_form"):

        st.markdown("### Enter Storage and Survey Details")

        col1, col2 = st.columns(2)

        with col1:
            crop = st.selectbox(
                "Crop",
                ["Maize", "Groundnut"],
            )

            prev_crop = st.selectbox(
                "Previous Crop",
                ["Maize", "Groundnut", "Other"],
            )

            dry_method = st.selectbox(
                "Drying Method",
                ["Sun", "Other"],
            )

            stor_type = st.selectbox(
                "Storage Type",
                ["Bag", "Other"],
            )

            stor_length = st.number_input(
                "Storage Duration (months)",
                min_value=0.0,
                max_value=60.0,
                value=3.0,
                step=1.0,
            )

        with col2:
            altitude = st.number_input(
                "Altitude",
                min_value=0.0,
                max_value=10000.0,
                value=500.0,
                step=50.0,
            )

            wealth = st.selectbox(
                "Wealth Category",
                ["Low", "Medium", "High"],
            )

            awar_myco = st.selectbox(
                "Awareness of Mycotoxins",
                ["Yes", "No"],
            )

        submitted = st.form_submit_button(
            "Assess Preliminary Risk",
            use_container_width=True,
        )

    if submitted:
        try:
            risk_model = load_risk_model(str(MODEL_FILE))

            input_data = pd.DataFrame(
                [
                    {
                        "Crop": crop,
                        "PrevCrop": prev_crop,
                        "DryMeth": dry_method,
                        "StorType": stor_type,
                        "StorLeng": stor_length,
                        "Altitude": altitude,
                        "Wealth": wealth,
                        "AwarMyco": awar_myco,
                    }
                ]
            )

            prediction = risk_model.predict(input_data)[0]

            st.markdown("### Model Result")

            st.write(f"Predicted category: **{prediction}**")

            st.caption(
                "This output reflects the saved model and its training "
                "data. It should not be interpreted as a validated "
                "food-safety decision."
            )

        except Exception as error:
            st.error(
                "The risk assessment could not be completed. "
                f"Details: {error}"
            )

            st.info(
                "The model may require different category names or "
                "input formats. If this error appears, share the "
                "complete error message so we can correct the code."
            )


# --------------------------------------------------
# 9. GRAIN IMAGE INSPECTION (VISUAL REVIEW SUPPORT)
# --------------------------------------------------

elif page == "Image Inspection":

    st.markdown("## Grain Image Inspection")
    st.write(
        "Upload a grain image to create a visual inspection record. "
        "This first version checks image quality and displays the image; "
        "it does not claim to identify grain defects using a trained model."
    )

    uploaded_image = st.file_uploader(
        "Upload a grain image",
        type=["jpg", "jpeg", "png"],
        help="Choose a clear, well-lit image of a small grain sample."
    )

    if uploaded_image is not None:
        from PIL import Image, ImageStat

        image = Image.open(uploaded_image).convert("RGB")
        img_col, details_col = st.columns([1.25, 1])

        with img_col:
            st.image(image, caption="Uploaded grain sample", use_container_width=True)

        with details_col:
            width, height = image.size
            stats = ImageStat.Stat(image)
            brightness = float(np.mean(stats.mean))
            contrast = float(np.mean(stats.stddev))

            st.markdown("### Image quality summary")
            st.metric("Image dimensions", f"{width} × {height}")
            st.metric("Average brightness", f"{brightness:.1f}/255")
            st.metric("Average contrast", f"{contrast:.1f}")
            if width < 300 or height < 300:
                st.warning("The image is relatively small. A closer, higher-resolution image may be easier to inspect.")
            elif brightness < 45:
                st.warning("The image appears dark. Retake it in brighter, even lighting.")
            elif brightness > 220:
                st.warning("The image may be overexposed. Reduce glare and retake it.")
            else:
                st.success("The image is loaded. Check that individual grains are clearly visible.")

        st.info(
            "Next inspection step: visually review for broken kernels, discoloration, "
            "insect damage, mould-like growth, and foreign material. These visual signs "
            "cannot confirm aflatoxin. Use laboratory testing for confirmation."
        )

        st.download_button(
            "Download uploaded image",
            data=uploaded_image.getvalue(),
            file_name=uploaded_image.name,
            mime=uploaded_image.type
        )
    else:
        st.info("Upload a JPG or PNG image to begin the visual inspection workflow.")

    st.markdown("### What this module can and cannot do")
    st.markdown(
        "- **Can do:** accept and preview sample images, report basic image quality, and support a documented manual inspection.\n"
        "- **Cannot yet do:** automatically classify healthy/damaged grains or detect aflatoxin. A labelled image dataset and a trained, tested vision model are required for that."
    )


# --------------------------------------------------
# 10. AI ASSISTANT (DATA-GROUNDED RULE-BASED HELPER)
# --------------------------------------------------

elif page == "AI Assistant":

    st.markdown("## GrainGuard AI Assistant")
    st.write(
        "Ask about the available sensor history, configured review limits, "
        "or the meaning of your preliminary risk assessment."
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: Which sensor readings need attention?"
    )

    if question:
        q = question.lower()

        if not SENSOR_FILE.exists():
            st.warning("The historical sensor CSV is not available, so I cannot summarize sensor readings.")
        else:
            try:
                assistant_data = load_sensor_data(str(SENSOR_FILE))
                latest_rows = (
                    assistant_data.sort_values("Timestamp")
                    .groupby("Sensor", as_index=False)
                    .tail(1)
                )

                if any(word in q for word in ["sensor", "latest", "temperature", "humidity", "reading", "monitor"]):
                    st.markdown("### Sensor history summary")
                    a, b, c = st.columns(3)
                    a.metric("Sensors in dataset", int(assistant_data["Sensor"].nunique()))
                    b.metric("Historical readings", f"{len(assistant_data):,}")
                    c.metric("Latest timestamp", str(assistant_data["Timestamp"].max()))
                    st.dataframe(
                        latest_rows[["Sensor", "Timestamp", "Temperature", "Humidity"]]
                        .sort_values("Sensor"),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption(
                        "This answer is calculated from the historical CSV. "
                        "It is not live monitoring."
                    )
                elif any(word in q for word in ["aflatoxin", "contamination", "safe", "safety"]):
                    st.info(
                        "The saved model provides a preliminary category based on survey and storage inputs. "
                        "Sensor thresholds and image appearance cannot confirm aflatoxin. "
                        "A validated laboratory test is needed to establish contamination."
                    )
                elif any(word in q for word in ["alert", "risk", "recommend", "action"]):
                    st.markdown("### Practical review checklist")
                    st.markdown(
                        "1. Review the latest temperature and humidity readings for each sensor.\n"
                        "2. Check repeated or unusual changes rather than relying on one reading.\n"
                        "3. Inspect the grain and storage area for moisture, damaged kernels, pests, or visible mould.\n"
                        "4. Improve drying, ventilation, and storage hygiene where appropriate.\n"
                        "5. Arrange laboratory testing if aflatoxin contamination is suspected."
                    )
                else:
                    st.info(
                        "I can summarize sensor history, explain preliminary risk outputs, "
                        "or provide a storage review checklist. Try asking about sensors, alerts, "
                        "temperature, humidity, or aflatoxin."
                    )
            except Exception as error:
                st.error(f"Unable to prepare a data summary: {error}")

    st.caption(
        "Prototype assistant: responses use simple rules and the local historical dataset. "
        "This is not a connected large-language-model service."
    )


# --------------------------------------------------
# 11. REPORTS
# --------------------------------------------------

elif page == "Reports":

    st.markdown("## Reports & Exports")
    st.write("Download the historical records and analysis outputs available in this prototype.")

    if SENSOR_FILE.exists():
        report_data = load_sensor_data(str(SENSOR_FILE))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Environmental readings")
            st.metric("Available records", f"{len(report_data):,}")
            st.download_button(
                "Download sensor history CSV",
                data=report_data.to_csv(index=False).encode("utf-8"),
                file_name="grainguard_sensor_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            st.markdown("### Latest sensor snapshot")
            latest_report = (
                report_data.sort_values("Timestamp")
                .groupby("Sensor", as_index=False)
                .tail(1)
                .sort_values("Sensor")
            )
            st.dataframe(latest_report, use_container_width=True, hide_index=True)
            st.download_button(
                "Download latest sensor snapshot",
                data=latest_report.to_csv(index=False).encode("utf-8"),
                file_name="grainguard_latest_sensor_snapshot.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("Sensor dataset was not found. Check data/processed/sensor_readings_cleaned.csv.")

    st.info(
        "Reports contain historical demonstration data. They are not certificates of grain quality "
        "or laboratory confirmation of aflatoxin."
    )


# --------------------------------------------------
# 12. ABOUT
# --------------------------------------------------

elif page == "About":

    st.markdown("## About GrainGuard AI")

    st.write(
        """
        GrainGuard AI is a prototype for grain-storage monitoring
        and preliminary aflatoxin risk assessment.
        """
    )

    st.markdown("### Current Features")

    st.markdown(
        """
        - Historical temperature and humidity monitoring
        - Sensor-wise trend charts
        - Configurable environmental review limits
        - Dashboard summary metrics
        - Overall environmental monitoring status across sensors
        - Historical alert history with filters
        - CSV downloads for readings and alerts
        - Preliminary survey-based risk assessment\n        - Grain image upload and image-quality review (not automated defect classification)\n        - Data-grounded prototype assistant\n        - Historical sensor anomaly exploration can be added after validating the current data and environment
        """
    )

    st.markdown("### Data and Limitations")

    st.write(
        """
        The environmental monitoring demonstration uses historical
        cold-storage sensor data. It is not a grain-specific,
        experimentally validated aflatoxin prediction dataset.

        The risk assessment uses a separate historical survey model.
        The two components are not currently connected into a single
        validated predictive pipeline.
        """
    )

    st.warning(
        "Environmental alerts indicate conditions for review; they "
        "do not confirm fungal growth or aflatoxin contamination. "
        "Laboratory testing is required to establish contamination."
    )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "GrainGuard AI | Hackathon prototype | Historical monitoring "
    "and preliminary risk assessment only. Laboratory testing is "
    "required to confirm aflatoxin contamination."
)
