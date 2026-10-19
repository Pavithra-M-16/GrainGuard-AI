from pathlib import Path

import joblib
import pandas as pd
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
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f8f4;
        color: #1e3025;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 0.8rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 22px 28px;
        border-radius: 16px;
        background: linear-gradient(115deg, #153d2a 0%, #2e7048 100%);
        color: #ffffff;
        margin-bottom: 10px;
        box-shadow: 0 6px 18px rgba(26, 70, 44, 0.08);
    }

    .hero h1 {
        color: #ffffff;
        margin: 0 0 6px 0;
        font-size: 2.15rem;
    }

    .hero p {
        color: #e5f0e7;
        font-size: 1rem;
        margin: 0;
    }

    .section-heading {
        color: #174c32;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0e9e1;
        padding: 15px 16px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(20, 61, 42, 0.04);
    }

    div[data-testid="stSegmentedControl"] {
        margin: 0 0 8px 0;
    }

    div[data-testid="stSegmentedControl"] button {
        border-radius: 10px !important;
        font-weight: 600;
    }

    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] [data-selected="true"] {
    background-color: #24613e !important;
    background: #24613e !important;
    color: #ffffff !important;
    border-color: #24613e !important;
    box-shadow: none !important;
}

div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p,
div[data-testid="stSegmentedControl"] button[aria-checked="true"] p {
    color: #ffffff !important;
}

    div[data-testid="stSegmentedControl"] button:hover {
        border-color: #8fb49a !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: #24613e !important;
    color: #ffffff !important;
    border-color: #24613e !important;
    box-shadow: none !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
    padding-top: 0.55rem;
    padding-bottom: 0.55rem;
}

div[data-testid="stAlert"] p {
    font-size: 0.93rem;
    line-height: 1.45;
}

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e1e9e1;
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #24613e;
        color: #ffffff;
        border: 1px solid #24613e;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #17462b;
        color: #ffffff;
        border-color: #17462b;
    }

    h2, h3 {
        color: #174c32;
    }

    hr {
        border-color: #e0e8e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
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


@st.cache_resource
def load_risk_model(file_path):
    return joblib.load(file_path)


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🌾 GrainGuard AI</h1>
        <p>
            Smart grain-storage monitoring and preliminary
            aflatoxin risk assessment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 5. SIMPLE TOP NAVIGATION
# --------------------------------------------------

page = st.segmented_control(
    "Navigation",
    options=["Dashboard", "Risk Assessment", "About"],
    default="Dashboard",
    label_visibility="collapsed",
)

st.caption(
    "Historical demonstration data • Not a live monitoring system"
)


# --------------------------------------------------
# 6. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown("## Storage Monitoring Dashboard")

    if not SENSOR_FILE.exists():
        st.error(
            "Sensor data file not found. Please check that "
            "data/processed/sensor_readings_cleaned.csv exists."
        )
        st.stop()

    try:
        sensor_data = load_sensor_data(str(SENSOR_FILE))

    except Exception as error:
        st.error(f"Unable to load sensor data: {error}")
        st.stop()

    if sensor_data.empty:
        st.warning("No valid sensor readings are available.")
        st.stop()

    # ----------------------------------------------
    # REVIEW LIMITS
    # ----------------------------------------------

    st.markdown("### Environmental Review Limits")

    st.caption(
        "These are configurable demonstration limits, not "
        "validated grain-storage recommendations or official "
        "aflatoxin safety thresholds."
    )

    limit_col1, limit_col2 = st.columns(2)

    with limit_col1:
        temperature_limit = st.number_input(
            "Temperature review limit (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=30.0,
            step=1.0,
        )

    with limit_col2:
        humidity_limit = st.number_input(
            "Humidity review limit (% RH)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=5.0,
        )

    # ----------------------------------------------
    # SUMMARY CARDS
    # ----------------------------------------------

    temperature_alerts = (
        sensor_data["Temperature"] >= temperature_limit
    )

    humidity_alerts = (
        sensor_data["Humidity"] >= humidity_limit
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Total Readings",
        f"{len(sensor_data):,}",
    )

    metric2.metric(
        "Sensors Monitored",
        sensor_data["Sensor"].nunique(),
    )

    metric3.metric(
        "Temperature Alerts",
        f"{int(temperature_alerts.sum()):,}",
    )

    metric4.metric(
        "Humidity Alerts",
        f"{int(humidity_alerts.sum()):,}",
    )

    st.caption(
        "Alert counts represent historical readings across all "
        "included sensors. They are not live alerts or confirmed "
        "grain contamination."
    )

    st.divider()

    # ----------------------------------------------
    # DOWNLOAD DATA
    # ----------------------------------------------

    st.markdown("### Download Sensor Data")

    csv_data = sensor_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned Sensor Data",
        data=csv_data,
        file_name="grainguard_sensor_readings.csv",
        mime="text/csv",
    )

    st.divider()

    # ----------------------------------------------
    # SENSOR SELECTION
    # ----------------------------------------------

    st.markdown("### Individual Sensor Monitoring")

    sensors = sorted(sensor_data["Sensor"].unique().tolist())

    selected_sensor = st.selectbox(
        "Select a sensor",
        sensors,
    )

    selected_data = sensor_data[
        sensor_data["Sensor"] == selected_sensor
    ].copy()

    latest = selected_data.iloc[-1]

    sensor_col1, sensor_col2, sensor_col3 = st.columns(3)

    sensor_col1.metric(
        "Latest Temperature",
        f"{latest['Temperature']:.1f} °C",
    )

    sensor_col2.metric(
        "Latest Humidity",
        f"{latest['Humidity']:.1f} %",
    )

    sensor_col3.metric(
        "Readings in Latest 24-Hour Data Window",
        f"{int((selected_data['Timestamp'] >= latest['Timestamp'] - pd.Timedelta(hours=24)).sum()):,}",
    )

    st.caption(
        f"Selected sensor: {selected_sensor} | "
        f"Latest recorded timestamp: {latest['Timestamp']}"
    )

    latest_temperature_alert = (
        latest["Temperature"] >= temperature_limit
    )

    latest_humidity_alert = (
        latest["Humidity"] >= humidity_limit
    )

    if latest_temperature_alert or latest_humidity_alert:
        messages = []

        if latest_temperature_alert:
            messages.append("temperature is at or above its review limit")

        if latest_humidity_alert:
            messages.append("humidity is at or above its review limit")

        st.warning(
            "Review required: " + " and ".join(messages) + "."
        )
    else:
        st.success(
            "The latest reading is below both configured review limits."
        )

    # ----------------------------------------------
    # IMPROVEMENT 1: OVERALL ENVIRONMENTAL STATUS
    # ----------------------------------------------

    st.markdown("## Overall Environmental Monitoring Status")

    st.caption(
        "This summary checks the latest recorded reading from each sensor "
        "against the review limits above. It is based on historical data, "
        "not live monitoring or a validated aflatoxin prediction."
    )

    latest_by_sensor_status = (
        sensor_data.sort_values("Timestamp")
        .groupby("Sensor", as_index=False)
        .tail(1)
        .copy()
    )

    latest_by_sensor_status["Temperature Alert"] = (
        latest_by_sensor_status["Temperature"] >= temperature_limit
    )

    latest_by_sensor_status["Humidity Alert"] = (
        latest_by_sensor_status["Humidity"] >= humidity_limit
    )

    latest_by_sensor_status["Needs Review"] = (
        latest_by_sensor_status["Temperature Alert"]
        | latest_by_sensor_status["Humidity Alert"]
    )

    sensors_needing_review = int(
        latest_by_sensor_status["Needs Review"].sum()
    )

    total_latest_sensors = len(latest_by_sensor_status)

    latest_temp_alert_count = int(
        latest_by_sensor_status["Temperature Alert"].sum()
    )

    latest_humidity_alert_count = int(
        latest_by_sensor_status["Humidity Alert"].sum()
    )

    if sensors_needing_review > 0:
        st.warning(
            f"**Review needed:** {sensors_needing_review} of "
            f"{total_latest_sensors} sensors have a latest recorded "
            "temperature or humidity reading at/above a configured limit."
        )
    else:
        st.success(
            f"**No review-limit exceedances in the latest records:** "
            f"all {total_latest_sensors} sensors have latest recorded "
            "readings below both configured limits."
        )

    status_col1, status_col2, status_col3 = st.columns(3)

    status_col1.metric(
        "Sensors Needing Review",
        f"{sensors_needing_review} / {total_latest_sensors}",
    )

    status_col2.metric(
        "Latest Temperature Exceedances",
        latest_temp_alert_count,
    )

    status_col3.metric(
        "Latest Humidity Exceedances",
        latest_humidity_alert_count,
    )

    status_table = latest_by_sensor_status[
        [
            "Sensor",
            "Timestamp",
            "Temperature",
            "Humidity",
            "Temperature Alert",
            "Humidity Alert",
            "Needs Review",
        ]
    ].sort_values("Sensor").copy()

    status_table["Temperature"] = status_table["Temperature"].round(2)
    status_table["Humidity"] = status_table["Humidity"].round(2)

    status_table = status_table.rename(
        columns={
            "Temperature Alert": "Temperature Limit Reached",
            "Humidity Alert": "Humidity Limit Reached",
            "Needs Review": "Needs Review",
        }
    )

    st.dataframe(
        status_table,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Interpretation: 'Needs Review' means a recorded environmental "
        "reading crossed a user-configured review limit. It does not "
        "confirm fungal growth, aflatoxin presence, or grain safety."
    )

    st.divider()

    # ----------------------------------------------
    # SENSOR CHARTS
    # ----------------------------------------------

    st.markdown("### Environmental Trends")

    temperature_chart = go.Figure()

    temperature_chart.add_trace(
        go.Scatter(
            x=selected_data["Timestamp"],
            y=selected_data["Temperature"],
            mode="lines",
            name="Temperature",
            line=dict(width=1.5),
        )
    )

    temperature_chart.add_hline(
        y=temperature_limit,
        line_dash="dash",
        annotation_text="Temperature review limit",
    )

    temperature_chart.update_layout(
        title="Temperature Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Temperature (°C)",
        height=350,
        template="plotly_white",
    )

    st.plotly_chart(
        temperature_chart,
        use_container_width=True,
    )

    humidity_chart = go.Figure()

    humidity_chart.add_trace(
        go.Scatter(
            x=selected_data["Timestamp"],
            y=selected_data["Humidity"],
            mode="lines",
            name="Humidity",
            line=dict(width=1.5),
        )
    )

    humidity_chart.add_hline(
        y=humidity_limit,
        line_dash="dash",
        annotation_text="Humidity review limit",
    )

    humidity_chart.update_layout(
        title="Humidity Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Relative Humidity (%)",
        height=350,
        template="plotly_white",
    )

    st.plotly_chart(
        humidity_chart,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------
    # ALERT HISTORY
    # ----------------------------------------------

    st.markdown("## 🚨 Alert History")

    st.write(
        "Review historical readings that meet or exceed "
        "the configured temperature or humidity limits."
    )

    st.caption(
        "This table is generated from recorded data. It does not "
        "represent real-time alerts or confirmed aflatoxin detection."
    )

    alert_history = sensor_data.copy()

    alert_history["Temperature Alert"] = (
        alert_history["Temperature"] >= temperature_limit
    )

    alert_history["Humidity Alert"] = (
        alert_history["Humidity"] >= humidity_limit
    )

    alert_history = alert_history[
        alert_history["Temperature Alert"]
        | alert_history["Humidity Alert"]
    ].copy()

    def get_alert_type(row):
        if row["Temperature Alert"] and row["Humidity Alert"]:
            return "Temperature + Humidity"

        if row["Temperature Alert"]:
            return "Temperature"

        return "Humidity"

    if not alert_history.empty:

        alert_history["Alert Type"] = alert_history.apply(
            get_alert_type,
            axis=1,
        )

        alert_history = alert_history.sort_values(
            "Timestamp",
            ascending=False,
        )

        sensor_filter_options = ["All Sensors"] + sensors

        selected_alert_sensor = st.selectbox(
            "Filter alert history by sensor",
            sensor_filter_options,
            key="alert_sensor_filter",
        )

        if selected_alert_sensor != "All Sensors":
            alert_history = alert_history[
                alert_history["Sensor"] == selected_alert_sensor
            ]

        alert_type_options = [
            "All Alert Types",
            "Temperature",
            "Humidity",
            "Temperature + Humidity",
        ]

        selected_alert_type = st.selectbox(
            "Filter by alert type",
            alert_type_options,
            key="alert_type_filter",
        )

        if selected_alert_type != "All Alert Types":
            alert_history = alert_history[
                alert_history["Alert Type"] == selected_alert_type
            ]

        alert_col1, alert_col2 = st.columns(2)

        with alert_col1:
            st.metric(
                "Matching Historical Alerts",
                f"{len(alert_history):,}",
            )

        with alert_col2:
            st.metric(
                "Sensors in Filtered Results",
                alert_history["Sensor"].nunique(),
            )

        display_columns = [
            "Timestamp",
            "Sensor",
            "Temperature",
            "Humidity",
            "Alert Type",
        ]

        if alert_history.empty:
            st.info(
                "No historical alerts match the selected filters."
            )
        else:
            display_history = alert_history[display_columns].copy()

            display_history["Temperature"] = (
                display_history["Temperature"].round(2)
            )

            display_history["Humidity"] = (
                display_history["Humidity"].round(2)
            )

            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
            )

            alert_csv = display_history.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Filtered Alert History",
                data=alert_csv,
                file_name="grainguard_alert_history.csv",
                mime="text/csv",
            )

    else:
        st.success(
            "No historical readings meet the currently configured "
            "review limits."
        )

    st.divider()

    # ----------------------------------------------
    # LATEST READING BY SENSOR
    # ----------------------------------------------

    st.markdown("### Latest Reading by Sensor")

    latest_by_sensor = (
        sensor_data.sort_values("Timestamp")
        .groupby("Sensor", as_index=False)
        .tail(1)
        .sort_values("Sensor")
    )

    latest_by_sensor = latest_by_sensor[
        ["Sensor", "Timestamp", "Temperature", "Humidity"]
    ].copy()

    latest_by_sensor["Temperature"] = (
        latest_by_sensor["Temperature"].round(2)
    )

    latest_by_sensor["Humidity"] = (
        latest_by_sensor["Humidity"].round(2)
    )

    st.dataframe(
        latest_by_sensor,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 7. RISK ASSESSMENT
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
# 8. ABOUT
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
        - Preliminary survey-based risk assessment
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
