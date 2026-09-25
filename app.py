import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("model/grain_risk_model.pkl")


# Page title
st.set_page_config(
    page_title="GrainGuard AI",
    page_icon="🌾",
    layout="centered"
)


# Project heading
st.title("🌾 GrainGuard AI")
st.subheader("Post-Harvest Grain Contamination Risk Predictor")

st.write(
    "Enter the grain storage conditions below to estimate "
    "the contamination-related risk level."
)

st.info(
    "Prototype warning: This system provides a data-driven risk "
    "estimate and does not replace laboratory testing."
)


# User inputs
crop = st.selectbox(
    "Crop",
    ["Maize", "Groundnut"]
)

prev_crop = st.selectbox(
    "Previous Crop",
    [1, 2, 3, 4]
)

dry_method = st.selectbox(
    "Drying Method",
    [1, 2]
)

storage_type = st.selectbox(
    "Storage Type",
    [0, 1]
)

storage_length = st.number_input(
    "Storage Duration",
    min_value=1.0,
    value=180.0
)

altitude = st.number_input(
    "Altitude",
    min_value=0.0,
    value=500.0
)

wealth = st.selectbox(
    "Wealth Category",
    [0, 1]
)

awareness = st.selectbox(
    "Mycotoxin Awareness",
    [1, 2]
)


# Prediction button
if st.button("🔍 Predict Risk"):

    sample = pd.DataFrame([{
        "Crop": crop,
        "PrevCrop": prev_crop,
        "DryMeth": dry_method,
        "StorType": storage_type,
        "StorLeng": storage_length,
        "Altitude": altitude,
        "Wealth": wealth,
        "AwarMyco": awareness
    }])

    prediction = model.predict(sample)[0]

    st.subheader("Prediction Result")

    if prediction == "Higher":
        st.error("⚠️ Higher Risk")
        st.write(
            "The entered storage conditions are associated with "
            "a higher contamination-risk category in the prototype model."
        )
    else:
        st.success("✅ Lower Risk")
        st.write(
            "The entered storage conditions are associated with "
            "a lower contamination-risk category in the prototype model."
        )