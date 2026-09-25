# 🌾 GrainGuard AI

## Post-Harvest Grain Contamination Risk Predictor

GrainGuard AI is a machine-learning prototype that estimates the contamination-related risk category of stored grains using storage and crop-related information.

## Problem

Post-harvest grain contamination can affect food safety and agricultural losses. Early identification of potentially higher-risk storage conditions can help support preventive action.

## Solution

GrainGuard AI uses a Random Forest classification model trained on a real public maize and groundnut survey dataset containing aflatoxin measurements and related agricultural/storage information.

The system takes information such as:

- Crop
- Previous crop
- Drying method
- Storage type
- Storage duration
- Altitude
- Wealth category
- Mycotoxin awareness

and predicts a prototype risk category:

- Lower Risk
- Higher Risk

## Machine Learning

The project uses:

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- Streamlit

The trained model achieved approximately **75% accuracy** on the held-out test set used in this prototype.

## Dataset

The prototype uses the public **Maize and Groundnut Aflatoxin Survey 2012** dataset from the International Institute of Tropical Agriculture (IITA).

The dataset contains measured aflatoxin values along with agricultural and storage-related information.

## Important Note

The risk categories used in this prototype are derived from the distribution of the available dataset and are **not official food-safety thresholds**.

This prototype does not directly measure fungal contamination or aflatoxin and does not replace laboratory testing.

## Application

The Streamlit dashboard allows users to enter storage conditions and receive an immediate prototype risk prediction.

## Project Structure

```text
GrainGuard-AI/
│
├── data/
│   ├── maize_and_groundnut-survey-2012.csv
│   ├── maize-and-groundnut-survey-2012_metadata.csv
│   └── cleaned_grain_data.csv
│
├── model/
│   └── grain_risk_model.pkl
│
├── notebooks/
│
├── prepare_data.py
├── train_model.py
├── test_model.py
├── app.py
├── requirements.txt
└── README.md