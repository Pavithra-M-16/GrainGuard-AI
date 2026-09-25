import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/grain_risk_model.pkl")

# Example storage conditions
sample = pd.DataFrame([{
    "Crop": "Maize",
    "PrevCrop": 2,
    "DryMeth": 1,
    "StorType": 1,
    "StorLeng": 180,
    "Altitude": 500,
    "Wealth": 0,
    "AwarMyco": 1
}])

# Make prediction
prediction = model.predict(sample)

print("Predicted Risk:", prediction[0])