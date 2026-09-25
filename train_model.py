import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


# Load cleaned dataset
df = pd.read_csv("data/cleaned_grain_data.csv")


# Create a dataset-derived prototype target
# 1.2 ppb is the median of this dataset, NOT an official safety limit.
df["Risk"] = df["TotalAfla"].apply(
    lambda x: "Lower" if x <= 1.2 else "Higher"
)


# Separate inputs and target
X = df.drop(["TotalAfla", "Risk"], axis=1)
y = df["Risk"]


# Categorical columns
categorical_columns = [
    "Crop",
    "PrevCrop",
    "DryMeth",
    "StorType"
]


# Numerical columns
numeric_columns = [
    "StorLeng",
    "Altitude",
    "Wealth",
    "AwarMyco"
]


# Convert categorical values into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Random Forest classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Combine preprocessing and model
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train
pipeline.fit(X_train, y_train)


# Predict
predictions = pipeline.predict(X_test)


# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("Binary classification model trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy, 2))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Save model
joblib.dump(pipeline, "model/grain_risk_model.pkl")

print("\nModel saved successfully!")