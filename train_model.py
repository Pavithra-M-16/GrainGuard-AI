
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "cleaned_grain_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "grain_risk_model.pkl"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = [
    "Crop",
    "PrevCrop",
    "DryMeth",
    "StorType",
    "StorLeng",
    "Altitude",
    "Wealth",
    "AwarMyco",
    "TotalAfla",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()


# --------------------------------------------------
# 3. CREATE PROTOTYPE TARGET
# --------------------------------------------------

# 1.2 is the dataset median used for this prototype.
# It is NOT an official aflatoxin safety threshold.

df["Risk"] = df["TotalAfla"].apply(
    lambda value: "Lower" if value <= 1.2 else "Higher"
)


# --------------------------------------------------
# 4. DEFINE INPUTS AND TARGET
# --------------------------------------------------

feature_columns = [
    "Crop",
    "PrevCrop",
    "DryMeth",
    "StorType",
    "StorLeng",
    "Altitude",
    "Wealth",
    "AwarMyco",
]

X = df[feature_columns].copy()
y = df["Risk"]


# --------------------------------------------------
# 5. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# --------------------------------------------------

categorical_columns = [
    "Crop",
    "PrevCrop",
    "DryMeth",
    "StorType",
    "Wealth",
    "AwarMyco",
]

numeric_columns = [
    "StorLeng",
    "Altitude",
]


# --------------------------------------------------
# 6. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        (
            "numerical",
            "passthrough",
            numeric_columns,
        ),
    ]
)


# --------------------------------------------------
# 7. BUILD MODEL PIPELINE
# --------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", classifier),
    ]
)


# --------------------------------------------------
# 8. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 9. TRAIN AND EVALUATE
# --------------------------------------------------

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# 10. SAVE TRAINED MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, MODEL_FILE)

print("\nModel saved successfully:")
print(MODEL_FILE)
