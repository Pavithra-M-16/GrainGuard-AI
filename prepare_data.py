import pandas as pd

# Load the real IITA dataset
df = pd.read_csv("data/maize_and_groundnut-survey-2012.csv")

# Select the columns we need
columns = [
    "Crop",
    "PrevCrop",
    "DryMeth",
    "StorType",
    "StorLeng",
    "Altitude",
    "Wealth",
    "AwarMyco",
    "TotalAfla"
]

df = df[columns]

# Remove rows where the target value is missing
df = df.dropna(subset=["TotalAfla"])

# Treat undocumented StorType code 11 as missing
df["StorType"] = df["StorType"].replace(11, pd.NA)

# Fill missing values in numeric columns with the median
numeric_columns = [
    "StorLeng",
    "Altitude",
    "Wealth",
    "AwarMyco"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing values in categorical columns with the most common value
categorical_columns = [
    "Crop",
    "PrevCrop",
    "DryMeth",
    "StorType"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Save the cleaned dataset
df.to_csv("data/cleaned_grain_data.csv", index=False)

print("Cleaned dataset created successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())