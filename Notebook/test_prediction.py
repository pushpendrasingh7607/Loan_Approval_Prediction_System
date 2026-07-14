import os
import joblib
import pandas as pd

MODEL_DIR = "Model"

try:
    model = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
    columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("Models loaded successfully.")
    print("Scaler features:", scaler.feature_names_in_)
    print("Model features:", columns)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Input data matching the original 11 features
no_of_dependents = 2
education = 1 # Graduate
self_employed = 0 # No
income_annum = 5000000
loan_amount = 10000000
loan_term = 10
cibil_score = 750
residential_assets_value = 20000000
commercial_assets_value = 10000000
luxury_assets_value = 15000000
bank_asset_value = 8000000

scaler_features = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
]

input_data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]
})

# Reorder columns to match scaler features
input_data = input_data[scaler_features]

# Scale
input_scaled = scaler.transform(input_data)

# Extract only model columns
input_scaled_df = pd.DataFrame(input_scaled, columns=scaler_features)
input_model_ready = input_scaled_df[columns]

# Predict
prediction = model.predict(input_model_ready)
print("Prediction:", prediction)
