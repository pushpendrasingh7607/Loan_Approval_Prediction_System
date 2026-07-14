import os
import joblib
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")

st.markdown("""
This system uses a machine learning model combined with a **CIBIL Score Eligibility Policy** to determine if a loan should be approved.

### 📋 CIBIL Score Eligibility Criteria:
* **Loan Amount < 500,000**: Minimum CIBIL score of **600** required.
* **Loan Amount 500,000 - 2,000,000**: Minimum CIBIL score of **700** required.
* **Loan Amount > 2,000,000**: Minimum CIBIL score of **750** required.
""")

# -----------------------------
# Load Model Files
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Model")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
    columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()

# -----------------------------
# User Inputs
# -----------------------------

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    value=10
)

cibil_score = st.slider(
    "CIBIL Score",
    300,
    900,
    750
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=1000000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=500000
)

# -----------------------------
# Encode Categorical Features
# -----------------------------

education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Loan Status"):

    # 1. First run the CIBIL Eligibility rules checks
    eligible = True
    cibil_rejection_reason = ""
    
    if loan_amount < 500000:
        if cibil_score < 600:
            eligible = False
            cibil_rejection_reason = f"CIBIL score is {cibil_score}, but a minimum of 600 is required for loans under 500,000."
    elif loan_amount <= 2000000:
        if cibil_score < 700:
            eligible = False
            cibil_rejection_reason = f"CIBIL score is {cibil_score}, but a minimum of 700 is required for loans between 500,000 and 2,000,000."
    else:
        if cibil_score < 750:
            eligible = False
            cibil_rejection_reason = f"CIBIL score is {cibil_score}, but a minimum of 750 is required for loans above 2,000,000."

    if not eligible:
        st.error("❌ Loan Rejected")
        st.warning(f"**Reason**: {cibil_rejection_reason} (CIBIL Score Policy Check)")
    else:
        # 2. Fall back to Machine Learning Model Prediction
        # Note: Model was trained on raw (unscaled) features subset selected via SelectKBest
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

        # Match training columns (selects only the 8 columns in the exact order model expects)
        input_data = input_data.reindex(columns=columns, fill_value=0)

        # Prediction on raw features
        prediction = model.predict(input_data)

        # In training label encoding: ' Approved' -> 0, ' Rejected' -> 1
        if prediction[0] == 0:
            st.success("✅ Loan Approved")
            st.info("The CIBIL score meets the policy requirements, and the ML model predicted approval.")
        else:
            st.error("❌ Loan Rejected")
            st.info("The CIBIL score meets the policy requirements, but the ML model predicted rejection based on other features (e.g. assets, income, etc.).")