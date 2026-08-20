# 🏦 Loan Approval Prediction System

A machine learning-powered web application that predicts loan approval status using an XGBoost model combined with a rule-based CIBIL Score Eligibility Policy.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Overview

This project predicts whether a loan application will be **Approved** or **Rejected** based on applicant details such as income, credit score, assets, and loan amount. It uses a **two-stage decision pipeline**:

1. **CIBIL Score Policy Check** — Enforces tiered minimum credit score requirements based on the loan amount.
2. **ML Model Prediction** — If CIBIL eligibility is met, an XGBoost classifier makes the final prediction.

---

## 🎯 Features

- 📊 Real-time loan approval prediction via an interactive web UI
- 🔒 Rule-based CIBIL score policy enforcement
- 🤖 XGBoost machine learning model trained on real-world loan data
- 📋 Feature selection using `SelectKBest`
- 🖥️ Clean and intuitive Streamlit interface

---

## 📋 CIBIL Score Eligibility Criteria

| Loan Amount | Minimum CIBIL Score Required |
|---|---|
| Less than ₹5,00,000 | **600** |
| ₹5,00,000 – ₹20,00,000 | **700** |
| More than ₹20,00,000 | **750** |

---

## 🗂️ Project Structure

```
Loan_Approval_Prediction_System/
│
├── Dataset/
│   └── loan.csv                   # Raw dataset used for training
│
├── Model/
│   ├── loan_model.pkl             # Trained XGBoost model
│   ├── columns.pkl                # Selected feature columns
│   └── scaler.pkl                 # Standard scaler (used during training)
│
├── Notebook/
│   └── test_prediction.py         # Script for testing model predictions
│
├── Streamlit_App/
│   └── app.py                     # Main Streamlit web application
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/pushpendrasingh7607/Loan_Approval_Prediction_System.git
cd Loan_Approval_Prediction_System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit pandas scikit-learn xgboost joblib
```

### 4. Run the App

```bash
streamlit run Streamlit_App/app.py
```

The app will open in your browser at **http://localhost:8501**

---

## 🧾 Input Features

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of financial dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income (₹) |
| `loan_amount` | Requested loan amount (₹) |
| `loan_term` | Loan repayment period (years) |
| `cibil_score` | Credit score (300–900) |
| `residential_assets_value` | Value of residential property (₹) |
| `commercial_assets_value` | Value of commercial property (₹) |
| `luxury_assets_value` | Value of luxury assets (₹) |
| `bank_asset_value` | Value of bank/liquid assets (₹) |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Feature Selection | SelectKBest (top 8 features) |
| Label Encoding | `Approved` → 0, `Rejected` → 1 |
| Training Data | `Dataset/loan.csv` |

---

## 📸 App Preview

The app provides:
- Input fields for all applicant details
- A **"Predict Loan Status"** button
- Clear **✅ Approved** or **❌ Rejected** result with reason

---

## 👤 Author

**Pushpendra Singh**
- GitHub: [@pushpendrasingh7607](https://github.com/pushpendrasingh7607)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
