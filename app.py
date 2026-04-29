import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Credit Risk App", layout="centered")

# Load model
model = joblib.load("credit.pkl")

# Title
st.title("💳 Credit Default Prediction")

st.write("Enter customer details:")

# ================= INPUT =================

LIMIT_BAL = st.slider("💰 Credit Limit", 10000, 1000000, 50000)
AGE = st.slider("🎂 Age", 18, 80, 30)

SEX = st.selectbox("👤 Gender", ["Male", "Female"])
SEX = 1 if SEX == "Male" else 2

EDUCATION = st.selectbox("🎓 Education", ["Graduate", "University", "High School"])
EDUCATION = {"Graduate":1, "University":2, "High School":3}[EDUCATION]

MARRIAGE = st.selectbox("💍 Marital Status", ["Single", "Married", "Other"])
MARRIAGE = {"Married":1, "Single":2, "Other":3}[MARRIAGE]

PAY_0 = st.slider("📉 Recent Payment Delay (PAY_0)", -2, 8, 0)
PAY_2 = st.slider("PAY_2", -2, 8, 0)
PAY_3 = st.slider("PAY_3", -2, 8, 0)

BILL_AMT1 = st.number_input("Bill Amount 1", 0, 1000000)
BILL_AMT2 = st.number_input("Bill Amount 2", 0, 1000000)
BILL_AMT3 = st.number_input("Bill Amount 3", 0, 1000000)

PAY_AMT1 = st.number_input("Payment Amount 1", 0, 1000000)
PAY_AMT2 = st.number_input("Payment Amount 2", 0, 1000000)
PAY_AMT3 = st.number_input("Payment Amount 3", 0, 1000000)

# ================= PREDICTION =================

if st.button("Predict"):

    input_dict = {
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure all required features exist
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Match exact order
    input_df = input_df[model.feature_names_in_]

    # Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # ================= OUTPUT =================

    st.subheader("📊 Prediction Result")

    st.progress(float(prob))

    if prob > 0.7:
        st.error(f"🔴 High Risk ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f"🟠 Medium Risk ({prob:.2f})")
    else:
        st.success(f"🟢 Low Risk ({prob:.2f})")

    # ================= EXPLANATION =================

    st.subheader("🔍 Why this prediction?")

    reasons = []

    if PAY_0 > 1:
        reasons.append("Recent payment delay is high")

    if PAY_AMT1 < BILL_AMT1:
        reasons.append("Payment is less than recent bill")

    if LIMIT_BAL > 500000:
        reasons.append("High credit exposure")

    if BILL_AMT1 > 50000:
        reasons.append("High recent bill amount")

    if len(reasons) == 0:
        st.info("No major risk factors detected")
    else:
        for r in reasons:
            st.write("•", r)

# ================= MODEL COMPARISON =================

st.markdown("---")
st.subheader("📈 Model Comparison")

st.write("""
| Model | Precision | Recall | F1 Score |
|------|----------|--------|----------|
| Logistic Regression | 0.37 | 0.62 | 0.46 |
| Random Forest | 0.63 | 0.36 | 0.46 |
| XGBoost (Final Model) | 0.50 | 0.58 | 0.54 |
""")
