import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("credit.pkl")

st.title("💳 Credit Default Prediction")

st.write("Enter customer details:")

# ================= INPUT =================

LIMIT_BAL = st.number_input("Credit Limit", 10000, 1000000)
AGE = st.number_input("Age", 18, 100)

PAY_0 = st.number_input("PAY_0 (Recent Delay)", -2, 8)
PAY_2 = st.number_input("PAY_2", -2, 8)
PAY_3 = st.number_input("PAY_3", -2, 8)

BILL_AMT1 = st.number_input("Bill Amount 1", 0, 1000000)
BILL_AMT2 = st.number_input("Bill Amount 2", 0, 1000000)
BILL_AMT3 = st.number_input("Bill Amount 3", 0, 1000000)

PAY_AMT1 = st.number_input("Payment Amount 1", 0, 1000000)
PAY_AMT2 = st.number_input("Payment Amount 2", 0, 1000000)
PAY_AMT3 = st.number_input("Payment Amount 3", 0, 1000000)

# ================= PREDICTION =================

if st.button("Predict"):

    # ⚠️ MUST match training feature order EXACTLY
    input_data = np.array([[ 
        LIMIT_BAL, 2, 2, 2, AGE,
        PAY_0, PAY_2, PAY_3, 0, 0, 0,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, 0, 0, 0,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, 0, 0, 0
    ]])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ High Risk of Default ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk ({prob:.2f})")

    # ================= SIMPLE EXPLANATION =================

    st.subheader("🔍 Why this prediction?")

    reasons = []

    if PAY_0 > 1:
        reasons.append("Recent payment delay is high")

    if PAY_AMT1 < BILL_AMT1:
        reasons.append("Payment is less than recent bill")

    if LIMIT_BAL > 500000:
        reasons.append("High credit limit exposure")

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
