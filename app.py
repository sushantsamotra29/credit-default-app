import streamlit as st
import numpy as np
import joblib

model = joblib.load("credit.pkl")

st.title("💳 Credit Default Prediction")

LIMIT_BAL = st.number_input("Credit Limit", 10000, 1000000)
AGE = st.number_input("Age", 18, 100)

PAY_0 = st.number_input("PAY_0", -2, 8)
PAY_2 = st.number_input("PAY_2", -2, 8)
PAY_3 = st.number_input("PAY_3", -2, 8)

BILL_AMT1 = st.number_input("Bill Amt 1", 0, 1000000)
BILL_AMT2 = st.number_input("Bill Amt 2", 0, 1000000)
BILL_AMT3 = st.number_input("Bill Amt 3", 0, 1000000)

PAY_AMT1 = st.number_input("Pay Amt 1", 0, 1000000)
PAY_AMT2 = st.number_input("Pay Amt 2", 0, 1000000)
PAY_AMT3 = st.number_input("Pay Amt 3", 0, 1000000)

# Feature Engineering
TOTAL_BILL = BILL_AMT1 + BILL_AMT2 + BILL_AMT3
TOTAL_PAY = PAY_AMT1 + PAY_AMT2 + PAY_AMT3

PAY_RATIO = TOTAL_PAY / (abs(TOTAL_BILL) + 1)
AVG_DELAY = (PAY_0 + PAY_2 + PAY_3) / 3
MAX_DELAY = max(PAY_0, PAY_2, PAY_3)

TOTAL_BILL = np.log1p(abs(TOTAL_BILL))
TOTAL_PAY = np.log1p(TOTAL_PAY)

if st.button("Predict"):

    input_data = np.array([[ 
        LIMIT_BAL, 2, 2, 2, AGE,
        PAY_0, PAY_2, PAY_3, 0, 0, 0,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, 0, 0, 0,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, 0, 0, 0,
        TOTAL_BILL, TOTAL_PAY, PAY_RATIO, AVG_DELAY, MAX_DELAY
    ]])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"High Risk ({prob:.2f})")
    else:
        st.success(f"Low Risk ({prob:.2f})")