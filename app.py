import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("credit.pkl")

# ---------------- HEADER ----------------

st.title("💳 Credit Default Prediction")
st.markdown("Predict whether a customer is likely to default on the next month's credit card payment.")

# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

with col1:

    LIMIT_BAL = st.number_input(
        "💰 Credit Limit",
        min_value=10000,
        max_value=1000000,
        value=50000,
        step=1000
    )

    AGE = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=80,
        value=30
    )

    SEX = st.selectbox(
        "👤 Gender",
        ["Male", "Female"]
    )

    EDUCATION = st.selectbox(
        "🎓 Education",
        ["Graduate", "University", "High School"]
    )

    MARRIAGE = st.selectbox(
        "💍 Marital Status",
        ["Single", "Married", "Other"]
    )

with col2:

    PAY_0 = st.selectbox(
        "📉 PAY_0",
        [-2,-1,0,1,2,3,4,5,6,7,8]
    )

    PAY_2 = st.selectbox(
        "📉 PAY_2",
        [-2,-1,0,1,2,3,4,5,6,7,8]
    )

    PAY_3 = st.selectbox(
        "📉 PAY_3",
        [-2,-1,0,1,2,3,4,5,6,7,8]
    )

    BILL_AMT1 = st.number_input(
        "Bill Amount 1",
        min_value=0,
        value=0,
        step=100
    )

    BILL_AMT2 = st.number_input(
        "Bill Amount 2",
        min_value=0,
        value=0,
        step=100
    )

    BILL_AMT3 = st.number_input(
        "Bill Amount 3",
        min_value=0,
        value=0,
        step=100
    )

st.subheader("💵 Payment Information")

col3, col4, col5 = st.columns(3)

with col3:
    PAY_AMT1 = st.number_input(
        "Payment Amount 1",
        min_value=0,
        value=0,
        step=100
    )

with col4:
    PAY_AMT2 = st.number_input(
        "Payment Amount 2",
        min_value=0,
        value=0,
        step=100
    )

with col5:
    PAY_AMT3 = st.number_input(
        "Payment Amount 3",
        min_value=0,
        value=0,
        step=100
    )

# ---------------- ENCODING ----------------

SEX = 1 if SEX == "Male" else 2

EDUCATION = {
    "Graduate": 1,
    "University": 2,
    "High School": 3
}[EDUCATION]

MARRIAGE = {
    "Married": 1,
    "Single": 2,
    "Other": 3
}[MARRIAGE]

# ---------------- PREDICTION ----------------

if st.button("🚀 Predict Risk", use_container_width=True):

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

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric(
        "Default Probability",
        f"{prob*100:.1f}%"
    )

    st.progress(float(prob))

    if pred == 1:
        st.error("🔴 Likely to Default")
    else:
        st.success("🟢 Not Likely to Default")

    st.subheader("🔍 Risk Factors")

    reasons = []

    if PAY_0 > 1:
        reasons.append("Recent payment delay is high")

    if PAY_AMT1 < BILL_AMT1:
        reasons.append("Payment amount is lower than outstanding bill")

    if LIMIT_BAL > 500000:
        reasons.append("High credit exposure")

    if BILL_AMT1 > 50000:
        reasons.append("Large outstanding bill amount")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.info("No major risk factors detected")

# ---------------- MODEL COMPARISON ----------------

st.markdown("---")
st.subheader("📈 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Precision": [0.37, 0.63, 0.50],
    "Recall": [0.62, 0.36, 0.58],
    "F1 Score": [0.46, 0.46, 0.54]
})

st.dataframe(comparison_df, use_container_width=True)
