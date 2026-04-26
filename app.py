import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

os.environ["PYTHONWARNINGS"] = "ignore"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_files()

st.title("🏦 Customer Churn Prediction System")
st.markdown("### Predict whether a customer is likely to leave the bank")

st.write(
    "This project uses Machine Learning to predict customer churn "
    "based on customer profile and banking activity."
)

st.divider()

st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        18, 92, 35
    )

    tenure = st.slider(
        "Tenure",
        0, 10, 5
    )

with col2:
    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    num_products = st.slider(
        "Number of Products",
        1, 4, 2
    )

    has_cr_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active_member = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

st.divider()

gender = 1 if gender == "Male" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

if st.button("🔮 Predict Churn", use_container_width=True):

    input_data = np.array([[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
        geo_germany,
        geo_spain
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if probability > 0.70:
        risk_level = "🔴 High Risk"
    elif probability > 0.40:
        risk_level = "🟠 Medium Risk"
    else:
        risk_level = "🟢 Low Risk"

    st.write(f"### Risk Level: {risk_level}")

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
        st.info("Suggested Action: Offer retention benefits and personalized support.")
    else:
        st.success("✅ Customer is likely to STAY")
        st.info("Suggested Action: Maintain customer engagement and loyalty.")

st.markdown("---")
st.caption(
    "Built using Scikit-learn + Streamlit | Random Forest Classifier"
)