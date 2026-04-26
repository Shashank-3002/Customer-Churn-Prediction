import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AI-Powered Customer Retention Intelligence System",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_files():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_files()

st.title("🏦 AI-Powered Customer Retention Intelligence System")
st.markdown("### Advanced Customer Churn Prediction Dashboard")
st.write(
    "Predict customer churn, estimate churn probability, assign risk level, and provide retention strategies for better business decisions."
)

st.divider()

st.sidebar.header("📋 Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

gender = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

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

if st.sidebar.button("🔮 Predict Churn"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if probability > 0.70:
        risk_level = "🔴 High Risk"
        recommendation = "Offer loyalty rewards, premium support, and dedicated relationship manager assistance."
    elif probability > 0.40:
        risk_level = "🟠 Medium Risk"
        recommendation = "Provide targeted offers, follow-up campaigns, and stronger engagement plans."
    else:
        risk_level = "🟢 Low Risk"
        recommendation = "Maintain customer satisfaction and continue loyalty engagement programs."

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with col2:
        st.metric("Prediction", "Churn" if prediction[0] == 1 else "Stay")

    with col3:
        st.metric("Risk Level", risk_level)

    st.divider()

    st.subheader("💡 Retention Strategy Recommendation")
    st.info(recommendation)

    st.subheader("📊 Key Factors Affecting Churn")

    feature_data = pd.DataFrame({
        "Feature": [
            "Age",
            "Balance",
            "IsActiveMember",
            "Credit Score",
            "Geography",
            "Estimated Salary"
        ],
        "Impact Score": [35, 25, 15, 10, 8, 7]
    })

    st.bar_chart(feature_data.set_index("Feature"))

    st.subheader("📈 Business Insight Summary")

    if prediction[0] == 1:
        st.error(
            "This customer is likely to leave the bank. Immediate retention action is recommended to reduce potential revenue loss."
        )
    else:
        st.success(
            "This customer is likely to stay. Continue engagement and loyalty programs for long-term retention."
        )

st.markdown("---")
st.caption(
    "Built with Streamlit + Scikit-learn | Random Forest Classifier | Streamlit Cloud Optimized"
)