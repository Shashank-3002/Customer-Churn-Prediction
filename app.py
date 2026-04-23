import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# -------------------------------
# Load model & preprocessing
# -------------------------------
model = tf.keras.models.load_model("churn_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:   # ColumnTransformer
    ct = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("💳 Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn.")

# -------------------------------
# User Inputs
# -------------------------------
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age", 18, 92, 30)
tenure = st.slider("Tenure", 0, 10, 3)

balance = st.number_input("Balance", value=50000.0)
credit_score = st.number_input("Credit Score", value=650)

num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# -------------------------------
# Prepare Input Data
# -------------------------------
input_dict = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}

input_df = pd.DataFrame([input_dict])

# Encode gender
input_df["Gender"] = label_encoder.transform(input_df["Gender"])

# Apply ColumnTransformer (handles Geography encoding)
input_encoded = ct.transform(input_df)

# Scale
input_scaled = scaler.transform(input_encoded)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):

    prob = model.predict(input_scaled)[0][0]

    # 🔥 Improved threshold (from your model tuning)
    threshold = 0.4
    prediction = 1 if prob > threshold else 0

    st.subheader("📊 Prediction Result")

    st.write(f"**Churn Probability:** {prob:.2f}")

    # -------------------------------
    # Risk Segmentation (UNIQUE FEATURE)
    # -------------------------------
    if prob > 0.7:
        risk = "🔴 High Risk"
    elif prob > 0.4:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.write(f"**Risk Level:** {risk}")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Model: ANN with class imbalance handling, ROC optimization & threshold tuning")