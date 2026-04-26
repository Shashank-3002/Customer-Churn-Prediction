# ============================================================
# AI CUSTOMER RETENTION INTELLIGENCE — Streamlit Cloud Safe
# app.py
# ============================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, features, threshold


try:
    model, scaler, FEATURES, THRESHOLD = load_artifacts()
except Exception as e:
    st.error("Model files not found or corrupted.")
    st.write(str(e))
    st.stop()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def risk_label(prob):
    if prob > 0.70:
        return "🔴 HIGH RISK"
    elif prob > 0.40:
        return "🟠 MEDIUM RISK"
    return "🟢 LOW RISK"


def retention_strategy(prob, num_products, is_active, balance, age):
    strategies = []

    if prob > 0.70:
        if num_products == 1:
            strategies.append("Cross-sell another product like savings or insurance.")
        if not is_active:
            strategies.append("Launch reactivation campaign with cashback offer.")
        if balance > 80000:
            strategies.append("Assign dedicated relationship manager.")
        if age > 50:
            strategies.append("Offer premium support and loyalty fee waiver.")
        if not strategies:
            strategies.append("Immediate outreach with loyalty rewards.")

    elif prob > 0.40:
        strategies = [
            "Targeted engagement email campaign.",
            "Savings interest rate boost offer.",
            "Quarterly relationship review call."
        ]

    else:
        strategies = [
            "Maintain loyalty through newsletters.",
            "Enroll customer into referral program."
        ]

    return strategies


def plot_probability_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    categories = ["Stay", "Churn"]
    values = [1 - prob, prob]

    colors = ["#2ecc71", "#e74c3c"]

    ax.barh(categories, values)
    ax.set_xlim(0, 1)
    ax.set_title("Churn Probability")
    ax.set_xlabel("Probability")

    plt.tight_layout()
    return fig


def preprocess_single_input(
    credit_score,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active,
    estimated_salary,
    geography
):
    gender_enc = 1 if gender == "Male" else 0
    has_cc_enc = 1 if has_cr_card == "Yes" else 0
    active_enc = 1 if is_active == "Yes" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    input_data = np.array([[
        credit_score,
        gender_enc,
        age,
        tenure,
        balance,
        num_products,
        has_cc_enc,
        active_enc,
        estimated_salary,
        geo_germany,
        geo_spain
    ]])

    input_scaled = scaler.transform(input_data)

    return input_scaled, active_enc


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("Customer Profile")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

age = st.sidebar.slider(
    "Age",
    18,
    92,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=80000.0
)

num_products = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    2
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    ["Yes", "No"]
)

is_active = st.sidebar.selectbox(
    "Is Active Member",
    ["Yes", "No"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

predict_button = st.sidebar.button("Predict Churn")

# ------------------------------------------------------------
# MAIN PAGE
# ------------------------------------------------------------

st.title("📊 Customer Churn Intelligence Dashboard")
st.write("XGBoost + SMOTE + Threshold Optimization + Streamlit Cloud Safe Version")

tab1, tab2 = st.tabs([
    "Single Prediction",
    "Bulk CSV Analysis"
])

# ============================================================
# TAB 1
# ============================================================

with tab1:
    if predict_button:
        input_scaled, active_enc = preprocess_single_input(
            credit_score,
            gender,
            age,
            tenure,
            balance,
            num_products,
            has_cr_card,
            is_active,
            estimated_salary,
            geography
        )

        probability = float(model.predict_proba(input_scaled)[0][1])
        prediction = "Likely to Churn" if probability >= THRESHOLD else "Likely to Stay"
        risk = risk_label(probability)

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", prediction)
        col2.metric("Churn Probability", f"{probability:.2%}")
        col3.metric("Risk Level", risk)

        st.pyplot(plot_probability_gauge(probability))

        st.subheader("Recommended Retention Strategy")

        strategies = retention_strategy(
            probability,
            num_products,
            active_enc,
            balance,
            age
        )

        for i, strategy in enumerate(strategies, 1):
            st.write(f"{i}. {strategy}")

        st.subheader("Customer Snapshot")

        snapshot = pd.DataFrame({
            "Feature": [
                "Credit Score",
                "Age",
                "Tenure",
                "Balance",
                "Products",
                "Active Member",
                "Geography"
            ],
            "Value": [
                credit_score,
                age,
                tenure,
                balance,
                num_products,
                is_active,
                geography
            ]
        })

        st.dataframe(snapshot, use_container_width=True)

    else:
        st.info("Fill customer details from sidebar and click Predict Churn.")

# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.subheader("Bulk Customer Analysis")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success(f"{len(df)} customers loaded successfully")

        df_work = df.copy()

        drop_cols = [
            "RowNumber",
            "CustomerId",
            "Surname",
            "Exited"
        ]

        for col in drop_cols:
            if col in df_work.columns:
                df_work.drop(columns=col, inplace=True)

        if "Gender" in df_work.columns:
            df_work["Gender"] = df_work["Gender"].map({
                "Male": 1,
                "Female": 0
            })

        if "Geography" in df_work.columns:
            df_work = pd.get_dummies(
                df_work,
                columns=["Geography"],
                drop_first=True
            )

        required_cols = [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_Germany",
            "Geography_Spain"
        ]

        for col in required_cols:
            if col not in df_work.columns:
                df_work[col] = 0

        df_work = df_work[required_cols]

        X_scaled = scaler.transform(df_work)

        probs = model.predict_proba(X_scaled)[:, 1]

        df["Churn Probability %"] = (probs * 100).round(2)
        df["Prediction"] = np.where(
            probs >= THRESHOLD,
            "Churn",
            "Stay"
        )

        df["Risk Level"] = [
            risk_label(p) for p in probs
        ]

        df = df.sort_values(
            by="Churn Probability %",
            ascending=False
        )

        st.subheader("Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Customers",
            len(df)
        )

        c2.metric(
            "Predicted Churn",
            (df["Prediction"] == "Churn").sum()
        )

        c3.metric(
            "Average Churn Probability",
            f"{probs.mean():.2%}"
        )

        st.subheader("Prediction Results")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )