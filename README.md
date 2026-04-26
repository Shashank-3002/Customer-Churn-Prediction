# 📊 Customer Churn Intelligence using Machine Learning

An end-to-end **AI-Powered Customer Churn Prediction & Retention Intelligence System** built using **XGBoost + SMOTE + SHAP Explainability + Streamlit Dashboard**, designed to help businesses identify high-risk customers and take proactive retention actions.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://customer-churn-prediction-naultpgvcxvfrr9yv3utvc.streamlit.app/)

---

## 🧠 Problem Statement

Customer churn is one of the biggest business challenges in the banking and financial sector. Retaining an existing customer is significantly more cost-effective than acquiring a new one.

This project aims to build an **intelligent churn prediction system** that can:

- Predict whether a customer is likely to churn
- Calculate churn probability
- Classify customer risk level
- Suggest retention strategies
- Support both single customer prediction and bulk customer analysis

This helps businesses take **proactive customer retention decisions** instead of reactive ones.

---

## 📝 Project Overview

This project uses an advanced **Machine Learning Classification Model (XGBoost)** trained on customer banking data to predict churn probability based on features such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

The project goes beyond basic prediction by adding:

- Business-oriented threshold optimization
- Retention strategy recommendations
- Bulk customer risk analysis
- Interactive executive dashboard

---

## ⚙️ Key Features

---

## 🔹 Machine Learning Model

### XGBoost Classifier

Used a highly optimized gradient boosting model for superior prediction performance.

### Why XGBoost?

- Better than traditional ANN for tabular banking data
- Handles non-linearity effectively
- Stronger performance on imbalanced datasets
- Faster inference for deployment
- Production-friendly and scalable

---

## 🔹 Class Imbalance Handling

### SMOTE (Synthetic Minority Oversampling Technique)

Customer churn datasets are naturally imbalanced.

To solve this:

- Applied **SMOTE**
- Balanced churn vs non-churn classes
- Improved recall for churn prediction

This helps reduce false negatives (missing actual churners).

---

## 🔹 Repository Structure

├── app.py                     # Streamlit dashboard

├── train_model.py             # Model training pipeline

├── Churn_Modelling.csv        # Dataset

├── churn_model.pkl            # Trained XGBoost model

├── scaler.pkl                 # StandardScaler

├── feature_names.pkl          # Feature names

├── threshold.pkl              # Optimized prediction threshold

├── shap_importance.png        # SHAP feature importance plot

├── requirements.txt           # Dependencies

├── runtime.txt                # Python version

└── README.md                  # Documentation

--- 

## 🚀 How to Run Locally
bash
### Clone the repository
git clone <your-repo-link>

### Navigate into project folder
cd customer-churn-project

### Install dependencies
pip install -r requirements.txt

### Run the app
streamlit run app.py

---
