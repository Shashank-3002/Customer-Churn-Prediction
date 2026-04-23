# 📊 Customer Churn Prediction using Deep Learning (ANN)

An end-to-end **Customer Churn Prediction System** built using an Artificial Neural Network (ANN), featuring **class imbalance handling, ROC-AUC optimization, and a Streamlit deployment** for real-time predictions.

[![Streamlit App]([https://static.streamlit.io/badges/streamlit_badge.svg)](https://ann-project-implementation-4rrbmvvkbdqkmfkukxthsd.streamlit.app/](https://customer-churn-prediction-bxzkw6uapa7svn8bet6dxh.streamlit.app/))

---

## 🔗 Live Demo
👉 **[Try the App](https://ann-project-implementation-4rrbmvvkbdqkmfkukxthsd.streamlit.app/)**

---

## 🧠 Problem Statement
Customer churn is a critical issue in the banking sector. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to **predict whether a customer will churn**, enabling proactive retention strategies.

---

## 📝 Project Overview
This project uses a **deep learning classification model** to predict churn probability based on customer attributes such as:

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Activity Status  

---

## ⚙️ Key Features

### 🔹 Deep Learning Model
- ANN architecture: **128 → 64 → 32 → 1**
- Activation: ReLU (hidden), Sigmoid (output)
- Regularization:
  - Dropout
  - Batch Normalization
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

### 🔹 Advanced Evaluation Metrics
- ROC-AUC Score  
- Precision, Recall, F1-score  
- Confusion Matrix  

> Focused on **recall optimization** to minimize missed churners.

---

### 🔹 Class Imbalance Handling
- Implemented **class weights** to ensure churn cases are properly learned.

---

### 🔹 Threshold Tuning (Business-Oriented)
- Optimized decision threshold (not default 0.5)
- Improves detection of high-risk customers

---

### 🔹 Risk Segmentation (Unique Feature)
Customers are categorized into:

- 🔴 High Risk  
- 🟠 Medium Risk  
- 🟢 Low Risk  

---

### 🔹 Streamlit Web Application
- Interactive UI for real-time predictions  
- Displays:
  - Churn probability  
  - Risk level  
  - Final decision  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend / ML:** TensorFlow / Keras  
- **Data Processing:** Pandas, NumPy  
- **Preprocessing:** Scikit-learn  
  - StandardScaler  
  - LabelEncoder  
  - ColumnTransformer (OneHot Encoding)

---

## 📂 Project Structure
├── app.py # Streamlit application
├── churn_model.h5 # Trained ANN model
├── scaler.pkl # Feature scaler
├── encoder.pkl # ColumnTransformer encoder
├── label_encoder.pkl # Gender encoder
├── requirements.txt # Dependencies
├── runtime.txt # Python version
└── README.md # Documentation

---

## 📊 Model Performance

- Strong ROC Curve  
- Expected ROC-AUC: **~0.85–0.90**  
- Balanced precision and recall  
- Improved churn detection using threshold tuning  

---

## 💡 Key Insights

- Customers with **1 product** have higher churn risk  
- Customers with **2 products** are most stable  
- Active members are less likely to churn  
- Geography and balance significantly influence churn behavior  

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone <your-repo-link>

# Navigate into project folder
cd customer-churn-project

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

---
