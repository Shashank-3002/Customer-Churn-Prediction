# 📊 Customer Churn Prediction using ANN

An interactive web application that uses an **Artificial Neural Network (ANN)** to predict the likelihood of a customer leaving a bank. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://ann-project-implementation-4rrbmvvkbdqkmfkukxthsd.streamlit.app/)

---

## 🔗 Live Demo
Access the deployed application here: 
👉 **[ANN Churn Predictor](https://ann-project-implementation-4rrbmvvkbdqkmfkukxthsd.streamlit.app/)**

---

## 📝 Project Overview
Customer Churn is a major challenge for service-based industries. This project focuses on building a deep learning model to identify high-risk customers based on features such as credit score, geography, gender, age, tenure, and balance.

### Key Features:
* **Interactive Dashboard:** Users can adjust customer parameters using sliders and dropdowns.
* **Deep Learning Inference:** Uses a trained TensorFlow model to calculate churn probability.
* **Instant Results:** Displays whether a customer is "Likely to Churn" or "Likely to Stay" based on a 50% threshold.

---

## 🛠️ Tech Stack
* **Framework:** [Streamlit](https://streamlit.io/)
* **Deep Learning Library:** [TensorFlow / Keras](https://www.tensorflow.org/)
* **Data Manipulation:** Pandas & NumPy
* **Preprocessing:** Scikit-Learn (StandardScaler & LabelEncoder)

---

## 📂 Project Structure
```text
├── app.py              # Main Streamlit application
├── churn_model.h5      # Pre-trained Artificial Neural Network model
├── requirements.txt    # Python library dependencies
├── runtime.txt         # Environment configuration (Python 3.11)
└── README.md           # Project documentation
