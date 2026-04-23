import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn. preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

##Load the trained ANN model
model = tf.keras.models.load_model('churn_model.h5')

##Load the scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
##Load the one-hot encoder for the 'Geography' column
with open('onehot_encoder.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)
##Load the label encoder for the 'Gender' column
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

## Streamlit app
st.title("Customer Churn Prediction")

##User input fields
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember':[1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary':[estimated_salary],
})

## One-hot encode the 'Geography' column
# Use a DataFrame with the same column name as used during encoder fitting to avoid warnings.
geo_encoded = geo_encoder.transform(pd.DataFrame({'Geography': [geography]})).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

## Input dataframe without the 'Geography' column (we already excluded it)
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure the features are in the same order as used during training
expected_columns = list(scaler.feature_names_in_)
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_columns]

## Scale the input data using the loaded scaler
scaled_input = scaler.transform(input_data)

##Make prediction
prediction = model.predict(scaled_input)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_probability:.2f}.")
else:
    st.write(f"The customer is unlikely to churn with a probability of {1-prediction_probability:.2f}.")
