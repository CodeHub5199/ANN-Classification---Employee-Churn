import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

# Input fields
creditScore = st.number_input('Enter Credit Score', min_value=300, max_value=850, value=650)
geography = st.selectbox('Select Country', options=label_encoder_geo.categories_[0])
gender = st.selectbox('Select Gender', options=label_encoder_gender.classes_)
age = st.number_input('Enter Age', min_value=18, max_value=100, value=30)
tenure = st.slider('Select Tenure (years)', 0, 10, 1)
balance = st.number_input('Enter Account Balance', min_value=0, value=50000)
numOfProducts = st.selectbox('Number of Products', options=[1, 2, 3, 4])
hasCrCard = st.radio('Has Credit Card?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
isActiveMember = st.radio('Is Active Member?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimatedSalary = st.number_input('Enter Estimated Salary', min_value=0, value=100000)

# Prepare input_data dictionary
input_data = {
    'CreditScore': creditScore,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': numOfProducts,
    'HasCrCard': hasCrCard,
    'IsActiveMember': isActiveMember,
    'EstimatedSalary': estimatedSalary
}

input_df = pd.DataFrame([input_data])
# st.write(label_encoder_geo.get_feature_names_out(['Geography']))

# One-hot encoding Geography
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Encode Categorical value
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Scale the data
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)

prediction_proba = prediction[0][0]

st.write(prediction_proba)

if prediction_proba > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
