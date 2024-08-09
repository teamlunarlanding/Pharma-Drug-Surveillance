#!/usr/bin/env python
# coding: utf-8

# In[61]:


import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Suppress all warnings
warnings.filterwarnings('ignore')


# In[56]:


# Function to clean NDC9 code
def clean_ndc9_code(ndc9_code):
    if isinstance(ndc9_code, bytes):
        ndc9_code = ndc9_code.decode('utf-8')
    elif not isinstance(ndc9_code, str):
        ndc9_code = str(ndc9_code)
    return re.sub('-', '0', ndc9_code)

# Load the model, encoder, and scaler
model = joblib.load('Pretrained_Model_and_Apps/rf.joblib')
encoder = joblib.load('Pretrained_Model_and_Apps/encoder.joblib')
scaler = joblib.load('Pretrained_Model_and_Apps/scaler.joblib')

# Define the categorical and numerical columns
cats = ['sex','report_source', 'ndc9']
nums = ['weight', 'age', 'unit_price']


# Function to transform data (same as used for training)
def transform_data(data):
    # Apply one-hot encoding to categorical features
    encoded_features = encoder.transform(data[cats])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cats))
    
    # Apply standardization to numerical features
    standardized_features = scaler.transform(data[nums])
    standardized_df = pd.DataFrame(standardized_features, columns=nums)
    
    # Concatenate the standardized numerical features with the encoded categorical features
    return pd.concat([data.drop(nums + cats, axis=1).reset_index(drop=True),
                      standardized_df.reset_index(drop=True),
                      encoded_df.reset_index(drop=True)], axis=1)

# Map prediction outcome to text label
outcome_labels = {0: 'Nonserious', 1: 'Serious', 2: 'Death'}


# In[ ]:


# Streamlit app
st.title('Adverse Drug Event Outcome Prediction')

# User inputs
age = st.number_input('Age', min_value=0, max_value=120, value=30)
weight = st.number_input('Weight (kg)', min_value=0.0, max_value=200.0, value=70.0)
unit_price = st.number_input('Unit Price', min_value=0.0, max_value=1000.0, value=10.0)
sex = st.radio('Select Sex:', options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
report_source = st.selectbox(
    'Select Report Source:', 
    options=[1, 2, 3], 
    format_func=lambda x: 'Physician' if x == 1 else 'Pharmacist' if x == 2 else 'OtherHealthcareProfessional'
)
ndc9_code = st.text_input('NDC-9 Drug Code: e.g., 59212-562')

# Clean the NDC9 code
cleaned_ndc9_code = clean_ndc9_code(ndc9_code)

# Prepare initial data
input_data = pd.DataFrame({
    'weight': [weight],
    'age': [age],
    'unit_price': [unit_price],
    'sex': [sex],
    'report_source': [report_source],
    'ndc9': [cleaned_ndc9_code]
})


# Display the raw input data for debugging
st.subheader('Raw Input Data (Before Encoding)')
st.write(input_data)

# Ensure that all required categorical columns are in the DataFrame
if not all(col in input_data.columns for col in cats):
    st.error(f"Input data is missing required columns: {', '.join([col for col in cats if col not in input_data.columns])}")
else:
    # One-hot encode `sex`, `report_source`, and `ndc9`
    input_data_encoded = transform_data(input_data)

    # Display the encoded input data for debugging
    st.subheader('Encoded Input Data (After Encoding)')
    st.write(input_data_encoded)

    # Make prediction
    prediction = model.predict(input_data_encoded)[0]

    # Display the prediction result with a label
    st.subheader('Prediction')
    st.write(f'The predicted outcome is: {outcome_labels[prediction]}')

