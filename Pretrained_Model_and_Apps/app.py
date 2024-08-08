#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np


# In[4]:


# Load the model and scalers
model = joblib.load('Pretrained_Model_and_Apps/rf.joblib')
age_scaler = joblib.load('Pretrained_Model_and_Apps/age_scaler.joblib')
weight_scaler = joblib.load('Pretrained_Model_and_Apps/weight_scaler.joblib')
price_scaler = joblib.load('Pretrained_Model_and_Apps/price_scaler.joblib')


# In[16]:


# Example list of NDC9 codes used in the model
ndc9_codes = joblib.load('Pretrained_Model_and_Apps/ndc9_codes.joblib')


# In[ ]:


# Create a mapping from NDC9 code to index
ndc9_code_to_index = {code: idx for idx, code in enumerate(ndc9_codes)}


# In[6]:


# Function to clean NDC9 code
def clean_ndc9_code(ndc9_code):
    return re.sub('-', '0', ndc9_code)

# Define the one-hot encoding function
def one_hot_encode_ndc9(ndc9_code, ndc9_code_to_index, num_codes=len(ndc9_codes)):
    one_hot = np.zeros(num_codes)
    if ndc9_code in ndc9_code_to_index:
        one_hot[ndc9_code_to_index[ndc9_code]] = 1
    return one_hot

# Define the one-hot encoding function for report_source
def one_hot_encode_report_source(report_source):
    one_hot = np.zeros(3)
    if report_source == 'Physician':
        one_hot[0] = 1
    elif report_source == 'Pharmacist':
        one_hot[1] = 1
    elif report_source == 'OtherHealthcareProfessional':
        one_hot[2] = 1
    return one_hot

# Define the prediction function
def predict(age, weight, sex_2, unit_price, report_source, ndc9_code):
    # Clean the NDC9 code
    ndc9_code = clean_ndc9_code(ndc9_code)
    
    # Preprocess the inputs
    sex_2 = 1 if sex_2.lower() == 'female' else 0
    age_scaled = age_scaler.transform([[age]])[0][0]
    weight_scaled = weight_scaler.transform([[weight]])[0][0]
    price_scaled = price_scaler.transform([[drug_unit_price]])[0][0]
    one_hot_report_source = one_hot_encode_report_source(report_source)
    one_hot_ndc9 = one_hot_encode_ndc9(ndc9_code, ndc9_code_to_index)
    input_data = np.concatenate(([age_scaled, weight_scaled, sex_2, price_scaled], one_hot_report_source, one_hot_ndc9))
    input_data = input_data.reshape(1, -1)  # Reshape to 2D array
    prediction = model.predict(input_data)
    return prediction[0]


# In[7]:


# Streamlit app interface
st.title('Adverse Drug Reaction Outcome Prediction')

age = st.number_input('Age (yr)', min_value=0, max_value=120, value=30)
weight = st.number_input('Weight (kg)', min_value=0.0, value=70.0)
sex = st.selectbox('Sex', ['Male', 'Female'])
unit_price = st.number_input('Drug Price per Unit', min_value=0.0, value=10.0)
report_source = st.selectbox('Reporting Authority Qualification', ['Physician', 'Pharmacist', 'OtherHealthcareProfessional'])
ndc9_code = st.text_input('Drug NDC9 Code (e.g., 12345-6789)')

if st.button('Predict'):
    prediction = predict(age, weight, sex_2, unit_price, report_source, ndc9_code)
    if prediction == 0:
        st.success('Prediction: Non-serious')
    elif prediction == 1:
        st.warning('Prediction: Serious')
    else:
        st.error('Prediction: Death')


# In[ ]:




