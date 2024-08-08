#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re


# In[22]:


# Load the model and scalers
model = joblib.load('Pretrained_Model_and_Apps/rf.joblib')
age_scaler = joblib.load('Pretrained_Model_and_Apps/age_scaler.joblib')
weight_scaler = joblib.load('Pretrained_Model_and_Apps/weight_scaler.joblib')
price_scaler = joblib.load('Pretrained_Model_and_Apps/price_scaler.joblib')


# In[23]:


# Example list of NDC9 codes used in the model
ndc9_codes = joblib.load('Pretrained_Model_and_Apps/ndc9_codes.joblib')


# In[24]:


# Create a mapping from NDC9 code to index
ndc9_code_to_index = {code: idx for idx, code in enumerate(ndc9_codes)}


# In[25]:


# Function to clean NDC9 code
def clean_ndc9_code(ndc9_code):
    return re.sub('-', '0', ndc9_code)

# Define the one-hot encoding function
def one_hot_encode_ndc9(ndc9_code, ndc9_code_to_index, num_codes):
    one_hot = np.zeros(num_codes)
    if ndc9_code in ndc9_code_to_index:
        one_hot[ndc9_code_to_index[ndc9_code]] = 1
    return one_hot

# Define the one-hot encoding function for report_source
def one_hot_encode_report_source(report_source, categories):
    one_hot = np.zeros(len(categories))
    if report_source in categories:
        one_hot[categories.index(report_source)] = 1
    return one_hot

# Define the categories for report_source
categories_report_source = ['Physician', 'Pharmacist', 'OtherHealthcareProfessional']

# Define the one-hot encoding function for sex
def one_hot_encode_sex(sex):
    return np.array([1 if sex == 'Female' else 0])

# Define the prediction function
def predict(age, weight, sex, unit_price, report_source, ndc9_code, ndc9_code_to_index, categories_report_source, age_scaler, weight_scaler, price_scaler, model):
    # Clean the NDC9 code
    ndc9_code = clean_ndc9_code(ndc9_code)
    
    # Preprocess the inputs
    age_scaled = age_scaler.transform([[age]])[0][0]
    weight_scaled = weight_scaler.transform([[weight]])[0][0]
    price_scaled = price_scaler.transform([[unit_price]])[0][0]
    one_hot_sex = one_hot_encode_sex(sex)
    one_hot_report_source = one_hot_encode_report_source(report_source, categories_report_source)
    one_hot_ndc9 = one_hot_encode_ndc9(ndc9_code, ndc9_code_to_index, num_codes=len(ndc9_code_to_index))
    
    input_data = np.concatenate(([age_scaled, weight_scaled, price_scaled], one_hot_sex, one_hot_report_source, one_hot_ndc9))
    input_data = input_data.reshape(1, -1)  # Reshape to 2D array
    
    prediction = model.predict(input_data)
    return prediction[0]


# In[7]:


# Streamlit app interface
st.title('Adverse Drug Reaction Prediction')

age = st.number_input('Age (yr)', min_value=0, max_value=120, value=30)
weight = st.number_input('Weight (kg)', min_value=0, value=70)
sex = st.selectbox('Sex', ['Male', 'Female'])
unit_price = st.number_input('Drug Price per Unit', min_value=0.00000, value=10.00000)
report_source = st.selectbox('Reporting Authority Qualification', categories_report_source)
ndc9_code = st.text_input('Drug NDC9 Code (e.g., 12345-6789)')

if st.button('Predict'):
    prediction = predict(age, weight, sex, unit_price, report_source, ndc9_code, 
                         ndc9_code_to_index, categories_report_source, age_scaler, weight_scaler, price_scaler, model)
    if prediction == 0:
        st.success('Prediction: Nonserious')
    elif prediction == 1:
        st.warning('Prediction: Serious')
    else:
        st.error('Prediction: Death')


# In[ ]:




