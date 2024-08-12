# Adverse Drug Reaction Surveillance System

# Project Background
Almost half of Americans take prescription pharmaceutical drugs every month. Side effect profiles and warnings issued by drug manufacturers are limited by clinical trial data, and therefore lack variance for individual differences. To track the epidemiological impacts of drugs, the Food and Drug Administration (FDA) created the Adverse Event Reporting System (FAERS) which receives millions of reports each year of adverse reactions to pharmaceutical drugs, underscoring the size of the public health burden. Many systems have been developed to model side effects and adverse drug reactions based on FAERS data, but they lack interpretability of individual differences and economic impacts. 

# Method
Here, we document the method of an extract-transform-load pipeline that synthesizes multiple sources of public data about pharmaceutical drugs into a SQL database (FAERS, Medicaid drug prices, RxNorm). Data is queried from the database to train, tune, and test machine learning models to classify outcomes of adverse drug events. The optimal pre-trained model is imported into PowerBI via PowerQuery and displayed as an interactive dashboard, creating a user-friendly end-product of drug-related data. 

Add Photo of Architecture here

# Results
Our system shows novel insights that are clinically relevant and add a precision public health approach to adverse event risk. Specifically, individual differences like age, weight, and sex, and economic factors like drug prices, are significant features for classifying adverse outcomes in relation to specific pharmaceutical drugs. EXPLAIN MODEL RESULTS IN ONE OR TWO SENTENCE.

# System Outputs
* PowerBI Dashboard
* Streamlit Dashboard
* Public Access to SQL Database: UN and PW

# Programming Languages
* Python
* SQL
  
# Notes on System Use
* Note 1
* Note 2

# Style (black) add banner and run lint to clean up code
