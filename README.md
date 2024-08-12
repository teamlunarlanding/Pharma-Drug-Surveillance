# Adverse Drug Reaction Surveillance System

# Project Background
Almost half of Americans take prescription pharmaceutical drugs every month. Side effect profiles and warnings issued by drug manufacturers are limited by clinical trial data, and therefore lack variance for individual differences. To track the epidemiological impacts of drugs, the Food and Drug Administration (FDA) created the Adverse Event Reporting System (FAERS) which receives millions of reports each year of adverse reactions to pharmaceutical drugs, underscoring the size of the public health burden. Many systems have been developed to model side effects of adverse drug reaction (ADR) outcomes based on FAERS data, but they lack interpretability of individual differences and economic impacts. 

# Method
This project created an extract-transform-load pipeline that synthesizes multiple sources of public data about pharmaceutical drugs into an SQL database (FAERS, Medicaid drug prices, RxNorm). Data is queried from the database to train, tune, and test machine learning models to classify outcomes of adverse drug reaction (ADR) events. The optimal pre-trained model (random forest) is deployed in a Streamlit application where users can assess risk of ADR outcome. The data is imported into PowerBI via PowerQuery and displayed as an interactive dashboard. The system creates user-friendly tools for ADR risk and surveillance.  


![ADR Surveillance System Architecture](/ImageLibrary/DataArchSquare.png)

# Results
The system shows novel insights that are clinically relevant and add a precision public health approach to adverse drug reaction risk. Specifically, individual differences like age, weight, and sex, and economic factors like drug prices, are significant features for classifying adverse outcomes, more so than specific pharmaceutical drugs. More specifically, younger, overweight females are at highest risk for death, compared to older, lower weight males, implying that differences in pharmacokinetics are related to outcome seriousness.

# System Outputs
* PowerBI Dashboard
![PowerBIDashboard](/ImageLibrary/dashboard.png) 

* Streamlit Application(https://pharma-drug-surveillance-atgm5xpvbuhuvolftviaun.streamlit.app/)
![Streamlit](/ImageLibrary/streamlit.png)

* Public Access to SQL Database: UN and PW

# Programming Languages
* Python
* SQL
  
# Notes on System Use
* Note 1
* Note 2

# Style
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
