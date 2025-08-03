import streamlit as st
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
st.set_page_config(layout='wide')
class Label_Map_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_map):
        self.encoding_map = encoding_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Apply map to each column individually
            return X.apply(lambda col: col.map(self.encoding_map).fillna(0))
        elif isinstance(X, pd.Series):
            return X.map(self.encoding_map).fillna(0).to_frame()
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")


st.title("🧠 Mental Wellness Analysis and Support Strategy")
st.subheader("Capstone Project 2025 | OpenLearn Cohort 1.0")
st.markdown("---")

st.markdown("""
### 📌 Project Overview
This project aims to uncover key factors influencing mental health issues among tech industry employees and propose data-driven strategies to promote well-being and proactive support mechanisms.
""")

st.markdown("""
### 🎯 Objectives
- **Classification Task:** Predict whether an individual is likely to seek mental health treatment.
- **Regression Task:** Predict the **age** of an individual using personal and workplace features to inform targeted interventions.
- **Unsupervised Learning Task:** Segment employees based on mental wellness indicators to enable personalized HR strategies.
""")

st.markdown("""
### 🧾 Dataset Overview
- **Source:** Mental Health in Tech Survey  
- **Collected by:** [OSMI (Open Sourcing Mental Illness)](https://osmihelp.org/)  
- **Features include:**
  - Demographic attributes (age, gender, country)
  - Workplace environment (mental health benefits, leave policies)
  - Personal mental health history (self and family)
  - Attitudes and perceptions around mental health in the workplace
""")

st.markdown("""
### 🧪 Case Study Scenario
You are working as a **Machine Learning Engineer** at *NeuronInsights Analytics*, contracted by a consortium of leading tech firms like **CodeLab**, **QuantumEdge**, and **SynapseWorks**.  
These companies are confronting a surge in employee burnout, disengagement, and turnover—fueled by unmanaged mental health challenges.

They have entrusted you with:
- Analyzing mental health survey data from over **1,500 tech professionals**
- Identifying risk patterns and drivers of mental health conditions
- Enabling data-backed, targeted HR strategies and interventions
""")

st.markdown("""
### 🔍 Key Questions Explored
- Who is most at risk of silently suffering and avoiding treatment?
- How do remote work setups, support policies, and mental health benefits impact employee well-being?
- Can we cluster employees into meaningful profiles to enable **tailored outreach** and **mental wellness programs**?
""")

st.markdown("---")
st.markdown("👨‍💻 *This project has been developed under the mentorship and guidance of the [OpenLearn Community](https://www.openlearn.org.in/). Proudly part of the OpenLearn Cohort 1.0.*")
