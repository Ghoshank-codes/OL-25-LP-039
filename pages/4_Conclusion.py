import streamlit as st

st.set_page_config(page_title="Mental Wellness Project Summary", layout="wide")

st.title("🧠 Mental Wellness Analysis and Support Strategy")
st.subheader("OpenLearn Capstone Project 2025 — Cohort 1.0")

st.markdown("""
## 🎯 Objective
To understand key factors influencing mental health in the tech workforce and develop solutions using:
- **Classification**: Predict likelihood of seeking mental health treatment.
- **Regression**: Predict age from demographic and workplace attributes.
- **Clustering**: Segment employees by mental health indicators.

---
## 📊 Dataset Overview
- **Source**: OSMI (Open Sourcing Mental Illness) — Mental Health in Tech Survey
- **Attributes**:
  - Demographics (Age, Gender, Country)
  - Workplace Environment (Benefits, Leave Policies)
  - Personal History (Mental illness, Family history)
  - Attitudes (Seeking help, Supervisor support)

---
## 🧪 Exploratory Data Analysis Insights
- The majority of employers don't offer sufficient mental health assistance.
- Even when there is support, employees frequently don't know about it.
- Neglect of mental health is fuelled by a disconnect between both employers and staff members.
- Non-binary people and women are more likely to seek treatment.
- Men are less likely to ask for assistance.
- Programs for mental wellness are not well-organised in small businesses.
- Although there is little awareness, larger organisations offer assistance.
- Many people who sought treatment did so without support from their employers.
- Employees avoid using offered assistance due to comfort and stigma.
- Wellness initiatives and awareness are frequently lacking among remote workers.
- Srong correlation between family history and mental health awareness.

---
## 🔍 Cluster Analysis — All Employees
### **Cluster 0 – "Well-Supported but Private Individuals"**
- Moderate openness, high support, and moderate experience.
- Sought treatment, but may hesitate to open up.

### **Cluster 1 – "Isolated and Inexperienced Individuals"**
- Very low support and openness.
- Most do **not seek treatment**.

### **Cluster 2 – "Aware and Treatment-Seeking Individuals"**
- High experience, moderate openness, low support.
- Still seek treatment due to **awareness**.

### **Cluster 3 – "Open but Unsupported Individuals"**
- High openness, low support and experience.
- **Least treatment** despite openness.

**Key Takeaway**: **Experience with mental health issues** is the strongest predictor of treatment-seeking behavior.

---
## 🖥️ Cluster Insights – Tech Professionals
### **Cluster 0 – "Silent Sufferers"**
- High experience, low openness/support.
- Seek treatment independently.

### **Cluster 1 – "Well-Supported and Aware"**
- High support, openness, and awareness.
- Most proactive in seeking help.

### **Cluster 2 – "Ignorant Sufferers"**
- Open, low experience/awareness.
- **Highest untreated** group.

**Insight**: **Support and awareness** empower employees to take action — **openness alone is not enough**.

---
## ✅ Recommendations
### For Cluster 0:
- 1-on-1 counselor sessions to **build openness and trust**.

### For Cluster 1:
- Promote them as **mental health ambassadors** in the workplace.

### For Cluster 2:
- Run **awareness programs** and self-assessment workshops.

### For Cluster 3:
- Introduce **peer mentorship** and mental health training programs.

---
## 🔚 Final Thought
**Tech companies** must go beyond offering benefits — they must ensure employees are **aware**, **comfortable**, and **empowered** to access mental health care.
"""
)


