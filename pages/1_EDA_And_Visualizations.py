import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')

@st.cache_data
def load_data():
    data=pd.read_csv('cleaned_survey.csv')
    return data
df=load_data()


st.title("🕵️ Explorartory Data Analysis and Visualiztions")
st.subheader("Capstone Project 2025 | OpenLearn Cohort 1.0")
st.divider()

st.header("📊 Dataset Overview: OSMI Mental Health in Tech Survey")
st.markdown("""
The dataset used in this project is sourced from the **Open Sourcing Mental Illness (OSMI)** initiative. It contains responses from people working in the tech industry regarding their experiences, attitudes, and treatment around mental health.

### 🗂️ Dataset Source
- **Platform:** [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Collected by:** [Open Sourcing Mental Illness (OSMI)](https://osmihelp.org/)
- **Purpose:** To raise awareness and improve mental health support in the tech workplace.

---
""")
st.subheader("🧾 Basic Information")
st.write(f"**Shape of the cleaned dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
st.write("**Sample Records:**")
st.dataframe(df.head(5), use_container_width=True)
st.subheader("📝 Key Features")
st.markdown("""
- `Age` – Respondent’s age
- `Gender` – Self-reported gender
- `Country` – Country of residence
- `self_employed` – Whether the respondent is self-employed
- `family_history` – Family history of mental illness
- `treatment` – Whether the respondent has sought treatment for mental health
- `work_interfere` – How mental health affects work
- `no_employees` – Company size
- `remote_work` – Whether the respondent works remotely
- `tech_company` – Whether the employer is a tech company
- `benefits`, `care_options`, `wellness_program`, `seek_help` – Workplace support indicators
""")
st.subheader("📌 Notes")
st.markdown("""
- The dataset originally contained several **missing values** and **inconsistent entries** (e.g., in the `Gender` column: Male, male, M, etc.).
- Multiple **categorical features** had unclear or inconsistent formatting.
- **Data cleaning** was necessary to handle these issues before analysis.

👉 **The glimpse of the dataset shown above represents the cleaned version**, which does not contain missing or unusual values.

🔍 For accurate insights, all preprocessing steps were done before moving to EDA and modeling.
""")

st.success("This dataset forms the foundation of our analysis to understand the mental health landscape in tech.")
st.divider()

st.title("📊 Mental Wellness Visual Insights")

sns.set_theme(style="whitegrid", palette="Set2")

st.subheader("Gender vs Mental Health Support & Treatment")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, y="Gender", hue="seek_help", ax=ax,palette="muted")
    ax.set_title("Help Provided by Employer")
    ax.set_xlabel("")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, y="Gender", hue="treatment", ax=ax)
    ax.set_title("Treatment Taken by Gender")
    ax.set_xlabel("")
    st.pyplot(fig)
with st.expander("See Insights💡"):
    st.markdown('''
**Less Attention Given by Employers**  
Mental health services are **not available** to employees in **most organizations**.

**Support Capability Missing**  
A significant number of employees **do not recognize** these services due to **poor communication**.

**Communication Disconnect**  
This **gap between availability and awareness** results in **poor mental health outcomes**, as many employees **do not receive essential support**.

**Gender-Based Observations** 
- **Women** and **non-binary employees** are generally **more proactive** in seeking mental health assistance.  
- A **considerable number of male employees** do **not seek help**, indicating a **prevalent stigma or negative attitude** toward mental health.

''')


col1,col2=st.columns(2)
with col1:
    st.subheader("Benefits vs Care Options")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(df["benefits"], df["care_options"], margins=True),
                annot=True, fmt='d', cmap='YlOrBr', ax=ax)
    ax.set_xlabel("Care Options")
    ax.set_ylabel("Benefits")
    st.pyplot(fig)
with col2:
    st.subheader("Work Interference vs Treatment")
    fig, ax= plt.subplots()
    sns.heatmap(pd.crosstab(df["work_interfere"], df["treatment"], margins=True),
                annot=True, fmt='d', cmap='Purples', ax=ax)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Work Interference")
    st.pyplot(fig)

with st.expander("See Insights💡"):
    st.markdown('''
- **Smaller Firms Lack Mental Health Support:**  
  Small companies often **do not offer mental health services**. This is usually due to **fewer employees** and **limited resources**.

- **Larger Organizations Provide Support, But Awareness is Low:**  
  Companies with **over 500 employees** typically **do have mental health provisions**, but a large portion of staff is **unaware** of these services.

- **Treatment Rates Remain Consistent:**  
  The **percentage of employees receiving treatment** stays **fairly consistent** across small and large firms.  
  Therefore, it is **inconclusive** whether company size directly impacts mental health treatment uptake.
''')

col1,col2=st.columns(2)
with col1:
    st.subheader("Age & Family History")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x="Age", hue="treatment",ax=ax, fill=True, palette="Set2")
    plt.title("Age Distribution by Treatment")
    plt.xlabel("Age")
    st.pyplot(fig)

with col2:
    st.subheader("Remote Work vs Wellness Program")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(df["remote_work"], df["wellness_program"], margins=True),
            annot=True, fmt='d', cmap='Oranges', ax=ax)
    ax.set_xlabel("Wellness Program")
    ax.set_ylabel("Remote Work")
    st.pyplot(fig)

with st.expander("See Insights💡"):
    st.markdown('''
- **Lack of Awareness of Health Benefits:**  
  Health benefits are available in many organizations; however, employees are often **unaware** of these benefits.

- **Mental Health Care Options Unknown:**  
  The same applies to **mental health care services**—employees often **don’t know** they exist.  
  It is the **organization’s responsibility** to ensure that employees are properly informed.

- **Supportive Organizations Provide Leave More Easily:**  
  Companies that offer mental health care options are also **more likely to grant leave** easily.

- **Remote Workers Often Left in the Dark:**  
  Employees working remotely often **do not know** if mental wellness programs exist or if they are being offered at all.

- **Non-Remote Workers Also Lack Support:**  
  Even employees working on-site **seldom receive** proper support for mental wellness from their organizations.
''')





