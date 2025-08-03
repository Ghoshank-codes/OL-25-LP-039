import streamlit as st
import joblib
import pandas as pd
from utilities import Label_Map_Encoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score, confusion_matrix
st.set_page_config(layout='wide')

@st.cache_data
def load_data():
    data=pd.read_csv('cleaned_survey.csv')
    return data

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df=load_data()
classification_model=load_model('Light_GBM_Classification.pkl')
regression_model=load_model('SVR_regressor.pkl')


st.title("🧠 Mental Health Support Prediction")
st.subheader("Capstone Project 2025 | OpenLearn Cohort 1.0")
st.markdown("Built with machine learning to assess mental health support needs and estimate respondent age.")
st.divider()
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Prediction Type", [
    '🔍 Predict Treatment Seeking Behavior',
    '📈 Predict Estimated Age'
])

st.header("📌 Supervised Learning Overview")

st.markdown("### 🟢 **Classification Task**: Predict if someone is likely to seek mental health treatment")
st.markdown("**Input Features:**")
st.code("'Age', 'no_employees', 'facilities', 'priotize_mental_health','Gender', 'Country', 'self_employed', 'family_history', 'work_interfere', 'remote_work', 'tech_company', 'wellness_program', 'coworkers', 'supervisor', 'obs_consequence'")

st.markdown("**Evaluation Metrics:**")
st.success("""
- Accuracy: 95.22%
- F1 Score: 95.65%
- Precision: 94.96%
- Recall: 96.35%
- ROC AUC: 95.11%
""")
st.markdown("🧠 **Model Confidence:** _95.22%_")

st.markdown("### 🔵 **Regression Task**: Predict the respondent's age")
st.markdown("**Input Features:**")
st.code("'Country','no_employees','Gender','tech_company'")

st.markdown("**Evaluation Metrics:**")
st.info("""
- Mean Absolute Error: 5.29  
- Mean Squared Error: 45.92  
- RMSE: 6.78  
- R² Score: 0.05
""")
st.markdown("📉 **Model Confidence:** _5%_")

st.divider()

def treatment_prediction():
    st.header("📝 Take the Survey and Get Prediction")

    with st.form("mental_health_form"):
        col1,col2,col3=st.columns(3)
        with col1:
            age=st.number_input("Enter your Age",min_value=10,max_value=80,value=25)
            gender = st.selectbox("Gender", ['Male','Female','Other'])
            country = st.selectbox("Country of Residence",['United States', 'Canada', 'United Kingdom', 'Bulgaria', 'France',
        'Portugal', 'Netherlands', 'Switzerland', 'Poland', 'Australia',
        'Germany', 'Russia', 'Mexico', 'Brazil', 'Slovenia', 'Costa Rica',
        'Austria', 'Ireland', 'India', 'South Africa', 'Italy', 'Sweden',
        'Colombia', 'Latvia', 'Romania', 'Belgium', 'New Zealand', 'Spain',
        'Finland', 'Uruguay', 'Israel', 'Bosnia and Herzegovina',
        'Hungary', 'Singapore', 'Japan', 'Nigeria', 'Croatia', 'Norway',
        'Thailand', 'Denmark', 'Greece', 'Moldova', 'Georgia', 'China',
        'Czech Republic', 'Philippines'])
            self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
            no_employees=st.slider("Company Size",min_value=10,max_value=2500,step=10,value=50)
            remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
            tech_company = st.selectbox("Is your employer a tech company?", ["Yes", "No"])
            family_history = st.selectbox("Any family history of mental illness?", [ "No","Yes"])
        with col2:
            benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
            care_options = st.selectbox("Are you aware of care options provided by your employer?", ["Yes", "No", "Not sure"])
            wellness_program = st.selectbox("Does your employer have wellness programs?", ["Yes", "No", "Don't know"])
            seek_help = st.selectbox("Does your employer encourage seeking help for mental health?", ["Yes", "No", "Don't know"])
            anonymity = st.selectbox("Do you feel your anonymity is protected if you choose to take advantage of mental health services?",["Yes", "No", "Don't know"])
            leave = st.selectbox(
            "How easy is it to take medical leave for a mental health condition?",['Somewhat easy', "Don't know", 'Somewhat difficult','Very difficult', 'Very easy'])
            mental_health_consequence = st.selectbox("Do you think discussing a mental health issue with your employer could have negative consequences?",["Yes", "No", "Maybe"])
            phys_health_consequence = st.selectbox("Do you think discussing a physical health issue with your employer could have negative consequences?",["Yes", "No", "Maybe"])
        with col3:
            coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?",["Yes", "No", "Some of them"])
            supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your supervisor(s)?",["Yes", "No", "Some of them"])
            mental_health_interview = st.selectbox("Would you bring up a mental health issue in a job interview?",["Yes", "No", "Maybe"])
            phys_health_interview = st.selectbox("Would you bring up a physical health issue in a job interview?",["Yes", "No", "Maybe"])
            mental_vs_physical = st.selectbox("Do you feel that mental health is treated as seriously as physical health in your workplace?",["Yes", "No", "Don't know"])
            obs_consequence = st.selectbox("Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",["Yes", "No"])
            work_interfere = st.selectbox("How does your mental health interfere with your work?",['Often', 'Rarely', 'Never', 'Sometimes'])
        
        submit = st.form_submit_button("🔍 Submit and Predict")
    if submit:
        data = {
                'Age': [age],
                'Gender': [gender],
                'Country': [country],
                'self_employed': [self_employed],
                'no_employees': [no_employees],
                'remote_work': [remote_work],
                'tech_company': [tech_company],
                'family_history': [family_history],
                'benefits': [benefits],
                'care_options': [care_options],
                'wellness_program': [wellness_program],
                'seek_help': [seek_help],
                'anonymity': [anonymity],
                'leave': [leave],
                'mental_health_consequence': [mental_health_consequence],
                'phys_health_consequence': [phys_health_consequence],
                'coworkers': [coworkers],
                'supervisor': [supervisor],
                'mental_health_interview': [mental_health_interview],
                'phys_health_interview': [phys_health_interview],
                'mental_vs_physical': [mental_vs_physical],
                'obs_consequence': [obs_consequence],
                'work_interfere': [work_interfere]
            }
        survey_df = pd.DataFrame(data)
        def facilties(row):
            '''
            This function will be used to create the new feature 'facilities'
            '''
            label=0

            # benifits
            if row['benefits'] == 'Yes':
                label += 1
            elif row['benefits'] == "No":
                label += -1
            
            # care_options
            if row['care_options'] == 'Yes':
                label += 1
            elif row['care_options'] == "No":
                label += -1
            
            # seek_help
            if row['seek_help'] == 'Yes':
                label += 1
            elif row['seek_help'] == "No":
                label += -1
            
            # leave
            if row['leave'] == 'Very easy':
                label += 1
            elif row['leave'] == 'Somewhat easy':
                label += 0.5
            elif row['leave'] == "Don't know":
                label += 0
            elif row['leave'] == 'Somewhat difficult':
                label += -0.5
            elif row['leave'] == 'Very difficult':
                label += -1
            
            # anonymity
            if row['anonymity'] == 'Yes':
                label += 1
            elif row['anonymity'] == "No":
                label += -1

            # mental_vs_physical
            if row['mental_vs_physical'] == 'Yes':
                label += 1
            elif row['mental_vs_physical'] == "No":
                label += -1
            
            return label
        survey_df["facilities"]=survey_df.apply(facilties, axis=1)
        survey_df.drop(columns=['benefits', 'care_options', 'seek_help', 'leave',"anonymity","mental_vs_physical"], inplace=True,axis=1)
        def priotize_mental_health(row):
            '''
            This function will be used to create the new feature 'priortize mental health'
            '''

            label=0

            # mental_health_consequence
            if row['mental_health_consequence'] == 'Yes':
                label += 1
            elif row['mental_health_consequence'] == "No":
                label += -1
            
            # phys_health_consequence
            if row['phys_health_consequence'] == 'Yes':
                label += -1
            elif row['phys_health_consequence'] == "No":
                label += 1
            
            # mental_health_interview
            if row['mental_health_interview'] == 'Yes':
                label += 1
            elif row['mental_health_interview'] == "No":
                label += -1
            
            # phys_health_interview
            if row['phys_health_interview'] == 'Yes':
                label += -1
            elif row['phys_health_interview'] == "No":
                label += 1
            
            return label
        survey_df["priotize_mental_health"]=survey_df.apply(priotize_mental_health, axis=1)
        survey_df.drop(columns=['mental_health_consequence', 'phys_health_consequence', 'mental_health_interview','phys_health_interview'], inplace=True)
        result=classification_model.predict(survey_df)
        if result == 1:
            st.success("✅ Prediction: The respondent is **likely to seek mental health treatment.**")
        else:
            st.warning("❌ Prediction: The respondent is **unlikely to seek mental health treatment.**")
    else:
        st.info("Please fill and submit the survey to see predictions.")

def age_prediction():
    st.header("🔢 Age Prediction (Regression Model)")

    with st.form('age_prediction_form'):
        gender = st.selectbox("Gender", ['Male','Female','Other'])
        country = st.selectbox("Country of Residence",['United States', 'Canada', 'United Kingdom', 'Bulgaria', 'France',
            'Portugal', 'Netherlands', 'Switzerland', 'Poland', 'Australia',
            'Germany', 'Russia', 'Mexico', 'Brazil', 'Slovenia', 'Costa Rica',
            'Austria', 'Ireland', 'India', 'South Africa', 'Italy', 'Sweden',
            'Colombia', 'Latvia', 'Romania', 'Belgium', 'New Zealand', 'Spain',
            'Finland', 'Uruguay', 'Israel', 'Bosnia and Herzegovina',
            'Hungary', 'Singapore', 'Japan', 'Nigeria', 'Croatia', 'Norway',
            'Thailand', 'Denmark', 'Greece', 'Moldova', 'Georgia', 'China',
            'Czech Republic', 'Philippines'])
        tech_company = st.selectbox("Is your employer a tech company?", ["Yes", "No"])
        no_employees=st.slider("Company Size",min_value=10,max_value=2500,step=10,value=50)
        submit = st.form_submit_button("📈 Predict Age")

    if submit:
        data={
            'Gender': [gender],
            'Country': [country],
            'tech_company': [tech_company],
            'no_employees': [no_employees]
        }
        survey_df = pd.DataFrame(data)
        result=int(regression_model.predict(survey_df)[0])
        st.success(f' Estimated Age: **{result} years**')
    else:
        st.info("Please fill and submit the survey to see predictions.")

if option == '🔍 Predict Treatment Seeking Behavior':
    treatment_prediction()
elif option == '📈 Predict Estimated Age':
    age_prediction()




