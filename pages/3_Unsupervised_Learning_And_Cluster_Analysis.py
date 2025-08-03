import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
st.set_page_config(layout='wide')
@st.cache_data
def load_data():
    data=pd.read_csv('cleaned_survey.csv')
    return data

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df=load_data()
unsupervised_model=load_model('KMeans_clustering.pkl')
tech_unsupervised_model=load_model('Tech_only_cluster.pkl')

def mental_support(row):
    '''
    This function is to score the mental support given by company to the applicant
    '''
    score=0
    # benefits
    if row["benefits"]=='Yes':
        score+=1
    elif row["benefits"]=="Don't know":
        score+=0.5
    
    # care_options
    if row['care_options']=='Yes':
        score+=1
    elif row['care_options']=='Not sure':
        score=0.5
    
    # wellness_program
    if row["wellness_program"]=='Yes':
        score+=1
    elif row["wellness_program"]=="Don't know":
        score+=0.5

    # seek_help
    if row["seek_help"]=='Yes':
        score+=1
    elif row["seek_help"]=="Don't know":
        score+=0.5
    
    # anonymity
    if row["anonymity"]=='Yes':
        score+=1
    elif row["anonymity"]=="Don't know":
        score+=0.5
   
    # leave
    if row["leave"]=='Very easy':
        score+=2
    elif row['leave']=='Somewhat easy':
        score+=1.5
    elif row['leave']=="Don't know":
        score+=1
    elif row['leave']=='Somewhat difficult':
        score+=0.5
    
    # mental_vs_physical
    if row["mental_vs_physical"]=='Yes':
        score+=1
    elif row["mental_vs_physical"]=="Don't know":
        score+=0.5

    return score

def openness(row):
    '''
    This function is score the openness og employees to discuus their mental problems 
    '''
    score=0
    # coworkers
    if row['coworkers']=='Yes':
        score+=1
    elif row['coworkers']=='Some of them':
        score+=0.5
    
    # supervisor
    if row['supervisor']=='Yes':
        score+=1
    elif row['supervisor']=='Some of them':
        score+=0.5

    # mental_health_interview
    if row['mental_health_interview']=='Yes':
        score+=1
    elif row['mental_health_interview']=='Maybe':
        score+=0.5
    
    return score

def experience(row):
    '''
    This is the function to score the past experience an employee has on the basis of mental heath consequences
    '''
    score=0

    # family history
    if row['family_history']=='Yes':
        score+=1

    # work_interfere
    if row['work_interfere']=='Often':
        score+=1.5
    elif row['work_interfere']=='Rarely':
        score+=1
    elif row['work_interfere']=='Sometimes':
        score+=0.5
    
    # mental_health_consequence
    if row['mental_health_consequence']=='Yes':
        score+=1
    elif row['mental_health_consequence']=='Maybe':
        score+=0.5

    # obs_consequence
    if row['obs_consequence']=='Yes':
        score+=1

    # treatment
    if row['treatment']=='Yes':
        score+=1
    
    return score

def _3D_visual(data,model,mapper):
    test=pd.DataFrame()
    test["mental_support"]=data.apply(mental_support,axis=1)
    test["openness"]=data.apply(openness,axis=1)
    test["mental_exp"]=data.apply(experience,axis=1)
    test['label']=model.predict(test)
    test['label']=test['label'].map(mapper)
    fig= px.scatter_3d(
        test,
        x='mental_support',
        y='openness',
        z='mental_exp',
        color='label',
        title="3D KMeans Cluster Visualization",
        opacity=0.7,
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="Mental Health Support",
            yaxis_title="Openness to Discuss",
            zaxis_title="Mental Health Experience"
        )
    )
    fig.update_traces(marker=dict(size=5))
    return fig
    


st.title("🧠 Employee Clustering based on Mental Health Behavior")
st.subheader("Capstone Project 2025 | OpenLearn Cohort 1.0")
st.markdown("Using **Unsupervised Learning (Clustering)** to segment employees for mental health support strategies.")
st.divider()

st.sidebar.header("Choose Analysis Type")
analysis_type = st.sidebar.radio("Select Cluster Analysis:", ["All Employees", "Only Tech Employees"])

if analysis_type=="All Employees":
    data=df
    model=unsupervised_model
    mapper={ 0:"Well-Supported but Private Individuals",1:"Isolated and Inexperienced Individuals",2:"Aware and Treatment-Seeking Individuals",3:"Open but Unsupported Individuals"}
    st.plotly_chart(_3D_visual(data,model,mapper), use_container_width=True)
    st.divider()
else:
    data=df[df['tech_company']=='Yes']
    model=tech_unsupervised_model
    mapper={0:'Silent Sufferes',1:'Well-Supported and Aware',2:'Ignorant Sufferers', }
    st.plotly_chart(_3D_visual(data,model,mapper), use_container_width=True)
    st.divider()

st.header("🔍 Cluster-wise Insights")

if analysis_type == "All Employees":
    st.markdown("### **Cluster-Wise Analysis**")
    
    with st.expander("🔵 Cluster 0 – *Well-Supported but Private Individuals*"):
        st.markdown("""
        - High mental support, moderate openness and experience.
        - Majority took treatment.
        - Need expressive support system to open up further.
        """)
    
    with st.expander("🔴 Cluster 1 – *Isolated and Inexperienced Individuals*"):
        st.markdown("""
        - Low openness, support, and experience.
        - Highest count of no treatment taken.
        - Awareness and facility lacking.
        """)
        
    with st.expander("🟢 Cluster 2 – *Aware and Treatment-Seeking Individuals*"):
        st.markdown("""
        - High experience, moderate openness, low support.
        - Most proactive in seeking treatment.
        - Can be role models.
        """)
        
    with st.expander("🟡 Cluster 3 – *Open but Unsupported Individuals*"):
        st.markdown("""
        - High openness, but lack support and experience.
        - Majority haven’t taken treatment.
        - Need structured awareness and wellness programs.
        """)

    st.markdown("### **Recommendations Summary**")
    st.markdown("""
    - **Cluster 0:** Assign mental health counselors to encourage openness.
    - **Cluster 1:** Launch awareness campaigns and confidential therapy channels.
    - **Cluster 2:** Use as mental health ambassadors, improve support delivery.
    - **Cluster 3:** Conduct proactive wellness workshops and peer programs.
    """)

else:
    st.markdown("### **Cluster-Wise Analysis for Tech Employees**")
    
    with st.expander("🟢 Cluster 0 – *Silent Sufferers*"):
        st.markdown("""
        - Low support, low openness, but high experience.
        - Many took treatment despite minimal support.
        - Need emotional safety and structured support.
        """)

    with st.expander("🔵 Cluster 1 – *Well-Supported and Aware*"):
        st.markdown("""
        - High support, high openness, high awareness.
        - Most balanced and low-risk group.
        - Can become mental health ambassadors.
        """)

    with st.expander("🔴 Cluster 2 – *Ignorant Sufferers*"):
        st.markdown("""
        - Moderate support, high openness, low experience.
        - Least treatment taken.
        - High risk due to lack of awareness.
        """)

    st.markdown("### **Recommendations Summary**")
    st.markdown("""
    - **Cluster 0:** Provide resources and open discussion platforms.
    - **Cluster 1:** Encourage to lead peer awareness initiatives.
    - **Cluster 2:** Immediate awareness campaigns and counseling drives.
    """)

# === Final Insight Summary ===
st.divider()
st.subheader("📌 Key Insight Summary")

if analysis_type == "All Employees":
    st.markdown("""
    - **Mental health experience** most strongly influences treatment-seeking behavior.
    - **Low openness** reduces treatment rates; **high openness alone** doesn’t guarantee help-seeking.
    - **Support availability** helps but isn’t a must — awareness can still drive action.
    """)
else:
    st.markdown("""
    - Mental health **experience** has the strongest influence on treatment-seeking behavior in tech workers.
    - **Support and openness** both positively influence action.
    - Companies play a critical role in providing structure, encouragement, and a safe space for disclosure.
    """)
st.divider()

st.subheader("🧠 Curious which mental health support category you belong to?")

with st.form("mental_health_survey_form"):
    st.markdown("📋 **Fill out this quick survey to get personalized suggestions based on your mental health openness, support, and experience.**")
    col1, col2, col3 = st.columns(3)

    with col1:
        treatment = st.radio("Have you ever taken mental health treatment?", ["Yes", "No"])
        benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Are you aware of care options provided by your employer?", ["Yes", "No", "Not sure"])
        wellness_program = st.selectbox("Does your employer have wellness programs?", ["Yes", "No", "Don't know"])
        seek_help = st.selectbox("Does your employer encourage seeking mental health help?", ["Yes", "No", "Don't know"])

    with col2:
        anonymity = st.selectbox("Is your anonymity protected when using mental health services?", ["Yes", "No", "Don't know"])
        leave = st.selectbox("Ease of taking leave for mental health?", ['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult'])
        mental_vs_physical = st.selectbox("Is mental health treated as seriously as physical health?", ["Yes", "No", "Don't know"])
        coworkers = st.selectbox("Would you discuss a mental health issue with coworkers?", ["Yes", "No", "Some of them"])
        supervisor = st.selectbox("Would you discuss a mental health issue with your supervisor?", ["Yes", "No", "Some of them"])

    with col3:
        mental_health_interview = st.selectbox("Would you bring up a mental health issue in a job interview?", ["Yes", "No", "Maybe"])    
        family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
        mental_health_consequence = st.selectbox("Could discussing mental health with your employer have negative consequences?", ["Yes", "No", "Maybe"])
        obs_consequence = st.selectbox("Have you observed negative consequences for others at work?", ["Yes", "No"])
        work_interfere = st.selectbox("How much does mental health interfere with your work?", ['Never', 'Rarely', 'Sometimes', 'Often'])

    submitted = st.form_submit_button("🔍 Submit and See Your Result")

if submitted:
    data = {    
                'treatment':[treatment],
                'family_history': [family_history],
                'benefits': [benefits],
                'care_options': [care_options],
                'wellness_program': [wellness_program],
                'seek_help': [seek_help],
                'anonymity': [anonymity],
                'leave': [leave],
                'mental_health_consequence': [mental_health_consequence],
                'coworkers': [coworkers],
                'supervisor': [supervisor],
                'mental_health_interview': [mental_health_interview],
                'mental_vs_physical': [mental_vs_physical],
                'obs_consequence': [obs_consequence],
                'work_interfere': [work_interfere]
            }
    survey_df = pd.DataFrame(data)
    survey=pd.DataFrame()
    survey["mental_support"]=survey_df.apply(mental_support,axis=1)
    survey["openness"]=survey_df.apply(openness,axis=1)
    survey["mental_exp"]=survey_df.apply(experience,axis=1)
    result = unsupervised_model.predict(survey)[0]
    cluster_info = {
            0: {
                "title": "ℹ️ You belong to **'Well-Supported but Private Individuals'**",
                "msg": "You have access to mental health resources but may be hesitant to share your experiences. Personalized counseling and safe spaces may help build trust."
            },
            1: {
                "title": "🚫 You belong to **'Isolated and Inexperienced Individuals'**",
                "msg": "You may lack both support and awareness. Consider exploring educational resources or workshops to better understand mental health and its impact."
            },
            2: {
                "title": "✅ You belong to **'Aware and Treatment-Seeking Individuals'**",
                "msg": "You're self-aware, experienced, and proactive. You could become a mental health advocate and support others in your environment."
            },
            3: {
                "title": "⚠️ You belong to **'Open but Unsupported Individuals'**",
                "msg": "You’re willing to talk about mental health but lack the support systems to act. Advocate for institutional support or peer-led programs."
            }
        }
    cluster = cluster_info[result]
    if cluster:
        st.subheader(cluster["title"])
        st.write(cluster["msg"])
    else:
        st.error("Unexpected cluster result. Please try again.")

else:
    st.info("📌 Please fill out the survey form to discover your mental health behavior category.")


