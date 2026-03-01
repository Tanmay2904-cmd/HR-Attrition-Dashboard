import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
st.title("👥 HR Attrition Analytics Dashboard")

# Data aur model load karo
df = pd.read_csv('data/hr_data.csv')
model = joblib.load('models/attrition_model.pkl')

# Do tabs banao
tab1, tab2 = st.tabs(["📊 Overview", "🔮 Predict Attrition"])

# ─── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Company Attrition Overview")

    # KPI Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("Attrition Rate", f"{df['Attrition'].mean()*100:.1f}%")
    col3.metric("At Risk Employees", int(df['Attrition'].sum()))

    # Bar Chart — Department wise attrition
    dept_attrition = df.groupby('Department')['Attrition'].mean() * 100
    dept_attrition = dept_attrition.reset_index()
    dept_attrition.columns = ['Department', 'Attrition %']

    fig = px.bar(dept_attrition, x='Department', y='Attrition %',
                 title="Attrition % by Department",
                 color='Attrition %', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2: PREDICT ────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔮 Predict Employee Attrition")
    st.write("Employee ki details bharo — model predict karega ki woh jayega ya nahi!")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 22, 60, 30)
        income = st.slider("Monthly Income (₹)", 20000, 100000, 50000)
        satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)

    with col2:
        department = st.selectbox("Department", ['Sales', 'Engineering', 'HR', 'Finance', 'Marketing', 'Operations'])
        job_role = st.selectbox("Job Role", ['Manager', 'Senior Executive', 'Junior Executive', 'Analyst', 'Intern'])
        overtime = st.selectbox("OverTime", ['Yes', 'No'])

    if st.button("🔮 Predict Now"):
        dept_map = {'Finance': 0, 'Engineering': 1, 'HR': 2, 'Marketing': 3, 'Operations': 4, 'Sales': 5}
        role_map = {'Analyst': 0, 'Intern': 1, 'Junior Executive': 2, 'Manager': 3, 'Senior Executive': 4}
        ot_map   = {'No': 0, 'Yes': 1}

        input_data = pd.DataFrame([{
            'Age': age,
            'Department': dept_map[department],
            'Job role': role_map[job_role],
            'MonthlyIncome': income,
            'JobSatisfaction': satisfaction,
            'OverTime': ot_map[overtime]
        }])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ High Attrition Risk! This employee may leave.")
        else:
            st.success("✅ Low Attrition Risk! Employee likely to stay.")
