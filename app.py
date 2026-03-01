import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide", page_icon="👥")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background gradient */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* KPI Cards */
    .kpi-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .kpi-label { color: #a0aec0; font-size: 13px; letter-spacing: 1px; text-transform: uppercase; }
    .kpi-value { color: #ffffff; font-size: 38px; font-weight: 700; margin: 6px 0; }
    .kpi-sub   { color: #68d391; font-size: 13px; }

    /* Title */
    h1 { background: linear-gradient(90deg, #667eea, #764ba2);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-size: 2.5rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #a0aec0; font-size: 15px; }
    .stTabs [aria-selected="true"] { color: #667eea !important; border-bottom: 2px solid #667eea !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        padding: 12px 32px; font-size: 16px; font-weight: 600;
        width: 100%; margin-top: 10px;
    }
    .stButton > button:hover { opacity: 0.85; transform: scale(1.02); }

    /* Sliders & Selects */
    .stSlider > div > div > div > div { background: #667eea; }
    label { color: #cbd5e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data & Model ────────────────────────────────────────────────────────
df    = pd.read_csv('data/hr_data.csv')
model = joblib.load('models/attrition_model.pkl')

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 👥 HR Attrition Analytics")
st.markdown("*Predict employee turnover risk using Machine Learning*")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊  Overview", "🔮  Predict Attrition"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Employees</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">↑ Across all departments</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        rate = df['Attrition'].mean() * 100
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Attrition Rate</div>
            <div class="kpi-value">{rate:.1f}%</div>
            <div class="kpi-sub" style="color:#fc8181;">⚠ Industry avg: 15%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        at_risk = int(df['Attrition'].sum())
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">At-Risk Employees</div>
            <div class="kpi-value">{at_risk}</div>
            <div class="kpi-sub" style="color:#fc8181;">Immediate attention needed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_a, col_b = st.columns(2)

    with col_a:
        dept = df.groupby('Department')['Attrition'].mean() * 100
        dept = dept.reset_index()
        dept.columns = ['Department', 'Attrition %']
        fig1 = px.bar(dept, x='Department', y='Attrition %',
                      title="📊 Attrition % by Department",
                      color='Attrition %', color_continuous_scale='Purpor',
                      template='plotly_dark')
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           title_font_color='white')
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        ot = df.groupby('OverTime')['Attrition'].mean() * 100
        fig2 = go.Figure(go.Bar(
            x=ot.index, y=ot.values,
            marker_color=['#667eea', '#e53e3e'],
            text=[f"{v:.1f}%" for v in ot.values], textposition='outside'
        ))
        fig2.update_layout(title="⏰ Overtime vs Attrition",
                           template='plotly_dark',
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           title_font_color='white', yaxis_title='Attrition %')
        st.plotly_chart(fig2, use_container_width=True)

    # Salary vs Attrition scatter
    fig3 = px.box(df, x='Attrition', y='MonthlyIncome',
                  color='Attrition', title="💰 Salary Distribution: Stays vs Leaves",
                  color_discrete_map={0: '#667eea', 1: '#e53e3e'},
                  labels={'Attrition': 'Left Company (1=Yes)', 'MonthlyIncome': 'Monthly Income (₹)'},
                  template='plotly_dark')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       title_font_color='white')
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Predict Employee Attrition Risk")
    st.markdown("Fill in the employee details below and let the model decide.")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age         = st.slider("🎂 Age", 22, 60, 30)
        income      = st.slider("💰 Monthly Income (₹)", 20000, 100000, 50000, step=1000)
        satisfaction= st.slider("😊 Job Satisfaction (1=Low → 4=High)", 1, 4, 3)
    with col2:
        department  = st.selectbox("🏢 Department", ['Sales','Engineering','HR','Finance','Marketing','Operations'])
        job_role    = st.selectbox("💼 Job Role", ['Manager','Senior Executive','Junior Executive','Analyst','Intern'])
        overtime    = st.selectbox("⏰ Works OverTime?", ['Yes', 'No'])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Attrition Risk"):
        dept_map = {'Finance':0,'Engineering':1,'HR':2,'Marketing':3,'Operations':4,'Sales':5}
        role_map = {'Analyst':0,'Intern':1,'Junior Executive':2,'Manager':3,'Senior Executive':4}
        ot_map   = {'No':0, 'Yes':1}

        input_data = pd.DataFrame([{
            'Age': age,
            'Department': dept_map[department],
            'Job role': role_map[job_role],
            'MonthlyIncome': income,
            'JobSatisfaction': satisfaction,
            'OverTime': ot_map[overtime]
        }])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100

        if prediction == 1:
            st.error(f"⚠️ **HIGH ATTRITION RISK** — {prob:.0f}% probability of leaving")
            st.markdown("""
            **Recommended Actions:**
            - 💸 Review compensation package
            - 🕐 Reduce overtime load
            - 🎯 Offer career growth opportunities
            - 💬 Schedule 1-on-1 feedback session
            """)
        else:
            st.success(f"✅ **LOW ATTRITION RISK** — {prob:.0f}% probability of leaving")
            st.markdown("This employee is likely to stay. Keep up the good work! 🎉")
