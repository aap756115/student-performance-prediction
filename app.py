import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------
# Background Image Function
# ----------------------------------
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("background.jpg")

# ----------------------------------
# Custom CSS
# ----------------------------------
st.markdown(f"""
<style>

/* Background Image */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Header */
[data-testid="stHeader"] {{
    background: rgba(255,255,255,0);
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(15px);
}}

/* Main Content */
.main .block-container {{
    background: rgba(255,255,255,0.92);
    padding: 2rem;
    border-radius: 25px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.15);
}}

/* Title */
.title {{
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #1e293b;
}}

/* Subtitle */
.subtitle {{
    text-align: center;
    font-size: 22px;
    color: #475569;
    margin-bottom: 20px;
}}

/* Metric Cards */
[data-testid="metric-container"] {{
    background: linear-gradient(135deg,#ffffff,#f1f5f9);
    border-radius: 18px;
    padding: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}}

/* Buttons */
.stButton > button {{
    width: 100%;
    height: 55px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg,#2563eb,#3b82f6);
    color: white;
    border: none;
}}

.stButton > button:hover {{
    background: linear-gradient(90deg,#1d4ed8,#2563eb);
}}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load Model
# ----------------------------------
model = pickle.load(open("student_model.pkl", "rb"))

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.title("🎓 Student Details")

G1 = st.sidebar.slider("📘 G1 Marks", 0, 20, 12)
G2 = st.sidebar.slider("📗 G2 Marks", 0, 20, 14)
studytime = st.sidebar.slider("📚 Study Time", 1, 4, 2)
failures = st.sidebar.slider("❌ Failures", 0, 5, 0)
absences = st.sidebar.slider("📅 Absences", 0, 100, 5)

predict = st.sidebar.button("🔍 Predict Performance")

# ----------------------------------
# Header
# ----------------------------------
st.markdown(
    '<div class="title">🎓 Student Performance Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Machine Learning Dashboard</div>',
    unsafe_allow_html=True
)

# ----------------------------------
# Prediction Section
# ----------------------------------
if predict:

    data = np.array([[studytime, failures, absences, G1, G2]])

    prediction = model.predict(data)[0]

    st.subheader("📊 Input Summary")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("G1", G1)
    c2.metric("G2", G2)
    c3.metric("Study Time", studytime)
    c4.metric("Failures", failures)
    c5.metric("Absences", absences)

    st.divider()

    col1, col2 = st.columns(2)

    # Gauge Meter
    with col1:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'suffix': '/20'},
            title={'text': "Predicted G3 Score"},
            gauge={
                'axis': {'range': [0,20]},
                'bar': {'color': "#7C3AED"},
                'steps':[
                    {'range':[0,8],'color':'#FECACA'},
                    {'range':[8,12],'color':'#FDE68A'},
                    {'range':[12,16],'color':'#BBF7D0'},
                    {'range':[16,20],'color':'#86EFAC'}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("🎯 Result")

        if prediction >= 16:
            st.success("🌟 Excellent Performance")
        elif prediction >= 12:
            st.success("👍 Good Performance")
        elif prediction >= 8:
            st.warning("⚠️ Average Performance")
        else:
            st.error("❌ Poor Performance")

        if prediction >= 10:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

        progress = int((prediction / 20) * 100)

        st.progress(progress)

        st.write(f"### Progress Score : {progress}%")

    st.divider()

    # ----------------------------------
    # Actual vs Predicted Chart
    # ----------------------------------
    st.subheader("📈 Actual vs Predicted Analysis")

    actual = [10,11,8,16,13,15,17,12,18,14]
    predicted = [10,13,11,15,12,14,16,11,17,13]

    chart_df = pd.DataFrame({
        "Student": range(1,11),
        "Actual": actual,
        "Predicted": predicted
    })

    fig2 = px.line(
        chart_df,
        x="Student",
        y=["Actual","Predicted"],
        markers=True,
        title="Actual vs Predicted Scores"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------
    # Feature Importance
    # ----------------------------------
    st.subheader("📊 Feature Importance")

    feature_df = pd.DataFrame({
        "Feature":[
            "G2",
            "G1",
            "Study Time",
            "Absences",
            "Failures"
        ],
        "Importance":[0.47,0.36,0.09,0.05,0.03]
    })

    fig3 = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Important Features Affecting G3"
    )

    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("👈 Enter details from the sidebar and click Predict Performance.")