from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Zynxis Intern Performance Predictor", layout="wide")

st.title("Zynxis Intern Performance Predictor")
st.caption("Estimate how an intern is likely to perform during their internship program.")

model = load_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("Internship domain", ["AI/ML", "Web Development", "Data Science", "Product", "UX Research", "Marketing"])
        education_level = st.selectbox("Education level", ["Bachelor's", "Master's", "Diploma"])
        prior_experience = st.selectbox("Prior experience", [0, 1])
        coding_score = st.slider("Coding score", 0, 100, 75)
        communication_score = st.slider("Communication score", 0, 100, 72)
        teamwork_score = st.slider("Teamwork score", 0, 100, 74)

    with col2:
        discipline_score = st.slider("Discipline score", 0, 100, 80)
        attendance_pct = st.slider("Attendance (%)", 0, 100, 88)
        hours_logged = st.slider("Hours logged per week", 20, 100, 45)
        project_quality = st.slider("Project quality", 0, 100, 76)
        mentor_rating = st.slider("Mentor rating", 0, 100, 78)

    submitted = st.form_submit_button("Predict performance")

if submitted:
    features = pd.DataFrame(
        [{
            "domain": domain,
            "education_level": education_level,
            "coding_score": coding_score,
            "communication_score": communication_score,
            "teamwork_score": teamwork_score,
            "discipline_score": discipline_score,
            "attendance_pct": attendance_pct,
            "hours_logged": hours_logged,
            "project_quality": project_quality,
            "prior_experience": prior_experience,
            "mentor_rating": mentor_rating,
        }]
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_order = model.classes_
    confidence = max(probabilities)
    confidence_index = list(probabilities).index(confidence)

    st.subheader("Prediction")
    st.success(f"Predicted performance: {prediction}")
    st.write(f"Confidence: {confidence:.0%} ({class_order[confidence_index]})")

    st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=class_order))

    if prediction == "Excellent":
        st.info("This intern is very likely to perform strongly and may need advanced mentorship or challenging tasks.")
    elif prediction == "Good":
        st.info("This intern is on track and should continue with structured growth opportunities.")
    else:
        st.warning("This intern may need additional support in technical skills, communication, or consistency.")

    st.caption("Model output is advisory and should be used alongside manager review.")
