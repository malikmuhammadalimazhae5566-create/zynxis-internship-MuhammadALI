import pandas as pd
import streamlit as st

from utils.predict import predict_species
from utils.preprocess import build_input_frame

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

with open("assets/style.css", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

if "sepal_length" not in st.session_state:
    st.session_state.sepal_length = 5.1
if "sepal_width" not in st.session_state:
    st.session_state.sepal_width = 3.5
if "petal_length" not in st.session_state:
    st.session_state.petal_length = 1.4
if "petal_width" not in st.session_state:
    st.session_state.petal_width = 0.2

st.title("🌸 Iris Species Predictor")
st.caption("A polished machine-learning demo that turns flower measurements into a fast, clear species prediction.")

with st.sidebar:
    st.header("Quick help")
    st.write("The model uses sepal and petal measurements to classify iris flowers.")
    if st.button("Load sample values"):
        st.session_state.sepal_length = 5.1
        st.session_state.sepal_width = 3.5
        st.session_state.petal_length = 1.4
        st.session_state.petal_width = 0.2
    if st.button("Clear form"):
        st.session_state.sepal_length = 0.0
        st.session_state.sepal_width = 0.0
        st.session_state.petal_length = 0.0
        st.session_state.petal_width = 0.0

with st.form("prediction_form"):
    st.subheader("Enter flower measurements")
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=st.session_state.sepal_length, step=0.1, key="sepal_length")
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=st.session_state.sepal_width, step=0.1, key="sepal_width")
    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=st.session_state.petal_length, step=0.1, key="petal_length")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=st.session_state.petal_width, step=0.1, key="petal_width")

    submitted = st.form_submit_button("Predict Species", use_container_width=True)

if submitted:
    features = build_input_frame(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
    )
    prediction, probabilities = predict_species(features)

    top_class, top_probability = max(probabilities.items(), key=lambda item: item[1])
    confidence_level = "high" if top_probability >= 0.8 else "medium" if top_probability >= 0.6 else "low"

    if confidence_level == "high":
        st.success(f"Prediction: **{prediction}**")
    elif confidence_level == "medium":
        st.warning(f"Prediction: **{prediction}**")
    else:
        st.info(f"Prediction: **{prediction}**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted species", prediction, help="The class with the highest model probability")
    with col_b:
        st.metric("Top confidence", f"{top_probability * 100:.1f}%", help="Model confidence for the winning class")

    species_details = {
        "setosa": {
            "description": "Setosa is a compact iris species with short, wide petals and a delicate, elegant look.",
            "traits": ["Small petals", "Short petal length", "Narrow and simple shape", "Often appears more delicate than the other two species"],
            "fun_fact": "Setosa is often considered the easiest iris to distinguish because of its very small petals.",
        },
        "versicolor": {
            "description": "Versicolor is a medium-sized iris with balanced petals and a strong, classic flower structure.",
            "traits": ["Medium petals", "Balanced sepal and petal dimensions", "Moderate overall size", "Often has a graceful but sturdy look"],
            "fun_fact": "Versicolor sits between setosa and virginica in size and shape.",
        },
        "virginica": {
            "description": "Virginica is the largest iris species, with broad petals and a bold, impressive flower form.",
            "traits": ["Large petals", "Long petal length", "Broad and prominent structure", "Usually the most visually striking species"],
            "fun_fact": "Virginica is known for its big petals and strong overall size compared with the other iris species.",
        },
    }

    detail = species_details.get(prediction, species_details["versicolor"])
    with st.expander(f"About {prediction.title()}", expanded=True):
        st.write(detail["description"])
        st.write("**Key traits:**")
        for trait in detail["traits"]:
            st.write(f"- {trait}")
        st.write(f"**Fun fact:** {detail['fun_fact']}")

    chart_df = pd.DataFrame({"Probability": probabilities.values()}, index=probabilities.keys()).sort_values("Probability", ascending=False)
    st.subheader("Confidence by class")
    st.bar_chart(chart_df)

    with st.expander("Show input values"):
        st.dataframe(features)
