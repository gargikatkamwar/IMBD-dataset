import streamlit as st
import joblib
import re
import string
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    vectorizer_path = "tfidf_vectorizer.pkl"
    model_path = "logistic_regression_model.pkl"

    if not os.path.exists(vectorizer_path):
        st.error(f"File not found: {vectorizer_path}")
        st.stop()

    if not os.path.exists(model_path):
        st.error(f"File not found: {model_path}")
        st.stop()

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    return vectorizer, model


vectorizer, model = load_model()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# UI
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review below and click **Predict** to detect its sentiment."
)

review = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was amazing!",
    height=180,
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        cleaned = clean_text(review)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None

        st.divider()

        if str(prediction).lower() == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        if confidence:
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))

        if hasattr(model, "classes_") and hasattr(model, "predict_proba"):
            st.subheader("Probability")

            for label, prob in zip(model.classes_, probabilities):
                st.write(f"**{label}** : {prob*100:.2f}%")

        with st.expander("Processed Text"):
            st.write(cleaned)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")

st.sidebar.write("""
**Machine Learning Model**

- TF-IDF Vectorizer
- Logistic Regression
- Streamlit Web App

Developed for IMDB Movie Review Sentiment Analysis.
""")
