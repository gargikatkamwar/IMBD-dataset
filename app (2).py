import streamlit as st
import joblib
import re
import string

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🎬 IMDB Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logistic_regression_model.pkl")
    return vectorizer, model

try:
    tfidf_vectorizer, model = load_models()
except Exception as e:
    st.error(f"❌ Error loading model files:\n\n{e}")
    st.stop()

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)          # Remove HTML tags
    text = re.sub(r"\d+", "", text)            # Remove numbers
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()   # Remove extra spaces
    return text

# -----------------------------
# App Title
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review below to predict whether it expresses a **Positive** or **Negative** sentiment."
)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "✍️ Enter your movie review:",
    height=180,
    placeholder="Example: This movie was absolutely amazing! The acting was brilliant..."
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:

        cleaned_text = clean_text(user_input)

        # Transform text
        input_vector = tfidf_vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        # Confidence
        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("Prediction Result")

        if prediction.lower() == "positive":
            st.success("😊 **Positive Review**")
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            st.error("😞 **Negative Review**")
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence:.2f}%")

        # Probability Details
        st.subheader("Prediction Probabilities")

        try:
            class_labels = model.classes_

            for label, prob in zip(class_labels, probabilities):
                st.write(f"**{label.capitalize()}** : {prob*100:.2f}%")

        except Exception:
            st.write(f"Positive : {probabilities[1]*100:.2f}%")
            st.write(f"Negative : {probabilities[0]*100:.2f}%")

        # Show cleaned text
        with st.expander("📝 View Processed Text"):
            st.write(cleaned_text)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
This application predicts the sentiment of IMDB movie reviews using:

- TF-IDF Vectorization
- Logistic Regression
- Streamlit

Simply type or paste a movie review and click **Analyze Sentiment**.
"""
)
