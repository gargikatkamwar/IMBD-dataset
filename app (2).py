import streamlit as st
import joblib
import re
import string

# Load the saved TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('logistic_regression_model.pkl')

# Text cleaning function (must be the same as used during training)
def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = ''.join([char for char in text if char not in string.punctuation]) # Remove punctuation
    return text

st.title('IMDB Movie Review Sentiment Analyzer')
st.write('Enter a movie review below to predict its sentiment (positive/negative).')

# Text input
user_input = st.text_area('Enter your review here:', '')

if st.button('Analyze Sentiment'):
    if user_input:
        # Clean the input text
        cleaned_input = clean_text(user_input)

        # Transform the cleaned text using the loaded TF-IDF vectorizer
        # tfidf_vectorizer expects an iterable, so pass cleaned_input in a list
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # Make prediction
        prediction = model.predict(input_tfidf)
        prediction_proba = model.predict_proba(input_tfidf)

        st.subheader('Prediction:')
        if prediction[0] == 'positive':
            st.success(f'The review is: Positive (Confidence: {prediction_proba[0][1]*100:.2f}%)')
        else:
            st.error(f'The review is: Negative (Confidence: {prediction_proba[0][0]*100:.2f}%)')
    else:
        st.warning('Please enter a review to analyze.')
