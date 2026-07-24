import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer

# Render (and most non-Heroku hosts) don't auto-install NLTK data from
# nltk.txt the way Heroku's buildpack does, so download it here instead.
# quiet=True keeps this from spamming the Streamlit logs on every restart;
# NLTK skips the download automatically if the data is already present.
for _resource in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(_resource, quiet=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]
    # Skip single-character tokens (punctuation remnants, single letters)
    tokens = [ps.stem(t) for t in tokens if len(t) > 1]
    return " ".join(tokens)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
