import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model and vectorizer
with open("./model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("./model/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.strip()
    return text

# Streamlit app
st.title("Fake News Classifier")
st.write("Enter a news headline to classify it as Real or Fake.")

# Input from the user
headline = st.text_input("News Headline:")

# Prediction button
if st.button("Classify"):
    if headline.strip():
        if len(headline.strip()) < 5:
            st.warning("The headline is too short to classify. Please provide a longer headline.")
        else:
            
            # Preprocess and transform the headline
            processed_headline = preprocess_text(headline)
            headline_vectorized = vectorizer.transform([processed_headline])

            # Obtener la probabilidad de predicción
            if hasattr(model, "predict_proba"):
                headline_proba = model.predict_proba(headline_vectorized)[0, 1]

                # Ajustar el umbral
                threshold = 0.50
                prediction = int(headline_proba >= threshold)

            # Display the result
            if prediction == 1:
                st.success("📰 The headline is **Real**.")
            else:
                st.error("🚨 The headline is **Fake**.")
    else:
        st.warning("Please enter a valid headline.")
