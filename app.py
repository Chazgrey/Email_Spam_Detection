import streamlit as st
import re
from nltk.corpus import stopwords
from joblib import load

# Load saved model and vectorizer
model = load("spam_model.joblib")
vectorizer = load("vectorizer.joblib")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.title("📧 Spam Detection App")

user_input = st.text_area("Enter an email text:")

if st.button("Predict"):
    cleaned = preprocess_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]  # probability scores

    # Display results
    st.write("### Result:", prediction)
    st.write("### Confidence Scores:")
    for label, prob in zip(model.classes_, probabilities):
        st.write(f"- {label}: {prob*100:.2f}%")