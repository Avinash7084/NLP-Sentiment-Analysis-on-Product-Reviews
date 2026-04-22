import streamlit as st
import joblib

# Load the pre-trained pipeline (vectorizer + model)
model = joblib.load('sentiment_model.pkl')

# UI Design
st.set_page_config(page_title="Sentiment Analyzer")
st.title(" Product Review Classifier")
st.write("Enter a product review below to see if it's Positive or Negative.")

# User Input
user_review = st.text_area("Review Text:", placeholder="Type something like 'This product is amazing'...")

if st.button("Predict Sentiment"):
    if user_review.strip():
        # Prediction
        prediction = model.predict([user_review])[0]
        probability = model.predict_proba([user_review]).max()
        
        # Display Results
        if prediction == 1:
            st.success(f"Positive Sentiment (Confidence: {probability:.2%})")
        else:
            st.error(f"Negative Sentiment (Confidence: {probability:.2%})")
    else:
        st.warning("Please enter some text first!")