import streamlit as st
import numpy as np
import joblib

# Load the model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit config
st.set_page_config(
    page_title="Mental Wellness AI",
    page_icon="🌿",
    layout="centered"
)

# Soft blue background
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 Mental Wellness AI")
st.caption("A calm AI space to understand your emotions.")

# User input
user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Analyze Emotion"):
    if user_input.strip() != "":
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        confidence = np.max(model.predict_proba(transformed)) * 100

        if prediction == "positive":
            st.success("😊 You seem positive. Keep shining!")
            st.write("• Continue what makes you happy.")
            st.write("• Share positivity with someone.")
            st.write("• Use this energy productively.")

        elif prediction == "negative":
            st.error("🌼 You may be feeling low.")
            st.write("• Try deep breathing for 2 minutes.")
            st.write("• Talk to someone you trust.")
            st.write("• Take a short walk.")

        else:
            st.info("🌿 You seem balanced.")
            st.write("• Maintain your routine.")
            st.write("• Stay hydrated.")
            st.write("• Do something productive.")

        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter your thoughts.")