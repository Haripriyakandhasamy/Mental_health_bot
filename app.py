import streamlit as st
import numpy as np
import joblib

# Load the model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit config
st.set_page_config(
    page_title="Mental Wellness AI",
    page_icon="😊",
    layout="centered"
)

st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
}

/* Main container text */
.block-container {
    padding-top: 2rem;
    color: #000000;  /* Make all normal text black */
}

/* Titles, headers, captions */
h1, h2, h3, h4, h5, h6, .stText, .css-10trblm { 
    color: #000000 !important;  /* Force black for all headings & captions */
}

/* Buttons text color */
.stButton button {
    color: #000000 !important;   /* Button text black */
    background-color: #bfdbfe !important; /* Optional: slightly darker blue button */
}

/* Text area input text */
textarea, input {
    color: #000000 !important;  /* Input text black */
}

/* Streamlit messages (success, warning, error, info) */
.stAlert {
    color: #000000 !important;
}

/* Links */
a {
    color: #1d4ed8 !important;  /* Dark blue links */
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
