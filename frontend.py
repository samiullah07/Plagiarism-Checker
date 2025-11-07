import streamlit as st
import time
from backend import finger_print_data

st.title("Plagiarism Detector")

# Brief instruction
st.write("Paste your text below and click 'Check for Plagiarism'. You’ll get a quick, clear analysis.")

# Text input area for user
user_input = st.text_area("Paste text here...")

# Simplified 'Check' button
if st.button("Check for Plagiarism"):
    with st.spinner("Analyzing..."):
        time.sleep(1)  # brief loading effect
        if user_input:
            analysis = finger_print_data(user_input)
            st.subheader("Analysis Result:")
            st.write(analysis)
        else:
            st.warning("Please paste some text to check.")

