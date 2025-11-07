import streamlit as st
import requests

st.title("Plagiarism Detection Tool")

option = st.selectbox("Input Type", ["Text", "PDF"])
methods = st.multiselect("Detection Methods", ["string_matching", "citation_analysis", "stylometry", "semantic_analysis", "cross_lingual", "metadata_analysis"])
sensitivity = st.slider("Sensitivity (%)", 60, 100, 80)

if option == "Text":
    text = st.text_area("Paste your text here")
    if st.button('Run Analysis'):
        response = requests.post("http://localhost:8000/analyze/text", json={
            "text": text,
            "methods": methods,
            "sensitivity": sensitivity
        })
        st.write(response.json())
else:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button('Run Analysis'):
        files = {'file': uploaded_file.getvalue()}
        response = requests.post(f"http://localhost:8000/analyze/pdf", files=files)
        st.write(response.json())
