import streamlit as st
import requests
import time

st.title("Plagiarism Detection Tool")

# Backend configuration
BACKEND_URL = "http://localhost:8000"

# Check backend status
def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

# Display backend status
if check_backend_health():
    st.success("✅ Backend is connected")
else:
    st.error("❌ Backend is not running. Please start the FastAPI server first.")
    st.info("Run this command in your terminal: `uvicorn main:app --reload`")

option = st.selectbox("Input Type", ["Text", "PDF"])
methods = st.multiselect(
    "Detection Methods", 
    ["string_matching", "citation_analysis", "stylometry", "semantic_analysis", "cross_lingual", "metadata_analysis"],
    default=["string_matching", "citation_analysis", "stylometry"]
)
sensitivity = st.slider("Sensitivity (%)", 60, 100, 80)

def analyze_with_retry(endpoint, payload=None, files=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if files:
                response = requests.post(f"{BACKEND_URL}{endpoint}", files=files, timeout=30)
            else:
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
            
            # PRINT RAW RESPONSE FOR DEBUGGING
            st.write("Raw response text:", response.text)
            st.write("Response status:", response.status_code)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"Server returned status {response.status_code}. Retrying...")
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to backend. Please make sure the server is running at {BACKEND_URL}")
            break
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            break
    return None


if option == "Text":
    text = st.text_area("Paste your text here", height=200)
    if st.button('Run Analysis') and text.strip():
        with st.spinner('Analyzing text for plagiarism...'):
            payload = {
                "text": text,
                "methods": methods,
                "sensitivity": sensitivity
            }
            
            result = analyze_with_retry("/analyze/text", payload=payload)
            
            if result:
                # st.subheader("Raw Backend JSON Result")
                # st.json(result)

                final_result = result.get("result", {})
                overall_score = final_result.get("overall_plagiarism_score", 0)
                st.metric("Overall Plagiarism Score", f"{overall_score}%")

                st.subheader("Method Breakdown")
                method_scores = final_result.get("method_breakdown", {})
                for method, score in method_scores.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{method.replace('_', ' ').title()}:**")
                    with col2:
                        st.progress(score/100, text=f"{score}%")

                
                # Display detailed findings
                if "detailed_findings" in result and result["detailed_findings"]:
                    st.subheader("Detailed Findings")
                    for finding in result["detailed_findings"]:
                        st.warning(f"**{finding.get('issue', 'Issue')}**: {finding.get('explanation', '')}")
                
                # Display recommendations
                if "recommendations" in result and result["recommendations"]:
                    st.subheader("Recommendations")
                    for rec in result["recommendations"]:
                        st.info(f"• {rec}")

else:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None and st.button('Run Analysis'):
        with st.spinner('Analyzing PDF for plagiarism...'):
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }
            result = analyze_with_retry("/analyze/pdf", files=files)
            if result:
                # st.subheader("Raw Backend JSON Result")
                # st.json(result)

                final_result = result.get("result", {})
                overall_score = final_result.get("overall_plagiarism_score", 0)
                st.metric("Overall Plagiarism Score", f"{overall_score}%")

                st.subheader("Method Breakdown")
                method_scores = final_result.get("method_breakdown", {})
                for method, score in method_scores.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{method.replace('_', ' ').title()}:**")
                    with col2:
                        st.progress(score/100, text=f"{score}%")
