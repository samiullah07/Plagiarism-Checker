import streamlit as st
import requests
import time
import os

st.title("Plagiarism Detection Tool")

BACKEND_URL = "http://localhost:8000"

def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

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

# --- Threshold Tuning Controls ---
PLAGIARISM_THRESHOLD = st.slider("Plagiarism Score Warning Threshold (%)", 10, 80, 30)
GOOGLE_MATCH_THRESHOLD = st.slider("Google Snippet Warning Threshold (%)", 10, 80, 35)

def badge(score):
    if score >= 70:
        return "🟥"
    elif score >= 40:
        return "🟧"
    elif score >= 20:
        return "🟨"
    else:
        return "🟩"

def analyze_with_retry(endpoint, payload=None, files=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if files:
                response = requests.post(f"{BACKEND_URL}{endpoint}", files=files, timeout=30)
            else:
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
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

# --- MAIN TEXT INPUT LOGIC ---
if option == "Text":
    # Allow UI input OR env variable for SerpAPI key
    serpapi_key = st.text_input(
        "SerpAPI Key (for Google search plagiarism check)", 
        value=os.getenv("SERPAPI_KEY", ""), 
        type="password"
    )

    # --- Quick Test Buttons ---
    if 'test_text' not in st.session_state:
        st.session_state['test_text'] = ""

    colA, colB = st.columns(2)
    with colA:
        if st.button("Paste Example Plagiarized Text"):
            st.session_state['test_text'] = (
                "Artificial intelligence (AI) is intelligence demonstrated by machines, "
                "as opposed to the natural intelligence displayed by animals including humans. "
                "Leading AI textbooks define the field as the study of intelligent agents: "
                "any device that perceives its environment and takes actions that maximize its "
                "chance of achieving its goals."
            )
    with colB:
        if st.button("Paste Example Original Text"):
            st.session_state['test_text'] = (
                "Modern artificial intelligence uses data-driven methods to offer tailored solutions. "
                "Computer systems leverage vast data to make predictions, assist users, and solve real-world "
                "problems without mimicking human consciousness."
            )
    text = st.text_area("Paste your text here", height=200, value=st.session_state["test_text"])

    if st.button('Run Analysis') and text.strip():
        with st.spinner('Analyzing text for plagiarism...'):
            payload = {
                "text": text,
                "methods": methods,
                "sensitivity": sensitivity
            }
            if serpapi_key:
                payload["serpapi_key"] = serpapi_key
            
            result = analyze_with_retry("/analyze/text", payload=payload)
            
            if result:
                final_result = result.get("result", {})
                overall_score = final_result.get("overall_plagiarism_score", 0)
                st.metric("Overall Plagiarism Score", f"{overall_score}%")

                # --- Plagiarism/Warning Flags ---
                google_score = final_result.get("method_breakdown", {}).get("google_snippet", 0)
                if overall_score >= PLAGIARISM_THRESHOLD:
                    st.error(f"⚠️ Plagiarism WARNING! Overall score exceeds {PLAGIARISM_THRESHOLD}%")
                elif google_score >= GOOGLE_MATCH_THRESHOLD:
                    st.warning(f"⚠️ Possible Google Plagiarism Match! Highest snippet is {google_score}%")

                method_scores = final_result.get("method_breakdown", {})
                
                # --- Google Details Block ---
                if "google_snippet" in method_scores:
                    badge_color = badge(google_score)
                    st.subheader(f"Google Search Match {badge_color}")
                    st.write(f"Highest Google snippet similarity: {google_score}%")

                    # Display ALL top snippets
                    top_snippets = method_scores.get("google_top_snippets", [])
                    for i, match in enumerate(top_snippets):
                        st.markdown(f"**Match {i+1}: {badge(match['score'])} {match['score']}%**")
                        if match['snippet']:
                            st.info(f"Matched Snippet: {match['snippet']}")
                        if match['title'] and match['link']:
                            st.write(f"Source: [{match['title']}]({match['link']})")
                        elif match['title']:
                            st.write(f"Source: {match['title']}")
                        st.markdown("---")

                # --- Method Breakdown Block ---
                st.subheader("Method Breakdown")
                for method, score in method_scores.items():
                    # Only show progress for numeric scores, skip lists/dicts (e.g. top_snippets)
                    if method not in [
                        "google_snippet",
                        "google_snippet_title",
                        "google_snippet_link",
                        "google_snippet_text",
                        "google_top_snippets"
                    ]:
                        if isinstance(score, (int, float)):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"{badge(score)} **{method.replace('_', ' ').title()}:**")
                            with col2:
                                st.progress(score/100, text=f"{score}%")

                # Detailed findings
                if "detailed_findings" in result and result["detailed_findings"]:
                    st.subheader("Detailed Findings")
                    for finding in result["detailed_findings"]:
                        st.warning(f"**{finding.get('issue', 'Issue')}**: {finding.get('explanation', '')}")

                # Recommendations
                if "recommendations" in result and result["recommendations"]:
                    st.subheader("Recommendations")
                    for rec in result["recommendations"]:
                        st.info(f"• {rec}")
if option == "PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    # Sensitivity and thresholds are reused from above

    if uploaded_file is not None and st.button('Run Analysis PDF'):
        with st.spinner('Analyzing PDF for plagiarism...'):
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }
            # PDF only—`payload` not needed
            result = analyze_with_retry("/analyze/pdf", files=files)

            if result:
                final_result = result.get("result", {})
                overall_score = final_result.get("overall_plagiarism_score", 0)
                st.metric("Overall Plagiarism Score", f"{overall_score}%")

                # --- Plagiarism/Warning Flags ---
                google_score = final_result.get("method_breakdown", {}).get("google_snippet", 0)
                if overall_score >= PLAGIARISM_THRESHOLD:
                    st.error(f"⚠️ Plagiarism WARNING! Overall score exceeds {PLAGIARISM_THRESHOLD}%")
                elif google_score >= GOOGLE_MATCH_THRESHOLD:
                    st.warning(f"⚠️ Possible Google Plagiarism Match! Highest snippet is {google_score}%")

                method_scores = final_result.get("method_breakdown", {})

                # --- Google Details Block ---
                if "google_snippet" in method_scores:
                    badge_color = badge(google_score)
                    st.subheader(f"Google Search Match {badge_color}")
                    st.write(f"Highest Google snippet similarity: {google_score}%")

                    # Display ALL top snippets
                    top_snippets = method_scores.get("google_top_snippets", [])
                    for i, match in enumerate(top_snippets):
                        st.markdown(f"**Match {i+1}: {badge(match['score'])} {match['score']}%**")
                        if match['snippet']:
                            st.info(f"Matched Snippet: {match['snippet']}")
                        if match['title'] and match['link']:
                            st.write(f"Source: [{match['title']}]({match['link']})")
                        elif match['title']:
                            st.write(f"Source: {match['title']}")
                        st.markdown("---")

                # --- Method Breakdown Block ---
                st.subheader("Method Breakdown")
                for method, score in method_scores.items():
                    # Only show progress for numeric scores, skip lists/dicts (e.g. top_snippets)
                    if method not in [
                        "google_snippet",
                        "google_snippet_title",
                        "google_snippet_link",
                        "google_snippet_text",
                        "google_top_snippets"
                    ]:
                        if isinstance(score, (int, float)):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"{badge(score)} **{method.replace('_', ' ').title()}:**")
                            with col2:
                                st.progress(score/100, text=f"{score}%")

                # Detailed findings
                if "detailed_findings" in result and result["detailed_findings"]:
                    st.subheader("Detailed Findings")
                    for finding in result["detailed_findings"]:
                        st.warning(f"**{finding.get('issue', 'Issue')}**: {finding.get('explanation', '')}")

                # Recommendations
                if "recommendations" in result and result["recommendations"]:
                    st.subheader("Recommendations")
                    for rec in result["recommendations"]:
                        st.info(f"• {rec}")
