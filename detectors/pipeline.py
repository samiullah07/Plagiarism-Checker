from detectors.methods import (
    run_string_matching, run_citation_analysis, run_stylometry,
    run_semantic_analysis, run_cross_lingual, run_metadata_analysis, run_google_snippet_matching
)
from detectors.pdf import extract_pdf_text, extract_pdf_metadata
import os

def analyze_document(text, methods=None, sensitivity=80, serpapi_key=None):
    # Always load key from env only if not passed in



    breakdown = {}
    breakdown['string_matching'] = run_string_matching(text, sensitivity=sensitivity)
    breakdown['citation_analysis'] = run_citation_analysis(text, sensitivity=sensitivity)
    breakdown['stylometry'] = run_stylometry(text, sensitivity=sensitivity)
    breakdown['semantic_analysis'] = run_semantic_analysis(text, sensitivity=sensitivity)
    breakdown['cross_lingual'] = run_cross_lingual(text, sensitivity=sensitivity)
    breakdown['metadata_analysis'] = 0  # Non-PDF

    # Only run Google checking if API key is provided
    if serpapi_key is None:
        serpapi_key = os.getenv("SERPAPI_KEY")

    if serpapi_key:
        google_result = run_google_snippet_matching(text, serpapi_key, sensitivity=sensitivity)
        breakdown["google_snippet"] = google_result["score"]
        breakdown["google_top_snippets"] = google_result["top_snippets"]  # This is a list!
        # Optionally keep keys for the single best match (for UI compatibility)
        best = google_result["top_snippets"][0] if google_result["top_snippets"] else {}
        breakdown["google_snippet_title"] = best.get("title")
        breakdown["google_snippet_link"] = best.get("link")
        breakdown["google_snippet_text"] = best.get("snippet")

    scores = [v for k, v in breakdown.items() 
          if isinstance(v, (int, float)) and v is not None]

    if "google_snippet" in breakdown and breakdown["google_snippet"] is not None:
        scores.append(breakdown["google_snippet"])

    result = {
        "overall_plagiarism_score": int(sum(scores) // len(scores)) if scores else 0,
        "method_breakdown": breakdown,
        "confidence_level": "high",
        "detailed_findings": [],
        "flagged_sections": [],
        "recommendations": []
    }
    return result

def analyze_pdf(content, methods=None, sensitivity=80, serpapi_key=None):
    if serpapi_key is None:
        serpapi_key = os.getenv("SERPAPI_KEY")

    text = extract_pdf_text(content)
    metadata_score = run_metadata_analysis(content, sensitivity=sensitivity)
    result = analyze_document(text, methods, sensitivity, serpapi_key=serpapi_key)
    result['method_breakdown']['metadata_analysis'] = metadata_score
    scores = [v for k, v in result['method_breakdown'].items() if v is not None and not k.startswith("google_snippet_")]
    if "google_snippet" in result['method_breakdown'] and result['method_breakdown']["google_snippet"] is not None:
        scores.append(result['method_breakdown']["google_snippet"])

    result['overall_plagiarism_score'] = int(sum(scores) // len(scores)) if scores else 0
    return result
