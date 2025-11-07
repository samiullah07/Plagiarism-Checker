from detectors.methods import (
    run_string_matching, run_citation_analysis, run_stylometry,
    run_semantic_analysis, run_cross_lingual, run_metadata_analysis
)
from detectors.pdf import extract_pdf_text, extract_pdf_metadata

def analyze_document(text, methods=None, sensitivity=80):
    result = {
        "overall_plagiarism_score": 0,
        "method_breakdown": {},
        "confidence_level": "high",
        "detailed_findings": [],
        "flagged_sections": [],
        "recommendations": []
    }
    # Run methods as required
    breakdown = {}
    breakdown['string_matching'] = run_string_matching(text)
    breakdown['citation_analysis'] = run_citation_analysis(text)
    breakdown['stylometry'] = run_stylometry(text)
    breakdown['semantic_analysis'] = run_semantic_analysis(text)
    breakdown['cross_lingual'] = run_cross_lingual(text)
    # Only run for PDFs:
    breakdown['metadata_analysis'] = 0
    # Weighted average & confidence:
    scores = [v for k,v in breakdown.items() if v is not None]
    result['overall_plagiarism_score'] = sum(scores) // len(scores)
    result['method_breakdown'] = breakdown
    return result

def analyze_pdf(content, methods=None, sensitivity=80):
    text = extract_pdf_text(content)
    metadata_score = run_metadata_analysis(content)
    result = analyze_document(text, methods, sensitivity)
    result['method_breakdown']['metadata_analysis'] = metadata_score
    scores = [v for v in result['method_breakdown'].values()]
    result['overall_plagiarism_score'] = sum(scores) // len(scores)
    return result
