"""
upgraded_plagiarism_tool.py

Single-file upgraded plagiarism detection pipeline:
- detectors: string matching, citation analysis, stylometry, semantic, cross-lingual, metadata
- serpapi integration: multi-chunk search and combined string+semantic scoring
- analyze_document() and analyze_pdf() aggregator functions

Save and import in your app:
from upgraded_plagiarism_tool import analyze_document, analyze_pdf
"""

import os
import re
import math
import requests
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------
# Global model (load once)
# ---------------------------------------------------------------------
semantic_model = SentenceTransformer("all-mpnet-base-v2")

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sliding_chunks(text: str, chunk_size_words: int = 50, stride: Optional[int] = None) -> List[str]:
    if stride is None:
        stride = max(1, chunk_size_words // 2)
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, max(1, len(words) - chunk_size_words + 1), stride):
        chunk = " ".join(words[i:i + chunk_size_words])
        chunks.append(chunk)
    # include tail
    if len(words) > chunk_size_words and (len(words) - chunk_size_words) % stride != 0:
        tail = " ".join(words[-chunk_size_words:])
        if tail not in chunks:
            chunks.append(tail)
    if not chunks:
        chunks = [" ".join(words)]
    return chunks


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def partial_ratio(str1: str, str2: str) -> float:
    shorter, longer = (str1, str2) if len(str1) <= len(str2) else (str2, str1)
    m = SequenceMatcher(None, shorter, longer)
    scores = []
    for block in m.get_matching_blocks():
        start = block[1]
        substring = longer[start:start + len(shorter)]
        scores.append(SequenceMatcher(None, shorter, substring).ratio())
    return max(scores) if scores else 0.0


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw_text": resp.text, "status_code": resp.status_code}

# ---------------------------------------------------------------------
# SerpAPI helper (multi-chunk + semantic/string combined scoring)
# ---------------------------------------------------------------------
def serpapi_google_search(query: str, serpapi_key: str, num_results: int = 7) -> List[Dict[str, Any]]:
    if not serpapi_key:
        return []
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": serpapi_key, "engine": "google", "num": num_results}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = safe_json(r)
        return data.get("organic_results", []) or []
    except Exception:
        return []


def score_snippet_against_chunk(chunk: str, snippet: str, chunk_emb=None) -> Dict[str, Any]:
    s_str = sequence_ratio(normalize_text(chunk), normalize_text(snippet))
    s_sem = 0.0
    if chunk_emb is not None and snippet.strip():
        try:
            s_emb = semantic_model.encode([snippet], convert_to_tensor=True)
            s_sem = float(util.pytorch_cos_sim(chunk_emb, s_emb).item())
        except Exception:
            s_sem = 0.0
    # combine semantic heavier to catch paraphrase
    combined = 0.45 * s_str + 0.55 * s_sem
    return {"string_score": s_str, "semantic_score": s_sem, "combined": combined, "snippet": snippet}


def run_google_snippet_matching_fulltext(text: str, serpapi_key: Optional[str], chunk_size_words: int = 40,
                                         max_chunks: int = 6, top_n_per_chunk: int = 3) -> Dict[str, Any]:
    """
    Break text into chunks, run SerpAPI search per chunk, score returned snippets using
    both string and semantic similarity; return top matches across chunks.
    """
    out = {"score": 0, "top_snippets": []}
    if not serpapi_key or not text:
        return out

    chunks = sliding_chunks(normalize_text(text), chunk_size_words=chunk_size_words)
    chunks = chunks[:max_chunks]  # limit API calls/cost
    scored_snippets = []

    for chunk in chunks:
        # query SerpAPI with chunk (prefer longer snippets; if chunk too short, include next chunk)
        query = chunk if len(chunk) <= 300 else chunk[:300]
        hits = serpapi_google_search(query, serpapi_key, num_results=top_n_per_chunk)
        if not hits:
            continue
        # embed chunk once
        try:
            chunk_emb = semantic_model.encode([chunk], convert_to_tensor=True)
        except Exception:
            chunk_emb = None

        for hit in hits:
            snippet = hit.get("snippet") or hit.get("title") or ""
            title = hit.get("title", "")
            link = hit.get("link", hit.get("url", ""))
            scored = score_snippet_against_chunk(chunk, snippet, chunk_emb=chunk_emb)
            scored.update({"title": title, "link": link})
            scored_snippets.append(scored)

    if not scored_snippets:
        return out

    # sort by combined score and return top overall results
    scored_snippets.sort(key=lambda x: x["combined"], reverse=True)
    top = scored_snippets[:10]
    top_details = [{
        "score": round(item["combined"] * 100, 2),
        "string_score": round(item["string_score"] * 100, 2),
        "semantic_score": round(item["semantic_score"] * 100, 2),
        "snippet": item["snippet"],
        "title": item.get("title"),
        "link": item.get("link")
    } for item in top]
    out["top_snippets"] = top_details
    out["score"] = int(top_details[0]["score"]) if top_details else 0
    return out

# ---------------------------------------------------------------------
# 1) String Matching (chunk-aware)
# ---------------------------------------------------------------------
def run_string_matching(text: str, source_texts: Optional[List[str]] = None, sensitivity: int = 80) -> Dict[str, Any]:
    text_clean = normalize_text(text)
    chunks = sliding_chunks(text_clean, chunk_size_words=50)
    if not chunks:
        return {"score": 0, "details": {"reason": "empty_text"}}

    if not source_texts:
        source_texts = [
            "in this paper we present", "as shown in figure", "previous research has shown",
            "the results indicate that", "it is important to note", "based on our analysis"
        ]

    chunk_scores = []
    for chunk in chunks:
        best = 0.0
        for src in source_texts:
            s = max(sequence_ratio(chunk, normalize_text(src)), partial_ratio(chunk, normalize_text(src)))
            best = max(best, s)
        chunk_scores.append(best)

    avg_chunk_sim = float(np.mean(chunk_scores)) if chunk_scores else 0.0
    peak_sim = float(np.max(chunk_scores)) if chunk_scores else 0.0
    threshold = sensitivity / 100.0
    density = sum(1 for s in chunk_scores if s >= threshold) / len(chunk_scores)
    base = (0.7 * peak_sim) + (0.3 * avg_chunk_sim)
    score_val = int(min(100, base * 100 * (1 + density)))
    details = {
        "avg_chunk_similarity": round(avg_chunk_sim * 100, 2),
        "peak_chunk_similarity": round(peak_sim * 100, 2),
        "chunk_density_above_threshold": round(density * 100, 2),
        "chunks_evaluated": len(chunk_scores)
    }
    return {"score": score_val, "details": details}

# ---------------------------------------------------------------------
# 2) Citation Analysis
# ---------------------------------------------------------------------
def extract_reference_section(text: str) -> List[str]:
    ref_keywords = ["references", "bibliography", "works cited"]
    lines = text.lower().split("\n")
    section, capture = [], False
    for line in lines:
        if any(k in line for k in ref_keywords):
            capture = True
            continue
        if capture and line.strip():
            section.append(line)
    return section


def run_citation_analysis(text: str, sensitivity: int = 80) -> Dict[str, Any]:
    patterns = [
        r"\([A-Z][a-z]+ et al\.?, \d{4}\)",
        r"\([A-Z][a-z]+ and [A-Z][a-z]+, \d{4}\)",
        r"\[[\d,\s]+\]",
        r"[A-Z][a-z]+ \(\d{4}\)"
    ]
    citations = []
    for p in patterns:
        citations.extend(re.findall(p, text))
    refs = extract_reference_section(text)
    word_count = max(1, len(text.split()))
    citation_density = len(citations) / word_count * 1000
    academic = detect_academic_content(text)

    if len(citations) == 0 and academic:
        suspicion = 80
    else:
        norm_density = min(citation_density / 5.0, 100)
        # reference matching: naive author occurrence heuristic
        if not refs:
            ref_match = 0
        else:
            author_matches = 0
            for c in citations:
                authors = re.findall(r"[A-Z][a-z]+", c)
                for a in authors:
                    if any(re.search(r"\b" + re.escape(a) + r"\b", ref) for ref in refs):
                        author_matches += 1
                        break
            ref_match = min(100, (author_matches / max(1, len(citations))) * 100)
        suspicion = int(max(0, 100 - (0.6 * norm_density + 0.4 * ref_match)))
    return {"score": int(suspicion), "details": {"citations_found": len(citations), "reference_lines": len(refs)}}


def detect_academic_content(text: str) -> bool:
    academic_keywords = [
        "research", "study", "findings", "results", "data", "analysis",
        "according to", "previous work", "literature", "experiment", "methodology"
    ]
    text_lower = text.lower()
    hits = sum(1 for k in academic_keywords if k in text_lower)
    return hits >= 2

# ---------------------------------------------------------------------
# 3) Stylometry
# ---------------------------------------------------------------------
def calculate_vocabulary_richness(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    if not words:
        return 0.0
    ttr = len(set(words)) / len(words)
    return min(100.0, ttr * 200.0)


def check_passive_voice(text: str) -> float:
    patterns = [r"\b(is|are|was|were|be|been|being) [a-z]+ed\b", r"\bby [a-z]+\b"]
    sentences = re.split(r"[.!?]+", text)
    passive_count = sum(1 for s in sentences if any(re.search(p, s.lower()) for p in patterns))
    ratio = passive_count / len(sentences) if sentences else 0
    return round(min(100, ratio * 100), 2)


def analyze_punctuation(text: str) -> float:
    total_sentences = max(1, len(re.split(r"[.!?]+", text)))
    count = (text.count(",") + text.count(";") + text.count(":")) / total_sentences
    if 1 <= count <= 3:
        return 85.0
    if 0.5 <= count <= 4:
        return 70.0
    return 40.0


def run_stylometry(text: str, sensitivity: int = 80) -> Dict[str, Any]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
    if len(sentences) < 3:
        return {"score": 50, "details": {"reason": "too_short"}}
    lengths = [len(s.split()) for s in sentences]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    ttr = calculate_vocabulary_richness(text)
    passive = check_passive_voice(text)
    punctuation = analyze_punctuation(text)
    style_consistency = (max(0, 100 - std_len * 2) * 0.3) + (ttr * 0.4) + (passive * 0.1) + (punctuation * 0.2)
    style_consistency = max(0, min(100, style_consistency))
    suspicion = int(100 - style_consistency)
    details = {"mean_sentence_length": round(mean_len, 2), "std_sentence_length": round(std_len, 2),
               "type_token_ratio": round(ttr, 2), "passive_voice_percent": passive, "punctuation_score": punctuation}
    return {"score": suspicion, "details": details}

# ---------------------------------------------------------------------
# 4) Semantic analysis (compare against provided comparisons or generic phrases)
# ---------------------------------------------------------------------
def run_semantic_analysis(text: str, comparison_texts: Optional[List[str]] = None, sensitivity: int = 80) -> Dict[str, Any]:
    if not text:
        return {"score": 0, "details": {}}
    if comparison_texts is None:
        comparison_texts = [
            "in this paper we present", "based on our analysis", "previous research has shown"
        ]
    try:
        text_emb = semantic_model.encode([text], convert_to_tensor=True)
        sims = []
        for c in comparison_texts:
            c_emb = semantic_model.encode([c], convert_to_tensor=True)
            sims.append(float(util.pytorch_cos_sim(text_emb, c_emb).item()))
        avg = float(np.mean(sorted(sims, reverse=True)[:3])) if sims else 0.0
    except Exception:
        avg = 0.0
    # map to score
    score = int(min(100, avg * 100 + (100 - sensitivity) * 0.2))
    return {"score": score, "details": {"avg_similarity": round(avg, 4), "comparisons": len(comparison_texts)}}

# ---------------------------------------------------------------------
# 5) Cross-lingual
# ---------------------------------------------------------------------
def detect_translation_patterns(text: str) -> float:
    patterns = [
        r"\b(a|the|an) [a-z]+(ed|ing)\b",
        r"\b(very|quite|rather) [a-z]+\b",
        r"\b(do |make |have ) [a-z]+\b"
    ]
    matches = sum(len(re.findall(p, text.lower())) for p in patterns)
    words = len(text.split())
    density = (matches / max(1, words)) * 1000
    return min(100.0, density * 2)


def check_idiom_usage(text: str) -> float:
    idioms = ["piece of cake", "break a leg", "hit the books", "cost an arm and a leg"]
    total, correct = 0, 0
    for idiom in idioms:
        if idiom in text.lower():
            total += 1
            if len(text.split()) > 50:
                correct += 1
    if total == 0:
        return 50.0
    return (correct / total) * 100.0


def run_cross_lingual(text: str, sensitivity: int = 80) -> Dict[str, Any]:
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(1, len(text))
    tl_score = detect_translation_patterns(text)
    idiom_score = check_idiom_usage(text)
    weight = sensitivity / 100.0
    non_ascii_part = min(non_ascii_ratio * 200 * weight, 100)
    total = (non_ascii_part + tl_score * weight + idiom_score * weight) / 3
    return {"score": int(total), "details": {"non_ascii_ratio": round(non_ascii_ratio, 4), "translation_pattern_score": tl_score, "idiom_score": idiom_score}}

# ---------------------------------------------------------------------
# 6) Metadata / PDF forensics
# ---------------------------------------------------------------------
def check_dates(meta: Dict[str, Any]) -> int:
    c, m = meta.get("creationDate"), meta.get("modDate")
    if c and m:
        return 20 if c == m else 80
    return 50


def check_author_info(meta: Dict[str, Any]) -> int:
    author, prod = meta.get("author", ""), meta.get("producer", "")
    if author and prod:
        if "Microsoft" in prod and not author:
            return 30
        elif len(author) > 1:
            return 85
    return 50


def check_fonts(doc) -> int:
    fonts = set()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    fonts.add(span.get("font", ""))
    return 30 if len(fonts) > 3 else 60 if len(fonts) > 1 else 90


def check_content_flow(doc) -> int:
    total_text = "".join(page.get_text() for page in doc)
    sents = re.split(r"[.!?]+", total_text)
    if len(sents) < 5:
        return 30
    avg_len = sum(len(s.split()) for s in sents) / len(sents)
    return 80 if 10 <= avg_len <= 25 else 40


def run_metadata_analysis(pdf_bytes: bytes, sensitivity: int = 80) -> Dict[str, Any]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = doc.metadata or {}
        checks = [check_dates(meta), check_author_info(meta), check_fonts(doc), check_content_flow(doc)]
        doc.close()
        base = int(sum(checks) / len(checks))
        score = int(base * (sensitivity / 100.0))
        return {"score": score, "details": {"metadata": meta}}
    except Exception as e:
        return {"score": 50, "details": {"error": str(e)}}

# ---------------------------------------------------------------------
# Aggregator: analyze_document and analyze_pdf
# ---------------------------------------------------------------------
def analyze_document(text: str = "", pdf_bytes: Optional[bytes] = None,
                     serpapi_key: Optional[str] = None, sensitivity: int = 80) -> Dict[str, Any]:
    """
    Runs all detector modules and returns a structured report with weighted scoring.
    overall_plagiarism_score: 0-100 (higher => more plagiarized/suspicious)
    """
    # Load SerpAPI key from env if not provided
    serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")

    text = text or ""
    # Precompute google snippet matches (multi-chunk)
    google_result = run_google_snippet_matching_fulltext(text, serpapi_key) if serpapi_key and text else {"score": 0, "top_snippets": []}

    # Run detectors
    m_string = run_string_matching(text, sensitivity=sensitivity)
    m_citation = run_citation_analysis(text, sensitivity=sensitivity)
    m_stylo = run_stylometry(text, sensitivity=sensitivity)
    comparison_texts = [s["snippet"] for s in google_result["top_snippets"]] if google_result.get("top_snippets") else None
    m_semantic = run_semantic_analysis(text, comparison_texts=comparison_texts, sensitivity=sensitivity)
    m_cross = run_cross_lingual(text, sensitivity=sensitivity)
    m_meta = {"score": 0, "details": {}} if pdf_bytes is None else run_metadata_analysis(pdf_bytes, sensitivity=sensitivity)

    breakdown = {
        "string_matching": m_string,
        "citation_analysis": m_citation,
        "stylometry": m_stylo,
        "semantic_analysis": m_semantic,
        "cross_lingual": m_cross,
        "metadata_analysis": m_meta,
        "google_snippet": {"score": google_result.get("score", 0), "top_snippets": google_result.get("top_snippets", [])}
    }

    # Weighted scoring (tune weights as needed)
    weights = {
        "string_matching": 0.30,
        "semantic_analysis": 0.30,
        "citation_analysis": 0.05,
        "stylometry": 0.10,
        "cross_lingual": 0.05,
        "metadata_analysis": 0.05,
        "google_snippet": 0.15
    }

    module_scores = {
        "string_matching": m_string.get("score", 0),
        "semantic_analysis": m_semantic.get("score", 0),
        "citation_analysis": m_citation.get("score", 0),
        "stylometry": m_stylo.get("score", 0),
        "cross_lingual": m_cross.get("score", 0),
        "metadata_analysis": m_meta.get("score", 0),
        "google_snippet": google_result.get("score", 0)
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for k, w in weights.items():
        val = module_scores.get(k, 0)
        if val is None:
            continue
        weighted_sum += val * w
        total_weight += w

    overall = int(weighted_sum / total_weight) if total_weight else 0
    # clamp
    overall = max(0, min(100, overall))

    report = {
        "overall_plagiarism_score": overall,
        "method_breakdown": breakdown,
        "raw_module_scores": module_scores,
        "weights": weights,
        "confidence_level": "high" if overall >= 20 else "low",
        "google_top_snippets": google_result.get("top_snippets", []),
        "recommendations": []
    }

    # Add simple recommendations
    if overall >= 75:
        report["recommendations"].append("High overlap detected — consider full rewrite or proper quoting and citation.")
    elif overall >= 40:
        report["recommendations"].append("Moderate overlap — check highlighted snippets and verify sources.")
    else:
        report["recommendations"].append("Low overlap detected — review citation quality as needed.")

    return report


def analyze_pdf(pdf_bytes: bytes, serpapi_key: Optional[str] = None, sensitivity: int = 80) -> Dict[str, Any]:
    # extract text
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        text = ""
    return analyze_document(text=text, pdf_bytes=pdf_bytes, serpapi_key=serpapi_key, sensitivity=sensitivity)

# ---------------------------------------------------------------------
# If run directly, quick demo (no API key)
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     sample_text = """Artificial intelligence has revolutionized the way data is analyzed in scientific research.
#     By enabling machines to identify complex patterns and relationships, researchers can now process vast datasets
#     in a fraction of the time it once took."""
#     print("Running demo (no SerpAPI key used)...")
#     r = analyze_document(sample_text, serpapi_key=None, sensitivity=80)
#     import json
#     print(json.dumps(r, indent=2))
