# methods_refactored.py
import re
import os
import math
import requests
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional

# Load semantic model once
semantic_model = SentenceTransformer('all-mpnet-base-v2')

# -------------------------
# Utilities
# -------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def sliding_chunks(text: str, chunk_size_words: int = 50, stride: Optional[int] = None) -> List[str]:
    if stride is None:
        stride = max(1, chunk_size_words // 2)
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, max(1, len(words) - chunk_size_words + 1), stride):
        chunk = ' '.join(words[i:i + chunk_size_words])
        chunks.append(chunk)
    # include tail if not covered
    if len(words) > chunk_size_words and (len(words) - chunk_size_words) % stride != 0:
        chunk = ' '.join(words[-chunk_size_words:])
        if chunk not in chunks:
            chunks.append(chunk)
    if not chunks:
        chunks = [' '.join(words)]
    return chunks


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw_text": resp.text, "status_code": resp.status_code}

# -------------------------
# 0) Google / SerpApi helper (semantic + string scoring)
# -------------------------
def serpapi_google_search(query: str, serpapi_key: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query SerpApi. Returns list of organic_results items (may be empty).
    Caller must provide a valid serpapi_key.
    """
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


def run_google_snippet_matching(chunk: str, serpapi_key: str, top_n: int = 3) -> Dict[str, Any]:
    """
    For a given text chunk, run a small search and compute combined string+semantic similarity
    against returned snippets. Returns dict with best score (0-100) and top_snippets details.
    """
    result = {"score": 0, "top_snippets": []}
    if not chunk or not serpapi_key:
        return result

    query = chunk if len(chunk) <= 200 else chunk[:200]
    hits = serpapi_google_search(query, serpapi_key, num_results=7)
    if not hits:
        return result

    # embed chunk once
    try:
        chunk_emb = semantic_model.encode([chunk], convert_to_tensor=True)
    except Exception:
        chunk_emb = None

    scored = []
    for hit in hits:
        snippet = hit.get("snippet") or hit.get("title") or ""
        title = hit.get("title", "")
        link = hit.get("link", hit.get("url", ""))
        # string similarity
        ssim = sequence_ratio(normalize_text(chunk), normalize_text(snippet))
        # semantic similarity via embeddings if available
        sem = 0.0
        if chunk_emb is not None and snippet.strip():
            try:
                s_emb = semantic_model.encode([snippet], convert_to_tensor=True)
                sem = float(util.pytorch_cos_sim(chunk_emb, s_emb).item())
            except Exception:
                sem = 0.0
        # combine: weight semantic higher for paraphrase detection
        combined = (0.4 * ssim) + (0.6 * sem)
        scored.append({
            "combined_score": combined,
            "string_score": ssim,
            "semantic_score": sem,
            "snippet": snippet,
            "title": title,
            "link": link
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    top = scored[:top_n]
    # convert to percent
    top_details = [{
        "score": round(item["combined_score"] * 100, 2),
        "string_score": round(item["string_score"] * 100, 2),
        "semantic_score": round(item["semantic_score"] * 100, 2),
        "snippet": item["snippet"],
        "title": item["title"],
        "link": item["link"]
    } for item in top]
    best_score = top_details[0]["score"] if top_details else 0
    result["score"] = int(best_score)
    result["top_snippets"] = top_details
    return result

# -------------------------
# 1) String Matching (chunk aware, fingerprint-lite)
# -------------------------
def run_string_matching(text: str, source_texts: Optional[List[str]] = None, sensitivity: int = 80) -> Dict[str, Any]:
    """
    Compute string-based similarity using sliding chunks and optional source_texts.
    Returns {'score':int, 'details': {...}} where higher score => more overlap.
    """
    text_clean = normalize_text(text)
    chunks = sliding_chunks(text_clean, chunk_size_words=50)
    if not chunks:
        return {"score": 0, "details": {}}

    # If no external sources provided, use built-in common phrases
    if not source_texts:
        source_texts = [
            "in this paper we present", "as shown in figure", "previous research has shown",
            "the results indicate that", "it is important to note", "based on our analysis"
        ]
    # compute chunk->source best similarities
    chunk_scores = []
    for chunk in chunks:
        best = 0.0
        for src in source_texts:
            s = max(sequence_ratio(chunk, normalize_text(src)), partial_ratio(chunk, normalize_text(src)))
            best = max(best, s)
        chunk_scores.append(best)
    # overall metrics
    avg_chunk_sim = float(np.mean(chunk_scores)) if chunk_scores else 0.0
    peak_sim = float(np.max(chunk_scores)) if chunk_scores else 0.0
    # density: fraction of chunks above threshold
    threshold = sensitivity / 100.0
    density = sum(1 for s in chunk_scores if s >= threshold) / len(chunk_scores)
    # scoring: combine peak and density
    base = (0.7 * peak_sim) + (0.3 * avg_chunk_sim)
    score = int(min(100, base * 100 * (1 + density)))
    details = {
        "avg_chunk_similarity": round(avg_chunk_sim * 100, 2),
        "peak_chunk_similarity": round(peak_sim * 100, 2),
        "chunk_density_above_threshold": round(density * 100, 2),
        "chunks_evaluated": len(chunk_scores)
    }
    return {"score": score, "details": details}


# -------------------------
# 2) Citation Analysis (returns suspicion score: higher => suspicious)
# -------------------------
def extract_reference_section(text: str) -> List[str]:
    ref_keywords = ['references', 'bibliography', 'works cited']
    lines = text.lower().split('\n')
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

    # Heuristic: well-cited texts are less suspicious; no citations in academic text -> suspicious
    academic = detect_academic_content(text)
    if len(citations) == 0 and academic:
        suspicion = 80  # high suspicion if academic and no citations
    else:
        # Normalize density to 0-100 and convert to suspicion (more density => less suspicion)
        norm_density = min(citation_density / 5.0, 100)  # arbitrary scale
        # compute ref match ratio
        ref_match = 50 if not refs else (min(100, sum(1 for c in citations if any(re.search(r'\b' + re.escape(a) + r'\b', ' '.join(refs)) for a in re.findall(r'[A-Z][a-z]+', c))) / max(1, len(citations)) * 100))
        # suspicion lowers with better density/match
        suspicion = int(max(0, 100 - (0.6 * norm_density + 0.4 * ref_match)))
    details = {"citations_found": len(citations), "reference_lines": len(refs)}
    return {"score": int(suspicion), "details": details}


def detect_academic_content(text: str) -> bool:
    academic_keywords = [
        'research', 'study', 'findings', 'results', 'data', 'analysis',
        'according to', 'previous work', 'literature', 'experiment', 'methodology'
    ]
    text_lower = text.lower()
    hits = sum(1 for k in academic_keywords if k in text_lower)
    return hits >= 2


# -------------------------
# 3) Stylometry (suspicion score)
# -------------------------
def run_stylometry(text: str, sensitivity: int = 80) -> Dict[str, Any]:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    if len(sentences) < 3:
        return {"score": 50, "details": {"reason": "too_short"}}

    lengths = [len(s.split()) for s in sentences]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    ttr = calculate_vocabulary_richness(text)
    passive = check_passive_voice(text)
    punctuation = analyze_punctuation(text)

    # Combine into a style-consistency metric (higher => consistent/human)
    style_consistency = (max(0, 100 - std_len * 2) * 0.3) + (ttr * 0.4) + (passive * 0.1) + (punctuation * 0.2)
    style_consistency = max(0, min(100, style_consistency))
    # Convert to suspicion (higher when style is unlikely/robotic)
    suspicion = int(100 - style_consistency)
    details = {
        "mean_sentence_length": round(mean_len, 2),
        "std_sentence_length": round(std_len, 2),
        "type_token_ratio": round(ttr, 2),
        "passive_voice_percent": passive,
        "punctuation_score": punctuation
    }
    return {"score": suspicion, "details": details}


def calculate_vocabulary_richness(text: str) -> float:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    if not words:
        return 0.0
    ttr = len(set(words)) / len(words)
    return min(100.0, ttr * 200.0)


def check_passive_voice(text: str) -> float:
    patterns = [r'\b(is|are|was|were|be|been|being) [a-z]+ed\b', r'\bby [a-z]+\b']
    sentences = re.split(r'[.!?]+', text)
    passive_count = sum(1 for s in sentences if any(re.search(p, s.lower()) for p in patterns))
    ratio = passive_count / len(sentences) if sentences else 0
    return round(min(100, ratio * 100), 2)


def analyze_punctuation(text: str) -> float:
    total_sentences = max(1, len(re.split(r'[.!?]+', text)))
    count = (text.count(',') + text.count(';') + text.count(':')) / total_sentences
    if 1 <= count <= 3:
        return 85.0
    if 0.5 <= count <= 4:
        return 70.0
    return 40.0


# -------------------------
# 4) Semantic Analysis (compare to search hits if provided)
# -------------------------
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

    # Sensitivity mapping: higher sensitivity => lower threshold to mark as similar
    # Convert avg to 0-100 score (higher means more similar => more likely plagiarized)
    score = int(min(100, avg * 100 + (100 - sensitivity) * 0.2))
    details = {"avg_similarity": round(avg, 4), "comparisons": len(comparison_texts)}
    return {"score": score, "details": details}


# -------------------------
# 5) Cross-lingual detection
# -------------------------
def detect_translation_patterns(text: str) -> float:
    patterns = [
        r'\b(a|the|an) [a-z]+(ed|ing)\b',
        r'\b(very|quite|rather) [a-z]+\b',
        r'\b(do |make |have ) [a-z]+\b'
    ]
    matches = sum(len(re.findall(p, text.lower())) for p in patterns)
    words = len(text.split())
    density = (matches / max(1, words)) * 1000
    return min(100.0, density * 2)


def check_idiom_usage(text: str) -> float:
    idioms = ['piece of cake', 'break a leg', 'hit the books', 'cost an arm and a leg']
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


# -------------------------
# 6) Metadata / PDF forensics
# -------------------------
def run_metadata_analysis(pdf_bytes: bytes, sensitivity: int = 80) -> Dict[str, Any]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = doc.metadata or {}
        checks = [
            check_dates(meta),
            check_author_info(meta),
            check_fonts(doc),
            check_content_flow(doc)
        ]
        doc.close()
        base = int(sum(checks) / len(checks))
        score = int(base * (sensitivity / 100.0))
        return {"score": score, "details": {"metadata": meta}}
    except Exception as e:
        return {"score": 50, "details": {"error": str(e)}}


def check_dates(meta: Dict[str, Any]) -> int:
    c, m = meta.get('creationDate'), meta.get('modDate')
    if c and m:
        return 20 if c == m else 80
    return 50


def check_author_info(meta: Dict[str, Any]) -> int:
    author, prod = meta.get('author', ''), meta.get('producer', '')
    if author and prod:
        if 'Microsoft' in prod and not author:
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
    sents = re.split(r'[.!?]+', total_text)
    if len(sents) < 5:
        return 30
    avg_len = sum(len(s.split()) for s in sents) / len(sents)
    return 80 if 10 <= avg_len <= 25 else 40

# -------------------------
# Aggregator
# -------------------------
def analyze_document(text: str = "", pdf_bytes: Optional[bytes] = None, serpapi_key: Optional[str] = None, sensitivity: int = 80) -> Dict[str, Any]:
    """
    Runs all modules and returns a structured report:
    {
      "overall_plagiarism_score": float,  # 0-100 where higher means more plagiarized
      "modules": { "string_matching": {...}, ... },
      "google_top_snippets": [...]
    }
    """
    # Normalize inputs
    text = text or ""
    # Prepare chunk-level Google matching for stronger signal
    chunks = sliding_chunks(normalize_text(text), chunk_size_words=40)
    google_top_snippets = []
    google_scores = []
    if serpapi_key and chunks:
        # query only a subset to limit cost
        for c in chunks[:6]:
            g = run_google_snippet_matching(c, serpapi_key, top_n=2)
            if g.get("top_snippets"):
                google_top_snippets.extend(g["top_snippets"])
                google_scores.append(g.get("score", 0))
    # Run modules
    m_string = run_string_matching(text, sensitivity=sensitivity)
    m_citation = run_citation_analysis(text, sensitivity=sensitivity)
    # For stylometry/semantic/cross-lingual, run on full text (or could be chunk-based)
    m_stylo = run_stylometry(text, sensitivity=sensitivity)
    # Provide google snippets text as comparison_texts to semantic analysis (if available)
    comparison_texts = [s["snippet"] for s in google_top_snippets] if google_top_snippets else None
    m_semantic = run_semantic_analysis(text, comparison_texts=comparison_texts, sensitivity=sensitivity)
    m_cross = run_cross_lingual(text, sensitivity=sensitivity)
    m_meta = {"score": 0, "details": {}}
    if pdf_bytes:
        m_meta = run_metadata_analysis(pdf_bytes, sensitivity=sensitivity)

    # Weighting (tunable)
    weights = {
        "string_matching": 0.45,
        "semantic_analysis": 0.25,
        "citation_analysis": 0.05,
        "stylometry": 0.10,
        "cross_lingual": 0.05,
        "metadata_analysis": 0.10
    }
    # Ensure all module scores exist
    scores = {
        "string_matching": m_string.get("score", 0),
        "semantic_analysis": m_semantic.get("score", 0),
        "citation_analysis": m_citation.get("score", 0),
        "stylometry": m_stylo.get("score", 0),
        "cross_lingual": m_cross.get("score", 0),
        "metadata_analysis": m_meta.get("score", 0)
    }
    overall = sum(scores[k] * weights.get(k, 0) for k in scores)
    # Map module "suspicion" to plagiarism percent more directly:
    overall_plagiarism_percent = round(min(100, overall), 2)

    report = {
        "overall_plagiarism_score": overall_plagiarism_percent,
        "modules": {
            "string_matching": m_string,
            "semantic_analysis": m_semantic,
            "citation_analysis": m_citation,
            "stylometry": m_stylo,
            "cross_lingual": m_cross,
            "metadata_analysis": m_meta
        },
        "google_top_snippets": google_top_snippets[:5],
        "raw_module_scores": scores,
        "weights": weights
    }
    return report
