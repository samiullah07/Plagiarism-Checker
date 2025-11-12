# legal_rag_plagiarism_tool.py
"""
Advanced AI + Plagiarism + RAG toolkit (Streamlit single-file app)
Features:
 - BM25 lexical search (rank_bm25)
 - Semantic search (SentenceTransformers + FAISS)
 - Async chunked SerpAPI plagiarism checks
 - Improved burstiness metric
 - Perplexity-based AI detection (with model fallback)
 - Dynamic AI likelihood scoring
 - Smart plagiarism classification (Direct/Paraphrase/Patchwriting)
 - SQLite history logging
 - PDF export of expert report
 - Dashboard metrics & verdict
 - Placeholders for: DetectGPT, ColBERT integration, RoBERTa stylometry, benchmarking scripts
"""

import os
import re
import time
import math
import json
import sqlite3
import asyncio
import aiohttp
import tempfile
from urllib.parse import urljoin
from difflib import SequenceMatcher
from typing import List, Dict

import streamlit as st
import requests
import fitz  # pymupdf
from fpdf import FPDF

# NLP libs
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# Optional but recommended libs
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from langdetect import detect as lang_detect
except Exception:
    lang_detect = None

# ---------------------
# Config & Helpers
# ---------------------
st.set_page_config(page_title="Legal RAG + Plagiarism + AI Detection", layout="wide")

APP_DB = "analysis_history.db"

def init_db():
    conn = sqlite3.connect(APP_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            input_excerpt TEXT,
            verdict TEXT,
            plag_score REAL,
            ai_score_text TEXT,
            full_report TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_analysis(mode, input_excerpt, verdict, plag_score, ai_score_text, full_report):
    conn = sqlite3.connect(APP_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses (timestamp, mode, input_excerpt, verdict, plag_score, ai_score_text, full_report)
        VALUES (datetime('now'),?,?,?,?,?)
    """, (mode, input_excerpt[:500], verdict, plag_score, ai_score_text, full_report))
    conn.commit()
    conn.close()

# ---------------------
# Model Loading (perplexity + embeddings)
# - Attempt a stronger causal LM (phi-2 preferred), else fallback to gpt2
# ---------------------
@st.cache_resource(show_spinner=False)
def load_perplexity_model(preferred="microsoft/phi-2"):
    try:
        tok = AutoTokenizer.from_pretrained(preferred, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(preferred, dtype=torch.float32, device_map=None)
        device = next(model.parameters()).device
        return {"name": preferred, "tokenizer": tok, "model": model, "device": device}
    except Exception as e:
        st.warning(f"Could not load {preferred} (maybe no GPU). Falling back to gpt2: {e}")
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = next(model.parameters()).device
        return {"name": "gpt2", "tokenizer": tok, "model": model, "device": device}

@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name="all-mpnet-base-v2"):
    return SentenceTransformer(model_name)

PERP_MODEL = load_perplexity_model()
ST_MODEL = load_sentence_transformer()

# ---------------------
# Perplexity calculation (sliding window)
# ---------------------
def compute_perplexity(text: str, model_bundle, stride=512, max_length=1024):
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"]
    n_tokens = input_ids.size(1)
    if n_tokens == 0:
        return float("inf")
    nlls = []
    for i in range(0, n_tokens, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, n_tokens)
        chunk_ids = input_ids[:, begin_loc:end_loc].to(device)
        if chunk_ids.size(1) <= 1:
            continue
        target_ids = chunk_ids.clone()
        # mask the earlier tokens
        target_ids[:, :-stride] = -100
        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
            nll = outputs.loss * (end_loc - begin_loc)
            nlls.append(nll)
    if not nlls:
        return float("inf")
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / n_tokens).item()
    return round(float(ppl), 2)

# ---------------------
# Burstiness (improved)
# ---------------------
def burstiness_score(text: str):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    avg_words = [len(s.split()) for s in sentences]
    variance = float(np.var(avg_words)) if len(avg_words) > 0 else 0.0
    punctuation_variety = len(set(re.findall(r'[,:;—\-()]', text))) / max(1, len(sentences))
    score = round(((math.sqrt(variance) + punctuation_variety) / (max(1, math.sqrt(len(avg_words))))) , 3)
    return score

# fallback import of numpy if not earlier (we use above)
import numpy as np

# ---------------------
# AI detection pipeline
# ---------------------
def ai_detection_pipeline(text: str, model_bundle):
    ppl = compute_perplexity(text, model_bundle)
    burst = burstiness_score(text)
    if ppl < 30 and burst < 0.2:
        label = "Likely AI-generated"
    elif ppl > 70 and burst > 0.25:
        label = "Likely Human-written"
    else:
        label = "Mixed/Uncertain"
    return {"perplexity": ppl, "burstiness": burst, "ai_likelihood": label}

def get_ai_likelihood_score(perplexity, burstiness, ai_label):
    if ai_label == "Likely Human-written":
        return "15% - Likely Human"
    elif ai_label == "Likely AI-generated":
        return "90% - Likely AI-Generated"
    else:
        if perplexity < 30 and burstiness < 0.3:
            score = 75
        elif perplexity > 70 and burstiness > 0.7:
            score = 35
        else:
            score = 50
        return f"{score}% - Uncertain"

# ---------------------
# Chunking utilities for SerpAPI search
# ---------------------
def split_to_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def chunk_text(text, sentences_per_chunk=3, overlap=1):
    sents = split_to_sentences(text)
    chunks = []
    i = 0
    while i < len(sents):
        chunk = " ".join(sents[i:i+sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
        i += max(1, sentences_per_chunk - overlap)
    return chunks if chunks else [text[:500]]

# ---------------------
# Async SerpAPI calls
# ---------------------
async def serpapi_query(session, q, serpapi_key, num=3):
    url = "https://serpapi.com/search"
    params = {"q": q, "api_key": serpapi_key, "engine": "google", "num": num}
    try:
        async with session.get(url, params=params, timeout=20) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("organic_results", [])
    except Exception:
        return []

async def serpapi_run_chunks(chunks, serpapi_key, concurrency=4):
    results = []
    conn = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [serpapi_query(session, c, serpapi_key) for c in chunks]
        pages = await asyncio.gather(*tasks)
        for p in pages:
            results.extend(p or [])
    return results

def run_serpapi_search(text, serpapi_key):
    if not serpapi_key:
        return {"overall_plagiarism_score": 0, "top_matches": []}
    chunks = chunk_text(text, sentences_per_chunk=3)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    raw_results = loop.run_until_complete(serpapi_run_chunks(chunks, serpapi_key))
    # dedupe
    seen = set()
    aggregated = []
    for r in raw_results:
        snippet = r.get("snippet","")
        link = r.get("link","")
        title = r.get("title","")
        key = (snippet, link)
        if key in seen:
            continue
        seen.add(key)
        # compute similarity scores
        s_str = SequenceMatcher(None, text, snippet).ratio()
        emb1 = ST_MODEL.encode([text], convert_to_tensor=True)
        emb2 = ST_MODEL.encode([snippet], convert_to_tensor=True)
        s_sem = float(util.pytorch_cos_sim(emb1, emb2).item())
        combined = 0.45 * s_str + 0.55 * s_sem
        aggregated.append({"snippet": snippet, "link": link, "title": title,
                           "string_score": round(s_str*100,1),
                           "semantic_score": round(s_sem*100,1),
                           "combined_score": round(combined*100,1)})
    top = max(aggregated, key=lambda x: x["combined_score"], default=None)
    overall = top["combined_score"] if top else 0
    return {"overall_plagiarism_score": overall, "top_matches": aggregated}

# ---------------------
# Plagiarism classification
# ---------------------
def classify_matches(matches):
    flagged = []
    for m in matches:
        s_score = m.get("string_score",0)
        sem_score = m.get("semantic_score",0)
        if s_score > 70:
            typ = "Direct Copying"
            sev = "High"
            analysis = "Verbatim overlap with web source."
            improvement = "Use quotes + cite or rewrite."
        elif sem_score > 70:
            typ = "Paraphrased"
            sev = "Medium"
            analysis = "High semantic overlap indicates close paraphrase."
            improvement = "Rephrase and cite properly."
        elif s_score > 50 or sem_score > 50:
            typ = "Possible Patchwriting"
            sev = "Low"
            analysis = "Moderate overlap; risk of patchwriting."
            improvement = "Rewrite more originally."
        else:
            continue
        flagged.append({**m, "type": typ, "severity": sev, "analysis": analysis, "improvement": improvement})
    return flagged

# ---------------------
# BM25 index (lexical retrieval)
# ---------------------
@st.cache_resource(show_spinner=False)
def build_bm25_from_docs(doc_texts: List[str]):
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 not installed. pip install rank-bm25")
    tokenized = [doc.split() for doc in doc_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

# ---------------------
# FAISS semantic index utilities
# ---------------------
@st.cache_resource(show_spinner=False)
def build_faiss_from_docs(doc_texts: List[str], embed_model_name="all-mpnet-base-v2"):
    # returns (index, embeddings, docs)
    embed_model = SentenceTransformer(embed_model_name)
    embeddings = embed_model.encode(doc_texts, convert_to_tensor=False, show_progress_bar=False)
    import numpy as np
    idx = None
    if faiss is None:
        st.warning("faiss not available; semantic search will use in-memory cosine compare.")
        return None, np.array(embeddings), doc_texts, embed_model
    dim = embeddings.shape[1] if hasattr(embeddings, "shape") else len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    # normalize for inner product -> cosine
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(np.array(emb_norm).astype("float32"))
    return index, np.array(embeddings), doc_texts, embed_model

def faiss_search(index, query, emb_model, docs, top_k=5):
    q_emb = emb_model.encode([query], convert_to_tensor=False)
    import numpy as np
    qn = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    if index is None:
        # fallback: compute cosine manually
        sims = (docs @ qn.T).flatten()
        idxs = np.argsort(-sims)[:top_k]
        return [{"doc": docs[i], "score": float(sims[i])} for i in idxs]
    D, I = index.search(qn.astype("float32"), top_k)
    return [{"doc": docs[i], "score": float(D[0][j])} for j,i in enumerate(I[0])]

# ---------------------
# Stylometry (placeholder for RoBERTa fine-tune / DetectGPT)
# ---------------------
@st.cache_resource(show_spinner=False)
def load_stylometry_model():
    # Placeholder: In production you would fine-tune roberta-base on PAN or author corpora.
    # For now we return a sentence-transformer to compute style embeddings + a simple distance metric.
    return SentenceTransformer("all-mpnet-base-v2")

STYLO_MODEL = load_stylometry_model()

def stylometry_score(text, known_author_embeddings: List[np.ndarray]=None):
    # Simple heuristic: compute embedding and distance to known author centroids if provided.
    emb = STYLO_MODEL.encode([text], convert_to_tensor=True)
    if not known_author_embeddings:
        # return a neutral score
        return {"prob_same_author": 0.5, "notes": "No author reference embeddings provided."}
    # compute similarity to first (example)
    sim = float(util.pytorch_cos_sim(emb, known_author_embeddings[0]).item())
    return {"prob_same_author": round(sim,3), "notes": "Compare to author centroid."}

# DetectGPT: placeholder for the DetectGPT algorithm (requires multiple forward passes & perturbations)
def detectgpt_stub(text, llm_bundle):
    # Placeholder: implement DetectGPT following the paper by OpenAI if you want robust detection
    # For now return uncertain
    return {"detectgpt_score": 0.5, "notes": "DetectGPT not implemented; this is a stub."}

# ---------------------
# Report generation (markdown + PDF)
# ---------------------
def generate_markdown_report(title, plag_scores, flagged, ai_scores, ai_hallmarks, verdict):
    lines = []
    lines.append(f"# {title}\n")
    lines.append("## PART 1 — Plagiarism Analysis\n")
    lines.append(f"- **Overall Plagiarism Score:** {plag_scores['overall']}\n")
    if flagged:
        lines.append("### Flagged Sections\n")
        for i,f in enumerate(flagged,1):
            badge = "🟥" if f['severity']=="High" else "🟧" if f['severity']=="Medium" else "🟨"
            lines.append(f"{i}. {badge} **{f['type']}** — score(s): string {f['string_score']}%, semantic {f['semantic_score']}%")
            lines.append(f"   - Snippet: {f['snippet']}")
            lines.append(f"   - Source: {f.get('link','')}")
            lines.append(f"   - Analysis: {f['analysis']}")
            lines.append(f"   - Fix: {f['improvement']}\n")
    else:
        lines.append("No significant matches found.\n")
    lines.append("## PART 2 — AI Authorship Detection\n")
    lines.append(f"- **AI Likelihood:** {ai_scores['score']}")
    lines.append(f"- **Perplexity:** {ai_scores['ppl']}")
    lines.append(f"- **Burstiness:** {ai_scores['burst']}")
    lines.append(f"- **Label:** {ai_scores['summary']}\n")
    lines.append("## PART 3 — Verdict & Recommendations\n")
    lines.append(f"- **Verdict:** {verdict}\n")
    lines.append("- **Recommendations:**")
    lines.append("  - Re-write high-severity flagged text or add explicit citations.")
    lines.append("  - Inject original examples and personal analysis.")
    lines.append("  - If AI was used, disclose and edit for voice and depth.\n")
    lines.append("---\n")
    lines.append("_This report is heuristic & research-grade. Use professional services for official determinations._\n")
    return "\n".join(lines)

from fpdf import FPDF

def create_pdf_report(report_text, title="Expert Report"):
    """
    Create a Unicode-safe PDF report.
    Works for English, Urdu, em-dashes, quotes, etc.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Use a built-in Unicode font (DejaVuSans supports most chars)
    pdf.add_font("DejaVu", "", r"C:\Windows\Fonts\DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)
    
    pdf.multi_cell(0, 7, txt=title)
    pdf.ln(5)
    
    for line in report_text.split("\n"):
        pdf.multi_cell(0, 6, txt=line)
    
    # Output as bytes (no latin-1 encoding!)
    return bytes(pdf.output(dest="S").encode("utf-8"))


# ---------------------
# Verdict logic & confidence score
# ---------------------
def compute_verdict_and_confidence(plag_score, ai_score_percent):
    # ai_score_percent should be 0-100 numeric
    # Confidence: combine plagiarism & ai signals
    conf = round((100 - plag_score)*0.6 + (100 - ai_score_percent)*0.4, 1)  # higher is better for originality
    if ai_score_percent < 40 and plag_score < 20:
        verdict = "✅ Human-written & original"
    elif ai_score_percent > 70 or plag_score > 50:
        verdict = "⚠️ Possibly AI-generated or copied"
    else:
        verdict = "🟡 Mixed content"
    return verdict, conf

# ---------------------
# UI: Sidebar config
# ---------------------
st.title("📚 Legal RAG + Plagiarism + AI Detection Toolkit")
with st.sidebar:
    st.header("Configuration")
    serpapi_key = st.text_input("SerpAPI Key (SerpAPI)", value=os.getenv("SERPAPI_KEY",""), type="password")
    use_bm25 = st.checkbox("Enable BM25 (fast lexical search)", value=True)
    use_faiss = st.checkbox("Enable FAISS semantic search (local)", value=True)
    emb_model_name = st.text_input("Embedding model (sentence-transformers)", value="all-mpnet-base-v2")
    prefer_perp_model = st.text_input("Preferred perplexity model", value="microsoft/phi-2")
    show_history = st.checkbox("Show analysis history", value=True)
    st.markdown("---")
    st.write("Notes:")
    st.write("- Advanced features (DetectGPT, ColBERT, fine-tuned stylometry) are optional and need local setup.")
    st.write("- FAISS requires significant RAM if you store many documents.")

# reload preferred perp model if changed (not cached)
PERP_MODEL = load_perplexity_model(preferred=st.sidebar.text_input("Perplexity model (re-run to reload)", value="microsoft/phi-2"))

# ---------------------
# Main UI input selection
# ---------------------
mode = st.radio("Input mode", ["Text", "PDF", "RAG Search"], index=0, horizontal=True)

if mode == "Text":
    txt = st.text_area("Paste / type your text", height=300)
    run_btn = st.button("Analyze Text")
    if run_btn and txt.strip():
        with st.spinner("Running analysis..."):
            lang = None
            if lang_detect:
                try:
                    lang = lang_detect(txt)
                except Exception:
                    lang = None
            ai_res = ai_detection_pipeline(txt, PERP_MODEL)
            plag_res = run_serpapi_search(txt, serpapi_key) if serpapi_key else {"overall_plagiarism_score":0,"top_matches":[]}
            flagged = classify_matches(plag_res["top_matches"])
            # stylometry stub
            styl = stylometry_score(txt)
            detectgpt = detectgpt_stub(txt, PERP_MODEL)
            # produce ai numeric percent from get_ai_likelihood_score string (parse)
            ai_percent = 0
            try:
                ai_percent = int(get_ai_likelihood_score(ai_res['perplexity'], ai_res['burstiness'], ai_res['ai_likelihood']).split("%")[0])
            except Exception:
                ai_percent = 50
            verdict, confidence = compute_verdict_and_confidence(float(plag_res['overall_plagiarism_score']), ai_percent)
            # Build ai_scores dict
            ai_scores = {
                "score": get_ai_likelihood_score(ai_res['perplexity'], ai_res['burstiness'], ai_res['ai_likelihood']),
                "summary": ai_res['ai_likelihood'],
                "burst": ai_res['burstiness'],
                "ppl": ai_res['perplexity']
            }
            plag_scores = {"overall": round(float(plag_res['overall_plagiarism_score']),1), "summary": "Similarity with online content found via SerpAPI"}
            md = generate_markdown_report("Expert Analysis", plag_scores, flagged, ai_scores, [], verdict)
            st.markdown("### Verdict")
            st.metric("Verdict", verdict, delta=f"Confidence: {confidence}%")
            st.markdown("### Expert Report")
            st.markdown(md.replace("\n","  \n"))
            # Download buttons
            pdf_bytes = create_pdf_report(md, title="Expert Report")
            st.download_button("Download PDF", data=pdf_bytes, file_name="expert_report.pdf", mime="application/pdf")
            # Save history
            save_analysis("text", txt[:500], verdict, plag_res['overall_plagiarism_score'], ai_scores['score'], md)

elif mode == "PDF":
    up = st.file_uploader("Upload PDF", type=["pdf"])
    if up:
        with st.spinner("Extracting PDF text..."):
            try:
                doc = fitz.open(stream=up.read(), filetype="pdf")
                pages_text = [p.get_text() for p in doc]
                doc.close()
                pdf_text = "\n".join(pages_text)
            except Exception as e:
                st.error(f"PDF error: {e}")
                pdf_text = ""
        st.text_area("Extracted Text (editable)", value=pdf_text, height=250)
        if st.button("Analyze PDF") and pdf_text.strip():
            with st.spinner("Analyzing PDF..."):
                ai_res = ai_detection_pipeline(pdf_text, PERP_MODEL)
                plag_res = run_serpapi_search(pdf_text, serpapi_key) if serpapi_key else {"overall_plagiarism_score":0,"top_matches":[]}
                flagged = classify_matches(plag_res["top_matches"])
                ai_percent = int(get_ai_likelihood_score(ai_res['perplexity'], ai_res['burstiness'], ai_res['ai_likelihood']).split("%")[0])
                verdict, confidence = compute_verdict_and_confidence(float(plag_res['overall_plagiarism_score']), ai_percent)
                ai_scores = {"score": get_ai_likelihood_score(ai_res['perplexity'], ai_res['burstiness'], ai_res['ai_likelihood']), "summary":ai_res['ai_likelihood'], "burst": ai_res['burstiness'], "ppl": ai_res['perplexity']}
                plag_scores = {"overall": round(float(plag_res['overall_plagiarism_score']),1), "summary": "Similarity with online content found via SerpAPI"}
                md = generate_markdown_report("Expert Analysis (PDF)", plag_scores, flagged, ai_scores, [], verdict)
                st.metric("Verdict", verdict, delta=f"Confidence: {confidence}%")
                st.markdown(md.replace("\n","  \n"))
                pdf_bytes = create_pdf_report(md, title="Expert Report")
                st.download_button("Download PDF", data=pdf_bytes, file_name="expert_report.pdf", mime="application/pdf")
                save_analysis("pdf", pdf_text[:500], verdict, plag_res['overall_plagiarism_score'], ai_scores['score'], md)

else:  # RAG Search mode
    st.subheader("RAG: Upload documents to build local retrieval index (BM25 + FAISS)")
    files = st.file_uploader("Upload one or more text/PDF files (or .txt)", accept_multiple_files=True)
    doc_texts = []
    if files:
        for f in files:
            name = f.name.lower()
            try:
                if name.endswith(".pdf"):
                    doc = fitz.open(stream=f.read(), filetype="pdf")
                    txt = "\n".join([p.get_text() for p in doc])
                    doc.close()
                    doc_texts.append(txt)
                else:
                    txt = f.getvalue().decode("utf-8", errors="ignore")
                    doc_texts.append(txt)
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")
    if doc_texts:
        st.success(f"Loaded {len(doc_texts)} documents.")
        if use_bm25 and BM25Okapi:
            bm25, tokenized = build_bm25_from_docs(doc_texts)
        else:
            bm25 = None
        if use_faiss:
            try:
                faiss_index, embeddings, docs_for_faiss, embed_model = build_faiss_from_docs(doc_texts, embed_model_name=emb_model_name)
            except Exception as e:
                st.error(f"FAISS build failed: {e}")
                faiss_index = None
                docs_for_faiss = doc_texts
                embed_model = ST_MODEL
        else:
            faiss_index = None
            docs_for_faiss = doc_texts
            embed_model = ST_MODEL

        q = st.text_input("Enter query for retrieval (e.g., 'Section 302 punishment'):")
        if st.button("Search RAG") and q.strip():
            with st.spinner("Searching..."):
                bm25_results = []
                if bm25:
                    tokenized_q = q.split()
                    bm25_scores = bm25.get_scores(tokenized_q)
                    topk = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:5]
                    bm25_results = [{"doc_index": i, "score": float(bm25_scores[i]), "excerpt": doc_texts[i][:500]} for i in topk]
                sem_results = []
                if embed_model:
                    # use embed_model + faiss_index (if present) or fallback to semantic ranking
                    if faiss_index is not None:
                        sem = faiss_search(faiss_index, q, embed_model, embeddings, top_k=5)
                        # sem returns doc references
                        sem_results = [{"doc_index": idx, "score": r["score"], "excerpt": doc_texts[idx][:500]} for idx,r in enumerate(sem)][:5]
                    else:
                        # compute similarity manually
                        q_emb = embed_model.encode([q], convert_to_tensor=True)
                        sims = [float(util.pytorch_cos_sim(q_emb, embed_model.encode([d], convert_to_tensor=True)).item()) for d in doc_texts]
                        topk = sorted(range(len(sims)), key=lambda i: -sims[i])[:5]
                        sem_results = [{"doc_index": i, "score": float(sims[i]), "excerpt": doc_texts[i][:500]} for i in topk]
                st.write("BM25 Results")
                st.json(bm25_results)
                st.write("Semantic Results")
                st.json(sem_results)
                # You can now select top-k passages to feed into an LLM for summarization/QA

# ---------------------
# Show history
# ---------------------
if show_history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis History (last 10)")
    conn = sqlite3.connect(APP_DB)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, mode, verdict, plag_score FROM analyses ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    for r in rows:
        st.sidebar.write(f"{r[1]} | {r[2]} | {r[3]} (plag {round(r[4],1)}%)")

# ---------------------
# Advanced placeholders / notes
# ---------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced integrations (how to)")
st.sidebar.markdown("""
**DetectGPT**: implement as in the DetectGPT paper — generate perturbed samples and compute prediction differences. Heavy compute required.

**ColBERT + FAISS**: use ColBERT/colbertv2 for dense retrieval and produce embeddings per token; combine with FAISS for efficient late interaction retrieval. See https://github.com/kimiyoung/colBERT

**Stylometry (RoBERTa)**: fine-tune roberta-base on PAN author identification datasets, save classifier, and load here to produce probability scores.

**Benchmarking**:
 - Download PAN plagiarism corpus, Wikipedia dumps, and curated essay datasets locally.
 - Write evaluation scripts: precision/recall on detected copied spans, ROC AUC for AI-detection, confusion matrices for stylometry.

**Deployment**:
 - Move heavy model inference to a GPU backend (FastAPI) and keep Streamlit as a UI.
""")

st.sidebar.markdown("----")
st.sidebar.write("If you want, I can help implement any specific advanced integration step-by-step.")
