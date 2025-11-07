import re
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF

# Initialize global semantic model once (avoid reloading per call)
semantic_model = SentenceTransformer('all-mpnet-base-v2')

# === Helper Utilities ===

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=50):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def partial_ratio(str1, str2):
    """Partial string matching ratio."""
    shorter, longer = (str1, str2) if len(str1) <= len(str2) else (str2, str1)
    m = SequenceMatcher(None, shorter, longer)
    scores = [
        SequenceMatcher(None, shorter, longer[block[1]:block[1]+len(shorter)]).ratio()
        for block in m.get_matching_blocks()
    ]
    return max(scores) if scores else 0


# === 1️⃣ String Matching and Fingerprinting ===

def run_string_matching(text, source_texts=None):
    if source_texts is None:
        source_texts = [
            "in this paper we present", "as shown in figure", "previous research has shown",
            "the results indicate that", "it is important to note", "based on our analysis"
        ]

    text = normalize_text(text)
    text_chunks = chunk_text(text, chunk_size=50)

    max_similarity = 0
    matches_found = 0

    for source in source_texts:
        src = normalize_text(source)
        ratio = SequenceMatcher(None, text, src).ratio()
        pr = partial_ratio(text, src)
        sim = max(ratio, pr)

        if sim > 0.8:
            matches_found += 1
        max_similarity = max(max_similarity, sim)

    if matches_found > 0:
        base = min(max_similarity * 100, 100)
        density = min((matches_found / len(text_chunks)) * 100, 50)
        score = (base + density) / 2
    else:
        score = max_similarity * 100
    
    print("String Matching Score:", score, "| Input:", text)
    return int(score)

# === 2️⃣ Citation Analysis ===

def extract_reference_section(text):
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


def check_citation_consistency(citations):
    if not citations:
        return 0
    formats = [
        'et_al' if 'et al' in c else
        'and' if 'and' in c else
        'bracketed' if c.startswith('[') else
        'other' for c in citations
    ]
    most_common = Counter(formats).most_common(1)[0][1]
    return most_common / len(citations) * 100


def check_reference_matches(citations, refs):
    if not citations or not refs:
        return 50
    author_matches = 0
    for c in citations:
        authors = re.findall(r'[A-Z][a-z]+', c)
        if any(any(a in ref for ref in refs) for a in authors):
            author_matches += 1
    return (author_matches / len(citations)) * 100


def run_citation_analysis(text):
    """Improved citation analysis without class dependencies"""
    
    citation_patterns = [
        r"\([A-Z][a-z]+ et al\.?, \d{4}\)",
        r"\([A-Z][a-z]+ and [A-Z][a-z]+, \d{4}\)", 
        r"\[[\d,\s]+\]",
        r"[A-Z][a-z]+ \(\d{4}\)"
    ]
    
    citation_count = 0
    for pattern in citation_patterns:
        citations = re.findall(pattern, text)
        citation_count += len(citations)
    
    word_count = len(text.split())
    
    # More nuanced scoring
    if citation_count == 0:
        # Check if this is academic/scientific content that SHOULD have citations
        academic_indicators = detect_academic_content(text)
        if academic_indicators:
            return 70  # Moderate suspicion for academic content without citations
        else:
            return 30  # Low suspicion for non-academic content
    
    # Calculate citation density
    citation_density = (citation_count / word_count) * 1000
    
    # Good citation density gets lower plagiarism score
    if citation_density > 5:  # Well-cited
        return 20
    elif citation_density > 2:  # Moderately cited
        return 40
    else:  # Poorly cited
        return 60

def detect_academic_content(text):
    """Detect if text contains academic/scientific content that needs citations"""
    academic_keywords = [
        'research', 'study', 'findings', 'results', 'data', 'analysis',
        'according to', 'previous work', 'literature', 'scholars',
        'experiment', 'hypothesis', 'methodology', 'conclusion'
    ]
    
    scientific_terms = [
        'significant', 'correlation', 'probability', 'statistical',
        'theory', 'model', 'framework', 'paradigm'
    ]
    
    text_lower = text.lower()
    
    academic_score = 0
    for keyword in academic_keywords:
        if keyword in text_lower:
            academic_score += 1
    
    for term in scientific_terms:
        if term in text_lower:
            academic_score += 1
    
    
    return academic_score >= 2  # At least 2 academic indicators
# === 3️⃣ Stylometry Analysis ===

def calculate_sentence_variation(sentences):
    if not sentences:
        return 0
    lengths = [len(s.split()) for s in sentences]
    v = np.std(lengths) / np.mean(lengths)
    return 90 if v < 0.5 else 70 if v < 1.0 else 30


def calculate_vocabulary_richness(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    if not words:
        return 0
    ttr = len(set(words)) / len(words)
    return min(ttr * 200, 100)


def calculate_readability(text):
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not sentences or not words:
        return 50
    syllables = sum(count_syllables(w) for w in words)
    asl, asw = len(words)/len(sentences), syllables/len(words)
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return max(0, min(score, 100))


def count_syllables(word):
    vowels = "aeiouy"
    word = word.lower()
    count = int(word[0] in vowels)
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(1, count)


def check_passive_voice(text):
    patterns = [r'\b(is|are|was|were|be|been|being) [a-z]+ed\b', r'\bby [a-z]+\b']
    sentences = re.split(r'[.!?]+', text)
    passive_count = sum(
        1 for s in sentences if any(re.search(p, s.lower()) for p in patterns)
    )
    ratio = passive_count / len(sentences) if sentences else 0
    return 80 if 0.1 <= ratio <= 0.3 else 40


def analyze_punctuation(text):
    total_sentences = len(re.split(r'[.!?]+', text))
    if not total_sentences:
        return 50
    count = (text.count(',') + text.count(';') + text.count(':')) / total_sentences
    return 85 if 1 <= count <= 3 else 70 if 0.5 <= count <= 4 else 40


def run_stylometry(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if len(sentences) < 3:
        return 50
    metrics = [
        calculate_sentence_variation(sentences),
        calculate_vocabulary_richness(text),
        calculate_readability(text),
        check_passive_voice(text),
        analyze_punctuation(text)
    ]
    return int(100 - (sum(metrics)/len(metrics)))


# === 4️⃣ Semantic Analysis ===

def run_semantic_analysis(text, comparison_texts=None):
    if comparison_texts is None:
        comparison_texts = [
            "in this paper we present", "based on our analysis", "previous research has shown"
        ]
    text_emb = semantic_model.encode([text], convert_to_tensor=True)
    sims = [
        util.pytorch_cos_sim(text_emb, semantic_model.encode([c], convert_to_tensor=True)).item()
        for c in comparison_texts
    ]
    avg = np.mean(sorted(sims, reverse=True)[:3]) if sims else 0
    return int(avg * 100)


# === 5️⃣ Cross-Lingual Plagiarism ===

def detect_translation_patterns(text):
    patterns = [
        r'\b(a|the|an) [a-z]+(ed|ing)\b',
        r'\b(very|quite|rather) [a-z]+\b',
        r'\b(do |make |have ) [a-z]+\b'
    ]
    matches = sum(len(re.findall(p, text.lower())) for p in patterns)
    words = len(text.split())
    density = (matches / words) * 1000 if words else 0
    return min(density * 2, 100)


def check_idiom_usage(text):
    idioms = ['piece of cake', 'break a leg', 'hit the books', 'cost an arm and a leg']
    total, correct = 0, 0
    for idiom in idioms:
        if idiom in text.lower():
            total += 1
            if len(text.split()) > 50:
                correct += 1
    return 50 if not total else (correct / total) * 100


def run_cross_lingual(text):
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
    scores = [
        min(non_ascii_ratio * 200, 100),
        detect_translation_patterns(text),
        check_idiom_usage(text)
    ]
    return int(sum(scores) / len(scores))


# === 6️⃣ Metadata and PDF Forensics ===

def run_metadata_analysis(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = doc.metadata
        checks = [
            check_dates(meta),
            check_author_info(meta),
            check_fonts(doc),
            check_content_flow(doc)
        ]
        doc.close()
        return int(sum(checks)/len(checks))
    except Exception as e:
        print(f"Metadata analysis error: {e}")
        return 50


def check_dates(meta):
    c, m = meta.get('creationDate'), meta.get('modDate')
    if c and m:
        return 20 if c == m else 80
    return 50


def check_author_info(meta):
    author, prod = meta.get('author', ''), meta.get('producer', '')
    if author and prod:
        if 'Microsoft' in prod and not author:
            return 30
        elif len(author) > 1:
            return 85
    return 50


def check_fonts(doc):
    fonts = set()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    fonts.add(span.get("font", ""))
    return 30 if len(fonts) > 3 else 60 if len(fonts) > 1 else 90


def check_content_flow(doc):
    total_text = "".join(page.get_text() for page in doc)
    sents = re.split(r'[.!?]+', total_text)
    if len(sents) < 5:
        return 30
    avg_len = sum(len(s.split()) for s in sents) / len(sents)
    return 80 if 10 <= avg_len <= 25 else 40
