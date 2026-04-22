import os
import re
import spacy
import nltk
import numpy as np

from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from dotenv import load_dotenv
from database.db import get_connection

load_dotenv()

# ─────────────────────────────────────────────
# ONE-TIME SETUP
# Download NLTK stopwords if not already present
# ─────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOPWORDS = set(stopwords.words("english"))

# Load spaCy model — used for sentence segmentation and noun chunk filtering
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# Load KeyBERT — uses a lightweight sentence-transformer under the hood
KBERT = KeyBERT(model="all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _split_into_chunks(text: str, max_chunk_chars: int = 1500) -> list[str]:
    """
    Splits large text into smaller chunks for processing.
    We split by paragraphs first, then merge until max_chunk_chars.

    Why: TF-IDF and KeyBERT both perform better on focused chunks
    than on one massive 20,000-character blob.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_chars:
            current_chunk += " " + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:max_chunk_chars]]


def _is_valid_topic(phrase: str) -> bool:
    """
    Filters out low-quality topic candidates.
    A valid topic must:
    - Be at least 2 characters long
    - Not be purely numeric
    - Not be a stopword
    - Contain at least one alphabetic word
    """
    phrase = phrase.strip().lower()

    if len(phrase) < 2:
        return False
    if phrase.isnumeric():
        return False
    if phrase in STOPWORDS:
        return False

    words = phrase.split()
    has_alpha = any(w.isalpha() and w not in STOPWORDS for w in words)
    return has_alpha


def _normalize_topic(phrase: str) -> str:
    """
    Normalizes a topic phrase:
    - Title case
    - Remove extra whitespace
    - Remove leading/trailing punctuation
    """
    phrase = phrase.strip().strip(".,;:-")
    phrase = re.sub(r'\s+', ' ', phrase)
    return phrase.title()


def _deduplicate_topics(topics: list[str], similarity_threshold: float = 0.75) -> list[str]:
    """
    Removes near-duplicate topics using character-level overlap.

    Example:
        "Binary Search" and "Binary Search Tree" → keeps "Binary Search Tree"
        "Neural Network" and "Neural Networks" → keeps "Neural Networks"

    Why not use embeddings here: We're staying within allowed libraries.
    This simple containment check works well for educational content.
    """
    topics = sorted(topics, key=len, reverse=True)  # Longer phrases first
    unique = []

    for topic in topics:
        topic_lower = topic.lower()
        is_duplicate = False

        for kept in unique:
            kept_lower = kept.lower()
            # If one is a substring of another, skip the shorter one
            if topic_lower in kept_lower or kept_lower in topic_lower:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(topic)

    return unique


# ─────────────────────────────────────────────
# EXTRACTION METHODS
# ─────────────────────────────────────────────

def extract_topics_tfidf(text: str, top_n: int = 20) -> list[str]:
    chunks = _split_into_chunks(text)

    if len(chunks) < 2:
        chunks = chunks * 2

    total_chunks = len(chunks)

    # Dynamically set min_df and max_df based on corpus size
    # For large PDFs (many chunks), min_df=1 with high max_df causes pruning
    min_df = 1
    max_df = min(0.95, max(0.5, 1 - (1 / total_chunks)))

    print(f"[TF-IDF] Chunks: {total_chunks}, min_df: {min_df}, max_df: {max_df:.2f}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=500,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"[a-zA-Z][a-zA-Z0-9\-]{2,}"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        # Fallback: relax all constraints completely
        print("[TF-IDF] Warning: Falling back to relaxed vectorizer settings.")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=500,
            min_df=1,
            max_df=1.0,     # No upper limit
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-]{2,}"
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)

    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    topics = []
    for idx in top_indices:
        term = feature_names[idx]
        if _is_valid_topic(term):
            topics.append(_normalize_topic(term))

    print(f"[Topic Extractor] TF-IDF extracted {len(topics)} candidate topics.")
    return topics


def extract_topics_keybert(text: str, top_n: int = 20) -> list[str]:
    """
    Extracts semantically meaningful keyphrases using KeyBERT.

    How it works:
    - KeyBERT encodes the document and candidate phrases using
      sentence-transformers (all-MiniLM-L6-v2)
    - Finds phrases whose embedding is most similar to the document embedding
    - MMR (Maximal Marginal Relevance) ensures diversity in results

    Args:
        text:  Full curriculum text (we use a truncated version).
        top_n: Number of keyphrases to extract.

    Returns:
        List of topic strings.
    """
    # KeyBERT struggles with very long text — use first 5000 chars
    # This is usually the most content-dense part of a curriculum
    text_sample = text[:5000]

    keywords = KBERT.extract_keywords(
        text_sample,
        keyphrase_ngram_range=(1, 3),   # Up to 3-word phrases
        stop_words="english",
        use_mmr=True,                   # Maximize diversity of results
        diversity=0.5,                  # 0=redundant, 1=maximally diverse
        top_n=top_n
    )

    topics = []
    for phrase, score in keywords:
        if score > 0.2 and _is_valid_topic(phrase):  # Confidence threshold
            topics.append(_normalize_topic(phrase))

    print(f"[Topic Extractor] KeyBERT extracted {len(topics)} candidate topics.")
    return topics


def extract_noun_chunks_spacy(text: str, top_n: int = 15) -> list[str]:
    """
    Extracts noun phrases using spaCy's dependency parser.

    How it works:
    - spaCy parses sentences and identifies noun chunks
      e.g. "the binary search algorithm" → "binary search algorithm"
    - We filter to chunks that appear frequently — frequency
      acts as a proxy for importance in the curriculum

    Why this is valuable:
    - TF-IDF finds terms, KeyBERT finds semantically similar phrases,
      but spaCy finds grammatically valid concept names
    - Together all three cover different failure modes

    Args:
        text:  Full curriculum text.
        top_n: Number of top noun chunks to return.

    Returns:
        List of topic strings.
    """
    # spaCy has a token limit — process first 10,000 chars
    doc = NLP(text[:10000])

    chunk_freq: dict[str, int] = {}
    for chunk in doc.noun_chunks:
        normalized = _normalize_topic(chunk.text)
        if _is_valid_topic(normalized) and len(normalized.split()) >= 2:
            chunk_freq[normalized] = chunk_freq.get(normalized, 0) + 1

    # Sort by frequency, return top_n
    sorted_chunks = sorted(chunk_freq.items(), key=lambda x: x[1], reverse=True)
    topics = [chunk for chunk, freq in sorted_chunks[:top_n]]

    print(f"[Topic Extractor] spaCy noun chunks extracted {len(topics)} candidate topics.")
    return topics


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def extract_and_save_topics(curriculum_id: int, text: str, top_n: int = 15) -> list[dict]:
    """
    Full topic extraction pipeline:
    1. Run TF-IDF, KeyBERT, and spaCy extraction
    2. Merge all results
    3. Deduplicate
    4. Save final topics to DB

    Args:
        curriculum_id: ID of the curriculum in the DB.
        text:          Full extracted curriculum text.
        top_n:         Final number of topics to save.

    Returns:
        List of dicts with topic_id and topic_name.
    """
    print(f"\n[Topic Extractor] Starting extraction for curriculum_id={curriculum_id}")

    # ── Run all three extractors ──
    tfidf_topics  = extract_topics_tfidf(text, top_n=25)
    keybert_topics = extract_topics_keybert(text, top_n=25)
    spacy_topics  = extract_noun_chunks_spacy(text, top_n=20)

    # ── Merge: KeyBERT first (highest quality), then TF-IDF, then spaCy ──
    merged = keybert_topics + tfidf_topics + spacy_topics

    # ── Deduplicate ──
    unique_topics = _deduplicate_topics(merged)

    # ── Limit to top_n after deduplication ──
    final_topics = unique_topics[:top_n]

    if not final_topics:
        raise ValueError(
            "No topics could be extracted. "
            "Check if the PDF text is readable and content-rich."
        )

    print(f"[Topic Extractor] Final topic count after dedup: {len(final_topics)}")

    # ── Save to DB ──
    conn = get_connection()
    cursor = conn.cursor()
    saved_topics = []

    for order_index, topic_name in enumerate(final_topics):
        cursor.execute("""
            INSERT INTO topics (curriculum_id, topic_name, order_index)
            VALUES (?, ?, ?)
        """, (curriculum_id, topic_name, order_index))

        topic_id = cursor.lastrowid
        saved_topics.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "order_index": order_index
        })

    conn.commit()
    conn.close()

    print(f"[Topic Extractor] {len(saved_topics)} topics saved to DB.")
    return saved_topics


def get_topics_for_curriculum(curriculum_id: int) -> list[dict]:
    """
    Fetches saved topics for a curriculum from the DB.
    Used by downstream modules (learning path, quiz generator).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, topic_name, order_index
        FROM topics
        WHERE curriculum_id = ?
        ORDER BY order_index ASC
    """, (curriculum_id,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {"topic_id": row["id"], "topic_name": row["topic_name"], "order_index": row["order_index"]}
        for row in rows
    ]