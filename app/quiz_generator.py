import os
import re
import random
import hashlib
import spacy
import nltk

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from database.db import get_connection

load_dotenv()

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")


# ─────────────────────────────────────────────
# ANSWER EXTRACTION — concise, 3–8 words
# ─────────────────────────────────────────────

# Signals that introduce a definition after them
_DEFN_PATTERNS = [
    r"(?:is defined as|refers to|is a type of|is an?|means|"
    r"is the process of|is the act of|is known as|is called)\s+(.+)",
]

# Signals that introduce a purpose clause
_PURPOSE_PATTERNS = [
    r"(?:used (?:to|for)|designed to|helps to|allows (?:us )?to|"
    r"enables|provides|responsible for|facilitates)\s+(.+)",
]

# Signals for how/mechanism
_HOW_PATTERNS = [
    r"(?:works by|operates by|functions by|achieved by|"
    r"accomplished by|performed by|implemented (?:using|by))\s+(.+)",
]


def _trim_to_phrase(raw: str, max_words: int = 8) -> str:
    """
    Extract the first clean noun phrase from raw text.
    Caps at max_words. Never returns an empty string.
    """
    doc = NLP(raw[:180])
    for chunk in doc.noun_chunks:
        text = chunk.text.strip(" .,;:")
        words = text.split()
        if 2 <= len(words) <= max_words:
            return text
    # Fallback: first max_words words, strip trailing punctuation
    words = raw.split()
    return " ".join(words[:max_words]).strip(" .,;:")


def _extract_answer_from_pattern(sentence: str, patterns: list[str], topic: str, max_words: int = 8) -> str | None:
    lower = sentence.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            start = m.start(1)
            raw_slice = sentence[start: start + len(m.group(1))]
            phrase = _trim_to_phrase(raw_slice, max_words)
            if len(phrase.split()) >= 2 and phrase.lower() != topic.lower():
                return phrase
    return None


def _normalize_length(text: str, max_words: int = 7) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip(" .,;:")


def _title(text: str) -> str:
    """Title-case but preserve existing uppercase acronyms."""
    return " ".join(
        w if w.isupper() and len(w) > 1 else w.capitalize()
        for w in text.split()
    )


# ─────────────────────────────────────────────
# SENTENCE SELECTION
# ─────────────────────────────────────────────

_NOISE_RE = re.compile(
    r"(chapter|section|figure|table|exercise|page|ref\.|et al\.|"
    r"https?://|www\.|^\s*\d+[\.\)]\s*$)",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^[A-Z][A-Z\s]{6,}$|^\d+\.\d+\s+[A-Z]")


def _is_usable_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < 50 or len(s) > 320:
        return False
    if s.endswith((":", "—", "…")):
        return False
    if _NOISE_RE.search(s) or _HEADING_RE.match(s):
        return False
    words = s.split()
    if len([w for w in words if w.isalpha()]) < 6:
        return False
    doc = NLP(s[:180])
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    has_subj = any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in doc)
    return has_verb and has_subj


def _get_topic_relevant_sentences(text: str, topic: str, top_n: int = 40) -> list[str]:
    all_sents = sent_tokenize(text)
    quality = [s.strip() for s in all_sents if _is_usable_sentence(s)]

    if not quality:
        return []

    topic_words = set(topic.lower().split())

    # Pass 1: keyword overlap
    hits = [s for s in quality if topic_words & set(s.lower().split())]
    if len(hits) >= top_n:
        return hits[:top_n]

    # Pass 2: TF-IDF cosine
    try:
        vec = TfidfVectorizer(stop_words="english", max_df=1.0, min_df=1)
        mat = vec.fit_transform([topic] + quality)
        sims = cosine_similarity(mat[0], mat[1:]).flatten()
        top_idx = sims.argsort()[::-1][:top_n]
        extra = [quality[i] for i in top_idx if sims[i] > 0.03 and quality[i] not in hits]
        return (hits + extra)[:top_n]
    except Exception:
        return hits[:top_n]


# ─────────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────────

def _fingerprint(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower().strip()).encode()).hexdigest()


def _is_duplicate_question(new_q: str, existing: list[str], threshold: float = 0.72) -> bool:
    if not existing:
        return False
    try:
        corpus = existing + [new_q]
        mat = TfidfVectorizer().fit_transform(corpus)
        sims = cosine_similarity(mat[-1], mat[:-1])
        return float(sims.max()) >= threshold
    except Exception:
        return new_q.lower() in [q.lower() for q in existing]


# ─────────────────────────────────────────────
# ANSWER TYPE CLASSIFICATION
# ─────────────────────────────────────────────

_TYPE_MAP = {
    "process":   ["processing", "computation", "execution", "transformation", "evaluation"],
    "technique": ["algorithm",  "technique",   "method",    "strategy",       "heuristic"],
    "structure": ["structure",  "framework",   "architecture", "model",        "schema"],
    "tool":      ["system",     "platform",    "library",   "module",         "toolkit"],
    "concept":   ["concept",    "principle",   "theory",    "paradigm",       "abstraction"],
    "other":     ["mechanism",  "component",   "procedure", "representation", "function"],
}

_TYPE_KEYWORDS = {
    "process":   ["process", "processing", "method", "procedure", "operation", "execution"],
    "technique": ["algorithm", "technique", "approach", "strategy", "heuristic"],
    "structure": ["structure", "model", "architecture", "layer", "framework", "tree", "graph"],
    "tool":      ["tool", "system", "platform", "library", "module", "compiler", "interpreter"],
    "concept":   ["concept", "principle", "theory", "notion", "idea", "abstraction"],
}


def _classify_answer_type(answer: str) -> str:
    lower = answer.lower()
    for atype, keywords in _TYPE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return atype
    return "other"


def _length_close(a: str, b: str, tol: int = 3) -> bool:
    return abs(len(a.split()) - len(b.split())) <= tol


# ─────────────────────────────────────────────
# DISTRACTOR GENERATION — typed, length-matched
# ─────────────────────────────────────────────

# Prefix seeds for synthetic distractors — domain-neutral adjectives/adverbs
_SEEDS = [
    "Adaptive", "Recursive", "Sequential", "Parallel", "Distributed",
    "Hierarchical", "Incremental", "Dynamic", "Static", "Greedy",
    "Heuristic", "Symbolic", "Probabilistic", "Deterministic", "Iterative",
    "Modular", "Layered", "Centralized", "Decentralized", "Virtual",
    "Abstract", "Concrete", "Explicit", "Implicit", "Composite",
]


def _make_typed_distractor(correct: str, atype: str, seed: str) -> str:
    suffix_pool = _TYPE_MAP.get(atype, _TYPE_MAP["other"])
    suffix = random.choice(suffix_pool)
    candidate = f"{seed} {suffix}"
    if candidate.lower() == correct.lower():
        alts = [s for s in suffix_pool if s != suffix]
        suffix = random.choice(alts) if alts else suffix_pool[0]
        candidate = f"{seed} {suffix}"
    return candidate


def _generate_distractors(correct: str, all_topics: list[str], sentences: list[str], count: int = 3) -> list[str]:
    correct_lower = correct.lower().strip()
    atype = _classify_answer_type(correct)
    seen = {correct_lower}
    result = []

    # ── Pass 1: typed distractors from other topic names ──────────────────
    topic_pool = [t for t in all_topics if t.lower().strip() != correct_lower]
    random.shuffle(topic_pool)
    for topic in topic_pool:
        if len(result) >= count:
            break
        seed = topic.split()[0]
        cand = _make_typed_distractor(correct, atype, seed)
        if cand.lower() not in seen and _length_close(cand, correct):
            seen.add(cand.lower())
            result.append(cand)

    # ── Pass 2: noun phrases from sentences, type-wrapped ─────────────────
    if len(result) < count:
        target_words = len(correct.split())
        for sent in sentences:
            if len(result) >= count:
                break
            doc = NLP(sent[:200])
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip(" .,;:")
                phrase = _normalize_length(phrase, target_words)
                if len(phrase.split()) < 2:
                    continue
                seed = phrase.split()[0]
                cand = _make_typed_distractor(correct, atype, seed)
                if cand.lower() not in seen and _length_close(cand, correct):
                    seen.add(cand.lower())
                    result.append(cand)
                if len(result) >= count:
                    break

    # ── Pass 3: synthetic fallback seeds ──────────────────────────────────
    if len(result) < count:
        shuffled = _SEEDS[:]
        random.shuffle(shuffled)
        for seed in shuffled:
            if len(result) >= count:
                break
            cand = _make_typed_distractor(correct, atype, seed)
            if cand.lower() not in seen and _length_close(cand, correct):
                seen.add(cand.lower())
                result.append(cand)

    return result[:count]


# ─────────────────────────────────────────────
# QUESTION TEMPLATES
# Each returns (question_text, short_answer) or None
# ─────────────────────────────────────────────

def _template_what_is(sentence: str, topic: str) -> tuple | None:
    answer = _extract_answer_from_pattern(sentence, _DEFN_PATTERNS, topic)
    if answer and len(answer.split()) >= 2:
        return f"What is {topic}?", _title(answer)
    return None


def _template_purpose(sentence: str, topic: str) -> tuple | None:
    answer = _extract_answer_from_pattern(sentence, _PURPOSE_PATTERNS, topic)
    if answer and len(answer.split()) >= 2:
        return f"What is the primary purpose of {topic}?", _title(answer)
    return None


def _template_how_it_works(sentence: str, topic: str) -> tuple | None:
    answer = _extract_answer_from_pattern(sentence, _HOW_PATTERNS, topic)
    if answer and len(answer.split()) >= 2:
        return f"How does {topic} operate?", _title(answer)
    return None


def _template_characteristic(sentence: str, topic: str) -> tuple | None:
    signals = ["always", "never", "must", "cannot", "ensures", "guarantees",
               "requires", "produces", "returns", "stores", "maintains"]
    if not any(sig in sentence.lower() for sig in signals):
        return None
    doc = NLP(sentence[:200])
    # Extract the predicate object as answer
    for token in doc:
        if token.dep_ in ("dobj", "attr", "pobj") and len(token.text.split()) >= 1:
            chunk_text = " ".join(
                [t.text for t in token.subtree]
            ).strip(" .,;:")
            answer = _normalize_length(chunk_text, 7)
            if len(answer.split()) >= 2 and answer.lower() != topic.lower():
                return f"Which of the following is a key characteristic of {topic}?", _title(answer)
    return None


def _template_application(sentence: str, topic: str) -> tuple | None:
    signals = ["applied in", "used in", "application in", "example of",
               "such as", "for example", "e.g.", "commonly used in", "found in"]
    if not any(sig in sentence.lower() for sig in signals):
        return None
    answer = _extract_answer_from_pattern(sentence, [
        r"(?:used in|applied in|found in|common in)\s+(.+)",
        r"(?:such as|for example|e\.g\.)[,\s]+(.+)",
    ], topic)
    if answer and len(answer.split()) >= 2:
        return f"In which context is {topic} commonly applied?", _title(answer)
    return None


def _template_fill_blank(sentence: str, topic: str) -> tuple | None:
    if topic.lower() not in sentence.lower():
        return None
    blanked = re.sub(re.escape(topic), "_____", sentence, flags=re.IGNORECASE, count=1)
    if len(blanked.split()) > 22 or "_____" not in blanked:
        return None
    return f"Fill in the blank: {blanked}", topic


TEMPLATES = [
    _template_what_is,
    _template_purpose,
    _template_how_it_works,
    _template_characteristic,
    _template_application,
    _template_fill_blank,
]

# Limit how many times the same template is reused per topic
_MAX_TEMPLATE_USES = 1


# ─────────────────────────────────────────────
# BUILD MCQ
# ─────────────────────────────────────────────

def _build_mcq(question: str, answer: str, distractors: list[str]) -> dict | None:
    if len(distractors) < 3:
        return None
    options = distractors[:3] + [answer]
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    correct_label = labels[options.index(answer)]
    return {
        "question_text":  question,
        "option_a":       options[0],
        "option_b":       options[1],
        "option_c":       options[2],
        "option_d":       options[3],
        "correct_option": correct_label,
    }


# ─────────────────────────────────────────────
# CORE GENERATION
# ─────────────────────────────────────────────

def generate_questions_for_topic(
    topic_id: int,
    topic_name: str,
    text: str,
    all_topics: list[str],
    questions_per_topic: int = 3,
) -> list[dict]:

    sentences = _get_topic_relevant_sentences(text, topic_name, top_n=50)
    if not sentences:
        return []

    questions = []
    used_sent_fps  = set()
    used_q_texts   = []
    template_uses  = {fn.__name__: 0 for fn in TEMPLATES}

    # Shuffle templates so each topic gets a different ordering
    template_order = TEMPLATES[:]
    random.shuffle(template_order)

    for sentence in sentences:
        if len(questions) >= questions_per_topic:
            break

        sent_fp = _fingerprint(sentence)
        if sent_fp in used_sent_fps:
            continue

        # Sort templates: prefer those used fewer times this topic
        sorted_templates = sorted(
            template_order,
            key=lambda fn: template_uses[fn.__name__]
        )

        for tmpl in sorted_templates:
            # Don't reuse same template style more than _MAX_TEMPLATE_USES times
            if template_uses[tmpl.__name__] >= _MAX_TEMPLATE_USES:
                continue

            result = tmpl(sentence, topic_name)
            if result is None:
                continue

            question, answer = result

            # Reject trivially short answers
            if len(answer.split()) < 2:
                continue

            # Hard cap on answer length
            if len(answer.split()) > 8:
                answer = _normalize_length(answer, 7)
                if len(answer.split()) < 2:
                    continue

            # Semantic duplicate check against already-generated questions
            if _is_duplicate_question(question, used_q_texts):
                continue

            distractors = _generate_distractors(answer, all_topics, sentences)
            if len(distractors) < 3:
                continue

            # Final length-consistency check across all 4 options
            all_options = distractors[:3] + [answer]
            max_words = max(len(o.split()) for o in all_options)
            min_words = min(len(o.split()) for o in all_options)
            if max_words - min_words > 4:
                # Trim the outliers
                all_options = [_normalize_length(o, min_words + 3) for o in all_options]
                answer      = all_options[-1]
                distractors = all_options[:3]

            mcq = _build_mcq(question, answer, distractors)
            if mcq is None:
                continue

            questions.append(mcq)
            used_sent_fps.add(sent_fp)
            used_q_texts.append(question)
            template_uses[tmpl.__name__] += 1
            break   # One question per sentence

    return questions


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def generate_and_save_quiz(curriculum_id: int, questions_per_topic: int = 3):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id as topic_id, topic_name FROM topics WHERE curriculum_id = ?",
        (curriculum_id,)
    )
    topics = cursor.fetchall()

    cursor.execute(
        "SELECT extracted_text_path FROM curricula WHERE id = ?",
        (curriculum_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not topics or not row or not row["extracted_text_path"]:
        raise ValueError(f"Missing topics or extracted text for curriculum_id={curriculum_id}")

    with open(row["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    all_topic_names = [t["topic_name"] for t in topics]

    # Global question fingerprint set — prevents cross-topic duplicates
    global_q_fps = set()

    conn = get_connection()
    cursor = conn.cursor()

    for topic in topics:
        topic_id   = topic["topic_id"]
        topic_name = topic["topic_name"]

        # Skip topics that already have questions (prevents duplicate inserts on re-run)
        cursor.execute("SELECT COUNT(*) FROM questions WHERE topic_id = ?", (topic_id,))
        if cursor.fetchone()[0] > 0:
            continue

        questions = generate_questions_for_topic(
            topic_id, topic_name, text, all_topic_names, questions_per_topic
        )

        for q in questions:
            q_fp = _fingerprint(q["question_text"])
            if q_fp in global_q_fps:
                continue
            global_q_fps.add(q_fp)

            cursor.execute(
                "INSERT INTO questions "
                "(topic_id, question_text, option_a, option_b, option_c, option_d, correct_option) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (topic_id, q["question_text"], q["option_a"], q["option_b"],
                 q["option_c"], q["option_d"], q["correct_option"])
            )

    conn.commit()
    conn.close()


def get_questions_for_topic(topic_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id as question_id, question_text, "
        "option_a, option_b, option_c, option_d, correct_option "
        "FROM questions WHERE topic_id = ?",
        (topic_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]