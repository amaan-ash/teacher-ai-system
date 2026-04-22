import os
import re
import random
import sqlite3
import spacy
import nltk

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from database.db import get_connection

load_dotenv()

nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")

EXTRACTED_DIR = os.getenv("EXTRACTED_DIR", "data/extracted")


# ─────────────────────────────────────────────
# QUESTION TEMPLATES
# Each template is a function:
#   input  → a key sentence (str)
#   output → (question_text, correct_answer) or None if template doesn't fit
# ─────────────────────────────────────────────

def _template_what_is(sentence: str, topic: str):
    """
    Template: "What is <topic>?"
    Answer  : The sentence itself (trimmed to answer length).
    Trigger : Sentence contains a definition signal word.
    """
    definition_signals = [
        "is defined as", "refers to", "is a", "is an",
        "can be defined", "means", "is the process",
        "is used to", "is known as"
    ]
    s_lower = sentence.lower()
    for signal in definition_signals:
        if signal in s_lower:
            question = f"What is {topic}?"
            answer = sentence.strip()
            if len(answer) > 120:
                answer = answer[:120].rsplit(" ", 1)[0] + "..."
            return question, answer
    return None


def _template_which_of(sentence: str, topic: str):
    """
    Template: "Which of the following is true about <topic>?"
    Answer  : The sentence itself.
    Trigger : Sentence contains factual assertion words.
    """
    factual_signals = [
        "always", "never", "must", "cannot", "only",
        "all", "every", "none", "ensures", "guarantees",
        "requires", "produces", "returns", "stores"
    ]
    s_lower = sentence.lower()
    for signal in factual_signals:
        if signal in s_lower:
            question = f"Which of the following is true about {topic}?"
            answer = sentence.strip()
            if len(answer) > 120:
                answer = answer[:120].rsplit(" ", 1)[0] + "..."
            return question, answer
    return None


def _template_purpose(sentence: str, topic: str):
    """
    Template: "What is the primary purpose of <topic>?"
    Answer  : The sentence.
    Trigger : Sentence contains purpose/function signal words.
    """
    purpose_signals = [
        "used to", "used for", "purpose of", "designed to",
        "helps to", "allows", "enables", "provides",
        "responsible for", "goal is", "aim is", "objective"
    ]
    s_lower = sentence.lower()
    for signal in purpose_signals:
        if signal in s_lower:
            question = f"What is the primary purpose of {topic}?"
            answer = sentence.strip()
            if len(answer) > 120:
                answer = answer[:120].rsplit(" ", 1)[0] + "..."
            return question, answer
    return None


def _template_fill_blank(sentence: str, topic: str):
    """
    Template: Fill-in-the-blank style converted to MCQ.
    We blank out the topic name from the sentence.
    Trigger : Topic name appears directly in the sentence.
    """
    if topic.lower() not in sentence.lower():
        return None

    # Create blanked sentence — replace topic with _____
    pattern = re.compile(re.escape(topic), re.IGNORECASE)
    blanked = pattern.sub("_____", sentence.strip(), count=1)

    if "_____" not in blanked:
        return None

    question = f"Fill in the blank: {blanked}"
    answer = topic
    return question, answer


TEMPLATES = [
    _template_what_is,
    _template_purpose,
    _template_which_of,
    _template_fill_blank,
]


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _get_topic_relevant_sentences(
    full_text: str,
    topic_name: str,
    top_n: int = 20
) -> list[str]:
    """
    Extracts sentences most relevant to a topic using two passes:

    Pass 1 — Keyword match:
        Sentences containing the topic name or its words directly.

    Pass 2 — TF-IDF cosine similarity:
        If Pass 1 gives fewer than 5 sentences, we compute TF-IDF
        similarity between the topic phrase and all sentences,
        picking the top matches.

    Why two passes:
        Keyword match is precise but misses paraphrased content.
        TF-IDF similarity catches related sentences even without
        exact topic word presence.
    """
    sentences = sent_tokenize(full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    topic_words = set(topic_name.lower().split())

    # Pass 1: Direct keyword match
    keyword_matches = []
    for s in sentences:
        s_words = set(s.lower().split())
        overlap = topic_words & s_words
        if len(overlap) >= max(1, len(topic_words) // 2):
            keyword_matches.append(s)

    if len(keyword_matches) >= 5:
        return keyword_matches[:top_n]

    # Pass 2: TF-IDF cosine similarity fallback
    if len(sentences) < 2:
        return keyword_matches

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_df=1.0, min_df=1)
        corpus = [topic_name] + sentences
        tfidf_matrix = vectorizer.fit_transform(corpus)

        topic_vec = tfidf_matrix[0]
        sentence_vecs = tfidf_matrix[1:]

        similarities = cosine_similarity(topic_vec, sentence_vecs).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]

        similarity_matches = [sentences[i] for i in top_indices if similarities[i] > 0.05]
        combined = keyword_matches + [s for s in similarity_matches if s not in keyword_matches]
        return combined[:top_n]

    except ValueError:
        return keyword_matches[:top_n]


def _score_sentence_for_question(sentence: str) -> float:
    """
    Scores a sentence on how suitable it is for question generation.

    Good question sentences:
    - Are declarative (not questions themselves)
    - Have subject-verb-object structure
    - Are not too short or too long
    - Don't start with pronouns (ambiguous reference)
    - Contain at least one content word

    Returns float score. Higher = better candidate.
    """
    score = 0.0
    s = sentence.strip()

    # Length sweet spot: 60–200 chars
    length = len(s)
    if 60 <= length <= 200:
        score += 0.3
    elif length < 40 or length > 300:
        score -= 0.3

    # Not already a question
    if s.endswith("?"):
        return 0.0

    # Doesn't start with pronoun
    first_word = s.split()[0].lower() if s.split() else ""
    if first_word in ["it", "this", "that", "they", "he", "she", "its"]:
        score -= 0.2

    # Contains a verb (basic check via spaCy POS)
    doc = NLP(s[:200])  # Limit to 200 chars for speed
    has_verb = any(token.pos_ == "VERB" for token in doc)
    has_noun = any(token.pos_ == "NOUN" for token in doc)

    if has_verb:
        score += 0.3
    if has_noun:
        score += 0.2

    # Penalize sentences with too many numbers (likely stats/tables)
    num_count = len(re.findall(r'\b\d+\b', s))
    if num_count > 3:
        score -= 0.2

    return score


def _generate_distractors(
    correct_answer: str,
    all_topics: list[str],
    all_sentences: list[str],
    count: int = 3
) -> list[str]:
    """
    Generates wrong answer options (distractors) for MCQ.

    Strategy (in priority order):
    1. Other topic names from the curriculum (most relevant distractors)
    2. Sentence fragments from unrelated sentences
    3. Generic fallback placeholders

    We ensure distractors:
    - Are not the correct answer
    - Are not duplicates of each other
    - Are roughly similar in length to the correct answer
    """
    distractors = []
    correct_lower = correct_answer.lower().strip()

    # Strategy 1: Other topic names
    topic_distractors = [
        t for t in all_topics
        if t.lower().strip() != correct_lower and len(t) > 3
    ]
    random.shuffle(topic_distractors)

    for t in topic_distractors:
        if len(distractors) >= count:
            break
        if t not in distractors:
            distractors.append(t)

    # Strategy 2: Sentence fragments
    if len(distractors) < count:
        random.shuffle(all_sentences)
        for sent in all_sentences:
            if len(distractors) >= count:
                break
            # Extract a noun phrase fragment
            doc = NLP(sent[:200])
            for chunk in doc.noun_chunks:
                fragment = chunk.text.strip().title()
                if (
                    fragment.lower() != correct_lower
                    and fragment not in distractors
                    and len(fragment) > 5
                    and len(fragment) < 100
                ):
                    distractors.append(fragment)
                    break

    # Strategy 3: Generic fallbacks
    fallbacks = [
        "None of the above",
        "All of the above",
        "It depends on context",
        "Cannot be determined"
    ]
    for fb in fallbacks:
        if len(distractors) >= count:
            break
        if fb not in distractors:
            distractors.append(fb)

    return distractors[:count]


def _build_mcq(
    question_text: str,
    correct_answer: str,
    distractors: list[str]
) -> dict:
    """
    Assembles a complete MCQ dict with shuffled options.

    Returns:
        {
            question_text, option_a, option_b, option_c, option_d,
            correct_option  (one of "A", "B", "C", "D")
        }
    """
    options = distractors[:3] + [correct_answer]
    random.shuffle(options)

    option_labels = ["A", "B", "C", "D"]
    correct_label = option_labels[options.index(correct_answer)]

    return {
        "question_text": question_text,
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
        "correct_option": correct_label
    }


# ─────────────────────────────────────────────
# CORE GENERATION FUNCTION
# ─────────────────────────────────────────────

def generate_questions_for_topic(
    topic_id: int,
    topic_name: str,
    full_text: str,
    all_topic_names: list[str],
    questions_per_topic: int = 3
) -> list[dict]:
    """
    Generates MCQ questions for a single topic.

    Pipeline:
    1. Find sentences relevant to this topic
    2. Score and rank those sentences
    3. Apply templates to top sentences
    4. Generate distractors
    5. Assemble MCQ

    Args:
        topic_id:           DB ID of the topic.
        topic_name:         Name of the topic.
        full_text:          Full curriculum text.
        all_topic_names:    All topics (used for distractor generation).
        questions_per_topic: Target number of questions.

    Returns:
        List of MCQ dicts.
    """
    relevant_sentences = _get_topic_relevant_sentences(full_text, topic_name, top_n=30)

    if not relevant_sentences:
        print(f"[Quiz Generator] No relevant sentences found for: {topic_name}")
        return []

    # Score and sort sentences
    scored = [
        (sent, _score_sentence_for_question(sent))
        for sent in relevant_sentences
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s for s, score in scored if score > 0.0]

    questions = []
    used_sentences = set()

    for sentence, _ in scored:
        if len(questions) >= questions_per_topic:
            break
        if sentence in used_sentences:
            continue

        # Try each template until one works
        for template_fn in TEMPLATES:
            result = template_fn(sentence, topic_name)
            if result is None:
                continue

            question_text, correct_answer = result

            # Skip if correct answer is too short or too long
            if len(correct_answer) < 4 or len(correct_answer) > 200:
                continue

            distractors = _generate_distractors(
                correct_answer,
                all_topic_names,
                top_sentences,
                count=3
            )

            if len(distractors) < 3:
                continue

            mcq = _build_mcq(question_text, correct_answer, distractors)
            questions.append(mcq)
            used_sentences.add(sentence)
            break   # One question per sentence

    return questions


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def generate_and_save_quiz(curriculum_id: int, questions_per_topic: int = 3) -> list[dict]:
    """
    Full pipeline:
    1. Fetch all topics for the curriculum
    2. Load extracted text from disk
    3. Generate questions for each topic
    4. Save to DB
    5. Return all questions grouped by topic

    Args:
        curriculum_id:       ID of the curriculum.
        questions_per_topic: Number of MCQs per topic.

    Returns:
        List of dicts: { topic_name, questions: [mcq, ...] }
    """
    print(f"\n[Quiz Generator] Starting quiz generation for curriculum_id={curriculum_id}")

    conn = get_connection()
    cursor = conn.cursor()

    # Fetch topics
    cursor.execute("""
        SELECT id as topic_id, topic_name
        FROM topics
        WHERE curriculum_id = ?
        ORDER BY order_index ASC
    """, (curriculum_id,))
    topics = [{"topic_id": row["topic_id"], "topic_name": row["topic_name"]}
              for row in cursor.fetchall()]

    # Fetch extracted text path
    cursor.execute("""
        SELECT extracted_text_path FROM curricula WHERE id = ?
    """, (curriculum_id,))
    row = cursor.fetchone()
    conn.close()

    if not topics:
        raise ValueError(f"No topics found for curriculum_id={curriculum_id}")
    if not row or not row["extracted_text_path"]:
        raise ValueError(f"No extracted text found for curriculum_id={curriculum_id}")

    with open(row["extracted_text_path"], "r", encoding="utf-8") as f:
        full_text = f.read()

    all_topic_names = [t["topic_name"] for t in topics]
    all_results = []

    conn = get_connection()
    cursor = conn.cursor()

    for topic in topics:
        topic_id   = topic["topic_id"]
        topic_name = topic["topic_name"]

        print(f"[Quiz Generator] Generating for: {topic_name}")

        questions = generate_questions_for_topic(
            topic_id,
            topic_name,
            full_text,
            all_topic_names,
            questions_per_topic
        )

        saved_questions = []
        for q in questions:
            cursor.execute("""
                INSERT INTO questions
                    (topic_id, question_text, option_a, option_b,
                     option_c, option_d, correct_option)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                topic_id,
                q["question_text"],
                q["option_a"],
                q["option_b"],
                q["option_c"],
                q["option_d"],
                q["correct_option"]
            ))
            q["question_id"] = cursor.lastrowid
            saved_questions.append(q)

        all_results.append({
            "topic_name": topic_name,
            "topic_id": topic_id,
            "questions": saved_questions
        })

    conn.commit()
    conn.close()

    total_q = sum(len(r["questions"]) for r in all_results)
    print(f"[Quiz Generator] Done. {total_q} questions saved across {len(topics)} topics.")
    return all_results


def get_questions_for_topic(topic_id: int) -> list[dict]:
    """
    Fetches saved questions for a topic from DB.
    Used by quiz engine (Step 6) and dashboard (Step 8).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id as question_id, question_text,
               option_a, option_b, option_c, option_d, correct_option
        FROM questions
        WHERE topic_id = ?
    """, (topic_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]