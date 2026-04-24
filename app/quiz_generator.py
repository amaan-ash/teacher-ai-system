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

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")


# ─────────────────────────────────────────────
# 🔥 NEW: SHORT ANSWER EXTRACTION
# ─────────────────────────────────────────────

def _extract_short_answer(sentence: str, topic: str) -> str:
    sentence = sentence.strip()

    patterns = [
        " is defined as ",
        " refers to ",
        " is a ",
        " is an ",
        " means ",
        " is the process of ",
    ]

    lower = sentence.lower()

    for p in patterns:
        if p in lower:
            parts = sentence.split(p, 1)
            if len(parts) > 1:
                answer = parts[1].strip()
                break
    else:
        answer = sentence

    answer = answer.strip().strip(".")

    if len(answer) > 80:
        answer = answer[:80].rsplit(" ", 1)[0]

    return answer


# ─────────────────────────────────────────────
# QUESTION TEMPLATES
# ─────────────────────────────────────────────

def _template_what_is(sentence, topic):
    if " is " in sentence.lower():
        return f"What is {topic}?", _extract_short_answer(sentence, topic)
    return None


def _template_purpose(sentence, topic):
    if any(k in sentence.lower() for k in ["used to", "helps", "allows", "purpose"]):
        return f"What is the purpose of {topic}?", _extract_short_answer(sentence, topic)
    return None


def _template_fill_blank(sentence, topic):
    if topic.lower() not in sentence.lower():
        return None

    blanked = re.sub(re.escape(topic), "_____", sentence, flags=re.IGNORECASE, count=1)
    return f"Fill in the blank: {blanked}", topic


TEMPLATES = [
    _template_what_is,
    _template_purpose,
    _template_fill_blank
]


# ─────────────────────────────────────────────
# SENTENCE SELECTION
# ─────────────────────────────────────────────

def _get_topic_relevant_sentences(text, topic, top_n=20):
    sentences = sent_tokenize(text)

    sentences = [
        s.strip() for s in sentences
        if 40 < len(s.strip()) < 200
        and not s.strip().endswith("?")
    ]

    topic_words = set(topic.lower().split())

    matches = []
    for s in sentences:
        words = set(s.lower().split())
        if topic_words & words:
            matches.append(s)

    return matches[:top_n]


# ─────────────────────────────────────────────
# 🔥 IMPROVED DISTRACTORS
# ─────────────────────────────────────────────

def _generate_distractors(correct_answer, all_topics, all_sentences, count=3):

    distractors = []
    correct_lower = correct_answer.lower()

    # 1. Topic-based distractors
    topic_pool = [
        t for t in all_topics
        if t.lower() != correct_lower and len(t) > 3
    ]
    random.shuffle(topic_pool)

    for t in topic_pool:
        if len(distractors) >= count:
            break
        distractors.append(t)

    # 2. Noun phrase fallback
    if len(distractors) < count:
        for sent in all_sentences:
            doc = NLP(sent[:150])
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().title()

                if (
                    phrase.lower() != correct_lower
                    and phrase not in distractors
                    and 5 < len(phrase) < 60
                ):
                    distractors.append(phrase)

                if len(distractors) >= count:
                    break

            if len(distractors) >= count:
                break

    return distractors[:count]


# ─────────────────────────────────────────────
# BUILD MCQ
# ─────────────────────────────────────────────

def _build_mcq(question, answer, distractors):
    options = distractors[:3] + [answer]
    random.shuffle(options)

    labels = ["A", "B", "C", "D"]
    correct = labels[options.index(answer)]

    return {
        "question_text": question,
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
        "correct_option": correct
    }


# ─────────────────────────────────────────────
# CORE GENERATION
# ─────────────────────────────────────────────

def generate_questions_for_topic(topic_id, topic_name, text, all_topics, questions_per_topic=3):

    sentences = _get_topic_relevant_sentences(text, topic_name)

    questions = []
    used_sentences = set()
    used_questions = set()

    for sentence in sentences:

        if len(questions) >= questions_per_topic:
            break

        if sentence.lower() in used_sentences:
            continue

        for template in TEMPLATES:
            result = template(sentence, topic_name)
            if not result:
                continue

            question, answer = result

            if len(answer) < 5:
                continue

            if question.lower() in used_questions:
                continue

            distractors = _generate_distractors(answer, all_topics, sentences)

            if len(distractors) < 3:
                continue

            mcq = _build_mcq(question, answer, distractors)

            questions.append(mcq)
            used_sentences.add(sentence.lower())
            used_questions.add(question.lower())

            break

    return questions


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def generate_and_save_quiz(curriculum_id, questions_per_topic=3):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id as topic_id, topic_name
        FROM topics
        WHERE curriculum_id = ?
    """, (curriculum_id,))
    topics = cursor.fetchall()

    cursor.execute("""
        SELECT extracted_text_path FROM curricula WHERE id = ?
    """, (curriculum_id,))
    row = cursor.fetchone()
    conn.close()

    with open(row["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    all_topic_names = [t["topic_name"] for t in topics]

    conn = get_connection()
    cursor = conn.cursor()

    for topic in topics:
        topic_id = topic["topic_id"]
        topic_name = topic["topic_name"]

        questions = generate_questions_for_topic(
            topic_id,
            topic_name,
            text,
            all_topic_names,
            questions_per_topic
        )

        for q in questions:
            cursor.execute("""
                INSERT INTO questions
                (topic_id, question_text, option_a, option_b, option_c, option_d, correct_option)
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

    conn.commit()
    conn.close()


def get_questions_for_topic(topic_id):
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