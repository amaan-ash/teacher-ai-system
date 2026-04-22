import os
from datetime import datetime
from database.db import get_connection


# ─────────────────────────────────────────────
# SESSION MANAGEMENT
# A "session" = one student attempting one topic's quiz.
# We track sessions in memory (dict) during runtime.
# Persisted data goes to student_attempts table.
# ─────────────────────────────────────────────

# In-memory session store
# Structure:
# {
#   session_id: {
#       student_name: str,
#       curriculum_id: int,
#       topic_id: int,
#       question_ids: [int],
#       current_index: int,
#       answers: { question_id: selected_option },
#       started_at: str
#   }
# }
_SESSIONS: dict = {}


def _generate_session_id(student_name: str, topic_id: int) -> str:
    """
    Generates a simple unique session ID.
    Format: studentname_topicid_timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = student_name.strip().lower().replace(" ", "_")
    return f"{safe_name}_{topic_id}_{timestamp}"


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _fetch_questions_for_topic(topic_id: int) -> list[dict]:
    """
    Fetches all questions for a topic from the DB.

    Returns:
        List of question dicts with all MCQ fields.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            id          as question_id,
            question_text,
            option_a,
            option_b,
            option_c,
            option_d,
            correct_option
        FROM questions
        WHERE topic_id = ?
        ORDER BY id ASC
    """, (topic_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def _fetch_single_question(question_id: int) -> dict | None:
    """
    Fetches a single question by ID.
    Used when serving individual questions during a session.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            id          as question_id,
            question_text,
            option_a,
            option_b,
            option_c,
            option_d,
            correct_option
        FROM questions
        WHERE id = ?
    """, (question_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def _save_attempt(
    student_name: str,
    question_id: int,
    selected_option: str,
    is_correct: bool
) -> int:
    """
    Saves a single answer attempt to the DB.

    Args:
        student_name:    Name of the student.
        question_id:     ID of the question answered.
        selected_option: The option chosen ("A", "B", "C", or "D").
        is_correct:      Whether the answer was correct.

    Returns:
        attempt_id: Auto-generated DB row ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO student_attempts
            (student_name, question_id, selected_option, is_correct)
        VALUES (?, ?, ?, ?)
    """, (
        student_name.strip(),
        question_id,
        selected_option.upper(),
        1 if is_correct else 0
    ))
    conn.commit()
    attempt_id = cursor.lastrowid
    conn.close()
    return attempt_id


def _validate_option(option: str) -> bool:
    """Checks that the student's selected option is A, B, C, or D."""
    return option.strip().upper() in {"A", "B", "C", "D"}


# ─────────────────────────────────────────────
# SESSION FUNCTIONS
# ─────────────────────────────────────────────

def start_quiz_session(
    student_name: str,
    curriculum_id: int,
    topic_id: int
) -> dict:
    """
    Starts a new quiz session for a student on a specific topic.

    Checks:
    - Topic must have at least 1 question
    - Student name must be non-empty

    Args:
        student_name:  Name of the student.
        curriculum_id: Curriculum being quizzed.
        topic_id:      Topic to quiz on.

    Returns:
        {
            session_id: str,
            student_name: str,
            topic_id: int,
            total_questions: int,
            first_question: dict  (the first MCQ, without correct_option)
        }
    """
    student_name = student_name.strip()
    if not student_name:
        raise ValueError("Student name cannot be empty.")

    questions = _fetch_questions_for_topic(topic_id)
    if not questions:
        raise ValueError(
            f"No questions found for topic_id={topic_id}. "
            "Run quiz generation first."
        )

    session_id = _generate_session_id(student_name, topic_id)

    _SESSIONS[session_id] = {
        "student_name":  student_name,
        "curriculum_id": curriculum_id,
        "topic_id":      topic_id,
        "question_ids":  [q["question_id"] for q in questions],
        "current_index": 0,
        "answers":       {},   # question_id → selected_option
        "started_at":    datetime.now().isoformat()
    }

    # Return first question — without exposing correct_option to student
    first_q = questions[0].copy()
    first_q.pop("correct_option")

    print(f"[Quiz Engine] Session started: {session_id} | "
          f"Student: {student_name} | Questions: {len(questions)}")

    return {
        "session_id":      session_id,
        "student_name":    student_name,
        "topic_id":        topic_id,
        "total_questions": len(questions),
        "current_index":   0,
        "first_question":  first_q
    }


def submit_answer(session_id: str, question_id: int, selected_option: str) -> dict:
    """
    Submits an answer for the current question in the session.

    Validates:
    - Session must exist
    - Question must belong to this session
    - Selected option must be A/B/C/D
    - Question must not already be answered

    Returns:
        {
            is_correct: bool,
            correct_option: str,
            next_question: dict | None   (None if quiz is complete)
            session_complete: bool
        }
    """
    # ── Validate session ──
    if session_id not in _SESSIONS:
        raise ValueError(f"Session '{session_id}' not found or expired.")

    session = _SESSIONS[session_id]

    # ── Validate question belongs to this session ──
    if question_id not in session["question_ids"]:
        raise ValueError(
            f"question_id={question_id} does not belong to session '{session_id}'."
        )

    # ── Validate option ──
    if not _validate_option(selected_option):
        raise ValueError(
            f"Invalid option '{selected_option}'. Must be A, B, C, or D."
        )

    # ── Prevent re-answering ──
    if question_id in session["answers"]:
        raise ValueError(
            f"question_id={question_id} has already been answered in this session."
        )

    # ── Fetch full question to check correct answer ──
    question = _fetch_single_question(question_id)
    if not question:
        raise ValueError(f"question_id={question_id} not found in DB.")

    selected_option = selected_option.strip().upper()
    is_correct = (selected_option == question["correct_option"])

    # ── Save attempt to DB ──
    _save_attempt(
        session["student_name"],
        question_id,
        selected_option,
        is_correct
    )

    # ── Update session state ──
    session["answers"][question_id] = selected_option
    session["current_index"] += 1

    # ── Determine next question ──
    next_question = None
    session_complete = False

    if session["current_index"] < len(session["question_ids"]):
        next_qid = session["question_ids"][session["current_index"]]
        next_q_full = _fetch_single_question(next_qid)
        if next_q_full:
            next_question = next_q_full.copy()
            next_question.pop("correct_option")   # Never expose to student
    else:
        session_complete = True
        print(f"[Quiz Engine] Session complete: {session_id}")

    return {
        "is_correct":       is_correct,
        "correct_option":   question["correct_option"],
        "correct_text":     question[f"option_{question['correct_option'].lower()}"],
        "next_question":    next_question,
        "session_complete": session_complete,
        "current_index":    session["current_index"]
    }


def get_session_result(session_id: str) -> dict:
    """
    Returns the final result of a completed quiz session.

    Calculates:
    - Total questions attempted
    - Correct answers
    - Score percentage
    - Per-question breakdown

    Args:
        session_id: The session to summarize.

    Returns:
        {
            student_name, topic_id, total, correct,
            score_percent, breakdown: [...]
        }
    """
    if session_id not in _SESSIONS:
        raise ValueError(f"Session '{session_id}' not found.")

    session = _SESSIONS[session_id]
    student_name = session["student_name"]
    topic_id = session["topic_id"]

    conn = get_connection()
    cursor = conn.cursor()

    # Fetch all attempts for this student + topic combination
    cursor.execute("""
        SELECT
            sa.question_id,
            sa.selected_option,
            sa.is_correct,
            q.question_text,
            q.correct_option,
            q.option_a, q.option_b, q.option_c, q.option_d
        FROM student_attempts sa
        JOIN questions q ON sa.question_id = q.id
        WHERE sa.student_name = ?
          AND q.topic_id = ?
        ORDER BY sa.attempted_at DESC
    """, (student_name, topic_id))

    rows = cursor.fetchall()
    conn.close()

    # De-duplicate: keep only latest attempt per question
    seen = set()
    breakdown = []
    for row in rows:
        if row["question_id"] not in seen:
            seen.add(row["question_id"])
            breakdown.append({
                "question_id":    row["question_id"],
                "question_text":  row["question_text"],
                "selected":       row["selected_option"],
                "correct":        row["correct_option"],
                "is_correct":     bool(row["is_correct"]),
                "selected_text":  row[f"option_{row['selected_option'].lower()}"],
                "correct_text":   row[f"option_{row['correct_option'].lower()}"]
            })

    total   = len(breakdown)
    correct = sum(1 for b in breakdown if b["is_correct"])
    score_percent = round((correct / total) * 100, 1) if total > 0 else 0.0

    return {
        "session_id":    session_id,
        "student_name":  student_name,
        "topic_id":      topic_id,
        "total":         total,
        "correct":       correct,
        "wrong":         total - correct,
        "score_percent": score_percent,
        "breakdown":     breakdown
    }


def end_session(session_id: str):
    """
    Clears the session from memory after it's complete.
    Call this after get_session_result() is done.
    """
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]
        print(f"[Quiz Engine] Session cleared: {session_id}")


# ─────────────────────────────────────────────
# BULK FETCH — Used by Analytics (Step 7)
# ─────────────────────────────────────────────

def get_all_attempts_for_curriculum(curriculum_id: int) -> list[dict]:
    """
    Fetches every student attempt for all topics in a curriculum.
    This is the raw data feed for the analytics + heatmap module.

    Returns:
        List of attempt dicts with topic_id, student_name,
        question_id, is_correct, selected_option.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            sa.id           as attempt_id,
            sa.student_name,
            sa.question_id,
            sa.selected_option,
            sa.is_correct,
            sa.attempted_at,
            q.topic_id,
            t.topic_name
        FROM student_attempts sa
        JOIN questions q  ON sa.question_id = q.id
        JOIN topics    t  ON q.topic_id     = t.id
        WHERE t.curriculum_id = ?
        ORDER BY sa.attempted_at ASC
    """, (curriculum_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_topic_score_summary(curriculum_id: int) -> list[dict]:
    """
    Returns per-topic score summary across all students.
    Used directly by analytics module for heatmap generation.

    Returns:
        List of dicts:
        {
            topic_id, topic_name,
            total_attempts, correct_attempts,
            accuracy_percent, struggle_score
        }
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            t.id            as topic_id,
            t.topic_name,
            COUNT(sa.id)    as total_attempts,
            SUM(sa.is_correct) as correct_attempts
        FROM topics t
        LEFT JOIN questions q    ON q.topic_id = t.id
        LEFT JOIN student_attempts sa ON sa.question_id = q.id
        WHERE t.curriculum_id = ?
        GROUP BY t.id, t.topic_name
        ORDER BY t.order_index ASC
    """, (curriculum_id,))
    rows = cursor.fetchall()
    conn.close()

    summary = []
    for row in rows:
        total   = row["total_attempts"]   or 0
        correct = row["correct_attempts"] or 0
        accuracy = round((correct / total) * 100, 1) if total > 0 else None

        # struggle_score: 0.0 = no data, 1.0 = everyone got everything wrong
        struggle = round(1.0 - (correct / total), 3) if total > 0 else 0.0

        summary.append({
            "topic_id":        row["topic_id"],
            "topic_name":      row["topic_name"],
            "total_attempts":  total,
            "correct_attempts": correct,
            "accuracy_percent": accuracy,
            "struggle_score":  struggle    # Higher = more struggle
        })

    return summary