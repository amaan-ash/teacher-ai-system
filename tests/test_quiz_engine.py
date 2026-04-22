import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import initialize_db
from app.pdf_processor import process_pdf
from app.topic_extractor import extract_and_save_topics
from app.learning_path import build_learning_path
from app.quiz_generator import generate_and_save_quiz, get_questions_for_topic
from app.quiz_engine import (
    start_quiz_session,
    submit_answer,
    get_session_result,
    end_session,
    get_topic_score_summary
)


def simulate_student(student_name: str, curriculum_id: int, topic: dict):
    """Simulates one student attempting a quiz on one topic."""
    print(f"\n── Student: {student_name} | Topic: {topic['topic_name']} ──")

    session = start_quiz_session(student_name, curriculum_id, topic["topic_id"])
    session_id = session["session_id"]
    current_q  = session["first_question"]

    while current_q:
        # Simulate student picking a random option
        import random
        chosen = random.choice(["A", "B", "C", "D"])
        print(f"  Q: {current_q['question_text'][:60]}...")
        print(f"  Student picks: {chosen}")

        result = submit_answer(session_id, current_q["question_id"], chosen)
        status = "✓ Correct" if result["is_correct"] else "✗ Wrong"
        print(f"  {status} | Correct was: {result['correct_option']}")

        current_q = result["next_question"]

    # Get final result
    final = get_session_result(session_id)
    print(f"\n  Score: {final['correct']}/{final['total']} "
          f"({final['score_percent']}%)")
    end_session(session_id)
    return final


def test_quiz_engine():
    initialize_db()

    pdf_path = "sample_data/sample_curriculum.pdf"
    if not os.path.exists(pdf_path):
        print("[TEST] Place a PDF at sample_data/sample_curriculum.pdf")
        return

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # Full pipeline setup
    pdf_result    = process_pdf(file_bytes, "sample_curriculum.pdf")
    curriculum_id = pdf_result["curriculum_id"]

    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    topics_saved = extract_and_save_topics(curriculum_id, text, top_n=5)
    build_learning_path(curriculum_id)
    generate_and_save_quiz(curriculum_id, questions_per_topic=3)

    # Simulate 3 students on the first 2 topics
    from database.db import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id as topic_id, topic_name FROM topics
        WHERE curriculum_id = ? ORDER BY order_index ASC LIMIT 2
    """, (curriculum_id,))
    topics = [dict(r) for r in cursor.fetchall()]
    conn.close()

    students = ["Alice", "Bob", "Charlie"]
    for student in students:
        for topic in topics:
            simulate_student(student, curriculum_id, topic)

    # Print topic score summary
    print("\n===== TOPIC SCORE SUMMARY =====")
    summary = get_topic_score_summary(curriculum_id)
    for s in summary:
        acc = f"{s['accuracy_percent']}%" if s['accuracy_percent'] is not None else "No data"
        print(f"  {s['topic_name']:<35} | Accuracy: {acc:<8} | Struggle: {s['struggle_score']}")
    print("================================")


if __name__ == "__main__":
    test_quiz_engine()