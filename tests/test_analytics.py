import os
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import initialize_db
from app.pdf_processor import process_pdf
from app.topic_extractor import extract_and_save_topics
from app.learning_path import build_learning_path
from app.quiz_generator import generate_and_save_quiz
from app.quiz_engine import (
    start_quiz_session, submit_answer,
    get_session_result, end_session
)
from app.analytics import run_full_analytics


def simulate_students(curriculum_id: int, student_names: list[str]):
    """Simulates multiple students answering all topic quizzes."""
    from database.db import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id as topic_id, topic_name FROM topics
        WHERE curriculum_id = ? ORDER BY order_index ASC
    """, (curriculum_id,))
    topics = [dict(r) for r in cursor.fetchall()]
    conn.close()

    for student in student_names:
        for topic in topics:
            try:
                session = start_quiz_session(student, curriculum_id, topic["topic_id"])
                session_id = session["session_id"]
                current_q  = session["first_question"]

                while current_q:
                    # Bias: some students are weaker on some topics
                    # to make the heatmap more visually interesting
                    weight = random.random()
                    if weight < 0.35:
                        chosen = current_q.get("correct_option_hint", random.choice(["A","B","C","D"]))
                    else:
                        chosen = random.choice(["A","B","C","D"])

                    result = submit_answer(session_id, current_q["question_id"], chosen)
                    current_q = result["next_question"]

                get_session_result(session_id)
                end_session(session_id)

            except ValueError as e:
                print(f"  [Sim] Skipped {student}/{topic['topic_name']}: {e}")


def test_analytics():
    initialize_db()

    pdf_path = "sample_data/sample_curriculum.pdf"
    if not os.path.exists(pdf_path):
        print("[TEST] Place a PDF at sample_data/sample_curriculum.pdf")
        return

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # Full pipeline
    pdf_result    = process_pdf(file_bytes, "sample_curriculum.pdf")
    curriculum_id = pdf_result["curriculum_id"]

    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    extract_and_save_topics(curriculum_id, text, top_n=8)
    build_learning_path(curriculum_id)
    generate_and_save_quiz(curriculum_id, questions_per_topic=3)

    # Simulate 5 students
    students = ["Alice", "Bob", "Charlie", "Diana", "Evan"]
    print(f"\n[TEST] Simulating {len(students)} students...")
    simulate_students(curriculum_id, students)

    # Run analytics
    results = run_full_analytics(curriculum_id)

    print("\n===== ANALYTICS RESULTS =====")
    print(f"Heatmap       : {results['heatmap_path']}")
    print(f"Accuracy Chart: {results['accuracy_chart_path']}")
    print(f"Leaderboard   : {results['leaderboard_path']}")

    print("\nWeak Topic Report:")
    for item in results["weak_topic_report"]:
        print(
            f"  #{item['rank']} {item['topic_name']:<35} "
            f"Accuracy: {item['accuracy_percent']}%  "
            f"{item['severity']}"
        )
    print("=============================")


if __name__ == "__main__":
    test_analytics()