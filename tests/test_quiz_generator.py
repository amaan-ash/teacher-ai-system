import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import initialize_db
from app.pdf_processor import process_pdf
from app.topic_extractor import extract_and_save_topics
from app.learning_path import build_learning_path
from app.quiz_generator import generate_and_save_quiz


def test_quiz_generation():
    initialize_db()

    pdf_path = "sample_data/sample_curriculum.pdf"
    if not os.path.exists(pdf_path):
        print("[TEST] Place a PDF at sample_data/sample_curriculum.pdf")
        return

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # Full pipeline
    pdf_result   = process_pdf(file_bytes, "sample_curriculum.pdf")
    curriculum_id = pdf_result["curriculum_id"]

    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    extract_and_save_topics(curriculum_id, text, top_n=15)
    build_learning_path(curriculum_id)
    results = generate_and_save_quiz(curriculum_id, questions_per_topic=3)

    print("\n===== GENERATED QUIZ =====")
    for topic_result in results:
        print(f"\nTopic: {topic_result['topic_name']}")
        for i, q in enumerate(topic_result["questions"], 1):
            print(f"  Q{i}: {q['question_text']}")
            print(f"       A. {q['option_a']}")
            print(f"       B. {q['option_b']}")
            print(f"       C. {q['option_c']}")
            print(f"       D. {q['option_d']}")
            print(f"       Correct: {q['correct_option']}")
    print("\n==========================")


if __name__ == "__main__":
    test_quiz_generation()