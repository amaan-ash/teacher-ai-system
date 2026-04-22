import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import initialize_db
from app.pdf_processor import process_pdf
from app.topic_extractor import extract_and_save_topics
from app.learning_path import build_learning_path


def test_learning_path():
    initialize_db()

    pdf_path = "sample_data/sample_curriculum.pdf"
    if not os.path.exists(pdf_path):
        print("[TEST] ERROR: Place a PDF at sample_data/sample_curriculum.pdf")
        return

    # Step 1: Process PDF
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    pdf_result = process_pdf(file_bytes, "sample_curriculum.pdf")
    curriculum_id = pdf_result["curriculum_id"]

    # Step 2: Extract topics
    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()
    extract_and_save_topics(curriculum_id, text, top_n=15)

    # Step 3: Build learning path
    modules = build_learning_path(curriculum_id)

    print("\n===== LEARNING PATH =====")
    for module in modules:
        print(f"\n{module['module_name']}  (~{module['estimated_hours']} hrs)")
        print("  Topics:")
        for t in module["topics"]:
            score = round(__import__('app.learning_path', fromlist=['_compute_complexity_score'])
                          ._compute_complexity_score(t["topic_name"]), 2)
            print(f"    [{t['order_index']+1:02d}] {t['topic_name']}  (complexity: {score})")
    print("\n=========================")


if __name__ == "__main__":
    test_learning_path()