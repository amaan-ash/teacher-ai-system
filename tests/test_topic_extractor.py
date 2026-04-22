import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import initialize_db
from app.pdf_processor import process_pdf
from app.topic_extractor import extract_and_save_topics, get_topics_for_curriculum


def test_topic_extraction():
    initialize_db()

    # Step 1: Process the PDF first
    pdf_path = "sample_data/sample_curriculum.pdf"
    if not os.path.exists(pdf_path):
        print("[TEST] ERROR: Place a PDF at sample_data/sample_curriculum.pdf")
        return

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    pdf_result = process_pdf(file_bytes, "sample_curriculum.pdf")
    curriculum_id = pdf_result["curriculum_id"]

    # Step 2: Read extracted text
    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    # Step 3: Extract topics
    topics = extract_and_save_topics(curriculum_id, text, top_n=15)

    print("\n===== EXTRACTED TOPICS =====")
    for t in topics:
        print(f"  [{t['order_index']+1:02d}] {t['topic_name']}")

    # Step 4: Fetch back from DB to verify persistence
    fetched = get_topics_for_curriculum(curriculum_id)
    print(f"\n[TEST] Verified {len(fetched)} topics in DB.")
    print("============================")


if __name__ == "__main__":
    test_topic_extraction()