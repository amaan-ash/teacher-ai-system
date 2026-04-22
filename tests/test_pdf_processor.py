import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import initialize_db
from app.pdf_processor import process_pdf

def test_pdf_processing():
    initialize_db()

    # Load a real PDF from sample_data/
    pdf_path = "sample_data/sample_curriculum.pdf"

    if not os.path.exists(pdf_path):
        print("[TEST] ERROR: Place a sample PDF at sample_data/sample_curriculum.pdf")
        return

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    result = process_pdf(file_bytes, "sample_curriculum.pdf")

    print("\n===== TEST RESULT =====")
    print(f"Curriculum ID : {result['curriculum_id']}")
    print(f"Upload Path   : {result['upload_path']}")
    print(f"Extracted Path: {result['extracted_text_path']}")
    print(f"Char Count    : {result['char_count']}")
    print(f"\nText Preview  :\n{result['preview']}")
    print("=======================")

if __name__ == "__main__":
    test_pdf_processing()