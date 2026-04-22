import os
import re
import PyPDF2
from dotenv import load_dotenv
from database.db import get_connection

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
EXTRACTED_DIR = os.getenv("EXTRACTED_DIR", "data/extracted")


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _ensure_dirs():
    """Create upload and extracted directories if they don't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)


def _clean_text(raw_text: str) -> str:
    """
    Cleans raw PDF text:
    - Removes non-ASCII garbage characters
    - Collapses multiple blank lines into one
    - Strips leading/trailing whitespace per line
    - Removes lines that are pure noise (page numbers, single chars)
    """
    # Remove non-ASCII characters (common in scanned PDFs)
    text = raw_text.encode("ascii", errors="ignore").decode("ascii")

    # Normalize whitespace within lines
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines, single characters, and likely page numbers
        if not line:
            continue
        if len(line) <= 2:
            continue
        if re.fullmatch(r'\d+', line):   # Pure page numbers like "1", "23"
            continue

        cleaned_lines.append(line)

    # Join with newlines, collapse 3+ consecutive newlines to 2
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def save_uploaded_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Saves raw PDF bytes to the uploads directory.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename:   Original filename from the upload.

    Returns:
        Full path where the PDF was saved.
    """
    _ensure_dirs()

    # Sanitize filename — remove spaces and special characters
    safe_name = re.sub(r'[^\w\-.]', '_', filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    print(f"[PDF Processor] PDF saved to: {save_path}")
    return save_path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and cleans text from a PDF file page by page.

    Args:
        pdf_path: Path to the saved PDF file.

    Returns:
        Cleaned text string extracted from all pages.

    Raises:
        FileNotFoundError: If the PDF doesn't exist at given path.
        ValueError: If the PDF has no extractable text (e.g. scanned image PDF).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    raw_pages = []

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total_pages = len(reader.pages)

        if total_pages == 0:
            raise ValueError("PDF has no pages.")

        print(f"[PDF Processor] Extracting text from {total_pages} page(s)...")

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    raw_pages.append(page_text)
            except Exception as e:
                print(f"[PDF Processor] Warning: Could not extract page {i+1} — {e}")
                continue

    if not raw_pages:
        raise ValueError(
            "No text could be extracted. "
            "This may be a scanned/image-based PDF. "
            "OCR support is not included in this version."
        )

    raw_text = "\n".join(raw_pages)
    clean_text = _clean_text(raw_text)

    print(f"[PDF Processor] Extraction complete. {len(clean_text)} characters extracted.")
    return clean_text


def save_extracted_text(text: str, original_filename: str) -> str:
    """
    Saves the extracted text to the extracted/ directory as a .txt file.

    Args:
        text:              Cleaned text content.
        original_filename: Original PDF filename (used to name the .txt file).

    Returns:
        Path to the saved .txt file.
    """
    _ensure_dirs()

    base_name = os.path.splitext(original_filename)[0]
    safe_name = re.sub(r'[^\w\-]', '_', base_name)
    txt_path = os.path.join(EXTRACTED_DIR, f"{safe_name}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[PDF Processor] Extracted text saved to: {txt_path}")
    return txt_path


def log_curriculum_to_db(filename: str, upload_path: str, extracted_text_path: str) -> int:
    """
    Logs a processed curriculum into the SQLite database.

    Args:
        filename:             Original filename.
        upload_path:          Where the PDF is stored.
        extracted_text_path:  Where the .txt file is stored.

    Returns:
        curriculum_id: The auto-generated DB row ID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO curricula (filename, upload_path, extracted_text_path)
        VALUES (?, ?, ?)
    """, (filename, upload_path, extracted_text_path))

    conn.commit()
    curriculum_id = cursor.lastrowid
    conn.close()

    print(f"[PDF Processor] Curriculum logged to DB with ID: {curriculum_id}")
    return curriculum_id


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def process_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline: save PDF → extract text → save text → log to DB.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename from upload.

    Returns:
        A dict with curriculum_id, upload_path, extracted_text_path, and preview.
    """
    # Step 1: Save the raw PDF
    upload_path = save_uploaded_pdf(file_bytes, filename)

    # Step 2: Extract and clean text
    extracted_text = extract_text_from_pdf(upload_path)

    # Step 3: Save the cleaned text to disk
    extracted_text_path = save_extracted_text(extracted_text, filename)

    # Step 4: Log to database
    curriculum_id = log_curriculum_to_db(filename, upload_path, extracted_text_path)

    return {
        "curriculum_id": curriculum_id,
        "upload_path": upload_path,
        "extracted_text_path": extracted_text_path,
        "char_count": len(extracted_text),
        "preview": extracted_text[:500]   # First 500 chars for UI display
    }