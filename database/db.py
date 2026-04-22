import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "database/lesson_architect.db")


def get_connection():
    """Returns a SQLite connection. Creates the DB file if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables dict-like row access
    return conn


def initialize_db():
    """
    Creates all required tables on first run.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Stores each uploaded curriculum
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS curricula (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_path TEXT NOT NULL,
            extracted_text_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Stores extracted topics per curriculum
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            curriculum_id INTEGER NOT NULL,
            topic_name TEXT NOT NULL,
            order_index INTEGER,
            FOREIGN KEY (curriculum_id) REFERENCES curricula(id)
        )
    """)

    # Stores generated MCQ questions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,
            correct_option TEXT NOT NULL,
            FOREIGN KEY (topic_id) REFERENCES topics(id)
        )
    """)

    # Stores each student's quiz attempt
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            question_id INTEGER NOT NULL,
            selected_option TEXT NOT NULL,
            is_correct INTEGER NOT NULL,   -- 1 = correct, 0 = wrong
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES questions(id)
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


if __name__ == "__main__":
    initialize_db()