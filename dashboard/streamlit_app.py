import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from PIL import Image

from database.db import initialize_db
from app.pdf_processor import process_pdf
from app.topic_extractor import extract_and_save_topics
from app.learning_path import build_learning_path, get_learning_path
from app.quiz_generator import generate_and_save_quiz, get_questions_for_topic
from app.quiz_engine import (
    start_quiz_session, submit_answer,
    get_session_result, end_session,
    get_topic_score_summary
)
from app.analytics import run_full_analytics, generate_weak_topic_report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Lesson Architect",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL INIT
# ─────────────────────────────────────────────

initialize_db()

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────

defaults = {
    "curriculum_id":    None,
    "curriculum_name":  None,
    "pipeline_done":    False,
    "quiz_session_id":  None,
    "quiz_topic_id":    None,
    "quiz_topic_name":  None,
    "current_question": None,
    "quiz_complete":    False,
    "quiz_score":       None,
    "student_name":     "",
    "active_view":      "teacher"
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Base & Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background-color: #0D0F1A;
    }

    /* ── Remove horizontal scroll ── */
    .main .block-container {
        max-width: 100%;
        padding: 1.5rem 2rem 3rem 2rem;
        overflow-x: hidden;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111328 0%, #0D0F1A 100%);
        border-right: 1px solid #1E2140;
    }
    [data-testid="stSidebar"] * {
        color: #C8CADE !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #1E2140;
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid #1E2140;
        margin-bottom: 20px;
    }
    .sidebar-brand-icon {
        font-size: 1.8rem;
        line-height: 1;
    }
    .sidebar-brand-text {
        font-size: 1rem;
        font-weight: 700;
        color: #E94560 !important;
        letter-spacing: 0.01em;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 0.7rem;
        color: #6B7280 !important;
        font-weight: 400;
    }

    /* ── Sidebar status pills ── */
    .status-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 12px;
        border-radius: 8px;
        background: #141629;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }
    .status-dot-done { color: #10B981; font-size: 1rem; }
    .status-dot-wait { color: #6B7280; font-size: 1rem; }

    /* ── Sidebar curriculum info ── */
    .sidebar-curr-card {
        background: #141629;
        border: 1px solid #1E2140;
        border-left: 3px solid #E94560;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 0.82rem;
        word-break: break-word;
    }
    .sidebar-curr-card .curr-label {
        color: #6B7280 !important;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 3px;
    }
    .sidebar-curr-card .curr-name {
        color: #E0E3F0 !important;
        font-weight: 600;
    }

    /* ── Header strip ── */
    .header-strip {
        background: linear-gradient(135deg, #111328 0%, #16213E 60%, #1a1040 100%);
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid #1E2140;
        position: relative;
        overflow: hidden;
    }
    .header-strip::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 160px; height: 160px;
        background: radial-gradient(circle, rgba(233,69,96,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .header-strip h1 {
        color: #FFFFFF;
        margin: 0 0 6px 0;
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .header-strip h1 span { color: #E94560; }
    .header-strip p {
        color: #8B92B8;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #111328;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        border: 1px solid #1E2140;
        border-top: 3px solid #E94560;
        transition: transform 0.15s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card h3 {
        color: #E94560;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        line-height: 1;
    }
    .metric-card p {
        color: #6B7280;
        margin: 0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    /* ── Topic badge ── */
    .topic-badge {
        display: inline-block;
        background: rgba(233,69,96,0.12);
        color: #E94560;
        border: 1px solid rgba(233,69,96,0.3);
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ── Learning path topic row ── */
    .topic-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        background: #141629;
        border: 1px solid #1E2140;
        margin-bottom: 8px;
        transition: border-color 0.2s;
    }
    .topic-row:hover { border-color: #E94560; }
    .topic-row-name {
        flex: 1;
        color: #E0E3F0;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .topic-row-q {
        background: #1E2140;
        color: #8B92B8;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    /* ── Quiz card ── */
    .quiz-card {
        background: #111328;
        border-radius: 14px;
        padding: 24px 28px;
        border: 1px solid #1E2140;
        margin-bottom: 20px;
        line-height: 1.7;
    }
    .quiz-card .q-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #E94560;
        margin-bottom: 10px;
    }
    .quiz-card .q-text {
        font-size: 1.05rem;
        color: #E0E3F0;
        font-weight: 500;
    }

    /* ── Student topic card ── */
    .student-topic-card {
        background: #111328;
        border: 1px solid #1E2140;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: border-color 0.2s, transform 0.15s;
    }
    .student-topic-card:hover {
        border-color: #2E3360;
        transform: translateX(3px);
    }
    .student-topic-number {
        width: 32px; height: 32px;
        border-radius: 50%;
        background: rgba(233,69,96,0.1);
        border: 1px solid rgba(233,69,96,0.3);
        display: flex; align-items: center; justify-content: center;
        color: #E94560;
        font-weight: 700;
        font-size: 0.8rem;
        flex-shrink: 0;
    }
    .student-topic-name {
        flex: 1;
        color: #E0E3F0;
        font-weight: 500;
        font-size: 0.92rem;
    }

    /* ── Score result card ── */
    .score-card {
        text-align: center;
        padding: 36px 24px;
        background: #111328;
        border-radius: 16px;
        border: 1px solid #1E2140;
        margin-bottom: 24px;
    }
    .score-card .score-pct {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 10px;
    }
    .score-card .score-sub {
        color: #6B7280;
        font-size: 1rem;
    }

    /* ── Student name entry ── */
    .name-entry-card {
        max-width: 440px;
        margin: 60px auto;
        background: #111328;
        border: 1px solid #1E2140;
        border-radius: 16px;
        padding: 36px 32px;
        text-align: center;
    }
    .name-entry-card h2 {
        color: #E0E3F0;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .name-entry-card p {
        color: #6B7280;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }

    /* ── Severity badges ── */
    .sev-critical { color: #E74C3C; font-weight: 600; }
    .sev-high     { color: #F39C12; font-weight: 600; }
    .sev-medium   { color: #F1C40F; font-weight: 600; }
    .sev-low      { color: #10B981; font-weight: 600; }

    /* ── Section label ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6B7280;
        margin-bottom: 14px;
        margin-top: 4px;
    }

    /* ── Greeting banner ── */
    .greeting-bar {
        background: linear-gradient(90deg, rgba(233,69,96,0.08) 0%, transparent 100%);
        border-left: 3px solid #E94560;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #E0E3F0;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* ── Streamlit element overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #111328;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid #1E2140;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #6B7280;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #1E2140 !important;
        color: #E0E3F0 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"] {
        background: #E94560;
        border-color: #E94560;
    }
    .stButton > button[kind="primary"]:hover {
        background: #c73350;
        border-color: #c73350;
        transform: translateY(-1px);
    }

    .stTextInput input, .stSelectbox select {
        background: #141629 !important;
        border-color: #1E2140 !important;
        color: #E0E3F0 !important;
        border-radius: 8px !important;
    }

    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #C8CADE;
    }

    .stExpander {
        background: #111328;
        border: 1px solid #1E2140;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .stExpander summary {
        color: #C8CADE;
        font-weight: 500;
    }

    .stAlert {
        border-radius: 10px;
    }

    .stProgress > div > div {
        background: #E94560;
    }

    /* ── Dividers ── */
    hr { border-color: #1E2140; margin: 20px 0; }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Hide Streamlit default elements ── */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧠</div>
        <div>
            <div class="sidebar-brand-text">AI Lesson Architect</div>
            <div class="sidebar-brand-sub">Neural Nexus Hackathon 2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    view = st.radio(
        "View",
        ["👨‍🏫 Teacher Dashboard", "👨‍🎓 Student Portal"],
        index=0 if st.session_state.active_view == "teacher" else 1,
        label_visibility="collapsed"
    )
    st.session_state.active_view = "teacher" if "Teacher" in view else "student"

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.session_state.curriculum_name:
        st.markdown(f"""
        <div class="sidebar-curr-card">
            <div class="curr-label">Active Curriculum</div>
            <div class="curr-name">{st.session_state.curriculum_name}</div>
            <div style="color:#6B7280;font-size:0.72rem;margin-top:4px;">
                ID: {st.session_state.curriculum_id}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)

    done = st.session_state.pipeline_done
    steps = ["PDF Processed", "Topics Extracted", "Learning Path Built", "Quizzes Generated"]
    for step in steps:
        icon_cls = "status-dot-done" if done else "status-dot-wait"
        icon = "●" if done else "○"
        st.markdown(
            f"<div class='status-row'>"
            f"<span class='{icon_cls}'>{icon}</span>"
            f"<span style='color:{'#C8CADE' if done else '#6B7280'}'>{step}</span>"
            f"</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def show_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="header-strip">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def show_metric(label: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{value}</h3>
        <p>{label}</p>
    </div>
    """, unsafe_allow_html=True)


def reset_quiz_state():
    st.session_state.quiz_session_id  = None
    st.session_state.quiz_topic_id    = None
    st.session_state.quiz_topic_name  = None
    st.session_state.current_question = None
    st.session_state.quiz_complete    = False
    st.session_state.quiz_score       = None


# ─────────────────────────────────────────────
# VIEW 1 — TEACHER DASHBOARD
# ─────────────────────────────────────────────

def render_teacher_dashboard():
    show_header(
        "🧠 AI Lesson <span>Architect</span>",
        "Upload a curriculum PDF and let AI build your lesson plan, quizzes, and student analytics."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📤  Upload & Process",
        "📚  Learning Path",
        "📝  Quiz Preview",
        "📊  Analytics"
    ])

    # ════════════════════════════════════
    # TAB 1 — UPLOAD & PROCESS
    # ════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-label">Upload Curriculum PDF</div>', unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#8B92B8;font-size:0.88rem;'>"
            "Upload any curriculum, textbook chapter, or syllabus PDF. "
            "The system will extract topics and build the full learning path automatically."
            "</span>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Max recommended size: 50MB"
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            num_topics = st.slider(
                "Topics to extract",
                min_value=5, max_value=25, value=10, step=1
            )
        with col2:
            questions_per_topic = st.slider(
                "Questions per topic",
                min_value=1, max_value=5, value=3, step=1
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if uploaded_file and st.button("🚀  Process Curriculum", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    file_bytes = uploaded_file.read()

                    progress = st.progress(0, text="Extracting text from PDF...")
                    pdf_result = process_pdf(file_bytes, uploaded_file.name)
                    curriculum_id = pdf_result["curriculum_id"]
                    progress.progress(25, text="Text extracted — running NLP topic detection...")

                    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
                        text = f.read()
                    extract_and_save_topics(curriculum_id, text, top_n=num_topics)
                    progress.progress(50, text="Topics extracted — building learning path...")

                    build_learning_path(curriculum_id)
                    progress.progress(75, text="Learning path ready — generating quiz questions...")

                    generate_and_save_quiz(curriculum_id, questions_per_topic)
                    progress.progress(100, text="✅ All done!")

                    st.session_state.curriculum_id   = curriculum_id
                    st.session_state.curriculum_name = uploaded_file.name
                    st.session_state.pipeline_done   = True

                    st.success(
                        f"✅ Curriculum processed successfully! "
                        f"Extracted {num_topics} topics with "
                        f"{num_topics * questions_per_topic} quiz questions."
                    )

                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")

        if st.session_state.pipeline_done:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Processing Summary</div>', unsafe_allow_html=True)

            cid    = st.session_state.curriculum_id
            topics  = get_learning_path(cid)
            summary = get_topic_score_summary(cid)
            attempts = sum(s["total_attempts"] for s in summary)

            c1, c2, c3, c4 = st.columns(4)
            with c1: show_metric("Topics Extracted", str(len(topics)))
            with c2: show_metric("Quiz Questions", str(len(topics) * questions_per_topic))
            with c3: show_metric("Active Topics", str(
                len(set(s["topic_name"] for s in summary if s["total_attempts"] > 0))
            ))
            with c4: show_metric("Total Attempts", str(attempts))

    # ════════════════════════════════════
    # TAB 2 — LEARNING PATH
    # ════════════════════════════════════
    with tab2:
        if not st.session_state.pipeline_done:
            st.info("📤 Upload and process a curriculum first.")
            return

        st.markdown('<div class="section-label">Generated Learning Path</div>', unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#8B92B8;font-size:0.88rem;'>"
            "Topics ordered from foundational to advanced using graph-based dependency analysis."
            "</span>",
            unsafe_allow_html=True
        )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        cid    = st.session_state.curriculum_id
        topics = get_learning_path(cid)

        if not topics:
            st.warning("No topics found.")
            return

        module_size = 3
        for module_idx in range(0, len(topics), module_size):
            group      = topics[module_idx:module_idx + module_size]
            module_num = (module_idx // module_size) + 1

            with st.expander(
                f"Module {module_num} — {group[0]['topic_name']}  ·  ~{len(group) * 1.5:.0f} hrs",
                expanded=(module_num == 1)
            ):
                for t in group:
                    q_count = len(get_questions_for_topic(t["topic_id"]))
                    st.markdown(
                        f"""
                        <div class="topic-row">
                            <span class="topic-badge">#{t['order_index']+1}</span>
                            <span class="topic-row-name">{t['topic_name']}</span>
                            <span class="topic-row-q">{q_count} questions</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # ════════════════════════════════════
    # TAB 3 — QUIZ PREVIEW
    # ════════════════════════════════════
    with tab3:
        if not st.session_state.pipeline_done:
            st.info("📤 Upload and process a curriculum first.")
            return

        st.markdown('<div class="section-label">Quiz Question Preview</div>', unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#8B92B8;font-size:0.88rem;'>Browse all generated questions by topic.</span>",
            unsafe_allow_html=True
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        cid    = st.session_state.curriculum_id
        topics = get_learning_path(cid)

        if not topics:
            st.warning("No topics found.")
            return

        topic_names         = [t["topic_name"] for t in topics]
        selected_topic_name = st.selectbox("Select a topic to preview", topic_names)
        selected_topic      = next(t for t in topics if t["topic_name"] == selected_topic_name)

        questions = get_questions_for_topic(selected_topic["topic_id"])

        if not questions:
            st.warning("No questions generated for this topic.")
        else:
            st.markdown(
                f"<div style='color:#8B92B8;font-size:0.85rem;margin:12px 0 16px;'>"
                f"{len(questions)} question(s) for: <strong style='color:#E0E3F0'>{selected_topic_name}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
            for i, q in enumerate(questions, 1):
                with st.expander(f"Q{i}: {q['question_text'][:80]}..."):
                    st.markdown(
                        f"<div style='color:#E0E3F0;font-weight:500;margin-bottom:14px;line-height:1.6;'>"
                        f"{q['question_text']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    option_map = {
                        "A": q["option_a"],
                        "B": q["option_b"],
                        "C": q["option_c"],
                        "D": q["option_d"]
                    }
                    for label, text in option_map.items():
                        if label == q["correct_option"]:
                            st.markdown(
                                f"<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);"
                                f"border-radius:8px;padding:8px 14px;margin-bottom:6px;color:#10B981;font-weight:600;'>"
                                f"✅ {label}. {text}</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div style='background:#141629;border:1px solid #1E2140;"
                                f"border-radius:8px;padding:8px 14px;margin-bottom:6px;color:#8B92B8;'>"
                                f"{label}. {text}</div>",
                                unsafe_allow_html=True
                            )

    # ════════════════════════════════════
    # TAB 4 — ANALYTICS
    # ════════════════════════════════════
    with tab4:
        if not st.session_state.pipeline_done:
            st.info("📤 Upload and process a curriculum first.")
            return

        st.markdown('<div class="section-label">Student Analytics</div>', unsafe_allow_html=True)

        cid     = st.session_state.curriculum_id
        summary = get_topic_score_summary(cid)
        has_data = any(s["total_attempts"] > 0 for s in summary)

        if not has_data:
            st.info(
                "No student attempts yet. "
                "Switch to the **Student Portal** to attempt quizzes, "
                "then come back here."
            )
            return

        if st.button("🔄  Refresh Analytics"):
            with st.spinner("Generating charts..."):
                results = run_full_analytics(cid)
                st.session_state["analytics_results"] = results

        if "analytics_results" not in st.session_state:
            with st.spinner("Generating analytics..."):
                st.session_state["analytics_results"] = run_full_analytics(cid)

        results = st.session_state["analytics_results"]

        st.markdown('<div class="section-label" style="margin-top:16px;">🚨 Topics Needing Attention</div>', unsafe_allow_html=True)
        report = generate_weak_topic_report(cid)
        if report:
            report_df = pd.DataFrame(report)[
                ["rank", "topic_name", "accuracy_percent", "severity", "total_attempts"]
            ]
            report_df.columns = ["Rank", "Topic", "Accuracy (%)", "Severity", "Attempts"]
            st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-label">🌡️ Doubt Heatmap</div>', unsafe_allow_html=True)
            if results.get("heatmap_path") and os.path.exists(results["heatmap_path"]):
                img = Image.open(results["heatmap_path"])
                st.image(img, use_container_width=True)
            else:
                st.warning("Heatmap not available yet.")

        with col_right:
            st.markdown('<div class="section-label">📈 Topic Accuracy</div>', unsafe_allow_html=True)
            if results.get("accuracy_chart_path") and os.path.exists(results["accuracy_chart_path"]):
                img = Image.open(results["accuracy_chart_path"])
                st.image(img, use_container_width=True)
            else:
                st.warning("Accuracy chart not available yet.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">🏆 Student Leaderboard</div>', unsafe_allow_html=True)
        if results.get("leaderboard_path") and os.path.exists(results["leaderboard_path"]):
            img = Image.open(results["leaderboard_path"])
            st.image(img, use_container_width=True)
        else:
            st.warning("Leaderboard not available yet.")


# ─────────────────────────────────────────────
# VIEW 2 — STUDENT PORTAL
# ─────────────────────────────────────────────

def render_student_portal():
    show_header(
        "👨‍🎓 Student <span>Quiz Portal</span>",
        "Attempt topic quizzes and get instant feedback on your performance."
    )

    if not st.session_state.pipeline_done:
        st.warning("⚠️ No curriculum loaded yet. Ask your teacher to upload a curriculum first.")
        return

    cid    = st.session_state.curriculum_id
    topics = get_learning_path(cid)

    if not topics:
        st.warning("No topics available.")
        return

    # ── Student name entry ──
    if not st.session_state.student_name:
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown("""
            <div class="name-entry-card">
                <div style="font-size:2rem;margin-bottom:12px;">👤</div>
                <h2>Welcome!</h2>
                <p>Enter your name to begin the quiz session.</p>
            </div>
            """, unsafe_allow_html=True)
            name_input = st.text_input("Your name", placeholder="e.g. Alice", label_visibility="collapsed")
            if st.button("Continue →", type="primary", use_container_width=True) and name_input.strip():
                st.session_state.student_name = name_input.strip()
                st.rerun()
        return

    st.markdown(
        f"<div class='greeting-bar'>👋 Hello, <strong>{st.session_state.student_name}</strong>! "
        f"Choose a topic below to start your quiz.</div>",
        unsafe_allow_html=True
    )

    # ── Topic selector (if no active quiz) ──
    if not st.session_state.quiz_session_id:
        st.markdown('<div class="section-label">Available Topics</div>', unsafe_allow_html=True)

        summary = {s["topic_id"]: s for s in get_topic_score_summary(cid)}

        for t in topics:
            tid  = t["topic_id"]
            name = t["topic_name"]
            s    = summary.get(tid, {})
            acc  = s.get("accuracy_percent")

            if acc is not None:
                color = "#10B981" if acc >= 60 else "#F39C12" if acc >= 40 else "#E74C3C"
                acc_html = f"<span style='color:{color};font-size:0.82rem;font-weight:600;'>{acc}% class avg</span>"
            else:
                acc_html = "<span style='color:#4B5563;font-size:0.82rem;'>Not attempted</span>"

            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(
                    f"""
                    <div class="student-topic-card">
                        <div class="student-topic-number">{t['order_index']+1}</div>
                        <div class="student-topic-name">{name}</div>
                        {acc_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                if st.button("Start →", key=f"start_{tid}", use_container_width=True):
                    reset_quiz_state()
                    try:
                        session = start_quiz_session(
                            st.session_state.student_name,
                            cid,
                            tid
                        )
                        st.session_state.quiz_session_id  = session["session_id"]
                        st.session_state.quiz_topic_id    = tid
                        st.session_state.quiz_topic_name  = name
                        st.session_state.current_question = session["first_question"]
                        st.session_state.quiz_complete    = False
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Could not start quiz: {e}")
        return

    # ── Active quiz ──
    if not st.session_state.quiz_complete:
        st.markdown(
            f"<div style='color:#E94560;font-size:0.78rem;font-weight:600;letter-spacing:0.08em;"
            f"text-transform:uppercase;margin-bottom:8px;'>Active Quiz</div>"
            f"<div style='color:#E0E3F0;font-size:1.15rem;font-weight:700;margin-bottom:20px;'>"
            f"📝 {st.session_state.quiz_topic_name}</div>",
            unsafe_allow_html=True
        )

        q = st.session_state.current_question
        if not q:
            st.session_state.quiz_complete = True
            st.rerun()
            return

        st.markdown(
            f"<div class='quiz-card'>"
            f"<div class='q-label'>Question</div>"
            f"<div class='q-text'>{q['question_text']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        option_map = {
            "A": q["option_a"],
            "B": q["option_b"],
            "C": q["option_c"],
            "D": q["option_d"]
        }

        selected = st.radio(
            "Choose your answer:",
            options=list(option_map.keys()),
            format_func=lambda x: f"{x}.  {option_map[x]}",
            key=f"radio_{q['question_id']}"
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col_btn, col_space = st.columns([1, 3])
        with col_btn:
            if st.button("Submit Answer →", type="primary", use_container_width=True, disabled=(selected is None)):
                try:
                    result = submit_answer(
                        st.session_state.quiz_session_id,
                        q["question_id"],
                        selected
                    )

                    if result["is_correct"]:
                        st.success("✅ Correct!")
                    else:
                        st.error(
                            f"❌ Wrong. Correct answer: **{result['correct_option']}. "
                            f"{result['correct_text']}**"
                        )

                    if result["session_complete"]:
                        final = get_session_result(st.session_state.quiz_session_id)
                        st.session_state.quiz_score    = final
                        st.session_state.quiz_complete = True
                        end_session(st.session_state.quiz_session_id)

                    st.session_state.current_question = result["next_question"]
                    st.rerun()

                except ValueError as e:
                    st.error(f"Error: {e}")

    # ── Quiz complete — score card ──
    else:
        final = st.session_state.quiz_score
        if not final:
            st.info("Quiz complete. Select another topic.")
            if st.button("← Back to Topics"):
                reset_quiz_state()
                st.rerun()
            return

        score_pct = final["score_percent"]
        color = "#10B981" if score_pct >= 60 else "#F39C12" if score_pct >= 40 else "#E74C3C"
        emoji = "🎉" if score_pct >= 60 else "💪" if score_pct >= 40 else "📖"

        st.markdown(f"""
        <div class="score-card">
            <div style="font-size:2.5rem;margin-bottom:8px;">{emoji}</div>
            <div class="score-pct" style="color:{color};">{score_pct}%</div>
            <div class="score-sub">
                {final['correct']} correct out of {final['total']} questions
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Answer Breakdown</div>', unsafe_allow_html=True)
        for item in final["breakdown"]:
            icon = "✅" if item["is_correct"] else "❌"
            with st.expander(f"{icon} {item['question_text'][:70]}..."):
                st.markdown(
                    f"<div style='color:#8B92B8;font-size:0.85rem;margin-bottom:10px;line-height:1.6;'>"
                    f"{item['question_text']}</div>",
                    unsafe_allow_html=True
                )
                if item["is_correct"]:
                    st.markdown(
                        f"<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);"
                        f"border-radius:8px;padding:8px 14px;color:#10B981;font-weight:600;'>"
                        f"✅ Your answer: {item['selected']}. {item['selected_text']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.3);"
                        f"border-radius:8px;padding:8px 14px;color:#E74C3C;margin-bottom:8px;'>"
                        f"❌ Your answer: {item['selected']}. {item['selected_text']}</div>"
                        f"<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);"
                        f"border-radius:8px;padding:8px 14px;color:#10B981;font-weight:600;'>"
                        f"✅ Correct: {item['correct']}. {item['correct_text']}</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Topics", use_container_width=True):
                reset_quiz_state()
                st.rerun()
        with col2:
            if st.button("Change Student", use_container_width=True):
                reset_quiz_state()
                st.session_state.student_name = ""
                st.rerun()


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

if st.session_state.active_view == "teacher":
    render_teacher_dashboard()
else:
    render_student_portal()