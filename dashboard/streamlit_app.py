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
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1A2E;
    }
    [data-testid="stSidebar"] * {
        color: #E0E0E0 !important;
    }

    /* Top header strip */
    .header-strip {
        background: linear-gradient(90deg, #16213E, #0F3460);
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-strip h1 {
        color: #E94560;
        margin: 0;
        font-size: 2rem;
    }
    .header-strip p {
        color: #A0A0B0;
        margin: 4px 0 0 0;
        font-size: 0.95rem;
    }

    /* Metric cards */
    .metric-card {
        background: #16213E;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border-left: 4px solid #E94560;
    }
    .metric-card h3 {
        color: #E94560;
        font-size: 1.8rem;
        margin: 0;
    }
    .metric-card p {
        color: #A0A0B0;
        margin: 4px 0 0 0;
        font-size: 0.85rem;
    }

    /* Topic badge */
    .topic-badge {
        display: inline-block;
        background: #0F3460;
        color: #E0E0E0;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
    }

    /* Quiz card */
    .quiz-card {
        background: #16213E;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #0F3460;
        margin-bottom: 16px;
    }

    /* Severity badges */
    .sev-critical { color: #E74C3C; font-weight: bold; }
    .sev-high     { color: #F39C12; font-weight: bold; }
    .sev-medium   { color: #F1C40F; font-weight: bold; }
    .sev-low      { color: #27AE60; font-weight: bold; }

    /* Divider */
    hr { border-color: #0F3460; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 AI Lesson Architect")
    st.markdown("---")

    view = st.radio(
        "Switch View",
        ["👨‍🏫 Teacher Dashboard", "👨‍🎓 Student Portal"],
        index=0 if st.session_state.active_view == "teacher" else 1
    )
    st.session_state.active_view = "teacher" if "Teacher" in view else "student"

    st.markdown("---")

    if st.session_state.curriculum_name:
        st.markdown(f"**📄 Curriculum:**")
        st.markdown(f"`{st.session_state.curriculum_name}`")
        st.markdown(f"**ID:** `{st.session_state.curriculum_id}`")
        st.markdown("---")

    st.markdown("**Pipeline Status**")
    status_icon = "✅" if st.session_state.pipeline_done else "⏳"
    st.markdown(f"{status_icon} PDF Processed")
    st.markdown(f"{'✅' if st.session_state.pipeline_done else '⏳'} Topics Extracted")
    st.markdown(f"{'✅' if st.session_state.pipeline_done else '⏳'} Learning Path Built")
    st.markdown(f"{'✅' if st.session_state.pipeline_done else '⏳'} Quizzes Generated")

    st.markdown("---")
    st.markdown(
        "<small>Neural Nexus Hackathon 2025<br>EduTech — Teacher Augmentation</small>",
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
        "🧠 AI Lesson Architect",
        "Upload a curriculum PDF and let AI build your lesson plan, quizzes, and student analytics."
    )

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Process",
        "📚 Learning Path",
        "📝 Quiz Preview",
        "📊 Analytics"
    ])

    # ════════════════════════════════════
    # TAB 1 — UPLOAD & PROCESS
    # ════════════════════════════════════
    with tab1:
        st.subheader("Upload Curriculum PDF")
        st.markdown(
            "Upload any curriculum, textbook chapter, or syllabus PDF. "
            "The system will extract topics and build the full learning path automatically."
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Max recommended size: 50MB"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            num_topics = st.slider(
                "Number of topics to extract",
                min_value=5, max_value=25, value=10, step=1
            )
        with col2:
            questions_per_topic = st.slider(
                "Questions per topic",
                min_value=1, max_value=5, value=3, step=1
            )

        if uploaded_file and st.button("🚀 Process Curriculum", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    file_bytes = uploaded_file.read()

                    # Stage 1: PDF extraction
                    progress = st.progress(0, text="Extracting text from PDF...")
                    pdf_result = process_pdf(file_bytes, uploaded_file.name)
                    curriculum_id = pdf_result["curriculum_id"]
                    progress.progress(25, text="Text extracted. Running NLP topic detection...")

                    # Stage 2: Topic extraction
                    with open(pdf_result["extracted_text_path"], "r", encoding="utf-8") as f:
                        text = f.read()
                    extract_and_save_topics(curriculum_id, text, top_n=num_topics)
                    progress.progress(50, text="Topics extracted. Building learning path...")

                    # Stage 3: Learning path
                    build_learning_path(curriculum_id)
                    progress.progress(75, text="Learning path ready. Generating quiz questions...")

                    # Stage 4: Quiz generation
                    generate_and_save_quiz(curriculum_id, questions_per_topic)
                    progress.progress(100, text="✅ All done!")

                    # Save to session state
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

        # Show preview if already processed
        if st.session_state.pipeline_done:
            st.markdown("---")
            st.subheader("📋 Processing Summary")
            cid = st.session_state.curriculum_id

            topics   = get_learning_path(cid)
            summary  = get_topic_score_summary(cid)
            attempts = sum(s["total_attempts"] for s in summary)

            c1, c2, c3, c4 = st.columns(4)
            with c1: show_metric("Topics Extracted", str(len(topics)))
            with c2: show_metric("Quiz Questions",   str(len(topics) * questions_per_topic))
            with c3: show_metric("Students Attempted", str(
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

        st.subheader("📚 Generated Learning Path")
        st.markdown("Topics are ordered from foundational to advanced using graph-based dependency analysis.")

        cid    = st.session_state.curriculum_id
        topics = get_learning_path(cid)

        if not topics:
            st.warning("No topics found.")
            return

        # Display as ordered modules
        module_size = 3
        for module_idx in range(0, len(topics), module_size):
            group = topics[module_idx:module_idx + module_size]
            module_num = (module_idx // module_size) + 1

            with st.expander(
                f"Module {module_num}: {group[0]['topic_name']}  "
                f"(~{len(group) * 1.5:.0f} hrs)",
                expanded=(module_num == 1)
            ):
                for t in group:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(
                            f"<span class='topic-badge'>#{t['order_index']+1}</span> "
                            f"**{t['topic_name']}**",
                            unsafe_allow_html=True
                        )
                    with col_b:
                        q_count = len(get_questions_for_topic(t["topic_id"]))
                        st.markdown(f"`{q_count} questions`")

    # ════════════════════════════════════
    # TAB 3 — QUIZ PREVIEW
    # ════════════════════════════════════
    with tab3:
        if not st.session_state.pipeline_done:
            st.info("📤 Upload and process a curriculum first.")
            return

        st.subheader("📝 Quiz Question Preview")
        st.markdown("Browse all generated questions by topic.")

        cid    = st.session_state.curriculum_id
        topics = get_learning_path(cid)

        if not topics:
            st.warning("No topics found.")
            return

        topic_names = [t["topic_name"] for t in topics]
        selected_topic_name = st.selectbox("Select a topic to preview", topic_names)
        selected_topic = next(t for t in topics if t["topic_name"] == selected_topic_name)

        questions = get_questions_for_topic(selected_topic["topic_id"])

        if not questions:
            st.warning("No questions generated for this topic.")
        else:
            st.markdown(f"**{len(questions)} question(s) for: {selected_topic_name}**")
            for i, q in enumerate(questions, 1):
                with st.expander(f"Q{i}: {q['question_text'][:80]}..."):
                    option_map = {
                        "A": q["option_a"],
                        "B": q["option_b"],
                        "C": q["option_c"],
                        "D": q["option_d"]
                    }
                    for label, text in option_map.items():
                        if label == q["correct_option"]:
                            st.markdown(f"**✅ {label}. {text}**")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;{label}. {text}")

    # ════════════════════════════════════
    # TAB 4 — ANALYTICS
    # ════════════════════════════════════
    with tab4:
        if not st.session_state.pipeline_done:
            st.info("📤 Upload and process a curriculum first.")
            return

        st.subheader("📊 Student Analytics")
        cid = st.session_state.curriculum_id

        summary = get_topic_score_summary(cid)
        has_data = any(s["total_attempts"] > 0 for s in summary)

        if not has_data:
            st.info(
                "No student attempts yet. "
                "Switch to the **Student Portal** to attempt quizzes, "
                "then come back here."
            )
            return

        if st.button("🔄 Refresh Analytics"):
            with st.spinner("Generating charts..."):
                results = run_full_analytics(cid)
                st.session_state["analytics_results"] = results

        # Auto-generate on first load
        if "analytics_results" not in st.session_state:
            with st.spinner("Generating analytics..."):
                st.session_state["analytics_results"] = run_full_analytics(cid)

        results = st.session_state["analytics_results"]

        # ── Weak topic report ──
        st.markdown("#### 🚨 Topics Needing Attention")
        report = generate_weak_topic_report(cid)
        if report:
            report_df = pd.DataFrame(report)[
                ["rank", "topic_name", "accuracy_percent", "severity", "total_attempts"]
            ]
            report_df.columns = ["Rank", "Topic", "Accuracy (%)", "Severity", "Attempts"]
            st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Charts ──
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🌡️ Doubt Heatmap")
            if results.get("heatmap_path") and os.path.exists(results["heatmap_path"]):
                img = Image.open(results["heatmap_path"])
                st.image(img, use_container_width=True)
            else:
                st.warning("Heatmap not available yet.")

        with col_right:
            st.markdown("#### 📈 Topic Accuracy")
            if results.get("accuracy_chart_path") and os.path.exists(results["accuracy_chart_path"]):
                img = Image.open(results["accuracy_chart_path"])
                st.image(img, use_container_width=True)
            else:
                st.warning("Accuracy chart not available yet.")

        st.markdown("---")
        st.markdown("#### 🏆 Student Leaderboard")
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
        "👨‍🎓 Student Quiz Portal",
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

    # ── Student name input ──
    if not st.session_state.student_name:
        st.subheader("Enter Your Name")
        name_input = st.text_input("Your name", placeholder="e.g. Alice")
        if st.button("Start", type="primary") and name_input.strip():
            st.session_state.student_name = name_input.strip()
            st.rerun()
        return

    st.markdown(f"**👋 Hello, {st.session_state.student_name}!**")
    st.markdown("---")

    # ── Topic selector (if no active quiz) ──
    if not st.session_state.quiz_session_id:
        st.subheader("Select a Topic to Attempt")

        summary = {s["topic_id"]: s for s in get_topic_score_summary(cid)}

        for t in topics:
            tid  = t["topic_id"]
            name = t["topic_name"]
            s    = summary.get(tid, {})
            acc  = s.get("accuracy_percent")

            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.markdown(f"**{t['order_index']+1}. {name}**")
            with col2:
                if acc is not None:
                    color = "#27AE60" if acc >= 60 else "#F39C12" if acc >= 40 else "#E74C3C"
                    st.markdown(
                        f"<span style='color:{color}'>{acc}% class avg</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("<span style='color:#888'>Not attempted</span>", unsafe_allow_html=True)
            with col3:
                if st.button(f"Start Quiz", key=f"start_{tid}"):
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
        st.subheader(f"📝 Quiz: {st.session_state.quiz_topic_name}")

        q = st.session_state.current_question
        if not q:
            st.session_state.quiz_complete = True
            st.rerun()
            return

        st.markdown(
            f"<div class='quiz-card'><b>Question:</b><br>{q['question_text']}</div>",
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
            format_func=lambda x: f"{x}. {option_map[x]}",
            key=f"radio_{q['question_id']}"
        )

        if st.button("Submit Answer", type="primary", disabled=(selected is None)):
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

        st.subheader("🏁 Quiz Complete!")

        score_pct = final["score_percent"]
        color = "#27AE60" if score_pct >= 60 else "#F39C12" if score_pct >= 40 else "#E74C3C"

        st.markdown(f"""
        <div style="text-align:center; padding: 30px; background:#16213E;
                    border-radius:12px; margin-bottom:20px;">
            <h1 style="color:{color}; font-size: 3rem;">{score_pct}%</h1>
            <p style="color:#A0A0B0; font-size:1.1rem;">
                {final['correct']} correct out of {final['total']} questions
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Answer Breakdown")
        for item in final["breakdown"]:
            icon = "✅" if item["is_correct"] else "❌"
            with st.expander(f"{icon} {item['question_text'][:70]}..."):
                st.markdown(f"**Your answer:** {item['selected']}. {item['selected_text']}")
                if not item["is_correct"]:
                    st.markdown(f"**Correct answer:** {item['correct']}. {item['correct_text']}")

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