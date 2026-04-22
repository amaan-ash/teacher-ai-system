import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for Streamlit + servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from dotenv import load_dotenv
from database.db import get_connection
from app.quiz_engine import get_topic_score_summary, get_all_attempts_for_curriculum

load_dotenv()

RESULTS_DIR = os.getenv("RESULTS_DIR", "data/results")


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_student_topic_matrix(curriculum_id: int) -> pd.DataFrame | None:
    """
    Builds a Student × Topic accuracy matrix for the heatmap.

    Rows    = students
    Columns = topics (in learning order)
    Values  = accuracy % for that student on that topic
              NaN if student hasn't attempted that topic

    Example:
                        Binary Search   Sorting   Graph Traversal
        Alice               66.7          33.3          100.0
        Bob                 33.3          66.7           NaN
        Charlie            100.0          33.3           66.7

    This is what gets rendered as the heatmap.
    """
    attempts = get_all_attempts_for_curriculum(curriculum_id)

    if not attempts:
        return None

    df = pd.DataFrame(attempts)

    # Aggregate: accuracy per student per topic
    grouped = (
        df.groupby(["student_name", "topic_name"])
          .agg(
              total=("is_correct", "count"),
              correct=("is_correct", "sum")
          )
          .reset_index()
    )
    grouped["accuracy"] = (grouped["correct"] / grouped["total"] * 100).round(1)

    # Pivot to matrix
    matrix = grouped.pivot(
        index="student_name",
        columns="topic_name",
        values="accuracy"
    )

    # Order columns by topic learning order
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic_name FROM topics
        WHERE curriculum_id = ?
        ORDER BY order_index ASC
    """, (curriculum_id,))
    ordered_topics = [row["topic_name"] for row in cursor.fetchall()]
    conn.close()

    # Keep only columns present in matrix, in learning order
    ordered_cols = [t for t in ordered_topics if t in matrix.columns]
    matrix = matrix[ordered_cols]

    return matrix


def _struggle_color_label(struggle_score: float) -> str:
    """Maps a struggle score to a readable severity label."""
    if struggle_score >= 0.7:
        return "🔴 Critical"
    elif struggle_score >= 0.5:
        return "🟠 High"
    elif struggle_score >= 0.3:
        return "🟡 Medium"
    elif struggle_score > 0.0:
        return "🟢 Low"
    else:
        return "⚪ No Data"


# ─────────────────────────────────────────────
# CHART 1 — DOUBT HEATMAP
# ─────────────────────────────────────────────

def generate_doubt_heatmap(curriculum_id: int) -> str | None:
    """
    Generates a Student × Topic accuracy heatmap and saves it as PNG.

    Color scale:
        Red   = low accuracy  (high doubt / struggle)
        Green = high accuracy (well understood)
        Grey  = no attempt

    The heatmap immediately shows teachers:
    - Which topics are universally struggled with (full red column)
    - Which students are struggling across the board (full red row)
    - Which topic-student combos need individual attention

    Returns:
        Path to saved PNG, or None if no data.
    """
    _ensure_results_dir()

    matrix = _get_student_topic_matrix(curriculum_id)
    if matrix is None or matrix.empty:
        print("[Analytics] No attempt data found for heatmap.")
        return None

    n_students = len(matrix.index)
    n_topics   = len(matrix.columns)

    # Dynamic figure size based on data dimensions
    fig_width  = max(10, n_topics * 1.4)
    fig_height = max(5,  n_students * 0.8 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom colormap: red (0%) → yellow (50%) → green (100%)
    # Grey for NaN (no attempt)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    cmap.set_bad(color="#CCCCCC")   # NaN = grey

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,              # Show accuracy numbers in each cell
        fmt=".1f",
        linewidths=0.5,
        linecolor="#EEEEEE",
        cbar_kws={
            "label": "Accuracy (%)",
            "shrink": 0.8
        },
        mask=matrix.isna()       # Grey out NaN cells
    )

    # Overlay NaN cells with "N/A" text
    for i, student in enumerate(matrix.index):
        for j, topic in enumerate(matrix.columns):
            if pd.isna(matrix.loc[student, topic]):
                ax.text(
                    j + 0.5, i + 0.5, "N/A",
                    ha="center", va="center",
                    fontsize=8, color="#888888"
                )

    ax.set_title(
        f"Doubt Heatmap — Student Performance by Topic\n"
        f"(Red = Struggling, Green = Strong, Grey = Not Attempted)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Topics (in learning order)", fontsize=10, labelpad=10)
    ax.set_ylabel("Students", fontsize=10, labelpad=10)

    # Rotate topic labels for readability
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        fontsize=8
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=9
    )

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"heatmap_curriculum_{curriculum_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Analytics] Doubt heatmap saved to: {save_path}")
    return save_path


# ─────────────────────────────────────────────
# CHART 2 — TOPIC ACCURACY BAR CHART
# ─────────────────────────────────────────────

def generate_topic_accuracy_chart(curriculum_id: int) -> str | None:
    """
    Generates a horizontal bar chart showing average accuracy per topic.

    Bars are colour-coded by struggle level:
        Red    < 40% accuracy
        Orange 40–60%
        Green  > 60%

    Returns:
        Path to saved PNG, or None if no data.
    """
    _ensure_results_dir()

    summary = get_topic_score_summary(curriculum_id)
    summary = [s for s in summary if s["total_attempts"] > 0]

    if not summary:
        print("[Analytics] No attempt data for accuracy chart.")
        return None

    topic_names = [s["topic_name"] for s in summary]
    accuracies  = [s["accuracy_percent"] or 0.0 for s in summary]

    # Colour per bar
    colors = []
    for acc in accuracies:
        if acc < 40:
            colors.append("#E74C3C")    # Red
        elif acc < 60:
            colors.append("#F39C12")   # Orange
        else:
            colors.append("#27AE60")   # Green

    fig_height = max(5, len(topic_names) * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(topic_names, accuracies, color=colors, edgecolor="white", height=0.6)

    # Add accuracy labels at end of each bar
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center", ha="left",
            fontsize=9, color="#333333"
        )

    # Reference lines
    ax.axvline(40, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.5, label="Critical (40%)")
    ax.axvline(60, color="#F39C12", linestyle="--", linewidth=0.8, alpha=0.5, label="Moderate (60%)")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Average Accuracy (%)", fontsize=10)
    ax.set_title(
        "Topic-wise Average Accuracy\n(All Students Combined)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()   # Highest topic at top

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"accuracy_chart_curriculum_{curriculum_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Analytics] Accuracy chart saved to: {save_path}")
    return save_path


# ─────────────────────────────────────────────
# CHART 3 — STUDENT LEADERBOARD
# ─────────────────────────────────────────────

def generate_student_leaderboard_chart(curriculum_id: int) -> str | None:
    """
    Generates a bar chart ranking students by overall accuracy.

    Returns:
        Path to saved PNG, or None if no data.
    """
    _ensure_results_dir()

    attempts = get_all_attempts_for_curriculum(curriculum_id)
    if not attempts:
        return None

    df = pd.DataFrame(attempts)

    leaderboard = (
        df.groupby("student_name")
          .agg(total=("is_correct", "count"), correct=("is_correct", "sum"))
          .reset_index()
    )
    leaderboard["accuracy"] = (leaderboard["correct"] / leaderboard["total"] * 100).round(1)
    leaderboard = leaderboard.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(leaderboard) * 0.6 + 1.5)))

    bar_colors = [
        "#27AE60" if acc >= 60 else "#F39C12" if acc >= 40 else "#E74C3C"
        for acc in leaderboard["accuracy"]
    ]

    bars = ax.barh(
        leaderboard["student_name"],
        leaderboard["accuracy"],
        color=bar_colors,
        edgecolor="white",
        height=0.5
    )

    for bar, acc in zip(bars, leaderboard["accuracy"]):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center", ha="left", fontsize=9
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Overall Accuracy (%)", fontsize=10)
    ax.set_title("Student Leaderboard", fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"leaderboard_curriculum_{curriculum_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Analytics] Leaderboard chart saved to: {save_path}")
    return save_path


# ─────────────────────────────────────────────
# REPORT — WEAK TOPIC ANALYSIS
# ─────────────────────────────────────────────

def generate_weak_topic_report(curriculum_id: int) -> list[dict]:
    """
    Returns a ranked list of topics by struggle score.
    Used by the dashboard to show teachers where to focus.

    Returns:
        List of dicts sorted by struggle_score descending:
        {
            rank, topic_name, accuracy_percent,
            struggle_score, severity, total_attempts
        }
    """
    summary = get_topic_score_summary(curriculum_id)
    summary = [s for s in summary if s["total_attempts"] > 0]

    if not summary:
        return []

    # Sort by struggle score descending
    sorted_summary = sorted(summary, key=lambda x: x["struggle_score"], reverse=True)

    report = []
    for rank, item in enumerate(sorted_summary, 1):
        report.append({
            "rank":            rank,
            "topic_name":      item["topic_name"],
            "accuracy_percent": item["accuracy_percent"],
            "struggle_score":  item["struggle_score"],
            "severity":        _struggle_color_label(item["struggle_score"]),
            "total_attempts":  item["total_attempts"]
        })

    return report


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_full_analytics(curriculum_id: int) -> dict:
    """
    Runs all analytics for a curriculum and returns paths + data.

    Returns:
        {
            heatmap_path,
            accuracy_chart_path,
            leaderboard_path,
            weak_topic_report
        }
    """
    print(f"\n[Analytics] Running full analytics for curriculum_id={curriculum_id}")

    heatmap_path       = generate_doubt_heatmap(curriculum_id)
    accuracy_path      = generate_topic_accuracy_chart(curriculum_id)
    leaderboard_path   = generate_student_leaderboard_chart(curriculum_id)
    weak_topic_report  = generate_weak_topic_report(curriculum_id)

    print(f"[Analytics] Complete.")

    return {
        "heatmap_path":       heatmap_path,
        "accuracy_chart_path": accuracy_path,
        "leaderboard_path":   leaderboard_path,
        "weak_topic_report":  weak_topic_report
    }