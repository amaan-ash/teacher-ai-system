import os
import re
from collections import defaultdict, deque
from database.db import get_connection

# ─────────────────────────────────────────────
# KEYWORD SIGNALS
# These lists define our heuristic ordering rules.
# Foundational topics use simple/basic keywords.
# Advanced topics use complexity/application keywords.
# ─────────────────────────────────────────────

FOUNDATIONAL_KEYWORDS = [
    "introduction", "intro", "basic", "basics", "overview",
    "fundamental", "definition", "concept", "what is",
    "history", "origin", "principle", "terminology",
    "types", "classification", "structure"
]

ADVANCED_KEYWORDS = [
    "advanced", "applied", "application", "implementation",
    "optimization", "analysis", "evaluation", "design",
    "extended", "complex", "comparison", "tradeoff",
    "performance", "case study", "project", "algorithm",
    "technique", "method", "framework", "architecture"
]

# Topics containing these are almost always prerequisites
PREREQUISITE_SIGNALS = [
    "number system", "data type", "variable", "operator",
    "introduction", "overview", "basic", "fundamental",
    "definition", "concept", "notation", "representation"
]


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _compute_complexity_score(topic_name: str) -> float:
    """
    Assigns a complexity score to a topic based on heuristics.
    Lower score = more foundational = should appear earlier.

    Scoring logic:
    - Foundational keyword found  → subtract points (earlier)
    - Advanced keyword found      → add points (later)
    - Word count                  → more words = more specific = later
    - All caps abbreviations      → likely technical = slightly later

    Returns:
        float between 0.0 (most foundational) and 1.0 (most advanced)
    """
    topic_lower = topic_name.lower()
    score = 0.5  # Neutral starting point

    for kw in FOUNDATIONAL_KEYWORDS:
        if kw in topic_lower:
            score -= 0.15

    for kw in ADVANCED_KEYWORDS:
        if kw in topic_lower:
            score += 0.15

    # Longer topic names tend to be more specific/advanced
    word_count = len(topic_name.split())
    if word_count == 1:
        score -= 0.1
    elif word_count >= 3:
        score += 0.05 * (word_count - 2)

    # Abbreviations like "OOP", "DBMS", "OS" are usually foundational context
    if re.search(r'\b[A-Z]{2,5}\b', topic_name):
        score -= 0.05

    # Clamp between 0 and 1
    return max(0.0, min(1.0, score))


def _is_prerequisite_of(topic_a: str, topic_b: str) -> bool:
    """
    Returns True if topic_a is likely a prerequisite of topic_b.

    Rules:
    1. If topic_a contains a prerequisite signal keyword → it goes first
    2. If topic_a's name is a substring of topic_b → it's more general → goes first
       e.g. "Sorting" is prerequisite of "Merge Sort Algorithm"
    3. If topic_a has significantly lower complexity score → goes first
    """
    a_lower = topic_a.lower()
    b_lower = topic_b.lower()

    # Rule 1: prerequisite signal
    for signal in PREREQUISITE_SIGNALS:
        if signal in a_lower and signal not in b_lower:
            return True

    # Rule 2: substring containment (general → specific)
    # "Graph" is prerequisite of "Graph Traversal Algorithm"
    a_words = set(a_lower.split())
    b_words = set(b_lower.split())
    if a_words and a_words.issubset(b_words) and a_lower != b_lower:
        return True

    # Rule 3: complexity gap
    score_a = _compute_complexity_score(topic_a)
    score_b = _compute_complexity_score(topic_b)
    if score_b - score_a > 0.3:  # Significant gap
        return True

    return False


def _build_dag(topics: list[dict]) -> dict[int, list[int]]:
    """
    Builds a Directed Acyclic Graph (DAG) from topic list.
    Edge: topic_a → topic_b means "learn A before B"

    Args:
        topics: List of dicts with topic_id and topic_name.

    Returns:
        adjacency: dict mapping topic_id → list of dependent topic_ids
    """
    adjacency = defaultdict(list)    # topic_id → [dependent_ids]
    in_degree = defaultdict(int)     # topic_id → number of incoming edges

    # Initialize all nodes
    for t in topics:
        _ = adjacency[t["topic_id"]]
        _ = in_degree[t["topic_id"]]

    # Build edges
    for i, topic_a in enumerate(topics):
        for j, topic_b in enumerate(topics):
            if i == j:
                continue
            if _is_prerequisite_of(topic_a["topic_name"], topic_b["topic_name"]):
                # Prevent duplicate edges
                if topic_b["topic_id"] not in adjacency[topic_a["topic_id"]]:
                    adjacency[topic_a["topic_id"]].append(topic_b["topic_id"])
                    in_degree[topic_b["topic_id"]] += 1

    return adjacency, in_degree


def _topological_sort(topics: list[dict], adjacency: dict, in_degree: dict) -> list[dict]:
    """
    Performs Kahn's Algorithm for topological sort on the DAG.

    Kahn's Algorithm:
    1. Start with all nodes that have in_degree = 0 (no prerequisites)
    2. Process each, reduce in_degree of its dependents
    3. Add dependents with in_degree = 0 to queue
    4. Repeat until queue is empty

    If a cycle is detected (queue empties before all nodes processed),
    we fall back to complexity-score-based sorting for remaining nodes.

    Returns:
        Ordered list of topic dicts.
    """
    topic_map = {t["topic_id"]: t for t in topics}

    # Initialize queue with nodes that have no prerequisites
    queue = deque()
    for t in topics:
        if in_degree[t["topic_id"]] == 0:
            queue.append(t["topic_id"])

    ordered = []
    visited = set()

    while queue:
        # Among all ready nodes, pick the one with lowest complexity score
        # This makes the ordering deterministic and pedagogically sensible
        ready_ids = list(queue)
        ready_ids.sort(
            key=lambda tid: _compute_complexity_score(topic_map[tid]["topic_name"])
        )
        current_id = ready_ids[0]
        queue.remove(current_id)

        ordered.append(topic_map[current_id])
        visited.add(current_id)

        for neighbor_id in adjacency[current_id]:
            in_degree[neighbor_id] -= 1
            if in_degree[neighbor_id] == 0:
                queue.append(neighbor_id)

    # Fallback: if cycle detected, sort remaining by complexity score
    remaining = [t for t in topics if t["topic_id"] not in visited]
    if remaining:
        print(f"[Learning Path] Cycle detected. Sorting {len(remaining)} remaining topics by complexity.")
        remaining.sort(key=lambda t: _compute_complexity_score(t["topic_name"]))
        ordered.extend(remaining)

    return ordered


def _group_into_modules(ordered_topics: list[dict], module_size: int = 3) -> list[dict]:
    """
    Groups ordered topics into learning modules.

    Why modules: Teachers don't teach topic-by-topic in isolation.
    They group related topics into lessons/units.

    Grouping logic:
    - Every `module_size` topics form one module
    - Module is named after its first (most foundational) topic
    - Each module gets an estimated duration

    Args:
        ordered_topics: Topologically sorted topic list.
        module_size:    How many topics per module.

    Returns:
        List of module dicts.
    """
    modules = []
    for i in range(0, len(ordered_topics), module_size):
        group = ordered_topics[i:i + module_size]
        module_number = (i // module_size) + 1

        modules.append({
            "module_number": module_number,
            "module_name": f"Module {module_number}: {group[0]['topic_name']}",
            "topics": group,
            "estimated_hours": len(group) * 1.5   # Rough: 1.5 hrs per topic
        })

    return modules


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def build_learning_path(curriculum_id: int) -> list[dict]:
    """
    Full pipeline:
    1. Fetch topics from DB
    2. Build DAG
    3. Topological sort
    4. Group into modules
    5. Update order_index in DB
    6. Return structured learning path

    Args:
        curriculum_id: ID of the curriculum.

    Returns:
        List of module dicts, each containing ordered topics.
    """
    print(f"\n[Learning Path] Building learning path for curriculum_id={curriculum_id}")

    # ── Fetch topics ──
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id as topic_id, topic_name, order_index
        FROM topics
        WHERE curriculum_id = ?
        ORDER BY order_index ASC
    """, (curriculum_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No topics found for curriculum_id={curriculum_id}. Run topic extraction first.")

    topics = [
        {"topic_id": row["topic_id"], "topic_name": row["topic_name"], "order_index": row["order_index"]}
        for row in rows
    ]

    print(f"[Learning Path] {len(topics)} topics fetched. Building DAG...")

    # ── Build DAG and sort ──
    adjacency, in_degree = _build_dag(topics)
    ordered_topics = _topological_sort(topics, adjacency, in_degree)

    # ── Update order_index in DB to reflect new ordering ──
    conn = get_connection()
    cursor = conn.cursor()
    for new_index, topic in enumerate(ordered_topics):
        cursor.execute("""
            UPDATE topics SET order_index = ? WHERE id = ?
        """, (new_index, topic["topic_id"]))
        topic["order_index"] = new_index
    conn.commit()
    conn.close()

    print(f"[Learning Path] Order updated in DB.")

    # ── Group into modules ──
    modules = _group_into_modules(ordered_topics, module_size=3)

    print(f"[Learning Path] {len(modules)} modules created.")
    return modules


def get_learning_path(curriculum_id: int) -> list[dict]:
    """
    Fetches the already-built learning path from DB.
    Used by the dashboard and quiz generator.

    Returns:
        Ordered list of topic dicts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id as topic_id, topic_name, order_index
        FROM topics
        WHERE curriculum_id = ?
        ORDER BY order_index ASC
    """, (curriculum_id,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {"topic_id": row["topic_id"], "topic_name": row["topic_name"], "order_index": row["order_index"]}
        for row in rows
    ]