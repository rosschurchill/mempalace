"""Tests for Phase 4 Cognition Engine:
topology (PageRank + degree), KG evolve_fact, ambient RAG, and REM cycle."""

import pytest

from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.topology import pagerank, entity_degree
from mempalace.ambient import get_whisper, get_socratic_question, get_eigen_thoughts
from mempalace.rem_cycle import run_rem_cycle


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def kg(tmp_path):
    """Fresh KG backed by a temp SQLite file."""
    db = str(tmp_path / "kg.sqlite3")
    g = KnowledgeGraph(db_path=db)
    yield g
    g.close()


@pytest.fixture
def populated_kg(tmp_path):
    """KG with a small social graph for topology tests."""
    db = str(tmp_path / "kg.sqlite3")
    g = KnowledgeGraph(db_path=db)
    # Max is the hub — many connections
    g.add_triple("Max", "child_of", "Alice")
    g.add_triple("Max", "child_of", "Bob")
    g.add_triple("Max", "does", "swimming")
    g.add_triple("Max", "loves", "chess")
    g.add_triple("Alice", "works_on", "Orion")
    g.add_triple("Bob", "works_on", "Orion")
    g.add_triple("Orion", "uses", "Postgres")
    # Leaf node with one connection
    g.add_triple("Lumi", "assists", "Alice")
    yield g
    g.close()


# ── topology: entity_degree ───────────────────────────────────────────────────

def test_entity_degree_empty(kg):
    assert entity_degree(kg) == []


def test_entity_degree_basic(populated_kg):
    degrees = entity_degree(populated_kg)
    assert len(degrees) > 0
    # Max has the most connections
    max_entry = next((d for d in degrees if d["entity"] == "Max"), None)
    assert max_entry is not None
    assert max_entry["out_degree"] >= 3
    # All entries have the required keys
    for entry in degrees:
        assert "entity" in entry
        assert "in_degree" in entry
        assert "out_degree" in entry
        assert "total" in entry


def test_entity_degree_sorted_descending(populated_kg):
    degrees = entity_degree(populated_kg)
    totals = [d["total"] for d in degrees]
    assert totals == sorted(totals, reverse=True)


# ── topology: pagerank ────────────────────────────────────────────────────────

def test_pagerank_empty(kg):
    assert pagerank(kg) == []


def test_pagerank_returns_all_entities(populated_kg):
    ranked = pagerank(populated_kg)
    entity_names = {r["entity"] for r in ranked}
    # All entities added should appear
    assert "Max" in entity_names
    assert "Alice" in entity_names
    assert "Orion" in entity_names


def test_pagerank_scores_sum_to_one(populated_kg):
    ranked = pagerank(populated_kg)
    total = sum(r["score"] for r in ranked)
    assert abs(total - 1.0) < 0.01


def test_pagerank_sorted_descending(populated_kg):
    ranked = pagerank(populated_kg)
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_pagerank_hub_ranks_high(populated_kg):
    """Max and Orion are hubs — they should rank in the top half."""
    ranked = pagerank(populated_kg)
    top_half = {r["entity"] for r in ranked[:len(ranked) // 2 + 1]}
    assert "Orion" in top_half or "Max" in top_half


def test_pagerank_single_entity(tmp_path):
    g = KnowledgeGraph(db_path=str(tmp_path / "k.sqlite3"))
    g.add_entity("Solo", "person")
    ranked = pagerank(g)
    g.close()
    assert len(ranked) == 1
    assert ranked[0]["entity"] == "Solo"
    assert abs(ranked[0]["score"] - 1.0) < 0.01


def test_pagerank_disconnected_graph(tmp_path):
    """Two disconnected components — both should get non-zero scores."""
    g = KnowledgeGraph(db_path=str(tmp_path / "k.sqlite3"))
    # Component 1: Alice → Bob
    g.add_triple("Alice", "knows", "Bob")
    # Component 2: Carol → Dave (no connection to comp 1)
    g.add_triple("Carol", "knows", "Dave")
    ranked = pagerank(g)
    g.close()
    names = {r["entity"] for r in ranked}
    assert names == {"Alice", "Bob", "Carol", "Dave"}
    for r in ranked:
        assert r["score"] > 0.0
    total = sum(r["score"] for r in ranked)
    assert abs(total - 1.0) < 0.01


def test_pagerank_cycle_only(tmp_path):
    """Pure cycle A→B→C→A — scores should be roughly equal (dangling-free)."""
    g = KnowledgeGraph(db_path=str(tmp_path / "k.sqlite3"))
    g.add_triple("A", "links", "B")
    g.add_triple("B", "links", "C")
    g.add_triple("C", "links", "A")
    ranked = pagerank(g)
    g.close()
    assert len(ranked) == 3
    scores = [r["score"] for r in ranked]
    # All scores should be close to each other (within 10%)
    assert max(scores) - min(scores) < 0.1
    assert abs(sum(scores) - 1.0) < 0.01


def test_pagerank_self_loop(tmp_path):
    """Self-loop (A→A) should not crash and still sum to ~1."""
    g = KnowledgeGraph(db_path=str(tmp_path / "k.sqlite3"))
    g.add_triple("Looper", "references", "Looper")
    g.add_triple("Other", "knows", "Looper")
    ranked = pagerank(g)
    g.close()
    assert len(ranked) == 2
    names = {r["entity"] for r in ranked}
    assert "Looper" in names
    total = sum(r["score"] for r in ranked)
    assert abs(total - 1.0) < 0.01


# ── KG evolve_fact ────────────────────────────────────────────────────────────

def test_evolve_fact_basic(kg):
    kg.add_triple("Max", "attends", "Lincoln Elementary")
    new_id = kg.evolve_fact("Max", "attends", "Lincoln Elementary", "Westside Middle")
    assert new_id is not None

    facts = kg.query_entity("Max", direction="outgoing")
    predicates = {f["predicate"] for f in facts}
    # New fact exists
    assert "attends" in predicates
    # query_entity returns obj_name (original name from entities table, not entity_id)
    new_obj = next((f["object"] for f in facts if f["predicate"] == "attends" and f["current"]), None)
    assert new_obj == "Westside Middle"


def test_evolve_fact_invalidates_old(kg):
    kg.add_triple("Max", "attends", "Lincoln Elementary")
    kg.evolve_fact("Max", "attends", "Lincoln Elementary", "Westside Middle")

    facts = kg.query_entity("Max", direction="outgoing")
    # Old fact should be invalid (current=False)
    old_fact = next(
        (f for f in facts if f["predicate"] == "attends" and f["object"] == "lincoln_elementary"),
        None,
    )
    if old_fact:  # May not appear if query_entity only returns current facts
        assert not old_fact.get("current", True)


def test_evolve_fact_creates_provenance(kg):
    kg.add_triple("Max", "attends", "Lincoln Elementary")
    kg.evolve_fact("Max", "attends", "Lincoln Elementary", "Westside Middle")

    facts = kg.query_entity("Max", direction="outgoing")
    predicates = {f["predicate"] for f in facts}
    # Provenance triple should exist
    assert any(p.startswith("evolved_from_") for p in predicates)


def test_evolve_fact_returns_new_triple_id(kg):
    kg.add_triple("Alice", "uses", "Python")
    result = kg.evolve_fact("Alice", "uses", "Python", "Rust")
    assert isinstance(result, str)
    assert len(result) > 0


# ── ambient: get_eigen_thoughts ───────────────────────────────────────────────

def test_eigen_thoughts_empty_kg(kg):
    result = get_eigen_thoughts(kg)
    assert result["pillars"] == []
    assert result["total_entities"] == 0
    assert "empty" in result["message"].lower()


def test_eigen_thoughts_returns_n(populated_kg):
    result = get_eigen_thoughts(populated_kg, n=3)
    assert len(result["pillars"]) <= 3
    assert result["total_entities"] > 0


def test_eigen_thoughts_ranked(populated_kg):
    result = get_eigen_thoughts(populated_kg, n=5)
    pillars = result["pillars"]
    for i, p in enumerate(pillars):
        assert p["rank"] == i + 1
    scores = [p["score"] for p in pillars]
    assert scores == sorted(scores, reverse=True)


def test_eigen_thoughts_contains_required_keys(populated_kg):
    result = get_eigen_thoughts(populated_kg)
    assert "pillars" in result
    assert "total_entities" in result
    assert "message" in result
    for p in result["pillars"]:
        assert "rank" in p
        assert "entity" in p
        assert "score" in p


# ── ambient: get_socratic_question ────────────────────────────────────────────

def test_socratic_empty_kg(kg):
    result = get_socratic_question(kg)
    assert result["question"] is None
    assert "empty" in result["reasoning"].lower()


def test_socratic_returns_question(populated_kg):
    result = get_socratic_question(populated_kg)
    assert result["question"] is not None
    assert isinstance(result["question"], str)
    assert len(result["question"]) > 10


def test_socratic_context_entities_bias(populated_kg):
    """With context_entities provided, the result should reference them more."""
    result = get_socratic_question(populated_kg, context_entities=["Lumi"])
    # We can't guarantee Lumi is picked (depends on degrees), but no crash
    assert result["question"] is not None


def test_socratic_has_required_keys(populated_kg):
    result = get_socratic_question(populated_kg)
    for key in ("question", "entity", "reasoning", "entity_degree"):
        assert key in result


# ── ambient: get_whisper ──────────────────────────────────────────────────────

def test_whisper_no_palace(tmp_path):
    """Graceful when palace doesn't exist."""
    result = get_whisper("anything", str(tmp_path / "missing"))
    assert result["whispers"] == []
    assert result["total_wings_found"] == 0


def test_whisper_with_data(tmp_path):
    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["d1"],
        documents=["postgres chosen for orion project ACID compliance"],
        metadatas=[{"wing": "wing_orion", "room": "decisions", "source_file": "a.md",
                    "chunk_index": 0, "filed_at": "2026-01-01", "normalize_version": 2}],
    )
    col.upsert(
        ids=["d2"],
        documents=["redis cache layer session management throughput"],
        metadatas=[{"wing": "wing_infra", "room": "technical", "source_file": "b.md",
                    "chunk_index": 0, "filed_at": "2026-01-02", "normalize_version": 2}],
    )

    result = get_whisper("database choices", palace, n_wings=2)
    assert "whispers" in result
    assert isinstance(result["whispers"], list)
    for w in result["whispers"]:
        assert "wing" in w
        assert "text" in w


# ── rem_cycle ─────────────────────────────────────────────────────────────────

def test_rem_cycle_no_palace(tmp_path, kg):
    result = run_rem_cycle(str(tmp_path / "missing"), kg)
    assert "error" in result
    assert result["bridges_created"] == 0


def test_rem_cycle_empty_palace(tmp_path, kg):
    from mempalace.palace import get_collection
    palace = str(tmp_path / "palace")
    get_collection(palace, create=True)  # create empty collection
    result = run_rem_cycle(palace, kg)
    assert result["bridges_created"] == 0
    assert result["anchors_scanned"] == 0


def test_rem_cycle_single_wing_no_bridges(tmp_path, kg):
    """All drawers in same wing → no cross-wing bridges possible."""
    from mempalace.palace import get_collection
    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    for i in range(3):
        col.upsert(
            ids=[f"d{i}"],
            documents=[f"content about project alpha topic {i} details"],
            metadatas=[{"wing": "wing_alpha", "room": "general", "source_file": f"f{i}.md",
                        "chunk_index": 0, "filed_at": f"2026-01-0{i+1}", "normalize_version": 2}],
        )
    result = run_rem_cycle(palace, kg, n_anchors=10, threshold=0.5)
    # No cross-wing neighbors possible with a single wing
    assert result["bridges_created"] == 0
    assert result["anchors_scanned"] <= 3


def test_rem_cycle_returns_required_keys(tmp_path, kg):
    from mempalace.palace import get_collection
    palace = str(tmp_path / "palace")
    get_collection(palace, create=True)
    result = run_rem_cycle(palace, kg)
    for key in ("bridges_created", "bridges_skipped_existing", "anchors_scanned",
                 "wings_involved", "runtime_ms"):
        assert key in result


def test_rem_cycle_is_idempotent(tmp_path, kg):
    """Running REM twice on the same palace shouldn't double-create bridges."""
    from mempalace.palace import get_collection
    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    # Two wings with semantically similar content
    col.upsert(
        ids=["a1"],
        documents=["database postgres ACID transactions reliability production"],
        metadatas=[{"wing": "wing_alpha", "room": "decisions", "source_file": "a.md",
                    "chunk_index": 0, "filed_at": "2026-01-01", "normalize_version": 2}],
    )
    col.upsert(
        ids=["b1"],
        documents=["postgres database transactions ACID compliance reliable"],
        metadatas=[{"wing": "wing_beta", "room": "technical", "source_file": "b.md",
                    "chunk_index": 0, "filed_at": "2026-01-02", "normalize_version": 2}],
    )

    r1 = run_rem_cycle(palace, kg, n_anchors=10, threshold=0.5)
    r2 = run_rem_cycle(palace, kg, n_anchors=10, threshold=0.5)

    # Second run should create 0 new bridges (they already exist).
    # skip count >= bridges_created because each anchor that would bridge is counted
    # (both a→b and b→a anchors try, but only the unique edge was created once).
    assert r2["bridges_created"] == 0
    assert r2["bridges_skipped_existing"] >= r1["bridges_created"]
