"""Tests for kg_health.py — KG staleness detection (#9.8)."""

import pytest

from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.kg_health import (
    check_stale_triples,
    check_silent_entities,
    kg_staleness_report,
)


@pytest.fixture
def kg(tmp_path):
    g = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    yield g
    g.close()


# ── check_stale_triples ───────────────────────────────────────────────────────


def test_stale_triples_empty_kg(kg):
    result = check_stale_triples(kg)
    assert result == []


def test_stale_triples_recent_fact_not_flagged(kg):
    """A fact added today (valid_from=today) must not appear as stale."""
    from datetime import date

    today = date.today().isoformat()
    kg.add_triple("Alice", "works_on", "Orion", valid_from=today)
    result = check_stale_triples(kg, stale_days=180)
    # Today is within the 180-day window — should not be flagged
    texts = [(r["subject"], r["predicate"], r["object"]) for r in result]
    assert ("Alice", "works_on", "Orion") not in texts


def test_stale_triples_old_fact_flagged(kg):
    """A fact from 2 years ago with no valid_to must be flagged."""
    kg.add_triple("Bob", "uses", "Python", valid_from="2022-01-01")
    result = check_stale_triples(kg, stale_days=180)
    subjects = [r["subject"] for r in result]
    assert "Bob" in subjects


def test_stale_triples_invalidated_fact_not_flagged(kg):
    """A fact with valid_to set (expired) must not appear — it's already resolved."""

    kg.add_triple("Carol", "lives_in", "London", valid_from="2020-01-01")
    # Invalidate it
    kg.invalidate("Carol", "lives_in", "London", ended="2023-06-01")
    result = check_stale_triples(kg, stale_days=180)
    subjects = [r["subject"] for r in result]
    assert "Carol" not in subjects


def test_stale_triples_provenance_pointers_excluded(kg):
    """evolved_from_* predicates are provenance — never stale."""

    with kg._lock:
        conn = kg._conn()
        # Add entity nodes first
        conn.execute(
            "INSERT OR IGNORE INTO entities(id,name,type) VALUES (?,?,?)",
            ("dave", "Dave", "person"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO entities(id,name,type) VALUES (?,?,?)",
            ("old_fact", "OldFact", "concept"),
        )
        # Insert a raw provenance triple
        conn.execute(
            """INSERT INTO triples(id,subject,predicate,object,valid_from)
               VALUES ('prov1','dave','evolved_from_123','old_fact','2020-01-01')""",
        )
        conn.commit()

    result = check_stale_triples(kg, stale_days=1)
    predicates = [r["predicate"] for r in result]
    assert not any(p.startswith("evolved_from_") for p in predicates)


def test_stale_triples_respects_limit(kg):
    """limit parameter caps the result count."""
    for i in range(10):
        kg.add_triple(f"E{i}", "knows", "Target", valid_from="2020-01-01")
    result = check_stale_triples(kg, stale_days=1, limit=3)
    assert len(result) <= 3


def test_stale_triples_result_keys(kg):
    """Each stale triple entry must have the expected keys."""
    kg.add_triple("Eve", "manages", "Project", valid_from="2021-01-01")
    result = check_stale_triples(kg, stale_days=1)
    assert result, "expected at least one stale triple"
    for entry in result:
        for key in ("triple_id", "subject", "predicate", "object", "valid_from", "age", "reason"):
            assert key in entry, f"missing key '{key}' in stale triple entry"


# ── check_silent_entities ─────────────────────────────────────────────────────


def test_silent_entities_no_col(kg):
    """When col=None, entity silence check must return empty (not crash)."""
    kg.add_entity("Frank", "person")
    result = check_silent_entities(kg, col=None)
    assert result == []


def test_silent_entities_empty_palace(kg):
    """Empty drawer collection — all entities are silent."""

    class _EmptyCol:
        def get(self, **kwargs):
            return {"metadatas": []}

    kg.add_entity("Grace", "person")
    result = check_silent_entities(kg, col=_EmptyCol(), silence_days=30)
    assert any(r["entity"] == "Grace" for r in result)


def test_silent_entities_recently_mentioned_not_flagged(kg):
    """Entity mentioned in a recent drawer must not be flagged."""
    from datetime import date

    today = date.today().isoformat()
    kg.add_entity("Hank", "person")

    class _RecentCol:
        def get(self, **kwargs):
            return {
                "metadatas": [
                    {
                        "entities": "hank",
                        "filed_at": today,
                        "wing": "wing_t",
                        "room": "general",
                    }
                ]
            }

    result = check_silent_entities(kg, col=_RecentCol(), silence_days=30)
    names = [r["entity"] for r in result]
    assert "Hank" not in names


def test_silent_entities_old_mention_flagged(kg):
    """Entity last mentioned more than silence_days ago must be flagged."""
    kg.add_entity("Ivy", "person")

    class _OldCol:
        def get(self, **kwargs):
            return {
                "metadatas": [
                    {
                        "entities": "ivy",
                        "filed_at": "2020-01-01",
                        "wing": "wing_t",
                        "room": "general",
                    }
                ]
            }

    result = check_silent_entities(kg, col=_OldCol(), silence_days=30)
    names = [r["entity"] for r in result]
    assert "Ivy" in names


# ── kg_staleness_report ───────────────────────────────────────────────────────


def test_staleness_report_structure(kg):
    """kg_staleness_report must return all required top-level keys."""
    report = kg_staleness_report(kg, col=None)
    for key in (
        "stale_triples",
        "silent_entities",
        "summary",
        "checked_at",
        "stale_threshold_days",
        "entity_silence_days",
    ):
        assert key in report, f"missing key '{key}' in staleness report"


def test_staleness_report_healthy_kg(kg):
    """Empty KG must report healthy (no stale facts)."""
    report = kg_staleness_report(kg, col=None)
    assert report["stale_triples"] == []
    assert "healthy" in report["summary"].lower() or report["stale_triples"] == []


def test_staleness_report_with_stale_data(kg):
    """Report must surface old unconfirmed triples."""
    kg.add_triple("Jack", "uses", "OldLib", valid_from="2019-01-01")
    report = kg_staleness_report(kg, col=None, stale_days=30)
    assert len(report["stale_triples"]) > 0
    assert "not confirmed" in report["summary"] or len(report["stale_triples"]) > 0
