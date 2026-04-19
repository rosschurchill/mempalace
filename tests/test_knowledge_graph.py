"""
test_knowledge_graph.py — Tests for the temporal knowledge graph.

Covers: entity CRUD, triple CRUD, temporal queries, invalidation,
timeline, stats, and edge cases (duplicate triples, ID collisions).
"""


class TestEntityOperations:
    def test_add_entity(self, kg):
        eid = kg.add_entity("Alice", entity_type="person")
        assert eid == "alice"

    def test_add_entity_normalizes_id(self, kg):
        eid = kg.add_entity("Dr. Chen", entity_type="person")
        assert eid == "dr._chen"

    def test_add_entity_upsert(self, kg):
        kg.add_entity("Alice", entity_type="person")
        kg.add_entity("Alice", entity_type="engineer")
        # Should not raise — INSERT OR REPLACE
        stats = kg.stats()
        assert stats["entities"] == 1


class TestTripleOperations:
    def test_add_triple_creates_entities(self, kg):
        tid = kg.add_triple("Alice", "knows", "Bob")
        assert tid.startswith("t_alice_knows_bob_")
        stats = kg.stats()
        assert stats["entities"] == 2  # auto-created

    def test_add_triple_with_dates(self, kg):
        tid = kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
        assert tid.startswith("t_max_does_swimming_")

    def test_duplicate_triple_returns_existing_id(self, kg):
        tid1 = kg.add_triple("Alice", "knows", "Bob")
        tid2 = kg.add_triple("Alice", "knows", "Bob")
        assert tid1 == tid2

    def test_invalidated_triple_allows_re_add(self, kg):
        tid1 = kg.add_triple("Alice", "works_at", "Acme")
        kg.invalidate("Alice", "works_at", "Acme", ended="2025-01-01")
        tid2 = kg.add_triple("Alice", "works_at", "Acme")
        assert tid1 != tid2  # new triple since old one was closed


class TestQueries:
    def test_query_outgoing(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", direction="outgoing")
        predicates = {r["predicate"] for r in results}
        assert "parent_of" in predicates
        assert "works_at" in predicates

    def test_query_incoming(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="incoming")
        assert any(r["subject"] == "Alice" and r["predicate"] == "parent_of" for r in results)

    def test_query_both_directions(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="both")
        directions = {r["direction"] for r in results}
        assert "outgoing" in directions
        assert "incoming" in directions

    def test_query_as_of_filters_expired(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2023-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "Acme Corp" in employers
        assert "NewCo" not in employers

    def test_query_as_of_shows_current(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2025-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "NewCo" in employers
        assert "Acme Corp" not in employers

    def test_query_relationship(self, seeded_kg):
        results = seeded_kg.query_relationship("does")
        assert len(results) == 2  # swimming + chess


class TestInvalidation:
    def test_invalidate_sets_valid_to(self, seeded_kg):
        seeded_kg.invalidate("Max", "does", "chess", ended="2026-01-01")
        results = seeded_kg.query_entity("Max", direction="outgoing")
        chess = [r for r in results if r["object"] == "chess"]
        assert len(chess) == 1
        assert chess[0]["valid_to"] == "2026-01-01"
        assert chess[0]["current"] is False


class TestTimeline:
    def test_timeline_all(self, seeded_kg):
        tl = seeded_kg.timeline()
        assert len(tl) >= 4

    def test_timeline_entity(self, seeded_kg):
        tl = seeded_kg.timeline("Max")
        subjects_and_objects = {t["subject"] for t in tl} | {t["object"] for t in tl}
        assert "Max" in subjects_and_objects

    def test_timeline_global_has_limit(self, kg):
        # Add > 100 triples
        for i in range(105):
            kg.add_triple(f"entity_{i}", "relates_to", f"entity_{i + 1}")
        tl = kg.timeline()
        assert len(tl) == 100  # LIMIT 100

    def test_timeline_entity_has_limit(self, kg):
        # Add > 100 triples all connected to a single entity
        for i in range(105):
            kg.add_triple(
                "hub", "connects_to", f"spoke_{i}", valid_from=f"2025-01-{(i % 28) + 1:02d}"
            )
        tl = kg.timeline("hub")
        assert len(tl) == 100  # LIMIT 100 on entity-filtered branch


class TestWALMode:
    def test_wal_mode_enabled(self, kg):
        conn = kg._conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


class TestStats:
    def test_stats_empty(self, kg):
        stats = kg.stats()
        assert stats["entities"] == 0
        assert stats["triples"] == 0

    def test_stats_seeded(self, seeded_kg):
        stats = seeded_kg.stats()
        assert stats["entities"] >= 4
        assert stats["triples"] == 5
        assert stats["current_facts"] == 4  # 1 expired (Acme Corp)
        assert stats["expired_facts"] == 1


# ── KnowledgeGraph.diff() (#9.9) ──────────────────────────────────────────────


class TestDiff:
    def test_diff_added_in_window(self, tmp_path):
        """Triples added today appear in a diff window covering today."""
        from datetime import date
        from mempalace.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        g.add_triple("Alice", "works_on", "Orion", valid_from="2026-01-01")
        today = date.today().isoformat()
        result = g.diff(since=today)
        g.close()

        subjects = [r["subject"] for r in result["added"]]
        assert "Alice" in subjects

    def test_diff_invalidated_in_window(self, tmp_path):
        """Triples invalidated today appear in invalidated list."""
        from datetime import date
        from mempalace.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        g.add_triple("Bob", "uses", "OldLib", valid_from="2024-01-01")
        g.invalidate("Bob", "uses", "OldLib")
        today = date.today().isoformat()
        result = g.diff(since=today)
        g.close()

        subjects = [r["subject"] for r in result["invalidated"]]
        assert "Bob" in subjects

    def test_diff_empty_window(self, tmp_path):
        """A window in the far past returns empty lists."""
        from mempalace.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        g.add_triple("Carol", "knows", "Dave", valid_from="2026-01-01")
        result = g.diff(since="2000-01-01", until="2000-01-02")
        g.close()

        assert result["added"] == []
        assert result["invalidated"] == []

    def test_diff_result_keys(self, tmp_path):
        """diff() result must contain required top-level keys."""
        from mempalace.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        result = g.diff(since="2026-01-01")
        g.close()

        for key in ("added", "invalidated", "since", "until"):
            assert key in result

    def test_diff_added_entry_keys(self, tmp_path):
        """Each added entry must have the expected fields."""
        from datetime import date
        from mempalace.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        g.add_triple("Eve", "manages", "Project", valid_from="2026-01-01")
        result = g.diff(since=date.today().isoformat())
        g.close()

        assert result["added"], "expected at least one added triple"
        entry = result["added"][0]
        for key in ("triple_id", "subject", "predicate", "object", "valid_from", "extracted_at"):
            assert key in entry, f"missing key '{key}' in added entry"

    def test_diff_mcp_tool_registered(self):
        """mempalace_kg_diff must be in the TOOLS dict."""
        from mempalace.mcp_server import TOOLS
        assert "mempalace_kg_diff" in TOOLS

    def test_diff_mcp_tool_requires_since(self):
        """mempalace_kg_diff schema must require 'since'."""
        from mempalace.mcp_server import TOOLS
        schema = TOOLS["mempalace_kg_diff"]["input_schema"]
        assert "since" in schema.get("required", [])
