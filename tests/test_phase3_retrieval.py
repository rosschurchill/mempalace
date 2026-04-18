"""Tests for Phase 3 retrieval intelligence: MMR, cross-wing balancing,
pipeline_trace, and the explain pipeline."""

from unittest.mock import MagicMock

from mempalace.searcher import mmr_rerank, _jaccard_sim, _cross_wing_balance
from mempalace.explain import (
    _extract_candidates,
    _detect_wing,
    _entity_to_wing_id,
    run_explain,
)


# ── MMR ──────────────────────────────────────────────────────────────────────


def _make_result(text: str, similarity: float, wing: str = "wing_a", distance: float = None):
    if distance is None:
        distance = round(1.0 - similarity, 3)
    return {
        "text": text,
        "similarity": similarity,
        "distance": distance,
        "effective_distance": distance,
        "wing": wing,
        "room": "general",
        "source_file": "test.md",
        "created_at": "",
        "closet_boost": 0.0,
        "matched_via": "drawer",
        "bm25_score": 0.0,
    }


def test_mmr_passthrough_when_results_le_n_final():
    """No reordering when result count <= n_final."""
    results = [_make_result(f"doc {i}", 0.9 - i * 0.1) for i in range(3)]
    out = mmr_rerank(results, n_final=5)
    assert out == results  # unchanged


def test_mmr_returns_n_final():
    results = [_make_result(f"unique doc number {i}", 0.9 - i * 0.01) for i in range(10)]
    out = mmr_rerank(results, n_final=4)
    assert len(out) == 4


def test_mmr_first_pick_is_highest_similarity():
    """When MMR runs, first selection must be the highest-similarity result."""
    results = [
        _make_result("doc about cats", 0.5),
        _make_result("doc about dogs", 0.9),
        _make_result("doc about fish", 0.7),
        _make_result("doc about birds", 0.4),
    ]
    # n_final < len(results) so MMR actually runs (no early-return passthrough)
    out = mmr_rerank(results, n_final=2)
    assert out[0]["text"] == "doc about dogs"


def test_mmr_avoids_duplicate_content():
    """Near-identical docs should not both rank in top-2 with lambda < 1."""
    text_a = "postgres database chosen for orion project acid compliance"
    text_b = "postgres was chosen for the orion project due to acid compliance requirements"
    text_c = "redis cache layer added to handle session state at high throughput"

    results = [
        _make_result(text_a, 0.95),
        _make_result(text_b, 0.92),  # near-duplicate of text_a
        _make_result(text_c, 0.80),  # unrelated
    ]
    out = mmr_rerank(results, n_final=2, mmr_lambda=0.5)
    # With diversity weighting, text_c should beat text_b for slot 2
    assert out[0]["text"] == text_a
    assert out[1]["text"] == text_c


def test_mmr_lambda_1_is_pure_relevance():
    """lambda=1.0 should return results in original similarity order."""
    results = [
        _make_result("alpha", 0.9),
        _make_result("beta", 0.8),
        _make_result("gamma", 0.7),
        _make_result("delta", 0.6),
    ]
    out = mmr_rerank(results, n_final=3, mmr_lambda=1.0)
    texts = [r["text"] for r in out]
    assert texts == ["alpha", "beta", "gamma"]


def test_jaccard_identical():
    assert _jaccard_sim("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert _jaccard_sim("cat dog", "fish bird") == 0.0


def test_jaccard_partial():
    score = _jaccard_sim("cat dog fish", "cat bird fish")
    assert 0.0 < score < 1.0


def test_jaccard_empty():
    assert _jaccard_sim("", "hello") == 0.0


# ── Cross-wing balancing ──────────────────────────────────────────────────────


def test_balance_single_wing_passthrough():
    """No balancing when only one wing present."""
    results = [_make_result(f"doc {i}", 0.9, wing="wing_a") for i in range(10)]
    out = _cross_wing_balance(results, n_results=5)
    assert out == results


def test_balance_limits_dominant_wing():
    """A wing with 8 results shouldn't take all 5 slots."""
    results = [_make_result(f"big wing {i}", 0.9 - i * 0.01, wing="wing_big") for i in range(8)] + [
        _make_result(f"small wing {i}", 0.7 - i * 0.01, wing="wing_small") for i in range(2)
    ]
    out = _cross_wing_balance(results, n_results=5)
    wings = [r["wing"] for r in out]
    assert "wing_small" in wings, "Small wing should appear in balanced results"


def test_balance_preserves_best_from_each_wing():
    """Best results from each wing should survive."""
    results = (
        [_make_result(f"a{i}", 0.9 - i * 0.1, wing="wing_a") for i in range(3)]
        + [_make_result(f"b{i}", 0.8 - i * 0.1, wing="wing_b") for i in range(3)]
        + [_make_result(f"c{i}", 0.7 - i * 0.1, wing="wing_c") for i in range(3)]
    )
    out = _cross_wing_balance(results, n_results=3)
    wings_present = {r["wing"] for r in out}
    # All three wings should appear in the balanced result
    assert wings_present == {"wing_a", "wing_b", "wing_c"}


def test_balance_empty_input():
    assert _cross_wing_balance([], n_results=5) == []


# ── pipeline_trace ────────────────────────────────────────────────────────────


def test_search_memories_returns_pipeline_trace(tmp_path):
    """search_memories() results should include pipeline_trace dict."""
    from mempalace.searcher import search_memories
    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["d1"],
        documents=["postgres was chosen for acid compliance"],
        metadatas=[
            {
                "wing": "wing_orion",
                "room": "decisions",
                "source_file": "t.md",
                "chunk_index": 0,
                "filed_at": "2026-01-01",
                "normalize_version": 2,
            }
        ],
    )

    result = search_memories("why postgres", palace, n_results=1)
    assert "results" in result
    if result["results"]:
        hit = result["results"][0]
        assert "pipeline_trace" in hit
        trace = hit["pipeline_trace"]
        assert "vector_sim" in trace
        assert "bm25_score" in trace
        assert "closet_boost" in trace
        assert "effective_distance" in trace


# ── explain pipeline ──────────────────────────────────────────────────────────


def test_extract_candidates_basic():
    candidates = _extract_candidates("Why did we choose Postgres for Orion?")
    assert "Postgres" in candidates
    assert "Orion" in candidates


def test_extract_candidates_filters_stopwords():
    candidates = _extract_candidates("Why did we choose this after the Monday meeting?")
    assert "Why" not in candidates
    assert "Monday" not in candidates


def test_extract_candidates_deduplicates():
    candidates = _extract_candidates("Orion Orion Orion")
    assert candidates.count("Orion") == 1


def test_entity_to_wing_id():
    assert _entity_to_wing_id("Orion") == "orion"
    assert _entity_to_wing_id("My Project") == "my_project"
    assert _entity_to_wing_id("Max-Well") == "max_well"


def test_detect_wing_finds_match():
    wings = {"wing_orion", "wing_alice", "wing_general"}
    detected = _detect_wing(["Orion", "Postgres"], wings)
    assert detected == "wing_orion"


def test_detect_wing_returns_none_on_miss():
    wings = {"wing_orion", "wing_alice"}
    detected = _detect_wing(["Unknown"], wings)
    assert detected is None


def test_detect_wing_first_match_wins():
    wings = {"wing_alice", "wing_bob"}
    # Alice comes first in candidates
    detected = _detect_wing(["Alice", "Bob"], wings)
    assert detected == "wing_alice"


def test_run_explain_no_palace(tmp_path):
    """run_explain gracefully handles no palace."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.entity_registry import EntityRegistry

    palace = str(tmp_path / "missing_palace")

    col_mock = MagicMock()
    col_mock.count.return_value = 0
    col_mock.get.return_value = {"metadatas": []}

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    registry = EntityRegistry.load(config_dir=tmp_path)

    result = run_explain(
        query="Why did we choose Postgres?",
        palace_path=palace,
        col=col_mock,
        kg=kg,
        entity_registry=registry,
    )
    # Should return a dict — either results or an error key
    assert isinstance(result, dict)
    assert "query" in result


def test_run_explain_with_data(tmp_path):
    """run_explain returns structured response with entities, kg_facts, results."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.entity_registry import EntityRegistry
    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["d1"],
        documents=["We chose Postgres for Orion because of ACID compliance and backup tooling."],
        metadatas=[
            {
                "wing": "wing_orion",
                "room": "decisions",
                "source_file": "t.md",
                "chunk_index": 0,
                "filed_at": "2026-01-01",
                "normalize_version": 2,
            }
        ],
    )

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Orion", "uses", "Postgres", valid_from="2025-11-01")

    registry = EntityRegistry.load(config_dir=tmp_path)

    result = run_explain(
        query="Why did we choose Postgres for Orion?",
        palace_path=palace,
        col=col,
        kg=kg,
        entity_registry=registry,
    )

    assert result["query"] == "Why did we choose Postgres for Orion?"
    assert "Orion" in result["entities_detected"] or "Postgres" in result["entities_detected"]
    assert isinstance(result["results"], list)
    assert isinstance(result["kg_facts"], list)
    assert "reasoning" in result
    # KG fact should be present
    kg_predicates = [f["predicate"] for f in result["kg_facts"]]
    assert "uses" in kg_predicates


def test_run_explain_falls_back_to_unscoped_when_scoped_returns_empty(tmp_path, monkeypatch):
    """Scoped search that returns zero hits must retry unscoped, with
    reasoning that names the attempted wing. Otherwise callers get empty
    results plus misleading 'auto-scoped to wing_X' reasoning and no
    signal that a broader search exists."""
    from mempalace import searcher
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.entity_registry import EntityRegistry

    # Force a wing match for 'Orion' without touching the embedder.
    # run_explain reads wings from col via _get_all_wings; stub that surface
    # so 'wing_orion' is considered an existing wing.
    col_mock = MagicMock()
    col_mock.count.return_value = 1
    col_mock.get.return_value = {"metadatas": [{"wing": "wing_orion"}]}

    # Fake search: return empty when scoped, non-empty when unscoped.
    calls: list = []

    def fake_search(query, palace_path, wing=None, room=None, n_results=5, **kwargs):
        calls.append({"wing": wing, "room": room})
        if wing is not None:
            return {"results": []}
        return {
            "results": [
                {
                    "text": "Orion is a project using Postgres",
                    "similarity": 0.8,
                    "wing": "wing_general",
                }
            ]
        }

    monkeypatch.setattr(searcher, "search_memories", fake_search)

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    registry = EntityRegistry.load(config_dir=tmp_path)

    result = run_explain(
        query="What did we decide about Orion?",
        palace_path=str(tmp_path / "palace"),
        col=col_mock,
        kg=kg,
        entity_registry=registry,
    )

    # Two searches: one scoped to wing_orion, one unscoped fallback
    assert len(calls) == 2
    assert calls[0]["wing"] == "wing_orion"
    assert calls[1]["wing"] is None

    # Response advertises the fallback explicitly
    assert result.get("scope_fallback") is True
    assert result.get("wing_scope_attempted") == "wing_orion"
    # wing_scope reflects what actually produced the returned hits (unscoped)
    assert result["wing_scope"] is None
    assert "fell back to unscoped" in result["reasoning"]
    assert len(result["results"]) == 1


def test_run_explain_no_fallback_when_scoped_has_hits(tmp_path, monkeypatch):
    """Sanity check: when the scoped search finds hits, no fallback happens
    and the response does not gain the fallback fields."""
    from mempalace import searcher
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.entity_registry import EntityRegistry

    col_mock = MagicMock()
    col_mock.count.return_value = 1
    col_mock.get.return_value = {"metadatas": [{"wing": "wing_orion"}]}

    calls: list = []

    def fake_search(query, palace_path, wing=None, room=None, n_results=5, **kwargs):
        calls.append({"wing": wing, "room": room})
        return {"results": [{"text": "scoped hit", "similarity": 0.9, "wing": "wing_orion"}]}

    monkeypatch.setattr(searcher, "search_memories", fake_search)

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    registry = EntityRegistry.load(config_dir=tmp_path)

    result = run_explain(
        query="Orion status",
        palace_path=str(tmp_path / "palace"),
        col=col_mock,
        kg=kg,
        entity_registry=registry,
    )

    assert len(calls) == 1
    assert calls[0]["wing"] == "wing_orion"
    assert "scope_fallback" not in result
    assert "wing_scope_attempted" not in result
    assert result["wing_scope"] == "wing_orion"
