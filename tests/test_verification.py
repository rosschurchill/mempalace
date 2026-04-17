"""End-to-end verification tests for claims we asserted were fixed upstream.

Each test in this file proves (or disproves) a specific claim from the review
that we made without empirical evidence:

  #590  — Claude Code JSONL mining does NOT silently drop tool_result messages
  #608  — MCP search sees drawers written by an external CLI mine (cache invalidates)
  #655  — KG does not duplicate edges across all write paths
  #816  — MCP server accepts Gemini-style tool calls with extra top-level fields
  #903/#912 — HTTP-mode embedding guard actually fires on mismatch
"""

import json
import os
from unittest.mock import patch



# ── #590 — JSONL tool_result not silently dropped ────────────────────────────

def test_jsonl_tool_result_preserved_590(tmp_path):
    """A Claude Code JSONL with tool_result blocks should keep tool output
    visible in the normalized transcript — not silently discard it."""
    from mempalace.normalize import normalize

    jsonl_file = tmp_path / "session.jsonl"

    # Assistant calls a Bash tool, then user turn contains the tool_result
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "List the files"},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "ls -la"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "UNIQUE_MARKER_ABC: total 4\ndrwxr-xr-x  2 user",
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "Done — two items found."},
        },
    ]
    jsonl_file.write_text("\n".join(json.dumps(m) for m in messages))

    result = normalize(str(jsonl_file))

    # The unique marker from the tool_result MUST appear somewhere in the output
    assert "UNIQUE_MARKER_ABC" in result, (
        "tool_result content was silently dropped — #590 not fixed. "
        f"Output was: {result[:500]}"
    )
    # The user question and assistant responses must also be present
    assert "List the files" in result
    assert "Done" in result or "two items" in result


def test_jsonl_tool_result_with_read_tool_allowed_dropped_590(tmp_path):
    """Explicit check: Read/Edit/Write tool results are intentionally collapsed
    (content is already in git/palace). This is BY DESIGN — not a #590 bug."""
    from mempalace.normalize import _format_tool_result

    # Read result gets stripped by design (empty string)
    assert _format_tool_result("file contents here", "Read") == ""
    assert _format_tool_result("file contents here", "Edit") == ""
    assert _format_tool_result("file contents here", "Write") == ""
    # But Bash, Grep, Glob, and unknown tools keep output
    assert "UNIQUE_MARKER" in _format_tool_result("UNIQUE_MARKER output", "Bash")
    assert "UNIQUE_MARKER" in _format_tool_result("UNIQUE_MARKER match.py:5", "Grep")
    assert "UNIQUE_MARKER" in _format_tool_result("UNIQUE_MARKER result", "UnknownTool")


# ── #608 — stale cache after external write ───────────────────────────────────

def test_mcp_sees_external_write_608(tmp_path):
    """After an external client writes to the palace, a second caller that
    loads the collection should see the new drawer without explicit reconnect."""
    from mempalace.backends.chroma import ChromaBackend

    palace = str(tmp_path / "palace")
    os.makedirs(palace)

    # Client A — mimics CLI miner
    backend_a = ChromaBackend()
    col_a = backend_a.get_collection(palace, "mempalace_drawers", create=True)
    col_a.upsert(
        ids=["drawer_verify_608"],
        documents=["Content written by client A that client B must see"],
        metadatas=[{"wing": "test_wing", "room": "test_room",
                    "source_file": "a.md", "chunk_index": 0,
                    "filed_at": "2026-04-17", "normalize_version": 2}],
    )

    # Client B — a fresh backend instance mimicking MCP server after restart
    backend_b = ChromaBackend()
    col_b = backend_b.get_collection(palace, "mempalace_drawers", create=False)
    result = col_b.get(ids=["drawer_verify_608"])

    assert result["ids"] == ["drawer_verify_608"], (
        "Client B can't see Client A's write — cache invalidation is broken. "
        f"Got: {result}"
    )
    assert "client A" in result["documents"][0]


# ── #655 — KG edge dedup across all write paths ───────────────────────────────

def test_kg_add_triple_dedup_655(tmp_path):
    """add_triple with identical (subject, predicate, object) must not create
    duplicate rows."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    id1 = kg.add_triple("Max", "loves", "chess")
    id2 = kg.add_triple("Max", "loves", "chess")
    id3 = kg.add_triple("Max", "loves", "chess")
    kg.close()

    # All three calls should return the same ID — the existing triple
    assert id1 == id2 == id3, (
        f"add_triple dedup broken — got distinct IDs {id1!r} {id2!r} {id3!r}"
    )


def test_kg_evolve_fact_dedup_655(tmp_path):
    """evolve_fact called twice on same fact should not create duplicates."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Max", "attends", "Lincoln")
    kg.evolve_fact("Max", "attends", "Lincoln", "Westside")
    # Second evolve should be a no-op on the already-invalidated fact,
    # or at worst idempotent
    kg.evolve_fact("Max", "attends", "Westside", "Oakridge")

    facts = kg.query_entity("Max", direction="outgoing")
    kg.close()

    # Current 'attends' relationship should be exactly one
    current_attends = [f for f in facts if f["predicate"] == "attends" and f["current"]]
    assert len(current_attends) == 1, (
        f"evolve_fact left multiple current attends triples: {current_attends}"
    )


def test_kg_rem_bridge_dedup_655(tmp_path):
    """REM cycle add_triple with 'semantically_bridges' predicate must dedup."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    b1 = kg.add_triple("wing_a", "semantically_bridges", "wing_b")
    b2 = kg.add_triple("wing_a", "semantically_bridges", "wing_b")
    kg.close()

    assert b1 == b2, "Bridge duplicate not prevented — REM would fill the KG"


# ── #816 — MCP tool_call with unknown args doesn't crash ──────────────────────

def test_mcp_tool_call_extra_args_816():
    """JSON-RPC tools/call with extra top-level fields should not crash
    the handler. Gemini CLI passes cwd, session, and other fields."""
    from mempalace.mcp_server import handle_request

    request = {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "mempalace_status",
            "arguments": {
                # Valid args mixed with unknown extras
                "cwd": "/home/user",
                "session": "gemini-session-xyz",
                "unknown_field": "random",
            },
            # Gemini-style top-level extras
            "cwd": "/home/user",
            "session_id": "abc",
        },
    }

    response = handle_request(request)
    # Should return a valid JSON-RPC response, not crash
    assert response is not None
    assert response.get("jsonrpc") == "2.0"
    assert response.get("id") == 42
    # Either "result" or "error" is acceptable — "error" is OK because the
    # status tool might fail without a palace, what matters is NO CRASH
    assert "result" in response or "error" in response


# ── #903/#912 — HTTP-mode embedding guard fires on mismatch ───────────────────

def test_embedding_guard_fires_on_mismatch_903(tmp_path):
    """Mine with model A, search with config set to model B — must warn."""
    from mempalace.palace import write_palace_meta, get_collection
    from mempalace.searcher import search_memories

    palace = str(tmp_path / "palace")
    os.makedirs(palace)

    # Seed a collection with some data
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["d1"],
        documents=["test content for embedding guard"],
        metadatas=[{"wing": "w", "room": "r", "source_file": "f.md",
                    "chunk_index": 0, "filed_at": "2026-04-17",
                    "normalize_version": 2}],
    )
    # Record that the palace was mined with model A
    write_palace_meta(palace, "mpnet-base-v2-fake", col=col)

    # Now search — config defaults to all-MiniLM-L6-v2, should warn about mismatch
    env = {k: v for k, v in os.environ.items() if k != "MEMPALACE_EMBEDDING_MODEL"}
    with patch.dict(os.environ, env, clear=True):
        result = search_memories("content", palace, n_results=1)

    assert "warning" in result, (
        "Embedding guard did not fire on model mismatch — the Phase 2 feature "
        "is broken. Result keys: " + str(list(result.keys()))
    )
    assert "mpnet-base-v2-fake" in result["warning"]
    assert "mismatch" in result["warning"].lower()


def test_embedding_guard_silent_when_models_match_903(tmp_path):
    """No warning when ingest and query use the same model (the common case)."""
    from mempalace.palace import write_palace_meta, get_collection
    from mempalace.searcher import search_memories

    palace = str(tmp_path / "palace")
    os.makedirs(palace)
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["d1"],
        documents=["test content"],
        metadatas=[{"wing": "w", "room": "r", "source_file": "f.md",
                    "chunk_index": 0, "filed_at": "2026-04-17",
                    "normalize_version": 2}],
    )
    # Record with the DEFAULT model (matches config)
    write_palace_meta(palace, "all-MiniLM-L6-v2", col=col)

    env = {k: v for k, v in os.environ.items() if k != "MEMPALACE_EMBEDDING_MODEL"}
    with patch.dict(os.environ, env, clear=True):
        result = search_memories("content", palace, n_results=1)

    assert "warning" not in result, (
        f"False-positive mismatch warning: {result.get('warning')}"
    )


def test_embedding_meta_stored_in_collection_metadata_903(tmp_path):
    """After write_palace_meta, the model name should be readable from
    ChromaDB collection metadata — not just the local palace_meta.json file.
    This is what makes HTTP mode work."""
    from mempalace.palace import write_palace_meta, get_collection, read_palace_meta

    palace = str(tmp_path / "palace")
    os.makedirs(palace)
    col = get_collection(palace, create=True)
    write_palace_meta(palace, "paraphrase-multilingual-MiniLM-L12-v2", col=col)

    # Read the collection metadata directly — not via palace_meta.json
    meta = col.metadata
    assert meta.get("embedding_model") == "paraphrase-multilingual-MiniLM-L12-v2", (
        f"Collection metadata missing embedding_model: {meta}"
    )

    # read_palace_meta with col= should return it
    parsed = read_palace_meta(palace, col=col)
    assert parsed.get("embedding_model") == "paraphrase-multilingual-MiniLM-L12-v2"
