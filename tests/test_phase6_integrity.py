"""Phase 6 data-integrity regression tests for issue #934 bundle + #929.

Each test corresponds to a specific item in the P1 #934 bundle, plus #929
(non-ASCII diary round-trip).
"""

import json
import os
from unittest.mock import patch

import pytest


# ── #934 item 4: tool_diary_write sanitizes topic ────────────────────────────

def test_diary_write_sanitizes_topic():
    """Unsanitized topic strings must be rejected, not silently stored."""
    from mempalace.mcp_server import tool_diary_write

    # Path traversal in topic — should be rejected
    result = tool_diary_write(
        agent_name="lumi",
        entry="test entry",
        topic="../../../etc/passwd",
    )
    assert result.get("success") is False or "invalid" in str(result).lower(), (
        f"Path traversal in topic was accepted: {result}"
    )


def test_diary_write_null_byte_topic_rejected():
    from mempalace.mcp_server import tool_diary_write
    result = tool_diary_write(
        agent_name="lumi",
        entry="test",
        topic="normal\x00topic",
    )
    assert result.get("success") is False


# ── #934 item 5: closet_llm retries on JSONDecodeError ───────────────────────

def test_closet_llm_retries_on_json_decode_error():
    """Malformed LLM responses should be retried up to 3 times, not
    silently skipped on the first failure."""
    from mempalace import closet_llm
    import urllib.request

    call_count = [0]

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return self.text.encode("utf-8")

    def fake_urlopen(req, timeout=30):
        call_count[0] += 1
        # First two attempts return malformed payloads, third returns valid
        if call_count[0] < 3:
            return FakeResponse("not valid json at all")
        return FakeResponse(json.dumps({
            "choices": [{"message": {"content": '{"topics": ["test"]}'}}],
            "usage": {"total_tokens": 10},
        }))

    cfg = closet_llm.LLMConfig(endpoint="http://fake", key="test", model="test")
    with patch.object(urllib.request, "urlopen", fake_urlopen):
        # Disable sleep so the test runs fast
        with patch.object(closet_llm.time, "sleep", lambda *a: None):
            parsed, _ = closet_llm._call_llm(cfg, "src.md", "w", "r", "test content")

    assert call_count[0] == 3, (
        f"Expected 3 attempts (retry on JSONDecodeError), got {call_count[0]}"
    )
    assert parsed is not None, "Should have succeeded on 3rd attempt"


# ── #934 item 6: tool_reconnect resets metadata cache ────────────────────────

def test_reconnect_resets_metadata_cache():
    """After tool_reconnect(), _metadata_cache must be None so the next
    read doesn't serve stale metadata for up to 5s."""
    from mempalace import mcp_server

    # Prime the cache
    mcp_server._metadata_cache = [{"wing": "old_data"}]
    mcp_server._metadata_cache_time = 12345.0

    # Simulate reconnect — but guard against connection errors
    try:
        mcp_server.tool_reconnect()
    except Exception:
        pass

    assert mcp_server._metadata_cache is None, (
        "Metadata cache not reset after reconnect — stale data can leak"
    )
    assert mcp_server._metadata_cache_time == 0, (
        "Metadata cache timestamp not reset"
    )


# ── #934 item 7: exporter refuses symlinked output paths ─────────────────────

def test_exporter_refuses_symlinked_output(tmp_path):
    """If output_dir is a symlink, exporter must refuse rather than follow it."""
    from mempalace.exporter import export_palace

    target_dir = tmp_path / "real_target"
    target_dir.mkdir()
    symlink_dir = tmp_path / "attacker_symlink"
    symlink_dir.symlink_to(target_dir)

    # Need a real palace to export — but we can use an empty one
    palace = str(tmp_path / "palace")
    os.makedirs(palace)
    from mempalace.palace import get_collection
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["d1"],
        documents=["test"],
        metadatas=[{"wing": "w", "room": "r", "source_file": "f.md",
                    "chunk_index": 0, "filed_at": "2026-04-17",
                    "normalize_version": 2}],
    )

    with pytest.raises(ValueError, match="symlink"):
        export_palace(palace, str(symlink_dir))


# ── #929: diary_write round-trip with non-ASCII ──────────────────────────────

def test_diary_write_read_non_ascii_round_trip(tmp_path):
    """Verify that our ensure_ascii=False fix (Phase 0) makes non-ASCII
    diary entries round-trip correctly (#929)."""
    from mempalace import mcp_server
    from mempalace.palace import get_collection

    # Redirect palace to tmp and reset caches
    mcp_server._config = type(mcp_server._config)()
    mcp_server._config._file_config = {"palace_path": str(tmp_path / "palace")}
    mcp_server._collection_cache = None
    mcp_server._client_cache = None
    mcp_server._metadata_cache = None

    os.makedirs(str(tmp_path / "palace"))
    get_collection(str(tmp_path / "palace"), create=True)

    non_ascii_entry = (
        "今天我们讨论了架构设计。 "
        "Decisions: Postgres → ACID compliance ✓. "
        "Alternatives rejected: MongoDB ✗, Redis ✗. "
        "Action items: ★ Update schema, ★ Run migrations."
    )

    write_result = mcp_server.tool_diary_write(
        agent_name="test_agent",
        entry=non_ascii_entry,
        topic="architecture",
    )
    assert write_result.get("success") is True, f"Write failed: {write_result}"

    read_result = mcp_server.tool_diary_read(agent_name="test_agent", last_n=1)
    assert "entries" in read_result
    assert len(read_result["entries"]) >= 1, f"No entries read back: {read_result}"

    content = read_result["entries"][0]["content"]
    # All non-ASCII chars must round-trip exactly
    assert "今天" in content, "Chinese chars lost"
    assert "→" in content, "Arrow char lost"
    assert "★" in content, "Star char lost"
    assert "✓" in content and "✗" in content, "Check/cross chars lost"
