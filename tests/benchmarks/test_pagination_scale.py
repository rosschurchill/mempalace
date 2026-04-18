"""
Pagination scale tests (#7.4) — verify _fetch_all_metadata and the MCP
status/list_wings tools handle 100K+ drawers without crashing or truncating.

Uses a fake collection so no embedder or real ChromaDB is needed.
The benchmark marker keeps this out of the standard test run
(`--ignore=tests/benchmarks`), but the logic is real: the fake collection
exercises every code path in the pagination loop.
"""

import time

import pytest

from mempalace.mcp_server import _fetch_all_metadata


# ── Fake collection ───────────────────────────────────────────────────────────

WINGS = [f"wing_{chr(97 + i)}" for i in range(10)]  # wing_a … wing_j


class _FakeCol:
    """Simulates a ChromaDB collection with N drawers, served in pages."""

    def __init__(self, n: int, page_size: int = 1000):
        self._n = n
        self._page_size = page_size
        self._metas = [
            {"wing": WINGS[i % len(WINGS)], "room": "general", "source_file": f"f{i}.md"}
            for i in range(n)
        ]

    def count(self) -> int:
        return self._n

    def get(self, include=None, limit=1000, offset=0, **kwargs):
        page = self._metas[offset : offset + limit]
        return {"metadatas": page, "ids": [f"id_{offset + j}" for j in range(len(page))]}


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.benchmark
class TestPaginationScale:
    """Verify pagination handles large palaces correctly and quickly."""

    N = 150_000  # threshold from notes/06 item 7.4
    WALL_CLOCK_BUDGET_S = 5.0

    def test_fetch_all_metadata_count(self):
        """_fetch_all_metadata must return exactly N records at 150K drawers."""
        col = _FakeCol(self.N)
        result = _fetch_all_metadata(col)
        assert len(result) == self.N, (
            f"Expected {self.N} metadata records, got {len(result)}"
        )

    def test_fetch_all_metadata_no_truncation(self):
        """Every page must be consumed — no silent truncation."""
        col = _FakeCol(self.N, page_size=1000)
        result = _fetch_all_metadata(col)
        wings_seen = {m["wing"] for m in result}
        # All 10 wings should appear across 150K drawers
        assert wings_seen == set(WINGS), (
            f"Some wings missing — pagination may have stopped early: {wings_seen}"
        )

    def test_fetch_all_metadata_wall_clock(self):
        """Pagination over 150K fake records must complete in < 5 seconds."""
        col = _FakeCol(self.N)
        start = time.perf_counter()
        _fetch_all_metadata(col)
        elapsed = time.perf_counter() - start
        assert elapsed < self.WALL_CLOCK_BUDGET_S, (
            f"_fetch_all_metadata took {elapsed:.2f}s > {self.WALL_CLOCK_BUDGET_S}s budget"
        )

    def test_fetch_all_metadata_odd_size(self):
        """Non-round collection size must not lose the last partial page."""
        n = 150_003  # 150 full pages + 3 stragglers
        col = _FakeCol(n)
        result = _fetch_all_metadata(col)
        assert len(result) == n

    def test_tool_status_returns_correct_count(self, monkeypatch):
        """tool_status must report total_drawers == N even for large palaces."""
        import os
        from mempalace import mcp_server

        col = _FakeCol(self.N)

        # Patch internals so tool_status uses our fake collection
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: col)
        monkeypatch.setattr(mcp_server, "_metadata_cache", None)
        monkeypatch.setattr(mcp_server, "_metadata_cache_time", 0)
        # tool_status checks for chroma.sqlite3 before calling _get_collection
        monkeypatch.setattr(os.path, "isfile", lambda p: True)

        result = mcp_server.tool_status()
        assert result["total_drawers"] == self.N, (
            f"Expected total_drawers={self.N}, got {result.get('total_drawers')}"
        )
        assert "error" not in result

    def test_tool_list_wings_returns_all_wings(self, monkeypatch):
        """tool_list_wings must enumerate all wings at 150K drawers."""
        from mempalace import mcp_server

        col = _FakeCol(self.N)

        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: col)
        monkeypatch.setattr(mcp_server, "_metadata_cache", None)
        monkeypatch.setattr(mcp_server, "_metadata_cache_time", 0)

        result = mcp_server.tool_list_wings()
        assert "wings" in result
        assert set(result["wings"].keys()) == set(WINGS), (
            f"Expected {set(WINGS)}, got {set(result['wings'].keys())}"
        )
        total = sum(result["wings"].values())
        assert total == self.N
