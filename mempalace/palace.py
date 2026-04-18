"""
palace.py — Shared palace operations.

Consolidates collection access patterns used by both miners and the MCP server.
"""

import contextlib
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from .backends.chroma import ChromaBackend


# ── Palace metadata ───────────────────────────────────────────────────────────
# palace_meta.json records embedding model info at mine time so the search path
# can warn when the query model differs from the ingest model (issue #903/#912).

_PALACE_META_FILE = "palace_meta.json"


def write_palace_meta(palace_path: str, embedding_model: str, col=None) -> None:
    """Record embedding model at mine time for drift detection (#903/#912).

    Writes to BOTH (when both targets are reachable):
      1. ChromaDB collection metadata (authoritative — works in both HTTP and
         embedded modes, survives palace moves, readable by any client).
      2. <palace_path>/palace_meta.json (embedded mode only — local cache so
         status/repair tools can read model info without opening the collection).

    Callers that already have an open collection should pass it as ``col`` to
    avoid an extra get_collection round-trip. When ``col`` is None, this
    function opens the collection itself (create=True — same behaviour as
    the miner which is the primary caller).

    HTTP mode: only #1 happens. Both writes are idempotent and non-fatal.
    """
    http_mode = bool(os.environ.get("MEMPALACE_CHROMA_URL"))
    now = datetime.now().isoformat()

    # 1. Write to ChromaDB collection metadata (source of truth)
    try:
        target_col = col if col is not None else get_collection(palace_path, create=True)
        target_col.set_metadata(embedding_model=embedding_model, last_mined=now)
    except Exception:
        pass  # non-fatal — metadata write failure must never abort a mine run

    # 2. Local cache file — embedded mode only
    if http_mode:
        return
    meta_path = Path(palace_path) / _PALACE_META_FILE
    try:
        existing: dict = {}
        if meta_path.is_file():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing["embedding_model"] = embedding_model
        existing["last_mined"] = now
        meta_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            meta_path.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
    except OSError:
        pass


def read_palace_meta(palace_path: str, col=None) -> dict:
    """Read palace metadata.

    When ``col`` is provided (an already-opened ChromaCollection), its metadata
    is the source of truth — works in both HTTP and embedded modes, survives
    palace relocation. Callers that have already loaded the collection should
    always pass it.

    When ``col`` is None, falls back to reading palace_meta.json (embedded mode
    only — HTTP deployments don't have a local file).

    Returns {} if no metadata is found.
    """
    # 1. Collection metadata (authoritative when available)
    if col is not None:
        try:
            meta = col.metadata
            relevant = {k: v for k, v in meta.items() if k in ("embedding_model", "last_mined")}
            if relevant:
                return relevant
        except Exception:
            pass

    # 2. Fall back to local file (embedded mode, legacy palaces, callers
    #    that don't have a collection handy like status-check CLI paths)
    meta_path = Path(palace_path) / _PALACE_META_FILE
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}

_DEFAULT_BACKEND = ChromaBackend()

# Schema version for drawer normalization. Bump when the normalization
# pipeline changes in a way that existing drawers should be rebuilt to pick up
# (e.g., new noise-stripping rules). `file_already_mined` treats drawers with
# a missing or stale `normalize_version` as "not mined", so the next mine pass
# silently rebuilds them — users don't need to manually erase + re-mine.
#
# v2 (2026-04): introduced strip_noise() for Claude Code JSONL; previous
#               drawers stored system tags / hook chrome verbatim.
NORMALIZE_VERSION = 2


def get_collection(
    palace_path: str,
    collection_name: str = "mempalace_drawers",
    create: bool = True,
):
    """Get the palace collection through the backend layer."""
    return _DEFAULT_BACKEND.get_collection(
        palace_path,
        collection_name=collection_name,
        create=create,
    )


def get_closets_collection(palace_path: str, create: bool = True):
    """Get the closets collection — the searchable index layer."""
    return get_collection(palace_path, collection_name="mempalace_closets", create=create)


CLOSET_CHAR_LIMIT = 1500  # fill closet until ~1500 chars, then start a new one
CLOSET_EXTRACT_WINDOW = 5000  # how many chars of source content to scan for entities/topics

# Common capitalized words that look like proper nouns but are usually
# sentence-starters or filler. Filtered out of entity extraction.
_ENTITY_STOPLIST = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "When",
        "Where",
        "What",
        "Why",
        "Who",
        "Which",
        "How",
        "After",
        "Before",
        "Then",
        "Now",
        "Here",
        "There",
        "And",
        "But",
        "Or",
        "Yet",
        "So",
        "If",
        "Else",
        "Yes",
        "No",
        "Maybe",
        "Okay",
        "User",
        "Assistant",
        "System",
        "Tool",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)


def build_closet_lines(source_file, drawer_ids, content, wing, room):
    """Build compact closet pointer lines from drawer content.

    Returns a LIST of lines (not joined). Each line is one complete topic
    pointer — never split across closets.

    Format: topic|entities|→drawer_ids
    """
    import re
    from pathlib import Path

    drawer_ref = ",".join(drawer_ids[:3])
    window = content[:CLOSET_EXTRACT_WINDOW]

    # Extract proper nouns (capitalized words, 2+ occurrences). Filter out
    # common sentence-starters that aren't real entities.
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", window)
    word_freq = {}
    for w in words:
        if w in _ENTITY_STOPLIST:
            continue
        word_freq[w] = word_freq.get(w, 0) + 1
    entities = sorted(
        [w for w, c in word_freq.items() if c >= 2],
        key=lambda w: -word_freq[w],
    )[:5]
    entity_str = ";".join(entities) if entities else ""

    # Extract key phrases — action verbs + context
    topics = []
    for pattern in [
        r"(?:built|fixed|wrote|added|pushed|tested|created|decided|migrated|reviewed|deployed|configured|removed|updated)\s+[\w\s]{3,40}",
    ]:
        topics.extend(re.findall(pattern, window, re.IGNORECASE))
    # Also grab section headers if present
    for header in re.findall(r"^#{1,3}\s+(.{5,60})$", window, re.MULTILINE):
        topics.append(header.strip())
    # Dedupe preserving order
    topics = list(dict.fromkeys(t.strip().lower() for t in topics))[:12]

    # Extract quotes
    quotes = re.findall(r'"([^"]{15,150})"', window)

    # Build pointer lines — each one is atomic, never split
    lines = []
    for topic in topics:
        lines.append(f"{topic}|{entity_str}|→{drawer_ref}")
    for quote in quotes[:3]:
        lines.append(f'"{quote}"|{entity_str}|→{drawer_ref}')

    # Always have at least one line
    if not lines:
        name = Path(source_file).stem[:40]
        lines.append(f"{wing}/{room}/{name}|{entity_str}|→{drawer_ref}")

    return lines


def purge_file_closets(closets_col, source_file: str) -> None:
    """Delete every closet associated with ``source_file``.

    Call this before ``upsert_closet_lines`` on a re-mine so stale topics
    from a prior schema/version don't survive in the closet collection.
    Mirrors the drawer-purge step in process_file().
    """
    try:
        closets_col.delete(where={"source_file": source_file})
    except Exception:
        pass


# Cap per-call payload so a single upsert never balloons in HTTP-mode (one
# round-trip per chunk was the prior pathology) and to keep the embedder's
# working set bounded on local mode. Tuned conservatively — ChromaDB accepts
# larger batches but we hit diminishing returns past ~100.
DRAWER_UPSERT_BATCH_SIZE = 100


def upsert_in_batches(collection, ids, documents, metadatas, batch_size=DRAWER_UPSERT_BATCH_SIZE):
    """Upsert in batches of ≤batch_size. Returns total count written.

    Collapses N per-chunk round-trips into ⌈N/batch_size⌉ — material on the
    HTTP backend (Phase 5) where each call is a network call.
    """
    n = len(ids)
    if not (len(documents) == n and len(metadatas) == n):
        raise ValueError("ids, documents, metadatas must be the same length")
    for i in range(0, n, batch_size):
        collection.upsert(
            documents=documents[i : i + batch_size],
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
    return n


def upsert_closet_lines(closets_col, closet_id_base, lines, metadata):
    """Write topic lines to closets, packed greedily without splitting a line.

    Closets are deterministically numbered (``..._01``, ``..._02``, …) and
    each ``upsert`` fully overwrites the prior content at that ID. Callers
    are expected to ``purge_file_closets`` first when re-mining a source
    file so stale-numbered closets from larger prior runs don't leak.

    Returns the number of closets written.
    """
    closet_num = 1
    current_lines: list = []
    current_chars = 0
    closets_written = 0

    def _flush():
        nonlocal closets_written
        if not current_lines:
            return
        closet_id = f"{closet_id_base}_{closet_num:02d}"
        text = "\n".join(current_lines)
        closets_col.upsert(documents=[text], ids=[closet_id], metadatas=[metadata])
        closets_written += 1

    for line in lines:
        line_len = len(line)
        # Would this line fit whole in the current closet?
        if current_chars > 0 and current_chars + line_len + 1 > CLOSET_CHAR_LIMIT:
            _flush()
            closet_num += 1
            current_lines = []
            current_chars = 0

        current_lines.append(line)
        current_chars += line_len + 1  # +1 for newline

    _flush()
    return closets_written


@contextlib.contextmanager
def mine_lock(source_file: str):
    """Cross-platform file lock for mine operations.

    Prevents multiple agents from mining the same file simultaneously,
    which causes duplicate drawers when the delete+insert cycle interleaves.
    """
    lock_dir = os.path.join(os.path.expanduser("~"), ".mempalace", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(
        lock_dir, hashlib.sha256(source_file.encode()).hexdigest()[:16] + ".lock"
    )

    lf = open(lock_path, "w")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception:
            pass
        lf.close()


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    Returns False (so the file gets re-mined) when:
      - no drawers exist for this source_file
      - the stored `normalize_version` is missing or older than the current
        schema (triggers silent rebuild after a normalization upgrade)
      - `check_mtime=True` and the file's mtime differs from the stored one

    When check_mtime=True (used by project miner), also re-mines on content
    change. When check_mtime=False (used by convo miner), transcripts are
    assumed immutable, so only the version gate triggers a rebuild.
    """
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        if not results.get("ids"):
            return False
        stored_meta = results.get("metadatas", [{}])[0] or {}
        # Pre-v2 drawers have no version field — treat them as stale.
        stored_version = stored_meta.get("normalize_version", 1)
        if stored_version < NORMALIZE_VERSION:
            return False
        if check_mtime:
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return abs(float(stored_mtime) - current_mtime) < 0.001
        return True
    except Exception:
        return False
