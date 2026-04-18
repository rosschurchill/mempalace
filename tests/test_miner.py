import os
import shutil
import tempfile
from pathlib import Path

import chromadb
import yaml

from mempalace.miner import load_config, mine, scan_project, status
from mempalace.palace import NORMALIZE_VERSION, file_already_mined


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scanned_files(project_root: Path, **kwargs):
    files = scan_project(str(project_root), **kwargs)
    return sorted(path.relative_to(project_root).as_posix() for path in files)


def test_project_mining():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        os.makedirs(project_root / "backend")

        write_file(
            project_root / "backend" / "app.py",
            "def main():\n    print('hello world')\n" * 20,
        )
        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_project",
                    "rooms": [
                        {"name": "backend", "description": "Backend code"},
                        {"name": "general", "description": "General"},
                    ],
                },
                f,
            )

        palace_path = project_root / "palace"
        mine(str(project_root), str(palace_path))

        client = chromadb.PersistentClient(path=str(palace_path))
        col = client.get_collection("mempalace_drawers")
        assert col.count() > 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_config_uses_defaults_when_yaml_missing():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        config = load_config(str(project_root))

        assert isinstance(config, dict)
        assert "wing" in config
        assert "rooms" in config
        assert config["wing"] == project_root.name
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_respects_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "ignored.py\ngenerated/\n")
        write_file(project_root / "src" / "app.py", "print('hello')\n" * 20)
        write_file(project_root / "ignored.py", "print('ignore me')\n" * 20)
        write_file(project_root / "generated" / "artifact.py", "print('artifact')\n" * 20)

        assert scanned_files(project_root) == ["src/app.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_respects_nested_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "*.log\n")
        write_file(project_root / "subrepo" / ".gitignore", "tasks/\n")
        write_file(project_root / "subrepo" / "src" / "main.py", "print('main')\n" * 20)
        write_file(project_root / "subrepo" / "tasks" / "task.py", "print('task')\n" * 20)
        write_file(project_root / "subrepo" / "debug.log", "debug\n" * 20)

        assert scanned_files(project_root) == ["subrepo/src/main.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_allows_nested_gitignore_override():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "*.csv\n")
        write_file(project_root / "subrepo" / ".gitignore", "!keep.csv\n")
        write_file(project_root / "drop.csv", "a,b,c\n" * 20)
        write_file(project_root / "subrepo" / "keep.csv", "a,b,c\n" * 20)

        assert scanned_files(project_root) == ["subrepo/keep.csv"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_allows_gitignore_negation_when_parent_dir_is_visible():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/*\n!generated/keep.py\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root) == ["generated/keep.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_does_not_reinclude_file_from_ignored_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/\n!generated/keep.py\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root) == []
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_disable_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "data/\n")
        write_file(project_root / "data" / "stuff.csv", "a,b,c\n" * 20)

        assert scanned_files(project_root, respect_gitignore=False) == ["data/stuff.csv"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_ignored_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "docs/\n")
        write_file(project_root / "docs" / "guide.md", "# Guide\n" * 20)

        assert scanned_files(project_root, include_ignored=["docs"]) == ["docs/guide.md"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_specific_ignored_file():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root, include_ignored=["generated/keep.py"]) == [
            "generated/keep.py"
        ]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_exact_file_without_known_extension():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "README\n")
        write_file(project_root / "README", "hello\n" * 20)

        assert scanned_files(project_root, include_ignored=["README"]) == ["README"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_include_override_beats_skip_dirs():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)

        assert scanned_files(
            project_root,
            respect_gitignore=False,
            include_ignored=[".pytest_cache"],
        ) == [".pytest_cache/cache.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_skip_dirs_still_apply_without_override():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)
        write_file(project_root / "main.py", "print('main')\n" * 20)

        assert scanned_files(project_root, respect_gitignore=False) == ["main.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_file_already_mined_check_mtime():
    tmpdir = tempfile.mkdtemp()
    try:
        palace_path = os.path.join(tmpdir, "palace")
        os.makedirs(palace_path)
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )

        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello world")

        mtime = os.path.getmtime(test_file)

        # Not mined yet
        assert file_already_mined(col, test_file) is False
        assert file_already_mined(col, test_file, check_mtime=True) is False

        # Add it with mtime + current normalize_version
        col.add(
            ids=["d1"],
            documents=["hello world"],
            metadatas=[
                {
                    "source_file": test_file,
                    "source_mtime": str(mtime),
                    "normalize_version": NORMALIZE_VERSION,
                }
            ],
        )

        # Already mined (no mtime check)
        assert file_already_mined(col, test_file) is True
        # Already mined (mtime matches)
        assert file_already_mined(col, test_file, check_mtime=True) is True

        # Modify file and force a different mtime (Windows has low mtime resolution)
        with open(test_file, "w") as f:
            f.write("modified content")
        os.utime(test_file, (mtime + 10, mtime + 10))

        # Still mined without mtime check
        assert file_already_mined(col, test_file) is True
        # Needs re-mining with mtime check
        assert file_already_mined(col, test_file, check_mtime=True) is False

        # Record with no mtime stored should return False for check_mtime
        col.add(
            ids=["d2"],
            documents=["other"],
            metadatas=[
                {
                    "source_file": "/fake/no_mtime.txt",
                    "normalize_version": NORMALIZE_VERSION,
                }
            ],
        )
        assert file_already_mined(col, "/fake/no_mtime.txt", check_mtime=True) is False
    finally:
        # Release ChromaDB file handles before cleanup (required on Windows)
        del col, client
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_dry_run_with_tiny_file_no_crash():
    """Dry-run must not crash when process_file returns 0 drawers (room was None)."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        # One normal file and one that falls below MIN_CHUNK_SIZE
        write_file(project_root / "good.py", "def main():\n    print('hello world')\n" * 20)
        write_file(project_root / "tiny.txt", "x")

        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_project",
                    "rooms": [{"name": "general", "description": "General"}],
                },
                f,
            )

        palace_path = project_root / "palace"
        # Should not raise TypeError on the summary print
        mine(str(project_root), str(palace_path), dry_run=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_status_missing_palace_does_not_create_empty_collection(tmp_path, capsys):
    palace_path = tmp_path / "missing-palace"

    status(str(palace_path))

    out = capsys.readouterr().out
    assert "No palace found" in out
    assert not palace_path.exists()


# ── normalize_version schema gate ───────────────────────────────────────
#
# When the normalization pipeline changes shape (e.g., strip_noise lands),
# `NORMALIZE_VERSION` is bumped so pre-existing drawers can be silently
# rebuilt on the next mine. These tests pin that contract.


def test_file_already_mined_returns_false_for_stale_normalize_version():
    """Pre-v2 drawers (no field, or older integer) must not short-circuit."""
    tmpdir = tempfile.mkdtemp()
    try:
        palace_path = os.path.join(tmpdir, "palace")
        os.makedirs(palace_path)
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection("mempalace_drawers")

        # Pre-v2 drawer: no normalize_version field at all
        col.add(
            ids=["d_old"],
            documents=["old"],
            metadatas=[{"source_file": "/fake/old.jsonl"}],
        )
        assert file_already_mined(col, "/fake/old.jsonl") is False

        # Explicitly older version
        col.add(
            ids=["d_v1"],
            documents=["v1"],
            metadatas=[{"source_file": "/fake/v1.jsonl", "normalize_version": 1}],
        )
        assert file_already_mined(col, "/fake/v1.jsonl") is False

        # Current version — short-circuits
        col.add(
            ids=["d_current"],
            documents=["cur"],
            metadatas=[
                {
                    "source_file": "/fake/current.jsonl",
                    "normalize_version": NORMALIZE_VERSION,
                }
            ],
        )
        assert file_already_mined(col, "/fake/current.jsonl") is True
    finally:
        del col, client
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_drawer_stamps_normalize_version(tmp_path):
    """Fresh drawers carry the current schema version so future upgrades work."""
    from mempalace.miner import add_drawer

    palace_path = tmp_path / "palace"
    palace_path.mkdir()
    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_or_create_collection("mempalace_drawers")
    try:
        added = add_drawer(
            collection=col,
            wing="test",
            room="notes",
            content="hello",
            source_file=str(tmp_path / "src.md"),
            chunk_index=0,
            agent="unit",
        )
        assert added is True
        stored = col.get(limit=1)
        meta = stored["metadatas"][0]
        assert meta["normalize_version"] == NORMALIZE_VERSION
    finally:
        del col, client


class _UpsertRecorder:
    """Captures upsert/delete/get calls. Enough stub for process_file."""

    def __init__(self):
        self.upserts: list = []
        self.deletes: list = []

    def get(self, **kwargs):
        return {"ids": [], "metadatas": []}

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def upsert(self, *, documents, ids, metadatas):
        self.upserts.append(
            {"documents": list(documents), "ids": list(ids), "metadatas": list(metadatas)}
        )


def test_upsert_in_batches_splits_into_fixed_size_calls():
    from mempalace.palace import DRAWER_UPSERT_BATCH_SIZE, upsert_in_batches

    fake = _UpsertRecorder()
    n = DRAWER_UPSERT_BATCH_SIZE * 2 + 50
    ids = [f"id_{i}" for i in range(n)]
    docs = [f"doc_{i}" for i in range(n)]
    metas = [{"i": i} for i in range(n)]

    written = upsert_in_batches(fake, ids, docs, metas)

    assert written == n
    assert len(fake.upserts) == 3
    assert [len(c["ids"]) for c in fake.upserts] == [
        DRAWER_UPSERT_BATCH_SIZE,
        DRAWER_UPSERT_BATCH_SIZE,
        50,
    ]
    # All ids preserved across batches, in order
    merged = [i for c in fake.upserts for i in c["ids"]]
    assert merged == ids


def test_process_file_batches_upserts_for_large_file(tmp_path):
    """Many chunks → upsert called in batches, not per-chunk.

    Regression guard: Phase 5 remote ChromaDB otherwise paid one HTTP RTT
    per chunk, which made mining large files visibly slow.
    """
    from mempalace.miner import process_file
    from mempalace.palace import DRAWER_UPSERT_BATCH_SIZE

    # 250 paragraphs each big enough to survive MIN_CHUNK_SIZE after strip.
    paragraphs = ["word " * 200 for _ in range(250)]
    src = tmp_path / "big.md"
    src.write_text("\n\n".join(paragraphs), encoding="utf-8")

    fake = _UpsertRecorder()
    drawers, _room = process_file(
        filepath=src,
        project_path=tmp_path,
        collection=fake,
        wing="w",
        rooms=[],
        agent="t",
        dry_run=False,
    )

    assert drawers > DRAWER_UPSERT_BATCH_SIZE, (
        "test expects >batch_size chunks to exercise batching"
    )
    expected_calls = (drawers + DRAWER_UPSERT_BATCH_SIZE - 1) // DRAWER_UPSERT_BATCH_SIZE
    assert len(fake.upserts) == expected_calls
    for call in fake.upserts:
        assert len(call["ids"]) <= DRAWER_UPSERT_BATCH_SIZE
    # Every drawer id is written exactly once
    merged = [i for c in fake.upserts for i in c["ids"]]
    assert len(merged) == drawers
    assert len(set(merged)) == drawers


# ── Content-hash dedup (#7.3) ─────────────────────────────────────────────────


def test_file_already_mined_detects_same_mtime_content_change(tmp_path):
    """When content changes but mtime stays the same, file_already_mined()
    must return False so the miner picks up the change.

    This guards the case where a tool or script writes content without
    bumping the mtime (coarse-grained file system, mtime-preserving copy,
    or clock skew). A pure mtime check would wrongly skip re-mining.
    """
    from mempalace.palace import _file_content_hash

    src = tmp_path / "notes.txt"
    src.write_text("original content here", encoding="utf-8")
    mtime = os.path.getmtime(str(src))
    original_hash = _file_content_hash(str(src))

    stored_meta = {
        "source_file": str(src),
        "source_mtime": mtime,
        "content_sha256": original_hash,
        "normalize_version": NORMALIZE_VERSION,
    }

    class _FakeCol:
        def get(self, where=None, limit=1):
            return {"ids": ["d1"], "metadatas": [stored_meta]}

    col = _FakeCol()

    # No changes — should appear mined
    assert file_already_mined(col, str(src), check_mtime=True) is True

    # Overwrite with different content but restore the exact same mtime
    src.write_text("completely different content", encoding="utf-8")
    os.utime(str(src), (mtime, mtime))

    # mtime matches but content changed — must return False
    assert file_already_mined(col, str(src), check_mtime=True) is False


def test_process_file_stores_content_sha256_in_metadata(tmp_path):
    """Drawers produced by process_file must carry content_sha256 in metadata."""
    from mempalace.miner import process_file

    src = tmp_path / "doc.txt"
    src.write_text("word " * 100, encoding="utf-8")  # above MIN_CHUNK_SIZE

    class _CaptureMeta:
        def __init__(self):
            self.upserts = []
            self.deletes = []

        def get(self, **kwargs):
            return {"ids": [], "metadatas": []}

        def delete(self, **kwargs):
            self.deletes.append(kwargs)

        def upsert(self, *, documents, ids, metadatas):
            self.upserts.append(metadatas)

    fake = _CaptureMeta()
    process_file(
        filepath=src,
        project_path=tmp_path,
        collection=fake,
        wing="w",
        rooms=[],
        agent="t",
        dry_run=False,
    )

    assert fake.upserts, "expected at least one upsert"
    first_batch_metas = fake.upserts[0]
    assert first_batch_metas, "metadata list must not be empty"
    first_meta = first_batch_metas[0]
    assert "content_sha256" in first_meta, "content_sha256 must be stored in drawer metadata"
    assert len(first_meta["content_sha256"]) == 64, "SHA256 hex digest is 64 chars"


def test_mine_refresh_forces_remining(tmp_path):
    """--refresh causes already-mined files to be re-mined."""
    import yaml as _yaml
    from mempalace.miner import mine

    src_dir = tmp_path / "project"
    src_dir.mkdir()
    (src_dir / "notes.md").write_text("word " * 100, encoding="utf-8")
    (src_dir / "mempalace.yaml").write_text(
        _yaml.dump({"wing": "test", "rooms": [{"name": "general", "description": "G"}]}),
        encoding="utf-8",
    )

    palace = str(tmp_path / "palace")

    # First mine — files not yet indexed
    mine(str(src_dir), palace, dry_run=True)  # dry-run to avoid embedder

    # Verify --refresh doesn't crash when passed; dry_run+refresh together is safe
    mine(str(src_dir), palace, dry_run=True, refresh=True)
