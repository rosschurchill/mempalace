"""Tests for destructive-operation safety in mempalace.migrate."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mempalace.migrate import migrate


def test_migrate_requires_palace_database(tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()

    result = migrate(str(palace_dir))

    out = capsys.readouterr().out
    assert result is False
    assert "No palace database found" in out


def test_migrate_aborts_without_confirmation(tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    # Presence of chroma.sqlite3 is the safety gate; validity is mocked below.
    (palace_dir / "chroma.sqlite3").write_text("db")

    mock_chromadb = SimpleNamespace(
        __version__="0.6.0",
        PersistentClient=MagicMock(side_effect=Exception("unreadable")),
    )

    with (
        patch.dict("sys.modules", {"chromadb": mock_chromadb}),
        patch("mempalace.migrate.detect_chromadb_version", return_value="0.5.x"),
        patch(
            "mempalace.migrate.extract_drawers_from_sqlite",
            return_value=[{"id": "id1", "document": "doc", "metadata": {"wing": "w", "room": "r"}}],
        ),
        patch("builtins.input", return_value="n"),
        patch("mempalace.migrate.shutil.copytree") as mock_copytree,
        patch("mempalace.migrate.shutil.rmtree") as mock_rmtree,
    ):
        result = migrate(str(palace_dir))

    out = capsys.readouterr().out
    assert result is False
    assert "Aborted." in out
    mock_copytree.assert_not_called()
    mock_rmtree.assert_not_called()


# ── #9.7 / #469: silent count=0 on schema-mismatched palace ──────────────────


def _make_palace_db(palace_dir):
    """Create a valid (but minimal) chroma.sqlite3 for migration tests."""
    import sqlite3
    db_path = palace_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS collections (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()


def test_migrate_detects_silent_count_zero(tmp_path, capsys):
    """When ChromaDB returns count=0 but SQLite has drawers, migration is triggered."""
    from unittest.mock import MagicMock, patch
    from mempalace.migrate import migrate

    palace = tmp_path / "palace"
    palace.mkdir()
    _make_palace_db(palace)

    fake_col = MagicMock()
    fake_col.count.return_value = 0  # ChromaDB 3.1.x silent mismatch

    fake_backend = MagicMock()
    fake_backend.get_collection.return_value = fake_col

    fake_drawers = [
        {"id": "d1", "document": "hello", "metadata": {"wing": "w", "room": "r"}}
    ]

    with (
        patch("mempalace.migrate.ChromaBackend", return_value=fake_backend),
        patch("mempalace.migrate.extract_drawers_from_sqlite", return_value=fake_drawers),
        patch("mempalace.migrate.confirm_destructive_action", return_value=False),
    ):
        migrate(str(palace), dry_run=False, confirm=False)

    out = capsys.readouterr().out
    assert "NOT readable" in out or "schema mismatch" in out or "Aborted" in out


def test_migrate_no_false_positive_on_genuinely_empty_palace(tmp_path, capsys):
    """When ChromaDB returns count=0 AND SQLite has no drawers, no migration triggered."""
    from unittest.mock import MagicMock, patch
    from mempalace.migrate import migrate

    palace = tmp_path / "palace"
    palace.mkdir()
    _make_palace_db(palace)

    fake_col = MagicMock()
    fake_col.count.return_value = 0

    fake_backend = MagicMock()
    fake_backend.get_collection.return_value = fake_col

    with (
        patch("mempalace.migrate.ChromaBackend", return_value=fake_backend),
        patch("mempalace.migrate.extract_drawers_from_sqlite", return_value=[]),
    ):
        result = migrate(str(palace), dry_run=False, confirm=True)

    out = capsys.readouterr().out
    assert result is True
    assert "No migration needed" in out


def test_migrate_nonzero_count_skips_sqlite_check(tmp_path, capsys):
    """When ChromaDB returns count>0, extract_drawers_from_sqlite must not be called."""
    from unittest.mock import MagicMock, patch
    from mempalace.migrate import migrate

    palace = tmp_path / "palace"
    palace.mkdir()
    _make_palace_db(palace)

    fake_col = MagicMock()
    fake_col.count.return_value = 42

    fake_backend = MagicMock()
    fake_backend.get_collection.return_value = fake_col

    with (
        patch("mempalace.migrate.ChromaBackend", return_value=fake_backend),
        patch("mempalace.migrate.extract_drawers_from_sqlite") as mock_extract,
    ):
        result = migrate(str(palace), dry_run=False, confirm=True)

    assert result is True
    mock_extract.assert_not_called()
