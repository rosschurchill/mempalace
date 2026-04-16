"""Tests for ChromaBackend HTTP client mode (#832).

All tests use mocking — no live ChromaDB server needed.
The env var MEMPALACE_CHROMA_URL triggers HTTP mode; its absence uses embedded.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from mempalace.backends.chroma import ChromaBackend, _get_http_client


# ── _get_http_client ──────────────────────────────────────────────────────────

def test_get_http_client_basic_url():
    """HttpClient receives correct host and port from MEMPALACE_CHROMA_URL."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://chroma.theshellnet.com:8001"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            _get_http_client()
            mock_http.assert_called_once()
            kwargs = mock_http.call_args[1]
            assert kwargs["host"] == "chroma.theshellnet.com"
            assert kwargs["port"] == 8001
            assert kwargs["ssl"] is False


def test_get_http_client_https():
    """SSL flag set when scheme is https."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "https://secure-chroma:443"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            _get_http_client()
            kwargs = mock_http.call_args[1]
            assert kwargs["ssl"] is True


def test_get_http_client_default_port():
    """Port defaults to 8000 for http:// URLs without explicit port."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://mempalace-chroma"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            _get_http_client()
            kwargs = mock_http.call_args[1]
            assert kwargs["port"] == 8000


def test_get_http_client_with_token():
    """Authorization header included when MEMPALACE_CHROMA_TOKEN is set."""
    env = {
        "MEMPALACE_CHROMA_URL": "http://chroma.theshellnet.com:8001",
        "MEMPALACE_CHROMA_TOKEN": "supersecret",
    }
    with patch.dict(os.environ, env):
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            _get_http_client()
            kwargs = mock_http.call_args[1]
            assert kwargs["headers"] == {"Authorization": "Bearer supersecret"}


def test_get_http_client_no_token():
    """No Authorization header when MEMPALACE_CHROMA_TOKEN is absent."""
    env = {"MEMPALACE_CHROMA_URL": "http://chroma.theshellnet.com:8001"}
    clean_env = {k: v for k, v in os.environ.items() if k != "MEMPALACE_CHROMA_TOKEN"}
    clean_env.update(env)
    with patch.dict(os.environ, clean_env, clear=True):
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            _get_http_client()
            kwargs = mock_http.call_args[1]
            assert kwargs.get("headers") == {}


# ── ChromaBackend._client() ───────────────────────────────────────────────────

def test_backend_uses_http_when_env_set():
    """_client() returns HttpClient when MEMPALACE_CHROMA_URL is set."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://localhost:8001"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_client = MagicMock()
            mock_http.return_value = mock_client
            backend = ChromaBackend()
            result = backend._client("/any/path")
            mock_http.assert_called_once()
            assert result is mock_client


def test_backend_uses_persistent_when_no_env(tmp_path):
    """_client() returns PersistentClient when MEMPALACE_CHROMA_URL is absent."""
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("MEMPALACE_CHROMA_URL", "MEMPALACE_CHROMA_TOKEN")}
    palace = str(tmp_path / "palace")
    os.makedirs(palace)
    with patch.dict(os.environ, clean_env, clear=True):
        with patch("chromadb.PersistentClient") as mock_pc:
            mock_pc.return_value = MagicMock()
            backend = ChromaBackend()
            backend._client(palace)
            mock_pc.assert_called_once()


def test_backend_http_client_cached():
    """_client() returns the same HttpClient instance on repeated calls."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://localhost:8001"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_client = MagicMock()
            mock_http.return_value = mock_client
            backend = ChromaBackend()
            r1 = backend._client("/p1")
            r2 = backend._client("/p2")
            assert r1 is r2
            assert mock_http.call_count == 1  # only one HttpClient created


# ── ChromaBackend.make_client() ───────────────────────────────────────────────

def test_make_client_uses_http_when_env_set():
    """make_client() returns HttpClient in HTTP mode."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://localhost:8001"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            ChromaBackend.make_client("/any/path")
            mock_http.assert_called_once()


def test_make_client_uses_persistent_when_no_env(tmp_path):
    """make_client() returns PersistentClient in embedded mode."""
    palace = str(tmp_path / "palace")
    os.makedirs(palace)
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("MEMPALACE_CHROMA_URL", "MEMPALACE_CHROMA_TOKEN")}
    with patch.dict(os.environ, clean_env, clear=True):
        with patch("chromadb.PersistentClient") as mock_pc:
            mock_pc.return_value = MagicMock()
            ChromaBackend.make_client(palace)
            mock_pc.assert_called_once()


# ── ChromaBackend.get_collection() ───────────────────────────────────────────

def test_get_collection_skips_mkdir_in_http_mode():
    """In HTTP mode, get_collection() should not call os.makedirs."""
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://localhost:8001"}):
        with patch("chromadb.HttpClient") as mock_http:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client.get_collection.return_value = mock_collection
            mock_http.return_value = mock_client

            backend = ChromaBackend()
            with patch("os.makedirs") as mock_mkdir:
                backend.get_collection("/nonexistent/path", "test_col", create=True)
                mock_mkdir.assert_not_called()


def test_get_collection_raises_for_missing_path_in_embedded_mode():
    """In embedded mode, get_collection(create=False) raises if path missing."""
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("MEMPALACE_CHROMA_URL", "MEMPALACE_CHROMA_TOKEN")}
    with patch.dict(os.environ, clean_env, clear=True):
        backend = ChromaBackend()
        with pytest.raises(FileNotFoundError):
            backend.get_collection("/definitely/does/not/exist", "col", create=False)


# ── palace.py write_palace_meta ───────────────────────────────────────────────

def test_write_palace_meta_noop_in_http_mode(tmp_path):
    """write_palace_meta() should not write any file in HTTP mode."""
    from mempalace.palace import write_palace_meta
    palace = str(tmp_path / "palace")
    with patch.dict(os.environ, {"MEMPALACE_CHROMA_URL": "http://localhost:8001"}):
        write_palace_meta(palace, "all-MiniLM-L6-v2")
    # No palace directory should be created
    assert not os.path.isdir(palace)


def test_write_palace_meta_writes_file_in_embedded_mode(tmp_path):
    """write_palace_meta() writes palace_meta.json in embedded mode."""
    import json
    from mempalace.palace import write_palace_meta
    palace = str(tmp_path / "palace")
    os.makedirs(palace)
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("MEMPALACE_CHROMA_URL", "MEMPALACE_CHROMA_TOKEN")}
    with patch.dict(os.environ, clean_env, clear=True):
        write_palace_meta(palace, "all-MiniLM-L6-v2")
    meta_file = os.path.join(palace, "palace_meta.json")
    assert os.path.isfile(meta_file)
    data = json.loads(open(meta_file).read())
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
