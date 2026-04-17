"""ChromaDB-backed MemPalace collection adapter.

Two modes — selected by environment variable:

Embedded (default):
    PersistentClient writing to a local directory.
    One process at a time; SQLite-backed HNSW index.

HTTP (opt-in):
    HttpClient connecting to a remote ChromaDB HTTP server.
    Set MEMPALACE_CHROMA_URL=http://host:port to activate.
    Optionally set MEMPALACE_CHROMA_TOKEN for Bearer token auth.
    Enables multi-process and multi-machine access to a shared palace.
"""

import logging
import os
import sqlite3
import urllib.parse

import chromadb
from chromadb.config import Settings

from .base import BaseCollection

# Disable ChromaDB's PostHog telemetry. Without this, every query sends data
# to posthog.com, violating MemPalace's local-first guarantee. Set at import
# time (env var) and again at client-creation time (Settings object) as belt-
# and-suspenders — the env var covers C-level telemetry before Python sees it.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _chroma_settings() -> Settings:
    """Return a fresh Settings instance.

    IMPORTANT: Do not share Settings across HttpClient / PersistentClient
    calls. ChromaDB mutates the Settings object during client init — HttpClient
    writes chroma_server_host/port into it, and a subsequent PersistentClient
    using the same object will see those host fields and silently switch to
    HTTP mode (failing with "Could not connect to a Chroma server" even when
    MEMPALACE_CHROMA_URL is unset). Every call creates a fresh object.
    """
    return Settings(anonymized_telemetry=False)


logger = logging.getLogger(__name__)


def _get_http_client():
    """Build a chromadb.HttpClient from MEMPALACE_CHROMA_URL / MEMPALACE_CHROMA_TOKEN.

    MEMPALACE_CHROMA_URL  — required, e.g. http://10.0.10.200:8001 or
                            http://mempalace-chroma:8000
    MEMPALACE_CHROMA_TOKEN — optional Bearer token (set on the ChromaDB server
                             via CHROMA_SERVER_AUTHN_CREDENTIALS)
    """
    raw_url = os.environ["MEMPALACE_CHROMA_URL"].rstrip("/")
    parsed = urllib.parse.urlparse(raw_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    ssl = parsed.scheme == "https"
    token = os.environ.get("MEMPALACE_CHROMA_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return chromadb.HttpClient(
        host=host,
        port=port,
        ssl=ssl,
        headers=headers,
        settings=_chroma_settings(),
    )


def _fix_blob_seq_ids(palace_path: str):
    """Fix ChromaDB 0.6.x -> 1.5.x migration bug: BLOB seq_ids -> INTEGER.

    ChromaDB 0.6.x stored seq_id as big-endian 8-byte BLOBs. ChromaDB 1.5.x
    expects INTEGER. The auto-migration doesn't convert existing rows, causing
    the Rust compactor to crash with "mismatched types; Rust type u64 (as SQL
    type INTEGER) is not compatible with SQL type BLOB".

    Must run BEFORE PersistentClient is created (the compactor fires on init).
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return
    try:
        with sqlite3.connect(db_path) as conn:
            for table in ("embeddings", "max_seq_id"):
                try:
                    rows = conn.execute(
                        f"SELECT rowid, seq_id FROM {table} WHERE typeof(seq_id) = 'blob'"
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if not rows:
                    continue
                updates = [(int.from_bytes(blob, byteorder="big"), rowid) for rowid, blob in rows]
                conn.executemany(f"UPDATE {table} SET seq_id = ? WHERE rowid = ?", updates)
                logger.info("Fixed %d BLOB seq_ids in %s", len(updates), table)
            conn.commit()
    except Exception:
        logger.exception("Could not fix BLOB seq_ids in %s", db_path)


class ChromaCollection(BaseCollection):
    """Thin adapter over a ChromaDB collection."""

    def __init__(self, collection):
        self._collection = collection

    def add(self, *, documents, ids, metadatas=None):
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def upsert(self, *, documents, ids, metadatas=None):
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def update(self, **kwargs):
        self._collection.update(**kwargs)

    def query(self, **kwargs):
        return self._collection.query(**kwargs)

    def get(self, **kwargs):
        return self._collection.get(**kwargs)

    def delete(self, **kwargs):
        self._collection.delete(**kwargs)

    def count(self):
        return self._collection.count()

    @property
    def metadata(self) -> dict:
        """Collection-level metadata (includes hnsw:space and palace config)."""
        return dict(self._collection.metadata or {})

    def set_metadata(self, **updates) -> None:
        """Merge updates into collection-level metadata (non-hnsw keys only).

        ChromaDB's ``modify(metadata=...)`` rejects any ``hnsw:*`` keys —
        the distance function is immutable after creation. We strip those
        keys before passing the merged dict, so user-level keys (like
        ``embedding_model``) are preserved while core hnsw config is left alone.

        Used to record embedding model info at mine time so searchers can
        detect model drift (issue #903/#912).
        """
        current = self.metadata
        current.update(updates)
        mutable = {k: v for k, v in current.items() if not k.startswith("hnsw:")}
        self._collection.modify(metadata=mutable)


class ChromaBackend:
    """Factory for MemPalace's default ChromaDB backend."""

    def __init__(self):
        # Per-instance client cache: palace_path -> chromadb.PersistentClient
        self._clients: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self, palace_path: str):
        """Return a cached ChromaDB client for *palace_path*.

        HTTP mode (MEMPALACE_CHROMA_URL set): returns a single shared HttpClient.
        The palace_path parameter is ignored — the server holds the data.

        Embedded mode (default): returns a PersistentClient keyed by palace_path.
        """
        if os.environ.get("MEMPALACE_CHROMA_URL"):
            # HTTP mode: one client shared across all callers in this process
            if "_http" not in self._clients:
                self._clients["_http"] = _get_http_client()
            return self._clients["_http"]
        # Embedded mode
        if palace_path not in self._clients:
            _fix_blob_seq_ids(palace_path)
            self._clients[palace_path] = chromadb.PersistentClient(
                path=palace_path, settings=_chroma_settings()
            )
        return self._clients[palace_path]

    # ------------------------------------------------------------------
    # Public static helpers (for callers that manage their own caching)
    # ------------------------------------------------------------------

    @staticmethod
    def make_client(palace_path: str):
        """Create and return a fresh ChromaDB client.

        HTTP mode  (MEMPALACE_CHROMA_URL set): returns a new HttpClient.
        Embedded mode (default): returns a fresh PersistentClient after
        running the BLOB seq_id migration.

        Intended for long-lived callers (e.g. mcp_server) that manage their
        own client cache.
        """
        if os.environ.get("MEMPALACE_CHROMA_URL"):
            return _get_http_client()
        _fix_blob_seq_ids(palace_path)
        return chromadb.PersistentClient(path=palace_path, settings=_chroma_settings())

    @staticmethod
    def backend_version() -> str:
        """Return the installed chromadb package version string."""
        return chromadb.__version__

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def get_collection(self, palace_path: str, collection_name: str, create: bool = False):
        http_mode = bool(os.environ.get("MEMPALACE_CHROMA_URL"))
        if not http_mode:
            # Embedded mode: validate and create local directory
            if not create and not os.path.isdir(palace_path):
                raise FileNotFoundError(palace_path)
            if create:
                os.makedirs(palace_path, exist_ok=True)
                try:
                    os.chmod(palace_path, 0o700)
                except (OSError, NotImplementedError):
                    pass

        client = self._client(palace_path)
        if create:
            collection = client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = client.get_collection(collection_name)
        return ChromaCollection(collection)

    def get_or_create_collection(
        self, palace_path: str, collection_name: str
    ) -> "ChromaCollection":
        """Shorthand for get_collection(..., create=True)."""
        return self.get_collection(palace_path, collection_name, create=True)

    def delete_collection(self, palace_path: str, collection_name: str) -> None:
        """Delete *collection_name* from the palace at *palace_path*."""
        self._client(palace_path).delete_collection(collection_name)

    def create_collection(
        self, palace_path: str, collection_name: str, hnsw_space: str = "cosine"
    ) -> "ChromaCollection":
        """Create (not get-or-create) *collection_name* with cosine HNSW space."""
        collection = self._client(palace_path).create_collection(
            collection_name, metadata={"hnsw:space": hnsw_space}
        )
        return ChromaCollection(collection)
