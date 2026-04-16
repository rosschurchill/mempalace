"""
rem_cycle.py — REM (Rapid Entity Mapping) Cycle for MemPalace.

Scans the N most-recently-filed drawers, searches the palace for each one's
semantic neighbors in a *different* wing, and creates "semantically_bridges"
triples in the KG when cross-wing similarity exceeds the threshold.

This is the automated discovery engine: it finds connections between projects
and topics that you never explicitly linked. A conversation about Python
performance in wing_myproject that semantically matches a discussion about
profiling in wing_research will get a bridge — so future explain() and
whisper() calls can traverse it.

Design constraints:
- Synchronous and bounded (n_anchors cap) — no daemon, no background threads
- Uses existing ChromaDB HNSW index — no new embedding infrastructure
- Uses existing kg.add_triple() for bridges — no schema changes
- Idempotent: skips pairs that already have a bridge
- Non-destructive: only adds triples, never modifies existing ones
"""

import logging
from datetime import datetime

logger = logging.getLogger("mempalace_mcp")


def run_rem_cycle(
    palace_path: str,
    kg,
    n_anchors: int = 50,
    threshold: float = 0.15,
) -> dict:
    """Discover and record semantic bridges between wings.

    Args:
        palace_path: Path to the ChromaDB palace directory.
        kg: KnowledgeGraph instance (bridges stored as triples here).
        n_anchors: Max number of recent drawers to use as starting points.
                   Bounded to prevent O(n²) behaviour on large palaces.
        threshold: Cosine *distance* threshold (not similarity). 0.15 ≈ 0.85
                   cosine similarity — only very strong semantic matches bridge.
                   Lower = stricter. 0.0 = identical vectors only.

    Returns:
        {
            "bridges_created": int,
            "bridges_skipped_existing": int,
            "anchors_scanned": int,
            "wings_involved": list[str],
            "runtime_ms": int,
        }
    """
    from .palace import get_collection

    start = datetime.now()

    # ── Load collection ───────────────────────────────────────────────────────
    try:
        col = get_collection(palace_path, create=False)
    except Exception as e:
        logger.warning("REM cycle: no palace at %s — %s", palace_path, e)
        return {
            "bridges_created": 0,
            "bridges_skipped_existing": 0,
            "anchors_scanned": 0,
            "wings_involved": [],
            "runtime_ms": 0,
            "error": f"No palace found: {e}",
        }

    # ── Fetch most-recent anchors ─────────────────────────────────────────────
    try:
        total = col.count()
        if total == 0:
            return {
                "bridges_created": 0,
                "bridges_skipped_existing": 0,
                "anchors_scanned": 0,
                "wings_involved": [],
                "runtime_ms": 0,
                "message": "Palace is empty.",
            }

        # Get a larger sample and sort by filed_at to find the most recent ones.
        # We fetch up to n_anchors * 3 and sort client-side — cheaper than
        # paginating the entire palace.
        fetch_limit = min(n_anchors * 3, total, 500)
        raw = col.get(
            include=["documents", "metadatas"],
            limit=fetch_limit,
        )
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        ids = raw.get("ids") or []

    except Exception as e:
        logger.exception("REM cycle: failed to fetch anchors")
        return {
            "bridges_created": 0,
            "bridges_skipped_existing": 0,
            "anchors_scanned": 0,
            "wings_involved": [],
            "runtime_ms": 0,
            "error": str(e),
        }

    # Sort by filed_at descending (most recent first)
    anchors = sorted(
        [
            {"doc": d, "meta": m, "id": i}
            for d, m, i in zip(docs, metas, ids)
            if d and m
        ],
        key=lambda x: x["meta"].get("filed_at", ""),
        reverse=True,
    )[:n_anchors]

    # ── Pre-load existing bridges to avoid duplicate triples ──────────────────
    existing_bridges: set = set()
    try:
        with kg._lock:
            conn = kg._conn()
            rows = conn.execute(
                "SELECT subject, object FROM triples WHERE predicate='semantically_bridges'"
            ).fetchall()
            for row in rows:
                existing_bridges.add((row["subject"], row["object"]))
                existing_bridges.add((row["object"], row["subject"]))  # undirected
    except Exception:
        pass  # If we can't read existing bridges, we'll just create duplicates
        # which KG dedup logic in add_triple() will prevent anyway

    # ── Main scan ─────────────────────────────────────────────────────────────
    bridges_created = 0
    bridges_skipped = 0
    wings_involved: set = set()

    for anchor in anchors:
        anchor_doc = anchor["doc"]
        anchor_meta = anchor["meta"]
        anchor_id = anchor["id"]
        anchor_wing = anchor_meta.get("wing", "")

        if not anchor_wing or not anchor_doc:
            continue

        try:
            # Query for semantic neighbors in a DIFFERENT wing
            # ChromaDB's $ne operator filters out the anchor's own wing
            results = col.query(
                query_texts=[anchor_doc],
                n_results=3,
                where={"wing": {"$ne": anchor_wing}},
                include=["metadatas", "distances"],
            )
        except Exception:
            continue

        distances = results.get("distances") or [[]]
        metadatas = results.get("metadatas") or [[]]
        if not distances or not distances[0]:
            continue

        for dist, meta in zip(distances[0], metadatas[0]):
            if dist is None or dist > threshold:
                continue
            neighbor_wing = (meta or {}).get("wing", "")
            if not neighbor_wing or neighbor_wing == anchor_wing:
                continue

            # Derive canonical entity IDs (same as KG._entity_id)
            src_id = anchor_wing.lower().replace(" ", "_").replace("'", "")
            dst_id = neighbor_wing.lower().replace(" ", "_").replace("'", "")

            if (src_id, dst_id) in existing_bridges:
                bridges_skipped += 1
                continue

            # Create the bridge triple
            try:
                kg.add_triple(
                    anchor_wing,
                    "semantically_bridges",
                    neighbor_wing,
                    source_closet=anchor_id,
                    confidence=round(1.0 - dist, 3),
                )
                existing_bridges.add((src_id, dst_id))
                existing_bridges.add((dst_id, src_id))
                bridges_created += 1
                wings_involved.add(anchor_wing)
                wings_involved.add(neighbor_wing)
                logger.info(
                    "REM bridge: %s ↔ %s (dist=%.3f)", anchor_wing, neighbor_wing, dist
                )
            except Exception:
                pass  # Never let a failed bridge abort the cycle

    runtime_ms = int((datetime.now() - start).total_seconds() * 1000)

    return {
        "bridges_created": bridges_created,
        "bridges_skipped_existing": bridges_skipped,
        "anchors_scanned": len(anchors),
        "wings_involved": sorted(wings_involved),
        "runtime_ms": runtime_ms,
    }
