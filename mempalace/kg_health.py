"""
kg_health.py — KG staleness detection for MemPalace.

Surfaces knowledge graph facts that may have become stale:
  - Triples whose valid_from is older than a staleness threshold and have
    never been invalidated (valid_to IS NULL) and were not recently confirmed.
  - Entities not mentioned in any drawer filed recently (no recent context).

Used by the mempalace_kg_staleness MCP tool.
"""

from datetime import datetime, timedelta


_DEFAULT_STALE_DAYS = 180  # 6 months — facts older than this are flagged
_DEFAULT_ENTITY_SILENCE_DAYS = 90  # entity not mentioned in drawers for >90 days


def check_stale_triples(
    kg,
    stale_days: int = _DEFAULT_STALE_DAYS,
    limit: int = 50,
) -> list:
    """Return triples that are probably stale.

    A triple is flagged when:
    - valid_to IS NULL (still marked as current)
    - valid_from is more than ``stale_days`` ago (or is NULL — created without a date)
    - predicate doesn't start with 'evolved_from_' (provenance pointers are never stale)

    Returns a list of dicts sorted by valid_from ascending (oldest first).
    """
    cutoff = (datetime.now() - timedelta(days=stale_days)).strftime("%Y-%m-%d")

    with kg._lock:
        conn = kg._conn()
        rows = conn.execute(
            """
            SELECT
                t.id,
                e_s.name  AS subject_name,
                t.predicate,
                e_o.name  AS object_name,
                t.valid_from,
                t.source_file,
                t.source_closet
            FROM triples t
            JOIN entities e_s ON e_s.id = t.subject
            JOIN entities e_o ON e_o.id = t.object
            WHERE t.valid_to IS NULL
              AND t.predicate NOT LIKE 'evolved_from_%'
              AND (t.valid_from IS NULL OR t.valid_from <= ?)
            ORDER BY t.valid_from ASC NULLS FIRST
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()

    results = []
    for row in rows:
        age_label = "unknown"
        if row["valid_from"]:
            try:
                age_days = (datetime.now() - datetime.strptime(row["valid_from"], "%Y-%m-%d")).days
                age_label = f"{age_days} days"
            except ValueError:
                age_label = "unknown"

        results.append(
            {
                "triple_id": row["id"],
                "subject": row["subject_name"],
                "predicate": row["predicate"],
                "object": row["object_name"],
                "valid_from": row["valid_from"],
                "age": age_label,
                "source_file": row["source_file"],
                "source_closet": row["source_closet"],
                "reason": "no_recent_confirmation",
            }
        )
    return results


def check_silent_entities(
    kg,
    col,
    silence_days: int = _DEFAULT_ENTITY_SILENCE_DAYS,
    limit: int = 20,
) -> list:
    """Return entities not mentioned in recently-filed drawers.

    Queries the palace collection for the most recent ``filed_at`` among
    drawers whose ``entities`` metadata field mentions each entity.
    Entities with no drawer mention since ``silence_days`` ago are flagged.

    Returns a list of dicts sorted by last_seen ascending (oldest first).
    Empty when the palace collection is unavailable.
    """
    silence_cutoff = (datetime.now() - timedelta(days=silence_days)).strftime("%Y-%m-%d")

    with kg._lock:
        conn = kg._conn()
        entity_rows = conn.execute(
            "SELECT id, name FROM entities LIMIT 200"
        ).fetchall()

    if not entity_rows or col is None:
        return []

    # Fetch all drawer metadata once; scan for entity mentions.
    # Capped at 5000 to keep this call fast — palace may be large.
    try:
        batch = col.get(include=["metadatas"], limit=5000)
        all_meta = batch.get("metadatas") or []
    except Exception:
        return []

    # Build: entity_id → most recent filed_at found in drawer metadata
    latest_seen: dict = {}
    for meta in all_meta:
        filed_at = meta.get("filed_at", "") or ""
        entities_raw = meta.get("entities", "") or ""
        if not entities_raw:
            continue
        # entities field is stored as a comma-separated string
        for eid in str(entities_raw).split(","):
            eid = eid.strip().lower()
            if not eid:
                continue
            if filed_at > latest_seen.get(eid, ""):
                latest_seen[eid] = filed_at

    silent = []
    for row in entity_rows:
        eid = row["id"]
        last = latest_seen.get(eid, "")
        if last and last >= silence_cutoff:
            continue  # recently seen — not silent
        silent.append(
            {
                "entity": row["name"],
                "entity_id": eid,
                "last_seen_in_drawer": last or None,
                "silence_days": silence_days,
                "reason": "not_mentioned_in_recent_drawers",
            }
        )

    silent.sort(key=lambda x: x["last_seen_in_drawer"] or "")
    return silent[:limit]


def kg_staleness_report(
    kg,
    col=None,
    stale_days: int = _DEFAULT_STALE_DAYS,
    entity_silence_days: int = _DEFAULT_ENTITY_SILENCE_DAYS,
    limit: int = 50,
) -> dict:
    """Build the full KG staleness report.

    Args:
        kg: KnowledgeGraph instance.
        col: Palace ChromaDB collection (for entity-silence check). Pass None
            to skip the entity-silence section.
        stale_days: Facts older than this (with no valid_to) are flagged.
        entity_silence_days: Entities not mentioned in drawers for this many
            days are flagged as potentially forgotten.
        limit: Max items per section.

    Returns a dict with:
        stale_triples     — list of possibly-stale facts
        silent_entities   — list of entities not recently mentioned in drawers
        summary           — human-readable summary string
        checked_at        — ISO timestamp
        stale_threshold_days
        entity_silence_days
    """
    stale = check_stale_triples(kg, stale_days=stale_days, limit=limit)
    silent = check_silent_entities(
        kg, col, silence_days=entity_silence_days, limit=limit // 2
    ) if col is not None else []

    summary_parts = []
    if stale:
        summary_parts.append(
            f"{len(stale)} triple(s) not confirmed in >{stale_days} days"
        )
    if silent:
        summary_parts.append(
            f"{len(silent)} entity/ies not mentioned in drawers for >{entity_silence_days} days"
        )
    if not summary_parts:
        summary_parts.append("KG appears healthy — no stale facts detected")

    return {
        "stale_triples": stale,
        "silent_entities": silent,
        "summary": "; ".join(summary_parts),
        "checked_at": datetime.now().isoformat(),
        "stale_threshold_days": stale_days,
        "entity_silence_days": entity_silence_days,
    }
