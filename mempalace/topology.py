"""
topology.py — Pure-Python graph analysis over the MemPalace Knowledge Graph.

No NetworkX. No scipy. Stdlib only (collections).
Reads the KG's SQLite triples directly for efficiency — one query per analysis,
not N calls to query_entity().

Public API:
    pagerank(kg, damping, iterations) → sorted list of {entity, score}
    entity_degree(kg)                → list of {entity, in_degree, out_degree, total}
"""

import collections


def pagerank(kg, damping: float = 0.85, iterations: int = 50) -> list:
    """Compute PageRank over KG entities.

    Uses the standard power-iteration formula:
        PR(u) = (1 - d) / N  +  d * Σ PR(v) / out_degree(v)   for all v → u

    Convergence is not checked — fixed iterations is sufficient for the small
    graphs typical of a personal KG (hundreds to low-thousands of nodes).

    Args:
        kg: KnowledgeGraph instance (uses kg._conn() for raw SQL).
        damping: Damping factor (0.85 is the standard web-graph value).
        iterations: Power-iteration steps (50 is overkill for small graphs).

    Returns:
        List of {"entity": name, "score": float} sorted by score descending.
        Returns [] if the KG has no entities.
    """
    with kg._lock:
        conn = kg._conn()
        # Fetch all entities
        entity_rows = conn.execute("SELECT id, name FROM entities").fetchall()
        if not entity_rows:
            return []

        id_to_name = {row["id"]: row["name"] for row in entity_rows}
        entity_ids = list(id_to_name.keys())
        n = len(entity_ids)

        # Build out-adjacency and in-adjacency from current (valid) triples only.
        # We only traverse edges where valid_to IS NULL (fact still true) so the
        # graph reflects the current state of the world, not stale facts.
        out_edges = collections.defaultdict(set)   # src → {dst, ...}
        in_edges = collections.defaultdict(set)    # dst → {src, ...}

        triple_rows = conn.execute(
            "SELECT subject, object FROM triples WHERE valid_to IS NULL"
        ).fetchall()
        for row in triple_rows:
            s, o = row["subject"], row["object"]
            if s in id_to_name and o in id_to_name:
                out_edges[s].add(o)
                in_edges[o].add(s)

    if n == 0:
        return []

    # Initialise scores uniformly
    scores = {eid: 1.0 / n for eid in entity_ids}
    # Dangling nodes (no out-edges) distribute their score equally to all nodes
    base = (1.0 - damping) / n

    for _ in range(iterations):
        new_scores: dict = {}
        dangling_sum = sum(scores[eid] for eid in entity_ids if not out_edges[eid])
        dangling_contrib = damping * dangling_sum / n

        for eid in entity_ids:
            incoming = sum(scores[src] / len(out_edges[src]) for src in in_edges[eid])
            new_scores[eid] = base + dangling_contrib + damping * incoming
        scores = new_scores

    return sorted(
        [{"entity": id_to_name[eid], "score": round(scores[eid], 6)} for eid in entity_ids],
        key=lambda x: x["score"],
        reverse=True,
    )


def entity_degree(kg) -> list:
    """Compute in-degree, out-degree, and total degree for every KG entity.

    Only counts current (valid_to IS NULL) triples so the degree reflects the
    live graph, not historical states.

    Returns:
        List of {"entity": name, "in_degree": int, "out_degree": int, "total": int}
        sorted by total degree descending.
    """
    with kg._lock:
        conn = kg._conn()
        entity_rows = conn.execute("SELECT id, name FROM entities").fetchall()
        if not entity_rows:
            return []

        id_to_name = {row["id"]: row["name"] for row in entity_rows}

        out_count = collections.Counter()
        in_count = collections.Counter()

        triple_rows = conn.execute(
            "SELECT subject, object FROM triples WHERE valid_to IS NULL"
        ).fetchall()
        for row in triple_rows:
            s, o = row["subject"], row["object"]
            if s in id_to_name:
                out_count[s] += 1
            if o in id_to_name:
                in_count[o] += 1

    result = []
    for eid, name in id_to_name.items():
        out_d = out_count[eid]
        in_d = in_count[eid]
        result.append({
            "entity": name,
            "in_degree": in_d,
            "out_degree": out_d,
            "total": in_d + out_d,
        })

    return sorted(result, key=lambda x: x["total"], reverse=True)
