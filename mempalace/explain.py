"""
explain.py — Decision archaeology pipeline for MemPalace.

mempalace_explain wires four previously disconnected modules into one
intelligent query pipeline:

  Entity Registry  → extract_people_from_query()     (exists, never called from search)
  Knowledge Graph  → query_entity()                  (exists, never called from search)
  Wing scoping     → 34% retrieval boost from filter  (exists, never auto-applied)
  Hybrid search    → search_memories()               (exists, always unscoped)

Usage (via MCP):
  mempalace_explain("Why did we choose Postgres for Orion?")

Pipeline:
  1. Extract entities from query (people from registry + capitalized candidates)
  2. Query KG for each entity's temporal facts
  3. Auto-detect which wing to scope to (entity name → wing_{entity_id})
  4. Run hybrid search scoped to detected wing
  5. Return structured result: entities + kg_facts + scoped search hits
"""

import re
import time
from typing import Optional


# Capitalized words that look like entities but aren't — sentence starters,
# common adjectives, days, months. Filtered out before entity detection.
_ENTITY_STOPWORDS = frozenset(
    {
        "The", "This", "That", "These", "Those", "When", "Where", "What", "Why",
        "Who", "Which", "How", "After", "Before", "Then", "Now", "Here", "There",
        "And", "But", "Or", "Yet", "So", "If", "Else", "Yes", "No", "Maybe",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December", "Did", "Does", "Do",
        "Can", "Should", "Would", "Could", "Will", "Was", "Were", "Has", "Have",
        "Had", "Are", "Is", "Be", "Been", "Being", "Let", "Make", "Get",
        "Want", "Need", "Use", "Used", "Using", "Also", "Just", "Very",
    }
)


def _extract_candidates(query: str) -> list:
    """Extract capitalized-word entity candidates from a query string.

    Returns unique candidates that aren't in the stop-word list, ordered
    by appearance.
    """
    # Match CamelCase, PascalCase, and plain Capitalized words (3+ chars)
    raw = re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", query)
    seen: dict = {}
    for word in raw:
        if word not in _ENTITY_STOPWORDS and word not in seen:
            seen[word] = None
    return list(seen.keys())


def _entity_to_wing_id(name: str) -> str:
    """Derive the canonical wing id from an entity name.

    Matches how wings are named at mine time:
        name.lower().replace(' ', '_').replace('-', '_')
    """
    return name.lower().replace(" ", "_").replace("-", "_")


def _detect_wing(entity_candidates: list, all_wings: set) -> Optional[str]:
    """Find the best wing match for the detected entities.

    Tries: wing_{entity_id} for each candidate, returns the first match.
    Falls back to None (unscoped search) if nothing matches.
    """
    for name in entity_candidates:
        candidate_wing = f"wing_{_entity_to_wing_id(name)}"
        if candidate_wing in all_wings:
            return candidate_wing
        # Also try without the wing_ prefix in case user stored as plain name
        plain = _entity_to_wing_id(name)
        if plain in all_wings:
            return plain
    return None


# Wings cache — avoids paging the entire palace on every explain() call.
# Keyed by id(col) since the same collection object is shared across calls
# within one MCP server session; TTL prevents stale results after mine.
_WINGS_CACHE: dict = {}
_WINGS_CACHE_TTL_SECONDS = 30.0


def _get_all_wings(col) -> set:
    """Read distinct wing names from palace metadata (paginated, cached).

    Without caching this pages through the full collection on every
    explain() call — O(N) per query on large palaces. The cache has a
    30-second TTL so it picks up new wings shortly after mining.
    """
    key = id(col)
    now = time.time()
    cached = _WINGS_CACHE.get(key)
    if cached and (now - cached[0]) < _WINGS_CACHE_TTL_SECONDS:
        return cached[1]

    wings: set = set()
    try:
        total = col.count()
        offset = 0
        while offset < total:
            batch = col.get(include=["metadatas"], limit=500, offset=offset)
            metas = batch.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                w = m.get("wing")
                if w:
                    wings.add(w)
            offset += len(metas)
    except Exception:
        pass

    _WINGS_CACHE[key] = (now, wings)
    return wings


def run_explain(
    query: str,
    palace_path: str,
    col,
    kg,
    entity_registry,
    wing_override: Optional[str] = None,
    memory_type: Optional[str] = None,
    include_kg: bool = True,
    n_results: int = 5,
) -> dict:
    """Core explain pipeline — called by the MCP tool handler.

    Args:
        query: Natural language question (e.g. "Why did we choose Postgres?")
        palace_path: Path to the palace directory (for search_memories).
        col: Palace ChromaDB collection (for wing enumeration).
        kg: KnowledgeGraph instance.
        entity_registry: EntityRegistry instance.
        wing_override: Explicit wing to scope to (skips auto-detection).
        memory_type: Optional room filter (e.g. "decisions", "problems").
        include_kg: Whether to enrich with KG facts (default True).
        n_results: Max results to return.

    Returns a dict with:
        query, entities_detected, kg_facts, wing_scope, results, reasoning
    """
    from .searcher import search_memories

    # ── Step 1: Entity extraction ─────────────────────────────────────────────
    # Try registry first (known people), then fall back to capitalized candidates.
    known_people: list = []
    try:
        known_people = entity_registry.extract_people_from_query(query)
    except Exception:
        pass

    capitalized_candidates = _extract_candidates(query)

    # Combine: known people first (higher confidence), then novel capitalized terms
    all_candidates = list(known_people)
    for c in capitalized_candidates:
        if c not in all_candidates:
            all_candidates.append(c)

    # ── Step 2: KG enrichment ─────────────────────────────────────────────────
    kg_facts: list = []
    kg_enriched_entities: list = []
    if include_kg and all_candidates:
        for entity in all_candidates[:5]:  # cap at 5 entities to avoid bloat
            try:
                facts = kg.query_entity(entity, direction="both")
                if facts:
                    kg_enriched_entities.append(entity)
                    for f in facts[:8]:  # cap per-entity facts
                        kg_facts.append({
                            "entity": entity,
                            "subject": f.get("subject", ""),
                            "predicate": f.get("predicate", ""),
                            "object": f.get("object", ""),
                            "valid_from": f.get("valid_from"),
                            "valid_to": f.get("valid_to"),
                            "current": f.get("current", True),
                            "source_closet": f.get("source_closet"),
                        })
            except Exception:
                pass

    # ── Step 3: Wing auto-detection ───────────────────────────────────────────
    detected_wing = wing_override
    reasoning_parts: list = []

    if not detected_wing and all_candidates:
        all_wings = _get_all_wings(col)
        detected_wing = _detect_wing(all_candidates, all_wings)
        if detected_wing:
            reasoning_parts.append(
                f"Auto-scoped to '{detected_wing}' based on entity detection."
            )
        else:
            reasoning_parts.append(
                "No matching wing found for detected entities — running unscoped search."
            )
    elif detected_wing:
        reasoning_parts.append(f"Wing override: '{detected_wing}'.")

    if not all_candidates:
        reasoning_parts.append("No entities detected in query — running unscoped search.")

    if kg_enriched_entities:
        reasoning_parts.append(
            f"KG enrichment: found {len(kg_facts)} facts for "
            f"{', '.join(kg_enriched_entities)}."
        )

    # ── Step 4: Scoped hybrid search ──────────────────────────────────────────
    search_result = search_memories(
        query=query,
        palace_path=palace_path,
        wing=detected_wing,
        room=memory_type,
        n_results=n_results,
    )

    hits = search_result.get("results", [])

    # ── Step 5: Build response ────────────────────────────────────────────────
    response: dict = {
        "query": query,
        "entities_detected": all_candidates[:10],
        "kg_facts": kg_facts,
        "wing_scope": detected_wing,
        "results": hits,
        "total_results": len(hits),
        "reasoning": " ".join(reasoning_parts) or "No additional context.",
    }

    if memory_type:
        response["room_filter"] = memory_type

    if search_result.get("warning"):
        response["warning"] = search_result["warning"]

    return response
