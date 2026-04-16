"""
ambient.py — Ambient RAG and Socratic Engine for MemPalace.

Three on-demand cognitive capabilities:

    get_whisper(query, palace_path, n_wings)
        Proactive context: surfaces the best verbatim hit from each of the top
        N wings, so the AI sees relevant cross-project memory it didn't know
        to ask about.

    get_socratic_question(kg, context_entities)
        Socratic prompting: finds structurally sparse entities in the KG
        (low-degree nodes = things you know exist but haven't explored deeply)
        and frames a question to surface their missing connections.

    get_eigen_thoughts(kg, n)
        Cognitive pillars: PageRank over the KG to find the entities that
        everything else points to or from. These are the load-bearing concepts
        in your knowledge graph.

All functions are local-only, zero-API, zero-new-deps.
"""

import random

from .topology import entity_degree, pagerank


# ── Whisper ───────────────────────────────────────────────────────────────────

def get_whisper(query: str, palace_path: str, n_wings: int = 3) -> dict:
    """Surface the best cross-wing hit per top wing for a query.

    Unlike mempalace_search (which returns the global top-N), whisper deliberately
    ensures representation from multiple wings so the AI sees context from projects
    it might not think to ask about.

    Pipeline:
    1. Run hybrid search with cross_wing_balance=True (already implemented).
    2. Group results by wing.
    3. Pick the single best hit (highest similarity) from each wing.
    4. Return the top n_wings winners.

    Args:
        query: What to look for.
        palace_path: Path to the palace directory.
        n_wings: How many distinct wings to surface (default 3).

    Returns:
        {
            "query": str,
            "whispers": [{"wing", "text", "similarity", "source_file", "room"}],
            "total_wings_found": int,
        }
    """
    from .searcher import search_memories

    # Request more results than needed so we have enough to pick one per wing
    raw = search_memories(
        query=query,
        palace_path=palace_path,
        n_results=n_wings * 6,
        cross_wing_balance=True,
    )

    hits = raw.get("results", [])
    if not hits:
        return {"query": query, "whispers": [], "total_wings_found": 0}

    # Group by wing, keep best hit per wing (already sorted by score)
    seen_wings: dict = {}
    for hit in hits:
        wing = hit.get("wing", "unknown")
        if wing not in seen_wings:
            seen_wings[wing] = {
                "wing": wing,
                "text": hit.get("text", ""),
                "similarity": hit.get("similarity", 0.0),
                "source_file": hit.get("source_file", ""),
                "room": hit.get("room", ""),
            }

    whispers = list(seen_wings.values())[:n_wings]
    return {
        "query": query,
        "whispers": whispers,
        "total_wings_found": len(seen_wings),
    }


# ── Socratic ──────────────────────────────────────────────────────────────────

# Question templates keyed by degree bucket (how isolated the entity is)
_QUESTION_TEMPLATES = {
    "isolated": [
        "We know {entity} exists in the knowledge graph but have almost no facts about them. What do we know about {entity}?",
        "{entity} appears in memory but has very few connections. What should we record about {entity}?",
        "What is the current status of {entity}? We have almost no facts about them.",
    ],
    "sparse": [
        "We have a few facts about {entity} but the picture feels incomplete. What else do we know?",
        "{entity} is mentioned in context: {context}. What happened after that?",
        "The knowledge graph shows {entity} connected to {connections}. What are we missing?",
    ],
    "medium": [
        "We know quite a bit about {entity}. Has anything changed recently?",
        "{entity} is connected to {connections}. Are all these facts still current?",
    ],
}


def get_socratic_question(kg, context_entities: list = None) -> dict:
    """Generate a Socratic question based on structural holes in the KG.

    Finds the entity that is most "underexplored" — present in the graph but
    with very few relationships relative to how important it seems. Prioritises
    entities that appear in context_entities (active session entities) so the
    question is grounded in what's being worked on now.

    Args:
        kg: KnowledgeGraph instance.
        context_entities: List of entity names currently in scope (optional).
                          When provided, entities in this list are weighted
                          higher so the question stays relevant.

    Returns:
        {
            "question": str,
            "entity": str,
            "reasoning": str,
            "entity_degree": int,
        }
        Returns {"question": None, ...} if the KG is empty.
    """
    degrees = entity_degree(kg)
    if not degrees:
        return {
            "question": None,
            "entity": None,
            "reasoning": "Knowledge graph is empty — add some facts first.",
            "entity_degree": 0,
        }

    # Score candidates: lower degree = higher priority, but context entities get a boost
    context_set = {e.lower() for e in (context_entities or [])}

    def _candidate_score(entry):
        degree = entry["total"]
        # Context boost: prefer entities relevant to current session
        context_boost = 3 if entry["entity"].lower() in context_set else 0
        # Avoid zero-degree noise: prefer entities with at least 1 connection
        if degree == 0:
            return -0.5 + context_boost
        return context_boost - degree  # lower degree → higher score

    candidates = sorted(degrees, key=_candidate_score, reverse=True)
    target = candidates[0]
    entity_name = target["entity"]
    degree = target["total"]

    # Fetch a sample of the entity's connections for template interpolation
    try:
        facts = kg.query_entity(entity_name, direction="both")
        connections = ", ".join(
            f"{f['predicate']} {f['object']}" for f in facts[:3]
        ) or "nothing yet"
        context_str = f"{facts[0]['predicate']} {facts[0]['object']}" if facts else "few facts"
    except Exception:
        connections = "nothing yet"
        context_str = "few facts"

    # Pick template bucket
    if degree == 0:
        bucket = "isolated"
    elif degree <= 2:
        bucket = "sparse"
    else:
        bucket = "medium"

    templates = _QUESTION_TEMPLATES[bucket]
    template = random.choice(templates)  # noqa: S311 — non-cryptographic
    question = template.format(
        entity=entity_name,
        connections=connections,
        context=context_str,
    )

    return {
        "question": question,
        "entity": entity_name,
        "reasoning": (
            f"'{entity_name}' has {degree} KG connection(s) — "
            f"{'present in current context, ' if entity_name.lower() in context_set else ''}"
            f"structurally underexplored."
        ),
        "entity_degree": degree,
    }


# ── Eigen thoughts ────────────────────────────────────────────────────────────

def get_eigen_thoughts(kg, n: int = 5) -> dict:
    """Return the top-N cognitive pillars via PageRank.

    The highest-PageRank entities are the ones everything else in your knowledge
    graph references — they are the load-bearing pillars of your personal
    knowledge architecture.

    Args:
        kg: KnowledgeGraph instance.
        n: How many pillars to return (default 5).

    Returns:
        {
            "pillars": [{"rank", "entity", "score"}],
            "total_entities": int,
            "message": str,  # human-readable interpretation
        }
    """
    ranked = pagerank(kg)
    if not ranked:
        return {
            "pillars": [],
            "total_entities": 0,
            "message": "Knowledge graph is empty — add some facts to discover pillars.",
        }

    top = ranked[:n]
    pillars = [
        {"rank": i + 1, "entity": entry["entity"], "score": entry["score"]}
        for i, entry in enumerate(top)
    ]

    top_names = ", ".join(p["entity"] for p in pillars[:3])
    message = (
        f"Your knowledge graph has {len(ranked)} entities. "
        f"The top {'pillar' if n == 1 else 'pillars'} — {top_names} — "
        f"are the concepts everything else references."
    )

    return {
        "pillars": pillars,
        "total_entities": len(ranked),
        "message": message,
    }
