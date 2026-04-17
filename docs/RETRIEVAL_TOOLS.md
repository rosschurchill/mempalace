# Retrieval Tools — When to Use Which

MemPalace exposes three retrieval tools that all "find stuff," but they're
optimised for different question shapes. Pick the one that matches your intent.

## Decision matrix

| Question shape | Tool | Why |
|---|---|---|
| "Find any drawers matching X" | **`mempalace_search`** | Generic hybrid search. Fast, unfiltered, returns top-N by similarity. |
| "Why did we choose X?" / "What did we decide about Y?" | **`mempalace_explain`** | Auto-detects entities, queries the KG for temporal facts, scopes search to the relevant wing, returns structured reasoning + alternatives. |
| "What's relevant to what I'm working on now?" (no specific target) | **`mempalace_whisper`** | Returns the best memory from each of the top-N wings. Surfaces context you didn't think to ask about. |

## Concrete examples

### `mempalace_search` — the default

```
mempalace_search(query="GraphQL migration", wing="myapp")
```

Use when:
- You know what you're looking for and just need the verbatim content
- You want raw hits without KG enrichment or wing auto-detection
- You're feeding results into another tool that will interpret them

### `mempalace_explain` — for decisions and reasoning

```
mempalace_explain(query="Why did we choose Postgres for Orion?")
```

Use when:
- Your question starts with "why", "how did we", "what did we decide"
- The query mentions a specific project, person, or concept by name
- You want to see both the reasoning AND the current state (KG facts)
- You care about alternatives that were considered and rejected

What makes it different: it reads the Entity Registry to find known entities,
queries the Knowledge Graph for temporal facts about each one, auto-scopes the
search to the wing matching the entity, and returns a structured response
with `entities_detected`, `kg_facts`, `wing_scope`, `results`, and a
`reasoning` field explaining what it did.

### `mempalace_whisper` — proactive context

```
mempalace_whisper(query="database performance")
```

Use when:
- Starting a new session and you want a broad sweep of relevant memory
- You want representation from multiple projects, not just the top hits from one
- You're mid-task and want to check "what else do I know about this that I might be forgetting?"

What makes it different: `search` returns the top-N by score (which can all
come from the same wing). `whisper` returns the best single hit from each of
the top-N *different* wings — so you see cross-project context that pure
similarity ranking would bury.

## Under the hood

All three use the same underlying hybrid search (BM25 + vector + closet boost +
MMR + cross-wing balancing). The difference is in:

- **Query interpretation** — `explain` parses entities; others pass through
- **Filtering** — `explain` auto-scopes to detected wing; others don't
- **Grouping** — `whisper` post-groups by wing; others rank globally
- **Enrichment** — `explain` adds KG facts; others return only drawer results

## Phase 4 cognition tools (different category)

These aren't retrieval — they're introspection:

- **`mempalace_pillars`** — PageRank over the KG. Who/what is most connected?
- **`mempalace_socratic`** — "What have you forgotten to record?"
- **`mempalace_rem_cycle`** — Discover semantic bridges between wings

## Rule of thumb

> "Find stuff" → `search`
> "Explain stuff" → `explain`
> "Remind me of stuff" → `whisper`
