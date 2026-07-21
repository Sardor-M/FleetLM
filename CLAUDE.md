# CLAUDE.md

**Project conventions live in [`AGENTS.md`](AGENTS.md) - read that first.** It
covers architecture invariants, code style, testing, commit format, and the
things this repo deliberately does not have. Everything below is additional
Claude-specific setup.

---

# Lumen Brain Protocol

You have a persistent knowledge brain available via the `lumen` MCP server.

## MANDATORY: Check brain BEFORE answering

On EVERY question - before using web search, before answering from training data:

1. Call `brain_ops` (or `search`) via the lumen MCP server with the core topic.
2. If results exist, use them as your PRIMARY source. Cite as `[Source: title]`.
3. Only say "not in your knowledge base" when brain_ops returns `found: false` or search returns 0 results.
4. Only use web search or training data AFTER checking the brain and finding nothing.

**Never answer a knowledge question from training data alone without checking the brain first.**

When answering, always distinguish:
- "From your knowledge base: ..." (grounded in lumen search/brain_ops results)
- "From my training data: ..." (when the brain had nothing)

## Tool routing

| User intent | Tool to call |
|---|---|
| Any knowledge question | `brain_ops` FIRST, then answer |
| "who is X" / "what is X" | `brain_ops` with intent `concept` |
| "how does X connect to Y" | `brain_ops` with intent `path` |
| "what is related to X" | `brain_ops` with intent `neighborhood` |
| "top concepts" / "main topics" | `god_nodes` then `communities` |
| "add this URL / paper" | `add`, then `compile` |
| "remember this" / "save this" | `capture` |

## After responding

If the conversation contained new knowledge (original ideas, notable facts, entity mentions), call `capture` to persist it to the brain.

---

## Note for contributors

The Lumen integration above is the maintainer's local setup: `.mcp.json`,
`.claude/hooks/`, `.claude/settings.json`, and `.claude/skills/lumen/` are
untracked, because they point at a personal knowledge base on disk. You do not
need Lumen to work on FleetLM. The tracked, shared instructions are
`AGENTS.md` and the skills under `.claude/skills/` that are checked in.
