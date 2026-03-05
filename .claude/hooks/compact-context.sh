#!/usr/bin/env bash
# Compaction recovery hook -- re-injects critical context after /compact

cat <<'CONTEXT'
=== ISALGRAPH COMPACTION RECOVERY ===

INSTRUCTION SET: {N,n,P,p,V,v,C,c,W}
  N/P: move primary pointer next/prev in CDLL
  n/p: move secondary pointer next/prev in CDLL
  V/v: insert new vertex + edge from primary/secondary, insert into CDLL after pointer
  C/c: connect primary->secondary / secondary->primary
  W: no-op

CRITICAL INVARIANT:
  Pointers are CDLL node indices, NOT graph node indices.
  To get graph node: cdll.get_value(pointer)
  NEVER pass pointer directly to SparseGraph.add_edge()
  Pointer does NOT move after V/v instruction.

KNOWN BUGS IN ORIGINAL CODE (documented, not yet fixed):
  1. SparseGraph._edge_count initialized to 1 (should be 0)
  2. generate_pairs_sorted_by_sum: sorts by a+b, should sort by |a|+|b|
  3. GraphToString while loop: uses AND (should be OR for nodes/edges remaining)
  4. GraphToString: pointers not updated after emitting movement instructions
  5. Debug print() left in GraphToString main loop

DEPENDENCY RULE:
  isalgraph.core = ZERO external deps (stdlib only)
  isalgraph.adapters = optional (networkx, igraph, pyg)

ENVIRONMENT:
  Conda env: isalgraph
  Tests: python -m pytest tests/ -v
  Lint: python -m ruff check src/ tests/
  Types: python -m mypy src/isalgraph/

ROUND-TRIP PROPERTY:
  S2G(w) ~ S2G(G2S(S2G(w), v0)) for all valid strings w
  ~ denotes graph isomorphism

KEY FILES:
  src/isalgraph/core/         -- Core implementation (zero deps)
  docs/ISALGRAPH_AGENT_CONTEXT.md -- Full mathematical + architectural spec (583 lines)
  docs/references/            -- Preprint + design notes PDFs
  .claude/CLAUDE.md           -- Project hub (<200 lines)

=== END COMPACTION RECOVERY ===
CONTEXT
