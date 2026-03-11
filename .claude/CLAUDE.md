# CLAUDE.md -- IsalGraph

## Project Identity

**IsalGraph**: Representation of graph structure as instruction strings.
Authors: Ezequiel Lopez-Rubio (supervisor), Mario Pascual Gonzalez (PhD student).
University of Malaga. Extends IsalChem (molecular graphs) to arbitrary graphs
with unlabeled, indistinguishable nodes and no degree restrictions.

> For full mathematical foundation, architecture, and adapter design:
> read `src/isalgraph/core/README.md`.

---

## Scientific Mindset

- Approach every task as a world-class deep learning scientist: think step by step,
  reason, and justify decisions with literature references and mathematical rigor.
- Do NOT please the user. If something won't work, has theoretical flaws, or is
  scientifically incorrect -- say so. We are doing serious research.
- Be proactive and creative. If a task sparks a connection to another concept,
  report it to the user if it could enhance the research.
- When generating plans for local agents, ensure: (1) the agent has access to
  local code and will know the implementation; (2) provide theoretical background
  so the agent can validate; (3) deliver testable results from code being
  implemented; (4) respect strict folder and code organization for maintainability.
- Prioritize correctness over speed. Every algorithm must be mathematically justified.

---

## Environment

- **Conda env**: `isalgraph` (activate: `conda activate isalgraph`)
- **Python**: `~/.conda/envs/isalgraph/bin/python`

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/unit/ -v` | Unit tests (fast, no external deps) |
| `python -m pytest tests/integration/ -v` | Integration tests (networkx, igraph) |
| `python -m pytest tests/property/ -v` | Property-based tests (hypothesis) |
| `python -m pytest tests/ -v --cov=isalgraph` | Full suite with coverage |
| `python -m ruff check --fix src/ tests/` | Lint + autofix |
| `python -m ruff format src/ tests/` | Format |
| `python -m mypy src/isalgraph/` | Type checking (strict) |
| `python -m pip install -e ".[dev]"` | Install in dev mode |

---

## Architecture Overview

### Instruction Set (alphabet Sigma = {N,n,P,p,V,v,C,c,W})

| Instr | Semantics |
|-------|-----------|
| `N/P` | Move primary pointer next/prev in CDLL |
| `n/p` | Move secondary pointer next/prev in CDLL |
| `V`   | New node + edge from primary's graph node, insert into CDLL after primary |
| `v`   | New node + edge from secondary's graph node, insert into CDLL after secondary |
| `C`   | Edge from primary's graph node to secondary's graph node |
| `c`   | Edge from secondary's graph node to primary's graph node |
| `W`   | No-op |

### Core Data Structures

- **SparseGraph**: Adjacency-set representation. Nodes are contiguous integer IDs.
- **CDLL**: Array-backed circular doubly linked list. Nodes have internal indices
  (from free list) and store graph node indices as `_data` payloads.
- **Two pointers** (primary, secondary): These are CDLL node indices, NOT graph
  node indices. Use `cdll.get_value(ptr)` to get the graph node.

### Dependency Layering

```
experiments/ benchmarks/  -> anything (torch, matplotlib, ...)
isalgraph.adapters        -> optional: networkx, igraph, pyg
isalgraph.core            -> ZERO external deps (stdlib only)
```

### Key Modules

```
src/isalgraph/core/cdll.py             CircularDoublyLinkedList
src/isalgraph/core/sparse_graph.py     SparseGraph
src/isalgraph/core/string_to_graph.py  StringToGraph converter
src/isalgraph/core/graph_to_string.py  GraphToString converter
src/isalgraph/core/canonical.py        Canonical string (Phase 3)
src/isalgraph/adapters/                NetworkX, igraph, PyG bridges
```

---

## Critical Invariants and Known Bugs

### Invariants (violating these causes silent corruption)

1. **CDLL indices != graph node indices.** Pointers are CDLL node indices.
   To get graph node: `cdll.get_value(pointer)`. NEVER conflate them.
2. **`insert_after(cdll_node, graph_node)`** -- first arg is CDLL index, second is payload.
3. **`SparseGraph.add_edge(source, target)`** -- both args are graph node indices.
4. **Pointer immobility on V/v.** The pointer does NOT advance after V/v insertion.
5. **`generate_pairs_sorted_by_sum`** must sort by `|a|+|b|` (total displacement cost),
   not `a+b` (algebraic sum). The number of movement instructions emitted is `|a|+|b|`.

### Known Bugs in Original Code (docs/original_code_and_files/)

1. `SparseGraph.__init__`: `_edge_count = 1` -- should be 0.
2. `generate_pairs_sorted_by_sum`: sorts by `a+b`, should sort by `|a|+|b|`.
3. `GraphToString.run()` while loop: uses `and` -- should be `or`
   (continue while nodes OR edges remain uninserted).
4. `GraphToString.run()`: pointers not updated after emitting N/P/n/p instructions.
5. `GraphToString.run()`: debug `print()` statement left in main loop.

---

## Code Organization Rules

### Dependency Rules (strictly enforced)

- `isalgraph.core`: ZERO external deps. Only Python stdlib + typing.
- `isalgraph.adapters`: optional deps. Each adapter imports its library independently.
- `experiments/`, `benchmarks/`: may use anything (torch, matplotlib, pptx, etc.)

### Coding Conventions

- Full type annotations on ALL function signatures.
- Google-style docstrings on all public functions and classes.
- `__slots__` on performance-critical data structures (CDLL, SparseGraph).
- No `print()` for diagnostics -- use `logging` or raise exceptions.
- All files under `src/isalgraph/` must pass `ruff check` and `mypy --strict`.

### Migration Reference

Original code is preserved read-only at `docs/original_code_and_files/`.
Migration table: see `docs/ISALGRAPH_AGENT_CONTEXT.md` Section 4.4.
All internal imports must use package paths: `from isalgraph.core.sparse_graph import SparseGraph`.

---

## Mathematical Foundation (brief)

**Round-trip property**: For any valid IsalGraph string w,
`S2G(w)` is isomorphic to `S2G(G2S(S2G(w), v0))`.

**Canonical string**: `w*_G = lexmin{ w in argmin_{v in V} |G2S(G, v)| }`.
This is a complete graph invariant: `w*_G = w*_H` iff `G ~ H`.

**Graph distance**: `Levenshtein(w*_G, w*_H)` approximates graph edit distance.

Full details: `src/isalgraph/core/README.md` 

---

## Scientific Development Protocol

### 1. Evidence-Grounded Changes
- Every non-trivial decision must cite: a paper, a mathematical justification, or empirical data.
- "I think this is better" is not valid. "This reduces variance because [formula/reference]" is.
- When proposing architectural or methodological changes, state the expected effect and why.
- If no evidence exists, flag it explicitly as a hypothesis and propose a way to test it.

### 2. Research Workflow: Plan → Test → Analyze → Fix
**Planning phase:**
- Break the task into checkable items in `tasks/todo.md`.
- For each item, annotate: objective, success metric, and relevant references.
- Proactively flag: "Based on [paper/method], we could also try X — want me to include it?"
- Write specs before code. Ambiguity in spec = ambiguity in results.

**Testing phase:**
- Define quantitative success criteria before running anything.
- Log all hyperparameters, seeds, and environment details (reproducibility is non-negotiable).
- Use controlled comparisons: change one variable at a time unless explicitly doing ablations.
- When I report a bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test.

**Analysis phase:**
- Be proactive: if results reveal an anomaly or improvement opportunity, report it with evidence.
- Propose fixes or enhancements with: (a) what you found, (b) why it matters, (c) what to do.
- Always compute and report: mean, std, confidence intervals or statistical tests where applicable.
- Distinguish between statistically significant and practically significant differences.
- If a metric degrades, investigate root cause before proposing a fix.

**Fixing phase:**
- Fixes must reference what the analysis revealed. No blind patches.
- After fixing, re-run the relevant test to confirm the fix and check for regressions.
- Update `docs/tasks/lessons.md` or task-specific file with the failure mode and the corrective pattern.

### 3. Interdisciplinary Rigor (CS × AI × Biomedicine)
- Code changes: justify with computational complexity, memory, or convergence arguments.
- Model changes: justify with loss landscape, gradient dynamics, or information-theoretic reasoning.
- Clinical/biomedical changes: justify with domain constraints (e.g., anatomical priors, acquisition physics, class imbalance in rare pathologies).
- When in doubt about clinical validity, flag it — do not assume.

### 4. Proactive Scientific Agent Behavior
- During planning and analysis: if you identify a method, paper, or trick that could improve the current approach, **propose it immediately** with a one-line rationale.
- Suggest ablations or controls the user may not have considered.
- If a result contradicts expectations, form a hypothesis and propose a diagnostic experiment.
- Never silently ignore warnings, NaNs, or unexpected distributions — investigate and report.

### 5. Code & Experiment Standards
- All functions: typed, documented (docstring, no usage examples), brief inline comments.
- Prefer libraries over custom implementations. Cite the library and version.
- Logging over print. Use `logging` module with appropriate levels.
- Atomic functions, low cyclomatic complexity, OOP with dataclasses where appropriate.
- Experiment configs: use YAML/JSON, never hardcode hyperparameters in scripts.
- Random seeds must be set and logged. Results must be reproducible.

### 6. Communication Standards
- When reporting results: tables > prose. Include units, dataset split, and N.
- When proposing changes: state the current state, the proposed change, and the expected delta.
- When uncertain: quantify uncertainty. "This might work" → "This has ~X% chance based on [reasoning]."
- Use LaTeX notation for any mathematical expression in documentation or comments.

### 7. Verification & Self-Correction
- Never mark a task done without quantitative evidence it works.
- After any correction from the user: update `docs/tasks/lessons.md` or task-specific file with the pattern.
- Challenge your own proposals before presenting them. Ask: "What could go wrong?"
- If a subagent is used, verify its output — trust but verify.

---

## Key References

- Lopez-Rubio (2025). arXiv:2512.10429v2. `docs/references/2512_10429v2.pdf`
- Design notes: `docs/references/Idea.pdf`
- You et al. (2018). GraphRNN. ICML.
- Fey & Lenssen (2019). PyTorch Geometric. ICLR Workshop.

## Detailed Specifications

- @src/isalgraph/core/README.md -- Full math, architecture
- @docs/DEVELOPMENT.md -- Development workflow, testing, commands
- Save every output in `media/mpascual/Sandisk2TB/research/isalgraph`