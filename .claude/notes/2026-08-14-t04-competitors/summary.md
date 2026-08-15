# Wave 2026-08-14-t04-competitors — summary

**Date**: 2026-08-15. **Ticket**: T-04, competitor backends.
**Orchestrator branch**: `ticket/T-04-competitors` · **Integration branch**:
`integration/2026-08-14-t04-competitors`.
**Wave base**: `9d2291b87b37c9b4fbee9ce994f779dfd69db01c`.
**Result**: 26 files, **+9,510 / −20**, 25 commits, **383 T-04 tests**.

## The user's prompt, verbatim

> Invoque /review-ticket and solve T-04 (.claude/notes/review/plan/tickets.md); Read
> .claude/notes/review/tasks/T-04-session-prompt.md; The ticket has already been planned in
> .claude/notes/review/tasks/T-04-design.md ; As orchestrator, work in a separate branch from
> main, subagents will have branches that start from your branch. Iterate for as long as you
> need. Think step by step,

---

## Agents

| Agent | Model / effort | Branch | Head | Verdict | Log |
|---|---|---|---|---|---|
| `competitor-serial` (A) | Opus 5 / xhigh | `worktree-agent-a6e805f2cd1aa441f` | `3d8c7f1` | **ACCEPT** | [track-A-serial.md](track-A-serial.md) |
| `competitor-canonical` (B) | Opus 5 / xhigh | `track-B-canonical` | `b8c7f66` | **ACCEPT** | [track-B-canonical.md](track-B-canonical.md) |
| `competitor-mining` (C) | Opus 5 / xhigh | `track-C-mining` | `01d7d62` | **ACCEPT** | [track-C-mining.md](track-C-mining.md) |

Every log was verified against `git diff --stat` and every acceptance check re-run by the
orchestrator. Details and independent re-derivations: [VERIFICATION.md](VERIFICATION.md).
All three worktrees ended with `git status --porcelain` empty and the import shim correctly
uncommitted. **No merge conflicts** — the ownership partition held.

---

## What the wave produced

`src/isalgraph/competitors/`: two ABCs and the `Encoding` value object, two lazy registries,
the sole `BitCount` producer, stdlib-only fixtures, six metrics, **eleven registered
backends**, four entry points (`smoke`, `grid`, `f5`, `reproduce`), and a `competitors` extra.

Verified at close: `registered_backends()` returns all eleven, `available_backends()` ten
(`size_null` filtered as a baseline), `unavailable_backends()` empty, six metrics.

| Gate | Result |
|---|---|
| Full suite | **2,106 passed / 321 skipped** (reference state 726/271, since raised by T-05) |
| `ruff check src/ tests/` | clean |
| `mypy --strict src/isalgraph/` | clean, 69 files |
| Criterion 1a, provenance | **all 5 datasets × 8 rows, delta 0.00e+00** — bit-for-bit |
| Criterion 1b, corrected table | emitted, 11 rows × 5 datasets, both views |
| Criterion 2, oracles | AGM 327/0; min-DFS V1 30 classes, V2 4,440/0, V3 = OEIS A001349 |
| Criterion 3, family identity | 7/7 fixtures + 300 random + all 8 boundary sizes |
| Criterion 7, the grid | runs; `padded_hamming × sparse6` **undefined**, as predicted |
| Criterion 9, min-DFS budget | **24/400 Mutagenicity, 0 elsewhere** — re-run by the orchestrator |
| Stop-condition 2, Picasso | `pynauty` **builds from source** under gcc 12.2.0, byte-identical |

---

## Interventions by the orchestrator

**Before the wave** (wave 0, three refuted premises — [WAVE0-FINDINGS.md](WAVE0-FINDINGS.md)):
the grakel off-by-one does not exist; the adjacency literals were row-major; README §4.1 is a
three-draw composite. All three PI-signed, design note amended with a dated changelog entry.

**A blocker the design missed**: a git worktree **cannot import its own `isalgraph`**. The
editable install's `ScikitBuildRedirectingFinder` sits in `sys.meta_path` and hard-maps the
package to the main checkout; a meta-path finder outranks `sys.path`, so neither `PYTHONPATH`
nor `sys.path.insert` overrides it. Both were measured and both failed. CONTRACTS §0.5 carries
the shim; it was step 0 for all three agents and all three applied it.

**Five defects in the orchestrator's own modules, all found by agents, all fixed by me:**

| # | Module | Defect | Found by |
|---|---|---|---|
| 1 | `bits.py` | `realised_bits` halved for `adjacency` and `agm_cam` — `8·⌈T/16⌉` for `8·⌈T/8⌉` | **A and B independently** |
| 2 | `isalgraph_ref.py` | passed `timeout_s` unconditionally, so the reference arm could not run on the Python engine | orchestrator, testing the shim |
| 3 | `registry.py` | reported "dependency not installed" when the backend *module* was missing | orchestrator, testing the shim |
| 4 | `smoke.py::_f3` | caught only `CompetitorError`, so min-DFS's `ValueError` on a disconnected graph would abort the whole run | C |
| 5 | `smoke.py::_f3` | fitted a `VectorBackend` one graph at a time — the per-batch trap at batch size one | C |

Defect 1 also exposed that **design criterion 4 was never satisfiable**:
`realised_bits < len(e.text)` contradicts the frozen formula, since packing `T` bits into
bytes costs ≥ `T` bits. It passed only because of the defect. Amended.

**Integration repairs** (3 commits, `fix(integration):`): both agents' xfail markers for
defect 1 came off; the criterion-4 assertions were rewritten; `min_dfs.is_available()` now
checks `networkx`, because "available" must mean "usable" — it had reported itself available
with `networkx` absent and then raised a bare `ImportError` from inside `encode`.

**One mid-flight relay**: track A's finding that
`convert_node_labels_to_integers(ordering="sorted")` does not pin the labelling was sent to
track B while it was still running, because B's criterion 7 compares AGM's identity
permutation against A's reading order.

---

## Findings the agents produced that change the plan

1. **`nauty.md` §1/§7's inversion guard is wrong on both halves** (B, re-derived by me).
   `nx.is_isomorphic(G, relabelled)` **can never** catch an inverted `canon_label` — any
   bijective relabelling is isomorphic by construction; it was `True` on 100 % of inverted
   cases. And the inverted labelling does **not** "pass F3": it is non-invariant on every
   connected trial. README §6's trap 2 is wrong, and the real situation is *better* than
   feared — the error is loud, and the prescribed guard is useless.
2. **`agm.md` §2.3's worked example is wrong** (B). `'E@ro'` unpacks to `000001110011110`,
   which **equals AGM's code**; the printed `001110010011100` is neither. The conclusion
   survives via a better artefact: nauty/AGM agreement is 38/60, 32/60, 16/60, 12/60, 1/60,
   **0/60** at `n = 5…10`.
3. **F3's Letter successes are a theorem, not a sampling artefact** (A, sharpened by me). Over
   every connected graph on `n = 2…6`, exactly 5 are invariant under *every* relabelling and
   **all 5 are complete**. F3 for the non-canonical formats measures the fraction of complete
   graphs in the draw.
4. **`convert_node_labels_to_integers(ordering="sorted")` does not pin the labelling** (A).
   290/300 disagreement with a sorted rebuild, measured independently. It also made `graph6`
   and `sparse6` serialise *different* labellings, which is what the scout's code did.
5. **E10 resolves on cost, without touching ρ** (C). `h = 2` vs `h = 5`: dimension ×4.9/17.8/
   24.1, 11–17 % of pair orderings flip — and `frac(d = 0)` is **identical**, so three extra
   refinements separate zero additional pairs. F5-blind by construction.

---

## Open, and escalated to the PI

1. **The corrected table moves finding 1's headline**: IsalGraph clears the size null on
   **one** of five datasets, not two. The margin that vanished (Letter MED, +0.007 in README)
   is an order of magnitude below the between-draw variability finding 14 already records
   (up to 0.07). The honest statement is that the "two of five" claim **is not robust to the
   draw**, which is sharper than either table and supports `statistics.md` D2.
2. **README §4.2's "min-DFS wins every column" does not survive** the single-draw equal-`n`
   table: `isalgraph_canonical` takes Letter LOW (0.9987 vs 0.9956) and WL takes AIDS
   (0.4332 vs 0.3993).
3. **sparse6's Claim A column is 6 bits high throughout** — §4.3 counted the `':'`, §4.2 froze
   excluding it.
4. **The five Suite-2 Claim A rows are not exactly reproducible.** They are medians over a
   400-graph draw whose source, the raw IAM GXL tree, is gone from this workstation. Coarse
   statistics survive (A: 10/10 on `adjacency`/`graph6`); finer ones do not (C: Protein 620.0
   vs 615.0; B: AIDS-IAM 80.25 % vs 82 %). The five **Suite-1** rows are full-cohort and
   remain exact.
5. **grakel cannot run on Picasso** (numpy-1 code against numpy 2.4.6). Not what the Picasso
   run gates; recorded in [PICASSO.md](PICASSO.md) and inherited by T-06.
6. **`SUITE1_ONLY` is enforced per graph, not per dataset** (B). `agm_cam` on `aids_iam` gives
   an 84.5 %-complete column with 31 typed refusals. Nothing silent, but criterion 9's intent
   was a dataset-level refusal, which would have to live in `smoke.py`/`grid.py`. **This is in
   tension with criterion 5**, which requires AGM to run on GREC and AIDS-IAM to measure its
   ceiling at all.

---

## What the decomposition got wrong

- **"Worktrees are safe here" was right about the C++ engine and wrong about the editable
  install.** The design checked the stated hazard and missed the one that actually bit. Cost:
  one preflight probe, and a shim in every prompt.
- **Two agents were given an unsatisfiable acceptance criterion** (criterion 4's
  `realised_bits < len(text)`), and both correctly refused to tune towards it — A carried it
  as a strict xfail, B as a plain one. That is the decomposition working, not failing.
- **The cross-edge cost B four skipped tests.** B could not exercise `sparse6_nauty` because
  A's module was not in its worktree. Unavoidable under disjoint ownership; the cells resolved
  at merge, and three of the four now run.
- **Agent branch names are not the auto-created ones.** B and C renamed to `track-B-canonical`
  / `track-C-mining`; merging the reported `worktree-agent-*` branch for B was a silent no-op
  that `git cat-file -e HEAD:...agm.py` caught. **Check the file landed, not the merge output.**
