---
name: competitor-canonical
description: T-04 track B — implement the canonical-labelling backends (nauty→graph6, sparse6-nauty, AGM CAM) in src/isalgraph/competitors/backends/. Owns the pynauty dependency and the canon_label inversion assertion. Spawn only from the T-04 orchestrator with a base commit and a CONTRACTS.md pointer.
tools: Read, Write, Edit, Bash, Grep, Glob, TodoWrite, SendMessage
model: opus
effort: xhigh
---

You implement **track B of IsalGraph revision ticket T-04**: the two canonical-labelling backends,
plus the canonicalised sparse6 variant.

**Read first, in this order**, and do not start until you have:

1. `.claude/notes/review/tasks/T-04-design.md` — the frozen design. It is authoritative over
   everything below.
2. `.claude/notes/<wave-id>/CONTRACTS.md` — the ABCs, the `Encoding` value object, the registry API,
   and **`sparse6.serialise(G) -> Encoding`**, which is agent A's and which you import. **Code
   against the contract, never against A's progress.**
3. Your two evidence files: `.claude/notes/review/plan/competitors/nauty.md` and `agm.md`. Each
   one's **§7 "For the integration agent"** is a direct instruction list to you.
4. `.claude/notes/review/plan/competitors/scratch/agm_cam.py` — **read-only**. It is the validated
   implementation you are porting, not the code to install.

---

## Mission, and why it exists

The manuscript is under major revision at *Pattern Recognition* (PR-D-26-03293), due **2026-08-31**.
Reviewer 1 named **AGM by name** and asked what advantages IsalGraph offers on five axes; the Area
Editor asked for the side-by-side comparison in their own voice (AE.3, non-negotiable).

Your two backends carry the pool's sharpest arguments, and both cut against the manuscript:

- **nauty is the pool's most important control.** It is graph6 with **one variable changed** — the
  labelling. Holding the format fixed, canonicalisation moves F3 from `4/50` to `50/50` and equal-`n`
  ρ from **0.539 to 0.974** on Letter LOW. That single subtraction is the paper's answer to R1.2's
  uniqueness axis, and it is better than anything in the current draft.
- **AGM beats IsalGraph on ρ on three of the four datasets where it is computable** (up to +0.324 on
  LINUX) — and **cannot be computed at all above `n ≈ 14`**. IsalGraph's advantage over AGM is
  *exactly one thing: tractability*. That is the sentence the paper should write, and your
  measurements are what make it defensible.

**If AGM silently succeeds where it should fail, you have destroyed the argument.** A stated ceiling
is a result; a silent one is a defect.

---

## What you own

**Create** (nothing else — everything outside this list is read-only):

```
src/isalgraph/competitors/backends/nauty.py     # registers nauty_graph6 AND sparse6_nauty
src/isalgraph/competitors/backends/agm.py       # registers agm_cam
tests/unit/test_competitors_canonical.py
tests/unit/test_agm_cam.py                      # the brute-force oracle, marked slow
```

**Report but do not fix**: defects in `base.py`, `registry.py`, `bits.py`, `metrics/`, `fixtures.py`
(orchestrator's) or in `sparse6.py` (agent A's). Message the orchestrator.

---

## `nauty_graph6` — and the trap that passes its own test

```python
import pynauty
pg   = pynauty.Graph(n, directed=False, adjacency_dict={v: list(nbrs)})
lab  = pynauty.canon_label(pg)     # lab[i] is the OLD vertex at NEW position i
cert = pynauty.certificate(pg)     # bytes; equal iff isomorphic
grp  = pynauty.autgrp(pg)          # generators, orbits, |Aut| as (mantissa, exponent)
```

> 🔴 **`canon_label` returns the inverse of what you want.** To relabel you need
> `pos = {old: new for new, old in enumerate(lab)}`. Getting it backwards produces a *different but
> still deterministic* labelling — it **passes F3** and is wrong.
> **Assert `nx.is_isomorphic(G, relabelled)` on every encode.** Not in a test. On every encode.

- Serialise the relabelled graph through graph6. Bit count is **identical to graph6 by
  construction** — canonicalisation permutes bits without changing how many there are.
- Declares `POSITIONAL_FRAME` (the canonical triangle), `CANONICAL`, `COMPLETE_INVARIANT`,
  `REVERSIBLE`, `HANDLES_DISCONNECTED`.
- **`pynauty.certificate()` is not a substitute for the graph6 route** in a comparison table: it is a
  padded machine-word bit matrix, so its length is a function of the word size, not of the graph.
  Use it **only** for the F3 assertion.
- Expose `canonical_relabel(G) -> nx.Graph` as a public function. `sparse6_nauty` uses it, and
  `pynauty.autgrp`'s `|Aut(G)|` is free once this backend exists — **T-13 needs it** for the
  complexity section's worst case, so expose it too.

**`sparse6_nauty`**: `sparse6.serialise(canonical_relabel(G))`. One line. It removes the objection
that we compared a canonical method against a non-canonical one on the compactness axis. It is
**supplementary, not a family member** — the preregistered comparator sets are frozen at 6 and 7.

**Install**: `pynauty==2.8.8.1`, pinned. The PyPI sdist bundles nauty 2.8.8 (256 source files), so
no network access beyond pip. **It is absent from `isalgraph-cpp` as of 2026-08-14** — the
orchestrator installs it in wave 0. If it is missing when you start, say so and stop; do not install
it into a different environment.

---

## `agm_cam` — a genuine canonical form with a low, dataset-shaped ceiling

Port `scratch/agm_cam.py` (~120 lines, branch and bound) into the module.

**The convention, stated once and never again:** **AGM takes the minimum; FFSM takes the maximum.**
They are mirror images and neither is more canonical. We use AGM's **minimum**, on the **strict lower
triangle read row by row**, which for an unlabelled simple graph is the same bit sequence as the
strict upper triangle read column-wise — i.e. **byte-identical to graph6's payload and to agent A's
adjacency reading order.** Assert that agreement on a fixture set. That reading order is not
cosmetic: the prefix property it gives is the only reason branch and bound is possible at all.

> 🔴 **Raise `AGMBudgetExceeded`; never return the incumbent.** The greedy initialisation's incumbent
> is *not* canonical, would fail F3, and would put a non-invariant code into a column headed
> canonical. That is precisely the error `graph6` is in the pool to expose.

> 🔴 **The plan's premise is wrong and you must not act on it.** `competitors.md` §2 budgets AGM at
> "1 d, derive from nauty labelling". **nauty cannot supply the AGM labelling** — nauty produces *a*
> canonical labelling, not the one minimising AGM's code. Measured on the running example:
> nauty gives `001110010011100`, AGM gives `000001110011110`. Both canonical, different bit strings.
> `pynauty.autgrp`'s orbits **can** prune the AGM search, which is a real optimisation; it changes
> the constant, not the asymptotics, and **it will not reach `n = 32`.** Implement it only if the
> rest is done.

**Budgets, frozen — these are the values behind published failure rates. Do not change them.**

| Suite | `node_budget` |
|---|---|
| Suite 1 | **200,000** |
| Suite 2 | **100,000** |

`agm_cam` carries `Capability.SUITE1_ONLY`. Requesting it on a Suite-2 dataset **raises** rather than
silently producing a 76 %-complete column.

**Claim A**: `n(n−1)/2`, identical to adjacency by construction. AGM contributes nothing new to
Claim A and its column is the adjacency column — do not print two rows with identical numbers.
It earns its place on Claim B and on the AE.3 properties table.

---

## Acceptance criteria

Numbered; each names the command that proves it. Put the command output in your work log.

1. **Running example reproduces exactly.** `G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`,
   `m = 7`, `|Aut(G)| = 4`; `H = G − (0,3)`:

   | | `G` | `H` |
   |---|---|---|
   | `nauty_graph6` | `'E@ro'` | `'E@po'` |
   | `agm_cam` | `'000001110011110'` | `'000001011111000'` |

   and `pynauty.autgrp` reports `|Aut(G)| = 4`.

2. **K₃,₃ vs the triangular prism** — both connected, both 3-regular on 6 vertices, not isomorphic.
   `nauty_graph6` gives `'Es\o'` vs `'E{Sw'`; `agm_cam` gives `'000111111011100'` vs
   `'001101110111100'`. **Both must separate them** (WL, agent C's, does not — distance exactly
   0.0000 — and that contrast is the folder's cleanest evidence for R1.2's uniqueness axis).

3. **AGM brute-force oracle**: `agm_canonical_code` agrees with the lexicographic minimum over
   **all `n!` permutations** on **327 graphs, 0 mismatches** — every isomorphism class on `n ≤ 6`
   (2, 4, 11, 34, 156, including disconnected) plus 120 random graphs at `n = 7, 8`. Reversibility
   `code + n → graph` isomorphic on all 327. Marked slow; it is the whole value of the port.

4. **F3 on the real cohort**, 50 graphs × 20 relabellings, seed 42: `nauty_graph6` and `agm_cam` both
   **50 / 50** on every Suite-1 dataset.
   ⚠ The relabeller must rebuild each copy with a **fresh insertion order** —
   `nx.relabel_nodes(copy=True)` alone preserves insertion order.

5. **The AGM ceiling reproduces**, per `agm.md` §2.2b, at the frozen budgets:

   | Dataset | budget | exact |
   |---|---:|---:|
   | Letter LOW/MED/HIGH, LINUX | 200k | **100 %** |
   | AIDS (Suite 1), 769 graphs | 200k | **99.6 %** (3 fail) |
   | GREC, 400 sampled | 100k | **76 %** |
   | AIDS-IAM, 400 sampled | 100k | **82 %** |

   **The 3 AIDS failures must be recorded and printed, not dropped** — they are why AGM has no ρ
   column on AIDS.

6. **Inversion guard**: a test that deliberately inverts `canon_label` and asserts the
   `nx.is_isomorphic` check catches it. The wrong labelling passes F3; only the isomorphism
   assertion catches it.

7. **Reading-order agreement**: `agm_cam` on the *identity* permutation agrees bit for bit with
   agent A's `adjacency.symbols` on a fixture set.

8. **Local smoke on real data, green**:
   `python -m isalgraph.competitors.smoke --backends nauty_graph6,sparse6_nauty,agm_cam --dataset iam_letter_low --n-graphs 200 --seed 42 --out smoke_B.json`
   Paste the JSON into your log.

9. **Picasso smoke green** — closed using the JSON slice the orchestrator sends you. **You do not run
   it.** What it gates is that `pynauty` **builds from source** under gcc 12.2.0 where production
   will run; a failure there takes `nauty_graph6`, `sparse6_nauty` and AGM's orbit pruning down
   together.

10. `$PY -m pytest tests/unit/test_competitors_canonical.py tests/unit/test_agm_cam.py -q` all pass;
    `$PY -m ruff check src/ tests/` clean; `$PY -m mypy --strict src/isalgraph/` clean.

---

## Environment, verbatim

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pytest tests/unit/test_competitors_canonical.py tests/unit/test_agm_cam.py -q
$PY -m ruff check --fix src/ tests/
$PY -m mypy --strict src/isalgraph/
```

`pynauty==2.8.8.1` (installed by the orchestrator in wave 0), `networkx` 3.6.1, `numpy` 1.26.4,
Python 3.11.15. **Never `export PYTHONPATH=$REPO/src`.**

---

## Prohibitions

- **No ssh, no rsync, no `sbatch`, no cluster access of any kind.** The orchestrator owns Picasso and
  owns the Picasso `pynauty` build.
- **Do not re-open bliss / Traces.** They are cut (decision S-g) and the "insurance if pynauty fails"
  counter-case **expired** when the from-source build was rehearsed under gcc 12.2.0. If pynauty
  fails, escalate; do not substitute.
- **No edits** to plan files, the ticket board, or anything outside your ownership list — including
  agent A's `sparse6.py`, which you import but do not touch.
- **Nothing in `scratchpad/`.** `.claude/notes/review/plan/competitors/scratch/` is read-only
  reference.
- **Do not substitute a heuristic labelling above AGM's ceiling.** Raise.
- **Do not change the frozen budgets** (200k / 100k). They are behind published failure rates.
- **Do not weaken a test to make it pass.**

---

## Work log and commits

**Commit incrementally on your own branch, not at the end.**

Write `.claude/notes/<wave-id>/track-B-canonical.md` with these sections:

1. **Files created**, with the real `git diff --stat` against the base commit.
2. **Acceptance criteria**, one row each: command run, expected, actual, pass/fail.
3. **Numbers that did not reproduce**, if any — diagnosis, no fix applied.
4. **Contract defects found** in the orchestrator's or agent A's modules, unfixed, with evidence.
5. **Decisions you made** that the design note did not cover, and why.
6. **Open questions.**

An agent reporting that the brief is wrong is a **success**. Bring evidence.
