# T-13 — complexity section: design note

**Ticket**: `tickets.md` T-13 — *Complexity section: `P(M)` recomputation, four costed operations,
three-way separation, the `|Aut(G)|` worst case.* Depends: none. P1.
**Answers**: R3.4b (is `P(M)` recomputed? cost the four operations), R3.4c (`n^{4.9}` vs `n^{9.0}` vs
"super-polynomial"), R3.7d (separate theory / worst case / empirical).
**Reading list**: `corrections.md` §5, `data.md` §4, `decisions.md` 17, `demands.md` R3.4b/c, R3.7d.

Base commit: `10eae30492982492cfc45db845b71c91a08e7883`.

---

## 1. State measured now, not assumed

Everything in this section was measured on 2026-08-26, not taken from the plan.

### 1.1 Environment

| Item | Plan said | Measured 2026-08-26 |
|---|---|---|
| Picasso C++ engine | `loginexa.yaml`: *"the C++ engine is absent, so `isalgraph.engine()` reads `python` here"* | 🔴 **STALE.** `engine() == 'cpp'`, `build_hash = 298fc1188bf1b051`, gcc 12.2.0, `x86-64-v3`, `.so` dated 2026-08-25 in `.../conda_envs/isalgraph/lib/python3.11/site-packages/isalgraph/core/`. T-06 built it. |
| Workstation engine | — | `cpp`, **`build_hash = 298fc1188bf1b051` — byte-identical to Picasso's.** Timings are comparable across the two machines modulo hardware. |
| `pynauty` | needed for `|Aut|` | **installed both sides, 2.8.8.1**; `autgrp` verified on `K_{1,3}` → 6.0 ✓ |
| `fscratch` quota | `CLAUDE.md`: 227.2k / 250.0k files | **235.1k / 250.0k soft, 400.0k hard.** ~15k headroom under soft. No new build tree is created by this ticket. |
| Cohort on Picasso | — | present: `.../datasets/isalgraph/{exported,exported_suite2}/`, 10 `.npz`, 16,370 graphs |
| Partitions | — | `cpu_partition` up, 7-day limit, 4 node families (`sd`/`bc`/`bl`/`sr`), ~3,500 idle cores |
| `|Aut|` helper | — | **already exists**: `competitors/backends/nauty.py:211 automorphism_group_size`, `:227 automorphism_orbits`, docstring says it was added *for T-13* |
| Operation counters | — | 🔴 **none exist anywhere** — not in `core/trace.py`, not in `viz/encoder_trace.py`, not in the native bindings. Only observables: `pairs_cache_size()`, `set_pairs_memo`, `set_branch_and_bound`. |
| `wl_subtree` on Picasso | — | grakel unusable (numpy 2). T-13 needs an **exact 1-WL partition refinement**, not a kernel — implemented self-contained, no numpy. |

### 1.2 `P(M)` — R3.4b's question has a definite answer, re-verified

**RECOMPUTED, once per frame, at every call site in the frozen Python reference.**
`graph_to_string.py:41 generate_pairs_sorted_by_sum` builds all `(2M+1)^2` pairs and sorts them —
`Θ(M² log M)` — and is called from `graph_to_string.py:155` (per greedy loop iteration),
`canonical.py:223` (per `_step` frame), `canonical_pruned.py:226` (per `_pruned_step` frame).
Nothing is memoised on the Python side. The C++ engine memoises per distinct `M`
(`native/include/isalgraph/pairs.hpp:12`), which is the single largest constant factor in the
implementation (25.5×–108.6× A/B at n = 6…10).

### 1.3 🔴 The plan's item 4 is refuted, and the replacement is stronger

`corrections.md` §5 item 4 and `decisions.md` §17 both assert that the incumbent triplet pruning key
is **"provably coarser than 1-WL"** and **"2.4–2.6× fewer classes, measured"**. Measured on a
25-graph draw from each of the ten datasets (250 graphs, `seed=13`):

| Claim | Measured |
|---|---|
| triplet key is **provably** coarser than 1-WL | **False in general.** Constructed witness — a *connected, 3-regular, 12-node* graph (3-prism spliced to `K_{3,3}`): stable 1-WL gives **1** class, the triplet key gives **4**. The two partitions are **incomparable**, not ordered. |
| the same, **empirically on real graphs** | Holds: WL refines the triplet key on **250/250**; the reverse holds on 124/250 (they coincide). Incomparable on 0/250. |
| **2.4–2.6× fewer classes** | **Not a cohort figure.** WL/triplet class ratio: **median 1.021**, min 1.000, max 3.200. The 2.4× came from one hand-picked graph (Mutagenicity/3703, 28 vs 66). |

And the finding that replaces them:

> **1-WL already attains the orbit partition on 250/250 graphs**, and the triplet key attains a
> median **97.9 %** of it (`WL/orbits` ratio: median 1.000, max 1.000; `triplet/orbits`: median
> 0.979, max 1.000; **zero** graphs where either partition is finer than the orbits — as
> Proposition 1 below requires).

This is the *characterised* worst case R3.7d asks for, and it is far stronger than the WL claim it
replaces: there is **no headroom for a finer invariant at all**. Decision 17's conclusion (do not
build `wl_pruned_canonical`) survives and is strengthened; only its stated rationale changes.

### 1.4 The cost law, and why an observational study is not enough

On the same 250 graphs, Spearman against `log t_pruned`:

| predictor | ρ (marginal) |
|---|---|
| `log n` | **+0.326** |
| `log m` | **+0.345** |
| density | −0.227 |
| `log \|Aut\|` | **+0.189** |

Marginally `|Aut|` looks *weakest*. But **within exact `(n, m)` strata** (13 strata, 84 graphs, each
with ≥ 4 graphs and varying `|Aut|`): median ρ(`log|Aut|`, `log t`) = **+0.655, positive in 12 of
13**. `|Aut|` is confounded with size in the IAM cohort, and the marginal correlation is the
confound, not the effect.

Banded, the step function is visible even at n = 250:

| `log10 \|Aut\|` | graphs | median `t_pruned` | max | median `n` |
|---|---:|---:|---:|---:|
| [0, 1) | 210 | 0.153 ms | 0.009 s | 9 |
| [1, 2) | 23 | 0.169 ms | 0.011 s | 18 |
| [2, 4) | 14 | **28.3 ms** | 5.015 s (censored) | 35 |
| [4, 8) | 3 | **5000 ms** | 5.006 s (censored) | 65 |

And the cleanest single contrast already in the draw:

| graph | `n` | `m` | `log10\|Aut\|` | `t_pruned` |
|---|---:|---:|---:|---:|
| `coil_del` | 58 | 161 | **0.00** | **8.9 ms** |
| `mutagenicity` | 77 | 81 | **5.09** | **censored at 5 s** |

Larger *and* denser encodes 560× faster when it is rigid.

**Conclusion for the design**: the real cohort cannot carry this claim on its own, because `n`, `m`,
density and `|Aut|` co-vary in it. **The primary evidence must be a controlled experiment on
constructed base graphs where one factor moves at a time.** The real cohort becomes the external
validity arm, not the proof.

---

## 2. The approach

### 2.1 Theory — the derivation (no compute)

Frame accounting for a greedy encode from a fixed start node: each payload instruction either
inserts a node (`n − 1` of them, one spanning tree) or adds a chord (`m − n + 1` of them), so a
greedy encode has **exactly `m` frames**, and
`|w| = m + Σ_f (|a_f| + |b_f|) = O(mn)`.

| Operation | Where | Per frame | Per encode |
|---|---|---|---|
| **pair generation** | `graph_to_string.py:41`, called `:155` / `canonical.py:223` / `canonical_pruned.py:226` | `Θ(M² log M)`, `M ≤ n` | `O(m n² log n)`; **memoisable to `Θ(n³ log n)` total** — the order depends on `M` alone, never on the graph |
| **pair scanning** | the `for a, b in pairs` loop, first-fit in cost order, early return at `canonical.py:267` | `O(M²)` worst, realised depth `D_f` measured | `O(m n²)` worst |
| **pointer walking** | `_move_pointer` `graph_to_string.py:321`, literal `for _ in range(\|steps\|)`; `_walk` `canonical.py:227` | `O(M)` per trial ⇒ `O(M³)` worst | **`O(m n³)` — the dominant worst-case term**, and incrementalisable to `O(m n²)` |
| **neighbour checks** | `_find_new_neighbor` `graph_to_string.py:331` (first match, `O(deg)`); canonical materialises all candidates `canonical.py:233`, `:276` | `O(Δ · D_f)` | `O(m Δ)` |
| **backtracking** | `_step` `canonical.py:202`, branch loop `:237`, in-place mutate/undo `:239–265` | branching `b_f = \|argmax τ over uninserted neighbours\|` | leaves `L ≤ n · Π_f b_f ≤ n · Δ^{n−1}` |

The `n ·` factor is the **start-node** union of Definition 2.6 — the branch Remark 2.7 omits (E13,
owner T-11). Total worst case: `O(n · Δ^{n−1} · m n³)`.

**Proposition 1 (invariance floor).** Let `κ` be any node invariant. For every `α ∈ Aut(G)` and
every `v`, `κ(v) = κ(α(v))`; hence the partition `V/κ` is a coarsening of the orbit partition
`V/Aut(G)`. *Proof:* an automorphism is an isomorphism `G → G`, so invariance applies directly. ∎

**Corollary 2 (irreducible branching).** Candidates lying in one orbit of the stabiliser of the
current partial configuration yield isomorphic continuations and therefore identical strings. No
invariant-based pruning key can separate them, so every one of them is expanded and all but one is
redundant. The redundant work is bounded below by the orbit sizes and is therefore governed by
`|Aut(G)|` — not by `n`, `m` or density, which enter only through the frame count.

**Corollary 3 (why the fix is automorphism detection, not a finer key).** Refining the pruning key
can only move the partition down to the orbit partition and no further. §1.3 measures that 1-WL
already sits *at* that floor on 250/250 real graphs. The remaining redundancy is therefore
irreducible without explicit automorphism detection — which is exactly what nauty, bliss and Traces
implement (individualisation–refinement with automorphism pruning). **This is deliverable 5's
justification, and it is now a theorem plus a measurement rather than a preference.**

### 2.2 The three-way separation (R3.7d)

| leg | object | statement |
|---|---|---|
| **(i) theoretical complexity** | the algorithm as *defined* | the table above; `S2G = Θ(\|w\|)`; `Levenshtein = Θ(\|w₁\|\|w₂\|)` |
| **(ii) worst-case search behaviour** | the search tree | Prop. 1 / Cor. 2: governed by `\|Aut(G)\|`. Characterised, predictable from a millisecond computation, and measured as a near-step function |
| **(iii) empirical runtime scaling** | this implementation on this cohort | `T ~ n^α`, fitted with CIs. **This is a property of the cohort, not of the algorithm** — it measures how `\|Aut\|` happens to co-vary with `n` in IAM data. Reporting it as complexity is precisely the conflation R3.7d objects to, and is why `n^{4.9}` and `n^{9.0}` could coexist. |

### 2.3 The controlled experiment — CE, the primary evidence

Constructed base graphs, one factor moved at a time. All families are standard textbook graphs with
**closed-form `|Aut|`**, verified against `pynauty` at build time (a family whose measured `|Aut|`
disagrees with its formula is a bug and aborts the run).

| family | `n` | `m` | `\|Aut\|` | role |
|---|---|---|---|---|
| `path` `P_n` | n | n−1 | 2 | rigid-ish tree |
| `cycle` `C_n` | n | n | 2n | low symmetry, fixed density |
| `star` `K_{1,n−1}` | n | n−1 | (n−1)! | **extreme `\|Aut\|` at minimum density** — the confound-breaker |
| `complete` `K_n` | n | n(n−1)/2 | n! | extreme `\|Aut\|` at maximum density |
| `complete_bipartite` `K_{a,a}` | 2a | a² | 2(a!)² | high `\|Aut\|`, medium density |
| `hypercube` `Q_d` | 2^d | d·2^{d−1} | 2^d·d! | high `\|Aut\|`, regular |
| `prism` `C_a × K_2` | 2a | 3a | 4a | low `\|Aut\|`, cubic |
| `caterpillar` | n | n−1 | 2^k | tunable `\|Aut\|` at tree density |
| `rigid_er` `G(n,p)` | n | ~p·C(n,2) | **1 a.s.** | the rigid control at any density |
| `symmetry_ladder` | **fixed (n,m)** | | tuned | **the matched design**: a base graph plus `k` symmetry-breaking edge swaps that hold `(n, m)` exactly and drive `\|Aut\|` down monotonically |

**Factorial**: family × `n ∈ {8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64}` × 5 replicates for the
random families, 1 for the deterministic ones. **Every cell is run for every representation.**

**Representations** — all thirteen registered arms, so the law is shown for canonical forms *as a
class* and the search-free serialisations act as the null:

- search-based canonical forms: `isalgraph_exhaustive`, `isalgraph_pruned`, `isalgraph_greedy`,
  `nauty_graph6`, `sparse6_nauty`, `min_dfs`, `agm_cam`
- search-free serialisations (the control — cost must be `Θ(n²)` or `Θ(n + m)` and **flat in
  `|Aut|`**): `adjacency`, `graph6`, `sparse6`, `wl_subtree`, `size_null`

**Engine-ablation arm** (leg (i) vs leg (iii), on a stratified subsample only):
`set_pairs_memo ∈ {on, off}` × `set_branch_and_bound ∈ {on, off}` — a 2×2 that prices the two
implementation optimisations separately from the algorithm.

### 2.4 The operation counters — OC

No counters exist, so T-13 builds an **instrumented mirror** of the frozen reference — the same
device T-09 used (`viz/encoder_trace.py`, validated on 134,609 (graph, start) pairs, `core/`
untouched). It counts frames, pair trials, cumulative scan depth, pointer steps, neighbour checks,
backtrack nodes and search-tree leaves, and **must emit byte-identical strings** to
`core.canonical` / `core.canonical_pruned` / `core.graph_to_string`. Counters validate the §2.1
derivation on the CE families; timings never come from it.

### 2.5 The real-cohort arm — RV

`|Aut|`, orbit count, triplet/1-WL/orbit partitions and the refinement tests, joined to controlled
re-timings, over all 16,370 graphs. This is external validity for CE, plus the material for §1.3's
corrected claim. T-06's per-graph `seconds` arrays
(`data/source/T06{,_exhaustive}/encodings/{suite1,suite2}/*.npz`) are a **consistency check only**,
never primary: they were produced under 5-way local concurrency and T-06 itself retracted its
derived rate figures as unprovenanced.

### 2.6 Rejected alternatives

| Rejected | Why |
|---|---|
| Observational study on the real cohort alone | `n`, `m`, density and `\|Aut\|` co-vary; the marginal ρ for `\|Aut\|` (+0.189) is *lower* than for `n`, and only the within-`(n,m)` contrast (+0.655) recovers the effect. A reviewer would call this correlational, correctly. |
| Reuse T-06's per-graph timings as primary | Produced under uncontrolled concurrency; T-06 retracted its own derived rates for exactly this reason. |
| Build `wl_pruned_canonical` to make the WL point | Decision 17, signed. And §1.3 now shows it would buy a median 2 % class-count improvement over a key already at 98 % of the invariance ceiling. |
| Instrument `core/` directly | The Python reference is frozen; changing it means re-proving C++ parity. |
| Report `T ~ n^α` as the complexity result | That *is* the R3.4c defect. It is leg (iii) and is labelled as such. |

---

## 3. Frozen before any run

1. **Budget** `T_max = 300 s` per (graph, representation, arm), matching D14 and T-06. Enforced by a
   killed subprocess, **not** `SIGALRM` (T-05 finding 5: `SIGALRM` does not interrupt the C++
   engine). Censored records carry `status = "censored"` with `seconds = T_max` and are analysed as
   **right-censored**, never dropped.
2. **Timing**: `time.process_time`, one warm-up then median of **3** repeats when the warm-up is
   `< 1 s`, otherwise **1** run (relative noise is negligible once the run is seconds long). One
   shard per dedicated core via `taskset`, whole-node `--exclusive` allocation, single-threaded
   engine (`ISALGRAPH_THREADS=1`).
3. **Engine assertion**: every worker asserts `isalgraph.engine() == "cpp"` and records
   `build_info()` in its output header. A shard whose `build_hash` differs from
   `298fc1188bf1b051` is discarded.
4. **`|Aut|` is reported as `log10|Aut|`**, computed as `log10(mantissa) + exponent` from
   `pynauty.autgrp`, never as a float product (it overflows above ~1e308).
5. **The refinement test is exact partition containment**, never a class-count comparison. Class
   counts are reported alongside but may not be used to assert subsumption.
6. **Family verification**: each constructed graph's measured `|Aut|` must equal its closed-form
   value; a mismatch aborts the run rather than being recorded.
7. **Primary analysis rule** (fixed before results exist): the `|Aut|` law is established by the
   **`symmetry_ladder` within-`(n, m)` contrast** — Spearman ρ(`log|Aut|`, `log t`) per
   `(family, n)` cell, sign-tested across cells. The marginal regression on `n`, `m`, density and
   `log|Aut|` is secondary and reported with its confounding stated. The real-cohort arm is
   external validity and is labelled descriptive.
8. **Supersession**: where a graph appears in both T-06's records and T-13's, T-13's controlled
   measurement is primary and the ratio distribution to T-06's is printed.
9. **Nothing in `src/isalgraph/` is modified.** All new code lives under
   `benchmarks/real_data/eval_t13_complexity/` with a `benchmarks/eval_t13_complexity` symlink.
   This is also what makes worktree isolation safe for this ticket (the editable-install finder
   would otherwise shadow a worktree's `src/`).

---

## 4. Acceptance criteria

| # | Criterion | Proof |
|---|---|---|
| 1 | `P(M)` recomputation is stated with all three call sites and the `Θ(M² log M)` cost | §2.1 table; `grep -n generate_pairs_sorted_by_sum src/isalgraph/core/*.py` returns exactly `graph_to_string.py:41,155`, `canonical.py:223`, `canonical_pruned.py:226` |
| 2 | Four operations costed, each with a per-frame and per-encode bound | §2.1 table |
| 3 | Counters reproduce the derived bounds on the CE families, and the instrumented mirror is **byte-identical** to the frozen reference | `pytest benchmarks/real_data/eval_t13_complexity/tests/ -q`, parity on ≥ 50,000 (graph, start) pairs, 0 mismatches |
| 4 | Prop. 1 / Cor. 2 / Cor. 3 stated and proved; no graph in any arm has an invariant partition finer than its orbit partition | `analysis.py` gate `G1`, 0 violations over all measured graphs |
| 5 | `|Aut|` law demonstrated **within fixed `(n, m)`** across the symmetry ladder, for every search-based representation, with the search-free controls flat | `figures/t13_cost_law.pdf`, `tables/t13_ladder.tex`, sign test per rule 7 |
| 6 | Three-way separation written out, with leg (iii)'s exponents re-fitted with bootstrap CIs and labelled a cohort property | `tables/t13_exponents.tex` |
| 7 | `corrections.md` §5 item 4 and `decisions.md` §17 corrected on the record | `review-close` propagation |
| 8 | Self-contained report at `/media/.../results/reports/T-13-complexity/` with `REPORT.md`, `T-13-FRAMING.md`, `PROVENANCE.md`, `data/`, `figures/`, `tables/` | directory listing + every number traceable to a file in `data/` |
| 9 | Full suite green, at or above **2,618 passed / 321 skipped** | `$PY -m pytest tests/ -q` |

---

## 5. Stop and ask

- Any measurement contradicting Prop. 1 (an invariant partition finer than the orbit partition) —
  that would be a bug in `pynauty`, in the partition code, or in the proposition. Halt.
- If the `symmetry_ladder` contrast comes out **null** — i.e. `|Aut|` does *not* govern cost under
  control — the ticket's deliverable 4 is refuted and the manuscript must say "exponential" after
  all. Halt and escalate; do not soften.
- If `nauty_graph6` shows **no** `|Aut|` degradation, the "property of canonical forms as a class"
  framing is wrong and must become "a property of canonical forms without automorphism pruning".
- Compute above ~5,000 core-hours. Current estimate: **≈ 400–700 core-h** (see §6).

---

## 6. Compute

Measured rates from the 250-graph probe: median `t_pruned` **0.167 ms**, tail censored at the
budget. The cost is entirely in the censored tail, which is bounded by construction:
`(#cells × #representations × T_max)`. CE is ≈ 700 graphs × 13 representations, worst case
`700 × 13 × 300 s = 758 core-h` if everything censored; realistically the search-free arms are
microseconds and only the high-`|Aut|` cells of the search-based arms censor. **Estimate 400–700
core-h**, one whole-node array on `cpu_partition`, ≥ 2 h per task (SCBI floor).
