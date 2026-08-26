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

This is the *characterised* worst case R3.7d asks for. ~~There is **no headroom for a finer
invariant at all**.~~ **See §1.3a — that sentence was drawn from a 250-graph probe and does not
survive the full cohort.** Decision 17's conclusion (do not build `wl_pruned_canonical`) survives;
its stated rationale changes, and the honest version is narrower than the probe suggested.

### 1.3a Corrected at cohort scale, 2026-08-26 — the "no headroom" claim was overstated

The 250-graph probe measured `WL/orbits = 1.000` on every graph and `triplet/orbits` at a median
0.979, which reads as "both partitions are already at the invariance ceiling". Re-run over the
**whole locked cohort, 16,370 graphs**, that splits in two and only one half holds:

| | 250-graph probe | **16,370-graph cohort** |
|---|---:|---:|
| 1-WL **==** orbit partition | 250/250 (100 %) | **16,360/16,370 (99.939 %)** |
| triplet key **==** orbit partition | ~ all | **6,854/16,370 (41.869 %)** |
| triplet resolution / ceiling | median 0.979 | **median 0.9130, mean 0.8571** |
| WL/triplet class ratio | median 1.021 | **median 1.0952, mean 1.2379, max 7.333** |
| Proposition 1 violations | 0 | **0** |
| incomparable | 0 | **0** |

**What survives, and it is the load-bearing half:** 1-WL attains the invariance ceiling on
**99.94 %** of the cohort, so *no invariant finer than 1-WL can help anywhere that matters*. Cor. 3
stands.

**What does not survive:** the incumbent **triplet key is not at the ceiling** — it agrees with the
orbit partition on only 41.9 % of graphs, and the shortfall is concentrated exactly where
canonicalisation is expensive:

| dataset | triplet == orbits | |
|---|---:|---|
| `iam_letter_low` | 100.00 % | small, sparse, cheap |
| `iam_letter_high` | 83.97 % | |
| `aids_iam` | 50.58 % | |
| `mutagenicity` | **14.50 %** | **the dataset T-06 measured 2.50 % censoring on** |
| `coil_del` | **10.33 %** | |
| `protein` | **10.54 %** | |

So there **is** headroom between the incumbent key and the ceiling, it is large on the hard
datasets, and 1-WL would capture essentially all of it. That partially rehabilitates the intuition
behind the plan's item 4 while leaving both of its stated claims refuted: the relation is
**incomparable in general** (§1.3's connected 3-regular witness), and the magnitude is a median
**1.0952×**, not 2.4–2.6×.

**Consequence for the ticket.** Whether that class-count headroom converts into *time* is not a
partition question — it is the cost law, and it is what the campaign measures. Decision 17 is not
reopened: T-16 was rejected on scope grounds (a new canonicalisation algorithm shipped during a
revision round whose opening comment questions the contribution's substance), and that argument is
untouched by this measurement. **Do not restate decision 17's rationale as "WL would buy nothing".**

### 1.3b 🔴 The cohort loader and the locked cohort are not the same ten datasets

`competitors.datasets.ALL_DATASETS` enumerates ten names summing to **16,320** graphs.
T-01's locked cohort is **16,370**. The difference is exact and identified:

- the loader's ten include Suite-1 **`aids` (769)** and **exclude `aids_graphedx`**;
- `aids_graphedx.npz` ships in `exported_suite2/` with **819** graphs and is **not reachable through
  the loader at all** — `datasets.load("aids_graphedx")` raises `DatasetNotFoundError`, because the
  name is absent from `SUITE2`;
- `16,320 − 769 + 819 = 16,370`, and swapping them reproduces the locked count **to the graph**.

Both figures above were therefore computed twice, and the table in §1.3a is the **locked ten**
(`aids_graphedx` in, Suite-1 `aids` out). The loader's ten give 16,310/16,320 and 6,848/16,320 —
the same conclusion.

**This is not T-13's to fix, and T-13 does not fix it**, but anything that iterates
`ALL_DATASETS` and calls the result "the cohort" is covering 16,320 graphs across a *different*
ten than the plan's. Owner: whoever next touches `competitors/datasets.py`. Carried to
`review-close`.

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

### 6.1 Amendment, 2026-08-26 — the RV arm does not re-time the whole cohort

**Frozen before any campaign runs.** Inspecting T-06's per-graph arrays
(`data/source/T06{,_exhaustive}/encodings/{suite1,suite2}/*.npz`) shows they already carry
`graph_ids, node_counts, edge_counts, length, status, fallback_used, seconds` for every
(graph, representation). Measured on Mutagenicity (4,040 graphs):

| arm | median `seconds` | max | censored |
|---|---:|---:|---:|
| `isalgraph_exhaustive` | 0.400 s | **31.3 s** | 888 (22.0 %) |
| `isalgraph_pruned` | 0.000771 s | **300.0 s** | 101 (**2.50 %** — reproduces T-06's headline) |

So T-06's exhaustive arm ran at a **≈ 30 s** budget and its pruned arm at **300 s**. Re-timing the
full cohort at a uniform 300 s would cost ≈ 74 core-h on Mutagenicity's exhaustive arm alone, to
reproduce numbers that already exist.

**Revised RV arm, in three parts:**

1. **Symmetry columns over all 16,370 graphs** — `log10|Aut|`, orbits, the triplet / 1-WL / orbit
   partitions and both refinement tests — joined onto T-06's existing per-graph records by
   `(dataset, graph_id)`. Cheap (`pynauty` plus pure-Python refinement). This is what generalises
   T-06's Mutagenicity-only `|Aut|` step function to all ten datasets.
2. **A controlled re-timing of a stratified subsample** (~2,000 graphs, strata crossed on
   `n`-band × `log10|Aut|`-band), run under the §3 timing rule on an exclusive node. Its purpose is
   to **bound the bias** in T-06's timings, which were produced under 5-way local concurrency: the
   ratio distribution `t_T13 / t_T06` is reported per stratum. A bias that is flat across `|Aut|`
   bands leaves T-06's records usable as descriptive support; one that is not, retires them.
3. **T-06's records are descriptive support, never primary**, and every figure drawn from them says
   so. The primary evidence remains CE.

**Revised estimate: 150–400 core-h.** Unchanged conclusion: no escalation needed (threshold 5,000).

### 6.2 Amendment, 2026-08-26 — which exhaustive arm, and the budget object

Two defects in CONTRACTS §5.2, both found by track-C and verified by the orchestrator in the source.

**1. `isalgraph_exhaustive`, not `isalgraph_canonical`, is the exhaustive arm above `n = 12`.**
`isalgraph_canonical` carries `Capability.SUITE1_ONLY` (`isalgraph_ref.py:267`) and
`_check_scope` raises `SuiteScopeError` above `n = 12` (`:163`), so the arm originally designated
primary cannot report a timing on most of the constructed grid. The two arms share one `encode`
path (`:196–212`) with no `except` and no reference to `fallback_variant` — **the D14 fallback is
declarative and is performed by the campaign driver, not by the backend** — so inside `measure.py`
they are behaviourally identical and differ only in that guard. The guard encodes a *cohort policy*
derived from T-04's finding that `canonical_string` costs 342 ms/graph with 12/400 timeouts on
Suite-2 AIDS; T-13 exists to characterise exactly that, so a guard encoding the conclusion may not
censor the experiment establishing it. Using `isalgraph_exhaustive` also keeps T-13 joinable to
T-06's Suite-2 records, which the RV arm needs.
**Consistency gate**: at `n ≤ 12` both arms run and must agree on `status` and `length_chars`.

Consequence for the record: `fallback_used` is **never `true`** under `measure.py`, because the
runner does not implement D14. It is `null` where no fallback is declared and `false` otherwise, and
a future `true` means someone added a substitution.

**2. One fully-populated `Budget` is threaded through every backend.** `base.py:180–194` documents
it as heterogeneous — *"only the fields a backend declares are read"*. Passing `budget=None` to the
non-IsalGraph arms would leave `max_projections` unset, and `min_dfs.py:372` reads
`cap = MAX_PROJECTIONS if budget is None else budget.max_projections`, so an incompletely
populated budget makes min-DFS **unbounded** and re-opens the OOM that killed the first Suite-2 run.
`max_projections = 50,000` is a **memory** cap, not a speed knob, and must stay set.
The resolved budget is serialised into the record as `budget_spec`: T-27 established for this
project that a method name without its options string is not a specification, and a censoring rate
is meaningless without the caps that produced it. Min-DFS truncation at the projection cap is
recorded `status="censored"`, `error_kind="max_projections"` — a different mechanism from a
wall-clock kill, and the analysis may not pool them.

### 6.3 Amendment, 2026-08-26 — the pilot sharpens Corollary 2 into a two-parameter statement

**Recorded before the campaign. Orchestrator's own pilot** on track-A's `spider_ladder` at `n = 33`
(a tree: `m = 32`, 8 legs of length 4, `n`, `m` **and the degree sequence** fixed across all four
rungs; workstation, `engine() == "cpp"`, 20 s budget). Times in ms:

| rung | `log10\|Aut\|` | `isalgraph_exhaustive` | `isalgraph_pruned` | `isalgraph_greedy` | `min_dfs` | `graph6` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.606 | 476.7 | **380.7** | 0.5 | **4471.7** | 0.4 |
| 1 | 2.857 | 487.3 | **74.1** | 0.5 | 75.9 | 0.4 |
| 2 | 1.380 | 515.1 | **14.7** | 0.5 | 10.9 | 0.4 |
| 3 | 0.301 | 468.2 | **3.0** | 0.5 | 8.6 | 0.4 |

**The two search arms separate the two governing parameters, and the ladder is what makes them
separable.**

- **The unpruned arm is flat**: 1.10× across a 4.3-order range in `|Aut|`. Mechanistically right —
  the unpruned search expands every branch, so its leaf count is `≤ n · Π_f b_f` where the `b_f`
  are neighbour-choice branching factors. Those are determined by the **degree sequence**, which
  the ladder holds fixed. Symmetry is invisible to it because it prunes nothing.
- **The pruned arm falls 127×** and is monotone in `|Aut|` on all four rungs. Also mechanistically
  right, and it is Corollary 2 measured: the triplet key removes every branch it can distinguish,
  and what it cannot distinguish is exactly the orbits, so the residual work is orbit redundancy.

~~So the paper's sentence becomes two named parameters rather than one hedge:~~

> ~~**The unpruned canonical search's cost is governed by the degree sequence; the pruned search's
> cost is governed by the automorphism group.** Measured on a ladder holding `n`, `m` and the degree
> sequence fixed, the unpruned arm varies by 1.10× across 4.3 orders of magnitude in `|Aut|` while
> the pruned arm varies by 127×.~~

> ## 🔴 RETRACTED 2026-08-26 by the campaign (array 2108126). Do not use this sentence.
>
> It was drafted from a **20 s pilot on one ladder** and generalised a cell into a law. At the
> frozen 300 s budget over all 21 cells, the unpruned arm's flatness is a property of **sparse**
> ladders, not of the unpruned algorithm:
>
> | ladder | `\|Aut\|` span | exhaustive fold |
> |---|---:|---:|
> | spider (tree) n = 31 / 33 / 65 | 3.7 / 4.3 / 4.6 | **1.11× / 1.14× / 1.09×** |
> | complete bipartite n = 8 / 10 / 12 / 14 | 2.5 / 4.5 / 6.0 / 7.7 | 1.76× / 3.58× / 6.96× / **14.78×** |
>
> and its rule-7 outcome is **ρ = +0.300, 7 of 9 cells, p = 0.18 — not significant**, on a sample
> **57.6 % censored**. The pilot cell reproduced exactly (1.14×); the generalisation did not.
>
> **What survives, and it is still Corollary 2.** Pruning does not make easy graphs faster — on the
> 56 graphs where both arms complete the median ratio is **1.00×**. It converts *censored* graphs
> into *completed* ones: **56 → 73 completions (+30 %)**, with the pruned arm's per-ladder dynamic
> range reaching **46,170×** against the exhaustive arm's 14.8×. Pruning removes what the invariant
> can discriminate, so what remains is automorphic redundancy and the `|Aut|` dependence gets
> **sharper**.
>
> **The confirmatory result is `isalgraph_pruned`: ρ = +0.892, 11 of 12 cells, p = 0.0064**, with
> the five search-free arms flat at fold 1.0–1.1× as the null.
>
> **And the pilot's unexplained nauty jump was real, not contention.** Measured on an exclusive
> node, `nauty_graph6` and `sparse6_nauty` correlate **negatively** (median ρ ≈ −0.61, 18 of 20
> cells) and complete 94.7 % against our 55.3 %. That is Corollary 3 confirmed: the one family
> implementing automorphism detection is the one that escapes the law. Full interpretation:
> `results/reports/T-13-complexity/T-13-FRAMING.md`.

That is a stronger answer to R3.7d than "the worst case is `|Aut|`-governed", because it says which
variant obeys which parameter and why, and it is the reason the pruned form is the one the paper
ships.

**Two consequences for the campaign, frozen here:**

1. The **exhaustive-versus-pruned contrast is a primary comparison**, not incidental. Both arms run
   on every ladder rung.
2. The **`no_bnb` ablation gains importance**: branch-and-bound exists only in the C++ engine
   (`native/src/canonical.cpp:193`) and has no counterpart in the manuscript's algorithm, so if the
   unpruned arm's flatness is a property of the *algorithm* it must survive `set_branch_and_bound(False)`.
   Run the ablation on every ladder rung, not only on a stratified subsample.

**Not carried forward from the pilot**: `nauty_graph6` and `sparse6_nauty` read 1.3 / 1.0 / 318.4 /
158.5 ms across the four rungs — a 300× jump between rungs 1 and 2 that both arms show together.
Two wave agents were running concurrently on this workstation during the pilot, so this is most
likely contention, not a nauty pathology. **It is not reported and must be re-measured under
exclusive allocation.** It is also the direct justification for the whole-node `--exclusive`
requirement in §3 rule 2.
