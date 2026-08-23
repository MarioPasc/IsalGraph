# T-04a — response-letter fragment

**Draft material for `reviews/response_to_reviewers.tex`. Not final prose** — `review-answer` writes
the answers; this supplies the claims and their provenance. Serves **AE.4a** (the requirement-modal
owner), **AE.3**, **R1.1**, **R1.2a/b**, **R3.6a**.

**Register**: the honest position, without spin. IsalGraph clears the size null on one of five
Suite-1 datasets and is the best representation on none of fifteen records. State that before a
reviewer computes it. The contribution that survives the comparison is *canonical **and**
edit-distance-compatible* — not "shortest", not "best correlated".

---

## R1.2a/b — why each competitor is compared under the distance it is compared under

Comparing representations on their correlation with graph edit distance presupposes that each one
*has* a distance worth comparing. We now establish that first, and separately from any GED.

Before computing a single correlation we measured the full **(representation × distance) grid** —
eleven representations, six distances, **66 cells**, every cell attempted and reported including the
failures — on one frozen 200-graph sample stratified over six node-count bands. Each representation's
primary distance is then the cheapest candidate that is **well-defined on every pair**, satisfies the
**metric axioms**, is **invariant under relabelling**, and is **non-degenerate**, with computational
cost as the tie-break. The protocol was committed before any measurement was taken, and the selection
never sees a GED: a test asserts that the selector's import closure reaches no GED loader.

Three representations — the raw adjacency triangle, graph6 and sparse6 — have **no admissible
distance at all**, each failing relabelling invariance on 49 of 50 graphs. They enter the comparison
table on their qualitative properties and carry no distance column. Of the rest, all six surviving
serialisations take **Levenshtein**; the Weisfeiler–Lehman subtree kernel takes its own kernel
distance. Padded Hamming, which an earlier draft of our protocol expected to be primary for the
canonical bit formats, is primary for **nothing**: it ties on well-definedness and loses on cost by
68× for nauty→graph6 and 8.6× for the AGM code.

**Why the screen is not a formality.** Applied to our own data it works against us. The raw adjacency
triangle fails relabelling invariance at 1/50 — its distance changes when the graph is relabelled and
nothing else — and yet on all-pairs correlation against certified exact GED it **beats IsalGraph
significantly on three of five datasets**: +0.500 on AIDS, +0.262 on LINUX, +0.123 on Letter HIGH,
Holm-adjusted p = 0.005 in each case. Restricted to equal-`n` pairs it beats IsalGraph on **none**,
and loses significantly on three. The reason is measurable: its distance correlates with |n₁−n₂| at
**0.83–0.93**, so it and GED correlate through size rather than through structure. A selection rule
allowed to look at GED correlation would have admitted it, and the comparison would have been a size
comparison wearing a structure comparison's label.

---

## R1.1 / R3.6a — what the comparison actually shows

Against T-03's certified exact GED, with the size null ρ(|n₁−n₂|, GED) reported beside every
correlation and a paired graph-level bootstrap on the difference (2,000 resamples, both arms on the
identical pairs and the identical resamples), IsalGraph's canonical string **exceeds the size null on
one of the five Suite-1 datasets**: Letter LOW, by +0.026 with a 95 % interval of [+0.008, +0.046]. On
the other four it falls below it, by −0.044 to −0.528. All five intervals exclude zero.

Under each representation's selected primary distance, IsalGraph is the best-correlated
representation on **none** of the fifteen dataset–reference records; the minimum DFS code leads on
four, the AGM canonical code on three, WL on three, sparse6-under-nauty on three and nauty→graph6 on
two. **The minimum DFS code out-correlates IsalGraph on all five Suite-1 datasets**, by +0.047 to
+0.277. We report this together with the compactness comparison, where IsalGraph is shorter on nine
of ten cohort profiles, because reporting either alone is the selective presentation R3.6a objects
to.

**What the encoding contributes is therefore not correlation and not compactness alone.** It is the
conjunction of four properties that no other member of the pool holds together, and we now measure
each. Relabelling sensitivity ψ — the median distance between a graph and a relabelled copy of
itself, in the units of the distance — is **exactly 0.0000** across all eleven draws for the seven
canonical representations, against 0.07–0.74 for the adjacency triangle, up to **1.003** for graph6
and up to **1.148** for sparse6, all intervals excluding zero. Over all 995 connected graphs on up to
seven vertices, enumerating all `n!` relabellings — 1,866,256 distinct labelled graphs — the only
graphs whose adjacency serialisation is relabelling-invariant are the **complete graphs**. On
completeness, the six complete invariants produce **zero** collisions and their zero-distance set
coincides exactly with the isomorphic set certified by VF2, bounding the collision rate at
≤ 3.4 × 10⁻⁵; the WL kernel produces **45 collisions in 183,016 pairs**, and on the two datasets that
contain no duplicate graphs every zero it emits is a false isomorphism certificate.

---

## AE.1 — the Suite-2 question, and why we report it as open

On the five large datasets, where exact GED is not computable, we evaluated the same comparison
against both ends of the T-05 bracket. **The verdict flips with the end of the bracket on all five.**
Against the `BRANCH_FAST` lower bound IsalGraph falls below the size null on 5 of 5 (−0.082 to
−0.289); against the `BIPARTITE` upper bound it exceeds it on 5 of 5 (+0.027 to +0.383). All ten
differences exclude zero.

We report this as undetermined rather than resolving it, and the reason is measured rather than
cautionary: ρ(|n₁−n₂|, LB) is **0.960–0.998**, so at these size spreads a valid lower bound and the
size null are very nearly the same statistic and no representation can exceed it. This is a property
of the regime, not a deficiency of the bound, which is the published BRANCH lower bound of Blumenthal
and Gamper. The upper bound correlates with size at 0.460–0.754, and on the datasets where exact GED
exists the truth correlates at 0.713–0.920 — between the two arms. Interpolating the bracket would
have produced a confident answer five times to a question the data does not settle.

---

## Provenance — one row per claim

| Claim | Artifact | Status |
|---|---|---|
| 66 cells, all attempted and reported | `results/grid_200.json`, `len(cells) == 66` | **measured** |
| sample: 200 graphs, strata `[33,33,33,33,34,34]`, seed 42, `n` 2–83, suite split 51/149 | `grid_200.json` `sample` block; reproducible from `(ALL_DATASETS, 200, 42)` | **measured** |
| primary distance per representation; `padded_hamming` primary for nothing | `grid_200.json` `primary_distance` + `selection_reason` | **measured** |
| F6 tie-break 68× (`nauty_graph6`) and 8.6× (`agm_cam`) | `selection_reason`, `levenshtein` at 0.001036 / 0.000575 ms/pair | **measured** |
| `k = 3` — `adjacency`, `graph6`, `sparse6`, F3 = 1/50 | `grid_200.json`, over preregistration §4.1's seven-member set | **measured** |
| `hamming` F1 = 0.032–0.035; `padded_hamming × sparse6` F1 = 0.0 | `grid_200.json` cells | **measured** |
| Letter LOW +0.026 [+0.008, +0.046]; all 15 differences exclude zero | `results/paired_null_ci.json` — **the authority for every interval** | **measured** |
| best representation per record: `min_dfs` 4, `agm_cam` 3, `wl_subtree` 3, `sparse6_nauty` 3, `nauty_graph6` 2, IsalGraph 0 | `results/f5_200.json`, all-pairs view | **measured** |
| min-DFS out-correlates IsalGraph on 5/5 Suite-1 by +0.047 to +0.277 | `f5_200.json`, point estimates on one draw | **measured**, point estimates |
| Suite-2 flip 5/5; ρ(\|Δn\|, LB) 0.960–0.998, UB 0.460–0.754, exact 0.713–0.920 | `paired_null_ci.json`, `f5_200.json` | **measured** |
| ψ = 0.0000 on 77/77 canonical rows; 1.003 graph6 / 1.148 sparse6 | `results/e1_invariance.json` | **measured** |
| invariant set of the n² family is exactly `{K_n}`, 995 graphs, 1,866,256 labelled | `e1_invariance.json` `exhaustive` block; count checks OEIS A001187 | **measured**, exhaustive |
| 0 collisions for six complete invariants; zero set ≡ VF2 isomorphic set; ≤ 2.0–3.4 × 10⁻⁵ | `results/e2_completeness.json` | **measured**, rule of three |
| WL 45/183,016 = 2.46 × 10⁻⁴ [1.79, 3.29] × 10⁻⁴; AIDS 11/11 CI [0.715, 1.000] | `e2_completeness.json` | **measured** |
| 0 axiom violations in 9,881,851 checks over 467,180 triples, `worst_excess = 0.0` | `results/e3_axioms.json` | **measured** — a correctness check, not a finding |
| adjacency +0.500 / +0.262 / +0.123, Holm p = 0.005; 0/5 equal-`n`; ρ(d, \|Δn\|) 0.83–0.93 | `results/e4_trap.json` | **measured** |
| certified exact GED, D6 cost model `[1,1,0,1,1,0]` | T-03 | inherited |
| `BRANCH_FAST` LB / `BIPARTITE` UB, all Suite-2 pairs | T-05, selected by T-27 on 46.8 M bound evaluations | inherited |
| compactness: IsalGraph shorter on 9 of 10 cohort profiles | T-04 | inherited |

> ⚠ **Four things this fragment must not be edited into.**
> 1. It does **not** claim IsalGraph is the best GED proxy. It is best on **0 of 15**.
> 2. It does **not** resolve the Suite-2 question. The verdict flips on 5/5 and the honest report is
>    that it is open.
> 3. It does **not** fault `BRANCH_FAST`. The degeneracy is a property of the size regime.
> 4. It does **not** state Letter LOW's +0.026 as "not a separation". Its paired interval excludes
>    zero; it is the one positive Suite-1 row and it is real.
>
> Full list: `T-04a-article-notes.md`, *"What is NOT claimable"*.
