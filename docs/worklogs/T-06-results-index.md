# T-06 — distance fidelity, under two families of reference

Two experiments, one representation side. **Every representation distance
`d_R(G,H)` is the same cached T-04a matrix in both halves** — read from
`data/source/T06/distances/`, never recomputed. Only the *reference* `d_ref`
changes. So a difference between `ged/` and `wl/` is a property of the yardstick,
not of the encoding.

```
T-06-full-recompute/
├── README.md          ← you are here
├── PROVENANCE.md      commit, engine, build hash, seed, cohort counts (covers both)
├── ged/               GRAPH EDIT DISTANCE — exact (Suite 1) and the LB/UB bracket (Suite 2)
│   ├── REPORT.md            the T-06 decision summary; opens with a five-line answer
│   ├── T-06-FRAMING.md      framings that were measured and REJECTED — read before re-deriving one
│   ├── T-06-design.md       the design note
│   ├── figures/             fig1 rho-vs-size, fig2 per-representation, fig3 absolute scale,
│   │                        fig4 information content (+ its caption)
│   ├── tables/              the four .tex tables
│   └── data/                rho_table, family_F0/F1/F2, claim_a_*, ladder, size_profile,
│                            censoring, collinearity, completion_rates, manifest, gates/
└── wl/                WL KERNEL + FOUR SPECTRAL λ-DISTANCE VARIANTS  (ticket T-28)
    ├── REPORT.md            addendum; §0 is a CORRECTION, read it first
    ├── figures/             fig_rho_vs_size_wl (WL alone),
    │                        fig_rho_vs_size_wl_vs_ged (WL beside GED, + its caption)
    └── data/                t28_bootstrap_verdicts, t28_probe_point_estimates,
                             t28_headtohead, t28_signtest_equal_n,
                             t28_size_profile_all_references
```

Restructured 2026-08-30. Previously `REPORT.md`, `figures/`, `data/` and
`tables/` sat at the top level and the WL results were mixed into them; anything
citing the old flat paths needs one directory inserted.

---

## Which reference is which

| key | what it is | where | scope |
|---|---|---|---|
| `exact` | exact GED, unit cost model | `ged/` | Suite 1 only, `n <= 12` |
| `lb` | `BRANCH_FAST` lower bound | `ged/` | Suite 2 |
| `ub` | `BIPARTITE` upper bound | `ged/` | Suite 2 |
| `wl` | WL subtree kernel distance, `h = 2`, unnormalised | `wl/` | every cell, exact at every `n` |
| `spectral` | λ-distance, normalised Laplacian, zero-padded — **pre-declared primary** | `wl/` | every cell |
| `spectral_comb` | λ-distance, combinatorial Laplacian | `wl/` | every cell |
| `spectral_adj` | λ-distance, adjacency spectrum | `wl/` | every cell |
| `spectral_esd` | 1-Wasserstein between empirical spectral distributions | `wl/` | every cell |

---

## The answer both halves are trying to give

**Does the canonical string's distance track a real graph distance better than
its competitors do, and better than counting nodes?**

### Under graph edit distance — no

`ged/REPORT.md`: the arm is best on none of the records landed and sits **below
its own `|n_i − n_j|` size null on 17 of 25 records**, including on 4 of 5
Suite-1 datasets against *exact* GED where no bracket argument applies. H3 is
refuted. The report says of that: *"No framing repairs this and none should be
attempted."* That still stands.

### Under the WL kernel — partly, and the partial is the useful part

`wl/REPORT.md`, paired bootstrap over 14 of 15 cells:

| WL reference, `all_pairs` | verdict |
|---|---|
| vs `sparse6_nauty` | **14 W / 0 T / 0 L** |
| vs `nauty_graph6` | **12 W / 0 T / 2 L** |
| vs `agm_cam` | 8 W / 4 T / 2 L (`all_pairs` only — 2 W / 5 T / 7 L under `equal_n`) |
| vs `min_dfs` | 3 W / 2 T / **9 L** |
| clears its own size null | **12 / 14**, against 6 / 14 under the best available GED reference |

**The load-bearing result is the size null, not the head-to-head.** On the five
Suite-1 datasets carrying exact GED, the arm's excess over its own null is
significantly positive on **1 of 5 against GED and 5 of 5 against WL** — same
datasets, same representation distances, only the reference moved. That is what
turns "the encoding fails" into "graph edit distance on these cohorts is a size
proxy, and under a structural reference the encoding does track structure".

### Under the spectral family — no

All four variants fail. The pre-declared primary (`spectral`, zero-padded
normalised Laplacian) clears the size null on **0 of 14** cells; the padding
convention makes it *more* size-dominated than GED. `spectral_esd` is the least
size-dominated reference of all eight and the encoding tracks it worst.
Reported win or lose, as the frozen design note required.

---

## Three things a reader will otherwise get wrong

1. **`min_dfs` is never beaten.** Not under exact GED, either bound, WL, or any
   spectral variant — see the table in `wl/REPORT.md` §0. One combination is
   nominally ahead (upper bound, `equal_n`) and its own lower bound reverses it
   on 6 of 9 cells. Do not quote it.
2. **🔴 `ged/figures/fig1_rho_vs_size.pdf` is wrong for the `wl_subtree` series**,
   and so are `ged/REPORT.md`'s `wl_subtree` sign-test rows and a manuscript
   claim at `05_results.tex:395` (`p = 0.012`). Cause and correction:
   `wl/REPORT.md` §4. **Only that one series**; every `levenshtein` arm is
   byte-identical across the two profiles. Fixed in code at `6b89b4f`.
3. **`suite2/mutagenicity` is missing from the WL bootstrap** — it timed out at
   10 h and was not rerun against the revision deadline. Every count in `wl/` is
   over 14 cells, and four of those cells are the Suite-1/Suite-2 duplicates
   (`linux`, `iam_letter_{low,med,high}`), so the distinct-dataset count is 10.
