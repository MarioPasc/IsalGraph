# Worklog — T-28: alternative similarity references for §5.4

**Ticket** T-28 · **Opened** 2026-08-29 · **Deadline** revision due 2026-08-31
**Design note (frozen)** `.claude/notes/review/tasks/T-28-design.md`
**Driver** PI request 2026-08-29: §5.4's GED correlation is weak enough to risk rejection;
add WL kernel and spectral λ-distance as alternative similarity references.

**Acceptance criterion (user):** the T-04a representation distances are *maintained*
(`levenshtein` for all competitors except `wl_subtree`, which uses `kernel`); only the
**reference** against which correlation is measured changes.

---

## 2026-08-29 — session 1

### Context secured

The Sandisk 2TB holding every T-06 artifact was about to be unplugged. Copied
`research/ISAL/completed/isalgraph/` (2.8 GB, 44,882 files) to
`/home/mpascual/research/data/isalgraph_archive/`; verified by file count and byte count on
both sides. Nothing else from the drive was copied.

### The decisive structural finding

**The representation side of the correlation is already computed and cached.**

```
data/source/T06/distances/{suite}/{dataset}__{rep}__{metric}.npz
    distance_matrix  float64 (G, G)     <- the expensive artifact
    graph_ids        <U16    (G,)
    node_counts      int32   (G,)
    defined_mask     bool    (G, G)
    metadata         0-d JSON (code_commit, build_hash, seed, ...)
```

515 MB across 15 (suite, dataset) cells × 7 arms. The reference matrices (`exact`, `lb`, `ub`)
live in separate trees and are joined on `graph_ids` at analysis time — the two sides are
**never** stored joined. Swapping the reference therefore costs **no re-encoding and no
re-computation of any representation distance**, which is exactly what the acceptance
criterion requires, and it is enforced structurally rather than by discipline.

`t06_f2.py::run_correlation_group` already takes `references: dict[str, ndarray]` and loops
over its keys, so adding `wl` and `spectral` is an extension, not a rewrite.

### Cost consequence

The 12 h `distances` SLURM stage is **skipped entirely**. Only the F2 statistics stage
(4 CPU / 24 G / 12 h wall, 15 shards) needs re-running, against ~2× the references. The
2-day budget is comfortable.

### Design decisions taken with the user before they left

1. **WL circularity.** The `wl_subtree` arm's own distance *is* the WL kernel distance, so
   under a WL reference it scores ρ = 1.0 by construction. Decision: report the row, mark it
   degenerate, **exclude it from win/loss counts**. Not silently dropped.
2. **Reporting.** Compute all references; produce the full grid **and** both manuscript
   variants (full-grid, winners-only); the PI decides at writing time.
3. **Spectral primary.** Normalised Laplacian `L_sym`, sorted spectrum, zero-padded,
   Euclidean — frozen before results, citing Wilson & Zhu, *Pattern Recognition*
   41(9):2833-2841 (2008).
4. **IsalChem p.7949 fallback.** Prepared in parallel on its own branch as insurance.

### Fast probe, and what it changed

Wrote `benchmarks/real_data/eval_reference_metrics/{spectral,probe_t28,summarise_t28}.py` —
point estimates only, no bootstrap — to answer the PI's question in minutes rather than
waiting on the full campaign. Smoke pass on AIDS + LINUX (`all_pairs`):

| reference | arm ρ (AIDS) | best competitor | **size null** | arm clears null? |
|---|---:|---:|---:|:--|
| GED exact | +0.2718 | +0.8213 `agm_cam` | **+0.8163** | no |
| **WL kernel** | +0.3154 | +0.4127 `min_dfs` | **+0.2127** | **yes** |
| spectral (norm L, padded) | +0.1359 | +0.8499 `agm_cam` | **+0.8717** | no |
| spectral (comb L, padded) | +0.3612 | +0.5990 `agm_cam` | +0.6265 | no |
| spectral (adjacency) | +0.1477 | +0.9312 `agm_cam` | +0.9592 | no |

Two results, and the second is the one that matters:

- **The zero-padded spectral distance is *more* size-dominated than GED** (null 0.87 vs 0.82).
  Mechanism, exact: `tr(L_sym) = n`, so `‖λ‖²` scales with `n`, and padding to `n_max` turns
  the Euclidean distance into a size proxy. **The padding convention carries the confound,
  not the normalisation.** This was not anticipated when the primary was frozen. It motivates
  `spectral_esd` (1-Wasserstein between eigenvalue *measures*), added as a declared,
  mechanism-driven variant — reported win or lose.
- **Under the WL kernel reference the size null collapses to ≈ 0.16–0.21 and IsalGraph
  clears it on 3/3 cells.** The WL kernel is a genuinely *structural* reference, not a size
  proxy. This directly addresses §5.4's headline concession — *"where the trivial baseline
  beats the representation, which competitor wins is second-order"* — which is the single
  most damaging sentence in the subsection. IsalGraph still loses the head-to-head to
  `min_dfs` on these two cells.

**Caveat, not yet resolved:** the probe uses per-arm pair masks, so it does not reproduce
T-06's `n_pairs` (234,258 vs 131,148 on AIDS) or its ρ (0.2718 vs 0.3266). T-06 intersects the
defined masks across the whole comparison group. The production run must use the group
intersection and reproduce T-06's `exact` column to 4 decimals — that is gate **G1**.

### Full probe — 15 cells, 1,070 records, point estimates

Raw records: `docs/worklogs/T-28-artifacts/probe_point_estimates.json`.

**Verdict against the *best* competitor** (the strictest reading), `all_pairs`:

| reference | win | loss | clears size null |
|---|---:|---:|---:|
| GED exact | 0 | 5 | 1/5 |
| GED lower bound | 0 | 10 | 0/10 |
| GED upper bound | 0 | 10 | 7/10 |
| **WL kernel** | **3** | 12 | **12/15** |
| spectral (norm L, padded) | 0 | 15 | 0/15 |
| spectral (comb L, padded) | 2 | 13 | 4/15 |
| spectral (adjacency) | 0 | 15 | 0/15 |

**Verdict per competitor** — the reading that answers the PI's *"gane a sus competidores"*.
Cells where the arm's ρ is higher, out of all cells; mean Δρ in brackets. `all_pairs`:

| reference | `agm_cam` | `min_dfs` | `nauty_graph6` | `sparse6_nauty` |
|---|---|---|---|---|
| GED exact | 1/5 (−0.225) | 0/5 (−0.154) | 3/5 (+0.053) | 4/5 (+0.136) |
| GED lower bound | 0/10 (−0.168) | 1/10 (−0.082) | 4/10 (+0.009) | 9/10 (+0.101) |
| GED upper bound | 6/10 (+0.106) | 1/10 (−0.042) | 2/10 (−0.034) | 2/10 (−0.024) |
| **WL kernel** | **15/15 (+0.177)** | 3/15 (−0.024) | **12/15 (+0.149)** | **15/15 (+0.206)** |
| spectral (norm L) | 0/15 (−0.268) | 1/15 (−0.126) | 7/15 (+0.015) | 11/15 (+0.114) |
| spectral (comb L) | 5/15 (−0.011) | 4/15 (−0.069) | 13/15 (+0.213) | 15/15 (+0.221) |
| spectral (adjacency) | 0/15 (−0.326) | 1/15 (−0.139) | 5/15 (−0.120) | 7/15 (−0.005) |

`wl_subtree` is excluded under the `wl` reference (degenerate, ρ ≡ 1.0 by construction).

### What this means

**The WL kernel reference carries the result; the spectral family does not.**

1. **Against the four non-degenerate competitors, IsalGraph wins three outright** — `agm_cam`
   15/15, `sparse6_nauty` 15/15, `nauty_graph6` 12/15 — and loses only to `min_dfs`, by a mean
   Δρ of **−0.024**, which is inside the interval width T-06 reported for comparable
   contrasts. This is the first reference under which the arm is not last.
2. **It clears its own `|Δn|` size null on 12 of 15 cells**, against 8 of 25 under GED and 1 of
   5 under *exact* GED. The mechanism is measurable: the size null against the WL kernel runs
   ρ ≈ 0.16–0.87 where against GED it runs 0.71–**0.997**. The WL kernel distance is a
   *structural* reference; graph edit distance on these cohorts is largely a size proxy.
   This is what repairs §5.4's most damaging sentence — *"where the trivial baseline beats the
   representation, which competitor wins is second-order"*.
3. **The spectral family is weaker and the primary variant fails outright.** The pre-declared
   `spectral` (normalised L, zero-padded) wins 0/15 against the best competitor and clears the
   size null 0/15. The padding artifact identified in the smoke pass is confirmed at full
   scale. `spectral_comb` is middling. `spectral_esd`, the size-controlled variant, is being
   built on track C and is not yet measured.

**Caveats, and they are not small.** These are point estimates with no confidence intervals,
computed on **per-arm** pair masks rather than T-06's group intersection, so they do not
reproduce T-06's `n_pairs` or ρ (AIDS: 234,258 pairs and ρ = 0.2718 here vs 131,148 and
0.3266 in T-06). **No verdict above is a significance verdict.** The production campaign must
use the group intersection, reproduce T-06's `exact` column to 4 decimals (gate G1), and
supply paired graph-level bootstrap intervals. A mean Δρ of −0.024 against `min_dfs` may well
be a tie; that is for the bootstrap to say, not the point estimate.

### Status

- [x] Archive secured off the Sandisk
- [x] Design note frozen and committed before results
- [x] Fast probe written; smoke pass done; full 15-cell run done
- [x] Full probe results reviewed — **WL kernel is the load-bearing reference**
- [ ] Production reference matrices in CONTRACTS §4 schema (track C)
- [ ] `t06_f2` reference plumbing + SLURM + Picasso submission (track A, mine)
- [ ] IsalChem p.7949 fallback prepared (track B)
