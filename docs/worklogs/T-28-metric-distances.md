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

### Status

- [x] Archive secured off the Sandisk
- [x] Design note frozen and committed before results
- [x] Fast probe written; smoke pass done; full 15-cell run launched
- [ ] Full probe results reviewed
- [ ] Production reference matrices in CONTRACTS §4 schema (track C)
- [ ] `t06_f2` reference plumbing + SLURM + Picasso submission (track A, mine)
- [ ] IsalChem p.7949 fallback prepared (track B)
