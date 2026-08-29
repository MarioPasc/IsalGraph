# T-28 — Alternative similarity references for §5.4 (distance fidelity)

**Opened** 2026-08-29. **Owner** Mario Pascual González. **Driver** PI request, 2026-08-29.
**Status** design frozen before results; see §7 for the freeze record.

---

## 1. Why this ticket exists

§5.4 (`sec:res-fidelity`) currently reports **0 win / 1 tie / 24 loss** for the canonical
string against graph edit distance, and — the headline risk — the arm falls **below its own
`|n_i − n_j|` size null on 17 of 25 records (68 %)**, including against **exact** GED on
Suite 1 where no bracket argument applies. T-06's own REPORT calls this "the headline risk".

The PI's judgement, 2026-08-29:

> Creo que hay que hacer algo con la comparativa de la Subsección 5.4 (fidelity, la del GED).
> Si la dejamos como está, corremos el riesgo de que rechacen el artículo en esta ronda. […]
> además del GED, pruebes con otras dos métricas de similitud de grafos: Weisfeiler-Lehman
> (WL) Kernel […] y Spectral Distance (λ-distance) […] Me quedaría contento con tal de que
> IsalGraph gane a sus competidores en al menos una de las dos métricas.

**What this ticket changes: the REFERENCE, never the representation distances.** The
representation side is frozen by T-04a and reused byte-for-byte from the T-06 cache.

---

## 2. What is frozen and what moves

| Side of the correlation | Status |
|---|---|
| **Representation distance** `d_R(G,H)` | **FROZEN — T-04a selections, read from cache, never recomputed.** `levenshtein` for the six admissible serialisations, `kernel` for `wl_subtree`, `size_null` for the baseline. |
| **Reference distance** `d_ref(G,H)` | **MOVES.** `exact` / `lb` / `ub` retained; `wl` and the spectral family added. |

This is the ticket's acceptance criterion and it is checked mechanically: every
representation distance is loaded from
`data/source/T06/distances/{suite}/{dataset}__{rep}__{metric}.npz`, whose `metadata` field
carries the T-06 `code_commit` and `build_hash`. No encode step runs in T-28.

---

## 3. The reference set

| key | definition | role |
|---|---|---|
| `exact` | exact GED, unit cost model (D6) | retained, Suite 1 |
| `lb` | `BRANCH_FAST` lower bound | retained, Suite 2 |
| `ub` | `BIPARTITE` upper bound | retained, Suite 2 |
| **`wl`** | WL subtree kernel distance, `sqrt(K(G,G)+K(H,H)−2K(G,H))`, linear kernel on colour-count multisets, `h = 2`, unnormalised, constant base colour | **new** |
| **`spectral`** | `‖λ(L_sym,G) − λ(L_sym,H)‖₂`, spectra sorted non-increasing, **zero-padded** to the cohort's `n_max` | **new — PRIMARY spectral variant** |
| `spectral_comb` | same, combinatorial `L = D − A` | sensitivity (the PI's literal wording) |
| `spectral_adj` | same, adjacency spectrum | sensitivity |
| `spectral_esd` | 1-Wasserstein between the **empirical spectral distributions** of `L_sym` | size-controlled variant; see §5 |

`L_sym = I − D^{−1/2} A D^{−1/2}`. Spectrum confined to `[0,2]`. Isolated vertices contribute a
zero row/column rather than a division by zero; the cohorts are connected, so this is a guard.

Primary spectral variant selected **before any result was inspected**, citing
Wilson & Zhu (2008), *A study of graph spectra for comparing graphs and trees*,
**Pattern Recognition 41(9):2833-2841**, DOI `10.1016/j.patcog.2008.03.011` — the target
venue's own literature.

### 3.1 🔴 The WL reference is DEGENERATE for the `wl_subtree` arm

The `wl_subtree` competitor arm's primary distance **is** the WL kernel distance (T-04a:
`kernel` primary for that arm). Under the `wl` reference, `d_R ≡ d_ref` for that arm and its
ρ is **exactly 1.0 by construction**.

**Decision (user, 2026-08-29): report the row, mark it degenerate, exclude it from every
win/loss/tie count.** It is not silently dropped — a reviewer must be able to see that we
know why it is 1.0. Implemented as `DEGENERATE = {("wl", "wl_subtree")}`; the head-to-head
"best competitor" is taken over the remaining five serialisations.

The `wl` reference matrix is **not recomputed**: it is the cached
`{dataset}__wl_subtree__kernel.npz` matrix itself, which guarantees the identity is exact
rather than approximate.

### 3.2 The WL reference is not a complete invariant

T-04a measured **45 collisions in 183,016** draws for WL. 1-WL cannot separate K₃,₃ from the
triangular prism at any `h`, so `d_wl = 0` occurs for non-isomorphic graphs. This is a
property of the reference and must be stated where the reference is introduced. It does not
invalidate the correlation; it bounds what a high ρ against `wl` can mean.

---

## 4. The size null is mandatory, not optional

Every new reference is reported **with `ρ(|n_i − n_j|, d_ref)` beside it**. This is
non-negotiable and it is the instrument that makes T-28 honest: §5.4's whole finding is that
GED on these cohorts is size-dominated (`|Δn|` null reaches **0.9971** on COIL-DEL), so a new
reference that is *also* size-dominated has changed nothing, however the head-to-head lands.

**A win against a size-dominated reference is not a result.** If IsalGraph wins under a
reference whose size null exceeds its own ρ, that is reported as a loss to the null and the
win is not claimed.

---

## 5. `spectral_esd`, and why adding it is not shopping

The smoke pass (2 datasets, point estimate, before the full run) showed the size null
against `spectral` at **ρ ≈ 0.87**, *higher* than against exact GED (**0.82**). The mechanism
is exact and was not anticipated: `tr(L_sym) = n` for a graph with no isolated vertices, so
`‖λ‖² ` grows with `n`, and zero-padding to `n_max` turns the Euclidean distance between
padded spectra into a size proxy. **The padding convention, not the normalisation, carries
the confound.**

`spectral_esd` compares the eigenvalue *measures* (each eigenvalue carrying mass `1/n`)
instead of padded vectors, which removes the `n`-scaling by construction.

**This is declared as a mechanism-driven addition, not a variant search.** The rule that
keeps it honest: **all four spectral variants are reported, win or lose**, and this design
note — written before the full run — is the record that `spectral` (zero-padded) was the
pre-declared primary. §7 freezes it.

---

## 6. Views, statistics, and what is reused

- **Views**: `all_pairs` and `equal_n` (`n_i = n_j`, where the size channel is removed by
  construction), as T-06.
- **Statistics**: unchanged from T-06 — Spearman ρ, graph-level bootstrap CI (resampling
  unit = graph, not pair), paired differences against the IsalGraph arm on identical pairs,
  sign test over equal-`n` strata. A loss inside a confidence interval is a **TIE**.
- **Pair masking**: the group-intersection convention of T-06, not per-arm masks — every arm
  in a group is correlated on the *same* pairs. (The fast probe used per-arm masks and
  therefore reports slightly different `n_pairs` and ρ than T-06; the production run must
  reproduce T-06's `exact` column to 4 decimals as its gate. See §8.)
- **Reused, never recomputed**: all representation distance matrices, all `size_null`
  matrices, all GED matrices (`exact`, `lb`, `ub`), the `wl` matrix.
- **Computed fresh**: the four spectral matrices only — `O(n³)` per graph on `n ≤ 98`,
  seconds per cohort.

---

## 7. Freeze record

Frozen 2026-08-29, before the full 15-cell run and before any spectral result beyond the
2-dataset smoke pass reported in §5:

1. `spectral` (normalised Laplacian, zero-padded, Euclidean) is the **primary** spectral
   reference. `spectral_comb`, `spectral_adj`, `spectral_esd` are secondary and all four are
   reported regardless of outcome.
2. The `wl_subtree` arm is **excluded from win counts under the `wl` reference only**.
3. The size null is reported against **every** reference, and a win against a reference whose
   size null exceeds the arm's ρ is **not claimed as a win**.
4. The head-to-head "best competitor" is the maximum ρ over the non-IsalGraph, non-degenerate
   representations present in the cell.

---

## 8. Acceptance gates

| # | Gate | Why |
|---|---|---|
| G1 | The `exact` / `lb` / `ub` columns reproduce T-06's `rho_table.json` to 4 decimals | proves the reference swap did not disturb the retained arms |
| G2 | Every representation distance matrix's `metadata.code_commit` matches T-06's | proves T-04a distances were reused, not recomputed |
| G3 | Every reference matrix is symmetric, zero-diagonal, finite, and joins on `graph_ids` | the T-06 structural gate, applied to the new references |
| G4 | `ρ(wl, wl_subtree) == 1.0` exactly | confirms the degeneracy is the expected identity and not a near-miss hiding a bug |
| G5 | Off-diagonal exact-zero fraction < 0.99 per reference matrix | the silent-zero failure shape from the GEDLIB correction |
| G6 | Size null reported on every emitted record | §4 |

---

## 9. Out of scope

- Re-running any encoder. T-28 touches no `encode()` path.
- Changing T-04a's primary-distance selections.
- The IsalChem p.7949 subgraph-based metrics — tracked separately as the **T-28b** fallback
  track, prepared in parallel as insurance and used only if `wl` and the spectral family both
  fail to produce a defensible result.
