# T-06 — artifact index

Every non-Python file T-06 produced, and where it lives. Written 2026-08-25, after
`ticket/T-06` merged to `main` at **`27bdfcb`** (parents `c1d36b1` + `09002bc`, pushed).

Three locations, and they are not interchangeable:

| | what it is | files | size |
|---|---|---|---|
| **A. the repo** | code, notes, plan edits — version-controlled | 72 (27 non-Python) | — |
| **B. the report** | the curated deliverable, meant to travel | 28 | 6.5 MB |
| **C. the source** | everything the campaign wrote; the ground truth | 448 | 535 MB |

The 45 Python modules and test files in the merge are deliberately not listed here.
They are under `benchmarks/real_data/eval_{encoding,distance,stats,size_profile}/`
and `tests/unit/test_t06_*.py`; `experiments/README.md` maps code to paper artifact.

---

## A. In the repo — committed at `27bdfcb`

### A.1 The paper deliverables — `.claude/notes/review/tasks/`

Read these in this order. The first two are what T-14 and T-20 actually consume.

| file | size | what it carries |
|---|---|---|
| `T-06-letter-fragment.md` | 155 ln | Fragments A–F for the response letter, plus a per-claim provenance table and a **"do not lift into the letter"** list |
| `T-06-article-notes.md` | 227 ln | ordered by consequence, every item names its owner and where it lands in the manuscript; ends with **"what is NOT claimable"** |
| `T-06-FRAMING.md` | 74 KB | §1–14. The defensible framings, the red lines, and every retraction with the evidence that forced it |
| `T-06-design.md` | 130 KB | the full decision log. §18.8 (intersection–union) and §18.9 (the `all_pairs` view ruling) live here |
| `T-06-HANDOFF.md` | 276 ln | durable context replacement, written when the orchestrator's window filled |
| `T-06-FILES.md` | — | this file |

### A.2 Plan files the ticket corrected

Propagated in place per `review-close` §3 — wrong text **struck, not deleted**.

- `.claude/notes/review/plan/statistics.md` — the last live copy of the `N_actual`
  formula missing two terms, struck at the point it was asserted; tier-3 MRM defect
  recorded in §5, **unassigned** (IsalSR and IsalHG inherit it)
- `.claude/notes/review/plan/preregistration.md` — RESULT section: which branch each
  pre-declared rule took, its standing request answered, and the three terms execution
  found undefined, with the enumerated counterfactual
- `.claude/notes/review/plan/tickets.md` — T-06 board row struck with headline numbers
- `.claude/notes/review/plan/data.md`
- `.claude/CLAUDE.md` — reference state → **2,550 passed / 321 skipped**

### A.3 Wave working notes — `.claude/notes/2026-08-16-t06-recompute/`

| file | what it carries |
|---|---|
| `CONTRACTS.md` | 441 ln. §3.1 symbols-not-characters and the `\x1f` separator rule; §3.2 the `status` / `error_kind` partition |
| `encoding.md`, `distance.md`, `stats.md` | per-track logs |
| `summary.md` | wave summary |
| `probe_encode.json` | pre-launch backend probe |
| `verify_canonical.json` | canonical-string verification probe |

### A.4 Shell drivers

Campaign entry points — `.claude/notes/2026-08-16-t06-recompute/`:
`run_encode_isalgraph.sh` · `run_encode_competitors.sh` · `run_encode_resume.sh` ·
`run_distances.sh` · `verify_f1_reference_arm.sh`

Pipeline — `experiments/paper_pipeline/`:
`run_f2.sh` · `run_f2_early.sh` · `run_size_profile.sh` · `archive_t06.sh`

### A.5 One symlink

`benchmarks/eval_distance` → `real_data/eval_distance` (mode `120000`, matching the
other 16). `eval_size_profile` has **no** symlink — `run_size_profile.sh` invokes
`benchmarks.real_data.eval_size_profile.*` fully qualified instead. A convention gap,
not a breakage.

---

## B. The curated report — 28 files, 6.5 MB

`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute/`

```
REPORT.md            21 KB   the decision summary (byte-identical to data/source/T06/DECISION_SUMMARY.md)
PROVENANCE.md        1.9 KB  engine, build hash, seed, budget, commits
T-06-design.md       130 KB  copy
T-06-FRAMING.md      74 KB   copy
```

### `data/` — 18 JSONs

| file | size | what it answers |
|---|---|---|
| `family_F2.json` | 266 KB | the confirmatory family. `cardinality.n_actual = 79`, `closed_form_expression = "101 - 5*3 - 7"`, `f0_demoted = true`, `d_applied = false`, `discrepancy = 0`; `bh_primary` m=79, rejects 75 |
| `family_F0.json` | 4.7 KB | calibration gate — fires on 4 of 5 |
| `family_F1.json` | 6.6 KB | bracket gate — `d = 7 of 10` (measured; outside the cardinality) |
| `rho_table.json` | 528 KB | every ρ, its size null, and the excess. Source for "below the null on 17 of 25" |
| `size_profile.json` | 1.1 MB | within-`n` decomposition. The 0.9656 → 0.0779 collapse |
| `size_profile_censoring_confound.json` | 586 KB | whether censoring drives the size profile |
| `claim_a_strata.json` | 838 KB | Claim A by size stratum — the 112/112 above n=20 |
| `claim_a_suite1.json`, `claim_a_suite2.json` | 119 / 225 KB | per-suite bit counts |
| `manifest.json` | 95 KB | every encoding cell with status and provenance |
| `ladder.json`, `ladder_suite1.json` | 4.7 / 2.9 KB | pair-accounting ladder; the 24,764,422-pair collision check |
| `censoring.json` | 28 KB | censoring by \|Aut\| — 0 % / 21.9 % / 100 % |
| `collinearity.json` | 4.9 KB | VIF screening. Why `aids_iam` and `coil_del` are not identifiable |
| `completion_rates.json` | 61 KB | per-cell completion; source for `c = 7` |
| `gates/gate_T06_reproduction.json` | 291 B | max \|Δ\| vs `corrected_rho_table.json` = **0.0000** |
| `gates/gate_T06_structural.json` | 93 KB | 190/190 distance matrices |
| `gates/repro_table.json` | 7.3 KB | the compared table |

### `figures/` — 3 figures, PDF + PNG

- `fig1_rho_vs_size` — ρ against graph size, the collapse
- `fig2_rho_by_representation` — per-competitor breakdown
- `fig3_absolute_scale` — twin-axis absolute scale with the LB/UB band

---

## C. The source of truth — 448 files, 535 MB

`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06/`

| dir | files | size | contents |
|---|---|---|---|
| `distances/` | 190 | 518 MB | every distance matrix, one per (representation, dataset) cell |
| `encodings/` | 166 | 9.0 MB | every encoding cell, `.npz` |
| `families/` | 29 | 2.0 MB | `family_F0/F1/F2.json`, `rho_table.json`, and **25 per-dataset partials** under `f2_partials/` (15) and `f2_partials_early/` (10) |
| `logs/` | 33 | 212 KB | campaign logs, UTC-stamped: `encode_competitors_*`, `distances_*`, `f2_*`, `f2early_*`, `confound_*` |
| `figures/size_profile/` | 6 | 2.3 MB | the three figures, PDF + PNG |
| `gates/` | 3 | 112 KB | as archived |
| `f1_recheck/encodings/suite2/` | 6 | 92 KB | `.npz` re-encodes for the F1 reference arm |

Loose at top level: `DECISION_SUMMARY.md`, the 12 result JSONs (copied into the
archive), `f1_recheck.log`, `f1_verify.log`, `resume.log`, `f1_idx_protein.txt`.

---

## Known gap — fix before the letter ships

`T-06-letter-fragment.md` cites **`families/f2_partials*/`** as the provenance for the
determinism claim ("14 model fits computed twice by independent processes hours apart
at the same seed are byte-identical").

Those 25 partials exist in **C** (`data/source/T06/families/`) but were **not copied
into B** — the archived report's `data/` holds 18 files and none of them are partials.
Anyone reproducing from the report alone cannot back that claim.

Two fixes, either is fine: copy `f2_partials/` and `f2_partials_early/` into
`results/reports/T-06-full-recompute/data/families/`, or repoint the provenance row at
the source tree and say so explicitly. **Unresolved as of 2026-08-25.**

---

## Reproduction parameters

Engine `cpp`, build hash `298fc1188bf1b051`, seed 42, 300 s per-graph encode budget.
`src_commit = c1d36b1` — T-06 changed no file under `src/isalgraph/`, so the campaign
ran the same encoder as `main`. Cohorts: 16,370 Suite-2 graphs / 21,710,892 pairs and
5,350 Suite-1 graphs / 3,897,911 pairs. Cost model: node and edge ins/del = 1,
substitutions free. Exact GED from `networkx` A\* below 12 nodes; above that the
BRANCH-FAST / IPFP bracket from GEDLIB, reported as two series and never interpolated.

---

## Addendum — the `T06_exhaustive` campaign (2026-08-26)

A **fourth** location, and it is not a copy of C. It holds the re-encoded IsalGraph arm
and the greedy ablation, and reuses everything else by symlink rather than by copy.

`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06_exhaustive/`

| dir | contents |
|---|---|
| `encodings/` | **185 cells over 13 representations.** 30 are real files from Picasso — `isalgraph_exhaustive` and `isalgraph_greedy`, 15 cells each. The other 155 are absolute symlinks into `T06/encodings/`. 20 of those (`isalgraph_pruned`, `isalgraph_canonical`) were **added while landing the campaign**: the run's own staging had omitted them, and without them `fig4` has no reference arm, since `design.REFERENCE_KEY` is `isalgraph_pruned` |
| `distances/` | 60 real new-arm matrices (188 MB) + 190 symlinks into `T06/distances/`. Picasso holds all 250 as real copies (699 MB); the 190 were **deliberately not pulled back** — copying them would replace a symlink with a duplicate, which is what F5 exists to prevent, for 518 MB |
| `families/` | 33 files, 2.3 MB. **`family_F2.json` (260 KB) is the confirmatory record; `rho_table.json` (534 KB) every ρ.** Plus three partial sets — see the table below |
| `gates/` | `gate_T06_reproduction.json`, `gate_T06_structural.json`, `repro_table.json` |
| `logs/` | 16 files — the encode log and the 15 per-dataset F2 logs |
| `completion_rates.json` | 70 KB, **11 representations** — the reference arm, the ablation and the 9 comparators. Not 13: `isalgraph_pruned` / `isalgraph_canonical` are figure-only and must not appear here (see the divergence note below) |

### `families/` — three partial sets, and only one is live

| dir | what it is |
|---|---|
| `f2_partials/` | **the live set.** 15 partials, 120 A1 cells (12 per suite2 partial), 300 ρ rows, 25 MRM fits. What `family_F2.json` was merged from |
| `f2_partials_claimB_only_20260826/` | run 1's 15 partials — 6 h 50 m of correct Claim B work with `a1_cells: []` throughout, because Picasso's `encodings/` held only the 30 new cells. Kept as the record of the defect, **not** an input |
| `_run2_stale_completion/` | the `N_actual = 86` `family_F2.json`, merged against a `completion_rates.json` built from a 30-cell tree. Internally consistent (`discrepancy = 0`) and **wrong**. Kept as the counter-example |

### The headline record

```
family_F2.json:  N_actual=79  closed_form=79  discrepancy=+0   (101 - 5*3 - 7)
                 BH over N_actual  : 76 rejected at q=0.05     (T-06: 75 of 79)
                 BH over N_max=182 : 73 rejected at q=0.05
                 120 a1 cells · 300 rho rows · 25 mrm fits
                 c cells: agm_cam x6, min_dfs x1 -- identical to T-06's
                 declared reps: adjacency agm_cam graph6 min_dfs nauty_graph6
                                sparse6 wl_subtree  (no IsalGraph arm is a member)
```

Produced by merge **2107278** after three failures; the chain and every failure are in
`.claude/notes/2026-08-25-t06-exhaustive/log.md`.

### ⚠ The Sandisk and Picasso trees legitimately differ

| | Sandisk | Picasso |
|---|---|---|
| `encodings/` | **185 cells / 13 reps** | **165 cells / 11 reps** |
| extra | `isalgraph_pruned`, `isalgraph_canonical` | in `_encodings_figure_only/` |

`fig4` needs the pruned arm — `design.REFERENCE_KEY` is `isalgraph_pruned` — and F2 must
**not** see it: it is not a declared family member, and its presence raises
`FamilyError: c names an undeclared representation`. Do not "fix" the divergence by
syncing one onto the other.

**Report artifacts are now 4 figures and 4 tables, not 3 figures.** `figures/` carries
`fig4_information_content` (PDF + PNG + `.caption.tex`) and `tables/` carries
`tab_representation_properties`, `tab_representation_headtohead`, `tab_bit_overhead` and
`tab_representation_summary`. The §B listing above predates all eight.

> **`fig1`–`fig3` do not show the new arms and cannot.** They derive from the frozen
> `data/size_profile.json`, which holds 7 representations. Only `fig4` and three of the
> four tables carry `isalgraph_exhaustive` / `isalgraph_greedy`;
> `tab_representation_headtohead` omits them correctly, being the *comparator* table.

Reproduction parameters differ from T-06's in one place that matters: the encode budget is
**30 s**, not 300 s. `code_commit = d6a9f4b`; `src_commit` reads `unknown` for every cell
(see `T-06-EXHAUSTIVE-HANDOFF.md` §7). Engine and build hash are unchanged.

### What the campaign measured — read these numbers from here, not from a figure

| quantity | value | where it comes from |
|---|---|---|
| median entropy bits, `n = 20` | exhaustive **114.1** · pruned 136.3 · nauty-sparse6 144.0 | `encodings/`, via `tables._bits_at` |
| median entropy bits, `n = 40` | exhaustive **342.4** · pruned 348.7 · nauty-sparse6 **336.0** | same |
| **does it clear nauty-sparse6 at `n = 40`** | **no** — 342.4 against 336.0, 6.3 of 12.7 bits closed | same |
| fallback rate at the `n = 40` stratum | **96.8 %** (3 of 93 graphs carry a true `w*_G`) | `fallback_used` in the `.npz` cells |
| gain where the search completes | **12.5–17.0 %** in every band, highest at `n = 31–40` | paired over 21,720 graphs |
| greedy non-invariance, `n = 5–9` | **50.7 / 68.8 / 81.2 / 91.0 / 84.7 %** against 0 % for both canonical arms | the ablation, `log.md` |
| minimality violations | **0** of 18,461 completed graphs, both directions | `length` in the `.npz` cells |

`psi` proper was **never measured** for `isalgraph_greedy` — the ablation's Part A did not
run, and a non-invariance rate is not a `psi`. `tables.PSI_MEASURED` carries `--` for it.

Full derivations, the three handoff corrections and every failure that had to be cleared:
`.claude/notes/2026-08-25-t06-exhaustive/log.md` § "Landing the campaign".
