# T-01 — Data lock: design, frozen 2026-08-13

**Status**: **decisions LOCKED; execution OUTSTANDING.** The ticket stays open on the board.
**Owner**: unassigned — hand to an agent via `/review-ticket T-01`.
**Decision**: [decisions](../plan/decisions.md) 22 · **Plan file**: [data](../plan/data.md)
**Estimate**: 0.5–1 day. Compute: minutes.
**Serves**: AE.1, AE.4b, R1.3a, R3.5a, E1 · **Blocks**: T-05, T-06 (and, through the cohort, everything)

---

## 1. State measured now, not assumed

Measured 2026-08-13 against the working tree and the full git history.

| Checked | Plan said | Measured | Δ |
|---|---|---|---|
| Surviving measurement scripts | 3 of 16 survive: `export_graphs.py`, `audit_recheck.py`, `audit_dropped.py`, `final_counts.py`, `gedlib_api.py` | **1 of 16.** The other four are absent from the tree **and from every commit that ever existed** (`git log --all --diff-filter=A`) | **yes — the loss is 15/16** |
| `export_graphs.py` scope | not stated | **Suite 1 only.** Five hardcoded `DatasetSpec` rows with asserted counts; `FILTER_N_MAX`, `assert_cohort` | **yes** |
| Loaders for the 5 new datasets | implied by §1's Suite-2 table | **none.** Grep for `GREC\|Mutagenicity\|COIL-DEL\|Protein` over `benchmarks/`, `src/`, `tests/` returns **zero hits**. Loaders present: `iam_letter_loader.py`, `graphedx_loader.py` | **yes** |
| Discarded-side machinery | — | `dataset_filter.FilterResult` **already carries** `dropped_indices`, `dropped_reasons`, `dropped_graph_ids`, `dropped_node_counts`, and both node-count histograms | **helps** — the discard audit needs no new filtering logic |
| `export_graphs` metadata | — | emits only `n_dropped_size` / `_disconnected` / `_trivial`. **Does not emit discarded `n̄`, `ñ`, `n_max`, `m̄` or density** | gap |
| Raw IAM data | `…/data/source/IAM_Database/extracted/` | **present**, and contains all nine: `AIDS/ COIL-DEL/ COIL-RAG/ Fingerprint/ GREC/ Letter/ Mutagenicity/ Protein/ Web/` | as stated |
| `scratchpad/` | 13 scripts lost from it | **empty** | consistent with the loss |
| `tests/unit/test_export_graphs.py` | "T-01 ports what survives" | **already exists**, 22 KB | the port is done; it was never the job |

**Consequence.** The entire **Suite-2 half of [data](../plan/data.md) §1** — 19,670 graphs,
40,024,242 pairs, `n_max = 98`, density span 0.094–0.607 — and **every discarded-side figure in §3
and §5** (1.92× / 2.27× / 1.58× / 1.19×) have **no reproducing code**. Those are the revision's
headline extension numbers: 3.7× graphs, 10.3× pairs, 8.2× larger.

---

## 2. Approach, and why

**Re-derive all ten datasets, retained and discarded, with tracked code.** Not "port what survives" —
there is nothing left to port.

The precedent is direct and recent. **T-25** re-derived six figures from the lost `ged_bounds.py`.
**None reproduced, and all six were flattering** — consistent with having been measured on one
dataset and printed as a cohort property. [data](../plan/data.md) §6 already states the rule this
ticket must now apply to itself: *treat any surviving number whose script is gone as unverified until
re-derived, not as presumptively correct.*

| Rejected | Why it lost |
|---|---|
| Re-derive Suite 2 + discards only, trust Suite 1 | Suite 1 *is* verified (`assert_cohort` + tests), so this is nearly the same work; and one audit script covering all ten is simpler than one covering five with a carve-out |
| Close on documentation, carry the re-derivation into T-06 | puts an unverified headline number on the critical path, in a revision whose central fix is that the previous numbers were not reproducible |
| Reconcile disagreements toward the plan | the failure mode this ticket exists to prevent. **What the script measures becomes the table** |

---

## 3. What must be frozen before running

| # | Frozen | Value |
|---|---|---|
| **1** | Filter | `min_nodes = 2`, `require_connected = True` — identical to `dataset_filter.filter_graphs`, unchanged from the submitted pipeline |
| **2** | `n_max` | **Suite 1: 12. Suite 2: none.** Both suites emitted by one run of one script |
| **3** | Splits | **merged** (decision 3). `export_graphs.py`'s origin-split retention for T-03 gate 0 is a separate concern and stays |
| **4** | Reconciliation rule | **the measurement wins.** Any disagreement with the current §1 is recorded as a finding with both values, and §1 is rewritten to the measured value. No value is adjusted toward the plan |
| **5** | Discard statistics | `n̄`, `ñ`, `n_max`, `m̄`, density, and the count, **per discard reason** (`size`, `disconnected`, `trivial`), per dataset — not pooled. This is what [data](../plan/data.md) §3 disclosure 1 promises to print |
| **6** | Density | `2m / (n(n−1))`, on the retained set and the discarded set separately. Closes **E1** |
| **7** | Pair counts | `C(kept, 2)` exactly. Any deviation is a bug |
| **8** | Determinism | no sampling anywhere in this ticket; the audit is a census. Nothing needs a seed |

---

## 4. Deliverables

| Artifact | Path |
|---|---|
| IAM GXL loader — GREC, AIDS (IAM), COIL-DEL, Mutagenicity, Protein, and Letter through one path | `benchmarks/real_data/eval_setup/iam_gxl_loader.py` |
| Cohort audit — both suites, retained **and** discarded, per discard reason | `benchmarks/real_data/eval_setup/cohort_audit.py` |
| Machine-readable output | `results/cohort_audit/{suite1,suite2}.json` + a markdown table |
| Unit tests | `tests/unit/test_iam_gxl_loader.py`, `tests/unit/test_cohort_audit.py` |
| Rewritten cohort tables | [data](../plan/data.md) §1, §3, §5, with a diff table of every changed value |

**Data root**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/IAM_Database/extracted/`
**GraphEdX root**: `…/data/source/GED_PRECOMPUTED/{AIDS,Letter,LINUX}`
**Env**: `PY=~/.conda/envs/isalgraph-cpp/bin/python`. Local only — no cluster.

---

## 5. Acceptance criteria

1. `$PY -m benchmarks.real_data.eval_setup.cohort_audit` emits both suites, retained and discarded,
   in one run, from tracked code.
2. **Suite 1 reproduces `export_graphs.py`'s asserted counts exactly** — 1,180 / 1,253 / 2,059 / 89 /
   769 and 3,897,911 total pairs. A mismatch here means the new script is wrong, not the old number.
3. Every Suite-2 row is measured; each value that differs from the current §1 is listed in a diff
   table with both figures and a one-line cause.
4. Discarded `n̄` and `n_max` printed **per dataset and per discard reason**, so [data](../plan/data.md)
   §3 disclosure 1 becomes a table rather than a promise.
5. `$PY -m pytest tests/unit/test_cohort_audit.py tests/unit/test_iam_gxl_loader.py -q` green.
6. Full suite unchanged or better than the reference state: **726 passed / 271 skipped** with the
   engine.
7. `ruff check` and `mypy --strict` clean on anything added under `src/`.
8. **Nothing in `scratchpad/`.** That directory is why this ticket exists.

## 6. Stop and ask

- **Suite 1 fails to reproduce.** That would mean `export_graphs.py`'s assertions and the submitted
  manuscript disagree, which is far larger than T-01.
- **`n_max = 98` does not reproduce**, or the retained ceiling moves. Decision 12 was re-affirmed on
  that number and the "8.2× extension" claim rests on it.
- **The discard bias inverts** — if the discarded set is *smaller* on average than the retained set on
  any of Mutagenicity / AIDS-IAM / Protein, [data](../plan/data.md) §3 disclosure 1 and
  [decisions](../plan/decisions.md) §6's scope limitation both need rewriting.
- **Total pairs move enough to change the compute budget** for T-05 (currently ≈ 1.05 core-h on
  40.0 M pairs).
- Any dataset that will not parse. `Web` was dropped for exactly that reason; a second one changes the
  cohort.

## 7. Debt this ticket does NOT own

- **I-05** (Fingerprint "2.3× → 1.19×"): Fingerprint is a **dropped** dataset. The corrected ratio is
  itself from the lost script. Recommendation for T-20: **do not cite Fingerprint's discard ratio at
  all**; use Mutagenicity / AIDS-IAM / Protein, which this ticket re-measures. Confirm with the PI
  before printing any Fingerprint number.
- **The pair-accounting ladder's later rungs** (`GED-available → GED > 0 → Lev > 0 → analysed`) need
  T-05/T-06 output. T-01 fixes the first two rungs, `raw → connected`, and the ladder's *definition*
  is frozen in [statistics](../plan/statistics.md) §10.
