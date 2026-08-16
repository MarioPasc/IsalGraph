# T-06 — design note

**Full recompute.** All experiments, C++ engine, ten-dataset cohort, competitor columns, the frozen
confirmatory family. Opened 2026-08-16.

**Depends**: T-02 ✅, T-03 ✅, T-04 ✅, T-05 ✅ — all struck. **T-04a is running and gates the
comparator distance matrices only** ([tickets](../plan/tickets.md) line 203); nothing else in this
ticket waits on it.

**Read first**: [statistics](../plan/statistics.md), [data](../plan/data.md),
[competitors](../plan/competitors.md), [labels](../plan/labels.md),
[preregistration](../plan/preregistration.md) (authoritative over `statistics.md` §9),
[approx_ged](../plan/approx_ged.md) §3–§4, and the four inherited-correction blocks in the
[board header](../plan/tickets.md).

---

## 1. State measured now, not assumed

Everything in this section was measured on 2026-08-16, not read out of a plan file. **Six values
differ from what the plan predicted; three of them change the design.**

### 1.1 Environment and engine

| Quantity | Measured | Source |
|---|---|---|
| `isalgraph.engine()` | **`cpp`** | direct call |
| `build_hash` | `298fc1188bf1b051` | `isalgraph.build_info()` — matches T-04's recorded hash, so the `.so` is not stale |
| compiler / ISA | gcc 12.2.0, `x86-64-v3`, `avx2=1`, `avx512f=0`, `NDEBUG=1` | same |
| repo HEAD | `f7ad283`, tree clean | `git status --porcelain` empty |

### 1.2 Picasso

| Quantity | Measured | Plan said |
|---|---|---|
| fscratch **files** | **227.2k / 250.0k soft, 400.0k hard** | "399.7k, EXCEEDED" (`CLAUDE.md`) — **stale; 22.8k of soft headroom exists** |
| fscratch space | 0.47 TB / 1.40 TB | — |
| home | 24.44 GB / 0.28 TB, 14.2k / 35.0k files | — |
| queue | **empty** — T-05's campaign fully drained | — |

**Consequence**: the GEDLIB build tree is gone and the quota pressure `T-23` was raised for has
cleared. T-06 has room, but 22.8k files is not much: an encoding campaign that writes one file per
graph would blow it. **T-06 writes one `.npz` per (dataset × representation), never per graph.**

### 1.3 🔴 `pruned_canonical_string` and `canonical_string` are BOTH complete invariants — verified, not assumed

The PI asked for this before the reference arm could be chosen. Measured over the **graph atlas**,
which enumerates exactly one representative per isomorphism class, so two distinct entries sharing a
string is a collision by construction and needs no isomorphism test:

| | connected graphs, 2 ≤ n ≤ 7 | distinct strings | collisions | invariance failures | errors |
|---|---:|---:|---:|---:|---:|
| `pruned_canonical_string` | 995 | **995** | **0** | **0** | 0 |
| `canonical_string` | 995 | **995** | **0** | **0** | 0 |

Per-`n` counts `1 / 2 / 6 / 21 / 112 / 853` reproduce **OEIS A001349**, confirming the enumeration is
the complete isomorphism-class set. Invariance was tested at **20 random relabellings per graph**
through the production `NetworkXAdapter`. Both forms therefore satisfy Thm 2.12 within the
undirected class.

> **But they are different canonical forms, and the difference is large.** They agree on only
> **137 / 995 (13.8 %)**, and the pruned string is **longer on 558 / 995 (56.1 %)**. The module
> docstring says so (`canonical_pruned.py:22–26`) and it is right. **Substituting one for the other
> changes every Claim A bit count**, so the reference arm cannot be chosen for convenience.

Artifact: `scratchpad/verify_canonical.json` — **to be promoted into `tests/` before close**, per
[data](../plan/data.md) §6's rule that no measurement a decision rests on may live in a scratchpad.

### 1.4 Data — two plan claims corrected

| Claim | Plan file | Measured 2026-08-16 |
|---|---|---|
| "The raw IAM GXL tree is **absent** from this workstation" | [T-04 article notes](T-04-article-notes.md) §7 | **FALSE.** `APPROX_GED/datasets/IAM_Database/extracted` exists with **33,187 `.gxl` files**, and `exported_suite2/manifest.json`'s `_totals.iam_root` points at it. **T-04's "Suite 2 is no longer reproducible from source" is refuted**; the five Suite-2 Claim A rows *can* be re-derived |
| `cohort_audit.py` "can no longer re-derive the LINUX and AIDS-GraphEdX rows" | [data](../plan/data.md) §7, T-05 correction | **Stands** — the two-root path defect is real. T-06 owns the fix (§5.4) |

**The frozen cohort is `exported_suite2/`, not the GXL tree.** CSR edge lists, `graph_ids` aligned
**index-for-index** with the `LB`/`UB` matrix rows, per-file `sha256`, `n_pairs_present =
21,710,892`. T-06 loads from here and re-runs the GXL loader only for the reproduction check.

### 1.5 What already exists, and must not be rebuilt

| Need | Exists at | Verdict |
|---|---|---|
| **Graph-level cluster bootstrap** | `ged_bakeoff_analysis.py:1403 bootstrap_dataset`, `approx_ged_analysis.py:698 _bootstrap_slopes_full` (`np.bincount` multiplicity weights + `np.einsum`, avoids materialising resampled pair lists) | **REUSE.** Two working implementations with tests pinning their semantics |
| Mantel (D3) | `correlation_metrics.py:99 mantel_test` | **REUSE** — already permutes *graph* labels jointly on rows and columns |
| Holm (D8) | `correlation_metrics.py:428 holm_bonferroni` | REUSE |
| Competitor encode/bits/distance | `src/isalgraph/competitors/` — 11 backends, 6 metrics, 383 tests | REUSE |
| Cohort loader | `exported_suite2/*.npz` | REUSE |
| **Pair-level** bootstrap | `correlation_metrics.py:174 bootstrap_correlation` | **DO NOT USE.** Resamples pairs (`:242`) — the exact defect R3.5c identified. `statistics.md` §11 already lists it as replaced, not supplemented |
| All-pairs Levenshtein | `levenshtein_computer.py:86` | **UNFIT AS-IS** — naive dense `N×N` loop, no chunking. Needs a sharded driver for 21.7 M |
| BH-FDR over a declared family | — | **ABSENT** — T-06 builds it |
| F0/F1/F2 family runner | — | **ABSENT** — T-06 builds it |

---

## 2. Decisions taken with the PI, 2026-08-16

| # | Question | Decision |
|---|---|---|
| **1** | Reference arm: `pruned` or `canonical`? | **Deferred pending §1.6's timing probe.** The premise behind "Suite 2 must use pruned" is `competitors/README` finding 5, whose source ticket recorded its figures under a **pure-Python engine**. Re-measuring with `engine() == 'cpp'` before choosing |
| **2** | The frozen family has no term for a representation computable on one suite only | **New pre-declared term `s`.** `N_actual(F2) = 182 − 15k − 8d − 20s`. Entered in [preregistration](../plan/preregistration.md) §8 **before any p-value exists** |
| **3** | Deadline | **No extension. 2026-08-31 stands.** A costed reduction is required *before* work starts — §4 |
| **4** | Labels tier (S-d) | **Tiers 0–1 committed, Tier 2 DECLINED.** See §2.1 |

### 2.1 The Tier-2 cost in `labels.md` is wrong by ~3,900×, and that is why it is declined

[labels](../plan/labels.md) §4 costs logging L1–L3 at **"≈ 0.3 core-hours"**. That figure derives
from the same ~100 µs/pair rate that [approx_ged](../plan/approx_ged.md) carried and **T-05 proved
wrong by ~3,750×**. Re-estimated from T-05's *realised* cost (≈ 2,140 core-h for three full-cohort
roles over 21,710,892 pairs ⇒ ≈ 713 core-h per role):

| | pairs | roles | ≈ core-h |
|---|---:|---:|---:|
| Labelled Suite-2 datasets (GREC, AIDS-IAM, COIL-DEL, Mutagenicity, Protein) | 17,773,306 | LB + UB | **≈ 1,170** |

**PI decision: declined.** No reviewer asked for Tier 2; `labels.md` says so itself. **The 0.3
core-h figure is corrected in place in `labels.md` at close** — leaving it would let a later ticket
re-propose Tier 2 on a cost that is off by three orders of magnitude.

### 2.2 The `s` term — exact statement, frozen before any p-value

> **`s` counts representations computable on Suite 1 but not on Suite 2.** Such a representation
> **loses its 10 A1 rows and its 10 B1a rows** (Suite-2 datasets), and **keeps its 5 B1e rows and its
> Suite-1 A1 rows**. Hence −20 per representation.
>
> `N_actual(F2) = 182 − 15·k − 8·d − 20·s`
>
> **Membership is decided by a pre-declared computability criterion, never by ρ**: a representation
> is Suite-2-computable iff it completes on **≥ 99 % of the graphs of every Suite-2 dataset** under
> the frozen per-graph budget. The threshold and the budget are fixed here, before the encoding
> campaign runs.

**Expected members, on T-04's measured ceilings** (`agm_cam` 76 % GREC / 82 % AIDS-IAM;
`isalgraph_canonical` unusable on Suite 2 — **pending §1.6's re-measurement under the engine**). The
criterion is F5-blind: it reads a completion rate, not a correlation. This is the same
justification decision 24 already accepts for `k`.

---

## 3. Approach

Three sequential **milestones**, each independently useful to T-20, so T-20 can start writing before
T-06 finishes. This is the only structure that fits §4's window.

| | Milestone | Delivers | Needs T-04a? |
|---|---|---|---|
| **M1** | **Encodings + Claim A** — 11 representations × both suites, both bit conventions, D14 censoring table, encode-time curves | Tab. 2, Fig. 1, Fig. 2, the censoring headline, T-18's and T-13's inputs | **no** |
| **M2** | **Distance matrices + ρ** — Levenshtein and comparator distances, the size null beside every ρ, the equal-`n` view | Tab. 3, Fig. 3, **T-05's deferred §7.5 debt**, the AE.1 answer | **yes**, for comparators only |
| **M3** | **The frozen family** — F0, F1 (→ `d`), F2 with BH over `N_actual`, bootstrap CIs, MRM, Friedman/CD | §3.2, §4, R3.5a/b/c | consumes M2 |

**Rejected alternatives:**

- *One monolithic recompute script.* Rejected: nothing is deliverable until everything is, and T-20
  cannot start. It also makes a mid-run failure unrecoverable.
- *Wait for T-04a before starting.* Rejected: M1 and the whole statistics engine are independent of
  it, and the window does not permit idling.
- *Reuse `levenshtein_computer.py` as-is.* Rejected: naive dense `N×N`, no sharding. COIL-DEL alone
  is 7.6 M pairs.
- *Subsample Suite-2 pairs.* Rejected: [data](../plan/data.md) §4 establishes no subsampling is
  needed outside D15's within-replicate rule, and subsampling the matrices would break the join
  against T-05's `graph_ids`.

---

## 4. The window, and the costed reduction — **PI: no extension**

**Today 2026-08-16. Due 2026-08-31 = 15 days.** Board budget for what remains on the critical path:
T-06 (10–14) → T-20 (5–7) → T-15 (2) → T-24 (1) = **18–24 days serial in a 15-day window.**

**It does not fit, and no reduction inside T-06 alone closes it** — the same finding
[schedule](../plan/schedule.md) recorded for option C. What T-06 can do is **overlap**, and cut the
items whose absence T-20 can write around.

### 4.1 Load-bearing — not cuttable

| Item | Why |
|---|---|
| Claim A bits, all representations, both conventions | **AE.4a is requirement-modal**; R3.6a; R1.1. Cheapest high-value item in the ticket |
| **ρ(Lev, ·) per dataset** | T-05's deferred §7.5 debt. **Without it there is no results section** |
| **The size null beside every ρ** | T-04 finding 1 — the item that changes what the paper may assert |
| **Equal-`n` view as primary** | T-04 §2 — where the canonical/non-canonical gap is 0.42 and the claim is defensible |
| **Graph-level bootstrap CIs** | T-04 §1 made these a **precondition for printing any ρ** |
| F0 + F1 gates | F1 sets `d`, which the frozen family needs |
| D14 censoring table | T-05: "a headline result of T-06, not a footnote" |
| Pair-accounting ladder | R3.5a, and cheap from matrices that already exist |
| AIDS density stratification | `statistics.md` §8 — **can refute `conclusion.tex:30–36`**, and better found by us |

### 4.2 Cut, deferred or descriptive — with what it costs us

| Cut | Returns | What we lose | Mitigation |
|---|---|---|---|
| **Tier 2 labels (L1–L3)** | ≈ 1,170 core-h + 0.5 d | round-2 insurance | PI-decided (§2.1). No reviewer asked |
| **D15 §5 rule 2's two subsample-validation arms** | ~0.5 d | the ratio-matched and structure-matched checks on tier 3 | **Report the tier assignment and state the validation was not run**, rather than run it and report nothing else. Pre-declared, so its absence is disclosed, not hidden |
| **Stratified analyses beyond AIDS-density and the D14 censoring strata** | ~1 d | symmetry / mean-degree strata | Already **exploratory by decision** (`preregistration` §6). Report the two that carry an argument |
| **`isalgraph_canonical` Suite-2 column** | ~1 d | — | Forced, not chosen — it is the `s` term (§2.2) |

### 4.3 What this still leaves for the PI

**The overrun is T-20's, not T-06's, and no T-06 cut fixes it.** The milestone structure buys T-20 a
start on day ~4 instead of day ~11. **If that is not enough, the decision is which *ticket* to cut,
and it belongs to the PI** — [schedule](../plan/schedule.md) §3 holds the cut order. Flagged here so
it is not discovered in week three.

---

## 5. Frozen before anything runs

Each rule below is committed with this note, before the run that produces the outcomes it selects
between.

| # | Rule | Value |
|---|---|---|
| **F-1** | Reference arm | **§1.6 decides, on measured cost under `engine() == 'cpp'`, then frozen** |
| **F-2** | Suite-2 computability threshold for the `s` term | **≥ 99 % of graphs per Suite-2 dataset** complete within the per-graph budget |
| **F-3** | Per-graph encoding budget (D14) | **300 s**, enforced by a **killed subprocess**, never `signal.setitimer` — CPython runs handlers between bytecode instructions, so `SIGALRM` stays queued through a native call and T-05 hung 25 min on one graph this way |
| **F-4** | D14 censored-graph handling | **Greedy-min fallback, graph retained and flagged**; complete-case sensitivity arm reported beside it. Never dropped |
| **F-5** | Bit conventions | **Both** reported for every method (entropy bound `L log₂|Σ|`; realised bytes). Primary named in the text |
| **F-6** | Seed | **42** everywhere |
| **F-7** | D15 tiers | **As frozen in `statistics.md` §5**, not recomputed at execution time |
| **F-8** | BH-FDR | Within each of F0, F1, F2 at **q = 0.05**; over **`N_actual`**; `N_max`, the exclusion list and a **BH-over-`N_max` sensitivity column** all printed |
| **F-9** | `N_actual` | **`182 − 15k − 8d − 20s`** (§2.2) |
| **F-10** | Bracket reporting | **No interpolation.** ρ against LB and UB separately; **absolute gap `UB − LB` leads**, relative width beside it with its denominator named |
| **F-11** | Every printed ρ carries | its **graph-level bootstrap CI**, the **size null**, and its **enumeration window** |
| **F-12** | Suite-2 join | on **`graph_ids`**, never positionally (`aids_graphedx` is 819 in Suite 2 against Suite 1's 769) |
| **F-13** | Censored GED pairs | carry **`inf`, not NaN** — filter `np.isfinite`, select on `certified_mask` |
| **F-14** | WL depth | **`h = 2` ≡ `n_iter = 2`**. `wl_kernel_computer.py`'s `n_iter = 5` default is **`h = 5`** and is a defect to fix (T-04 §4) |

---

## 6. Acceptance criteria

Each names the command or artifact that proves it. **A ticket without these cannot be closed.**

| # | Criterion | Proof |
|---|---|---|
| **A1** | Every one of the 16,370 Suite-2 and 5,350 Suite-1 graphs has an encoding under every applicable representation, or is **flagged censored with its fallback string** | `encodings/manifest.json`: `n_graphs` per file equals the cohort count; `censored` count matches the D14 table |
| **A2** | Claim A reports **both** bit conventions for **every** method | `claim_a.json` has `entropy_bits` and `realised_bits` non-null for all 6 serialisations; `wl_subtree`/`size_null` explicitly `BitCountUndefined` with the reason printed |
| **A3** | The D14 censoring rate is measured at the **frozen 300 s budget** with a killed subprocess, per dataset and per symmetry stratum | `censoring.json`; the runner contains no `signal.setitimer` |
| **A4** | Distance matrices join T-05's on `graph_ids`, and every matrix is symmetric, zero-diagonal, finite | `gate_T06_structural.json`, 0 violations |
| **A5** | **Every printed ρ carries a graph-level bootstrap CI and the size null** | `rho_table.json`: no row with `ci_low`/`ci_high`/`null_rho` null |
| **A6** | F0 and F1 run under BH at q = 0.05 and **`d` is determined by F1's pre-declared rule**, not chosen | `family_F0.json`, `family_F1.json`; `d` derived in code from the CI/threshold rule |
| **A7** | `N_actual` is computed from `182 − 15k − 8d − 20s` with `k`, `d`, `s` each traceable to the rule that set it, and the **BH-over-`N_max` sensitivity column is printed** | `family_F2.json` carries `k`, `d`, `s`, `N_actual`, `N_max`, and both threshold columns |
| **A8** | The bootstrap resamples **graphs**, never pairs | a test asserts the resampling unit; `correlation_metrics.bootstrap_correlation` is not imported by any T-06 module |
| **A9** | The pair-accounting ladder is emitted per dataset: `raw → connected → GED-available → GED > 0 → Lev > 0 → analysed` | `ladder.json`, 10 rows |
| **A10** | Full suite green, ruff clean, `mypy --strict` clean; nothing in `scratchpad/` | `$PY -m pytest tests/ -q`; ≥ 2,106 passed (T-04's reference state) |
| **A11** | Every number that contradicts a plan file is **written into that plan file**, not only into this log | `review-close` §3 |

---

## 7. Stop-and-ask conditions

I halt and bring a **diagnosed** problem with costed options, rather than proceed:

1. **F0 fails on ≥ 3 of 5 Suite-1 datasets.** The pre-declared branch makes exact GED primary and the
   whole large-`n` extension descriptive — that changes what the paper claims, and AE.1 is the
   Area Editor's first demand.
2. **D4's β₁ collapses.** `statistics.md` §6: the correlation was largely size agreement and **Claim B
   must be restated**. Given T-04's size-null finding this is a live possibility, not a formality.
3. **The cohort does not reproduce** — any count other than 16,370 graphs / 21,710,892 pairs / 5,350
   Suite-1 graphs / 3,897,911 pairs.
4. **A T-05 or T-03 matrix fails its structural gate**, or a `graph_ids` join is not exact.
5. **`d ≥ 5`** — half the Suite-2 datasets having an uninformative bracket would gut F2, and the
   remedy is a scope decision, not a technical one.
6. **Any measured compute above ~5,000 core-hours** not already in this note.
7. **A second failed iteration round with an agent.**

---

## 8. Log

| Date | Entry |
|---|---|
| 2026-08-16 | Opened. Plan read; state measured (§1). PI decided items 2–4 of §2; item 1 deferred to the §1.6 probe. Two plan claims corrected by measurement: the IAM GXL tree is **present**, and both canonical forms are **verified complete invariants** that agree on only 13.8 % of classes |
