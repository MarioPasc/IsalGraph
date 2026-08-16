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

### 1.6 The reference arm, decided by measurement under the engine

The PI declined to choose an arm on T-04's figures, because T-04's own reproduction header records
its Picasso numbers as coming from a **pure-Python engine**. Re-measured 2026-08-16 with
`engine() == "cpp"`, **15 uniformly random graphs per dataset, seed 42, each encode in its own
subprocess killed at 30 s**:

| dataset | `n` range | `pruned` median | `pruned` max | `pruned` killed | `canonical` median | `canonical` max | `canonical` killed |
|---|---|---:|---:|---:|---:|---:|---:|
| iam_letter_low | 2–7 | 0.86 ms | 1.27 ms | 0/15 | 0.83 ms | 1.14 ms | 0/15 |
| iam_letter_med | 2–8 | 0.79 ms | 1.26 ms | 0/15 | 0.80 ms | 1.28 ms | 0/15 |
| iam_letter_high | 2–9 | 0.99 ms | 1.81 ms | 0/15 | 0.85 ms | 2.17 ms | 0/15 |
| linux | 4–10 | 3.07 ms | 4.91 ms | 0/15 | 4.24 ms | 13.5 ms | 0/15 |
| aids_graphedx | 2–20 | 4.02 ms | 23.0 ms | 0/15 | 6.45 ms | 99.5 ms | 0/15 |
| grec | 4–24 | 3.58 ms | 15.4 ms | 0/15 | 5.18 ms | 31.6 ms | 0/15 |
| aids_iam | 2–85 | 3.65 ms | 6.56 ms | 0/15 | 5.46 ms | 17.7 ms | 0/15 |
| **coil_del** | 3–77 | 18.2 ms | 0.58 s | **0/15** | 117 ms | 7.37 s | **7/15** |
| **mutagenicity** | 4–98 | 126 ms | **28.3 s** | **0/15** | 1.47 s | 24.1 s | **3/15** |
| **protein** | 2–96 | 305 ms | 1.95 s | **0/15** | 2.07 s | 21.8 s | **10/15** |

> ## 🔴 RETRACTED 2026-08-16 (same day, by the production run) — **every absolute time in the table
> above is inflated by cold-start warm-up and must not be quoted.** The kill counts are addressed
> separately below.
>
> The probe encoded **one graph per subprocess**, so *every* measurement was a **first call** into the
> C++ extension and paid one-time warm-up. The production driver encodes thousands per process and
> pays it once.
>
> **Isolated on a single graph** (`coil_del` index 375, `n = 59`), all three producing the
> **identical** 497-character string:
>
> | probe (1 graph / process) | re-timed, 1st call in process | production (amortised) |
> |---:|---:|---:|
> | **578.95 ms** | 19.39 ms | **4.95 ms** |
>
> And in the same re-timing process the *second* and *third* graphs took **0.70 ms** and **0.40 ms**
> against production's **0.72 ms** and **0.43 ms** — agreeing to within 3 %. **Only the first call in
> a process is inflated**, which is exactly the design of the probe.
>
> **What survives**: the strings — verified identical to the frozen `pruned_canonical_string`
> reference on the largest COIL-DEL graphs and on the probe's three slowest, `match=True` throughout.
> The probe was measuring the right computation, badly.
>
> **What does not**: every median and maximum in the table. `data.md` §4's encode-cost figures are
> **not** superseded by it. **Quote the production `seconds` arrays instead**, which are amortised and
> per-graph.
>
> **Lesson for Fig. 2**: a per-graph timing harness that spawns a process per graph measures
> interpreter and engine start-up, not the encoder. This is the same shape as
> `competitors/README` finding 11 (timings must be language-matched) — a measurement contaminated by
> its own harness.

**`pruned` completed 150 / 150. `canonical` was killed on 20 of 45** across the three largest
datasets. **T-04's ceiling finding survives the engine** — it was not a pure-Python artefact, and the
PI's challenge is answered in the direction T-04 claimed.

**Decision (F-1): `isalgraph_pruned` is the reference arm on both suites.** It is the only variant
computable across the cohort, so the choice is forced by measurement rather than taken for
convenience — and it is the *less* flattering option, since T-04's equal-`n` table has `canonical`
ahead on Letter LOW (0.9987 vs 0.9806) and HIGH (0.6953 vs 0.6166). `isalgraph_canonical` is
retained as a **Suite-1-only descriptive arm**, reporting the pruned-vs-exhaustive gap that
[preregistration](../plan/preregistration.md) §6 already labels exploratory.

> **Two things this probe does NOT establish, and neither may be quoted as if it did.**
> (a) **It is not a 300 s censoring rate.** 15 graphs at a 30 s budget bound the tail; they do not
> measure it. Mutagenicity's `pruned` **maximum of 28.3 s** sits just under the probe budget, so a
> 4,040-graph cohort will contain graphs that exceed it. The full-cohort rate at 300 s is what the
> encoding campaign measures, and it is a D14 deliverable.
> (b) **It is not a per-pair or per-graph cost for publication.** Fig. 2's timings must be
> language-matched (`competitors/README` finding 11) and measured with no concurrent writer.

**This corrects `competitors/README` finding 7b**, which reports `pruned` failing 24/400 on
Mutagenicity and 4/400 on Protein and concludes *"an earlier note said `pruned` was fine to n = 98;
on real graphs it is not."* Those failures are at a **2 s** budget. At 30 s the same encoder is
0/150. **The finding is a statement about the budget, not about the encoder**, and the plan must not
carry it as the latter.

### 1.4 Data — two plan claims corrected

| Claim | Plan file | Measured 2026-08-16 |
|---|---|---|
| "The raw IAM GXL tree is **absent** from this workstation" | [T-04 article notes](T-04-article-notes.md) §7 | **FALSE.** `APPROX_GED/datasets/IAM_Database/extracted` exists with **33,187 `.gxl` files**, and `exported_suite2/manifest.json`'s `_totals.iam_root` points at it. **T-04's "Suite 2 is no longer reproducible from source" is refuted**; the five Suite-2 Claim A rows *can* be re-derived |
| `cohort_audit.py` "can no longer re-derive the LINUX and AIDS-GraphEdX rows" | [data](../plan/data.md) §7, T-05 correction | **Stands** — the two-root path defect is real. T-06 owns the fix (§5.4) |

**The frozen cohort is `exported_suite2/`, not the GXL tree.** CSR edge lists, `graph_ids` aligned
**index-for-index** with the `LB`/`UB` matrix rows, per-file `sha256`, `n_pairs_present =
21,710,892`. T-06 loads from here and re-runs the GXL loader only for the reproduction check.

### 1.4b 🔴 Two orchestrator sessions share one checkout, and every worktree reads it

**Measured 2026-08-16, and it changes how the wave is built.**

T-04a's session and this one were both operating in `/home/mpascual/research/code/IsalGraph`.
T-04a ran `git switch main → ticket/T-04a` (reflog `HEAD@{2}`); this session did not observe it and
committed its design note on top of T-04a's `7e96f4a`. Nothing was lost — `main` never moved from
`f7ad283` — but the commit was misplaced.

**Resolution**: T-06 works from a **dedicated worktree**,
`/home/mpascual/research/code/IsalGraph-T06` on branch `ticket/T-06`, cut from `f7ad283`. The design
commit was cherry-picked there (`8afa59e`). **The shared checkout is T-04a's; T-06 does not
`git switch`, commit or edit in it.**

**The consequence that actually matters.** Measured from the T-06 worktree:

```
isalgraph.__file__              -> /home/mpascual/research/code/IsalGraph/src/isalgraph/__init__.py
isalgraph.competitors.__file__  -> /home/mpascual/research/code/IsalGraph/src/isalgraph/competitors/__init__.py
```

The `scikit-build-core` editable finder is **path-pinned to the checkout it was installed from**, so
it outranks the worktree and `PYTHONPATH` alike. Every worktree agent therefore imports the *shared*
checkout's `src/` — which is **T-04a's branch, carrying T-04a's in-flight edits to
`src/isalgraph/competitors/`.**

**Rules this forces on the wave, restated in every spawn prompt:**

1. **No agent edits `src/isalgraph/` at all.** `competitors/` is T-04a's live territory.
2. **No agent asserts a numeric value produced by a competitor backend.** Tests assert *API shape*
   and invariants (`bits.count` returns both conventions; `distance` is symmetric and zero on
   identical input), never a specific bit count or ρ — those can change under the agent with no error
   raised.
3. **No agent runs a production encoding campaign.** Agents build and validate machinery on
   synthetic fixtures; **the orchestrator runs production after T-04a merges**, with the shared
   checkout parked on a known commit.
4. **No agent reports a timing.** Three concurrent agents on one workstation contaminate every
   measurement. Timings are the orchestrator's, measured alone.
5. **Every output file records `isalgraph.build_info()['build_hash']` and the producing checkout's
   `git rev-parse HEAD`**, so a contaminated run is detectable afterwards rather than silently
   believed.

*Not* fixed by installing a second env from the T-06 worktree: that costs a C++ rebuild and a new
environment to remove a hazard that disappears when T-04a merges, inside a 15-day window.
**Detection beats isolation here — but the detection is not optional.**

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
| **1** | Reference arm: `pruned` or `canonical`? | **RESOLVED by measurement (§1.6): `isalgraph_pruned` on both suites.** The PI required the premise be re-checked under the C++ engine before choosing. It was, and it holds |
| **2** | The frozen family has no term for a representation computable on some datasets and not others | **New pre-declared term `c`, charged PER CELL.** `N_actual(F2) = 182 − 15k − 8d − c`, `N_actual` defined by enumeration. Entered in [preregistration](../plan/preregistration.md) §5.1/§5.2/§8 **before any p-value exists**. A first per-representation draft (`−20s`) was replaced the same day — see §2.3 |
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

### 2.2 The `c` term — exact statement, frozen before any p-value

Full text and precedence: [preregistration](../plan/preregistration.md) §5.1–§5.2. In brief:

> **`c` counts individual F2 cells, not representations.** A cell is `(row, representation,
> dataset)` for `row ∈ {A1, B1e, B1a}`, removed iff its representation **fails the computability
> criterion on that cell's dataset**: fewer than **99 %** of that dataset's graphs encode within the
> frozen **300 s** per-graph budget.
>
> **`N_actual(F2)` is defined by enumerating the admissible cell set**, with
> `182 − 15k − 8d − c` printed beside it as a check. Precedence `k → d → c`; a cell removed by an
> earlier term is never charged again.

The IsalGraph reference arm is **never charged to `c`** — D14 governs it, retaining a censored graph
with its greedy-min string and reporting the rate. `wl_subtree` has no A1 cell, so a WL failure costs
1 per dataset, not 2.

### 2.3 That term was wrong when first frozen, and was corrected the same day

The first draft charged **−20 per representation** (`s`). The T-04a session challenged it; the
challenge was **verified against the sources and upheld**. Three defects, recorded because the
correction is the kind a reviewer would otherwise find:

1. **The arithmetic over-charged, in the anti-conservative direction.** `competitors/README` finding
   5 measures `agm_cam` failing on **five** Suite-2 datasets (GREC 24 %, AIDS-IAM 18 %, COIL-DEL
   46 %, Protein 90 %, Mutagenicity 98 %) and completing at **100 %** on Letter ×3 and LINUX. An
   all-or-nothing gate deletes ~10 cells AGM does deliver — and **shrinking `N_actual` below what the
   data forces lowers the BH burden on every surviving test**. That is the reduction a reviewer
   attacks hardest, and it flatters us.
2. **The citation was wrong.** The reduction-rule hole is `competitors/README` finding **6**; finding
   **4** is the sparse6 `m/n` inversion. The "4" came from the *board header's* list, which numbers
   differently. Verified in the source before correcting.
3. **`k` and `s` were both ranged 0–6** although Claim B has **seven** comparators. `k` is now 0–7.

Added with the fix: the `k → d → c` precedence rule (three terms over one shared cell set can
double-count), the IsalGraph exemption, and the statement that **another ticket's completion rate at
its own budget is not a `c` determination** — T-04/T-04a measure at each backend's native budget
(`agm_cam` → `AGMBudgetExceeded`; `min_dfs` → `max_projections = 50,000`; `isalgraph_pruned` →
`CanonicalizationTimeoutError` at **2 s**), not at 300 s.

**No p-value existed under either version**, recorded in the §8 changelog as the rule requires.

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
| **F-9** | `N_actual` | **defined by enumeration of the admissible cell set**; closed form `182 − 15k − 8d − c` printed as a check; precedence `k → d → c`, no double-counting (`preregistration` §5.2) |
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

---

## 9. Inherited finding, 2026-08-16 — the Suite-2 reference is size-dominated

Reported by the T-04a session; recorded here because it bears on **70 of F2's 182 cells** (B1a).

**Measured** (T-04a, seed-42 200-graph draw per dataset), `ρ(|n₁−n₂|, bound)`:

| | GREC | AIDS-IAM | COIL-DEL | Mutagenicity | Protein |
|---|---:|---:|---:|---:|---:|
| **LB** (`BRANCH_FAST`) | 0.9803 | 0.9600 | **0.9978** | 0.9876 | 0.9699 |
| **UB** (`BIPARTITE`) | 0.7061 | 0.7117 | 0.7362 | 0.7538 | 0.4596 |

**Against ground truth**, on the four cohorts T-05's `PROVENANCE` certifies identical between suites,
same certified pairs:

| | Letter LOW | Letter MED | Letter HIGH | LINUX |
|---|---:|---:|---:|---:|
| ρ(\|Δn\|, **UB**) | 0.7482 | 0.7363 | 0.7080 | 0.3479 |
| ρ(\|Δn\|, **exact**) | 0.9139 | 0.9146 | 0.9195 | 0.7134 |
| ρ(\|Δn\|, **LB**) | 0.9804 | 0.9740 | 0.9224 | 0.8838 |

### How this must be stated, and how it must not

**Not "the lower bound is broken."** `|n₁−n₂|` *is itself a valid lower bound on GED under D6* —
every surplus node costs one insertion — so a lower-bound method correlating with size is
mechanically expected, and `BRANCH_FAST` is a proven bound that T-27 selected on measurement over
3.8 M certified pairs. Writing it as a defect reads as an attack on decision 11, which it is not.

**And not "the size channel is an artefact of the bound" either.** `ρ(|Δn|, exact)` is **0.91–0.92**
on the three Letter sets. **The truth is size-dominated too.** The size channel is a real component
of GED at these sizes, not something the bracket invented.

**What the measurement does establish**, and it is narrower than either of those: `UB < exact < LB`
in **4 of 4** — the bracket holds, but each arm's size-dependence is biased in a **fixed direction**,
the LB overstating and the UB understating, and **neither reproduces the truth's own
size-dependence.** So a conclusion drawn from the LB arm alone is not a conclusion about GED.

### Three consequences

1. **`approx_ged.md` §4's no-interpolation rule gains a measurement.** Its current justification is an
   argument from ignorance — *"we do not know where in the bracket the truth lies"*. It is now
   stronger: the two arms disagree **systematically**, not merely noisily, so a midpoint would
   inherit both biases with the direction unknown. → propagate at close; **T-20**.
2. **`d` is likely to come in high.** F1's statistic is `ρ(Lev, LB) − ρ(Lev, UB)`, and the two
   references differ this much in what they correlate with. **§7 stop-and-ask condition 5 (`d ≥ 5`)
   is now a live expectation rather than a remote one**, and each uninformative dataset removes 8 F2
   cells.
3. **The equal-`n` view is where the structural claim is defensible on Suite 2** — it removes the
   size channel from both sides at once. This was already load-bearing in §4.1 on T-04's finding 1;
   this measurement is independent support.

### What does NOT change, and why

**B1a's pair population stays as frozen: all pairs** — but *not* for the reason first given here.

> ## ⚠ CORRECTED 2026-08-16 — the cancellation argument was WRONG. The conclusion survives on
> the other two reasons; what B1a may be **claimed to show** does not.
>
> This section originally led with: *"B1a is a difference of two correlations against the same
> reference, so a size component inflating both arms partly cancels."* **That is false, and it was
> the load-bearing half.** Refuted by the T-04a session, analytically and empirically, and the
> refutation holds.
>
> **Analytically**: a correlation is **not additive in its components**, so there is no offset to
> cancel. Spearman is rank-based, so for strictly monotone `f`, `ρ(X, f(|Δn|)) = ρ(X, |Δn|)`
> *exactly*. With `R ≈ f(|Δn|)` — and `ρ(|Δn|, LB)` is **0.96–0.998** — it follows that
> `ρ(X,R) − ρ(Y,R) ≈ ρ(X,|Δn|) − ρ(Y,|Δn|)`. The difference does not *remove* the size
> channel; it makes the test almost entirely **about** it, reweighted by each representation's own
> size-sensitivity — which is exactly the axis B1a compares.
>
> **Empirically**, from `corrected_rho_table.json`, Kendall τ between the all-pairs ranking and the
> equal-`n` ranking. **Under a pure additive offset τ would be 1.00 with zero inversions:**
>
> | dataset | reps | τ(all-pairs, equal-`n`) | inversions vs `isalgraph_pruned` |
> |---|---:|---:|---:|
> | AIDS | 9 | **−0.111** | 6 |
> | LINUX | 10 | 0.067 | 4 |
> | Letter MED | 10 | 0.111 | 5 |
> | Letter HIGH | 10 | 0.200 | 3 |
> | Letter LOW | 10 | 0.467 | 2 |
>
> Representations **cross** rather than shift together: `adjacency` 0.756 → 0.157 and `nauty_graph6`
> 0.425 → 0.137 on AIDS; `nauty_graph6` 0.646 → **0.952** on Letter MED. And this is measured against
> **exact** GED (size-dependence 0.71–0.92); B1a's reference is the **LB** at 0.96–0.998, so the
> effect there is **larger**, not smaller.
>
> **The inversions cut both ways** — on Letter LOW the equal-`n` view lifts `agm_cam`
> (0.918 → 0.994) and `sparse6_nauty` (0.638 → 0.981) *above* IsalGraph. This is not a view that
> flatters us.

**The two surviving reasons, which are sufficient:**

- **Moving a frozen family's population after seeing a measurement is what pre-registration exists to
  prevent.** The measurement is F5-blind — `ρ(|Δn|, LB)` never touches Levenshtein — so a change
  would be *defensible*, but "defensible" is a lower bar than "not required".
- The equal-`n` view **is computed and reported regardless**, at the cost of a mask over matrices we
  already hold.

### 9.1 What the refutation DOES change — three claim-scoping rules, frozen

The finding does not change what B1a is **computed on**. It changes what B1a may be **claimed to
show**, and that is a constraint on T-20 as much as on T-06.

1. **B1a all-pairs may NOT be reported as "which representation is the better GED proxy."** It is
   *which representation better tracks the reference, whose dominant component at these sizes is
   graph size*. **That sentence goes in the text, beside the table.**
2. **The equal-`n` companion is not a robustness check.** A robustness check returning τ = −0.111
   against the primary confirms nothing. **Both views get equal prominence, and where they disagree,
   the disagreement IS the result** — not a footnote, and never an average of the two.
3. **If Claim B rests on B1a's all-pairs outcome and the companion inverts it, that is the finding**
   and it is reported as such. Written down **now, before the family runs**, so the framing cannot be
   drafted around whichever view survives.

Neither view is the true one. They answer different questions, and the paper states which question
each answers.

---

## 10. ⚠ The size-null verdict INVERTS across the bracket on 5 of 5 Suite-2 datasets

Reported by the T-04a session (F5 arm, 200-graph seed-42 draw per dataset, all-pairs, each null
computed against **the same reference** as the representation). This supersedes the forecast in §9's
consequence 2 and is the sharpest result the ticket has produced.

`ρ(Lev, ·) − ρ(|Δn|, ·)`:

| dataset | vs **LB** (`BRANCH_FAST`) | vs **UB** (`BIPARTITE`) | |
|---|---:|---:|---|
| GREC | −0.214 | **+0.122** | flips |
| AIDS-IAM | −0.248 | **+0.027** | flips |
| COIL-DEL | −0.082 | **+0.197** | flips |
| Mutagenicity | −0.295 | **+0.078** | flips |
| Protein | −0.233 | **+0.383** | flips |

**Against the proven lower bound IsalGraph fails the size null on every Suite-2 dataset; against the
proven upper bound it clears it on every one. GED lies between them, so neither answer is licensed.**

### Why — and it is mechanical, not empirical

`ρ(|Δn|, LB)` is **0.960–0.998** (§9). The lower bound very nearly *is* the size null, so
`ρ(X, LB) ≈ ρ(X, |Δn|)` for **any** `X`, and the null's own score against LB is ≈ 1. **No
representation can beat it. The comparison is degenerate by construction**, and a "failure" there is
a statement about `BRANCH_FAST`, not about IsalGraph. `ρ(|Δn|, UB)` is 0.460–0.754, which leaves
room. On Suite 1, where truth exists, `ρ(|Δn|, exact)` is **0.713–0.920 — between the two arms.**

### Status of the evidence — read before quoting

- On the **LB** side all five margins put the null outside IsalGraph's 95 % bootstrap CI.
- On the **UB** side **four** of five do. **AIDS-IAM's +0.027 sits inside [0.636, 0.818] and is within
  noise.** The honest form is *"4 of 5 flip with both ends individually significant; the fifth flips
  within noise."*
- ⚠ **This is a point-estimate-against-an-interval check, which is weaker than what D13 requires** —
  a **paired bootstrap CI of the difference on shared resamples** (D7). T-04a explicitly declined to
  synthesise that from these numbers, correctly. **Computing it is T-06's**, and it needs the real
  Levenshtein matrices this ticket is producing.

### Consequences

1. **`d` is no longer a tail risk; plan for it to be large.** D13 tests `ρ(Lev, LB) − ρ(Lev, UB)`;
   for IsalGraph that raw difference is **0.05–0.11**, straddling the pre-declared 0.05 threshold,
   and the *comparative* conclusion inverts on 5/5. **Plan for the bracket to be declared
   uninformative on most or all ten datasets, and treat a small `d` as the surprise.** At `d = 10`
   the reduction is 80 cells. **`N_actual` must be re-forecast before the PI is shown the old
   number**, and §7 stop-and-ask condition 5 (`d ≥ 5`) should be expected to fire rather than
   guarded against.
2. **This is the empirical core of the no-interpolation rule** — stronger than §9's bias table. A
   midpoint would have produced a *single confident answer* to a question the data leaves open, on
   all five datasets.
3. **T-20 may write neither "IsalGraph clears the size null on the larger datasets" nor its
   negation.** The sentence that survives: *against the proven lower bound it does not clear the null
   on any Suite-2 dataset; against the proven upper bound it clears it on all five; and since GED
   lies between them the comparison is undetermined at these sizes.* **That is a publishable and
   honest result, and a better answer to AE.1 than a number would be** — AE.1 asks for the size
   impact to be made clear, and "the reference degrades faster than the representation at these
   sizes" is exactly that.

**Suite 1 is unaffected**: exact GED, no bracket, IsalGraph clears the null on **1 of 5** (Letter LOW
+0.026; MED −0.044, HIGH −0.220, LINUX −0.239, AIDS −0.528) — reproducing T-04's corrected finding on
an independently written pipeline.

### A cross-validation gate this hands us for free

T-04a's F5 Suite-1 ρ values reproduce `corrected_rho_table.json` at **max |delta| = 0.0000** across
every shared cell — **two independently written pipelines, same seed-42 draw, bit-identical.**
**T-06's distance driver is checked against that same target** and any disagreement is a defect in
one of three implementations rather than a judgement call. Added as an acceptance criterion.

---

## 11. RESULT — the D14 censoring rate at the frozen 300 s budget, measured

**The full Suite-2 cohort, `isalgraph_pruned`, 300 s per graph, killed subprocess, C++ engine
(`build_hash 298fc1188bf1b051`), `src_commit 6c3e742`.** This is the measurement
[statistics](../plan/statistics.md) D14 called *"a headline result of T-06, not a footnote"*, and it
had never been made on the full cohort at the production budget.

| dataset | graphs | **censored** | rate | median length | solver-seconds | max s |
|---|---:|---:|---:|---:|---:|---:|
| iam_letter_low | 1,180 | 0 | 0.00 % | 4 | 0.1 | 0.0 |
| iam_letter_med | 1,253 | 0 | 0.00 % | 4 | 0.1 | 0.0 |
| iam_letter_high | 2,059 | 0 | 0.00 % | 8 | 0.1 | 0.0 |
| linux | 89 | 0 | 0.00 % | 13 | 0.0 | 0.0 |
| aids_graphedx | 819 | 0 | 0.00 % | 19 | 0.1 | 0.0 |
| grec | 650 | 0 | 0.00 % | 22 | 0.1 | 0.0 |
| aids_iam | 1,811 | 0 | 0.00 % | 19 | 43.4 | 5.9 |
| coil_del | 3,900 | 0 | 0.00 % | 115 | 2.7 | 0.0 |
| **mutagenicity** | 4,040 | **101** | **2.50 %** | 54 | **37,381.7** | 300.0 |
| protein | 569 | 0 | 0.00 % | 148 | 47.1 | 15.0 |
| **TOTAL** | **16,370** | **101** | **0.62 %** | | ≈ **10.4 core-h** | |

**Verified, not assumed**: every file's `graph_ids` matches the frozen cohort **element-wise**
(asserted, not spot-checked), every `G` equals the cohort count, and the **D14 invariant holds on all
ten datasets** — `status == "censored"` ⟹ `fallback_used` ∧ `encoding != ""` ∧ `length >= 0`. No
censored graph left without its greedy-min string.

### This corrects D14's premise, and D14 survives anyway

D14 says censoring is *"a bulk property of two or three datasets, not a marginal tail"*, from T-05's
probe: `protein` 5/10, `coil_del` 5/10, `mutagenicity` 1/10. **Measured at production settings:
`protein` 0/569, `coil_del` 0/3,900, `mutagenicity` 101/4,040.**

**This is not a contradiction of T-05 — it is a different configuration**, and both differences push
the same way. T-05 probed **`canonical_string`** (exhaustive) at a **15 s** budget; production runs
**`pruned_canonical_string`** at **300 s**. Either change alone would lower the rate. **What must not
happen is D14's prediction being quoted as if it described the production encoder.**

**D14's machinery is still required and still earns its place.** 101 graphs censor, and they are
exactly the hard ones — Mutagenicity is the dataset `data.md` §4 already identifies as
`|Aut|`-pathological (`n = 98`, density 0.021, *"does not finish in 5 minutes"* under the exhaustive
encoder). Dropping them would remove the hardest cases from the very dataset that carries the
scaling argument. The **complete-case sensitivity arm is not optional**: 101 of 4,040 graphs touch
roughly **5 % of Mutagenicity's 8,158,780 pairs**, so the primary and sensitivity arms can differ
materially there and nowhere else.

### What may be said, and what may not

- **May**: *"At a 300 s per-graph budget the pruned canonical encoder censors 2.50 % of Mutagenicity
  and 0.00 % of the other nine datasets, 0.62 % cohort-wide."*
- **May not**: any cohort-level censoring rate without naming Mutagenicity — the rate is **not** a
  cohort property, it is one dataset.
- **May not**: this as a rate for `canonical_string`, which is a different encoder, or at any other
  budget.
- **`seconds` is in-worker solver time at `--jobs 6`**, so it is a cost *floor*, not job consumption,
  and it is not a publishable per-graph timing. Mutagenicity's 37,382 s ≈ **10.4 core-hours** is the
  cohort's entire encoding cost to within rounding; every other dataset together is under 100 s.

### 11.1 F-1 re-verification — first attempt measured a scope guard, not a timeout

The first re-check ran `isalgraph_canonical` through the competitors registry and returned
`ok = 10 / 49 / 19` of 200 on protein / coil_del / mutagenicity. **Those are not timeouts.** Every
failure is `error_kind = SuiteScopeError`, the split is exactly at the boundary (`n(ok) max = 12`,
`n(err) min = 13`), and the counts equal `#{n <= 12}` in each sample precisely. Slowest surviving
encode: **21 ms**.

**The registry refuses `isalgraph_canonical` above `n = 12` by policy, before attempting an
encode.** A completion rate measured through it is a property of that guard, not of the encoder —
the same shape as the warm-up contamination it was sent to check, and the second harness-measures-
itself error of the day.

**This changes the F-1 question rather than answering it.** Two separable facts:

1. **Is `canonical_string` fast enough on Suite-2 graphs?** Open — re-measured by `f1_verify.py`,
   which calls `isalgraph.core.backends` directly and amortises warm-up across a band.
2. **Would the packaged backend produce Suite-2 columns even if it were?** **No.** `SUITE1_ONLY` is a
   frozen T-04 policy, not a performance outcome. Reversing it is a change to another closed
   ticket's decision, not a T-06 measurement call — **so even a fast result does not by itself
   reopen F-1; it would have to go to the PI.**

**And it vindicates T-04a's insistence** that `SuiteScopeError` be reported as a separate `error_kind`
from `AGMBudgetExceeded`. Summing them would have produced a "completion rate" here that was 100 %
scope policy and 0 % budget, read as if it were the latter — and `c` would have been determined by it.
