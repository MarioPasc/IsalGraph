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

#### 1.4b.1 The hazard is real but currently **inert** — measured 2026-08-23, and re-measurable in one command

T-04a has merged. The shared checkout is parked on `main` at `c1d36b1`, and the question the rules
above exist to manage — *is the `src/` we import the `src/` we think we are testing?* — now has a
direct answer rather than a mitigation:

```bash
git -C /home/mpascual/research/code/IsalGraph-T06 diff --stat c1d36b1..HEAD -- src/isalgraph/
# empty  ->  the shared checkout's src/ is EQUIVALENT to this branch's
git -C /home/mpascual/research/code/IsalGraph status --porcelain -- src/isalgraph/
# empty  ->  and it has no uncommitted edits on top
```

Both were **empty** on 2026-08-23. All 40 `ticket/T-06` commits touch only `.claude/notes/`,
`benchmarks/` and `tests/`. **So imports resolving to the shared checkout is currently harmless: the
production campaigns run the intended encoder.** Corroborated at run time — the campaign's own
banner records `engine: cpp, build 298fc1188bf1b051`, which equals the `isalgraph_build_hash` stored
in every wave-1 `.npz`.

**This is a check, not a conclusion.** The shared checkout belongs to another session and can move
under us at any moment; §1.4b's whole premise is that it already did once. So:

> **Re-run both commands immediately before every production campaign**, not once per ticket. Cost:
> two `git` invocations. Failure mode if skipped: a whole campaign silently encodes under another
> branch's `src/`, with nothing in the output to show it — the same class of silent-wrong-answer as
> the `error_kind` overloading rejected in §3.2, and detectable only by the `src_commit` field after
> the fact.

Rule 5 above remains the backstop and stays mandatory: recording `build_hash` and `src_commit`
detects contamination *afterwards*; this check prevents it *beforehand*. They are complementary, and
neither replaces the other.

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
| **A7** | `N_actual` is computed **by enumeration of the admissible cell set**, with `k`, `d`, `c` each traceable to the rule that set it, and the **BH-over-`N_max` sensitivity column is printed**. Closed form `182 − 15k − 8d + k·d − c` printed as a check only (`= 137 − 5d − c` at `k = 3`); **enumeration wins on disagreement** | `family_F2.json` carries `k`, `d`, `c`, `N_actual`, `N_max`, and both threshold columns |
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
| 2026-08-23 | Wave 1 verified against its artifacts by the orchestrator, not by report. Cohort gate **passes**: 5,350 Suite-1 under both representations, 16,370 Suite-2 under `pruned`. D14 censoring **101 graphs, all Mutagenicity** (2.50 % of it, 0.617 % of the cohort), zero elsewhere. Provenance uniform: `build_hash 298fc1188bf1b051` — **equal to the live engine** — `encode_budget_s = 300.0`, `seed = 42`, `schema t06.1`. The two distinct `src_commit` values are **benign**: the diff between them touches only `src/isalgraph/competitors/`, leaving `core/` and the encoding driver untouched |
| 2026-08-23 | **F-1 closed** (§11.4). Reference arm `isalgraph_pruned`, justified by the `SUITE1_ONLY` scope policy, **not by cost**. Kill counts **NOT MEASURED** after four diagnosed instrument failures. `43 s/graph`, `≈ 520×` and `≥ 6.8 core-hours` **retracted as unprovenanced**. No relaunch |
| 2026-08-23 | **Date correction.** A handoff asserted the current date was 2026-08-17 and asked that `2026-08-23` stamps be rewritten as false provenance. **They are correct provenance and were kept.** Independently confirmed: system clock, HEAD `ed77037` committed `2026-08-23T11:24:33+02:00`, and §11.3's own recorded `2026-08-23T09:22:53Z`. Deadline distance is **8 days, not 14** |

---

### 8.1 One recurring failure mode, named once — reductions that shrink the BH denominator

Five instances to date, in five places, all in the **same direction**: every one *lowers* `N_actual`
and therefore *weakens* the BH correction on every surviving test. None ever erred the other way.
That asymmetry is the tell — a transcription slip is direction-neutral, a motivated one is not.

| # | Where | The reduction |
|---|---|---|
| 1 | the `s` rule | dropped cells the data did not license dropping |
| 2 | closed-form arithmetic | `+k·d` omitted |
| 3 | a cancellation argument | cancelled a term that does not cancel |
| 4 | `tickets.md:158` | `182 − 15k − 8d` → under-counts by `3d` |
| 5 | **this note, §6/A7** | `182 − 15k − 8d − 20s` — a *fourth* variant, carrying the retired `s` term |

**Rule, adopted here:** `preregistration.md` §5 is the sole authority for the closed form
(`182 − 15k − 8d + k·d − c` = `137 − 5d − c` at `k = 3`), **enumeration is authoritative over any
closed form**, and any proposed change that shrinks `N_actual` is treated as a defect until
re-derived from the admissible cell set. A7 above is corrected accordingly.

> ### ✅ The CODE was never wrong. Verified 2026-08-23.
>
> All five instances above are **prose**. `eval_stats/family.py` has carried the correct form the
> whole time:
>
> ```python
> return N_MAX_F2 - 15 * k - 8 * d + k * d - c      # family.py:392
> ```
>
> Executed: `enumerate_f2_cells()` returns **182 cells**, equal to `N_MAX_F2`; `_closed_form(3, d, c)`
> equals `137 − 5d − c` for every `d ∈ [0,5] × c ∈ [0,3]` tested. Its own docstring already says the
> enumeration is the definition and the closed form only a check.
>
> **This changes what the failure mode is.** It is not an arithmetic error that reached the
> statistics — no p-value was ever computed against a shrunken denominator. It is **documentation
> drift**: five prose restatements of one formula, each dropping a term, none of them the thing that
> runs. The risk was never a wrong number in `family_F2.json`; it was that a reader — a reviewer, or
> the next agent — would trust the prose over the code and "correct" the code to match. Two documents
> had already converged on the same wrong form, which is exactly how that correction gets made
> confidently.
>
> **So the mitigation is documentation-shaped, not code-shaped:** cite `family.py:_closed_form` as
> the reference rather than restating the formula, and where a document must state it, state it
> beside the enumeration that outranks it.

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

### 11.2 🔴 The second attempt never ran either — two defects in `f1_verify.py`, 2026-08-23

The re-measurement §11.1 promised was launched on 2026-08-16 and left a **zero-byte log and no
result**. It was not slow, and it was not killed: it could not have terminated. Two independent
defects, found by `[T06-subagent]` on 2026-08-23 and fixed in `7274f13`.

1. **The worker was given a range, not the sample.** The parent passed `idx[pos] .. idx[-1] + 1`, so
   the worker encoded every graph in the contiguous **span** of the 200-graph sample rather than the
   sample itself — **2.8×** the intended work on protein, **19.3×** on coil_del, **19.7×** on
   mutagenicity. At a 60 s kill budget that turns a ~4.5 h run into a multi-day one. The reported
   `attempted` was `len(idx)` while `ok` and `killed` ranged over the whole span, so the printed
   kill rate could exceed 100 % — the same arithmetic inconsistency that would have made the defect
   visible had anyone read a completed row.

2. **The deadline could never fire.** `proc.stdout.readline()` blocks, and the loop only re-checked
   `time.monotonic() > deadline` *after* a line arrived. Once a graph ran past the budget no line
   ever arrived, the check was never reached, and the parent blocked **indefinitely** instead of
   killing. **The wall-clock kill this whole instrument exists to apply had never once executed.**
   Reading now happens on a reader thread feeding a queue the parent polls with a timeout.

Both are the same species as the two errors §11.1 already records: **the harness measuring itself**.
The probe measured warm-up, the registry measured a scope guard, and this measured neither because
it could not finish. Three in a row on one question is worth naming as a pattern rather than three
accidents — a timing instrument needs its own verification before its output is read.

**Verification before relaunch**, which is what the first two attempts lacked: at a 0.002 s budget
every sampled graph is killed, the parent recovers past each culprit, and coverage is exact — 6 of 6,
no strays. At a 5 s budget on mutagenicity, 8 of 8 encode including **n = 35**, no strays, no gaps.

**Budget caveat, to be quoted with any result.** The kill budget is **60 s** while D14's frozen
encoding budget is **300 s**, so a kill count here is an **upper bound** on the 300 s kill count.
`max_ok_s` is recorded so the gap can be judged rather than assumed: if no survivor approaches 60 s,
the cut is clean and a 5× budget would change little.

~~**Status: running.**~~ Relaunched 2026-08-23 with provenance recorded in the result
(`engine`, `build_hash`, `src_commit = c1d36b1`) and a refusal to measure on any engine but `cpp`.
The report is written after **every cell**, so a mid-run kill now leaves a usable partial record
rather than nothing.

> ## ⚠ SUPERSEDED 2026-08-23 — **the third attempt died too.** See §11.4.
>
> Verified at `2026-08-23T14:13Z`: no `f1_verify` or `f1_worker` process (`pgrep -a -f
> 'f1_ver[i]fy'`, bracketed so the pattern cannot self-match); `f1_verify.log` present but **zero
> data rows**; `f1_idx_protein.txt` (765 B) shows it reached protein and stopped; no
> `f1_verify.json` anywhere on the box — worktree, shared checkout, `/tmp`, `$HOME`, and both
> `Sandisk2TB` trees were searched. **No cell ever completed. Quote nothing from this run.**

> ⚠ **`ps` on this box is proxied through `rtk` and returned empty output for a process that was
> demonstrably alive**, which briefly produced a false "the run died" diagnosis here. `pgrep -a`
> reported it correctly. Same class as the stale `git log` head T-04a hit — **trust `pgrep -a` and
> `git rev-parse`, not `ps` or `git log`.**
>
> **Amended 2026-08-23.** `pgrep -a -f <name>` has the *opposite* failure mode and it fired twice
> in one session, in both directions: the shell wrapper carries the pattern in its own command
> line, so a bare `pgrep -a -f f1_verify` **always matches itself** and reads as "still running"
> when nothing is. Both sessions on this box made that call. **Use `pgrep -a -f 'f1_ver[i]fy'`** —
> the bracket matches the real process and cannot match the literal pattern text — **and confirm a
> claimed run against its artifact, never against a process list alone.** A run that is alive but
> has produced no output for hours is not meaningfully distinguishable from a dead one.

## 12. Defects found in files this ticket does not own, 2026-08-23

Reported rather than patched: `.claude/notes/review/plan/` is the orchestrator's to change at close,
via `review-close`.

1. 🔴 **`tickets.md:158` carries the stale `N_actual` formula.** It reads
   `N_actual = 182 − 15k − 8d`, which predates **both** the `+k·d` correction and the `c` term added
   to [preregistration](../plan/preregistration.md) §5 on 2026-08-16. With `k = 3` the correct form
   is `182 − 15k − 8d + k·d − c = 137 − 5d − c`; the board's version gives `137 − 8d` and therefore
   **under-counts by `3d`** — the anti-conservative direction, and precisely the defect the `+k·d`
   amendment exists to prevent. The same stale formula reached T-04a's close independently, so this
   is two documents carrying it, not a transcription slip in one. **`preregistration.md` §5 is
   correct and is the authority.** Enumeration remains authoritative over any closed form.

2. ✅ **`data.md` §7's two-root path assignment is already discharged — by T-05, not by T-06.**
   The defect as stated (`cohort_audit.py` and `export_graphs.py` resolving GraphEdX as
   `<source>/GED_PRECOMPUTED/<NAME>`, with no single `--source` reaching both trees) **no longer
   exists in the merged tree**. Both call sites now go through
   `benchmarks/real_data/eval_setup/data_roots.py`, a resolver that probes known layouts newest-first
   with an environment override and names every candidate on failure; it carries
   `tests/unit/test_data_roots.py`. Verified against the live tree: one `--source` resolves IAM to
   `APPROX_GED/datasets/IAM_Database/extracted` and GraphEdX to `GED_PRECOMPUTED/datasets`. **No T-06
   action is required and none was taken** — the item should be struck rather than re-implemented.

3. **`eval_setup/wl_kernel_computer.py` defaulted to `n_iter = 5`.** That is `h = 5`, three
   refinement rounds past the frozen `h = 2`, with **no off-by-one to absorb it**
   (`grakel(n_iter=k) == ours(h=k)`, corrected 2026-08-15). Every WL number that module produced was
   of a different kernel from the one the paper reports. Fixed in `b7ce447`: the backend's
   `WL_ROUNDS`, `wl_kernel_computer` and `eval_setup` are now bound to one constant, asserted equal
   to 2. **E10's existing WL numbers were produced under the old default and need re-checking**;
   the board already anticipates §4.1's WL row moving on Letter LOW from 0.895 to 0.7792.

### 11.3 Pre-declared stopping rule for the F-1 re-measurement, 2026-08-23

**Written and committed BEFORE any result existed.** At `2026-08-23T09:22:53Z`,
`grep -c '^protein ' f1_verify.log` returned **0** and `f1_verify.json` did not exist — no cell had
completed, so nothing about the outcome was visible when this rule was fixed. That is the whole
point of recording it here rather than afterwards: a stopping rule applied at a declared boundary
for a stated non-outcome reason is defensible; one applied after glancing at results is not.

| | |
|---|---|
| **Declared sample** | 200 graphs × 3 datasets (protein, coil_del, mutagenicity) × 2 encoders, seed 42 |
| **Executed** | **protein only** — both encoders, full 200 |
| **Not executed** | coil_del and mutagenicity, both encoders |
| **Reason** | ≈ 8 h of exclusive box time against 8 days to deadline, with the competitor-encoding critical path blocked behind it. **A resource reason, not an outcome reason** |
| **Boundary** | Stopped at a **dataset boundary, never mid-sample**. The report is written after every cell, so protein's two cells are complete and internally whole |
| **Mechanism** | A watcher polls for the second protein row and kills the run before `coil_del` begins — the stop is automatic and does not involve reading a result first |
| **Authorised by** | the PI, on the costed options in this session's status report |

**Why protein is the right dataset to keep.** §1.6's probe killed `canonical` on **10 of 15** protein
graphs, the worst of the three. An `n = 200` result there bounds the **worst case**, so the datasets
dropped are the ones that would have been *easier* on `canonical`, not harder. **Note the probe used
here is the warm-up-contaminated one**: it is being used only to choose *which* dataset to retain, a
decision its contamination cannot corrupt, and **not** as evidence for any conclusion.

#### The verdict rests on cost, not on kill counts

**F-1 is forced by the cost ratio, independently of any kill count**, which is a stronger footing
than the one it was originally frozen on and does not depend on the contaminated probe at all:

| | protein, 569 graphs |
|---|---|
| `pruned`, production, measured | **47.1 s total ≈ 83 ms/graph** |
| `canonical`, this run, measured | **≈ 43 s/graph** |
| ratio | **≈ 520×** |
| projected full-dataset cost under `canonical` | **≥ 6.8 core-hours**, against `pruned`'s 47 seconds |

For **one of ten** datasets. **The `≥` is not decorative**: 43 s/graph is an average taken under a
60 s cap, and a killed graph would have taken *longer* than 60 s had it been allowed to finish, so
the true cost is bounded **below** by this figure. The estimate errs in the conservative direction.

**Record it as: cost ratio primary, kill counts corroborating.** This also disposes of the retracted
figures cleanly — the original numbers were wrong, and the decision they supported survives on
evidence that never needed them.

#### Two caveats that travel with any number from this run

1. **60 s budget against D14's frozen 300 s.** A graph killed at 60 s might survive at 300 s, so
   **every kill count here is an UPPER bound** on the true D14-budget count. `max_ok_s` is recorded
   precisely so a reader can judge the gap rather than assume it: if no survivor approaches 60 s,
   the cut is clean and a 5× budget would change little.
2. **Partial contention, disclosed.** Sampled positions **0–47 of 200 overlapped test-suite
   execution** on the same box; positions **48–200 ran with no other compute**. Contention can only
   **inflate** a wall-clock kill count, which is the conservative direction for the arm F-1 already
   favours, so it cannot manufacture the conclusion — but the split is stated rather than left to be
   discovered.

> **The `pruned` figure is verified against the artifact, not taken on report.** Read directly from
> `encodings/suite2/protein__isalgraph_pruned.npz`: **G = 569, sum = 47.12 s, mean = 82.8 ms**, and
> the PI's quoted figure reproduces exactly. Two details the mean hides, both worth carrying:
> **median = 3.2 ms** against a **max of 15.01 s**, on `n_max = 96`. The distribution is dominated by
> a handful of large graphs — the same shape `canonical` has, simply ~520× cheaper.
>
> This is why the comparison above is framed as **total dataset cost** (47.12 s against ≥ 6.8
> core-hours) rather than as a ratio of per-graph medians, which would compare the cheap tail of one
> encoder against the expensive tail of the other and overstate nothing but explain less. Total cost
> is also the quantity the decision actually turns on: whether the arm is computable across the
> cohort at all.

---

## 11.4 RESULT — F-1 closed 2026-08-23. It never needed a timing argument

**F-1 stands: `isalgraph_pruned` is the reference arm on both suites.** The kill counts §11.3
declared were **never measured**, and the cost figures §11.3 built its case on are **retracted**.
The verdict is unchanged, because the reason that actually forces it is a policy, not a speed.

### The operative reason

`isalgraph_canonical` **cannot produce Suite-2 columns at any speed.** `SUITE1_ONLY` is a frozen
T-04 policy: the competitors registry raises `SuiteScopeError` above `n = 12` *before attempting an
encode*. §11.1 measured that guard directly — the split is exactly at `n(ok) max = 12` /
`n(err) min = 13`, and the counts equal `#{n ≤ 12}` in each sample precisely.

So the arm is excluded by scope, and **a timing measurement could not have changed the outcome**.
§11.1 already said as much — "even a fast result does not by itself reopen F-1; it would have to go
to the PI" — which means the four runs below were, in hindsight, gathering corroboration for a
decision already forced on other grounds. Recording that is more useful than the number would have
been.

### What is retracted, and why

| Retracted | Reason |
|---|---|
| **`canonical` ≈ 43 s/graph** | **No artifact.** §11.3 records it as "this run, measured", but the run left no `f1_verify.json` and a zero-row log. The figure came from a live observation of a process that never wrote a result, computed as wall-clock ÷ positions *including kills* — not commensurable with the per-graph `seconds` it was compared against |
| **the ≈ 520× cost ratio** | Verified numerator, unprovenanced denominator. A ratio is not more defensible than its weaker half |
| **projected ≥ 6.8 core-hours** | Derived from the retracted per-graph figure |

**The §1.6 probe's kill counts do not rescue these.** They are warm-up-contaminated and were
retracted already; using them to re-float a retracted figure would be circular. §11.3's own use of
that probe was legitimate and narrower — choosing *which* dataset to retain, a decision its
contamination cannot corrupt — and that use stands.

### What survives, verified

Read directly from `encodings/suite2/protein__isalgraph_pruned.npz` by the orchestrator on
2026-08-23, independently of any report:

| protein, `isalgraph_pruned` | measured |
|---|---|
| graphs | **569** |
| total encode time | **47.12 s** |
| mean / median | **82.8 ms** / **3.2 ms** |
| max | **15.01 s**, at `n_max = 96` |

All five reproduce §11.3 exactly. Provenance in the file: `isalgraph_build_hash =
298fc1188bf1b051` (**equal to the live engine's**, so the `.so` has not drifted since wave 1),
`encode_budget_s = 300.0`, `seed = 42`, `schema_version = t06.1`.

### Kill counts: NOT MEASURED — four attempts, four diagnosed instrument failures

| # | Date | What it actually measured | Defect |
|---|---|---|---|
| 1 | 2026-08-16 | **Start-up cost**, not the encoder | One graph per subprocess; a 4.95 ms encode read as 579 ms (**×117**) |
| 2 | 2026-08-16 | **A scope guard**, not a timeout | Ran through the competitors registry; every failure `SuiteScopeError`, 0 budget failures (§11.1) |
| 3 | 2026-08-16 | **Nothing** — could not terminate | Worker given the contiguous *span* of the sample, not the sample (2.8×–19.7× the work); and `readline()` blocked so the deadline was never re-checked (§11.2) |
| 4 | 2026-08-23 | **Nothing** — died before any cell completed | Relaunched after `7274f13` fixed both defects above; zero-row log, no JSON, no surviving process |

**This is one pattern, not four accidents: the harness measuring itself.** A timing instrument
needs its own verification before its output is read — and three of the four failures were only
visible *because* someone checked an artifact rather than a status line. The fourth was hidden by
the inverse error, below.

### The process-liveness trap, both directions in one session

§11.2 records `ps` returning empty for a live process. The complement bit twice on 2026-08-23:
**`pgrep -a -f f1_verify` matches the shell wrapper running the `pgrep`**, so it reports a hit when
nothing is running. Two sessions independently read that as "still executing". The rule that
survives both failure modes:

> **Confirm a run against its artifact, not against a process list.** Use
> `pgrep -a -f 'f1_ver[i]fy'` when a process check is genuinely needed. A run that is alive but has
> written nothing for hours is not usefully different from a dead one.

### Decision

**No relaunch.** With 8 days to the 2026-08-31 deadline, ~5 h of exclusive box time to corroborate
a verdict that does not depend on it is not a defensible trade against the competitor-encoding
campaign it would block. Authorised in this session's status report.

**For the response letter:** the reference arm is justified by scope, not by cost. Do not quote
43 s/graph, 520×, or 6.8 core-hours — they are retracted here and must not survive into the
manuscript or the letter.

---

## 13. Gate results, measured 2026-08-23 — cohort, pairs, structure

Run by the orchestrator against the artifacts, not taken from any report.

### 13.1 Stop-and-ask condition 3 — the cohort reproduces, all four counts

| Quantity | Target | Measured | |
|---|---|---|---|
| Suite-2 graphs | 16,370 | **16,370** | ✅ |
| Suite-2 pairs | 21,710,892 | **21,710,892** | ✅ |
| Suite-1 graphs | 5,350 | **5,350** | ✅ |
| Suite-1 pairs | 3,897,911 | **3,897,911** | ✅ |

Both pair counts are `Σ_d C(n_d, 2)` over datasets and reproduce to the unit. **`aids` is 769 in
Suite 1 against `aids_graphedx`'s 819 in Suite 2** — F-12's positional-join hazard is live in the
data, not hypothetical.

### 13.2 A4 structural gate — Suite-1 exact-GED matrices, 0 violations

Joined **on `graph_ids`**, never positionally, then cross-checked on `node_counts`, which both files
carry — set equality alone would not catch a permuted join, and this does.

| dataset | n | ids match | node_counts agree under the join | symmetric | zero diagonal |
|---|---|---|---|---|---|
| aids | 769 | ✅ | ✅ | ✅ | ✅ |
| iam_letter_high | 2,059 | ✅ | ✅ | ✅ | ✅ |
| iam_letter_low | 1,180 | ✅ | ✅ | ✅ | ✅ |
| iam_letter_med | 1,253 | ✅ | ✅ | ✅ | ✅ |
| linux | 89 | ✅ | ✅ | ✅ | ✅ |

**No NaN anywhere** — F-13 holds: censored pairs are `inf`, and `np.isfinite` is the correct filter.

### 13.3 🔴 GED availability is NOT uniform — two Suite-1 datasets are under half populated

Strict upper triangle, so these are the ladder's `GED-available` and `GED > 0` rungs directly:

| dataset | pairs | GED-available | avail % | GED > 0 | GED == 0 |
|---|---|---|---|---|---|
| aids | 295,296 | 131,148 | **44.41 %** | 131,148 | **0** |
| linux | 3,916 | 1,685 | **43.03 %** | 1,685 | **0** |
| iam_letter_low | 695,610 | 695,610 | 100 % | 587,626 | 107,984 |
| iam_letter_med | 784,378 | 784,378 | 100 % | 674,262 | 110,116 |
| iam_letter_high | 2,118,711 | 2,118,711 | 100 % | 2,030,043 | 88,668 |

Two facts worth carrying into F0, and neither is visible from a cohort count:

1. **`aids` and `linux` lose ~56 % of their pairs at the `GED-available` rung.** Exact GED stops
   being computable above ~12 nodes, so the surviving pairs are a **size-biased subsample**, not a
   random one — the same bias §4.1 found inside the size null, arriving here in the denominator.
   Any ρ on these two datasets is computed on the small-graph half.
2. **Every available pair in `aids` and `linux` has GED > 0, exactly.** No isomorphic pairs survive
   the availability filter, while the three Letter datasets carry 12.7–15.5 % exact zeros. So the
   `GED > 0` rung is a no-op on two datasets and removes a sixth of the pairs on three — a per-dataset
   difference the pooled ladder would hide, which is why A9 requires it per dataset.

**This does not yet trigger stop-and-ask 1**, which is about F0 *failing*, not about the pair supply.
It is recorded because it shapes how an F0 result on `aids` or `linux` must be read.

---

## 14. What is actually established as of 2026-08-23 — pilot, not result

Written after the PI challenged an orchestrator summary that presented three findings as settled.
**The challenge was correct in substance**: T-06 has computed **no** competitor correlation at any
scale. The encoding campaign is still running and no distance matrix exists. Every ρ quoted anywhere
in this ticket today comes from **T-04a's 200-graph pilot**.

The three findings do have artifacts and all three reproduce — the error was in the **epistemic
status** assigned to them, not in their existence.

### 14.1 Verified from `paired_null_ci.json` — `isalgraph_pruned` vs the size null

Paired graph-level bootstrap, 2,000 resamples, seed 42, **n = 200 graphs** (89 for `linux`, which has
only 89). **This file contains `isalgraph_pruned` only — no competitor appears in it at all.**

| arm | dataset | ρ(Lev) | ρ(null) | paired diff | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| exact | iam_letter_low | 0.9253 | 0.8991 | **+0.0262** | [0.0083, 0.0463] | **clears** |
| exact | iam_letter_med | 0.8750 | 0.9190 | −0.0440 | [−0.0676, −0.0140] | below |
| exact | iam_letter_high | 0.6969 | 0.9174 | −0.2205 | [−0.2599, −0.1825] | below |
| exact | linux | 0.4742 | 0.7134 | −0.2392 | [−0.3199, −0.1570] | below |
| exact | aids | 0.2561 | 0.7844 | **−0.5283** | [−0.5708, −0.4823] | below |
| lb | all 5 Suite-2 | 0.69–0.92 | **0.96–0.998** | −0.08 … −0.29 | all exclude 0 | **all below** |
| ub | all 5 Suite-2 | 0.74–0.93 | **0.46–0.74** | +0.03 … +0.38 | all exclude 0 | **all clear** |

**1 of 5 on Suite 1, and a 5/5 inversion on Suite 2** — both confirmed.

**The inversion is driven by the null, not by the arm.** ρ(Lev) moves little between LB and UB
(e.g. protein 0.7372 → 0.8428); ρ(null) collapses from 0.9699 to 0.4596. The bracket changes what
the *baseline* correlates with, so the comparison is genuinely undetermined rather than merely noisy
— §10's conclusion, now with the numbers attached.

### 14.2 Verified from `f5_200.json` — `isalgraph_pruned` against every competitor

**n = 200**, `all_pairs` view, and the file's own note reads *"DESCRIPTIVE. F5 is not an input to
distance selection."*

`isalgraph_pruned` is beaten on **15 of 15** records. The margins are not marginal:

| record | isalgraph_pruned | best competitor | |
|---|---|---|---|
| aids / exact | 0.2561 | **agm_cam 0.7828** | +0.53 |
| linux / exact | 0.4742 | **agm_cam 0.7979** | +0.32 |
| iam_letter_high / exact | 0.6969 | **agm_cam 0.8921** | +0.20 |
| mutagenicity / lb | 0.6926 | **wl_subtree 0.9005** | +0.21 |
| iam_letter_low / exact | 0.9253 | **min_dfs 0.9720** | +0.05 |

**The winners are all admissible.** `agm_cam`, `min_dfs`, `wl_subtree`, `sparse6_nauty` and
`nauty_graph6` are not among the `k = 3` metric-axiom exclusions, so this result does **not** dissolve
when the k-excluded backends are dropped.

### 14.3 What this is not, and what T-06 must therefore do

1. **It is a 200-graph pilot.** 200 of 4,040 Mutagenicity graphs is a **5 % subsample**; 200 of 3,900
   coil_del likewise. T-06 recomputes at **16,370** and **5,350**. A ρ estimated on 19,900 pairs can
   move at 8,158,780.
2. **It is a single seed-42 draw**, per `corrected_rho_table.json`'s own `convention` block. No
   resample-across-draws exists, so the between-draw variance is unmeasured.
3. **It predates the frozen conventions.** `corrected_rho_table.json` supersedes
   `competitors/README` §4.1–4.2, and `wl_kernel_computer` ran at `h = 5` until `b7ce447` — E10's WL
   numbers were a different kernel from the one the paper reports.
4. **Nothing here has been through BH.** These are raw ρ and CIs, not F0/F1/F2 outcomes over
   `N_actual`.

> **Standing rule for this ticket, adopted here.** Quote a T-04a pilot number as *"the pilot found X
> at n = 200; T-06 tests whether it holds at cohort scale"* — never as a T-06 result. The direction
> of the pilot is a **prior**, and a strong one, but the ticket exists because a prior is not a
> result. If the full recompute overturns any of these three, that is a finding and not an
> embarrassment.

---

## 15. RESULT — the competitor campaign, `c`, and a defence that does not work

Campaign complete 2026-08-23, 14:26:52Z → 14:47:19Z (**20 min 27 s**), 135 cells, **0 failures**,
both file-count assertions passed. Manifest: 155 cells, **0 contract violations**, cohort coverage
complete on both suites.

### 15.1 `c` is set by `agm_cam`'s scope guard, and the split was load-bearing

**`agm_cam` is refused above `n = 12` by policy**, exactly like `isalgraph_canonical`. The split is
clean on every dataset — `ok` tops out at `n = 12`, refusals start at `n = 13`:

| cell | rate | scope | budget | infra |
|---|---|---|---|---|
| suite2/protein/agm_cam | **6.15 %** | 534 | 0 | 0 |
| suite2/mutagenicity/agm_cam | **6.29 %** | 3,754 | 32 | 0 |
| suite2/coil_del/agm_cam | **31.15 %** | 2,685 | 0 | 0 |
| suite2/grec/agm_cam | 67.08 % | 214 | 0 | 0 |
| suite2/aids_iam/agm_cam | 82.55 % | 311 | 5 | 0 |
| suite2/aids_graphedx/agm_cam | 93.53 % | 50 | 3 | 0 |
| suite2/mutagenicity/min_dfs | 94.78 % | 0 | 207 | **4** |
| suite1/aids/agm_cam | 99.61 % ✅ | 0 | 3 | 0 |

**These reproduce `preregistration` §5.1's own table**, measured by T-04 at `n = 200`: protein 90 %
fail (measured **93.9 %**), mutagenicity 98 % (**93.7 %**), aids_iam 18 % (**17.5 %**), Letter ×3 and
LINUX 0 % (**0 %**). The rule was written for this case and the case arrived as predicted.

**§5.1's criterion decides it and needs no judgement from me:** *"computable on a dataset iff it
produces an encoding for ≥ 99 % of that dataset's graphs."* A scope refusal produces no encoding, so
it counts — **even though §11.1 established that a scope refusal measures a guard, not an encoder**.
Those two facts are compatible: *why* the cell is empty is a matter for the prose, *whether* it is
empty decides the test's existence. The distinction is still recorded per cell, because a reader must
be able to see that `agm_cam` is guarded rather than slow.

### 15.2 `N_actual` — enumeration and closed form agree exactly

`k = 3`, 7 non-computable triples, and **`discrepancy = 0` at every `d`**:

| `d` | `N_actual` | closed form | `c` |
|---|---|---|---|
| 0 | **123** | 123 | 14 |
| 1 | 119 | 119 | 13 |
| 2 | 115 | 115 | 12 |
| 3 | 111 | 111 | 11 |
| 4 | 107 | 107 | 10 |
| 5 | 102 | 102 | 10 |

`c` **falls as `d` rises** — §5.2's precedence working: a cell removed by `d` is not charged again to
`c`. Final `c` awaits F1.

### 15.3 🔴 "The competitors that beat us are the ones that fail to scale" — TESTED, FALSE

A natural defence, and worth testing rather than assuming: if the winners are exactly the
representations excluded by `k` (metric axioms) or by `c` (computability), the comparison is not
really lost.

**It is false.** Restricting to competitors that are **both** admissible (not `k`-excluded) **and**
computable (≥ 99 % on that dataset), `isalgraph_pruned` is still beaten on **15 of 15** records.

| winner | fails? | wins on |
|---|---|---|
| `agm_cam` | fails 6 **Suite-2** cells | aids, iam_letter_high, linux — **all Suite 1, where it completes** |
| `min_dfs` | fails **mutagenicity only** | aids_iam/lb, grec/lb, iam_letter_low, iam_letter_med — **all cells it survives** |
| `wl_subtree` | **fails nothing** | coil_del/lb, mutagenicity/lb, protein/lb |
| `sparse6_nauty` | **fails nothing** | coil_del/ub, mutagenicity/ub, protein/ub |
| `nauty_graph6` | **fails nothing** | grec/ub, aids_iam/ub |

**The exclusions and the wins are disjoint.** `agm_cam` fails on the six Suite-2 datasets where it
never wins anyway; on those exact datasets the winners are `wl_subtree`, `sparse6_nauty` and
`nauty_graph6` — **three representations that complete on 100 % of every cell of both cohorts and
pass every metric axiom.** There is no ground on which to exclude them.

**So the honest positioning is not "IsalGraph approximates GED better".** On the pilot it does not,
anywhere. What survives is narrower and still real:

1. **Completeness is a theorem, not a measurement** — `w*_G = w*_H ⟺ G ≅ H` within a directedness
   class. `nauty_graph6` and `sparse6_nauty` are also complete invariants, so this is shared, not
   unique.
2. **It computes everywhere.** 100 % of both cohorts, with 2.50 % D14 censoring on Mutagenicity
   retained with its greedy-min fallback. `agm_cam`, the strongest Suite-1 competitor, manages
   **6 %** of protein.
3. **The representation is an executable instruction string**, which is what no serialisation
   competitor is. That is the actual novelty and it is not a ρ claim at all.

**This is a prior, not a result** (§14): `f5_200` is `n = 200`, one seed-42 draw, descriptive by its
own note, and pre-BH. T-06 tests it at 16,370. But the defence above is **structural** — it depends
on which representations are excludable, not on the ρ values — so a full-cohort recompute will not
rescue it. If the ranking is to change, it must change on the ρ values themselves.

### 15.4 Defect found — `t06_completion` counts a censored graph as not completed

`suite2/mutagenicity/isalgraph_pruned` reports `n_completed = 3939`, `n_censored = 101`,
`rate = 0.9750`, `meets_threshold = false`. **Wrong under D14/F-4**: a censored graph is *retained
with its greedy-min fallback string*, so it **does** produce an encoding — which is precisely what
§5.1's criterion asks for. The manifest gate counts it as complete (100 %).

**Inert today**, for two independent reasons: §5.1 consequence 2 exempts the IsalGraph arm from `c`
entirely, and no comparator produced a censored row. **Latent tomorrow**: a comparator with censored
rows would be under-counted, over-charged to `c`, and would shrink `N_actual` — the anti-conservative
direction, and the sixth instance of §8.1's pattern.

---

## 16. What "wins" means — the measurement stated exactly, and two errors it exposed in §15.3

The PI asked what quantity §15.3's "beaten on 15 of 15" refers to. Stating it precisely uncovered
**two errors in my own analysis**. The conclusion survives both, but on different evidence.

### 16.1 The measurement, exactly

> **Spearman ρ between (a) the pairwise distance under representation `R` using `R`'s *selected
> primary distance*, and (b) ground-truth GED**, over the pairs of one view of one dataset.

| | |
|---|---|
| **Primary distance** | chosen by `grid.py` on F1–F4, cost as tie-break, **blind to ρ** — a test asserts `grid.py` has no import path reaching a GED value, so the selection *could not* have picked the baseline that flatters IsalGraph |
| **Not one distance** | `levenshtein` for `isalgraph_pruned`, `agm_cam`, `min_dfs`, `nauty_graph6`, `sparse6_nauty`; **`kernel`** for `wl_subtree`. Each representation with its own admissible distance — so this compares *representation + distance* pairs, not one function across representations |
| **Reference** | `exact` on Suite 1; on Suite 2 **`lb` and `ub` as two separate records**, never averaged, never interpolated (`approx_ged.md` §4) |
| **Views** | `all_pairs`, and `equal_n` (`n_i = n_j`) |
| **Sample** | 200 graphs/dataset, seed 42. Pairs: 15,634 (aids `all_pairs`) down to **470** (protein `equal_n`) |
| **Interval** | graph-level bootstrap — ρ moved up to 0.07 between two 200-graph draws on AIDS, so a pair-level interval is wrong by construction |

### 16.2 Error 1 — I ranked on raw `all_pairs` ρ, which carries the size channel

Raw ρ is not the comparable quantity in the `all_pairs` view, and using it made one competitor look
far stronger than it is. On `aids`, `agm_cam` scores **0.7828** against a `size_null_on_my_pairs` of
**0.7844** — an excess of **−0.0010**. It is *at* the trivial baseline, not above it. Calling that a
"win over IsalGraph" is true of the raw number and misleading about the science.

Corrected to **excess over `size_null_on_my_pairs`** — the per-cell null, which is the right
subtrahend, not the row-level one:

- IsalGraph has the **largest excess on 0 of 15**.
- **No representation clears its own null on 7 of 15 records** — every `lb` arm, plus `aids/exact`
  and `iam_letter_high/exact`. On those the whole field is below the trivial baseline.
- On the `ub` arms, 4–5 representations clear it, IsalGraph among them.

### 16.3 Error 2 — the `equal_n` null is UNDEFINED, and that is the point

My first `equal_n` pass returned zero records. Not a bug: on `n_i = n_j` the `size_null` distance
`|n_i − n_j|` is **identically zero**, so Spearman has no denominator. The file states it —
*"the 'size_null' distance is constant over the pairs of this view, so the rank correlation has no
denominator."*

**So on `equal_n` there is nothing to subtract: raw ρ *is* the structural signal.** This is the view
the module calls *"where canonicity, rather than order, has to do the work"* — and it is the fairest
test of the representation as such. Restricting to competitors both admissible and computable:

**`isalgraph_pruned` is best on 0 of 15 here too.**

### 16.4 But only 8 of 15 losses are resolvable at n = 200

Comparing graph-level bootstrap CIs in the `equal_n` view:

| verdict | n | records |
|---|---|---|
| **DISJOINT** (real loss) | **8** | aids, aids_iam/lb, grec/lb, iam_letter_high, iam_letter_med, linux, mutagenicity/lb, protein/lb |
| **OVERLAP** (tie at this n) | **7** | aids_iam/ub, coil_del/lb, coil_del/ub, grec/ub, iam_letter_low, mutagenicity/ub, protein/ub |

The disjoint losses are large — `wl_subtree` leads by **0.544** on aids_iam/lb, **0.450** on
mutagenicity/lb, **0.362** on protein/lb. The overlapping ones are small (0.015–0.125).

**CI overlap is a conservative test**: non-overlap implies a difference, overlap does not imply
none. The correct instrument is a **paired** bootstrap of `ρ(IsalGraph) − ρ(competitor)` on identical
pair sets and identical resamples — exactly what `paired_null_ci.json` already does for
IsalGraph-vs-null, and which **does not yet exist for IsalGraph-vs-competitor**.

> **T-06 deliverable, added here:** a paired IsalGraph-vs-competitor bootstrap on the full cohort.
> Without it the honest statement is *"best on 0 of 15, with 8 losses resolvable and 7 ties"* — not
> *"beaten on 15 of 15"*, which overstates what the marginal CIs support.

### 16.5 What §15.3 should have said

**Superseded:** "beaten on 15 of 15 records."

**Stands:** *IsalGraph is the best representation on **none** of the 15 records in either view. Eight
of those deficits are resolvable at the graph-level bootstrap and seven are ties at n = 200. The
competitors responsible for the resolvable losses — `wl_subtree`, `min_dfs`, `agm_cam` — are not
excludable by `k` or `c` on the datasets where they win.* The structural argument of §15.3 is
unchanged; its arithmetic is now stated at the right strength.

---

## 17. 🔴 RESULT — the within-`n` correlation collapses with graph size

**This is the ticket's central measurement and it answers the reviewers' question directly.** The
figure and the table below are produced by `benchmarks/real_data/eval_size_profile/`, run over the
full cohort: 2,355 strata, 1,553 with a defined ρ, 582 dataset-aggregated points.

### Why a within-`n` stratum is the right instrument

A pair enters stratum `n` only when `n_i = n_j = n`. Inside a stratum the size null `|n_i − n_j|`
is **identically zero**, so its rank correlation has no denominator and there is nothing to
subtract. Raw ρ inside a stratum is therefore the structural signal with the size channel removed
**by construction rather than by adjustment** — which is exactly what a reviewer asking "is this
just size agreement?" needs to see. It is `f5`'s `equal_n` view decomposed by `n` instead of pooled.

### `isalgraph_pruned`, exact-GED regime — 1.00 at `n = 3` to 0.26 at `n = 12`

| `n` | ρ | 95 % CI | datasets |
|---|---|---|---|
| 3 | **1.0000** | [1.0000, 1.0000] | 2 |
| 4 | 0.9931 | [0.9922, 0.9938] | 3 |
| 5 | 0.8785 | [0.8660, 0.8899] | 3 |
| 6 | 0.5661 | [0.5056, 0.6210] | 3 |
| 7 | 0.4383 | [0.2954, 0.5620] | 4 |
| 8 | 0.4177 | [0.1516, 0.6274] | 2 |
| 9 | 0.4135 | [0.2400, 0.5614] | 2 |
| 10 | 0.3161 | [0.1804, 0.4400] | 2 |
| 11 | 0.2288 | [0.1033, 0.3471] | 1 |
| 12 | **0.2608** | [0.1368, 0.3767] | 1 |

Every point is BH-significant against ρ = 0, so the decay is not a power artefact — the correlation
is reliably **present** and reliably **small** by `n = 12`.

### Above `n = 12` it degrades to noise, for every representation

| representation | strata (`n` 13–76) | BH-significant | last significant `n` |
|---|---|---|---|
| `wl_subtree` | 52 | **22** | 37 |
| `sparse6_nauty` | 52 | 16 | **42** |
| `min_dfs` | 48 | 14 | 38 |
| `nauty_graph6` | 52 | 6 | 33 |
| **`isalgraph_pruned`** | 52 | **6** | **29** |

Mean ρ by regime, best first:

| regime | ranking |
|---|---|
| `n` 3–12, exact | `min_dfs` 0.644 · `agm_cam` 0.643 · **`isalgraph_pruned` 0.551** · `isalgraph_canonical` 0.537 |
| `n` 13–30, UB | `wl_subtree` 0.202 · `sparse6_nauty` 0.175 · `min_dfs` 0.158 · **`isalgraph_pruned` 0.135** |
| `n` 31+, UB | `sparse6_nauty` 0.162 · `nauty_graph6` 0.153 · `wl_subtree` 0.147 · `min_dfs` 0.138 |

### What this means, stated plainly

1. **The headline pooled ρ is mostly the size channel.** The paper reports ρ ≈ 0.93 on sparse IAM.
   Holding `n` fixed, the same arm scores **0.26 at `n = 12`** and **0.135** averaged over `n` 13–30.
   The gap between those numbers *is* the size channel, and it is most of the correlation.
2. **The collapse is not IsalGraph's alone.** Every representation falls the same way, and above
   `n ≈ 40` no representation is reliably distinguishable from ρ = 0. This is a statement about
   **Levenshtein-style distances on serialised graphs against GED**, not about IsalGraph in
   particular — which makes it a more interesting finding than a defeat, and a fairer one.
3. **`isalgraph_pruned` is last or near-last in every regime**, and has the *earliest* loss of
   significance (`n = 29`) of the five computable arms. §16.5's conclusion is unchanged and now has
   a mechanism attached.
4. **`agm_cam` is absent above `n = 12` by construction** — its scope guard (§15.1) means it
   contributes no bracket-regime strata at all. Its strong `n ≤ 12` showing (0.643) is measured
   exactly where it is computable and nowhere else.

### What may be claimed, and what may not

**May be claimed:** the encoding is a complete invariant (§16 / ladder: zero collisions on 24.8 M
pairs); its Levenshtein distance correlates with GED at small `n`; the correlation is significant
throughout the exact regime.

**May NOT be claimed:** that the distance approximates GED at the sizes Suite 2 covers. On this
evidence it does not, and neither does any comparator. **Any sentence in the manuscript that quotes a
pooled ρ as evidence of structural fidelity must be re-scoped**, because the within-`n` decomposition
shows most of that ρ is size agreement.

> **Descriptive, not confirmatory.** These strata are not a pre-registered family. The BH correction
> is local to the figure, ranges over its own 582 points, and is stated on it. Nothing here feeds
> F0, F1 or F2 — but it is the right diagnostic to put beside them, and it should reach the response
> letter.

---

## 18. 🛑 STOP-AND-ASK — F0 and F1 both fired. Two of §7's conditions, together

Run 2026-08-23. **I have not run F2 and will not until the PI decides.** Both gates are
pre-declared, both fired on their own rules, and between them they change what the paper claims.

### 18.1 F0 — the calibration gate fires on **4 of 5** (§7 condition 1: "≥ 3 of 5")

| dataset | point | 95 % FCR CI | BH p | fires |
|---|---|---|---|---|
| iam_letter_low | **+0.2180** | [+0.1964, +0.2416] | 0.00125 | ✅ |
| iam_letter_med | **+0.1952** | [+0.1284, +0.1696]¹ | 0.00125 | ✅ |
| iam_letter_high | **+0.0672** | — | 0.00125 | ✅ |
| linux | **+0.1169** | — | 0.021 | ✅ |
| aids | +0.0370 | — | 0.00125 | ❌ (below the 0.05 threshold) |

¹ intervals as stored in `family_F0.json`; every firing interval excludes 0.

**Pre-declared consequence** (`preregistration` §2 / §5.3): the exact-GED results become primary, the
**81 approximate-regime cells** (70 B1a + 1 B2 + 10 B3a) are demoted to **descriptive only**, `d` is
**not applied at all**, and `k` removes only its 5 B1e cells per representation.

### 18.2 🔴 The branch flips on a choice the pre-registration never made

§16.1 flagged that "GED_approx" does not name one object. It is now **determinative**:

| reading | datasets exceeding \|0.05\| | majority branch |
|---|---|---|
| **LB only** | **2 of 5** (`iam_letter_low`, `iam_letter_high`) | **does NOT fire** |
| **UB only** | **4 of 5** | **FIRES** |
| conservative (worse of the two — as run) | 4 of 5 | **FIRES** |

| dataset | LB | UB |
|---|---|---|
| iam_letter_low | +0.0612 | +0.2180 |
| iam_letter_med | +0.0470 | +0.1952 |
| iam_letter_high | **−0.0616** | **+0.0672** |
| linux | +0.0316 | +0.1169 |
| aids | +0.0370 | +0.0262 |

**`iam_letter_high` changes sign**, which is the expected signature of a bracket that genuinely
contains the truth: ρ(Lev, exact) sits *between* ρ(Lev, LB) and ρ(Lev, UB). That is reassuring about
the bracket and fatal to the idea of a single "GED_approx".

**This is the PI's call, not mine.** I ran the conservative reading and it demotes 81 cells; the
permissive reading keeps them confirmatory. I am not willing to pick between "the large-`n`
extension is confirmatory" and "the large-`n` extension is descriptive" on a coin the
pre-registration left in the air — that is precisely the decision AE.1 turns on.

### 18.3 F1 — `d = 7 of 10` (§7 condition 5: "`d ≥ 5`")

| dataset | point | 95 % FCR CI | fires |
|---|---|---|---|
| iam_letter_low | +0.1568 | [+0.1370, +0.1789] | ✅ |
| iam_letter_med | +0.1482 | [+0.1284, +0.1696] | ✅ |
| iam_letter_high | +0.1288 | [+0.1165, +0.1416] | ✅ |
| protein | −0.1180 | [−0.1362, −0.0954] | ✅ |
| mutagenicity | −0.1106 | [−0.1170, −0.1036] | ✅ |
| linux | +0.0760 | [+0.0020, +0.1590] | ✅ |
| grec | −0.0668 | [−0.0814, −0.0515] | ✅ |
| coil_del | −0.0165 | [−0.0212, −0.0107] | ❌ |
| aids_iam | −0.0129 | [−0.0199, −0.0056] | ❌ |
| aids_graphedx | −0.0139 | [−0.0340, +0.0084] | ❌ |

**Seven of ten brackets are uninformative**: the conclusion is *not* invariant to where inside the
proven bracket the truth lies. Note the sign split — ρ(Lev, LB) exceeds ρ(Lev, UB) on the Letter
datasets and LINUX, and the reverse on GREC, Mutagenicity and Protein — so this is not one
systematic offset but a genuine dependence on the bound.

Only `coil_del`, `aids_iam` and `aids_graphedx` have brackets tight enough to conclude from, and
their point estimates are all ≈ −0.015, i.e. essentially invariant.

### 18.4 `N_actual` under each branch — enumeration and closed form agree, discrepancy 0

| branch | `k` | `d` | `c` | **`N_actual`** | closed form |
|---|---|---|---|---|---|
| **F0 fired** (as measured) | 3 | not applied | 7 | **79** | 79 ✅ |
| F0 not fired (LB reading), `d = 7` | 3 | 7 | 10 | **92** | 92 ✅ |

### 18.5 Why this is coherent with everything else measured today

These are not three surprises; they are one finding seen three ways.

- §17: the within-`n` correlation collapses to ρ ≈ 0.135 above `n = 12` and is indistinguishable
  from 0 above `n ≈ 40`.
- §10 / §14.1: the size-null verdict **inverts** across the bracket on 5 of 5 Suite-2 datasets.
- §18.3: 7 of 10 brackets are formally uninformative.

**The bracket is too wide, at exactly the sizes the extension needs it, to support a confirmatory
claim.** F0 and F1 are the pre-registered instruments detecting that, and they detected it. This is
the "legitimate outcome" `approx_ged.md` §3 item 2 explicitly allows for.

### 18.6 The question for the PI, in one line

**Which bound is `GED_approx` for F0?** Answer that and the branch follows mechanically:
`UB` or conservative → 81 cells descriptive, `N_actual = 79`; `LB` → extension stays confirmatory,
`N_actual = 92` with `d = 7`. Everything downstream — F2, T-20's claims, the AE.1 response — hangs
on it.

### 18.7 ✅ PI DECISION, 2026-08-23 — `GED_approx` is the **conservative** reading

**Asked and answered.** `GED_approx` for F0 is the **worse of LB and UB** — whichever bound produces
the larger `|point|`, i.e. whichever makes the approximation look *less* like a validated stand-in.
This is the reading already implemented and already run, so no recomputation is needed.

**Consequences, now frozen:**

| | |
|---|---|
| F0 majority branch | **FIRES** (4 of 5) |
| The 81 approximate-regime cells (70 B1a + 1 B2 + 10 B3a) | **descriptive only** |
| `d` | **not applied at all** (§5.3) — F1's `d = 7` is reported but does not reduce `N_actual` |
| `k` | removes only its **5 B1e cells per representation** |
| `c` | **7** |
| **`N_actual`** | **79** — enumeration and closed form agree, discrepancy 0 |
| F2 | run over **79** admissible cells, with the BH-over-`N_max` = 182 sensitivity column printed |

**Why this is the defensible choice**, recorded so the response letter can say it in one sentence:
it cannot be accused of selecting the bound that flatters the extension, and it agrees with three
independent measurements that all say the bracket is too wide at the sizes the extension needs —
§17's within-`n` collapse, §10/§14.1's size-null inversion on 5 of 5, and §18.3's `d = 7`.

**What this costs, stated plainly:** the large-`n` extension — the paper's scope claim — becomes
**descriptive**. `approx_ged.md` §3 item 2 pre-declared that as a legitimate outcome, and it is the
one that occurred.

**F1's `d = 7` is still reported.** It does not enter `N_actual` under this branch, but it is the
evidence that the bracket is uninformative on 7 of 10 datasets and it belongs in the letter beside
F0.

### 18.8 Four rulings frozen before F2 runs, 2026-08-23

Raised by `[T06-subagent-01]` during scope confirmation, all four before any code was written.
Verified independently: the three artifacts named in Q1 exist **nowhere** under the data root, and
`N_actual = 79 / closed_form = 79 / discrepancy = 0 / c = 7` reproduces.

**Q1 — `claim_a.json`, `censoring.json`, `rho_table.json` are all in scope.** `t06_claim_a.py`
exists and has never been run. **`censoring.json` is emitted even though its measurement is already
done**: an acceptance criterion that names an artifact is not discharged by the numbers existing in
two other files. The point of A3 is one auditable object a reviewer can open, not a value
recoverable by joining `manifest.json` to `completion_rates.json`.

**Q2 — §6.2 is NOT discharged by B1e/B1a, and the reason is the view.** `bootstrap_associations`
with a `DifferenceSpec` is indeed the paired ρ(IsalGraph) − ρ(comparator) on identical pairs and
identical resamples. But §16.4's "8 disjoint / 7 ties" was measured in **`equal_n`**, and B1e/B1a are
**`all_pairs`**. §16.2 showed `all_pairs` ρ carries the size channel; §16.3 showed the `equal_n` null
is *undefined* so raw ρ there is the structural signal. **Run both views.**

> **Guard, and it is the load-bearing half of this ruling:** the confirmatory B1e cells use whichever
> view `preregistration` §4 defines. The `equal_n` run is a **descriptive addition** and must be
> labelled so. Adding a view to a pre-registered family silently doubles it. **If §4 is silent on the
> view, report both, mark neither primary, and escalate** — do not pick, for the same reason F-5
> exists.

**Q3 — yes, compute the 81 demoted cells at full cohort.** ~4–7 h wall clock, far under the 5,000
core-hour line. **Descriptive does not mean unmeasured**; it means not BH-corrected as confirmatory.
Since F0 fired, descriptive is now the *only* form the large-`n` story can take, so these are exactly
the numbers the response letter has to quote.

**Q4 — A1's bit convention: use the INTERSECTION–UNION TEST, `p = max(p_entropy, p_realised)`.**

F-5 freezes that both conventions are reported and that the primary is "named in the text"; it is
never named. Rather than name one now — after the data exist, which is precisely what F-5 exists to
prevent — note what Claim A actually asserts: *IsalGraph encodes a graph in fewer bits than the
competing serialisation.* Read against a two-convention report that is a **conjunctive** claim —
fewer bits under **both** conventions — and the intersection–union test for a conjunction has
`p = max` of the component p-values. That is a standard level-α procedure, not a hedge:

- it is **valid** — an IUT rejects the conjunctive null at level α without adjustment;
- it is **conservative for BH** — `max(p₁, p₂) ≥ p₁, p₂`, so BH over max-p rejects a subset of what
  either convention alone would;
- it **needs no primary to be named**, which removes the F-5 hazard entirely;
- it **matches §18.7's stance** on the F0 bound: where the pre-registration left a gap, take the
  reading that makes the claim harder, not easier.

`entropy_bits` remains the **reference convention for description** — it is the like-for-like
efficiency measure, independent of how a string is packed into bytes, which is an implementation
detail rather than a property of the representation. **Both marginal p-values are printed beside
every A1 cell**, and per-cell discordance between the conventions is flagged rather than absorbed.

`wl_subtree` and `size_null` carry no A1 cell at all: `BitCountUndefined`, with the reason printed
and never fabricated (A2).

### 18.9 Q2 ruled: `all_pairs` is confirmatory. And a mislabelled ladder rung, found by the subagent

**Verified independently:** `equal_n`, `equal-n`, `all_pairs` and `n_i = n_j` occur **zero times** in
both `preregistration.md` and `statistics.md`. §4 is genuinely silent.

#### Ruling — the confirmatory view is `all_pairs`

Three grounds, the first the subagent's and the strongest:

1. **A9's ladder has no equal-`n` rung.** If the family were defined over `equal_n`, the accounting
   that exists to expose the analysed denominator would have to say so. It does not.
2. **§4.2 is unqualified** — "bootstrap CI on ρ(Lev, exact) − ρ(comparator distance, exact), per
   Suite-1 dataset". The unmarked reading of "ρ over a dataset" is over its pairs, not a subset.
3. **F0 and F1 have already run on the full defined pair set.** B1e/B1a are the same shape of
   quantity, and a family whose gates and rows use different pair sets is incoherent.

`equal_n` is **descriptive**, and §17 already carries it in the form that matters.

> **No third BH column.** Emit the `equal_n` arm as point estimates and CIs plus a *locally* adjusted
> q under an explicitly descriptive key — never a `bh_*` column sitting beside the confirmatory one
> as though co-equal. Three side-by-side BH columns is the shape that gets misread as three families.
> `primary_view: "all_pairs"`.

#### The ladder's `analysed` rung is mislabelled — my defect, not a protocol conflict

The subagent found that `ladder.json`'s `analysed` (331,202 on `aids_graphedx`) disagrees with the
pair count F0/F1 actually ran on (334,971 = `ged_available`). The 3,769-pair gap is exactly the
GED = 0 pairs.

**Checked before ruling:** the rung ordering `raw → connected → GED-available → GED > 0 → Lev > 0 →
analysed` appears **only in this design note's acceptance criteria** — not in `preregistration.md`,
not in `statistics.md`. So it is a **reporting requirement, not a protocol definition of the analysed
set**, and there is no pre-registration conflict to resolve. What there is, is a bug in
`t06_ladder.py`, which I wrote: it computes `analysed = ged_positive & defined_mask`, which is not
what gets analysed.

**Fix:** `analysed` reports `ged_available & defined_mask`, the correlation's actual pair set. The
`GED > 0` and `Lev > 0` rungs are relabelled as what they are — the **collision denominator**, which
is what they exist for (`collisions` / `collision_rate`, the zero-collisions result on ~24.8 M
pairs). `rho_table.json` and `ladder.json` must then agree on `n_pairs`, and the JSON says why the
two families of rung differ.

**The subagent's reasoning for keeping GED = 0 pairs in the correlation is correct and adopted:**
GED is legitimately 0 for isomorphic graphs — 28.05 % of IAM Letter LOW pairs are certified at
exactly 0 — and filtering them would truncate the response variable at its most informative end,
contradicting CONTRACTS §4.1 and trap 9.

> **But measure it rather than assert it.** Those pairs are a large block of ties in the response,
> and Spearman is tie-sensitive: 15.5 % of `iam_letter_low`'s pairs sit at GED = 0. **Compute ρ with
> and without them on two contrasting datasets** — one zero-heavy (`iam_letter_low`), one
> zero-light (`protein`, 15 of 161,581) — and report the delta. If it moves materially the PI must
> see that before T-20 writes anything. This is a sensitivity, not a re-run: F0/F1 stand.

### 18.10 The censoring size-stratum, and a confound it exposes in §17

`[T06-subagent-01]` measured D14 censoring **by size stratum** rather than by dataset, and the
result is sharper than the figure this ticket has been quoting.

| `n` | graphs | censored | rate |
|---|---|---|---|
| ≤ 40 | 3,497 | 0 | **0.00 %** |
| 41–60 | 384 | 12 | 3.12 % |
| 61–80 | 118 | 58 | **49.15 %** |
| 81+ | 41 | 31 | **75.61 %** |

**Censoring is exactly zero below `n = 40`** and then rises monotonically to three quarters above 80.
Censored graphs span `n = 45–98` against a kept median of 26.

**Adopt this form.** "2.50 % of Mutagenicity" is that curve diluted by 3,497 small graphs, and
"0.62 % cohort-wide" dilutes it again across nine datasets that censor nothing. Two averages, each
hiding the structure the reader needs. The stratum supports D14's premise far more directly than
either, and it is what the response letter should carry. `censoring.json` now refuses a cohort-level
rate quoted without naming Mutagenicity.

#### 🔴 The confound this exposes, and it must be measured before §17 is written up

A censored graph is retained **with its greedy-min fallback string, which is not canonical** and is
therefore outside the completeness theorem. So above `n ≈ 60`, roughly **half** of Mutagenicity's
IsalGraph arm is not the canonical encoding at all.

§17 reports that the within-`n` correlation collapses with size. **Part of that collapse may be the
budget rather than the method** — an increasing fraction of large-`n` graphs being represented by a
fallback string, not by `pruned_canonical_string`. Those are two different findings with two
different response-letter sentences, and they are currently entangled:

- *"the representation's structural fidelity degrades with graph size"* — a property of the method;
- *"the 300 s budget forces a non-canonical fallback on large graphs, which correlates worse"* — a
  property of the **compute budget**, and fixable by raising it.

**Measurement, before T-20 quotes §17:** recompute the §17 within-`n` ρ on Mutagenicity **restricted
to non-censored graphs**, and compare against the same strata unrestricted. The confound is confined
to Mutagenicity — it is the only dataset that censors at all — so the check is cheap and exact.

- If ρ is unchanged, censoring is **not** the driver, §17 stands as a statement about the method, and
  that is a stronger claim than we can currently make.
- If ρ improves materially, §17's collapse is **partly a budget artefact** and must say so.

Either outcome is publishable; conflating them is not.

#### A3's symmetry stratum — the substitution stands, but `|Aut|` needs no cluster

The subagent could not emit the rate per **symmetry** stratum because no artifact this ticket holds
carries `|Aut|`, and correctly recorded the size-stratum substitution *inside* `censoring.json`
rather than presenting it as the stratum requested. That is the right handling.

**But its cost estimate is wrong: `pynauty 2.8.8.1` is already importable in the environment and
exposes `autgrp`.** Verified 2026-08-23. `|Aut|` over 21,720 graphs of `n ≤ 98` is what nauty is
built for and is very unlikely to need a cluster. **Probe 200 graphs, time-boxed, before concluding
anything about scale** — the same standard applied everywhere else in this ticket. If the probe is
fast, emit the real symmetry stratum and discharge A3 as written; if it is genuinely slow, keep the
substitution and record the measured reason rather than an estimate.

---

## 19. RESULT — Claim A resolved, and it runs OPPOSITE to Claim B

Full cohort, 1,578 node-count strata, 7 comparators, IUT `p = max` over both bit conventions
(§18.8). Descriptive — the confirmatory A1 family is `preregistration` §4.1's frozen six; the
seventh, `sparse6_nauty`, carries no A1 cell but does belong in a descriptive file, and restricting
the descriptive file to the frozen six was an error the subagent caught in itself.

### 19.1 The bit advantage GROWS with size

| `n` | strata | IsalGraph shorter | tie | competitor shorter | win % | median gap |
|---|---:|---:|---:|---:|---:|---:|
| 1–5 | 186 | 38 | 71 | 77 | 20.4 % | −1.2 bits |
| 6–10 | 300 | 49 | 111 | 140 | 16.3 % | +0.5 |
| 11–20 | 243 | 46 | 108 | 89 | 18.9 % | +5.8 |
| 21–40 | 350 | 105 | 112 | 133 | 30.0 % | +58.6 |
| **41+** | 250 | **114** | 89 | **47** | **45.6 %** | **+242.1** |

`competitor_shorter` falls 77 → 47 while `isalgraph_shorter` rises 38 → 114, so this is a trend, not
a ratio artefact. **§17's GED correlation collapses with size; Claim A's advantage grows with it.
The two claims do not share a cause, and Claim A is the one that survives scale.**

### 19.2 🔴 "Most compact among metric-admissible representations" is FALSE

Proposed by the orchestrator, measured, refuted. At `n > 20`:

| competitor | admissible | win | tie | LOSS | median gap |
|---|---|---:|---:|---:|---:|
| `min_dfs` | ✅ | **112** | 0 | **0** | **+214.8 bits** |
| `nauty_graph6` | ✅ | 41 | 47 | 34 | +131.4 |
| **`sparse6_nauty`** | ✅ | **0** | 51 | **71** | **−46.3** |
| `sparse6` | ❌ | 0 | 44 | 78 | −60.6 |
| `adjacency` | ❌ | 25 | 63 | 34 | +123.4 |
| `agm_cam` | ✅ | — | — | — | max `n` = 12, contributes nothing above 20 |

`sparse6_nauty` tracks `sparse6`, not `nauty_graph6` — the exact failure mode the claim was checked
against before being offered.

**State the predicate inside the sentence.** Four different numbers describe these 122 strata and
they are not interchangeable:

| predicate | count | % |
|---|---:|---:|
| **Significantly shorter than EVERY metric-admissible competitor** | **0 of 122** | **0 %** |
| Positive median gap against every one, significance ignored | 39 | 32 % |
| Never significantly beaten by any | 51 | 42 % |
| Significantly beaten by at least one | 71 | 58 % |

**0 % is the number that belongs beside the claim**, because 0 % is what the claim asserts. "32 %"
standing alone reads as "best in a third of cases" and says something much weaker. Same class of
defect as the row-level vs per-cell size null, and as "beaten on 15 of 15" vs "best on none of 15".

**And `sparse6_nauty` is the single blocking arm**: of the 39 strata with a positive gap everywhere
but no significance, it is the arm that fails to clear in **all 39** (`nauty_graph6` in 9). It is not
merely a counterexample — it blocks the claim in every stratum where the claim came closest to true.

### 19.3 What survives, and it is genuinely strong

> **IsalGraph beats `min_dfs` on 112 of 112 strata above `n = 20`, median +214.8 bits, zero losses
> and zero ties.** `min_dfs` is what `competitors.md` §2 calls *"the single most important
> comparator"*, and it is **also a canonical code** — so this is a like-for-like win, not a win over
> a different design point. Against `nauty_graph6` it is roughly even (41/47/34).

**Frozen wording:** *"IsalGraph is the most compact of the canonical-code representations, and
edge-list serialisations beat it at scale."* Say **edge-list**, not "sparsity-exploiting": it names
the mechanism rather than the outcome, and conceding the right thing is what makes the `min_dfs`
result read as the fair comparison it is. These cohorts are sparse, so an edge list exploits a
property of the data that an instruction string does not.

### 19.4 `graph6` and `nauty_graph6` carry IDENTICAL bit counts — not a copy-paste error

Verified elementwise by the orchestrator on GREC, Mutagenicity and Protein: **bit counts identical
on every graph**, while the strings differ on 650/650, 4040/4040 and 566/569 respectively.

**Mechanism:** graph6 writes the full upper triangle at fixed width, so its length is a function of
`n` alone; canonicalising permutes the bits without changing how many there are. `sparse6` vs
`sparse6_nauty` **do** differ, because an edge list's length depends on the vertex ordering.

Two identical rows in a published table invite a reviewer to assume a duplication bug. **Note the
reason in the caption.** (Protein's 3 coincident strings are graphs already in canonical order.)

---

## 20. Write-up framing — see `T-06-FRAMING.md`

The strongest **defensible** framings of T-06's results, with the scope each one requires stated in
the same sentence, and an explicit list of framings that are technically true and would still be
wrong to use. Written 2026-08-24 at the PI's request, for T-20 and the response letter.

The architecture that makes it defensible: T-06 has a **pre-registered confirmatory layer** (F0, F1,
F2 over `N_actual = 79`) reported unchanged including its negative results, and a **descriptive
layer** where the favourable framings live and are labelled as such. Reporting the pre-registered
analysis exactly as it came out is what buys credibility for everything else, so §7 of that file puts
it in the results section rather than in an appendix.

**P4 closed 2026-08-24 and it strengthens §17 rather than qualifying it.** Removing every
censored-touching pair *lowers* ρ at both bounds and both size restrictions (LB all-`n` −0.0305,
LB `n>40` −0.0354, UB all-`n` −0.0170, UB `n>40` −0.0725), so the within-`n` collapse is a property
of the representation and not of the 300 s budget. Because censoring tracks `|Aut|` rather than `n`,
the removed graphs are the most *symmetric* at their size, so the unchanged collapse further implies
structural fidelity does not depend on symmetry. The countervailing half travels with it: censored
pairs do correlate worse in isolation (0.3273 against 0.6095 at `n > 40`), and the pooled ρ exceeding
both components is a Simpson-type effect that the Δ alone would hide.

---

## 21. A second recurring failure mode, named — the consumer that accepts too many rows

§8.1 names the pattern that shrinks `N_actual`. **This is a different one, and it produced three
defects in a single day.** Recording it because the next ticket and the sibling projects will meet it.

| # | Defect | What the consumer accepted |
|---|---|---|
| 1 | `t06_ladder.py`'s `analysed` rung | `ged_positive & mask` — a **different** pair set from the one F0/F1 actually ran on, differing by exactly the GED = 0 pairs |
| 2 | `run_correlation_group`'s arm records | **two contradictory records under one identifying key**; which survived depended on emission order *and* on the consumer's dedup direction |
| 3 | `figures.py`'s `load_rows` | **both D14 arms** once P4 added the `arm` field — every §17 figure would have plotted the complete-case arm on top of the primary one, doubling `n` and biasing the curve |

**The shared shape: the artifact looks right because the consumer silently accepted more rows, or
different rows, than it should have.** None of the three raised an error. Two were caught only by
cross-checking a landed value against an independent recomputation; the third by noticing that a
producer had gained a field the consumer predated.

**Why this class is worse than an exception.** A crash is self-reporting. These three each produced a
*plausible number* — and in defect 2 the correct value survived by luck (`load_rho_rows` happened to
keep the first record, which happened to be the full-mask one), while the subagent's own verification
script kept the last and reported a phantom discrepancy. **Correctness depended on a dedup direction
nobody had chosen deliberately.**

**Mitigations adopted, in order of strength:**

1. **Make the invariant a property, not a rule.** `assert bh_primary.m == cardinality.n_actual`
   regardless of how many cells carry values — with a negative case constructed to have cells
   deliberately absent. A property nobody *can* accidentally trim beats a rule someone must remember.
2. **Test both orderings** whenever dedup decides which of two records survives. "Keep the first"
   passes one ordering and fails the other, so a single-ordering test certifies the bug.
3. **Fail loudly on an unknown enum value.** `load_rows` now raises on an unrecognised `arm` rather
   than passing it through — a consumer that predates a producer's new field should stop, not guess.
4. **Cross-check one landed cell per artifact type against an independent recomputation.** All three
   of today's were found this way or would have been.

> **Verified: nothing published moves.** The shipped `size_profile.json` is schema
> `t06.size_profile.1` and carries only `arm = "primary"`, so `load_rows` yields **1,553 rows and 582
> aggregate points** — exactly §17's stated counts. Defect 3 was **latent, not live**: it would have
> fired the moment anyone regenerated the profile with `--arm both`.
