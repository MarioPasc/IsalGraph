# T-03 — what belongs in the article about GED computation

**Written 2026-08-13 on T-03's completion.** Everything here is measured, and each item names where
in the manuscript or the letter it lands. Owners: **T-20** (manuscript), **T-14** (letter),
**T-21** (reproducibility), **T-05** (calibration ladder).

Sorted by consequence. The first three change what the paper can claim; the rest are reporting
obligations that a reviewer would otherwise raise.

---

## 1. Exact GED is not exact for 1.57 % of pairs, and 20.7 % of AIDS

**This is the single most important number T-03 produced, and it must not be buried.**

| Dataset | pairs | certified exact | interval-censored |
|---|---:|---:|---:|
| IAM Letter LOW / MED / HIGH | 3,598,699 | 100 % | **0 %** |
| LINUX | 3,916 | 98.83 % | 1.17 % |
| AIDS | 295,296 | 79.33 % | **20.67 %** |
| **Total** | **3,897,911** | **98.43 %** | **1.57 %** |

At a 60 s per-pair budget, **roughly one AIDS pair in five has no certified exact value**. Those
pairs enter the analysis as an interval `[LB, UB]` (D11), never dropped, and the rate is reported
**per stratum, never pooled** (D12).

**Where it goes**: the experimental-setup section must state the timeout, the censoring rate per
dataset, and the fact that the AIDS correlation rests on 79 % exact values plus 21 % intervals.
Any table row carrying an AIDS ρ needs a companion censoring figure.

**Why it cannot be omitted.** Dropping censored pairs would bias toward the pairs A* finishes
quickly — the small, low-symmetry ones — so the reported ρ would characterise the easy half of the
cohort while being printed under the whole cohort's header. That is the same defect class as the
connectivity discard ([data](../plan/data.md) §3) and the encoding-censoring D14 addresses.

---

## 2. The censoring rate is a property of *(cohort, timeout, machine)*, not of the cohort

**Measured, and the effect is large:**

| Same LINUX cohort, same 60 s timeout | censored |
|---|---:|
| Workstation (i7-13700KF) | **5 / 3,916 = 0.13 %** |
| Picasso `sr` (EPYC 7H12, 2.6 GHz) | **46 / 3,916 = 1.17 %** |

**A 9× difference from nothing but a slower core against a fixed wall.** A timeout-defined censoring
rate is not reproducible across hardware, and reporting one without the machine invites a reviewer
to fail to reproduce it.

**Where it goes**: state the CPU model and the per-pair budget beside every censoring figure, in the
same sentence. The reproducibility section (T-21) should say plainly that re-running on different
hardware will move the censoring rate and therefore the exact/censored split, though not the
*values* of the pairs that do complete.

---

## 3. The submitted pipeline recorded timed-out searches as exact

`benchmarks/real_data/eval_setup/ged_computer.py::compute_ged_pair` calls
`nx.graph_edit_distance(..., timeout=t)` and returns the result unless it is `None`.
**`networkx` returns its best-found-so-far cost when the budget expires** — it does not raise, and it
returns `None` only when *no* complete edit path was found at all. So a pair whose search was cut
off is stored indistinguishably from a certified optimum.

**Every "exact GED" value in the submitted study was produced this way.**

**Where it goes**: this is a candidate E-class disclosure. It is also the cleanest justification for
the new pipeline's design — we record, per pair, *whether the search completed*, and promote nothing
to exact without it. The letter can say: we no longer infer optimality from a value, we certify it
from completion.

**Do not overclaim.** On the submitted cohort the affected fraction is unknown, and on Letter it is
plausibly zero (100 % of Letter pairs complete well inside the budget here). The honest statement is
that the *mechanism* existed, not that a specific number of published values were wrong.

---

## 4. GraphEdX's published GED covers within-split pairs only

| Dataset | published pairs | all pairs | coverage |
|---|---:|---:|---:|
| LINUX | 1,685 | 3,916 | **43.0 %** |
| AIDS | 131,148 | 295,296 | **44.4 %** |

Both figures are now **measured** as the overlap between our all-pairs matrices and their published
entries, not inferred. The AIDS figure closes [data](../plan/data.md) §5's request to record
131,148's provenance, and confirms the population-matched gain as **2.25×**, not 1.62×.

**Consequence**: the submitted ρ = 0.433 (LINUX) and ρ = 0.349 (AIDS) are **within-split figures**,
which the source does not state. Our replacements are computed on 2.3× and 2.25× as many pairs and
are therefore not directly comparable. **Print both and explain, rather than substituting silently.**

---

## 5. ⚠ GraphEdX uses unit node costs — a plan premise that is wrong

The revision plan asserts ([gedlib](../plan/gedlib.md) §6, [statistics](../plan/statistics.md) D6)
that GraphEdX charges **zero** for node operations. Measured against the published file:

| pair | Δn | published | zero-node | unit-node |
|---|---:|---:|---:|---:|
| 241, 475 | 1 | 8.0 | 7.0 | **8.0** |
| 207, 377 | 3 | 8.0 | 5.0 | **8.0** |
| 135, 339 | 1 | 2.0 | 1.0 | **2.0** |
| 211, 67 | 4 | 9.0 | 5.0 | **9.0** |

**Unit 4/4, zero 0/4**, the published value exceeding the zero-node value by exactly `|n₁ − n₂|`.

**Two things follow for the article:**

1. **Do not write that the submission mixed two cost models** until the IAM Letter side is checked
   independently. That framing is D6's rhetorical lead in the current draft and it may be false.
2. **D6 still stands** on its own argument: zero node cost makes GED a *pseudo*metric, and
   validating a metric against a pseudometric reference is incoherent. That is an argument about
   cost models in general and survives intact. The recompute's remaining justifications — coverage
   (§4) and Letter having no published matrix at all — are both untouched and both sufficient.

**This also retracts a T-03 finding**: gate 0's "150 pairs below the published value, 0 above" was
the arithmetic of comparing under the wrong cost model, not evidence their reference is approximate.

---

## 6. GraphEdX's matrix is essentially exact — with two provable exceptions

Compared like-for-like under unit costs over the full overlap:

| | AIDS | LINUX |
|---|---:|---:|
| finite overlap pairs | 105,272 | 1,665 |
| agree | **105,270** | **1,665** |
| ours **higher** (would falsify our solver) | **0** | **0** |
| ours **lower** (their entry non-optimal) | **2** | 0 |

The two exceptions are `aids_train_0024`/`aids_train_0246` (ours 5, theirs 7) and
`aids_val_0016`/`aids_val_0036` (ours 7, theirs 9), both certified. Since GED is a minimum and A*
returns an achievable path, those two published entries are **provably not optimal**.

**Report this as a footnote, not an argument.** Two in 131,148 is a rounding error. The zero in the
falsifying row is the more useful number: it is independent evidence that our solver and our index
alignment are correct across 106,937 pairs.

**Tolerance matters and cost us two false alarms.** GED is integer-valued and GraphEdX stores
floats; comparisons at 1e-9 and then 1e-6 both reported storage noise as disagreement (7 LINUX
pairs, then 86 AIDS pairs, all with sub-integer deltas). **0.5 is the correct threshold.**

---

## 7. IAM Letter ships no GED matrix at all

`GED_PRECOMPUTED/Letter/{LOW,MED,HIGH}` holds **2,254 raw `.gxl` files per level** and no published
distances. Every Letter GED value in this study — and in the submitted version — was always ours.

**Where it goes**: the data section should not imply the IAM distribution supplied a reference for
Letter. It is also the second independent reason a single cost model across the cohort requires
recomputation rather than reuse.

---

## 8. Method and parameters to report (T-21, reproducibility)

- **Exact solver**: `networkx.graph_edit_distance` A*, run to completion, with completion recorded
  per pair. **Not** GEDLIB `ANCHOR_AWARE_GED` — see §9.
- **Bounds**: GEDLIB `BRANCH_FAST` (lower) and `IPFP` (upper), both symmetrised by evaluating both
  orientations and taking the minimum, because every GEDLIB upper-bound method builds its edit path
  from a *directed* assignment and is not symmetric.
- **Cost model**: `[1, 1, 0, 1, 1, 0]` — unit node and edge insert/delete, substitutions free (D6).
- **Per-pair budget**: **60 s**, the submission's own `ged_computer` default, kept unchanged
  deliberately so the comparison is like-for-like. **State it explicitly wherever a censoring rate
  appears.**
- **Hardware**: AMD EPYC 7H12 @ 2.6 GHz (`sr` nodes), 64 cores per task, one solver process per core.
- **Compute**: ≈ 2,081 core-hours for 3,897,911 pairs, of which AIDS is 99 %.
- **Determinism**: pair enumeration is by linear upper-triangle index, so the pair set is
  reproducible independent of scheduling; results merge deterministically by index.

---

## 9. `ANCHOR_AWARE_GED` is not usable as an exact solver — worth one sentence in related work

GEDLIB's `ANCHOR_AWARE_GED` is documented as exact. Measured on this cohort it is **non-deterministic
and not exact**: six fresh environments on one real AIDS pair returned `[10, 6, 6, 6, 6, 4]` where
exhaustive enumeration gives **2**, it disagreed with brute force on **4 of 18** small pairs (always
over, never under), and it reports `LB == UB` on those wrong values — a **false optimality
certificate**. No option (`--threads 1`, `--map-root-to-root`, `--search-method DFS`) restores it.
`networkx` A* was correct **18/18** on the same oracle.

**This is a defensible, self-contained empirical observation about a widely used library**, and it
directly justifies the paper's choice of solver. One or two sentences in the experimental setup, or
a footnote, with the version pinned. It should not become a claim about GEDLIB as a whole — the
`BRANCH_FAST` and `IPFP` bounds behaved correctly throughout, and the bracket held on every gate.

---

## 10. What is *not* claimable from T-03

Stated so nobody reaches for it later:

- **Not** that GraphEdX's published matrix is approximate — retracted, see §5.
- **Not** a specific count of wrong values in the submitted study — the timeout mechanism (§3)
  existed, but its incidence on the submitted runs was never measured.
- **Not** an exact-GED ceiling above n = 12. `ANCHOR_AWARE_GED` looked 70–5000× faster than A* and
  would have raised it, but it is not exact, so the ceiling stands where
  [approx_ged](../plan/approx_ged.md) §1 put it.
- **Not** a per-stratum censoring table yet. The data exists (`seconds_matrix`, `certified_mask`,
  and `aids_stage1_sampling_report.json` for the stratum assignment) but the table is T-05/T-06 work.
