# T-04a — what belongs in the article about metric admissibility

**Written 2026-08-23 on T-04a's completion.** Everything here is measured, and each item names where
in the manuscript or the letter it lands. Owners: **T-20** (manuscript), **T-14** (letter),
**T-17** (AE.3 table), **T-21** (reproducibility), **T-06** (recompute).

Sorted by consequence. The first four change what the paper can claim; the rest are reporting
obligations a reviewer would otherwise raise. **§8 is the section to read before quoting anything.**

Serves **AE.4a**, **AE.3**, **R1.1**, **R1.2a/b**, **R3.6a**.

---

## 1. IsalGraph clears the size null on 1 of 5, and is the best representation on 0 of 15

State this first. A reviewer who has our data computes it in an afternoon, and R3 checked thirteen of
thirteen checkable claims last round.

**Suite 1, against T-03's certified exact GED** — ρ(Levenshtein on the pruned canonical string, GED)
minus ρ(|n₁−n₂|, GED), paired graph-level bootstrap, 2,000 resamples, seed 42, both arms on the
identical pair set and the identical resamples:

| dataset | Δρ vs the size null | 95 % CI |
|---|---:|---|
| **Letter LOW** | **+0.026** | **[+0.008, +0.046]** |
| Letter MED | −0.044 | [−0.068, −0.014] |
| Letter HIGH | −0.220 | [−0.260, −0.183] |
| LINUX | −0.239 | [−0.320, −0.157] |
| AIDS | −0.528 | [−0.571, −0.482] |

**All five exclude zero, Letter LOW included.** Letter LOW is a genuine separation and must be
reported as one — it is small, its interval excludes zero, and calling it "not a separation" would
be a conservative error that gives the reviewer nothing and costs us the only positive Suite-1 row.

**Best representation per record**, each under its own selected primary distance, over the 15 records
(5 Suite-1 exact + 5 Suite-2 LB + 5 Suite-2 UB): `min_dfs` **4**, `agm_cam` **3**, `wl_subtree` **3**,
`sparse6_nauty` **3**, `nauty_graph6` **2**, **IsalGraph 0**.

**min-DFS out-correlates IsalGraph on all five Suite-1 datasets**, by +0.047 (Letter LOW) to +0.277
(AIDS) on this draw. Both halves go in: min-DFS wins the proxy comparison, and IsalGraph wins
compactness on 9 of 10 cohort profiles (T-04). Reporting one without the other is the failure mode
R3.6a already flagged.

**Where it goes**: §3.2.5 / Tab. 3, and the letter's answer to R1.1. **Provenance**: measured by
T-04a, `results/paired_null_ci.json` and `results/f5_200.json`.

**Why it cannot be omitted.** The submitted manuscript's ρ has no null baseline beside it. A
correlation of 0.93 against GED on a cohort where |n₁−n₂| alone scores 0.90 is not evidence that the
encoding tracks structure, and stating the null is what converts the number from a claim into a
measurement. This is the same discipline D2 imposes for the bootstrap.

---

## 2. The Suite-2 answer is *undetermined*, and saying so is the stronger result

Against T-05's bounds, the verdict on whether IsalGraph clears the size null **flips with the end of
the bracket on 5 of 5 Suite-2 datasets**:

| dataset | vs LB (`BRANCH_FAST`) | vs UB (`BIPARTITE`) |
|---|---:|---:|
| GREC | −0.214 | **+0.122** |
| AIDS-IAM | −0.248 | **+0.027** |
| COIL-DEL | −0.082 | **+0.197** |
| Mutagenicity | −0.289 | **+0.196** |
| Protein | −0.233 | **+0.383** |

**All ten differences exclude zero.** Fifteen of fifteen across both suites.

**The mechanism is measured and it exonerates the bound.** `ρ(|n₁−n₂|, LB) = 0.960–0.998` — the lower
bound very nearly *is* the size null, so no representation can beat it and the comparison is
degenerate by construction. This is **not** a defect in `BRANCH_FAST`, which is a proven lower bound
(Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018) that T-27 selected on 46.8 M measured bound
evaluations. `ρ(|n₁−n₂|, UB) = 0.460–0.754`, and on Suite 1, where truth exists,
`ρ(|n₁−n₂|, exact) = 0.713–0.920` — **between** the two arms. The bracket is valid; neither arm alone
is a stand-in for GED.

**Where it goes**: §3.2.5, as a reported non-result, and [approx_ged](../plan/approx_ged.md) §4's
no-interpolation rule cited as its consequence. A midpoint would have produced one confident answer
to a question the data leaves open, five times over.

**Why this is worth more than a number.** AE.1 asks whether the bounds are tight enough to support a
conclusion. Answering "on this question, at these sizes, they are not, and here is the correlation
that shows why" answers AE.1 directly and pre-empts the reviewer who would otherwise ask what the
bracket width means for the claim. A single interpolated ρ would have invited exactly that question
and had no defence.

---

## 3. The contribution that survives is **canonical *and* edit-distance-compatible**

Not "shortest" — sparse6 wins on bits. Not "best correlated" — min-DFS and AGM win on ρ. The
conjunction is what nothing else in the pool has, and T-04a supplies both halves as measurements
across three classes of representation.

| class | representative | relabelling-invariant? (E1, ψ) | complete? (E2, collisions) |
|---|---|---|---|
| **I — non-canonical serialisation** | `adjacency`, `graph6`, `sparse6` | **no**: ψ 0.07 – **1.148** | (moot) |
| **II — canonical but incomplete** | `wl_subtree` | ψ = 0.0000 | **no**: **45 / 183,016** = 2.46 × 10⁻⁴ [1.79, 3.29] × 10⁻⁴ |
| **III — canonical and complete** | `nauty_graph6`, `sparse6_nauty`, `agm_cam`, `min_dfs`, **`isalgraph_pruned`, `isalgraph_canonical`** | ψ = **0.0000**, all eleven draws | **0 collisions**; zero set **≡** VF2-certified isomorphic set |

**ψ** is the median distance between a graph and a relabelled copy of itself, in the units of the
distance. It is F3's binary predicate made continuous, and it is the number to print: graph6 peaks at
**1.003 [0.953, 1.054]** on LINUX and sparse6 at **1.148 [1.111, 1.187]** on AIDS — a representation
disagreeing with itself by more than one edit. All 33 intervals for the excluded three exclude 0; all
77 rows for the canonical seven are exactly 0.

**Exhaustively, not just on the cohort**: the invariant set of the n² family is **exactly `{K_n}`**,
over all 995 connected graphs to `n = 7` under full `n!` enumeration — **1,866,256** distinct
labelled graphs (OEIS A001187). There is no non-trivial subfamily on which an adjacency serialisation
is relabelling-invariant.

**Completeness upper bounds**: the six complete invariants collide 0 times, ≤ 2.0–3.4 × 10⁻⁵ by the
rule of three at 95 %. For WL, on LINUX and AIDS — which hold **no duplicate graphs** — every zero it
emits is a false isomorphism certificate: **1/1** and **11/11**, AIDS CI [0.715, 1.000]. **The LINUX
figure rests on one pair and establishes nothing on its own**; AIDS is the load-bearing row. As a
fraction of all pairs, 0.026 % and 0.055 %. K₃,₃ vs the triangular prism: WL exactly **0.0**, all six
complete invariants separate them.

**Where it goes**: §1 (the contribution statement), §3.2.5, and **T-17's AE.3 table** — this is the
evidence its "properties" column needs, on R1.2's five axes, populated from measurement rather than
asserted.

---

## 4. Every representation's primary distance was chosen before any GED was looked at

**The selection rule is F5-blind by construction** — cheapest candidate passing F1 (well-defined on
100 % of pairs), F2 (metric axioms), F3 (relabelling invariance), F4 (non-degenerate), with F6
(cost) as the tie-break and F5 (correlation with GED) never permitted to enter. A test asserts the
selector's import closure reaches no GED loader.

| representation | primary distance |
|---|---|
| `adjacency`, `graph6`, `sparse6` | **none admissible** — F3 = 1/50 |
| `nauty_graph6`, `sparse6_nauty`, `agm_cam`, `min_dfs`, `isalgraph_pruned`, `isalgraph_canonical` | `levenshtein` |
| `wl_subtree` | `kernel` |

**`padded_hamming` is primary for nothing**, losing the F6 tie-break to `levenshtein` by **68×** on
`nauty_graph6` (0.0010 vs 0.0704 ms/pair) and **8.6×** on `agm_cam`. `hamming` is undefined on ~97 %
of pairs everywhere (F1 = 0.032–0.035); `padded_hamming × sparse6` is defined on **no** pair
(F1 = 0.0) because sparse6 has no positional frame to pad into.

**`k = 3`** over the pre-registered seven-member Claim-B comparator set (`graph6`, `sparse6`,
`nauty_graph6`, `adjacency`, `agm_cam`, `min_dfs`, `wl_subtree`): `adjacency`, `graph6` and `sparse6`
have no admissible distance on any suite. **T-06 applies it** — `N_actual = 182 − 15k − 8d`.

**Where it goes**: §3.2.5's protocol paragraph and the supplementary 66-cell grid. **T-14** should
say plainly that the rule was frozen in a committed design note before any agent started and that
every cell, including the failures, is printed.

### 4a. Why F5-blindness is not pedantry — the trap, measured

`adjacency` — which fails F3 at 1/50, i.e. changes its answer when you relabel the graph and nothing
else — nonetheless **beats IsalGraph significantly on 3 of 5 datasets** on all-pairs ρ against exact
GED: AIDS **+0.500** [+0.443, +0.558], LINUX **+0.262** [+0.160, +0.362], Letter HIGH **+0.123**
[+0.079, +0.174], all Holm-adjusted p = 0.005; Letter MED +0.002 (ns), Letter LOW −0.037.

On **equal-`n`** pairs it wins **0 of 5** and loses significantly on three.

**Mechanism**: `ρ(d_adjacency, |n₁−n₂|) = 0.83–0.93`. The adjacency distance is very nearly a size
proxy; exact GED is substantially a size proxy; the two correlate through size, not through
structure. A selection rule with sight of F5 would have admitted it.

**Where it goes**: §3.2.5 as the justification for the admissibility screen, and the letter's answer
to R1.2b. **This is the single most persuasive item in the ticket** — it shows the screen doing work
against our own interest, on our own data.

---

## 5. The equal-`n` view is not supplementary material

On LINUX the **paired** CI on the difference is [−0.301, −0.007] and excludes zero; the **unpaired**
comparison of two marginal intervals is [−0.333, +0.034] and does not. **A difference between two
quantities is tested with an interval on the difference, never by eyeballing two marginal
intervals.** Every comparison in the paper follows the paired form, on shared resample indices
(D7).

**And the size null must be restricted to each representation's own pair set.** On Mutagenicity,
where IsalGraph loses 14/200 graphs to the canonicalisation budget and *every censored graph is
larger than every kept one* (mean 75.8 nodes, max 97, against 25.4 and 48), the whole-cohort null
scores 0.7538 and the restricted null 0.6363 — the UB margin moves **+0.078 → +0.196**. The
representation's own ρ does not move at all. This is D14's censoring bias appearing inside the
**baseline** rather than inside the arm, which is the direction nobody checks. `min_dfs` is censored
on Mutagenicity too, on a *different* 14 graphs, with a restricted null of 0.6817 — so one null per
dataset would have been wrong for at least one of them.

**Where it goes**: §3.3 (statistics) and **T-06's implementation**, which must compute the null per
representation, not per dataset.

---

## 6. The metric axioms hold — reported as a correctness check, not as a finding

**0 violations in 9,881,851 checks** (identity, symmetry, triangle inequality) over all **467,180**
triples of the 142 connected graphs on `n ≤ 6`, `worst_excess = 0.0` exactly — so the result is
non-vacuous rather than trivially satisfied.

**This is a check on our implementation, not a discovery.** Levenshtein is a metric by construction
(Levenshtein, 1966) and a metric composed with an injective encoding stays one; the exhaustive run
verifies that our encoders and metric wrappers do not break the property, which is what a
reproducibility reviewer wants and what a novelty claim must not be built on. Write it that way.

**Where it goes**: §3.3 or the supplementary, one sentence. **T-21** carries it as an artifact check.

---

## 7. Method and parameters to report (T-21, reproducibility)

- **Sample**: 200 graphs pooled over ten datasets, binned by node count into
  `[2,5] [6,9] [10,12] [13,20] [21,40] [41,∞)`, quotas `[33, 33, 33, 33, 34, 34]` with the remainder
  to the largest strata, one `random.Random(42)` consumed stratum by stratum. `n` 2–83, mean 20.92,
  suite split **51 / 149**. **A function of `(ALL_DATASETS, 200, 42)` alone**, reproducible without
  reference to anything that ran before it. **LINUX contributes zero graphs** — 89 graphs at
  `n ∈ [4,10]` against thousands in the strata it falls in. Report that, do not repair it.
- **F3 sub-sample**: 50 graphs drawn from the 200, quotas `[8, 8, 8, 8, 9, 9]`, 20 relabellings each
  via `fixtures.shuffled_copy`, each copy encoded once.
- **GED reference**: Suite 1 = T-03's certified exact GED, `networkx` A*, D6 unit cost model
  `[1, 1, 0, 1, 1, 0]`. Suite 2 = T-05's GEDLIB bracket, LB `BRANCH_FAST` and UB `BIPARTITE`,
  **reported as two values and never interpolated**.
- **Encoder budgets — name these wherever an F0 or a censoring rate appears.** `timeout_s = 2.0`,
  `max_projections = 50,000`, `search_nodes = 200,000`. **These are the backends' own budgets, not
  D14's 300 s**, and a different budget gives a different F0. T-06 re-measured `isalgraph_pruned` at
  a 30 s budget and it completed **150/150 across all ten datasets, zero kills** — so the 9/200 below
  is a property of *(representation, budget, machine)*, not an encoder ceiling.
- **Encodability (F0), at those budgets**: `agm_cam` 98/200 (101 `SuiteScopeError`, 1
  `AGMBudgetExceeded`), `isalgraph_canonical` 99/200 (101 `SuiteScopeError`), `isalgraph_pruned`
  191/200 (9 `CanonicalizationTimeoutError`), `min_dfs` 192/200 (8 `MinDfsBudgetExceeded`); the other
  seven 200/200. **All four are 1.00 on Suite 1** — every exclusion is a Suite-2 graph.
  **`SuiteScopeError` is a scope decision and must never be summed with a budget outcome**;
  `preregistration.md` §5 charges them differently.
- **Statistics**: Spearman ρ; paired graph-level bootstrap, **2,000 resamples**, seed 42, percentile
  interval, the same resamples reused across representations within a dataset (D7); Holm adjustment
  within the E4 family of five; Clopper–Pearson for proportions (D-A4); rule of three for zero-event
  upper bounds.
- **Engine**: `isalgraph.engine() == "cpp"`, conda env `isalgraph-cpp`, Python 3.11.15, gcc 12.2.0.
  `networkx` 3.6.1, `pynauty` 2.8.8.1, `grakel` 0.1.8, `rapidfuzz` 3.14.5.
- **Hardware and compute**: local workstation, no cluster. The whole campaign — 66 cells, F5, E1–E4,
  the paired bootstrap — is a workstation job of hours, not core-hours. **No Picasso, no SLURM.**
- **Determinism**: every draw is seeded and reproducible from its parameters; F5 reproduces T-04's
  `corrected_rho_table.json` at **max |delta| = 0.0000**.
- **Artifacts**:
  `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/experiments/metric-admissibility/` —
  self-contained, with `PROTOCOL.md`, `README.md`, `results/` (7 JSON), `code/` (13 files) and
  `manifest.json` carrying sha256 over 22 files. `results/paired_null_ci.json` is the authority for
  every interval quoted anywhere.

---

## 8. What is *not* claimable from T-04a

Stated so nobody reaches for it later.

- **Not** that IsalGraph is the best proxy for GED. It is best on **0 of 15** records. What it is, is
  the only representation in the pool that is canonical, complete, edit-distance-compatible **and**
  reversible — and those are four properties, not a ranking.
- **Not** that the Suite-2 result is negative. It is **undetermined**: the verdict flips with the end
  of the bracket on 5 of 5. Reporting either arm alone would be picking the answer.
- **Not** that `BRANCH_FAST` is a poor bound. It is a proven lower bound selected on measurement; the
  degeneracy is that at these size spreads a valid lower bound and the size null are nearly the same
  statistic. Blaming the bound would be wrong and a reviewer who knows the 2018 paper would catch it.
- **Not** a collision *rate* for the canonical string without its enumeration window. Zero events over
  108,290–183,016 pairs bounds the rate at ≤ 2.0–3.4 × 10⁻⁵; it does not establish 0.
- **Not** the WL LINUX false-certificate rate of 1/1 as a standalone number. **It rests on one pair.**
  AIDS's 11/11 with CI [0.715, 1.000] is the citable one, and even that interval is wide.
- **Not** the metric-axiom result as a contribution. It verifies our implementation; the mathematics
  was never in doubt.
- **Not** any F0 or censoring figure without its budget. `isalgraph_pruned`'s 9/200 is a 2.0 s
  artefact; at 30 s it is 0/150. A rate whose timeout is unstated is not reportable.
- **Not** `ψ` for `adjacency` under `padded_hamming` as evidence that padding helps. It is **0.988**
  against `levenshtein`'s 0.072 — **14× worse**. Padding aligns positions, and positions are what a
  relabelling permutes.
- **Not** T-04's Suite-2 LB figure of −0.295 for Mutagenicity, or the UB figure of +0.078. Both were
  computed against a null on the wrong pair set; the measured values are **−0.289** and **+0.196**.
- **Not** the production distance matrices. T-04a selects the distances; **T-06 computes the
  matrices** on the full cohorts. Every number here is on a 200-graph draw.
