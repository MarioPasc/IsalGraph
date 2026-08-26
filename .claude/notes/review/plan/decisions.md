# Decisions — signed, and still open

**Everything here is either settled or has a named owner and a date.** A ticket that finds itself
re-litigating a signed decision should stop and check this file first.

Related: [tickets](tickets.md) · [schedule](schedule.md) · [demands](demands.md)

---

## 1. Signed decisions

| # | Decision | Detail |
|---|---|---|
| 1 | **Re-execute everything with the C++ engine** | [data](data.md) §4 |
| 2 | Competitors enter the experiments, vendored **as backends in the IsalHG `iso_backends` style** | [competitors](competitors.md) §1 |
| 3 | **Merge all splits.** GED is symmetric and carries no train/test semantics | [exact_ged](exact_ged.md) |
| 4 | **Exact GED for `n ≤ 12`; proven bracket above it, on larger real datasets** — replaces the controlled-edit cohort, whose constructed pairs live in a `k`-ball around a base graph and are not distributed like real pairs | [approx_ged](approx_ged.md) §1 |
| 5 | **Decline the sequential-model experiment**; defer to future work, downgrade LM claims to conjecture | §3 below |
| 6 | Render an S2G/G2S example figure as in IsalSR / IsalHG | **author preference, no reviewer demand** — T-09, and the first page cut |
| 7 | Ignore the page budget while drafting; trim at the end | [manuscript](manuscript.md) §3.3 |
| 8 | gSpan vendored from `github.com/LasseRegin/gSpan` | [competitors](competitors.md) §2 |
| 9 | **[29] IsalChem is published** — `ThurnhoferHemsi:2025`, *J. Chem. Inf. Model.* **65(15):7936–7955, 2025**, already cited. Read the paper; the source repo is the implementation cross-check. **D19's [29] half is directly resolvable**. **T-07, 2026-08-26: [29] is CC BY open access** — PMC12344769, PMID 40720985 — so the delta rests on the paper itself. Caveat: the PMC/BioC text **strips all three algorithm listings**, every equation and every table body; anything depending on [29]'s pseudocode needs the publisher PDF. **D19 RESOLVED, both halves** | [corrections](corrections.md) §4, §7 |
| 10 | **[28] is and will remain arXiv-only** | [compliance](compliance.md) §5 |
| 11 | **GED comes from GEDLIB, not our own code.** ~~Exact = `ANCHOR_AWARE_GED`~~ (retracted by T-03 — non-deterministic and not exact; exact GED is `networkx`). Proven lower = **`BRANCH_FAST`, UPHELD and upgraded by T-27** — the survey §5.2.4 *proves* it equivalent to `BRANCH` under constant edge costs, and T-27 measured them identical on all 3,836,827 certified pairs, so this no longer rests on 400 LINUX pairs. ~~Proven upper = `IPFP`~~ → **`BIPARTITE` (T-27)**: `IPFP` is the tightest UB measured but costs **808× the frozen M7 gate**. **Options are part of the specification** — GEDLIB's UBs vary on 91–94 % of pairs at defaults | [gedlib](gedlib.md), [approx_ged](approx_ged.md) §2 RESULT |
| 12 | **Cohort locked** to the IAM Graph Database; **TUDataset dropped**. Reaches **n = 98 retained**, an 8.2× extension. Re-affirmed on the corrected number (the 417-node graph is disconnected). **Re-derived and re-affirmed 2026-08-13 (T-01)**: **16,370 graphs / 21,710,892 pairs**, ~~19,670 / 40,024,242~~ — see decision 27 | [data](data.md) §1–§2, §7 |
| 13 | **Labels are tiered.** The manuscript never claimed label handling; R1.3 asks for a discussion; a labelled variant is a different paper. **Tier 0 is not optional** | [labels](labels.md) |
| 14 | **Encoding-censored graphs are analysed, not dropped** — greedy-min fallback plus a complete-case sensitivity arm | [statistics](statistics.md) D14 |
| 15 | **No marked-up manuscript.** The letter's per-comment pointers are the change map | [manuscript](manuscript.md) §4.4 |
| 16 | **Query patcog@elsevier.com on day 1**: does supplementary material count toward the 35-page limit? The whole page strategy branches on the answer | [manuscript](manuscript.md) §3.2 |
| 17 | **T-16 rejected** — `wl_pruned_canonical` is not built. The WL *measurement* moves into T-13 | §2 below |
| 18 | **Competitor distances are selected by measurement (T-04a), not by assertion**, with the rule fixed in advance and ties broken on cost — **never on correlation with GED** | [competitors](competitors.md) §3 |
| 19 | **T-09 and T-10 merged**; both figures double as the refreshed **graphical abstract**, submitted separately and costing no manuscript pages | [manuscript](manuscript.md) §2 |
| 20 | **Disclose E1–E12**, but the reviewer's comment is answered first — ordering rule locked | [manuscript](manuscript.md) §4.3 |
| **21** | **T-03 runs in two stages** — a pre-declared stratified stage 1 is the reported analysis; the census runs unattended behind it and supersedes stage 1 only if it lands before the T-20 freeze | [exact_ged](exact_ged.md) §3 |
| **22** | **The cohort is re-derived, not ported.** 15 of 16 measurement scripts are gone and the whole Suite-2 half of the cohort table has no reproducing code. T-01 writes a tracked `cohort_audit.py` + IAM-GXL loader + tests and re-measures all ten datasets, **retained and discarded**. What it measures becomes the table | [data](data.md) §6, T-01 |
| **23** | **The confirmatory family is frozen at `N_max = 197`**, in three fixed-sequence families (F0 calibration gate 5, F1 bracket gate 10, F2 primary 182), BH-FDR at q = 0.05 within each | [preregistration](preregistration.md) — **authoritative** |
| **24** | **BH is computed over `N_actual`, with `N_max`, the exclusion list and a BH-over-`N_max` sensitivity column all printed.** Legitimate because T-04a's exclusion rule is **F5-blind** — ties break on cost, never on correlation with GED | [preregistration](preregistration.md) §5 |
| **25** | **D13 is confirmatory** (family F1), and `ρ(Lev, UB)` gets **no** per-dataset rows in the primary family. The upper bound is reported in full; its confirmatory weight is carried by the bracket-invariance claim rather than by 70 near-duplicate tests | [preregistration](preregistration.md) §3–§4.3, [approx_ged](approx_ged.md) §4 |
| **27** | **IAM datasets are enumerated by their split index, not by their directory.** Decision 3 merges *splits*, and the union of the splits is the dataset. This binds only **COIL-DEL**, where 7,200 `.gxl` files ship but the index defines **3,900** — exactly class-balanced, 100 classes × 39, while the other 3,300 carry **no class label**. Cost: the pair headline falls from 10.3× to **5.6×**; `n_max = 98`, `n̄ = 31.68` and the density span are **unchanged**, so AE.1's evidence is untouched | [data](data.md) §1.3 |
| ~~**26**~~ | ~~Both bracket ends are selected by measurement (T-27), not by default~~ → **CLOSED 2026-08-13 by T-27.** 60 cells, 46,774,932 evaluations, 0 M4 violations, ≈ 7 core-h. **LB `BRANCH_FAST` (5 of 5), UB `BIPARTITE` (5 of 5 by elimination).** The upper end is a **constraint** outcome — `BIPARTITE` is the loosest of seven and wins only because the frozen M7 gate excludes the rest; **PI ruled frozen-gate-primary plus a disclosed sensitivity arm.** Costs carried to T-05/T-06: D13 fires on **2 of 5** datasets, and `BIPARTITE`'s error grows **~10× faster in `n`** | [approx_ged](approx_ged.md) §2 RESULT, [REPORT](/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-27-ged-bound-bakeoff/REPORT.md) |

---

## 2. Decision 17 — why `wl_pruned_canonical` is rejected, not deferred

No reviewer or editor asked for it; it entered the board as a transfer from IsalSR and IsalHG, which
both carry a WL-pruned variant. Nothing in `mail.txt` requires a new canonicalisation algorithm, and
shipping one in revision would be **a new algorithm introduced during revision** — changing Tables 2
and 3 and Figures 2 and 4, needing its own claim scoping, **in a round whose opening comment (R3.1)
asks whether the contribution is substantive enough.** It would invite "is this a contribution or a
patch?" at the worst possible moment. **Removing it returns 3–4 days.**

**What is retained**: the WL finding as a *measurement* inside T-13 — 1-WL is **2.4–2.6× finer** than
the incumbent triplet pruning key and **strictly subsumes** it. That answers R3.7d's request for a
*characterised* worst case, costs hours instead of days, needs no parity re-proof and introduces no
new claim to scope. See [data](data.md) §4 and [corrections](corrections.md) §5.

---

## 3. Decision 5 — the sequential-model decline

R3.2's modal is the **softest in the report** — "*would substantially strengthen*", against "should
provide" / "should be narrowed" / "should be described". A credible sequence-model study is a paper,
not a subsection.

**The decline is only defensible if the claims come down with it. Non-negotiable:**

| Location | Required change |
|---|---|
| `main.tex:122–126` (abstract) | "language-model-compatible … **with direct applications in** graph similarity search, graph generation, graph-conditioned LM" → format compatibility as a **property**; applications as conjecture or dropped |
| `introduction.tex:35–37` | keep as motivation, **explicitly not a result** |
| `conclusion.tex:76` | already hedged — leave |
| `conclusion.tex:88–95` | **expand**: name the Transformer/LSTM study as the designated next step, citing [28] and [29] as templates |
| Limitations | **add R3.7a item 3 in substance**: no sequential model and no downstream pattern-recognition task is evaluated. **Requirement modal — the decline may not absorb it** |

**The decline is undercut by our own R3.1 table unless pre-empted.** T-07's delta table documents that
**both predecessors ran a sequence model and this paper does not** — R3.2's exact argument, in our own
words. Write the row as a stated scope decision; see [corrections](corrections.md) §4.1.

**Contingency, with a date.** If S-f's extension is granted *and* T-03 has finished, a minimal arm may
run: character-level Transformer, ≤ 2 M parameters, on canonical strings, **graph classification only**,
on the datasets that already carry class labels (~~Letter 15 classes, GREC 22~~ → **see the correction
below**; Mutagenicity 2, Protein 6, AIDS 2, COIL-DEL 100). Baselines: the WL subtree kernel (already computed) and the same model on
the **competitor** strings — which makes it a *representation* comparison rather than a weak claim
about Transformers. Fixed splits, one seed set, no architecture search. Reported as a **feasibility
demonstration**, explicitly not a benchmark result.

> ## ⚠ CORRECTED 2026-08-15 (T-05) — the class counts above are RAW counts and **two are false of
> the filtered cohort**. The contingency survives; its dataset list shrinks.
>
> The counts are what the IAM `.cxl` indices ship — re-verified, all correct **as raw figures**. They
> are not the counts that survive `require_connected`, which every Suite-2 cohort applies. Measured
> by T-05's exporter on the real tree:
>
> | dataset | raw classes | **retained after the filter** |
> |---|---:|---:|
> | IAM Letter LOW | 15 | **9** |
> | GREC | 22 | **17** |
> | IAM Letter HIGH | 15 | 15 |
> | Mutagenicity / Protein / COIL-DEL | 2 / 6 / 100 | 2 / 6 / 100 |
> | **LINUX**, **AIDS (GraphEdX)** | — | **0 — no class label at all** |
>
> **What this changes**: any manuscript sentence of the form "Letter, 15 classes" or "GREC, 22
> classes" is **false of the cohort actually analysed**, and a graph-classification arm cannot run on
> LINUX or AIDS-GraphEdX at all — `graphedx_loader` has no label field. This is the labels
> counterpart of the size-biased connectivity discard already recorded in §7. **Owners: T-18, T-06.**
>
> **What survives**: the contingency itself, its baselines, and its framing as a feasibility
> demonstration. Only the dataset list and the printed counts change. The realised count is a
> **measured output** recorded per file in `exported_suite2/*.npz` `metadata` and in that directory's
> `manifest.json` — never an assertion.

> **The contingency is live only if the extension is granted.** [schedule](schedule.md) establishes
> there is no slack — the critical path is 27.5–28.0 days minimum in a 19-day window — so an
> unconditional go/no-go on 08-22 would spend a decision cycle concluding what the schedule already
> knows.

**Residual risk**: R3 may hold the line in round 2. Mitigation: we chose the comparison the **Area
Editor endorsed** (AE.3) over the experiment **one reviewer suggested**, and the letter frames it as
exactly that exchange.

---

## 4. Open — awaiting sign-off

| # | Decision | Owner | Due | Blocks |
|---|---|---|---|---|
| ~~**S-e**~~ | ~~Validation gate 2~~ — **RESOLVED 2026-08-12 by execution, option A.** See §6 | — | closed | — |
| **S-f** | **The schedule does not fit** — extension, stage T-03, cut, or absorb | PI (Ezequiel sends) | **2026-08-13** | everything downstream of T-06 |
| **S-g** | **Two over-scope cuts**: bliss/Traces (1.0 d); split the T-09 bundle | PI | **2026-08-14** | T-04 backend build |
| **S-h** | **Bibliography** — 16–17 slots requested against 12 | PI | **2026-08-16** | T-19 search strategy |
| **S-d** | **Which [labels](labels.md) tier?** | PI | **2026-08-18** | T-06 configuration |

### S-f — the schedule does not fit

**Floor**: the board is **93.5 days upper / 54.8 lower** and the declared critical path is **27.5–28.0
days serial** against a **19-day window**. Full arithmetic in [schedule](schedule.md) §1.

| Option | Effort | Buys | Risks |
|---|---|---|---|
| **B — request a deadline extension** from patcog@elsevier.com | one email | Elsevier routinely grants 2–4 weeks on major revisions; converts an infeasible path into a feasible one | an extension request can read as a struggling revision — **against that, `:67` says these concerns "will strongly influence the potential impact of the work", which argues for doing them properly rather than quickly** |
| **E — stage T-03** (decision 21) | one paragraph of protocol | **~900–1,550 core-h and 2–5 elapsed days off the critical path, census kept** | the supersession rule must be written **before** either stage runs, or the choice between two ρ values becomes outcome-dependent |
| ~~A — subsample only~~ | — | dominated by E | the exact-GED story becomes "stratified sample" with no path back |
| **C — cut to fit** | see [schedule](schedule.md) §3 | keeps the deadline | **the cut list returns ~2 days against a 7.5-day lower-bound overrun. It is not large enough to close the gap** — a finding, not an opinion |
| **D — accept and absorb** | — | — | T-06 gets 5 of the 10–14 days it needs, on the ticket every downstream artifact depends on |

**Recommended: B immediately, with E as the technical structure.** B is nearly free, does not degrade
the science, and pairs naturally with decision 16's day-1 query to the same mailbox. E is free and is
the only option that shortens the critical path without cutting anything.

### S-g — two over-scope cuts, 1.0–1.5 d

Both are **sub-ticket** items, so neither appears in a cut order that operates on whole tickets.
Neither is on the critical path.

| Item | Returns | Recommendation | Counter-case |
|---|---|---|---|
| **bliss / Traces backends** | **1.0 d** + a bibliography slot | **Cut.** Absent from the `ReprBackend` set; functionally duplicate nauty; produce no table row; requested by nobody | cheap insurance if `pynauty` fails to build, which would otherwise take the graph6 **and** AGM rows down with it |
| **T-09 split** | **0.75 page** (the number that matters), 0.5–0.75 d | **Split the cut.** The search-space schematic answers **R3.7c** and its renderer already exists (~2–3 h); the S2G/G2S worked example answers **no demand at all**. Bundling protects the unasked-for figure behind the requested one | both feed the graphical abstract, which costs no manuscript pages — but **that argument does not distinguish them** |

> ## ✅ S-g RESOLVED 2026-08-25 — **neither was cut**; both were built, and the split was unnecessary
>
> The recommendation was to split the cut: keep the schematic (R3.7c), drop the worked
> example (no demand). Measured, the split had no cost to make and the "~2–3 h" estimate for
> the schematic was the wrong half to worry about — its renderer already existed and needed
> only re-pointing. The worked example was the work.
>
> **The counter-case is now decided, not by argument but by measurement.** The row noted that
> "both feed the graphical abstract, which costs no manuscript pages — but that argument does
> not distinguish them". It distinguishes nothing because **the graphical abstract was not
> regenerated at all**: its panel (b) carries numbers T-06 retired, so it cannot be reprinted
> in any form until T-24 rebuilds both panels. The no-page-cost argument supports neither
> figure and should not be used for either.
>
> **What survives**: the cut order itself. [manuscript](manuscript.md) §3.2 items 10 and 11
> are still the only two entries no reviewer requires, and item 11 is still first to go. The
> four worked-example panels are independent files, so that cut is now partial rather than
> all-or-nothing.

### S-h — bibliography

Full arithmetic and a fitting 12-slot allocation: [compliance](compliance.md) §2.
**Recommended: accept 55 as the working ceiling and re-scope T-19 to ≥ 4 additions dated 2025–26, at
least 3 at PR-field venues other than the PR journal, self-citations excluded.** As previously
written, T-19's criterion was satisfiable **without adding a single external reference**.

### S-d — labels tier

Full tier table and costs: [labels](labels.md) §2.
**Recommended: Tiers 0–1 committed, Tier 2 conditional on T-03 landing early, Tier 3 declined.**
**No reviewer asked for Tier 2**; its justification is round-2 insurance, which is legitimate but is
not a demand. Due one day before T-06 launches on a board that does not fit — deciding a default now
removes a gate from the critical path.

---

## 5. Resolved by measurement — do not reopen

| Item | Resolution |
|---|---|
| Letter graph-count reconciliation | **Closed.** The filter is `min_nodes = 2`, `require_connected = True`; applying it reproduces the manuscript **exactly** |
| Cohort / GREC | **Closed.** Add Mutagenicity, Protein, COIL-DEL, AIDS-IAM **and GREC**; drop COIL-RAG, Fingerprint, Web. GREC's 59.1 % retention is misleading — its discard is **size-unbiased**, the cleanest in the cohort |
| Exact-GED scope | **All-pairs approved**, then **staged** (decision 21). Applies to the five original datasets only |
| One cost model | **Unit node + unit edge** (D6). Published GraphEdX values will no longer match ours; stated in the text |
| Primary large-`n` reference | ⚠ **REOPENED 2026-08-13 by decision 26 — narrowed, not reversed.** The measurement stands (ρ(exact, LB) = **0.859** vs ρ(exact, UB) = **0.522**, bias **−26 %** vs **+135 %**) but its **population is 400 LINUX pairs at n̄ = 8.71**, licensing a regime to `n = 98`, and the UB arm compared *our own BP*, not `IPFP`. **T-27 re-selects both ends per dataset against T-03's 3.9 M certified exact values.** Quote the figure only with "on 400 LINUX pairs" attached |
| Bounds implementation | **GEDLIB** (decision 11) |
| Kendall τ-b | Spearman primary, τ-b as a robustness check (D1) |
| Confirmatory vs exploratory | **Decided** ([statistics](statistics.md) §9) and **the family is now enumerated and frozen** — [preregistration](preregistration.md), `N_max = 197`, T-02 closed 2026-08-13. ~~Outstanding: the family must be enumerated and its cardinality frozen in T-02 before T-06 runs~~ |
| G2S timeout | **Keep at 300 s**, record per-graph time, report the rate per stratum — and D14 fixes what the analysis does with a censored graph |
| Symmetry stratification | **Adopted** ([statistics](statistics.md) §8) |
| Exhaustive canonical above n = 12 | Measured — **fails on 55 % of Protein graphs**; report the pruned/exhaustive gap **as a result** |

---

## 6. S-e — resolved by execution, 2026-08-12

**S-e recommended option C then B — spot-check and retire. Option A was taken instead**, because it
turned out to cost hours rather than the budgeted 0.5–1 day, and because retiring the gate would have
left the *primary large-`n` reference* decision resting on numbers no surviving artifact could
produce.

**Delivered**, all tracked in the repository rather than a scratchpad — which is the defect that
created T-25 in the first place:

| Artifact | Path |
|---|---|
| BRANCH lower bound, Riesen–Bunke upper bound, exact A*, one parametrised cost model | `benchmarks/real_data/eval_setup/ged_bounds.py` |
| Gate runner, seeded sampling, per-pair JSON for the GEDLIB replay | `benchmarks/real_data/eval_setup/validate_ged_bounds.py` |
| Invariant tests — bracket validity, identity, symmetry, relabelling, cost-model tracking | `tests/unit/test_ged_bounds.py` — **35 passing** |

**Gate 2 result**: 400 LINUX pairs, unit costs, seed 42 — **0 bracket violations**. Reproduces the
GEDLIB Picasso smoke test exactly (P₄ vs C₄ → 1.00 / 1.00 / 1.00). Full table:
[exact_ged](exact_ged.md) §4.

**Three consequences, all carried forward:**

1. **The decision survives, its numbers do not.** BRANCH-FAST is confirmed as the better reference by
   a wide margin, but all six retired figures miss in the flattering direction — consistent with
   having been measured on IAM Letter and printed as a cohort property. **T-05 re-derives them per
   dataset**, and no draft may quote the old ones.
2. **The upper bound is not symmetric** — a finding the gate produced on its first run. Every GEDLIB
   upper-bound method builds its edit path from a directed assignment and shares the property, so a
   pairwise matrix filled in one orientation **is not a distance matrix**. T-05 must symmetrise with
   `min` or assert and fail loudly.
3. **Gate 2 is now two-sided.** Our numbers are on record; T-05 replays the same seeded sample
   through GEDLIB and compares. Disagreement is a bug in one of the two, and that is precisely what
   the gate exists to surface.

---

### Still open, and it is a scope limitation rather than a decision

**The connectivity discard is size-biased on the datasets added for scaling** — Mutagenicity discards
graphs 1.92× larger than it keeps, AIDS-IAM 2.27×, Protein 1.58×. Any "n̄ ≈ 30" claim is on a
subsample with the large graphs preferentially removed. **Report retained and discarded `n̄`/`n_max`;
state the precondition as a scope limitation with its measured cost.** Paired with D14 — the
*encoding* discard has the same structure.
