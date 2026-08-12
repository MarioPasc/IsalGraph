# Findings — Reviewer #3 · audit-2026-08-11b

**Agent**: `audit-r3` · **Demands owned**: R3.1a–b, R3.2, R3.3a–c, R3.4a–c, R3.5a–c, R3.6a–b, R3.7a–e (18) · **Date**: 2026-08-11
**Sources readable**: manuscript **yes** (all six `.tex`) · letter **yes** · plan **yes** · code **yes**

> **Provenance.** The agent's write was blocked by a harness hook; the orchestrator persisted the
> content and added Phase-5 stamps marked `[ORCH-VERIFIED]`.

## Verdict table

| ID | Modal | Verdict | One line |
|---|---|---|---|
| **R3.1a** | REQUIREMENT | **UNDER-blocking** | Second clause ("why substantive") has no owner anywhere |
| R3.1b | REQUIREMENT (scoping) | COVERED | B6 softens **and** unifies; T-17 supplies the comparison |
| R3.2 | SUGGESTION | COVERED | Decline legitimate; all five LM claim sites scheduled down |
| R3.3a | REQUIREMENT | COVERED | B1 names all four sites; body already correct |
| R3.3b | PREMISE | COVERED | Premise half-wrong; T-22's fix is stronger than asked |
| R3.3c | CLARIFICATION | COVERED | Proof audit forced by the correction; second customer verified |
| R3.4a | REQUIREMENT | COVERED | Confirmed: pseudocode wrong, implementation right |
| R3.4b | REQUIREMENT | COVERED | No complexity discussion exists at all; T-13 must build it |
| R3.4c | REQUIREMENT | COVERED | Contradiction is three-way; plan has all three |
| R3.5a | REQUIREMENT | COVERED | Criteria stated, never justified or counted |
| R3.5b | REQUIREMENT | COVERED | Literal ask is free (D5); recompute has verified second customers |
| R3.5c | REQUIREMENT | COVERED | D2/D3 stay inside the concession's bound |
| R3.6a | REQUIREMENT (`or`) | COVERED | Free branch taken unconditionally; expensive branch owned by AE.4a |
| R3.6b | REQUIREMENT | COVERED | Abstract unconditional, conclusion partial |
| R3.7a | REQUIREMENT | COVERED | Item 3 absent, items 1–2 partial; B5 covers all three |
| R3.7b | SUGGESTION | COVERED | `manuscript.md:29` creates the dedicated subsection |
| R3.7c | SUGGESTION | COVERED | Prose already says it; **cut order mis-bundles two figures** |
| R3.7d | REQUIREMENT | COVERED | T-13 three-way separation |
| R3.7e | REQUIREMENT | COVERED | All four statements located and booked |

**One UNDER, zero OVER.** Both rows the README nominated as likely over-scope (R3.5b, R3.6a) survive
the second-customer test on verified evidence — each carries a *different* defect, recorded below.

---

## R3.1a — the one hole · **UNDER-blocking**

**Operative clause** (`mail.txt:86`), verbatim:

> "The paper should provide a detailed side-by-side comparison that identifies which components are inherited, modified, or genuinely new, **and explain why the combined extension constitutes a sufficiently substantive contribution.**"

**Modal**: "should provide … and explain" → **REQUIREMENT**, two conjuncts.

### Q3 — Already in the manuscript?

| Clause | State | Evidence |
|---|---|---|
| 1 — inherited / modified / new comparison | **absent** | the entire prior-work comparison is **two sentences**, `introduction.tex:52–53` `[ORCH-VERIFIED]` — "Our previous work \cite{lopezrubio2025isalgraph} is substantially different … because the older approach requires a fixed ordering of the nodes and does not employ a circular doubly linked list of nodes. Also, our previous IsalChem methodology … is designed for chemical molecules only, while the current \IsalGraph{} methodology is devoted to general graphs." |
| 2 — **why the extension is substantive** | **absent** | Contributions `introduction.tex:55–62` `[ORCH-VERIFIED]` assert three items (instruction language + VM; canonical string completeness theorem; experiments). **None argues sufficiency against [28] / [29].** |

### What the plan currently commits to

Searched `substantive`, `why the combined`, `novelty` across `plan.md`, `gap-audit.md`,
`manuscript.md`, `competitors.md`, `statistics.md`, `labels.md`, `data.md` → two hits, both
incidental (`plan.md:813`, `gap-audit.md:434` — each the rationale for rejecting T-16). §0.5's row
(`plan.md:115`) quotes **only clause 1**; `manuscript.md:29` books the table only.
**Clause 2 is unowned.**

### Q6 — Second customer?

Clause 1 has three (AE.3, R3.7b, R1.2b). **Clause 2 has none** — which is why it fell out.

### Verdict: **UNDER-blocking**

**Why blocking**: strongest modal in R3's set, and `plan.md:503–509` (§6.1) shows the delta table will
document that *both predecessors ran a sequence model and this paper does not*. **Clause 1 delivered
without clause 2 hands R3 an argument that the extension is *less* substantive** — the table becomes
evidence against us.

**Effort**: current **0**. Proportionate: one paragraph, ~120–150 words, ≈0.1 page, closing the new
§2.x subsection, **≈2 h** once T-07's table exists — a re-ordering of facts already in hand (the
completeness theorem `methodology.tex:628–637`, the generic-topology redesign, the
unlabeled/unbounded-degree scope).

**Fix**: name clause 2 in `plan.md:115` and `manuscript.md:29`, assigned to **T-07**.

**Assumptions made**: the response letter is not a sufficient home — the clause says "**The paper**
should provide … and explain".

---

## R3.5b — the largest item in the revision, and what actually drives it

**Operative clause** (`mail.txt:104`), verbatim:

> "Because the datasets also differ substantially in density and size, the aggregated results in Figure 3 should be interpreted cautiously, with dataset-level correlations treated as the primary evidence."

**Modal**: "should be interpreted" → **REQUIREMENT**. The preceding sentence (IAM uniform vs
LINUX/AIDS topology-only costs) is **declarative — a premise, not an imperative**.

### Q3 — the literal ask is largely already how the paper is built

`results.tex:132–157` reports Spearman ρ **per dataset** (Table 4). Figure 3
(`results.tex:179–190`) aggregates 3,424,764 pairs, and the *only* quantity read off it is the pooled
OLS β = 0.80 / 0.78 / 0.82, which `conclusion.tex:38–41` then promotes to a headline.

**Residual gap**: one caveat sentence at `results.tex:187` + demote `conclusion.tex:38–41`. That is
**`statistics.md:42` D5** — driver "R3.5b", cost ≈ 0.

### Q6 — second customers, three verified findings

1. **F2 is a genuine independent driver.** `plan.md:174–182`: GraphEdX ships GED only *within splits*
   — LINUX 43.0 %, AIDS 43.9 % coverage, and the published ρ = 0.433 / 0.349 are undisclosed
   within-split figures. Missing AIDS pairs = 295,296 − 181,909 = **113,387**, at 12–20 s/pair =
   **378–630 core-hours**, *independent of any cost-model change*. With disclosure decided
   (`plan.md:953`), this must be fixed or caveated. **T-03 is not single-customer.**
2. **The new large-`n` cohort is NOT a driver — the wave prompt's hypothesis is false.**
   `data.md:544`: the entire 67.3 M-pair Suite-2 extension costs **1.24 core-hours**. The
   1,000–1,650 figure is Suite 1, 98 % of it AIDS exact GED (`data.md:298–301`).
3. **D6's lead justification is independent and formal.** `statistics.md:68–72`: zero node cost makes
   GED a *pseudo*metric while Cor. 2.13 asserts a metric. Re-used at `plan.md:676–678`.

### Verdict: **COVERED**, not OVER

The clause is answered by a locked, free decision; the expensive work has two verified non-R3.5b
drivers (cut guard 5).

**Effort**: current T-03 = 3–8 d elapsed + 1,000–1,650 core-h (16–26 h on 64 cores) + queue.
**Floor ≈ 0.5 h, 0 pages.**

**Two defects to record:**

- **Traceability**: §0.5's R3.5b row (`plan.md:125`) is titled "Heterogeneous cost models" — the
  *premise* — and lists **T-03, T-05 only**. **D5, the actual answer to the clause, appears in no
  ticket in that row.** If T-03 fails, §0.5 shows R3.5b unowned when in fact it is answered for free.
- **No zero-compute floor.** `plan.md:984` (risk R1) pre-approves a ~100 core-hour stratified
  subsample — a real 10× fallback. But **no branch is recorded that answers R3.5b with D5 + a caveat
  + stating the cost-model heterogeneity in text and no recompute at all.** On a 20-day clock where
  T-03 is gated on T-23 (quota grace expiring ≈ 2026-08-18) with unbudgeted queue time, that floor
  must be written down.
- Minor: `plan.md:925` marks Q3 resolved ("All-pairs, recover everything"); `data.md:686` still lists
  Q3 open.

---

## R3.6a — the explicit `or`

**Operative clause** (`mail.txt:109`), verbatim:

> "The authors should **either** narrow the claim accordingly **or** include comparisons with established reversible graph serializations."

### Q3

`computational_experiments.tex:162–176` defines `B_GED` and calls it "Under the **standard**
construction model" (`:166`) with no support — exactly the reviewer's target.

### Q4 — the plan takes the **cheap branch, unconditionally**

§10 B3 (`plan.md:907`): "GED **standard** construction" → "explicit-construction reference model".
Not contingent on anything. `statistics.md:81–98` already drafts the replacement derivation and
states it "answers R3.6a and R3.5b with a single paragraph".

**The expensive branch is not built for R3.6a.** `competitors.md:7–9` lists its drivers: **R1.1,
R1.2, AE.3, AE.4a — R3.6a is absent from the list.** AE.4a is editor-voiced and requirement-modal
(`mail.txt:66–67`). Independence confirmed from primary sources, not assumed.

### Verdict: **COVERED**. Free-branch effort: one paragraph replacing `:162–176`'s framing, ≈ 1 h, 0 pages.

---

## R3.5c — inside the concession's bound

**Concession** (`mail.txt:106`): "This does not invalidate Spearman's p as a descriptive measure, but
it could underestimate uncertainty and produce overly small p-values."
**Operative clause**: "The bootstrap procedure mentioned in Section 4.3 should be described and should
operate at the graph level rather than the pair level."

### Q3

"bootstrap" appears **exactly once** in the whole manuscript — `results.tex:176`, "(bootstrap 95\%
CIs overlap substantially)" — never described, never specified as pair- or graph-level. Significance
comes from the asymptotic test at `computational_experiments.tex:208–209`.

D2 (graph-level cluster bootstrap) is the literal ask. D3 (Mantel, 9,999 permutations) replaces the
asymptotic test — directly responsive to "produce overly small p-values", the concession's own words.
D1 keeps ρ descriptive, matching "does not invalidate Spearman's ρ". D15 is budgeting.

### Verdict: **COVERED** — inside the bound. No re-engineering beyond the ask.

---

## R3.4a — factual correction, confirmed against the code

Table 1: `C` = primary→secondary (`methodology.tex:102–104`), `c` = secondary→primary (`:105–107`).
Algorithm 2's `C` guard tests `(ṽ₂,ṽ₁) ∈ E` and duplicate-checks `(ℓ₂,ℓ₁) ∉ E(G_out)` (`:321–323`)
while *adding* `(ℓ₁,ℓ₂)` (`:324–325`); the `c` branch mirrors it (`:330–334`).

Implementation `graph_to_string.py:208–211` guards `tent_sec_in in neighbors(tent_pri_in)` and checks
`tent_sec_out not in neighbors(tent_pri_out)`, then adds `(tent_pri_out, tent_sec_out)` — **both**
guard and duplicate check match Table 1.

**Pseudocode wrong, implementation right.** The reviewer spotted only the guards; `plan.md:880` has
both. T-11 owns it. **Cut guard 1 — never cut.**

---

## R3.3b / R3.3c — T-22 is proportionate, with one item needing a label

**Theorem 2.12** (`methodology.tex:628–637`) **does not mention the `directed` flag**
`[ORCH-VERIFIED — the statement reads only "Let $G$ and $H$ be finite, simple, connected graphs. Then $w^*_G = w^*_H \iff G \cong H$."]`. Only the **proof** does: `:643–644` — "The decoder $\STG$ is a
deterministic function of $w$ **and the directed flag**" `[ORCH-VERIFIED]`.

**The reviewer's premise is half wrong, and the real defect is worse than described**: a load-bearing
hypothesis lives in the proof and not in the statement. T-22's restatement is therefore a **factual
correction** (guard 1), and re-checking the proof is forced by it rather than optional.

**Second customer verified**: Cor. 2.13's proof derives identity of indiscernibles "directly from
Theorem~\ref{thm:invariant}" (`methodology.tex:738–740`), and `statistics.md:68–72` makes Cor. 2.13
the **lead justification for D6** — the decision carrying the entire 1,000–1,650 core-hour recompute.
Auditing the corollary is a prerequisite for the argument justifying T-03.

**Label required (rubric §1 Q6)**: T-22's fourth item — the `tests/property/` directedness-collision
regression — is **unasked-for**. Cheap (two-graph fixture) and worth keeping, but it must be labelled
so it drops first if T-22 overruns. Cost of dropping: hours, no manuscript content.
*Assumption*: no such test exists; the agent did not enumerate that directory.

---

## R3.4b — the largest genuine content gap in this slice

`P(M)` recomputation is *derivable* from `methodology.tex:293–295` (`M ← |V(G_out)|` and the `\For`
both inside the `\While`) but never stated.

Grepping `complexity|O(|\Theta|polynomial|worst-case` across `methodology.tex`, `results.tex`,
`conclusion.tex` returns **five hits total**, all in methodology: `:478` (qualitative "exponential
worst-case"), `:498`/`:501` (triplet computation), `:783` (Levenshtein), `:826` (adjacency update).
**There is no complexity analysis of G2S or the canonical search anywhere** `[ORCH-VERIFIED — my own
grep returns the same five methodology hits; results.tex's are all figure/section labels]`.

Yet `main.tex:114–115` claims G2S runs "in time **polynomial** in the number of nodes"
`[ORCH-VERIFIED verbatim]` — an uncosted claim in the abstract. **The requested accounting is the
missing support for something the abstract already asserts, so it cannot be trimmed.**

T-13 + `manuscript.md:26` (new §2.2.x) own all four named operations. **Page cost ≈ 0.5 page,
additive, in a 35/35 document — a budget item, not a cut candidate.** An argument suffices; no
measurement.

---

## R3.4c — three-way, not two-way

`results.tex:88`/`:107`/`:239` give α = 4.9; `conclusion.tex:50` gives **n^9.0** *and* n^4.5;
`conclusion.tex:68` gives n^4.9; `conclusion.tex:80` calls the fitted curve "super-polynomial".
`plan.md:881` already records the `:50` vs `:68` disagreement **the reviewer missed**. Four drivers
(R3.7d, R3.7e-4, E3, E4). **COVERED.**

---

## R3.7a — three items, verified individually

| Item | State | Evidence |
|---|---|---|
| 1 — evaluation sizes ~12 nodes | **absent** | `conclusion.tex:68` bounds the *canonical method*, not the datasets; E1 records node counts are never reported |
| 2 — expensive + exponential worst case | partial | `methodology.tex:477–480` has it; `conclusion.tex:80` mislabels it |
| 3 — no sequential model / downstream task | **absent** | Limitations paragraph `conclusion.tex:67–71` lists only scaling, density and labels `[ORCH-VERIFIED]` |

B5 (`plan.md:911–912`) covers all three. **Wrinkle**: the n = 98 cohort *removes* item 1's premise,
so the emphasis correctly shifts to exact GED's ceiling (`plan.md:345–353`).

**Item 3 is R3.2's concession under a REQUIREMENT modal — it may not be cut even though R3.2 is
declined.**

---

## R3.7c — cut reasoning is unsound as written

`methodology.tex:462–470` (Remark 2.7, "What is and is not searched over") **already states, in
prose, the reviewer's exact sentence** `[ORCH-VERIFIED verbatim]`: "The priority order `V ≻ v ≻ C ≻ c`
and the minimum-displacement pair ordering $\mathcal{P}(M)$ … are intrinsic to the algorithm
definition and are *not* branched over. Only the identity of the uninserted neighbour chosen at each
`V`/`v` step contributes to the search space."

**The content is satisfied; only the figure is missing.** Renderer verified present:
`src/isalgraph/viz/search_tree.py`.

`plan.md:1002–1006` makes T-09 cut-candidate 4 at 1.5 d — but **T-09 is a merge of two figures with
different justifications**: the S2G/G2S worked example (author decision 6, **no reviewer demand**)
and the search-space schematic (R3.7c, soft modal, renderer already written). Bundling them means the
unasked-for figure is protected by the requested one, and vice versa.

**Recommendation**: split the cut. The schematic's marginal cost is ~2–3 h and ≈ 0.25–0.5 page; the
S2G/G2S example is the item rubric §4's cut order names **first** (unasked-for work with no second
customer). Both feed the graphical abstract, which costs no manuscript pages (`plan.md:50`), so the
graphical-abstract argument does not distinguish them.

---

## R3.1b, R3.3a, R3.6b, R3.7b, R3.7d, R3.7e — clean COVERED

- **R3.1b**: the absolute claim appears twice with **different property sets** —
  `introduction.tex:33` (compact / reversible / structure-preserving / canonicalisable) vs
  `conclusion.tex:74` (universal validity / reversibility / canonical completeness)
  `[ORCH-VERIFIED both verbatim]`. B6 (`plan.md:913`) explicitly **unifies as well as softens**, so
  the plan caught this.
- **R3.3a**: four sites — `main.tex:106–107`, `introduction.tex:33`, `:45–46`, `conclusion.tex:74`;
  B1's site list matches. The body is already correct (`methodology.tex:277`, `:352–358`, `:438`), so
  this is a scoping pass, ≈ 2 h, 0 pages. The abstract also self-contradicts between `:106` ("any
  finite, simple graph") and `:114` ("any connected graph") — that is E5.
- **R3.6b**: abstract `main.tex:120–122` unconditional; conclusion property (iv)
  `conclusion.tex:24–26` unconditional — but the degradation *is* already described at `:30–32` and
  `:69`. Partial → two sentences. B4.
- **R3.7b**: `manuscript.md:29` books "2.x Prior-work delta (**NEW subsection**)" — literally the ask.
- **R3.7d / R3.7e**: all four R3.7e statements located — `introduction.tex:16` (equivariance; the
  plan is right that `M → PMPᵀ` *is* equivariance and invariance is what breaks),
  `introduction.tex:33` / `conclusion.tex:74`, `main.tex:120–122` / `conclusion.tex:24–26`,
  `conclusion.tex:80`. All booked to T-11 / T-12.

---

## Notes for the orchestrator

**Cross-voice merges:**

- **R3.1a clause 1 ≡ AE.3 ≡ R3.7b** — one row, three drivers. Note AE.3 **broadens** it:
  `mail.txt:63–64` says "existing graph representations", whereas R3.1a is about our own [28]/[29].
  **T-07 (delta) and T-17 (comparison table) are both needed; neither substitutes for the other.**
- **R3.6a's expensive branch ≡ R1.1 / R1.2 / AE.4a** per `competitors.md:7–9`.
- **R3.7a item 3 ≡ R3.2's concession**, but under a **requirement** modal — do not let the R3.2
  decline absorb it.

**Flagged outside this slice**: `statistics.md` D4 (MRM) self-labels as "nothing — new … pre-empts
the size-confound attack" (`statistics.md:41`) — correctly labelled per Q6, **but it is promoted to
confirmatory**, so it enters D9's multiplicity family and can produce a headline. Cheap, but the
promotion is scope the label does not cover.

**Page budget, this slice only**: R3.4b ≈ 0.5 p (non-negotiable), R3.7b subsection ≈ 0.5–1 p, R3.7c
figure ≈ 0.25–0.5 p (cuttable), R3.1a clause 2 ≈ 0.1 p. Everything else is replacement text at ≈ 0
net.

**Could not verify**: D19 ([28] Transformer / [29] LSTM) — booked to T-07. Whether `tests/property/`
already holds a directedness-collision test.

**Factual errors in R3's own premises** (recorded neutrally; they strengthen our wording, not our
score): R3.3b says Theorem 2.12 "states" the flag hypothesis — it does not, the **proof** does, which
is a worse defect than described. R3.4a identifies reversed guards but not the equally reversed
duplicate checks. R3.4c names two contradictory exponents; there are **three**.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-11 | Created by `audit-r3` for audit-2026-08-11b; persisted by the orchestrator after a harness hook blocked the agent's write. `[ORCH-VERIFIED]` stamps added by the orchestrator against the manuscript source and the code. |
