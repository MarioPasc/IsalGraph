# Demands — the coverage contract

**One row per demand in `mail.txt`, locked to a decision, a ticket and a manuscript artifact.**
**A row with no ticket is a hole.** This is also the response letter's index: one fragment per row,
and an empty fragment cell is a visible hole rather than a late discovery.

**41 letter demands across 42 rows** (R3.1a is split into its two conjuncts). **All owned.**
Built from `../source/mail.txt` alone, twice, independently.

Legend — **✓** covered · **NEW** owner created by an audit · **FIX** a locked decision that was wrong

> **Two rows the letter does not contain, retained and relabelled.** **M3** (point-by-point response)
> is *inferred* — `:67` is the Area Editor's **priority statement**, not a request for a response
> document; the deliverable is right and journal convention justifies it, the citation was not.
> **R1.3b** is a *premise* — "the performance degradation on AIDS **may come from** the loss of label
> information" is declarative. **Neither carries an independent budget.**

---

## Submission mechanics

| ID | Demand (`mail.txt`) | Decision / experiment | Ticket | Artifact | |
|---|---|---|---|---|---|
| M1 | Upload **source files**, not PDF (`:22`) | assemble the LaTeX package — [compliance](compliance.md) §8 | **T-24** | — | ✓ |
| M2 | Due **2026-08-31** (`:20`) | dated calendar with gates — [schedule](schedule.md) | *(calendar, not a ticket)* | — | ✓ |
| M3 | Point-by-point response | **INFERRED.** `:67` requests no response document; T-14 stands on Editorial Manager convention instead. **`:67`'s real content — "Please address these concerns thoroughly, as they will strongly influence the potential impact of the work", a weighting over AE.1–AE.4c — is what converts AE.4a and AE.4b from soft to requirement** | T-14 | response letter | **FIX** |

## Editor-in-Chief — pass/fail, checked independently of the reviewers

Detail in [compliance](compliance.md).

| ID | Demand | Decision / experiment | Ticket | Artifact | |
|---|---|---|---|---|---|
| EiC.a1 | Bibliography **35–55** (`:126`) | 43 printed → 12 slots. **Allocations summed to 16–17.** Fitting allocation in [compliance](compliance.md) §2; "retire a dead entry" frees **nothing** | **T-26**, T-08, T-19 | `cas-refs.bib` | **FIX** |
| EiC.a2 | Cover **last and current year** (`:126`) | ≥ 4 from 2025–26, **self-citations excluded**. Measured: 5 refs postdate 2023, both 2025 entries are ours | **T-19** | related work | **FIX** |
| EiC.a3 | No excessive arXiv (`:126`) | strip `note = {arXiv:…}` from 5 peer-reviewed entries: **6 → 1**; state [28]'s status in one sentence | T-08 | `cas-refs.bib` | ✓ |
| EiC.a4 | No uncommented citation groups (`:126`) | split `\cite{garey1979,Zeng:2009}`; **do not "fix"** `introduction.tex:31` | T-08 | `methodology.tex:803` | ✓ |
| EiC.b | Cite **recent pattern-recognition** work (`:128`) | venue audit + targeted additions. Measured: **zero PR-field reference after 2023**, zero CVPR/ICCV/ECCV/ICPR/TPAMI/IJCV. Criterion: **≥ 3 additions at PR venues other than the PR journal** | **T-19** | related work | **FIX** |
| EiC.c | **≤ 35 pages** (`:130`) | page budget re-derived as **deltas, not gross sizes**; supplementary query day 1 | **T-26**, T-15 | whole document | **FIX** |

## Area Editor

| ID | Demand | Decision / experiment | Ticket | Artifact | |
|---|---|---|---|---|---|
| AE.1 | **Graph size impact must be clear** (`:59–60`) | Suite 1 (`n ≤ 12`, exact) / Suite 2 (`n ≤ 98`, proven bracket); **the ceiling is attributed to the reference, not to IsalGraph**; relative bracket width vs `n` | T-01, T-05, T-06 | §3.1, §4 | ✓ |
| AE.2 | Related-work framing + references (`:62`) | new §1.x: canonicalisation literature | T-08 | **§1.x (new)** | ✓ |
| AE.3 | **Side-by-side comparison of existing representations** — properties, strengths, limitations of each (`:63–64`) | comparison table as a **paper artifact**, axes from R1.2. **The Area Editor endorsed this in their own voice — non-negotiable** | **T-17** | comparison table (new) | ✓ |
| AE.4a | **Choice of benchmark models** (`:66`) | six competitor representations enter three experiments; **each distance selected by measurement** (T-04a). **Requirement-modal via `:67` — this, not R1.1, is what the competitors answer to** | T-04, **T-04a**, T-06 | Tabs. 2–3, Figs. 1–2 | ✓ |
| AE.4b | **Fully labeled vs partially labeled** (`:66`) | a **label-content column** in the dataset table (Tier 0) | **T-18** | §3.1 table | ✓ |
| AE.4c | Associated analysis of results (`:66`) | the whole [statistics](statistics.md) protocol, D1–D15 | T-02, T-06 | §3.2, §4 | ✓ |
| AE.5 | "**Additional comments … should also be addressed**" (`:69`) | Catch-all, requirement modal. Largely subsumed; the one unowned clause is **"rationale"** from R3's preamble (`:83`), which lands in R3.1a(ii)'s paragraph at **no marginal cost**. Plus one verification pass over `:73–116` inside T-14 | **T-14** + T-07 | response letter, §2.x | **NEW** |

## Reviewer 1

| ID | Demand | Decision / experiment | Ticket | Artifact | |
|---|---|---|---|---|---|
| R1.1 | GED runtime comparison unfair; compare against a similar problem setting (`:75`) | competitor encode-time curves; **per-graph and per-pair costs stop sharing an axis**. Two asks, not one: the "unfair" clause is a **defect report** and its fix is protected regardless of the closing modal | T-04, T-06, T-20 | Fig. 2 restructured | ✓ |
| R1.2a | AGM and gSpan uncited (`:77`) | both cited and discussed. **T-08 is the owner** — the ask is *discussion*, ≈ 0.5 d, satisfied by citation. gSpan's vendoring serves R1.1/AE.4a; **if T-04 slips, R1.2a is still answered** | **T-08** (T-04 enriches) | §1.x | **FIX** |
| R1.2b | Five axes: uniqueness, expressiveness, efficiency, scalability, **downstream learning** (`:77`) | all five are printed rows; downstream reads **"not evaluated"**. Marginal cost over the AE.3 table ≈ 0 | **T-17** | comparison table | ✓ |
| R1.3a | Density insufficient to explain AIDS (`:79`) | true density computed; **within-AIDS density stratification, which can refute `conclusion.tex:30–36`** | T-02, T-06 | §4.x | ✓ |
| R1.3b | **PREMISE** — "may come from the loss of label information" (`:79`) | rebuttal **leads and is free**: both sides of the correlation are topology-only. **Licenses no work of its own** | served by R1.3c / R1.2b / AE.4b | §4 paragraph + 2 columns | **FIX** |
| R1.3c | Discuss the limitation and its impact (`:79`) | the missing piece is the *connection* between the §5 limitation and the §4 AIDS interpretation | T-18, T-12 | §5 limitations | ✓ |
| R1.3d | Labels as future work (`:79`) | concrete `Σ × L` extension; **already named at `conclusion.tex:71`, `:81`** — make it concrete and point at it; [29] as precedent | T-07, T-12 | §5 future work | ✓ |

## Reviewer 3

| ID | Demand | Decision / experiment | Ticket | Artifact | |
|---|---|---|---|---|---|
| R3.1a**(i)** | Inherited / modified / new vs [28], [29] (`:86`) | read both **papers** ([29] is published); delta table | T-07 | §2.x table (new) | ✓ |
| **R3.1a(ii)** | "**…and explain why the combined extension constitutes a sufficiently substantive contribution**" (`:86`) | **one paragraph closing §2.x**, ~120–150 words. Re-orders facts T-07 already gathers — **no new investigation**. Without it, the delta table becomes evidence *against* us | **T-07** | §2.x closing paragraph | **NEW** |
| R3.1b | "No existing method satisfies all four" too absolute (`:86`) | B6 — softened **and unified**; the claim appears twice with *different* property sets | T-12, T-17 | §1, §5 | ✓ |
| R3.2 | **Sequential-model evaluation** (`:89`) | **DECLINED** (decision 5) + all five LM claim sites come down + a contingency **conditional on the S-f extension** | *(decision, not a ticket)* | abstract, §5 | ✓ |
| R3.3a | Narrow "any finite simple graph" / "arbitrary graphs" (`:92`) | B1: undirected **connected**; directed **root reaching all nodes**; S2G total, G2S partial | T-12 | abstract, §1, §5 | ✓ |
| R3.3b | Thm 2.12 and the `directed` flag (`:92`) | restate **within a fixed directedness class**; hypothesis moves from **proof to statement**; **re-verify all three proof steps and Cor. 2.13** | **T-22** | §2.3.3 | ✓ |
| R3.3c | Is the flag part of the representation? (`:92`) | **external metadata**; exact witness (`"V"` under both semantics); never quote a rate without its window | T-12, **T-22** | §2.3.3 | ✓ |
| R3.4a | Alg. 2 `C`/`c` vs Table 1 (`:95`) | pseudocode rewritten to match the implementation — **guards *and* duplicate checks** | T-11 | §2.2, Alg. 2 | ✓ |
| R3.4b | `P(M)` recomputed or precomputed; cost four operations (`:97`) | **recomputed per frame**; four operations costed; `\|Aut(G)\|`-governed worst case. **Supports a claim the abstract already makes — cannot be trimmed** | T-13 | §2.2.x (new) | ✓ |
| R3.4c | `n^{4.9}` vs `n^{9.0}`; "super-polynomial" (`:99`) | all exponents re-derived; three-way separation. **The contradiction is three-way, not two** | T-06, T-13 | §4.2, §5 | ✓ |
| R3.5a | Justify exclusions, report removals per dataset (`:102`) | pair-accounting ladder, per dataset | T-02, T-06 | ladder table | ✓ |
| R3.5b | **Interpret Fig. 3 cautiously; dataset-level correlations primary** (`:104`) | **D5 answers the literal clause at ~0 cost** (per-dataset primary, pooled demoted). **D6's recompute is a separate, deliberate choice** driven by F2 and Cor. 2.13 — direction, **not dose**: see [exact_ged](exact_ged.md) §3 | **D5** (floor) · T-03 stage 1, T-05 | §3.1, §4.3 | ✓ |
| R3.5c | Pair dependence; describe the bootstrap; graph level (`:106`) | D2 graph-level cluster bootstrap, D3 Mantel; D15 makes it affordable | T-02, T-06 | §3.2 | ✓ |
| R3.6a | "GED standard construction" not established (`:109`) | **explicit `or`. The free branch is taken unconditionally** — B3 rename + shared edit-operation alphabet. The expensive branch is owned by AE.4a, not by R3.6a | T-12, T-17 | §3.2.3, Tab. 2 | ✓ |
| R3.6b | "Strongly correlates" is not uniform (`:111`) | B4 — the results section's conditional framing propagates to abstract and conclusion | T-12 | abstract, §5 | ✓ |
| R3.7a | Three limitations to add (`:114`) | B5 — `n` ceiling **with its cause**, exponential worst case, **no sequential/downstream task**. Item 3 is R3.2's concession **under a requirement modal — the R3.2 decline may not absorb it** | T-12 | §5 | ✓ |
| R3.7b | Dedicated comparison subsection (`:116`) | §2.x delta subsection + §1.x | T-07, T-08 | §2.x | ✓ |
| R3.7c | Canonical search-space schematic (`:116`) | renderer exists: `viz/search_tree.py::canonical_search_tree_figure`. **Prose already states the reviewer's exact sentence** (Remark 2.7) — only the figure is missing | **T-09** | §2.3 figure | ✓ |
| R3.7d | Separate theory / worst case / empirical (`:116`) | three-way separation; the `\|Aut(G)\|` characterisation replaces "exponential" | T-13 | §2.2.x, §4.2 | ✓ |
| R3.7e | Four broad statements (`:116`) | equivariance → **invariance**; + B1, B4 | T-11, T-12 | §1, abstract, §5 | ✓ |

---

## Self-found defects — no reviewer raised these

Disclosed in response-letter Part 5, **after** every reviewer comment is answered
([manuscript](manuscript.md) §4.3). Detail in [corrections](corrections.md) §3.

| ID | Defect | Ticket |
|---|---|---|
| E1 | Density never computed; no node count reported | T-01, T-20 |
| E2 / F2 | 473,147-pair gap — **cause is within-split GED coverage, not filtering** | T-03 |
| E3 | Fits declared `n = 3–20`, greedy data to 50 | T-06, T-20 |
| E4 | A fourth node range (`n = 3–11`) | T-20 |
| E5 | Abstract self-contradiction (`:106` vs `:114`) | T-12 |
| E6 | "Labels present in all five datasets" — **false for LINUX**, two sites | **T-12** |
| E7 | Algorithms float to pp. 33–35, after the references | **T-11, before T-15** |
| E8 | Draft self-correction printed in Example 2.3 | T-11 |
| E9 | 13 dead bibliography entries | T-08 |
| E10 | WL kernel and Mantel computed, never reported | T-04, T-02 |
| E11 | Generative-AI declaration commented out | **T-24** |
| E12 | Orphaned figure PDFs; **`graphical_abtract.pdf` misspelt** | **T-24** |
| D19 | [28] Transformer / [29] LSTM claims unverified | T-07 |
| — | C++ engine and GEDLIB absent from the Implementation section; artifact release | **T-21** |
| — | Picasso `fscratch` file-count quota exceeded | **T-23 — blocking** |
| — | 13 of 16 measurement scripts absent; validation gate 2 unexecutable | **T-25 — blocking** |
