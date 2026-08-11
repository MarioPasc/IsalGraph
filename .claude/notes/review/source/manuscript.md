# Manuscript production, page budget and submission package

**Status**: v1.0, 2026-08-11. Owner: **T-20** (rewrite), **T-24** (package), **T-14** (letter).
Closes `gap-audit.md` GAP-5, GAP-6, GAP-9, GAP-10.

`plan.md` covers what gets computed. This covers what gets **written, cut, and uploaded** — the layer
between a finished recompute and a resubmission that the Editor-in-Chief will approve.

Manuscript root:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`
(its own git repository, separate from the code repo).

---

## 1. Section-by-section rewrite map

Every row has an owner. Rows marked **NEW** did not exist in the ticket board before this audit.

| Section | File | Change | Driver | Ticket |
|---|---|---|---|---|
| Abstract | `main.tex:106–126` | scope G2S (connected / root-reaching); "strongly correlates" → conditional; drop "direct applications in"; resolve the `:106` vs `:114` self-contradiction | R3.3a, R3.6b, R3.2, E5 | T-12 |
| Highlights, graphical abstract | `main.tex:129–141` | restate every scoped-down claim; **fix the `graphical_abtract.pdf` filename** | B4, B6, E12 | **T-24** |
| 1 Introduction | `introduction.tex:11–33` | "breaks permutation equivariance" → **invariance**; soften "no existing method"; hand positioning to the new §1.x | R3.7e/D20, R3.1b | T-11, T-12 |
| **1.x Related work (NEW section)** | new file | canonicalisation literature: AGM, gSpan, nauty/Traces, bliss, graph6/sparse6, Babai; the AE.3 comparison table | AE.2, AE.3, R1.2, R3.1 | T-08, **T-17** |
| 2.1–2.2 | `methodology.tex:28–420` | rewrite Alg. 2 lines 24–30 to match `graph_to_string.py:208–238`; delete the printed draft self-correction in Example 2.3 | R3.4a/D5, E8 | T-11 |
| **2.2.x Complexity (NEW subsection)** | `methodology.tex` §2.2 | `P(M)` recomputed per frame; cost pair scanning, pointer walking, neighbour checks, backtracking; three-way separation of theory / worst case / empirical | R3.4b, R3.4c, R3.7d | T-13 |
| 2.3 Canonicalization | `methodology.tex:421–470` | add the search-space schematic (renderer exists: `viz/search_tree.py::canonical_search_tree_figure`) | R3.7c | T-09 |
| 2.3.3 Proof | `methodology.tex:623–726` | restate Thm 2.12 **within a fixed directedness class**; move the `directed`-flag hypothesis from the proof into the statement; re-verify all three proof steps; propagate to **Cor. 2.13** | R3.3b/c, D3b, D4 | **T-22** |
| **2.x Prior-work delta (NEW subsection)** | `methodology.tex` or §1.x | inherited / modified / new versus [28] and [29] | R3.1a, R3.7b, AE.3 | T-07 |
| 3.1 Datasets | `computational_experiments.tex:14–58` | **5 → 10 datasets**; property table gains `n̄`, density, connectivity retention, **and the discarded subset's `n̄` / `n_max`**; Suite 1 / Suite 2 split | AE.1, E1, F1, open q. 15 | **T-20** |
| 3.2 Protocol | `computational_experiments.tex:90–233` | the whole statistical protocol is replaced; pair-accounting ladder; exclusion justifications with counts | R3.5a/b/c, `statistics.md` D1–D15 | **T-20** |
| 3.2.3 Message length | `computational_experiments.tex:141–189` | "GED **standard** construction" → "explicit-construction reference model"; real serializations beside it; the shared edit-operation alphabet derivation | R3.6a/B3 | **T-20**, T-17 |
| 3.3 Implementation | `computational_experiments.tex:234–` | **C++ engine and GEDLIB did not exist at submission**; versions; build constraints; artifact release | R3 "open implementation", `statistics.md` §8 | **T-21** |
| 4 Results — all | `results.tex` | every number re-derived; competitor columns; brackets; calibration; stratification; CD diagram | T-06 | **T-20** |
| 4.2 Empirical complexity | `results.tex:69–126` | **restructure Figure 2**: per-graph encoding cost and per-pair GED cost stop sharing an axis; competitor encode curves added; the `n = 3–20` / `n = 3–11` / `≤ 12` / `≤ 50` range confusion resolved | R1.1/D16, E3, E4 | **T-20** |
| **4.x Labels (NEW subsection)** | `results.tex` | collision count, label surplus, density-vs-label decomposition | R1.3, AE.4b | **T-18** |
| 5 Conclusion | `conclusion.tex` | every number; `n^{9.0}` deleted; "super-polynomial" deleted; "labels present in all five datasets" corrected; limitations expanded | D1, D2, E6, R3.7a, B1–B6 | T-12, **T-20** |

---

## 2. Artifact inventory

| # | Artifact | Disposition | Est. pages |
|---|---|---|---:|
| Fig. 1 | Message length scatter | **replace** — competitor serializations added | 0.75 |
| Fig. 2 | Empirical complexity | **restructure** — R1.1's category error; competitor curves | 1.0 |
| Fig. 3 | Aggregated GED/Lev heatmap | **demote** — per-dataset becomes primary (D5); pooled figure kept small or moved | 0.5 |
| Fig. 4 | Speed–quality trade-off | **replace** — new numbers, new methods | 0.75 |
| Tab. 1 | Instruction set | keep (hand-written LaTeX, no generator) | 0.5 |
| Tab. 2 | Dataset properties + information content | **replace** — 10 datasets, new columns | 1.25 |
| Tab. 3 | Spearman ρ summary | **replace** — competitors, brackets, CIs | 1.25 |
| **NEW** | AE.3 representation comparison (properties / strengths / limitations) | T-17 | 0.75 |
| **NEW** | [28] / [29] inherited-modified-new | T-07 | 0.75 |
| **NEW** | Pair-accounting ladder, per dataset | `statistics.md` §8 | 0.5 |
| **NEW** | Calibration: bracket width, certification rate, ρ-gap vs `n` | `statistics.md` §6.1 | 0.75 |
| **NEW** | Canonical search-space schematic | T-09 | 0.5 |
| **NEW** | S2G/G2S worked example | T-09 | 0.75 |
| reuse | **Graphical abstract**, regenerated from the two T-09 figures | T-09 → T-24 | **0** — submitted separately |
| **NEW** | Label surplus / collision table | T-18 | 0.75 |
| **NEW** | Critical-difference diagram (approximate regime only — `gap-audit.md` MF16) | D8 | 0.5 |
| reinstate | Generative-AI declaration | E11, Elsevier compliance | 0.2 |
| reinstate | Acknowledgements | funders, SCBI, NVIDIA | 0.2 |
| reinstate | Author biographies + photos | Pattern Recognition requirement | 0.6 |

**Gross addition ≈ 12–13 pages against 0 pages of headroom.** `main.pdf` is exactly 35 of 35.

---

## 3. The page budget — the binding constraint

### 3.1 Recoveries, in order of size

| Recovery | Est. pages | Note |
|---|---:|---|
| **Fix float placement (E7)** | **up to 2.0** | `\floatpagefraction{1}` + `\textfraction{.001}` (`main.tex:66–67`) push all three algorithms onto dedicated float pages 33–35, after the bibliography. Inlining them near their discussion recovers most of three near-empty pages. **Must run before any trim** — it changes pagination (`gap-audit.md` MF9) |
| Compress the Thm 2.12 proof (`methodology.tex:639–726`) | 0.75 | three steps; the execution-path bijection can be tightened |
| Restructure §1 into §1 + §1.x related work | 0.5 | the current positioning survey at `:11–33` is partly superseded by the new section |
| Trim worked examples (Ex. 2.3, Rem. 2.6, Rem. 2.11) | 0.5 | the new S2G/G2S figure carries the same content better |
| Demote Fig. 3 to a single small panel | 0.5 | D5 demotes the pooled analysis anyway |
| Merge the four §4 subsections' redundant preambles | 0.5 | |
| **Total recoverable in the main file** | **≈ 4.75** | |

**Recoveries (≈ 4.75) do not cover additions (≈ 12–13).** The gap is ≈ 8 pages and cannot be closed
by editing.

### 3.2 The one query that decides the strategy — **ask on day 1**

The decision letter names a mailbox and the plan has never used it:

> For any queries, please contact **only the Journal mailbox via patcog@elsevier.com** and the
> respective support team would check the query and respond. (`mail.txt:24`)

**Send one email, immediately:**

> Does supplementary material count toward the 35-page limit for a regular paper, and what file
> types are accepted for it at revision?

Everything below branches on the answer, and the answer costs nothing but latency. Asking on day 18
is worthless; asking on day 1 is free insurance on ~8 pages.

- **If supplementary does not count** — move the calibration tables, the stratified analyses, the
  pair-accounting ladder, the sensitivity arms and the full per-dataset result grids to
  supplementary, keep every *claim* and one summary table per claim in the main text. The gap closes
  comfortably and nothing scientific is lost.
- **If it does count** — the additive requests must be prioritised against each other. Pre-declared
  ranking, highest priority first, so the cut is not made under time pressure:

  1. **AE.3 comparison table** — the Area Editor endorsed it in their own voice. Non-negotiable.
  2. **Claim scoping (B1–B6)** — costs almost no space and answers four separate comments.
  3. **Factual corrections** — Alg. 2, `n^{9.0}`, "super-polynomial", equivariance, E5/E6/E8. Nearly
     free.
  4. **Statistics rewrite** — R3.5a/b/c are three of R3's seven comments.
  5. **Related-work section** — AE.2, R1.2.
  6. **[28]/[29] delta table** — R3.1, R3.7b.
  7. **Size-scaling results (Suite 2)** — AE.1.
  8. **Label results** — R1.3.
  9. **Complexity section** — R3.4b, R3.7d.
  10. **Search-space schematic** — R3.7c, explicitly a "would benefit from".
  11. **S2G/G2S example figure** — decision 6, **author preference, not a reviewer request**. First
      to go.

  Items 10 and 11 are the only two on the list no reviewer requires. Cutting them is the correct
  first move, and decision 6 should be re-affirmed with that understood.

### 3.3 Standing rule

Decision 7 — "ignore the page budget while drafting" — is right for drafting and dangerous at the
end. **T-15 is 2 days and must be scheduled with the same seriousness as a compute job**, because a
36-page manuscript is rejected on format by the EiC regardless of the science. Track the page count
at every commit to the manuscript repository from the moment the first new section lands.

---

## 4. Response letter

### 4.1 Structure

| Part | Content |
|---|---|
| 0 | One-page summary of changes: full recompute under one cost model, 5 → 10 datasets, six competitor representations, replaced statistical protocol, new related-work section |
| 1 | **Area Editor** — AE.1, AE.2, AE.3, AE.4a/b/c |
| 2 | **Reviewer 1** — R1.1, R1.2, R1.3 |
| 3 | **Reviewer 3** — R3.1 … R3.7, with sub-parts addressed individually |
| 4 | **Editor-in-Chief checklist** — EiC.a1–a4, EiC.b, EiC.c, each with the compliance evidence |
| 5 | **Corrections made on our own initiative** — E1–E12 (see §4.3) |

**Per-comment format**, fixed: verbatim quotation → response → *exact* pointer to the changed
location (section number and, where useful, the artifact label). R3 checked thirteen of thirteen
checkable claims against the sources; the letter must be equally checkable in the other direction.

### 4.2 Drafted incrementally, not at the end

T-14 is "3 days, depends on all", which puts 41 numbered demands into the last three days of a
20-day window. **Replace with: every ticket emits its response fragment when it closes.** T-14 then
assembles, harmonises the register and writes part 0 — three days is enough for that and is not
enough for the alternative.

`plan.md` §0.5's traceability matrix is the index: one row per comment ID, one fragment per row, and
an empty fragment cell is a visible hole rather than a late discovery.

### 4.3 Disclosure of E1–E12 — **DECIDED 2026-08-11: yes, and the reviewer is answered first**

> **The ordering rule, locked.** Nowhere does a self-found defect get raised before the reviewer's
> own comment has been answered on its own terms. This applies at both levels:
>
> - **Letter level** — Part 5 comes after Parts 1–4. The reviewers and editors see their comments
>   addressed before they see anything we volunteered.
> - **Comment level** — where a self-found defect touches a comment (E2 ↔ R3.5a, E1 ↔ R1.3a,
>   E6 ↔ R1.3b, E5 ↔ R3.3a, E3/E4 ↔ R3.4c), the reviewer's request is satisfied first and the extra
>   finding follows as *"in addressing this we also found…"*, never as the opening move.
>
> Rationale for the ordering: a response that leads with our own findings reads as deflection, and it
> buries the answer the reviewer is looking for. Answering first and volunteering second reads as
> diligence. Same content, opposite impression.

Twelve defects were found in our own sources that no reviewer raised, including the 473,147-pair
reconciliation gap (E2), density never being computed at all (E1), "labels present in all five
benchmark datasets" being false for LINUX (E6), and a printed draft self-correction in Example 2.3
(E8).

**Recommendation: disclose, in a short dedicated Part 5.**

1. The recompute changes every one of these numbers regardless. Silence would mean the printed values
   change with no stated reason — which invites the question rather than avoiding it.
2. R3 verified thirteen of thirteen checkable claims in round 1. Assuming round 2 is less thorough is
   not a plan.
3. E2 is *exactly* what R3.5a asked for and did not get. Volunteering it converts a latent finding
   into evidence of diligence.
4. It costs a third of a page.

The counter-argument — that it concedes errors nobody caught — is real, and it is answered by the
ordering rule above rather than by silence: the defects appear *after* every comment has been
addressed, so they read as the by-product of a thorough revision rather than as the headline.

### 4.4 No marked-up manuscript

`00-editor-and-decision.md` establishes that the letter requires neither a clean unhighlighted main
file nor a separate Summary of Changes — unlike TPAMI. Given that §§3.1–5 are rewritten wholesale, a
change-tracked version would be marked end to end and would carry no information.

**Locked: no marked-up manuscript. The letter's per-comment pointers are the change map.**

---

## 5. Submission package — checklist

Every item verified present before upload.

| # | Item | State today | Owner |
|---|---|---|---|
| 1 | **LaTeX source files, not PDF** (`mail.txt:22`) | sources present, package never assembled | T-24 |
| 2 | Main PDF, **≤ 35 pages**, double-spaced single column | 35 / 35 | T-15 |
| 3 | Response letter, parts 0–5 | not started | T-14 |
| 4 | **Generative-AI declaration** (E11) | commented out, `main.tex:198–202` | T-24 |
| 5 | **Author biographies + photos** — Pattern Recognition requires them | commented out, `main.tex:225–245` | T-24 |
| 6 | Acknowledgements (funders, SCBI, NVIDIA) | commented out, `main.tex:175–177` | T-24 |
| 7 | Highlights — updated for the scoped claims | not started | T-24 |
| 8 | Graphical abstract — updated; **filename misspelt** `graphical_abtract.pdf` (E12) | stale | T-24 |
| 9 | Declaration of competing interest | absent | T-24 |
| 10 | Data availability statement — GEDLIB, IAM, GraphEdX provenance | absent | T-21, T-24 |
| 11 | Code/data artifact updated: competitor backends, GEDLIB pin, new datasets, library versions | not started | T-21 |
| 12 | Bibliography **35–55** entries, arXiv `note` fields stripped from the five peer-reviewed entries, `\cite{garey1979,Zeng:2009}` group commented individually | 43 cited; 13 dead entries in the `.bib` (E9) | T-08, T-19 |

**Two traps carried from `codebase-pointers.md`:**

- the `.so` **does not rsync** — the extension is built on the cluster as part of environment setup;
- build flags are `-march=x86-64-v3`, **never** `-march=native` — Picasso is heterogeneous and
  `native` yields SIGILL on a fraction of nodes.

Both belong in the reproducibility statement (item 11), not only in internal notes.

---

## 6. Ordering constraints

Violating any of these means redoing work:

1. **E7 float fix → then page trim.** Fixing float placement changes pagination; trimming first
   measures the wrong document.
2. **T-06 (numbers) → T-12 (claim scoping).** Every scoped claim quotes a number that T-06 re-derives.
3. **T-07 ([28]/[29]) → T-08 (related work) → T-17 (comparison table).** The table's rows depend on
   what the delta analysis establishes.
4. **`statistics.md` §5 MRM and `plan.md` §8 AIDS stratification run in week 1.** Both can refute a
   central claim; a refutation in week 3 has no absorption time.
5. **patcog@elsevier.com query → §3.2 page strategy.** Day 1.
6. **Response fragments accrue per ticket → T-14 assembles.** Not the reverse.

---

## 7. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v1.0 | Created to close `gap-audit.md` GAP-5, GAP-6, GAP-9, GAP-10. Page arithmetic computed from the artifact inventory; supplementary-material query identified as the day-1 decision gate |
