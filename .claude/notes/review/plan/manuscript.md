# Manuscript production — rewrite map, page budget, response letter

**Owners**: T-20 (rewrite), T-15 (trim), T-14 (letter), T-26 (page-budget reconciliation)
**Serves**: EiC.c, M1, M3, and the delivery of every other row
**Status**: page budget is **the binding constraint**. `main.pdf` is **exactly 35 of 35**.

Related: [compliance](compliance.md) (the EiC checklist) · [corrections](corrections.md) (what changes) ·
[schedule](schedule.md) (when) · [demands](demands.md)

---

## 1. Section rewrite map

| Section | Source | Change | Drivers | Owner |
|---|---|---|---|---|
| Abstract | `main.tex:106–126` | scope G2S; "strongly correlates" → conditional; drop "direct applications in"; resolve the `:106` vs `:114` self-contradiction | R3.3a, R3.6b, R3.2, E5 | T-12 |
| Highlights, graphical abstract | `main.tex:129–141` | restate every scoped-down claim; **fix the `graphical_abtract.pdf` filename** | B4, B6, E12 | **T-24** |
| 1 Introduction | `introduction.tex:11–33` | equivariance → **invariance**; soften "no existing method"; hand positioning to §1.x | R3.7e, R3.1b | T-11, T-12 |
| **1.x Related work (NEW)** | new file | canonicalisation literature: AGM, gSpan, nauty/Traces, graph6/sparse6, Babai; **the AE.3 comparison table** | AE.2, AE.3, R1.2, R3.1 | T-08, **T-17** |
| 2.1–2.2 | `methodology.tex:28–420` | rewrite Alg. 2 lines 24–30 to match the implementation; delete the printed draft self-correction in Example 2.3 | R3.4a, E8 | T-11 |
| **2.2.x Complexity (NEW)** | `methodology.tex` §2.2 | `P(M)` recomputed per frame; four costed operations; three-way separation | R3.4b, R3.4c, R3.7d | T-13 |
| 2.3 Canonicalization | `methodology.tex:421–470` | add the search-space schematic (renderer exists: `viz/search_tree.py::canonical_search_tree_figure`) | R3.7c | T-09 |
| 2.3.3 Proof | `methodology.tex:623–726` | restate Thm 2.12 within a fixed directedness class; move the flag hypothesis into the statement; **re-verify all three steps; propagate to Cor. 2.13** | R3.3b/c | **T-22** |
| **2.x Prior-work delta (NEW)** | §1.x or `methodology.tex` | inherited / modified / new vs [28] and [29], **plus the sufficiency paragraph** | R3.1a(i)+(ii), R3.7b, AE.3, AE.5 | T-07 |
| 3.1 Datasets | `computational_experiments.tex:14–58` | **5 → 10 datasets**; table gains `n̄`, density, connectivity retention, **the discarded subset's `n̄`/`n_max`**, and a **label column**; Suite 1 / Suite 2 split | AE.1, AE.4b, E1, F1 | **T-20**, T-18 |
| 3.2 Protocol | `computational_experiments.tex:90–233` | the whole statistical protocol replaced; pair-accounting ladder; exclusion justifications with counts | R3.5a/b/c | **T-20** |
| 3.2.3 Message length | `computational_experiments.tex:141–189` | "standard construction" → "explicit-construction reference model"; real serializations beside it; shared edit-operation alphabet | R3.6a | **T-20**, T-17 |
| 3.3 Implementation | `computational_experiments.tex:234–` | **C++ engine and GEDLIB did not exist at submission**; versions; build constraints; artifact release | R3's "open implementation" | **T-21** |
| 4 Results — all | `results.tex` | every number re-derived; competitor columns; brackets; calibration; stratification; CD diagram | T-06 | **T-20** |
| 4.2 Empirical complexity | `results.tex:69–126` | **restructure Fig. 2**: per-graph encoding cost and per-pair GED cost stop sharing an axis; competitor encode curves; resolve the range confusion | R1.1, E3, E4 | **T-20** |
| **4.x Labels (NEW)** | `results.tex` | collision count; density-vs-label decomposition | R1.3, AE.4b | **T-18** |
| 5 Conclusion | `conclusion.tex` | every number; `n^{9.0}` deleted; "super-polynomial" deleted; "labels in all five datasets" corrected; limitations expanded | E6, R3.7a, B1–B6 | T-12, **T-20** |

---

## 2. Artifact inventory

| # | Artifact | Disposition | Est. pages |
|---|---|---|---:|
| Fig. 1 | Message length scatter | **replace** — competitor serializations added | 0.75 |
| Fig. 2 | Empirical complexity | **restructure** — R1.1's category error; competitor curves | 1.0 |
| Fig. 3 | Aggregated GED/Lev heatmap | **demote** — per-dataset becomes primary (D5) | 0.5 |
| Fig. 4 | Speed–quality trade-off | **replace** | 0.75 |
| Tab. 1 | Instruction set | keep | 0.5 |
| Tab. 2 | Dataset properties + information content | **replace** — 10 datasets, new columns | 1.25 |
| Tab. 3 | Spearman ρ summary | **replace** — competitors, brackets, CIs | 1.25 |
| NEW | AE.3 representation comparison | T-17 | 0.75 |
| NEW | [28]/[29] inherited-modified-new | T-07 | 0.75 |
| NEW | Pair-accounting ladder, per dataset | T-02 | 0.5 |
| NEW | Calibration: bracket width, certification rate, ρ-gap vs `n` | T-05 | 0.75 |
| NEW | Canonical search-space schematic | ~~T-09~~ **done** | 0.5 |
| ~~NEW~~ | ~~**S2G/G2S worked example**~~ → **four panels, not one** (see the T-09 RESULT below) | ~~T-09~~ **done** | ~~**0.75**~~ **T-26 re-prices** |
| reuse | Graphical abstract — ~~regenerated from the two T-09 figures~~ **NOT regenerated; panel (b) carries numbers T-06 retired** | ~~T-09 →~~ **T-24** | **0** — submitted separately |
| NEW | Label surplus / collision table | T-18 | 0.75 |
| NEW | Critical-difference diagram (approximate regime only) | T-06 | 0.5 |
| reinstate | Generative-AI declaration | T-24 | 0.2 |
| reinstate | Acknowledgements | T-24 | 0.2 |
| reinstate | Author biographies + photos | T-24 | 0.6 |


> ## ✅ T-09 RESULT, 2026-08-25 — three inventory rows above are corrected in place
>
> **The schematic is done and answers R3.7c.** `canonical_search_tree.pdf`, 7.0 × 3.4 in,
> drawn by the enumerator the canonicalisation itself uses, with one subtree per starting
> node.
>
> **The worked example became four panels, not one.** S2G and G2S, each for the exhaustive
> *and* the pruned canonical form of one running example; 7.0 × 2.84 in each. The `0.75`
> priced above is stale — **T-26 owns re-pricing this row**, and the figure dimensions in
> [T-09 article notes](../tasks/T-09-article-notes.md) §3 are dimensions, not page fractions.
>
> **§3.2's cut order still holds and is now cheaper to execute.** Items 10 (schematic) and
> 11 (worked example) remain the only two no reviewer requires, and the four worked-example
> panels are independent files, so the cut can be partial rather than all-or-nothing.
>
> **§3.1's recovery "Trim worked examples (Ex. 2.3, Rem. 2.6, Rem. 2.11), 0.5 p" is now
> executable**: the S2G panel carries Example 2.3's content, and Example 2.3's own text
> contains an in-line self-correction (*"so after 0 but before 1 in circular order --- actually
> [0,2,1]"*, `methodology.tex:203`) that **T-11** should remove whether or not the figure lands.
>
> 🔴 **The graphical abstract is NOT regenerated, by decision.** `graphical_abtract.pdf`
> panel (b) prints `Wins: 99.6 %`, `β = 0.537`, `R² = 0.947` and `14,108×` — all retired by
> T-06, which withdrew Claim B at scale. Regenerating panel (a) alone would make the stale
> half look freshly checked. **T-24 inherits it**, with the two panels for (a) already built.
>
> Artifacts: `…/results/reports/T-09-explanatory-figures/figures/`, five figures as `.pdf`
> and `.png`. Full record: [T-09 article notes](../tasks/T-09-article-notes.md).

**Gross addition ≈ 12–13 pages against 0 pages of headroom.**

> **The composition is wrong in both directions and the composition is what the strategy rests on.**
> It **over-counts ≈ 3.5–4.5 p** by pricing Figs. 1–4 and Tabs. 1–3 as new when they are
> *replacements* (Tab. 1 is "keep"; Fig. 3 "demote" shrinks), and **under-counts ≈ 4.8–5.8 p** by
> omitting §1.x related work, §2.2.x complexity, §3.2 protocol prose, §3.3 implementation, §5
> limitations and the Suite 1/2 framing — all committed in §1, none in the page column. The
> magnitudes nearly cancel, **but the supplementary-relief plan targets inventory rows (≈ 2–2.5 p)
> while the ≈ 5 p of genuine growth is main-text prose that cannot move to supplementary.**
> **T-26 re-derives this as deltas, not gross sizes.**

---

## 3. The page budget

### 3.1 Recoveries, in order of size

| Recovery | Est. pages | Note |
|---|---:|---|
| **Fix float placement (E7)** | **up to 2.0** | `\floatpagefraction{1}` + `\textfraction{.001}` (`main.tex:66–67`) push all three algorithms onto dedicated float pages 33–35, **after the bibliography**. Inlining them recovers most of three near-empty pages. **Must run before any trim** |
| Compress the Thm 2.12 proof (`methodology.tex:639–726`) | 0.75 | the execution-path bijection can be tightened |
| Restructure §1 into §1 + §1.x | 0.5 | the current positioning survey is partly superseded |
| Trim worked examples (Ex. 2.3, Rem. 2.6, Rem. 2.11) | 0.5 | the new S2G/G2S figure carries the same content better |
| Demote Fig. 3 to a single small panel | 0.5 | D5 demotes the pooled analysis anyway |
| Merge the four §4 subsections' redundant preambles | 0.5 | |
| **Total recoverable** | **≈ 4.75** | |

**Recoveries (≈ 4.75) do not cover additions (≈ 12–13). The gap is ≈ 8 pages and cannot be closed by
editing.**

### 3.2 The one query that decides the strategy — ask on day 1

The letter names a mailbox and the plan has never used it (`mail.txt:24`). **Send one email
immediately:**

> Does supplementary material count toward the 35-page limit for a regular paper, and what file types
> are accepted for it at revision?

Everything below branches on the answer, which costs nothing but latency. Asking on day 18 is
worthless; asking on day 1 is free insurance on ~8 pages.

- **If supplementary does not count** — move the calibration tables, stratified analyses, the
  pair-accounting ladder, the sensitivity arms and the full per-dataset grids to supplementary; keep
  every *claim* and one summary table per claim in the main text. The gap closes comfortably and
  nothing scientific is lost.
- **If it does count** — the additive requests are prioritised against each other. **Pre-declared
  ranking, so the cut is not made under time pressure:**

  1. **AE.3 comparison table** — the Area Editor endorsed it in their own voice. Non-negotiable.
  2. **Claim scoping (B1–B6)** — almost no space, answers four comments.
  3. **Factual corrections** — Alg. 2, `n^{9.0}`, "super-polynomial", equivariance, E5/E6/E8. Nearly free.
  4. **Statistics rewrite** — R3.5a/b/c are three of R3's seven comments.
  5. **Related-work section** — AE.2, R1.2.
  6. **[28]/[29] delta table** — R3.1, R3.7b.
  7. **Size-scaling results (Suite 2)** — AE.1.
  8. **Label results** — R1.3.
  9. **Complexity section** — R3.4b, R3.7d.
  10. **Search-space schematic** — R3.7c, explicitly a "would benefit from".
  11. **S2G/G2S example figure** — author preference, **not a reviewer request**. First to go.

  **Items 10 and 11 are the only two no reviewer requires**, and 11 is 0.75 p — ~9 % of the gap.

### 3.3 Standing rule

"Ignore the page budget while drafting" is right for drafting and dangerous at the end. **T-15 is 2
days and must be scheduled with the same seriousness as a compute job**, because a 36-page manuscript
is rejected on format by the EiC regardless of the science. **Track the page count at every commit**
from the moment the first new section lands.

---

## 4. Response letter

### 4.1 Structure

| Part | Content |
|---|---|
| 0 | One-page summary of changes: full recompute under one cost model, 5 → 10 datasets, six competitor representations, replaced statistical protocol, new related-work section |
| 1 | **Area Editor** — AE.1, AE.2, AE.3, AE.4a/b/c, AE.5 |
| 2 | **Reviewer 1** — R1.1, R1.2, R1.3 |
| 3 | **Reviewer 3** — R3.1 … R3.7, sub-parts addressed individually |
| 4 | **Editor-in-Chief checklist** — EiC.a1–a4, EiC.b, EiC.c, each with its compliance evidence |
| 5 | **Corrections made on our own initiative** — E1–E12 |

**Per-comment format, fixed**: verbatim quotation → response → *exact* pointer to the changed location
(section number and, where useful, the artifact label). R3 checked thirteen of thirteen checkable
claims against the sources; **the letter must be equally checkable in the other direction.**

### 4.2 Drafted incrementally, not at the end

T-14 is "3 days, depends on all", which puts 41 demands into the last three days. **Replace with:
every ticket emits its response fragment when it closes.** T-14 then assembles, harmonises the
register and writes part 0 — three days is enough for that and is not enough for the alternative.

[demands](demands.md) is the index: one row per comment ID, one fragment per row, **an empty fragment
cell is a visible hole rather than a late discovery.**

### 4.3 Disclosure of E1–E12 — decided: yes, and the reviewer is answered first

> **The ordering rule, locked.** Nowhere does a self-found defect get raised before the reviewer's own
> comment has been answered on its own terms.
>
> - **Letter level** — Part 5 comes after Parts 1–4.
> - **Comment level** — where a self-found defect touches a comment (E2 ↔ R3.5a, E1 ↔ R1.3a,
>   E6 ↔ R1.3b, E5 ↔ R3.3a, E3/E4 ↔ R3.4c), the reviewer's request is satisfied first and the extra
>   finding follows as *"in addressing this we also found…"*, **never as the opening move**.
>
> A response leading with our own findings reads as deflection and buries the answer the reviewer is
> looking for. Answering first and volunteering second reads as diligence. **Same content, opposite
> impression.**

Four reasons to disclose: the recompute changes every one of these numbers regardless, so silence
means printed values change with no stated reason; R3 verified thirteen of thirteen checkable claims
in round 1, and assuming round 2 is less thorough is not a plan; **E2 is exactly what R3.5a asked for
and did not get**, so volunteering it converts a latent finding into evidence of diligence; and it
costs a third of a page.

### 4.4 No marked-up manuscript

The letter requires neither a clean unhighlighted main file nor a separate Summary of Changes —
unlike TPAMI. Given that §§3.1–5 are rewritten wholesale, a change-tracked version would be marked end
to end and would carry no information. **Locked: the letter's per-comment pointers are the change map.**

---

## 5. Ordering constraints — violating any of these means redoing work

1. **E7 float fix → then page trim.** Fixing float placement changes pagination; trimming first
   measures the wrong document.
2. **T-06 (numbers) → T-12 (claim scoping).** Every scoped claim quotes a number T-06 re-derives.
3. **T-07 ([28]/[29]) → T-08 (related work) → T-17 (comparison table).**
4. **MRM (D4) and the AIDS density stratification run in week 1.** Both can refute a central claim; a
   refutation in week 3 has no absorption time.
5. **patcog@elsevier.com query → §3.2 page strategy.** Day 1.
6. **T-08 + T-19 → T-26 → T-15.** The slot and page arithmetics must reconcile before the trim.
7. **Response fragments accrue per ticket → T-14 assembles.** Not the reverse.
