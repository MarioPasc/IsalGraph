# Prose — the narrative architecture of the revised manuscript

**Owner**: T-20 (rewrite), with T-12 (scoping), T-17 (§2 tables), T-07 (§2.3), T-13 (§3.2.3), T-22 (§3.3.3)
**Serves**: the delivery of every row in [demands](demands.md)
**Status**: v1.0, 2026-08-26. Written after T-01, T-02, T-03, T-04, T-04a, T-05, T-06, T-06-exhaustive, T-09 and T-27 closed.

This file answers one question: **in what order do the sentences go, and what does each one rest on.**
It does not restate measurements. Every number here names the file that owns it, and where the two
disagree, **that file wins**.

Related: [manuscript](manuscript.md) (rewrite map, page budget, response letter) ·
[corrections](corrections.md) (what changes and why) · [competitors](competitors.md) ·
[statistics](statistics.md) · [preregistration](preregistration.md) · [demands](demands.md)

Evidence: `results/reports/{T-05-bounded-ged,T-06-full-recompute,T-07-prior-work-delta,`
`T-09-explanatory-figures,T-13-complexity,T-27-ged-bound-bakeoff}/` ·
`../tasks/T-06-{FRAMING,POSITIONING,article-notes,EXHAUSTIVE-HANDOFF}.md` ·
`T-13-complexity/T-13-FRAMING.md`

> **v1.1, 2026-08-26** — T-13 landed after v1.0 and is the largest change since. It converts §5.3
> from an observational correlation into a **controlled experiment**, supplies §3.2.3 wholesale,
> dissolves R3.4c rather than reconciling it, and **retracts two claims the plan had signed**
> (§5.3's closing note). §10 is the full artifact inventory.
>
> **v1.2, 2026-08-26** — float revision under the PI's rule (§10): *a reviewer asked for it → main
> text; nobody asked → a float only if it carries a claim no prose can*. The head-to-head table goes
> to supplementary and the per-dataset ρ table shrinks to an inline block — **−1.15 p**, and it also
> caught a **one-page arithmetic error** in v1.1's inventory total. The [28]/[29] delta stays a
> separate table, with the reason recorded in §10 so the question does not return. The search-space
> schematic is kept and gets a re-render spec (**§10.3**) after the renderer was found to draw
> 5.5 pt labels into a figure that is then scaled to 0.674.
>
> **v1.3, 2026-08-26** — the **canonical search tree and the S2G/G2S worked example leave the body**
> and become one figure serving as **the graphical abstract**: **−1.2 p**, §3 drops to 10.4 p, and
> `graphical_abtract.pdf`'s four retired numbers are retired with it. The manuscript now sits at
> **≈ 32.4 p**, and that margin answers the algorithms question by measurement (**§10.4**):
> **Alg. 1 and Alg. 2 stay in the body, Alg. 3 goes to S7.** One coverage risk is created and is
> flagged, not absorbed — **R3.7c asked for the schematic in a named section**, and the graphical
> abstract is not one (§10's ⚠ box).
>
> **v1.4, 2026-08-26** — **T-07 closed during this session** and is folded in. It replaces §2.3's
> sufficiency argument with a measured one (**zero** formal results in either predecessor), supplies
> **Tab. 3 and the sufficiency paragraph already built and verified**, adds **seven red lines** to
> §2, hands §6.3 the R3.2 pre-emption that Tab. 3 no longer carries, and records the
> **`\linespread{1}` trap** (§10.5 item 4) — without it a table is over a page instead of 0.67.

---

## 0. Four decisions taken 2026-08-26, and one consequence to accept

| # | Decision | Consequence |
|---|---|---|
| **P1** | **Thesis is the properties, with the GED failure reframed as a field-level result** — not abandoned, not defended | §1's contribution list and §5's order both follow from this. §6.2 exists because of it |
| **P2** | **`isalgraph_pruned` is the primary arm.** `isalgraph_exhaustive` is a declared measured improvement | The pre-registered family is not re-based. Every compactness figure is a *conservative* bound, which is a stronger sentence than the raw number |
| **P3** | **Supplementary is a separate uploaded PDF, never a LaTeX `\appendix`** | §8. This is the single largest page lever and it is a file-format decision, not a writing one |
| **P4** | **Raw `adjacency`, `graph6`, `sparse6` are removed from the paper entirely** | §9. Favourable and defensible — but it opens one hole in R1.2 that costs exactly one sentence to close |

### P3 — what is actually verified, and the one inference

Pattern Recognition's Guide for Authors, verbatim:

> *"Please limit your manuscript to between 20 and 35 pages (including figures and tables, ideally
> embedded in the text, as well as references, biosketches and **appendices**), double-spaced and in
> a single column with numbered pages."*

and, on supplementary material:

> *"Cite all supplementary files in the manuscript text. […] All supplementary materials provided
> will appear online in the **exact same way as received**. These files will not be checked,
> formatted or typeset by the production team."*

**Verified**: appendices count. Supplementary is a separately-uploaded, non-typeset artifact, and
*must* be cited from the main text. **Inferred, not quoted**: that a non-typeset attached file falls
outside a page limit stated over the typeset manuscript. The inference is standard Elsevier practice
and the two clauses are only consistent under it — a file that "appears exactly as received" is not
paginated with the article.

> 🔴 **The operative rule: nothing that must not count may live behind `\appendix`.** The plan's
> §3.2 query to `patcog@elsevier.com` was framed as *"does supplementary count"*; the sharper
> question, and the one that changes the LaTeX, is *"appendix or separate file"*. Build the
> supplement as `supplementary.pdf` with its own preamble. If the query is still unsent, send it —
> but the format decision does not wait on the answer, because the separate file is safe under both
> outcomes and the appendix is safe under neither.

---

## 1. The thesis

One paragraph, and every section is accountable to it.

> **IsalGraph encodes graph structure as an executable instruction string whose canonical form is a
> complete invariant. We verify that completeness on 24.8 million pairs, show the encoding is the
> most compact of the representations whose canonical form is intrinsic to the code, and characterise
> its cost: size sets how many frames the encoder emits, and at fixed size the branching is governed
> by |Aut(G)|. Its edit distance carries information about graph edit distance beyond node count and
> density — significantly on 19 of 19 identifiable fits — but node count carries more of it on 17 of
> those 19, and on a controlled family at three distortion levels the string's correlation falls from
> 0.93 to 0.67 while the trivial size baseline stays flat at 0.92. On the standard benchmarks
> node-count difference alone attains ρ = 0.71–0.997 against ground-truth GED, and above n ≈ 40 no
> representation we tested, ours or any competitor, is reliably distinguishable from ρ = 0. We report
> that as a property of the evaluation protocol, measured with the instrument the representation
> itself supplies.**

> ### 🔴 v1.6, 2026-08-26 — this paragraph carried TWO red-line violations and is rewritten
>
> Both were in the most load-bearing sentence in the plan, and both flattered us.
>
> **1. *"characterise its cost as governed by |Aut(G)| rather than by size"*** is T-13's banned
> sentence verbatim (§5.3's red lines). **Size sets the frame count and it matters** — the dense
> ladders censor everywhere for both canonical arms. |Aut| governs the **branching, at fixed size**.
>
> **2. *"Its edit distance tracks graph edit distance where structure is what varies and size is
> not"* is refuted by our own control experiment, in our favour.** The IAM Letter family **is** that
> condition — one generator, three distortion levels, node count 4.07 → 4.58 while mean edge count
> rises **49 %**. In exactly that regime the string runs **0.9278 → 0.8833 → 0.6660** while the size
> null runs **0.9139 → 0.9146 → 0.9195**. The representation degrades by a quarter where structure
> is what varies; the baseline does not move. **The clause asserted the inverse of C9, which is
> frozen three sections below it in this same file.**
>
> **Why it survived**: it is the most attractive positive sentence left in the paper. That is the
> whole reason it needed checking, and it is the argument for checking the sentences we most want to
> be true *first* rather than last.

**Why this is the fair setting the PI asked for.** It puts IsalGraph first where it is first, concedes
where it is beaten, and reports **two** negatives that are genuinely different: the benchmark is
size-dominated, which is measured on 21.7 M pairs and applies to every competitor identically; and
the representation itself degrades as structure becomes what varies, which is ours alone and is what
the IAM Letter control isolates. 🔴 **Do not collapse the second into the first.** The control exists
precisely to rebut "the benchmarks are size-dominated, so nothing could have done better" — there
node count is nearly constant, the null holds at 0.92, and the arm still collapses. A reader cannot
accuse it of special pleading because the same instrument that convicts IsalGraph convicts everything
else, and we are the ones who built it.

**The categorical differentiator, stated once and never argued from a table.** The instruction string
is a *program*: every prefix is a valid construction sequence producing a subgraph. No serialisation
in the comparison has this property, and neither experiment adjudicates it. That is where the novelty
lives, and it is the paragraph that reframes the comparison if the comparison reads as narrow.
It belongs in **§6.1**, not in §5, because it is not a result.

---

## 2. The hypothesis spine

The user-visible skeleton. Each H is **stated in §1**, **operationalised in §4**, **answered in §5**,
**interpreted in §6**. A reader should be able to trace any of the four end to end without
backtracking.

| | Hypothesis | Test | Verdict | Where |
|---|---|---|---|---|
| **H1** | The canonical string is a complete invariant — `w*_G = w*_H ⟺ G ≅ H` within a directedness class | Theorem (§3.3.3) + collision census over every GED-positive pair | **CONFIRMED.** 0 of 24,764,422 | §5.1 |
| **H2** | It is compact relative to established reversible serialisations | Per-stratum information content vs 4 competitors, both bit conventions, IUT | **CONFIRMED where scoped, REFUTED where not.** First among canonical codes above n ≈ 20 (112/112 vs min-DFS, +215 bits, advantage *grows* with n); beaten by nauty-sparse6 at scale | §5.2 |
| **H3** | Its edit distance approximates graph edit distance | ρ vs exact GED (n ≤ 12) and a proven bracket (n ≤ 98); size null; within-`n`; MRM; **and the same test against a second, structural reference** | **REFUTED against graph edit distance — and the refutation is substantially a property of the reference.** Below its `\|Δn\|` null on 17 of 25 GED records; above n ≈ 40 no representation tested is distinguishable from ρ = 0. **Against a Weisfeiler–Lehman kernel reference, the same distances on the same pairs clear the same null on 5 of 5 Suite-1 datasets where 1 of 5 clears against exact GED** | §5.4 |
| **H4** | Canonicalisation cost is governed by graph size | **A controlled ladder experiment** holding `n`, `m` and the degree sequence fixed while \|Aut(G)\| varies, with a search-free null arm; plus the cohort encode census | **REFUTED, and replaced by something better.** \|Aut(G)\| governs the branching at fixed size: ρ = +0.892, 11 of 12 cells, p = 0.0064, null arm flat at 1.0–1.1× | §5.3 |

**Two of four are refuted and the paper is stronger for it.** H4's refutation is *predictive* — a user
computes |Aut| with a mature library and knows in advance whether the method will encode their
graphs — and
it is the only claim in the paper established by a **controlled experiment** rather than an
observational one, which is worth saying in those words. H3's refutation is a statement about the
field. Say both plainly; neither needs softening, and softening
either forfeits the credibility that carries H1 and H2.

> **H3 is scoped, not softened, and the distinction is the whole argument.** *"The canonical string
> does not approximate graph edit distance on these benchmarks"* stays, in those words, with its
> numbers intact. What is added is the reason, and the reason is testable: §5.4 already argues that
> the reference is size-dominated (C5), and a claim of that form **predicts** that a reference which
> is *not* size-dominated will behave differently. It does. Holding every representation distance at
> its cached value and replacing only `d_ref`, the arm goes from clearing its own `|Δn|` null on 1 of
> 5 Suite-1 datasets to 5 of 5.
>
> This is the difference between *"the encoding failed"* and *"the encoding tracks structure, and
> graph edit distance on these cohorts mostly does not measure structure"* — and the second is both
> better for the paper and the one the data supports. It is **not** a rescue of H3 as stated: the
> hypothesis says *graph edit distance*, and against graph edit distance it is refuted. Write that
> sentence first and the scoping second, never the other way round.

> **Drafting rule.** Every subsection of §5 opens by naming the hypothesis it answers and closes with
> one sentence of interpretation. No result is presented without the reader knowing what question it
> was asked. This is what turns §5 from an inventory into an argument, and it costs about eight
> sentences in total.

---

## 3. Section map and page budget

> ## 🔴 v1.5, 2026-08-26 — SUPERSEDED BY MEASUREMENT. Wave A is built; read this before the table.
>
> **The table below is the planning estimate. Three of its rows are now measured and the estimate
> was wrong every time, always in our favour.** §1 **2.361**, §2 **4.14**, §3 **12.23** — measured
> from page spans on a green build, 30 p clean / 31 p blue.
>
> **Two arithmetic defects of mine, both confirmed:**
> - **§3 double-counted E7.** The row takes §3 from 14.0 to 10.4 *and* a separate `−2.0` row claims
>   E7's float recovery. The algorithms returning to the flow **is** that recovery and it is already
>   inside the measured 12.23. **There is no separate E7 row.** Counted once, §3's own itemised cuts
>   net **−1.85** from 14.0, giving **12.15** against a measured 12.23 — within 0.08 p.
>   **10.4 was never reachable.**
> - **The cut table's Net row says −1.35 where its items sum to −1.85.** The −1.35 was correct in
>   v1.2 and went stale when the schematic left for the graphical abstract in v1.3; I updated the
>   additions row and not the net.
>
> **Projection, with the undrafted half labelled as the estimate it is:**
>
> | | p | basis |
> |---|---:|---|
> | §1 + §2 + §3 | **18.73** | measured |
> | §4 + §5 + §6 + §7 | 13.6 | **estimate — from the file that has now been wrong three times** |
> | front matter with abstract | 0.9 | measured |
> | references + back matter at ~45 entries | 5.1 | measured 4.0 at 25, scaled |
> | **total** | **≈ 38.3** | **against a hard 35 — a gap of 3.3 p** |
>
> **The pre-declared cut order does not close it**, and this is measured rather than assumed:
> Alg. 1 → S7 is **0.79 p**, Tab. 5 → S5 **0.446**, Tab. 1 → S7 **0.531**. All three total **1.77 p**
> and land at **36.5**, still over, with every reserve spent. **See §3.1 for the decision.**

Delta against the submitted 35 pages, **not** gross sizes — the correction `manuscript.md` §2 demands.

| § | Section | Submitted | Target | Δ | Drivers |
|---|---|---:|---:|---:|---|
| — | Front matter, abstract, keywords | 1.0 | **0.9 ✔** | −0.1 | B1, B4, B6, E5, R3.2 |
| 1 | Introduction | 3.0 | **2.36 ✔** | **−0.64** | positioning moves to §2; R3.7e; R3.1b |
| **2** | **Related work and positioning** *(NEW)* | 0 | **4.14 ✔** | **+4.14** | AE.2, **AE.3**, R1.2a/b, R3.1a(i)+(ii), R3.7b |
| 3 | Method | 14.0 | **12.23 ✔** | **−1.77** | +T-13 cost model, −proof, −Alg. 3, −examples, **−3 figures to the graphical abstract** |
| 4 | Experimental design | 4.5 | 4.5 *(est.)* | 0 | AE.1, AE.4a/b/c, R3.5a/b/c, R3.6a |
| 5 | Results | 6.0 | 6.1 *(est.)* | **+0.1** | R1.1, AE.1, AE.4a, **T-13's controlled cost law**, R3.4c, E3/E4 |
| **6** | **Discussion and limitations** *(NEW/expanded)* | 0 | **1.8** | **+1.8** | R3.7a, R3.1b, R1.3c, R3.6b |
| 7 | Conclusions and future work | 3.0 | 1.2 | **−1.8** | numbers move to §5; R1.3d |
| — | References (43 → ~50) | 2.5 | 3.0 | +0.5 | EiC.a1, EiC.b, T-19 |
| — | Back matter + reinstated (AI decl., ack., bios) | 0.7 | 1.7 | **+1.0** | E11, E12, EiC compliance |
| — | ~~**E7 float-placement fix**~~ | — | **0** | **0** | 🔴 **Double-counted — deleted.** E7's recovery is the algorithms returning to the flow, and that is already inside §3's measured 12.23 |
| | **Total** | **34.7** *(est.)* | **~33.0** | **−1.7** | |

> **The submitted column is an estimate and sums to 34.7 against a measured 35.** The ~0.3 p
> difference is float and page-break slack that no section owns. **Do not spend it** — it is the
> reconciliation error, not headroom.

**It fits with ~2.6 p of real margin, and it did not two revisions ago.** Two moves bought it: v1.2
sent the head-to-head table to supplementary and shrank the per-dataset table to an inline block
(**−1.15 p**), and v1.3 merged the three §3 figures into the graphical abstract (**−1.2 p**). §10's
cut order holds a further **1.4 p** in reserve. **That margin is what makes §10.4's answer on the
algorithms a measurement rather than a preference.**

**Track the page count at every commit** anyway — E7 is still load-bearing, and a margin that exists
on a spreadsheet is not a margin that survives `pdflatex`.

> ### Measured against planned, 2026-08-26 — the scaffold is built and the floats are real
>
> Writer measured every float **in place, with its real caption**, against a 550.28 pt page. Two
> corrections to this table follow and both are already applied above.
>
> - **§2 is 3.05 p, not 2.5.** Its floats alone measure 1.40 p, leaving 1.10 p for prose that needs
>   about 1.65. **Approved rather than compressed**: §2.2 is the artifact the Area Editor endorsed in
>   their own voice and §2.3 is R3.1a(i)+(ii). Trimming either to hit a number set before T-07
>   existed would trade coverage for arithmetic.
> - **Front matter + references + back matter measure 4.0 p against the 5.7 planned** — a 1.7 p
>   over-estimate on my side, and where §2's overrun is paid from. **Do not spend the rest**: the
>   scaffold's bibliography is empty and EiC.a1 requires 35–55 items with ≥3 new PR-venue references
>   from T-19, so that line grows.
> - Individual floats came in close: comparison 0.671 (planned 0.80), delta 0.733 (0.70),
>   instruction set 0.531 (0.50), datasets 0.632 (**1.20** — I over-budgeted), bits 0.446 (0.30),
>   information content 0.882 (0.80), cost law 0.691 (0.70), rho 0.756 (0.90), Alg. 1 **0.814**
>   (0.60), Alg. 2 0.897 (0.90). **Alg. 1 is the one real overrun at +0.21 p, and it is CUT-2.**
>
> ### 🔴 Every prose page figure in this file is an estimate, and the one that got checked was out by 4×
>
> **The sufficiency paragraph.** §2.3 was budgeted ~0.8 p with Tab. 3 measured at 0.73, leaving
> ~0.07 p for the paragraph. Measured in isolation with `\pagetotal`: **226.0 pt = 0.411 p**,
> confirmed at 0.393 p by an independent harness. **Out by about 4×, on a frozen 145-word paragraph
> whose word count this file states.** §2.3 is 1.14 p, not 0.8.
>
> 🔴 **The fix is NOT a words-per-page constant.** That was proposed and it repeats the error in a
> new form: measured density is **249 words/page in §2 against 321 in §1** — same document, same
> class, 29 % apart. §2 carries **722 words against §1's 758** and occupies more pages, because
> three subsection headings, a section heading, thirteen `\cite`s, heavy `\texttt` and two floats
> are not words. §3 will differ again with algorithms and displayed maths. **A single conversion
> factor is the same mistake with a decimal point in it.**
>
> **The rule: budgets are measured per section, never converted.** Draft, build, read the page span.
> Until a section is drafted its number in §3's table is a planning estimate and must be labelled as
> one — which is what "est." on the submitted column has always meant and what the target column
> never said.
>
> **E7 is confirmed by measurement, and its cause was only half what this file said.** Algorithms now
> land on pp. 5–6 with the bibliography on p. 12, against pp. 33–35 in the submitted manuscript. But
> deleting `\floatpagefraction`/`\textfraction` was **necessary and not sufficient** — untreated by
> `\linespread{1}` the two listings are `Float too large for page` by 87 pt and 197 pt, so LaTeX
> could not place them anywhere. See §10.5 item 5. E7 is the largest single recovery on the
board and it is a two-line change to `main.tex:66–67`. **Run it first**; trimming before it measures
the wrong document ([manuscript](manuscript.md) §5.1).

### 3.1 The decision on the 3.3 p gap — taken 2026-08-26, before Wave B

**The gap is not closed by trimming, and trimming §5 would be the wrong failure mode**: §5 is where
the results live, and a compressed results section is how a revision loses the demands it was written
to answer. Four moves, in order of size.

**1. Theorem 2.12's proof moves to the supplement. Statement and a short sketch stay in §3.3.3.**
The single largest lever left, and it is principled under §10's own rule — *a reviewer asked for it →
main text*. **R3.3b asks for the statement to be restated within a fixed directedness class, with the
flag hypothesis moved from the proof into the statement, and for the proof steps to be re-verified.**
Re-verifying is an obligation we discharge; **printing the proof is not what was asked.** The
statement carries the demand, and T-22's re-verification carries the diligence.
🔴 **Cost it by `\iffalse` before committing**, the way Alg. 1 was costed at 0.79 p. This file has
been wrong four times by estimating; it does not get a fifth.

**2. Take the pre-declared cuts — 1.24 p, not 1.77.** Tab. 5 → S5 (0.446) and Alg. 1 → S7 (0.79)
are taken. **Alg. 2 stays**, without exception: R3.4a names its lines.

> ### 🔴 Tab. 1 → S7 is CANCELLED. The cut order was fixed before the answer that needs it existed.
>
> §3.2.2 discharges R3.4a by naming a **contradiction against Table 1**: *"both guards, and both
> duplicate checks, were stated on the transposed pair, which contradicts Table~1, where `C` inserts
> from the primary to the secondary pointer."* **R3 found that defect by comparing Algorithm 2
> against Table 1.** Moving Table 1 to a non-typeset attachment removes one side of the comparison
> the reviewer actually performed, in the one section where they already found a real inconsistency
> — the same defect §10.0 avoided for R3.5b.
>
> **A pre-declaration made in ignorance of a later fact yields to the fact.** The cut order exists so
> the decision is not made under deadline pressure, not so it survives evidence. **Tab. 1 stays.**

**3. §4–§7 are re-budgeted from measured density before Wave B drafts them, not trimmed after.**
Re-scoping is cheap now and expensive later. Densities to use: **249 words/page where a section
carries many subsection headings, citations and `\texttt`** (§2's shape — §4 has six subsections and
will behave the same), **321 where it is continuous prose** (§1's shape). **Do not use a single
constant** (§3's boxed rule).

**4. Bibliography targets ~40 cited entries, not 45.** EiC.a1's floor is 35 and T-19 needs ≥3 new
PR-venue references; 40 satisfies both with ~0.5 p less than 45.

### 3.2 The closing ladder — pre-declared 2026-08-26, executed only against a MEASURED total

**Nothing below is taken on a projection.** Every projection in this file has been wrong, always in
our favour, and pre-spending against one is how a demand gets cut to pay for an estimate. Wave B
lands, the document is measured, and then the ladder runs **top-down, stopping at 35**.

| # | Move | ≈ p | Cost |
|---|---|---:|---|
| 1 | Bibliography to EiC.a1's **floor of 35**, not 40 | 0.30 | none — 35 is compliant |
| 2 | Trim the information-content caption | 0.15 | none — it was written long |
| 3 | Cor. 2.13's proof → **S11** | 0.10 | none — five lines, statement stays |
| 4 | §3.4 topological structure, thinned again | ~0.4 | low — partly superseded by §6.1 |
| 5 | §7 Conclusions 1.2 → 1.0 | 0.20 | low — every number in it is already in §5 |
| 6 | §5.5 confirmatory family → the inline 4-row block plus three sentences | ~0.4 | **the split by direction must survive** |
| 7 | **Author biographies → dropped again** | **0.72** | none we own — they were cut from the *submitted* manuscript with the note *"to meet the 35-page limit"*, and Elsevier routinely takes them at proof stage. E12 asked for reinstatement; the authors' own earlier trade is the precedent for reversing it |

> **🔴 The acknowledgements are NOT on this ladder and are not ours to cut.** They measure 0.656 p and
> name PPRO-TIC163-G-2023, PID2022-136764OA-I00, ERDF, PUNI-003_2023, ATECH-25-02 and PI25/02129.
> Grant conditions frequently require the text verbatim. **That is a compliance decision for the
> authors, not a formatting choice for us.**

**🔴 Never, at any page count**: Algorithm 2 · the Theorem 2.12 **statement** · Tab. 1, 2, 3 or 4 ·
any claim · any scope clause. If the ladder runs out, the manuscript asks the Area Editor before it
drops a demand.

> ### The instruction that matters most for Wave B
>
> **§5's overflow goes to the supplement, never into compressed prose.** §5.4 alone carries eight
> numbered moves and is the section most likely to overrun. When it does, the MRM detail goes to
> **S6** and the per-competitor and per-dataset grids to **S8** — both already exist as supplementary
> sections and both are already cited from the main text. **The claims stay; the apparatus moves.**
> A results section that keeps every claim and points at its evidence answers the reviewers; one that
> keeps every table and drops a claim does not.

**Where the −1.85 in §3 comes from**, itemised so it is checkable rather than hoped for:

| Cut | p | Note |
|---|---:|---|
| Compress the Thm 2.12 proof (`methodology.tex:639–726`) | 0.75 | the execution-path bijection tightens; T-22 is re-verifying all three steps anyway |
| Algorithm 3 (pruned canonicalisation) listing → **S7** | 0.70 | keep a four-line prose description and the definition in the main text |
| Example 2.3, Remarks 2.6 / 2.11 | 0.50 | the S2G/G2S worked example carries them better; E8's printed self-correction dies here regardless |
| §2.4 Topological Structure, thinned | 0.40 | partly superseded by §6.1; its figure was already cut |
| Merge redundant subsection preambles | 0.30 | |
| **Additions** — §3.2.3 cost model (+0.8) | −0.80 | reviewer-requested and non-negotiable |
| **Net** | **−1.85** | **This is the whole of it.** E7's recovery is the algorithms returning to the flow and is inside these items, not a separate row |

---

## 4. Section-by-section brief

Each entry: **what it must say, in what order** · what it must **not** say · the artifact · the
supplementary pointer · the demands discharged.

### Abstract — five moves, in this order

1. **What it is.** Nine-instruction virtual machine; any **connected** simple graph (directed: a root
   reaching every node) encodes to a string; every string decodes to a valid graph. *(B1, E5)*
2. **The property.** The canonical string is a complete invariant **within a fixed directedness
   class**; verified with **zero collisions across 24.8 M pairs** over 10 datasets to `n = 98`. *(H1, R3.3b)*
3. **Compactness, scoped inline.** Most compact of the representations whose canonical form is
   intrinsic to the code — shorter than gSpan min-DFS on **112 of 112** strata above `n = 20` — and
   the advantage **grows with size**. Canonically-labelled edge-list serialisations are more compact
   at scale. *(H2)*
4. **Cost.** **Size sets the frame count; at fixed size the branching is governed by |Aut(G)|.** *(H4)* 🔴 **Never *"governed by |Aut|, not by n"*** — that is T-13's banned sentence, and until v1.6 this brief printed it verbatim.
5. **Fidelity, and the field result.** On these benchmarks node-count difference alone attains
   ρ = 0.71–0.997 against ground-truth GED; within fixed `n`, above `n ≈ 40`, no representation
   tested is reliably distinguishable from ρ = 0. *(H3, R3.6b, AE.1)*

**Delete outright**: "strongly correlates with graph edit distance", "language-model-compatible",
"with direct applications in graph similarity search, graph generation, and graph-conditioned
language modelling". *(R3.2, R3.6b, B4)* — the last is the sentence R3.2 is aimed at and it cannot
survive a paper with no sequential-model experiment.

---

### §1 Introduction — 2.2 p

**Order**: the problem (graphs as sequences, why a *program* rather than a description) → the four
properties, **softened and unified** (B6) → **scope**, stated positively (B1: undirected connected;
directed with a root reaching all nodes; **S2G total, G2S partial**) → **the four claims as testable
hypotheses**, §2's table in prose → roadmap.

> ### 🔴 The properties and the hypotheses are two DIFFERENT four-sets. Do not conflate them.
>
> An earlier version of this brief asked for "the four properties" and "the four claims" in
> consecutive clauses without saying they are different sets, which sent the §1 agent looking for a
> single list. They overlap in exactly one member.
>
> | | set |
> |---|---|
> | **Properties** (what the representation *is*) | validity · reversibility · canonicity · **compactness** |
> | **Hypotheses** (what we *test*) | H1 completeness · H2 **compactness** · H3 fidelity · H4 cost |
>
> Compactness is in both. Validity and reversibility are properties only; cost is a hypothesis only.
>
> **The submitted paper's own two lists disagree** — `introduction.tex:33` has four members
> (compact / reversible / structure-preserving / canonicalisable), `conclusion.tex:74` has three
> (universal validity / reversibility / canonical completeness). B6 requires unification, not just
> softening, so a merged set is mandatory rather than optional.
>
> **Resolution, shipped in §1 and binding on §7**: *structure-preserving* is renamed **locality** and
> **demoted from a property to a measured quantity carried by H3**. Listing locality among the
> properties IsalGraph *has* would contradict §5.4 inside the same paper — below its size null on 17
> of 25 records — and that self-contradiction is precisely what B6 exists to stop. Nothing is
> silently dropped: every member of both submitted lists survives, as a property or as a hypothesis.

- **equivariance → invariance** everywhere. `M → P M Pᵀ` *is* equivariance; invariance is what breaks. *(R3.7e)*
- **"No existing method satisfies all four"** → softened and pointed at **Tab. 2**. *(R3.1b, B6)*
- Positioning prose moves wholesale to §2. The introduction motivates; it no longer surveys.

> **The contribution list is the spine and is worth its 0.3 p.** Once H1–H4 are named here, every
> later section refers to them in three words instead of re-motivating itself. That is where the
> −0.8 p is actually recovered.

**Discharges**: R3.7e, R3.1b, B1, B6, E5 (partial).

---

### §2 Related work and positioning — 3.05 p — **NEW**

The Area Editor endorsed §2.2 in their own voice (AE.3). This section is non-negotiable and is the
largest single addition. Giving it its own top-level section, rather than a `§1.x` subsection, puts
AE.2, AE.3, R1.2a, R1.2b, R3.1a(i), R3.1a(ii) and R3.7b in **one place a reviewer can check**.

**§2.1 Graph canonicalisation and reversible serialisation** (~0.7 p)
nauty/Traces and canonical labelling; graph6/sparse6 as wire formats; AGM's canonical adjacency
matrix; gSpan's minimum DFS code; Babai on the complexity of the underlying problem; Weisfeiler–Lehman
as an invariant but **non-invertible** feature map. Each cited **individually with a comment** —
`\cite{a,b,c}` groups are an EiC.a4 failure. *(AE.2, R1.2a, EiC.a4, EiC.b)*

> 🔴 **The one sentence that P4 makes load-bearing.** With raw graph6/sparse6 gone as competitors,
> this is where R1.2's conceptual question is answered: *"graph6 and sparse6 serialise one specific
> vertex ordering, so an edit distance between them measures node ordering rather than structure
> unless a canonical labelling is applied first; we therefore evaluate them only in their
> nauty-canonicalised forms."* Cite **S4** for the measurement. One sentence, no table row, no
> competitor column. **See §9.**

**§2.2 A side-by-side comparison** (~1.0 p) — **Tab. 2**
Rows = representations, grouped by the a priori **family taxonomy** (§7). Columns = R1.2b's five axes
verbatim — **uniqueness, expressiveness, computational efficiency, scalability, downstream learning**
— plus reversibility, canonicity, metric admissibility, and the shared edit-operation alphabet R3.6a
asks for. **`downstream learning` reads "not evaluated" for every row including ours**, which is
honest, costs nothing, and pre-empts the obvious objection to R3.2's decline.
*(AE.3, R1.2b, R3.1b, R3.6a)*

**§2.3 Relation to the authors' prior work** (**measured 1.14 p** — Tab. 3 at 0.73 plus the
sufficiency paragraph at **0.41**) — **Tab. 3** + one paragraph
*(R3.1a(i), R3.1a(ii), R3.7b, AE.5)*

Tab. 3 is three columns — **inherited / modified / new** — against [28] and [29]. Then the
**sufficiency paragraph, 120–150 words, closing the section**. Without it the delta table is evidence
*against* us ([manuscript](manuscript.md) §1, [demands](demands.md) R3.1a(ii)).

> **The sufficiency argument, now measured rather than argued (T-07, closed 2026-08-26).** Over the
> complete text of both predecessors, the terms `theorem`, `proof`, `lemma`, `proposition`,
> `complete invariant` and `graph edit distance` occur **zero times each**. **Neither predecessor
> contains a single formal result.** [28]'s canonical string is *defined as* the greedy encoder's
> output on an adjacency matrix that presupposes a complete order on the vertices — canonical **per
> matrix**, not per isomorphism class. [29] argues relabelling invariance in **three sentences of
> prose, in one direction only**, resting on an explicit assumption, and no experiment there
> measures uniqueness or a collision rate.
>
> **What that licenses**: *"We prove what [29] asserted."*
> 🔴 **What it forbids**: *"[29] proves X and we extend it"* — rejected by anyone who has read [29].

**Tab. 3 and this paragraph are built and verified.** Do not rebuild either:
`…/results/reports/T-07-prior-work-delta/artifacts/{tab3_prior_work_delta,sufficiency_paragraph}.tex`.
Composition is frozen at **9 components — 3 inherited, 5 modified, 1 new**, under an attribution rule
committed **before any source was read**. The one "new" is the isomorphism invariance of the
canonical form. **Never add a second "new" row**: the metric corollary was deliberately folded into
the theorem row, because printing a result and its corollary as two novelties is the inflation
R3.1a is probing.

**Do not** write the sufficiency paragraph as a defence of R3.2's decline. It stands on the theorem,
and the drafted paragraph is clean of it. The absence of a sequential-model experiment is conceded
in §6.3, once, on its own terms.

> ### 🔴 Seven things §2 may not say — `T-07-article-notes.md` §5, read it in full
>
> - **No correlation statistic may be cited from [29].** It reports none — `pearson`, `spearman`,
>   `p-value` each occur zero times — despite its Discussion claiming a strong correlation. Cite the
>   qualitative monotone trend, and prefer the abstract's hedged wording.
> - **[29] is not a complexity comparison point.** It states no bound of any kind, including for its
>   own normalisation.
> - **No node-count ceiling and no disconnected-graph restriction may be attributed to [29].** The
>   paper is silent on both. What carries the "generic topology redesign" claim is narrower: the
>   container is a hydrogen list, the seed state is H₂, and insertion degree is a per-element constant.
> - **Do not quote [28]'s equation (1)** — it defines an edge as 0, contradicting its own equation (7)
>   and its encoder. Verified on the PDF, not a text-extraction artefact.
> - **Do not claim [29]'s normalisation searches only the starting atom.** The CC-BY text strips all
>   three algorithm listings, so it is not sourceable. Nothing printed depends on it.
> - **Do not print any criticism of [29]'s implementation.** Co-authored prior work, no bearing on
>   Tab. 3, out of scope.
> - The `24,764,422` in the sufficiency paragraph is **T-06's C1, inherited not re-measured**. If C1
>   moves, that paragraph moves with it.

---

### §3 Method — 10.4 p — *(was §2)*

Structure unchanged except where a demand forces it. **Three insertions, four cuts** (§3's table).

**§3.1 The virtual machine and instruction set** — **Tab. 1** kept as is. *(no demand)*

**§3.2 Graph-to-string**
- **§3.2.1** displacement pairs and cost ordering — the sort key is `(|a|+|b|, |a|, (a,b))`, all three
  components (invariant 5).
- **§3.2.2** the greedy algorithm — **Algorithm 2 lines 24–30 rewritten to match the implementation,
  guards *and* duplicate checks.** R3 found a real inconsistency with **Tab. 1** and the
  implementation is what is right. *(R3.4a)*
  **The S2G/G2S worked example is no longer a body figure** (§10). Where §3.2 wants to show the two
  directions as one object — the string draining into a graph and the graph draining into a string
  on the same running example — it **points at the graphical abstract**, which now carries both
  panels. Alg. 1 and Alg. 2 are what remain in the body, which is why §10.4 keeps them.
- **§3.2.3 Cost model** *(NEW, ~0.9 p)* — **the three-way separation R3.7d asks for, signposted as
  three named things**: (i) theoretical complexity, (ii) worst-case search behaviour, (iii) empirical
  scaling. Everything here is T-13. Five moves, in order:
  1. **`P(M)` is recomputed, at every frame**, at all three call sites, `Θ(M² log M)` each.
     R3.4b's question landed on **the largest single constant factor in the implementation**: over
     1,109,460 measured frames the pair lists match `Σ(2M+1)²` exactly and take only **12 distinct
     values of `M`**, which is why memoising them is worth **25.5×–108.6×** to the C++ engine. Say
     that — a reviewer who finds a real inefficiency should see it acknowledged and quantified.
  2. **The four named operations costed** — pair generation `O(m n² log n)`, pair scanning
     `O(m n²)` worst, pointer walking **`O(m n³)`, the dominant worst-case term**, neighbour checks
     `O(m Δ)`. Validated against realised counts on **178,886 (graph, start) parity pairs, 0
     mismatches**; `core/` was not modified to obtain them. Realised scan depth is **2.1 % of worst
     case** and the first pair is accepted in **26.45 %** of frames.
  3. **Proposition 1 (the invariance floor) and Corollaries 2–3.** Any node invariant is constant on
     every orbit, so its partition is a coarsening of the orbit partition; the residual branching is
     automorphic redundancy, and refining the key can reach the orbit partition and **no further**.
     Verified: **0 violations over 16,370 cohort graphs and 664 constructed graphs.** This replaces
     the word "exponential" with a characterisation.
  4. **The one-sided displacement lemma** — at every accepted `V` frame `b = 0` and at every accepted
     `v` frame `a = 0`, three lines from the scan order, **0 exceptions in 215,270 frames**. It makes
     `|w| = m + Σ_f(|a_f| + |b_f|)` exact, and it shows string length tracks the **cyclomatic number
     `m − n + 1`** more than `n`: insertion frames average 0.238 movement characters against chord
     frames' 1.480. **A small original result — without it the length formula reads as an error.**
  5. **The fitted exponent is labelled a cohort property, not a complexity result** — §10.1.
  *(R3.4b, R3.4c, R3.7d)*

**§3.3 Canonicalisation**
- **§3.3.1** the exhaustive canonical string. **The search-space schematic is no longer a body
  figure**: it is merged into the graphical abstract (§10). *(R3.7c — read §10's ⚠ box before
  drafting this subsection; the recommendation is to reproduce the merged figure as **Fig. S1** so
  this paragraph can cite something numbered.)*
  **Prose must now carry what the figure carried**, which is three sentences and no page cost: the
  search ranges over the **starting node** and over the uninserted-neighbour choice; the
  displacement ordering and the `V ≻ v ≻ C ≻ c` priority are intrinsic and never branched over; and
  on the running example greedy encoding from the six starting nodes gives strings of length
  **9, 10, 9, 11, 10, 10**, only one of which attains `w*_G`. That last measurement is what makes
  the start node a search dimension rather than a free choice.
  **Remark 2.7's "Only" is false and must be fixed in the same pass** — it denies exactly the
  starting-node branch the sentence above asserts. *(E13, and it is now a prose-only fix)*
- **§3.3.2** structural-triplet pruning — and the free correctness result the campaign produced:
  **the pruned form is never shorter than `w*_G`** (0 of 5,350 Suite-1 graphs; 0 of 18,461 completed
  in the exhaustive campaign), and is longer on 63.9 % of twelve-node graphs by a median of one
  symbol. **Therefore every compactness figure in §5.2 is a conservative lower bound on what the
  exhaustive form achieves.** That sentence is worth more than the number it qualifies. *(P2)*
- **§3.3.3** the completeness theorem — **restated within a fixed directedness class, with the flag
  hypothesis moved from the proof into the statement.** This is a *factual correction*: the theorem
  as printed does not mention the flag and the proof relies on it. The flag is **external metadata**;
  the witness is exact and needs no enumeration — a single undirected edge and a single directed arc
  both canonicalise to `"V"`. **Never quote a collision rate without its enumeration window.**
  Propagate to Corollary 2.13. *(R3.3b, R3.3c, B2, T-22)*

**§3.4 String distance and the induced graph distance** — Corollary, thinned.

**Cut to supplementary**: Algorithm 3's full listing → **S7**. **Cut outright**: Example 2.3's printed
draft self-correction (E8), Remarks 2.6 / 2.11.

---

### §4 Experimental design — 4.5 p — *(was §3)*

**This is the section the user's requirement (4) is aimed at: every decision states its justifying
experiment and points at the supplement that carries it.** The pattern, used six times:

> *decision, one sentence* → *why, one clause* → *(S-n)*

**§4.1 Datasets** (~1.2 p) — **Tab. 4**, 10 rows.
Columns: raw / kept / retention % / `n̄` / `ñ` / `n_max` / `m̄` / density / **discarded `n̄` and
`n_max`** / **label content** / suite. *(AE.1, AE.4b, E1, E6, F1)*
- **The Suite 1 / Suite 2 split is introduced here**, and the ceiling is attributed correctly:
  **exact GED has a 12-node ceiling; the encoder does not.** That is a constraint on the field, not
  on this work, and it is the sentence AE.1 needs. *(AE.1)*
- Density convention travels with the number: mean of per-graph `2m/(n(n−1))`, which differs by
  10–27 % from the ratio of means on this cohort. *(R1.3a)*
- **Two disclosures made first rather than caught on**: the connectivity discard is **size-biased
  across the whole cohort**, including the submitted five (Letter 1.23–1.32×, AIDS-IAM 2.27×,
  Mutagenicity 1.92×) — so `n̄ = 31.7` is the *connected subsample's* mean; and **the retained ceiling
  is 98, not 417** — the 417-node graph is disconnected and never enters. *(R3.5a)*
- **LINUX carries no node and no edge attribute**, which corrects a claim the submitted conclusion
  makes twice. *(E6)*

**§4.2 Reference graph edit distance** (~1.0 p)
- **One cost model throughout**: node ins/del = 1, edge ins/del = 1, substitutions free. The submitted
  paper inherited three heterogeneous published cost models, which is R3.5b's actual objection.
  Justification: Corollary 2.13. **(S1)**
- **Two regimes.** Exact below `n = 12` (`networkx` A*, so this is not a library compared against
  itself). A **proven bracket** above it. **Both ends selected by measurement, against 3.84 M
  certified exact values across 60 method×dataset cells and 46.8 M bound evaluations. (S2)**
- **The disclosure, in the main text, not the supplement**: the selected upper bound is the **loosest
  of the seven measured**, by 6.7×, and it wins **by elimination** under a cost gate frozen before any
  tightness result was visible. Its relative error grows **5–10× faster in `n`** than the alternatives — AIDS OLS slopes
  +0.294 against IPFP_MS's +0.029 and BP_BEAM_MS's +0.055, so "~10× than *any* alternative" is false
  against BP_BEAM_MS at 5.3×; **print the slopes** —
  and it trips the pre-registered uninformative-bracket rule on 2 of 5 datasets. A disclosed
  sensitivity arm quantifies the cost. **(S2)** *(AE.1 — this is adverse to us exactly where AE.1
  aims, and volunteering it is what makes the bracket credible)*
- **Bracket behaviour vs `n`, on both scales, because they disagree**: the **absolute** gap rises with
  `n` in **10 of 10** datasets while the relative width falls in 6 — the denominator grows.
  **Report the absolute gap as primary.** Certification runs 28.5 % → 0.03 %. **(S3)** *(AE.1)*
- **The guard that is right and the one that was wrong.** Per-read: reject non-finite and negative;
  reject zero **only where a zero distance is unattainable** — **Suite 1 holds 306,768 certified
  off-diagonal pairs at exact GED = 0**, and a blanket `> 0` assertion aborts a correct run.
  🔴 **Do not quote 28.05 % here.** That is the Suite-2 *certification* rate (`LB == UB`, GED proven
  exactly) — a different statistic on a different cohort. `CLAUDE.md` states both correctly and
  separately; this file conflated them, and the conflation reached a section scaffold before it was
  caught. Per-campaign: an accessor
  probe. Per-matrix: abort if the off-diagonal exact-zero fraction ≥ 0.99. **(S1)**

**§4.3 Representations compared** (~0.9 p)
- The five comparators and the **a priori family taxonomy** (§7), defined **without reference to any
  result** — that is the whole point of it.
- **Each representation's distance is selected by measurement, not assigned by inspection**, against
  six criteria fixed in advance, with **selection on correlation with GED explicitly forbidden**
  (F5 is measured and reported, never used to choose). A representation with no admissible distance
  enters **Tab. 2** on its properties and is excluded from the running comparison, with the reason
  printed. **(S4)** *(AE.4a, R3.6a, R1.1)*
- **`|n_i − n_j|` enters as a declared trivial baseline**, not as a representation. Say so plainly
  here; §5.4 depends on the reader already accepting it.

**§4.4 Bit accounting** (~0.5 p) — **Tab. 5** (payload bits per stored byte)
Both conventions reported for every method, as fixed before any bit count existed. The reason they
differ goes **on the page**: the frozen realised convention charges the adjacency triangle 7.50
payload bits per stored byte and the instruction string **3.17**, because IsalGraph has no published
wire format and one ASCII character per instruction is the default of writing it out as text. The
**entropy bound is primary**; the realised column is printed beside it. **(S5)** *(R3.6a)*

> **The likely objection and its answer, in one clause each.** *"You chose the convention that
> flatters you."* — three even-handed rules were evaluated and **all three agree**; both columns are
> printed so the reader adjudicates. Our own module already refuses eight-bits-per-character for the
> adjacency matrix and already flags min-DFS `inflated` for exactly the artefact IsalGraph suffered
> unflagged. **(S5)**

**§4.5 Statistical protocol** (~0.7 p)
- **Graph-level cluster bootstrap**, not pair-level — resample *graphs*, recompute over the induced
  submatrix. LINUX has **89 graphs**, not 3,916 independent observations. Describe it; R3.5c asked
  for a description and did not get one. *(R3.5c)*
- **Per-dataset correlations are primary; the pooled analysis is demoted.** *(R3.5b)*
- **A pre-registered confirmatory family, frozen before any p-value existed**, with BH-FDR at
  q = 0.05; everything else is labelled **descriptive**. **(S6)** *(AE.4c)*
- **A pair-accounting ladder, per dataset**, from raw graphs to analysed pairs, with every exclusion
  justified and counted — this is R3.5a's request answered literally. **(S6)** *(R3.5a)*
- The exclusions R3.5a names are justified here with their counts, and the self-found
  **473,147-pair gap** is disclosed as *"in addressing this we also found…"*, never as the opening
  move. *(E2/F2, and [manuscript](manuscript.md) §4.3's ordering rule)*

**§4.6 Implementation and reproducibility** (~0.2 p)
C++ engine (**neither it nor GEDLIB existed at submission** — say so), GEDLIB via `graphkit-learn`,
seed 42 throughout, 300 s per-graph encode budget enforced by a killed subprocess, build hash,
`-march=x86-64-v3`. **One sentence for the unplanned determinism check**: 14 regression fits computed
twice by independent processes in separate trees hours apart are byte-identical to the last stored
digit — not designed to succeed, which is why it is worth quoting. *(R3's "open implementation", T-21)*

---

### §5 Results — 6.1 p

**Order is load-bearing.** Unscoped and unattackable first; the negative fourth, by which point the
reader trusts the reporting. This is not burial — §5.4 is the longest subsection and §6.2 is built
on it.

**§5.1 Completeness at cohort scale** (~0.5 p) — *H1*
> **Zero encoding collisions across 24,764,422 GED-positive pairs.** Suite 1 against **exact** GED,
> so `GED > 0` certifies non-isomorphism: 3,424,764 certified pairs. Suite 2 at `LB > 0`: 21,339,658
> further pairs.

**It is a count, not an estimate — there is no interval to argue with**, which is why it opens.
One caveat clause, which costs nothing and pre-empts the obvious question: on Suite 2 certification is
`LB > 0`, so pairs the bound could not separate are outside that half; Suite 1 has no such gap.
It survives the censored arm — the 101 fallback graphs are outside the theorem and collide with
nothing either.

**§5.2 Compactness** (~1.5 p) — *H2* — **Fig. 1** (information content); the head-to-head table is
**S8**, so the four numbers below live in prose and **all four must appear**
1. The **scoped** headline: *"Among representations whose canonical form is intrinsic to the code,
   IsalGraph is the most compact above `n ≈ 20` — shorter than gSpan min-DFS on **112 of 112**
   strata, median **+215 bits**, no losses and no ties."* min-DFS is itself a canonical code, so this
   is like-for-like, and `competitors.md` calls it *"the single most important comparator"*.
2. **The scaling result, its own paragraph**: the advantage **grows** with size — 20.4 % of strata at
   `n ≤ 5` to 45.6 % above 40, median gap −1.2 → +242.1 bits. Where a length claim usually degrades
   with scale, this one improves.
3. **The concession, in this subsection and not in §6**: canonically-labelled **edge-list**
   serialisations beat it at scale. Say *edge-list* — it names the mechanism, and these cohorts are
   sparse, so an edge list exploits a property of the *data*. That is a difference in design point.
4. **The exhaustive arm, declared** *(P2)*. 🔴 **Do not quote a band percentage here — there are two
   and they answer different questions.** **8.8–12.2 %** is the *ceiling probe*, 25 graphs per node
   count at a 60 s budget, all graphs (`T-06-POSITIONING.md` §5). **12.5–17.0 %** is the *production
   campaign split by completion*, i.e. among graphs whose exhaustive search finished
   (`T-06-EXHAUSTIVE-HANDOFF.md` §0). Both are correct; neither source names its population.
   **Report the stratum medians instead** — they need no band statistic: **114.1 bits against
   nauty-sparse6's 144.0 at `n = 20`**, a 20.8 % win. At `n = 40` it is 342.4 against 336.0 — it
   closes 6.3 of a 12.7-bit gap and **does not overtake**. The reason is the clock, not the encoding:
   the budget expires on 96.8 % of that stratum. **Say both numbers.**
5. Pruned is primary and is **never shorter** than `w*_G`, so **the measured advantage is a lower
   bound**.

> 🔴 **The scope goes in the sentence, every time.** Pooled over all sizes the significant results run
> **10 against / 9 for**. *"IsalGraph produces shorter encodings"* unqualified is not a summary of
> this subsection, it is the opposite of one.

**§5.3 Encoding cost** (~1.4 p) — *H4* — **Fig. 3** (the cost law)

**This subsection got a great deal stronger on 2026-08-26** and its shape changed: T-13 turned an
observational correlation into a **controlled experiment**, which is a different class of evidence.
Six moves.

1. **The R1.1 fix, structural, and it comes first**: per-graph encoding cost and per-pair GED cost
   **no longer share an axis**. R1's "unfair" clause is a defect report and its fix is protected
   regardless of what else is cut. Competitor encode costs are reported per graph, against ours.
   *(R1.1)*
2. **The controlled result — the headline.** *"Across 12 ladder cells that hold `n`, `m` and the
   entire degree sequence fixed while |Aut(G)| varies by up to 71 orders of magnitude,
   ρ(log|Aut|, log t) is positive in 11, median +0.892, sign test p = 0.0064."* **State that it is a
   controlled experiment**, because on the real cohort |Aut| is the *weakest-looking* predictor
   (marginal ρ = +0.189 against +0.326 for `log n`) — `n`, `m`, density and |Aut| all co-vary there,
   and the ladders break the confound by construction. *(H4, R3.7d)*
3. **The null arm is what licenses reading the slope.** The five search-free serialisations are
   **flat across the entire |Aut| range — fold-change 1.0–1.1×**. Without it a rising curve could be
   any artefact of the ladder construction. One sentence, and it is the sentence that makes the
   figure evidence rather than decoration.
4. **The law is a property of the design point, not a defect of ours.** gSpan min-DFS obeys it
   (ρ = +0.686, p = 0.041) and so does AGM CAM. **Say this plainly**: |Aut|-driven blow-up is what
   happens when a canonical form is computed by searching an invariant-pruned tree.
5. **The concession, and it doubles as the future-work case.** *"The one family that escapes the law
   is the one that implements automorphism detection."* nauty-graph6 and nauty-sparse6 correlate
   **negatively** (median ρ ≈ −0.61, negative in 18 of 20 cells) and complete **94.7 %** of the grid
   against our 55.3 %. Concede it in one sentence, unhedged — the measurement and Corollary 3 agree,
   and that agreement is what makes the future-work statement a conclusion from evidence rather than
   an apology. *(This is the same move §5.2 makes with edge-list serialisations.)*
6. **The head-to-head, both halves in one paragraph.** Against min-DFS on the 66 graphs where both
   complete, IsalGraph's pruned form is a **median 3.39× faster and faster on 42 of 66** — and
   **min-DFS completes more of the grid, 83 of 132 against 73**. The first alone is selection; the
   second alone understates a real speed advantage.

**Then two supporting facts**, one line each: the observational cohort measurement (censoring **0 %
below |Aut| = 10⁴, 21.85 % at 10⁴–10⁸, 100 % above 10⁸**) as the *field* counterpart to the
controlled ladder result — report it **by stratum**, never as the diluted 2.50 % dataset figure; and
that the bottleneck has moved (exact GED grows ≈ 5× per added node near `n = 12`, encoding ≈ 1.15×).

> 🔴 **Three red lines specific to this subsection.**
> — *"cost is governed by |Aut|, not by size"* is **wrong**. Size sets the **frame count** and it
> matters: the dense ladders censor everywhere for both canonical arms. **|Aut| governs the
> branching, at fixed size.** Say which.
> — **No |Aut| claim for the exhaustive arm**: p = 0.18, 57.6 % censored. Unresolved, and reported so.
> — *"IsalGraph is the cheapest canonical form"* is **false on both axes against nauty** (median
> ratio 0.55×, i.e. nauty ~1.8× faster, and it completes 94.7 % to our 55.3 %). Quoting **127×** as
> *the* cost-law figure is also out — that is one cell; the per-ladder range is 1.1× to 46,170×.
> Quote the ρ and the sign test, or a fold **with its cell named**.

**One retraction to carry** *(self-found, disclosed under [manuscript](manuscript.md) §4.3's ordering
rule)*: the plan's claim that the triplet pruning key is *"provably coarser than 1-WL, 2.4–2.6×"* is
**refuted in both halves** — the partitions are incomparable in general, and the cohort ratio has
median 1.0952. What replaces it is stronger and belongs in §7's future work: **1-WL attains the orbit
partition on 99.939 % of the 16,370 cohort graphs; the incumbent key on 41.869 %** — so there is real
headroom below the invariance ceiling, and the ceiling itself is Proposition 1.

**§5.4 Distance fidelity** (~2.6 p) — *H3* — **Fig. 2** (within-`n` ρ) + two inline blocks: the
five Suite-1 per-dataset rows against exact GED **(R3.5b's primary evidence — it stays in the main
text because a reviewer asked for it in as many words)** and the three-row Letter control

The longest subsection, and it must be read in this order or it reads as excuse-making.

1. **The result, conceded first and without a bracket argument.** Against **exact** GED — no bound,
   no interpolation — the trivial `|n_i − n_j|` baseline beats the representation on **4 of 5**
   Suite-1 datasets, worst −0.4597 on AIDS. **No framing repairs this and none is attempted.**
   Over all 25 records: below its own size null on **17**, every one of them significantly, one
   undetermined, seven favouring the string.

   > **Concede it with the competitor columns beside it, because they change what the concession
   > means.** The baseline is not beating *us*; it is beating the field. Under **exact** GED no
   > representation clears it on more than **two of five** datasets — IsalGraph 1/5, min-DFS 2/5,
   > AGM CAM 2/5, nauty-graph6 0/5, nauty-sparse6 0/5 — and under the **proven lower bound** none
   > clears it on **any of nine**, for all five. Under the upper bound most clear on most. So the
   > sentence to write is not *"the canonical string fails against graph edit distance"* but
   > *"correlation with graph edit distance on these benchmarks does not separate the
   > representations, because the reference does not measure what the comparison needs it to."*
   > That is C16, it is the honest reading, and it is the one that makes moves 3–4 land instead of
   > sounding like recovery. **It exonerates no one, ourselves included** — which is exactly why a
   > reviewer will accept it. *(T-28)*
2. **Then the bracketed half, which is *undetermined* rather than failed.** On the same pairs the
   verdict inverts across the proven bracket on **7 of 10** datasets — below under LB, clearing under
   UB. **Both bounds printed, always.** The bracket being wide enough to flip a verdict on 21.7 M
   pairs *is* a finding, and it is the fourth independent detection of the same fact.
3. **Why — and this is the pivot from result to contribution.** On these benchmarks the reference
   itself is size-dominated: node-count difference alone attains **ρ = 0.71–0.997** against
   ground-truth GED, exceeding 0.96 on seven of ten Suite-2 datasets and reaching **0.9971** on
   COIL-DEL. This is a property of the **data**, not of the approximation — exact GED is itself
   ≈ 0.92 size-dominated on IAM Letter. *(AE.1, R3.5b)*
4. 🔴 **The diagnosis in 3 makes a prediction, and the prediction holds. This is the pivot of the
   subsection and the strongest object in it.** If the negative result is substantially a property of
   the *reference*, then a reference that is not size-dominated should let the same distances show
   structure. That is a falsifiable consequence, not a reframing, and it is tested by changing
   exactly one thing: **every representation distance is held at its cached value and only `d_ref` is
   replaced** — by a Weisfeiler–Lehman subtree kernel distance (`h = 2`, unnormalised), which is
   exact at every size and therefore carries no bracket and no ceiling.

   **The table is the argument: the same five datasets, the same pairs, the same strings, one column
   changed.** 2,000 graph-level bootstrap replicates; the `exact` half reproduces the §5.4 table
   above it to four decimals, which is what licenses reading the two halves as one experiment.

   | dataset | pairs | ρ vs exact GED | its size null | excess [95 % CI] | ρ vs WL | its size null | excess [95 % CI] |
   |---|---:|---:|---:|---|---:|---:|---|
   | `aids` | 131,148 | 0.3266 | 0.7863 | −0.4597 [−0.4983, −0.4210] | 0.3393 | 0.2272 | **+0.1121 [+0.0593, +0.1668]** |
   | `iam_letter_high` | 2,118,711 | 0.6660 | 0.9195 | −0.2536 [−0.2691, −0.2387] | 0.5959 | 0.4283 | **+0.1676 [+0.1434, +0.1911]** |
   | `iam_letter_low` | 695,610 | 0.9278 | 0.9139 | +0.0139 [+0.0057, +0.0235] | 0.7128 | 0.5696 | **+0.1432 [+0.1209, +0.1664]** |
   | `iam_letter_med` | 784,378 | 0.8833 | 0.9146 | −0.0313 [−0.0438, −0.0190] | 0.7109 | 0.5160 | **+0.1950 [+0.1715, +0.2192]** |
   | `linux` | 1,685 | 0.4850 | 0.7097 | −0.2247 [−0.3492, −0.0922] | 0.4798 | 0.1609 | **+0.3189 [+0.1699, +0.4454]** |
   | **clears its own size null** | | | | **1 of 5** | | | **5 of 5** |

   ⚠ **This move is carried by the table and by nothing else.** The size null is
   `ρ(|n_i − n_j|, d_ref)`, which is identically zero inside a stratum `n_i = n_j` — that is
   precisely why move 5's within-`n` view is the size-controlled one. So **Fig. 2 cannot show this
   result**, in either panel, and no rebuild of its profile adds the arm. Do not write a sentence
   that sends the reader to the figure for it.

   Note what the ρ column does **not** do: it barely moves, and on three datasets it *falls*. The
   representation did not get better. **The baseline it is being measured against collapsed** — the
   reference's own size null drops from a median of 0.914 to 0.516 — which is precisely what 3
   predicts and is the sentence to write. Over all fourteen cells measured the arm clears its null on
   **6 of 14 against the best available GED reference and 12 of 14 against WL**. *(T-28)*

5. **The correct instrument, which follows from 3 rather than being reached for, and now applies to
   both references.** Inside a stratum `n_i = n_j` the size null is identically zero, so raw ρ is the
   structural signal with the size channel removed **by construction**. **Report both bands in one
   table**: at `n ≤ 20` against exact GED the instruction string correlates significantly better than
   nauty-graph6 and nauty-sparse6 (p = 0.041, 0.041) and worse than min-DFS and AGM CAM; **above
   `n = 20` it is at best indistinguishable and under the upper bound significantly worse than all
   four.** Using an instrument the paper argues for is not cherry-picking; using it only where it
   helps is — so the WL reference is reported through the same instrument in **Fig. 2(a)**, where the
   nauty result holds and the AGM CAM result does not.

   > 🔴 **The WL comparison in this list was `p = 0.012` and is withdrawn.** The published figure and
   > its sign-test row were computed through `size_profile.py::_wl_counts`, which read the stored WL
   > encoding `h<level>:<colour>:<count>` as whole symbols. A symbol occurs once per sequence, so it
   > built a **presence indicator rather than a count vector** — 208 tokens with a largest cell of
   > 1.0, where the frozen encoding campaign fitted 69 colours with counts to 12. Recomputed against
   > the cached matrices the comparison **inverts**: 1 stratum higher, 18 lower, median Δρ −0.1116,
   > p = 7.6e-05. **Only the `wl_subtree` arm is affected** — every `levenshtein` arm is byte-identical
   > across 1,818 joined rows. Fixed at `6b89b4f`; `fig1_rho_vs_size.pdf` must be regenerated
   > **whether or not the WL reference goes in the paper**.

6. **The head-to-head under the structural reference, with both of its scopes in the same sentence.**
   Paired graph-level bootstrap over 14 of 15 dataset cells: the canonical string beats **both**
   canonically-labelled nauty serialisations on **every cell** — `sparse6_nauty` 14 W / 0 T / 0 L,
   `nauty_graph6` 12 W / 0 T / 2 L pooled; 10/3/1 and 11/2/1 within-`n`. **AGM CAM only pooled**
   (8/4/2, and 2/5/7 within-`n`, so most of that margin is size agreement — say so). **min-DFS beats
   it under every reference tested**, GED and structural alike, and that concession goes in the same
   paragraph rather than a later one.

   > 🔴 **And the nauty result is bounded by size, exactly as H2's is.** The per-stratum sign test
   > splits at `n = 20`: **at `n ≤ 20` IsalGraph is significantly higher than both** (58 strata
   > against 31, p = 0.0055 each); **above `n = 20` it is not** — a tie against `nauty_graph6`
   > (53/57, p = 0.78) and a **significant loss to `sparse6_nauty`** (40/70, p = 0.0054). Write
   > `n ≤ 20` into the sentence that makes the claim.
   >
   > **⚠ Two boundaries are in play and they are not the same one.** The sign test above splits at
   > `n = 20`, the paper's existing scope constant. **Fig. 2(a) breaks at `n = 12`**, the exact-GED
   > ceiling. Measured at the figure's own break, and this is what a caption may say:
   >
   > | | exact | lb | ub | wl |
   > |---|---|---|---|---|
   > | `nauty_graph6`, `n ≤ 12` | **higher** 15/5 | **higher** 41/11 | — 24/28 | **higher** 41/17 |
   > | `nauty_graph6`, `n > 12` | *(none)* | — 76/65 | — 59/82 | — 70/71 |
   > | `sparse6_nauty`, `n ≤ 12` | **higher** 15/5 | **higher** 37/15 | — 24/28 | **higher** 46/12 |
   > | `sparse6_nauty`, `n > 12` | *(none)* | **lower** 51/90 | **lower** 38/103 | **lower** 52/89 |
   >
   > *(higher/lower = sign test rejects at 0.05; — = does not resolve; exact GED has no strata above
   > the ceiling by construction.)*
   >
   > 🔴 **"Leads both below the ceiling and trails both above it, under every reference" is false on
   > both halves.** Below it, the **upper bound does not resolve** for either arm. Above it,
   > `sparse6_nauty` trails under all three but **`nauty_graph6` resolves under none of them**. What
   > *is* supported, and is the useful sentence: **within each band the references agree with each
   > other, and the direction changes with `n`** — so the split is a property of size, not of the
   > yardstick. Say that, and name the arm when claiming a trail above the ceiling.
   >
   > Note also that the `n ≤ 20` claim in C15 is carried by its `n ≤ 12` part: in the `13 ≤ n ≤ 20`
   > band nothing resolves for `nauty_graph6` under any reference, and for `sparse6_nauty` only the
   > upper bound does, *against* us.

   > **The two results do not conflict and the reason should be stated once.** The dataset sweep is
   > weighted by graphs and most graphs sit at small `n`, so it is carried by the same band the sign
   > test identifies; the per-stratum test weights every stratum equally and the high-`n` strata are
   > thin. This is also move 9 arriving early — the advantage dissolves at scale because *everything*
   > does. *(T-28)*
7. **The model that controls rather than stratifies**: `GED ~ β₁·Lev + β₂·|Δn| + β₃·|Δdensity|`,
   standardised. **β₁ is significant and positive on 19 of 19 identifiable fits — and the size
   coefficient exceeds it on 17 of 19.** Both halves, one sentence. Print a **VIF column**; four fits
   on two datasets are excluded as unidentifiable and two more because the point estimate falls
   outside its own bootstrap interval under tier-3 subsampling. Say why. **(S6)** *(E10)*
8. **The control that makes the limitation a *condition* rather than a vague degradation** —
   three inline rows. IAM Letter LOW/MED/HIGH is the same generator at three distortion levels:
   node count barely moves (4.07 → 4.58) while **mean edge count rises 49 %**. The size baseline stays
   **flat at ρ ≈ 0.92** while the string falls from **0.93 to 0.67**; β_lev halves, β_size doubles,
   and both instruments cross at MED. **The representation stops paying its way between LOW and MED
   distortion** — a named condition, not a coincidental size threshold. *(R1.3a, R1.3c)*
9. **The field-level statement**, which is the contribution: **above `n ≈ 40` not one of the
   representations tested — IsalGraph or any competitor — is reliably distinguishable from ρ = 0.**
   Measured on 21.7 M pairs. And it is **not** a compute artefact: removing every censored-touching
   pair *lowers* ρ at both bounds and both size restrictions. Report all three quantities, never the
   delta alone.
10. **R1.3b answered, and it leads because it is free**: the degradation on AIDS cannot come from
   discarded labels, because **both sides of the correlation are topology-only** — the reference GED
   is computed under a topology-only cost model on the same stripped graphs. Density is measured and
   is not sufficient either. The real decomposition is size versus structure. *(R1.3a, R1.3b)*


11. **What §5.4 can claim for the representation without winning the correlation.** Moves 1–10 settle
    that graph edit distance does not separate the representations on these cohorts (C16). That
    leaves the subsection needing a positive statement that does not depend on a correlation
    ranking, and there are three, in descending order of how well they are established.

    a. **Family, not method (C17).** Under the WL reference all three canonical codes clear their own
       node-count baseline on 12 of 14 cells and both edge-list serialisations clear it on 1 and 0.
       Canonical codes carry structure here; serialisations carry size. **Verified.** 🔴 IsalGraph is
       *not* distinctive within its family — min-DFS clears the same twelve with a marginally larger
       mean excess — so this is a claim about a class and must read as one.

    b. **Compactness against the same competitor, cross-referenced not restated.** §5.2 already
       establishes the canonical string is shorter than the gSpan minimum DFS code on **112 of 112**
       strata above `n ≈ 20`, median **+215 bits**, no losses and no ties. That is a clean sweep over
       the one competitor §5.4 concedes, and one sentence of cross-reference is the right weight —
       **restating it here duplicates H2 and drops its two mandatory scopes** (*above* `n ≈ 20`,
       *among canonical codes*). "IsalGraph produces shorter encodings" unqualified is already a red
       line. **Verified, and it is C2.**

    c. ✅ **The edit path stays in graph space. VERIFIED, matched protocol — and it must be
       stated on the right notion or it reverses.**

       §5.4's instrument is a Levenshtein distance between codes. That is a *graph* distance only if
       the codes along the path between two graphs are themselves graphs. Measured on **23,916 pairs
       × 5 alignments**, all five Suite-1 cohorts, both representations on the **same pairs** under
       the **same metric** — `levenshtein` over `Encoding.symbols`, one symbol being one atomic
       operation, which is what T-04a selected for both and what §5.4 already uses:

       | representation | intermediates | **valid** | 95 % CI | whole paths valid |
       |---|---:|---:|---|---:|
       | **IsalGraph** | 532,315 | **92.0 %** | [91.9, 92.2] | **80.5 %** |
       | **gSpan min-DFS** | 246,220 | **52.3 %** | [52.0, 52.5] | 38.5 % |

       Cluster bootstrap over pairs, 2,000 resamples. **Validity** = well-formed in the family's own
       language **and** denoting a simple, undirected, connected graph with `n ≥ 2` — the cohort's
       own filter. For min-DFS, well-formedness is membership in the DFS-code language of Yan & Han
       (*gSpan*, ICDM 2002, doi:`10.1109/ICDM.2002.1184038`, Defs. 4–6); for IsalGraph it is
       membership in `Σ*`, which is free because `S2G` is **total** — verified exhaustively over all
       597,870 strings of length 1–6. **That closure gap is the property**: an edited instruction
       string is still a program; an edited DFS code is generally not a code.

       > 🔴 **SAY WHICH NOTION, IN THE FIRST CLAUSE. On the canonical-code notion we LOSE, in all
       > five cohorts — IsalGraph 14.1 % against min-DFS 35.5 %.** A reader who takes "stays in graph
       > space" to mean "stays in *canonical*-code space" finds the opposite of the claim. The two
       > are in genuine tension and the mechanism is ours: `S2G`'s totality gives every graph a large
       > `Σ*` preimage, which is exactly what makes well-formedness easy and canonicality rare.
       > **The claim is about the language, never about the canonical form.**

       > **Three scopes that travel with the number.**
       >
       > **Cohort.** IsalGraph ranges **81.0 % – 100 %** across the five, tracking `C`/`c` frequency;
       > min-DFS is flat at 50.5 – 57.2 %. Never quote 92.0 % bare.
       >
       > **Every IsalGraph rejection is a self-loop** — 42,409 of 42,409, from a `C`/`c` executed
       > while both pointers sit on one node. No parse failure and no disconnection, ever. So the
       > **7.97 %** is the self-loop rate, and printing 92.0 % while withholding it is not coherent:
       > print both or neither.
       >
       > **Edit-unit asymmetry, and state it before a reviewer does.** min-DFS's alphabet is the
       > `O(n²)` index pairs; IsalGraph's is 9 operations at any `n`. One min-DFS edit is therefore a
       > larger semantic step. This is not a choice made for this experiment — it is the frozen
       > T-04a convention for both arms, and forcing character-level would instead charge min-DFS
       > about four edits per tuple. Name the mechanism (position-independent against
       > position-dependent alphabet) rather than let it be found.

       **Two numbers that must never appear apart.** Under each backend's own shipped decoder, which
       ignores the grammar, min-DFS reaches **89.5 %** — and `IsalGraphBackend.decode` copies edges
       under `u < v`, dropping exactly the self-loops that cause all our rejections, so IsalGraph
       reaches **100 %**. Grant both or neither; the ordering holds either way (92.0/52.3 or
       100.0/89.5). **Printing the min-DFS permissive row without ours beside it would be a
       misrepresentation.**

       **The alignment is not unique and the honest choice cost us.** Sampling uniformly over optimal
       paths gives min-DFS 52.3 %; the deterministic `rapidfuzz.editops` path gives **39.0 %**, 13
       points lower. **Using the default would have flattered us by 13 points**, so the sampled
       figure is the one to report and the fact that it is the less favourable one is worth a clause.

       Path lengths differ (5.4 against 3.0 edits per pair), which runs **against** us — more
       intermediates, more chances to fail — and the whole-path column agrees regardless.
       Suite 1 only (`n ≤ 12`); untested at scale. *(T-28)*

**§5.5 The pre-registered confirmatory family** (~0.8 p)
Reported **exactly as it came out**, including the negative results. `N_actual = 79`, 79 cells
carrying a p-value, **75 rejected at q = 0.05**.

> 🔴 **The count must never travel bare.** A rejection is against `H₀: Δ = 0` and can mean
> *significantly worse*. **Split by row and direction: 35 of the 69 directional rejections are for
> IsalGraph and 34 against.** Printed as a four-row table, it costs three lines and it is the single
> most credibility-bearing object in §5.

A pre-registered analysis reported unchanged is the strongest evidence of good faith the paper has,
and it is what buys the reader's acceptance of everything above it. **Do not bury it and do not
soften it.**

---

### §6 Discussion and limitations — 1.8 p — **NEW**

**§6.1 The trade-off surface** (~0.7 p)
> *"No single representation leads on both axes, and the two that lead each axis are undefined on the
> other: the most compact serialisation admits no distance satisfying the metric axioms, and the
> best-correlating representation admits no bit count. Among those measurable on both, IsalGraph is
> decisively more compact than min-DFS and nauty-graph6, and its correlation against them is
> bracket-dependent — indistinguishable under the lower bound, weaker under the upper. It is
> dominated on both axes by nauty-sparse6."*

**Say the last clause.** A trade-off framing that omits the one representation dominating us is the
most checkable dishonesty available in this paper, and conceding it is exactly what makes the min-DFS
result read as a finding rather than a selection.

**Then the categorical differentiator, and this is the paragraph that reframes the section**: the
instruction string is an **executable program** — every prefix is a valid construction sequence
producing a subgraph — and no serialisation in the comparison is. It is not adjudicated by ρ or by
bit counts, which is the point. Alphabet size is fixed at |Σ| = 9 independent of `n`; a sparse6 index
width is `⌈log₂ n⌉`.

**§6.2 On evaluating graph-distance surrogates** (~0.5 p)
The methodological contribution, stated as a recommendation to the field rather than as a defence:
report the `|n_i − n_j|` baseline alongside any GED-correlation claim, and decompose within fixed `n`.
Both are cheap; neither is standard; and on these benchmarks the difference between doing it and not
is the difference between ρ = 0.93 and ρ = 0.26. **This is the paper's most transferable result and
it costs half a page.**

**§6.3 Limitations** (~0.6 p)
R3.7a's three, each with its cause, plus ours:
1. **The `n` ceiling of the *evaluation*, with its cause** — exact GED, not the encoder. The encoder
   reaches 98; `networkx` A* reaches 12.
2. **The worst case**, characterised by |Aut(G)| rather than called exponential.
3. **No sequential model and no downstream pattern-recognition task is evaluated.** *(R3.7a item 3 —
   under a requirement modal; the R3.2 decline does not absorb it.)*
   **§6.3 now also carries the R3.2 pre-emption alone**: the sequence-model row was dropped from
   Tab. 3 by PI decision 2026-08-26, so nothing T-07 produced discharges it any more. Write it from
   T-07's measured facts, never from R3's paraphrase (`T-07-article-notes.md` §4):
   **neither predecessor ran a downstream graph-learning evaluation on a real benchmark.** [28]'s
   encoder-only Transformer is 3-way classification on a **synthetic, purpose-built** 3,000-sample
   set of ~12-node graphs, against **one** non-graph baseline (row-major binary flattening), with
   **no numeric result anywhere in its text** — eight figures, no table, no seed, no significance
   test. [29]'s is **LSTM *and* GRU** on **masked / random-position token prediction** over 10,000
   ZINC molecules against SMILES, SELFIES and InChI — token prediction, not graph learning.
   **Replicating either would not answer R3.2 as posed.**
   State it neutrally and in one short paragraph. **Both of R3's claims are true in kind — R3's
   accuracy record is intact and the letter must treat it that way.** No defence, no promise.
4. **Labels are discarded**, and the connection to §5.4's AIDS interpretation is made explicitly —
   this is the *connection* R1.3c asks for and the submitted paper does not make. *(R1.3c)*
5. The connectivity discard is **size-biased**, so cohort means are the connected subsample's.
6. The selected upper bound is the loosest measured and its error grows fastest in `n`.
7. Real-world machine-learning graphs are routinely far larger than 98 nodes.

---

### §7 Conclusions and future work — 1.2 p

Every number re-derived; **`n^{9.0}` deleted; "super-polynomial" deleted; "labels present in all five
datasets" corrected** (E6). No number appears here that does not appear in §5.

**Future work, two items, both concrete and both evidenced** — not gestured at:

1. **Labels** *(R1.3d)*: the `Σ × L` extension — a product alphabet carrying node and edge labels,
   with [29] as the precedent. R1 asks about this directly, and the current conclusion already names
   it twice without making it concrete.
2. **Automorphism-aware canonicalisation** *(from §5.3, and it is a conclusion rather than a
   concession)*: Corollary 3 says no node invariant can resolve below the orbit partition, and the
   one representation family that escapes the cost law is the one implementing individualisation–
   refinement. **There is also measured headroom below the ceiling**: 1-WL attains the orbit
   partition on 99.939 % of the cohort while the incumbent triplet key attains it on 41.869 % — and
   the shortfall concentrates exactly where canonicalisation is expensive (Mutagenicity 14.5 %,
   COIL-DEL 10.3 %, Protein 10.5 % against 100 % on Letter LOW). So the statement is precise: a
   finer invariant buys real headroom up to the ceiling, and only automorphism detection goes past
   it. **Say both halves** — the second alone reads as an excuse, the first alone overstates.

---

## 5. The claim register

**Frozen wordings.** Use these verbatim; they were measured, argued and in several cases retracted
before reaching this form. Full derivations: `../tasks/T-06-FRAMING.md`, `../tasks/T-06-POSITIONING.md`.

| # | Frozen sentence |
|---|---|
| C1 | *"Zero encoding collisions across 24,764,422 GED-positive pairs."* |
| C2 | *"Among representations whose canonical form is intrinsic to the code, IsalGraph is the most compact above `n ≈ 20` — shorter than gSpan min-DFS on 112 of 112 strata, median +215 bits, no losses and no ties. Canonically-labelled edge-list serialisations are more compact at scale, and we report that."* |
| C3 | 🔴 **AMENDED 2026-08-26 — the 8 clears must carry their reference, not only their significance.** *"On 17 of 25 records, node-count difference alone predicts graph edit distance significantly better than the canonical string. Of the 8 records that clear the baseline, **one clears against exact graph edit distance and seven clear only under the upper bound of the bracket, falling below the lower bound on the same pairs**; one of those eight is undetermined."* Both decompositions are true and only the second is load-bearing: `exact` 4 below / 1 clears, `lb` 10 / 0, `ub` 3 / 7. Reporting "7 favour the string" without the bracket scope reads as a clean win for results that **invert under the other bound** | |
| C4 | 🔴 **CORRECTED 2026-08-26 — the old wording named two datasets and the source has four.** *"On Suite 1, where ground-truth GED is exact, the size baseline outperforms the representation on **four of the five** datasets — `aids` (−0.4597), `iam_letter_high` (−0.2536), `linux` (−0.2247) and `iam_letter_med` (−0.0313); only `iam_letter_low` clears, by +0.0139. On Suite 2 the comparison is undetermined: the verdict inverts across the proven bracket on every dataset measured."* The superseded text was frozen when two Suite-1 cells had landed and never re-checked when five did — **an understatement of our own negative result**, and the one error class that cannot be allowed to ship | |
| C5 | *"On these benchmarks the reference itself is size-dominated: node-count difference alone attains ρ = 0.71–0.997 against ground-truth GED, exceeding 0.96 on seven of ten Suite-2 datasets. Correlation with GED on this data therefore measures size agreement more than structural fidelity — for every representation, ours included."* |
| C6 | *"Above `n ≈ 40`, not one of the representations tested is reliably distinguishable from ρ = 0."* |
| C7 | *"Censoring at the 300 s budget is 0 % for all 3,703 graphs with `\|Aut\| ≤ 10⁴`, 21.85 % at 10⁴–10⁸, and 100 % above 10⁸."* *(observational, one dataset — the field counterpart to C11)* |
| **C11** | *"Across 12 ladder cells holding `n`, `m` and the entire degree sequence fixed while `\|Aut(G)\|` varies by up to 71 orders of magnitude, ρ(log\|Aut\|, log t) is positive in 11, median +0.892, sign test p = 0.0064; the five search-free representations are flat over the same range (fold-change 1.0–1.1×)."* *(controlled — T-13)* |
| **C12** | *"Individualisation–refinement with automorphism detection, as implemented in nauty, bliss and Traces, removes the dependence we characterise; our measurements show its canonical form does not degrade with `\|Aut\|`. Re-implementing it is a project rather than a revision, and we state it as future work."* |
| **C13** | *"`P(M)` is recomputed at every frame, `Θ(M² log M)`. Over 1,109,460 frames the pair lists take only 12 distinct values of `M`, which is why memoising them is worth 25.5×–108.6× to the compiled engine."* |
| C8 | *"The canonical string contributes significant incremental information about graph edit distance beyond node-count and density difference — significant on 19 of 19 identifiable fits — but node-count difference carries more weight on 17 of 19."* |
| C9 | *"Holding the generator fixed and adding structural distortion, the trivial size baseline stays flat at ρ ≈ 0.92 while the canonical string's correlation falls from 0.93 to 0.67."* |
| C10 | *"No single representation leads on both axes, and the two that lead each axis are undefined on the other."* |
| **C16** | *"Under exact graph edit distance, no representation tested clears the node-count baseline on more than two of the five Suite-1 datasets, and under the proven lower bound none clears it on any of the nine Suite-2 datasets. Under the upper bound most clear on most. Which representation appears to track graph edit distance is therefore a property of which bound is read, not of the representation."* Measured with paired graph-level intervals: `exact` — IsalGraph 1/5, min-DFS 2/5, AGM CAM 2/5, nauty-graph6 0/5, nauty-sparse6 0/5; `lb` — **0/9 for all five**; `ub` — 5/9, 6/9, 9/9, 6/9, 5/9. **This is the opening move of §5.4** and it is stronger than "the representation fails", because it is a statement about the benchmark that happens to exonerate no one. It is C3 and C5 sharpened by the competitor columns, not a new claim *(T-28)* |
| **C17** | *"Under the Weisfeiler–Lehman reference the split is between families rather than between methods: all three canonical codes clear their own node-count baseline on twelve of fourteen dataset cells — IsalGraph, the minimum DFS code and AGM CAM alike — while the two canonically-labelled edge-list serialisations clear it on one and zero. Canonical codes carry structure on these cohorts; serialisations carry size."* 🔴 **IsalGraph is NOT distinctive here and the sentence must not imply it is** — min-DFS clears the same twelve cells with a slightly *larger* mean excess (+0.148 against +0.125). The finding is real, it is favourable, and it is about a **class** *(T-28)* |
| **C14** | *"Holding every representation distance fixed and replacing only the reference, the canonical string's correlation exceeds its own node-count baseline on five of the five Suite-1 datasets against a Weisfeiler–Lehman kernel, where it does so on one of five against exact graph edit distance. Its correlation does not rise — on three of the five it falls — and the reference's own size null drops from a median of 0.914 to 0.516. What changes is the baseline, not the representation."* **The last sentence is not optional.** Without it the paragraph reads as a rescue; with it, it is C5 confirmed by a second route and the honest reading of both. *(T-28)* |
| **C15** | 🔴 **CORRECTED 2026-08-30 — the frozen wording said "in both size bands" and that is false.** *"Under that reference the canonical string outranks both canonically-labelled nauty serialisations on every one of the fourteen dataset cells measured, and within equal node counts **at `n ≤ 20`** (58 strata higher against 31 lower, sign test p = 0.0055 against each). **Above `n = 20` the advantage does not hold**: it ties nauty-graph6 (53 against 57, p = 0.78) and is outranked by nauty-sparse6 (40 against 70, p = 0.0054). It is outranked by the gSpan minimum DFS code under every reference we tested."* **The `n ≤ 20` scope is load-bearing and belongs in the same sentence** — this is H2's scoping pattern mirrored, and the same reviewer checks both. The per-dataset verdict and the per-stratum test are **different estimands and do not conflict**: per-dataset aggregation weights by graphs, and most graphs sit at small `n`, so a 14-of-14 dataset sweep is carried by exactly the band where the sign test also finds the advantage |

> ## 🔴 Do not print a cardinal beside a list you will maintain
>
> A list that says what it contains and then contains it cannot drift. A list that says **six**
> acquires a second thing to maintain every time an item is added or struck — and the lists in this
> project are exactly the ones that change: the artifact inventory, the build risks, the self-found
> defects as the ledger is swept.
>
> **This heading and §10.5's have each been wrong once today**, and the response letter's Part 5 is
> safe only because it states no count at all. Where the count is genuinely useful to a reader,
> derive it or accept that it is a second fact to check; where it is decoration, drop it.

> ## 🔴 A second confirmation must come from a second COMPUTATION
>
> The mirror of the ratio rule, and the failure mode the *favour IsalGraph* directive creates. Every
> number true, every sentence defensible, and the whole thing still wrong — because one result is
> reported twice and presented as two.
>
> **Instance, 2026-08-26.** `ged_positive == lev_positive` in all fifteen ladder rows looked like the
> zero-collision result confirmed on a second denominator. It is not: `collisions == ged_positive −
> lev_positive` in 15 of 15 rows, so the equality **is** `collisions == 0` — one computation, two
> columns. Summing the column gives 3,424,764 / 21,339,658 / **24,764,422**, exact to the pair against
> what §5.1 already prints, so it is the *same cohort by a separate accounting path*.
>
> **The test, and it is arithmetic rather than judgement**: *if it reproduces a published figure
> exactly, it shares a pipeline with it.* A second computation on a second sample does not land
> to the pair. **Exactness is the tell, not the evidence.**
>
> **The rule**: same pipeline → the word is **reproduces**, and the strength claimed is **exactness**.
> Only a genuinely separate computation earns **corroborates**, and only that earns a claim of
> independence. Both agents caught this within an hour of the directive; I did not.

> ## 🔴 A ratio is not a number. It is a number **plus a denominator**, and the denominator ships.
>
> Three instances in one day, each one two *true* figures describing different comparisons, with this
> file carrying one and naming neither:
>
> | | the two figures | the difference |
> |---|---|---|
> | BIPARTITE's looseness | **6.7×** / **13.0×** | against the next-loosest bound, or against the tightest |
> | T-05's bracket slopes | **6 of 10** / **4 of 10** | datasets disagreeing in sign, or datasets carrying an unconfounded slope |
> | Letter LOW zero-distance | **28.05 %** / **306,768** | Suite-2 certification (`LB == UB`), or Suite-1 pairs at exact GED 0 |
| exhaustive-arm symbol saving | **8.8–12.2 %** / **12.5–17.0 %** | the ceiling probe over all graphs, or the production campaign split by completion |
| a censoring rate | **96.8 %** at 30 s / the D14 figure at 300 s | a rate belongs to the budget that produced it, and the two campaigns used different budgets |
>
> None of these is a wrong number. Each is a right number that answers a question nobody asked.
> **Print the denominator in the same sentence**, and where two are in circulation, print both and
> say which is which — this is the most reliable defect generator in the project.

### 🔴 The red lines — do not write these

Each is technically defensible and would still be wrong, and a reviewer who checks finds every one.
Condensed from `T-06-FRAMING.md` §6, `T-06-article-notes.md` §10 and **`T-13-FRAMING.md` §7**, which
remain authoritative. **§5.3's brief carries the three T-13 red lines local to the cost section**
(*"governed by |Aut|, not by size"*, any |Aut| claim for the exhaustive arm, *"the cheapest canonical
form"*) and they are not repeated here.

| ❌ | Why |
|---|---|
| *"clears the size baseline on 5 of 5 Suite-2 datasets"* | True under UB, false under LB — inverts on 7 of 10. The most damaging available sentence |
| 🔴 *"clears the size baseline on 5 of 5 datasets"*, **unscoped** | C14's result is **Suite 1** against the **WL kernel**. It sits one word away from the red line directly above it, which is the same shape and is false. Two scopes, both mandatory, both in the same sentence: **which suite, which reference.** Over all fourteen cells it is 12 of 14, not 5 of 5 |
| *"IsalGraph approximates graph edit distance after all"*, or any WL result offered as repairing H3 | H3 names **graph edit distance**. Against it the refutation stands with every number intact. The WL measurement changes what the failure is *attributable to*, not whether it happened — and a reviewer who reads it as a rescue will check the ρ column and find it **falls** on three of the five datasets |
| *"IsalGraph beats its competitors under the WL kernel"* | It beats **three of four** pooled and **two of four** within-`n`. **min-DFS is not beaten under any of the eight references, in either band** — and min-DFS is the competitor R3 named as the most important. The unscoped sentence is the one that gets checked first |
| 🔴 *"it outranks both nauty serialisations under the WL kernel"*, **without `n ≤ 20`** | **The advantage does not survive above `n = 20`**: a tie against nauty-graph6 and a *significant loss* to nauty-sparse6 by the per-stratum sign test (40/70, p = 0.0054). The 14-of-14 dataset sweep is real and is **carried by the small-`n` band**, because per-dataset aggregation weights by graphs. Quoting the sweep without the size scope is the same error H2's *"most compact"* makes, in the same paper, one subsection apart |
| 🔴 *"the nauty separation is visible under the WL reference and not under GED"* | **Measured and false.** The split is `n`, not the reference: IsalGraph leads both nauty arms at `n ≤ 20` under **exact GED as well as WL**, and trails at `n > 20` under **wl, lb and ub alike**. Above `n = 20` the nauty margin is in fact *smallest* under WL. The reading is an artefact of comparing Fig. 2(a), which is 52/62 strata above `n = 12`, against (b)'s wide panel, which stops at 12 — **different size ranges, and the shared tick values make that comparison easier to make by accident.** Compare (a) against (b)'s bracket small multiples instead |
| *"IsalGraph outranks min-DFS under the upper bound within equal `n`"* | The single (reference, view, competitor) cell of 64 in which it leads — 5 W / 3 T / 1 L — and **its own lower bound reverses the verdict on 6 of those 9 cells**. This is C3's bracket trap wearing a different hat, and the paper has already committed to printing both bounds |
| A win claimed from the spectral λ-distance family | All four variants lose to min-DFS and clear the size null on 0, 2, 0 and 0 of 14. `spectral_esd` is the **least** size-dominated of the eight references and the one the encoding tracks worst — report that, it is the evidence that the WL result is not reference-shopping |
| *"ρ ≈ 0.93 demonstrates structural fidelity"* | Mostly the size channel. This paper supplies the instrument that refutes it |
| *"competitive with the best representations"* on distance | Best on **none** of 25 records. Not a scoping — a contradiction |
| *"most compact among representations admitting a metric"* | **False.** True in 0 of 122 strata |
| *"no existing method satisfies all four properties"*, **softened or not** | 🔴 **False against our own frozen comparison table**, which is the worst place for it to be false. `nauty_graph6` and `sparse6_nauty` carry **every** tabulated property; **IsalGraph does not**, because it rejects disconnected input. B6 asks for softening, and softening a false claim leaves it false. **R3.1b is discharged by stating the position plainly** — two externally canonicalised serialisations satisfy the tabulated set and we do not — and that concession is what the §6.1 trade-off argument is built on anyway |
| *"IsalGraph produces shorter encodings"* unqualified | Pooled, the significant results run 10 against / 9 for. The `n ≈ 20` scope is load-bearing |
| *"it computes everywhere, unlike the competitors"* | Eight representations complete on 100 % of every cell. It ties; it does not lead |
| *"N of M cells are significant"* as success | 75 rejections split 35 for / 34 against |
| Any β₁ without β_size beside it | The coefficient inverts in meaning without its competitor |
| Any coefficient from `aids_iam` or `coil_del`; `mutagenicity`'s β_lev | Unidentifiable (VIF 18.1 / 16.2) or point outside its own CI. **RETRACTED** |
| A dose–response for the LB/UB straddle | **RETRACTED** — an artefact of correlating X with X − U |
| *"the exhaustive arm closes the gap to nauty-sparse6"* | **Measured 2026-08-26: it does not.** 342.4 vs 336.0 at `n = 40`. Claimable at `n = 20` |
| *"greedy-min is never shorter than the canonical string"* unqualified | Exact for **completed searches**; false over the delivered column |
| `43 s/graph`, `≈ 520×`, `≥ 6.8 core-hours` | Unprovenanced. **RETRACTED** |
| Any pre-registered result restated more favourably than it came out | Forfeits the protection for all of them |

**The general rule**: a scoped claim carries its scope **in the same sentence**. *"Most compact of the
canonical codes"* is fair. *"Most compact"*, with the qualifier moved to a limitations section, is not
— and the difference is precisely what a reviewer checks for.

---

## 6. Every decision, and the experiment that justifies it

The user's requirement (4), discharged as a checklist. **A row with no experiment is a row that will
be asked about.** Column 3 is the pointer the main text must actually print.

| Decision in the paper | Justifying experiment | Lives in |
|---|---|---|
| One unit cost model for all GED | Corollary 2.13 + the heterogeneity R3.5b objects to | §4.2 → **S1** |
| Exact GED below `n = 12` | `networkx` A* census, 3.90 M pairs, 60 s/pair budget | §4.2 → **S1** |
| `BRANCH_FAST` as lower bound | Provably equivalent to `BRANCH` under constant edge costs; **verified identical on all 3,836,827 certified pairs**; cheaper of the two | §4.2 → **S2** |
| `BIPARTITE` as upper bound | Bake-off, 12 methods × 5 datasets, 46.8 M evaluations; wins **by elimination** under a frozen cost gate; the loosest of seven, disclosed | §4.2 → **S2** |
| Method options pinned, not defaulted | GEDLIB defaults change 91.5–93.6 % of values between runs; pinned, **0.0000 %** | §4.6 → **S2** |
| Sensitivity arm on the upper bound | `BP_BEAM_DET` misses the gate by 17 %; quantifies what the gate cost | §4.2 → **S2** |
| Reporting the **absolute** bracket gap, not only the relative | The two disagree in sign on 6 of 10 datasets; the relative denominator grows with `n` | §4.2 → **S3** |
| Each competitor's distance | Six criteria fixed in advance; **selection on GED correlation explicitly forbidden** | §4.3 → **S4** |
| Raw serialisations excluded | No admissible distance — F3 fails at 1/50 relabellings | §2.1, §4.3 → **S4** |
| The family taxonomy | Defined by *where canonicity comes from*, without reference to any result | §4.3 |
| Entropy bound as the primary bit convention | Both conventions fixed before any bit count existed; **three even-handed rules agree**; the payload-bits-per-byte table shows why they differ | §4.4 → **S5** |
| Graph-level bootstrap | Pair-level treats 89 LINUX graphs as 3,916 independent observations | §4.5 → **S6** |
| Per-dataset primary, pooled demoted | R3.5b, and the cohorts differ in density, size and provenance | §4.5 → **S6** |
| The confirmatory family and its size | Frozen before any p-value existed; enumeration is the definition, closed form printed as a check | §4.5 → **S6** |
| Pair exclusions | Ladder, per dataset, per reason, with counts | §4.5 → **S6** |
| `isalgraph_pruned` as primary arm | It is the pre-registered arm, and it is **never shorter** than `w*_G` — so the figures are a conservative bound | §3.3.2, §5.2 |
| The 300 s encode budget and its censoring | Censoring characterised by \|Aut(G)\|; reported by stratum | §5.3 → **S7** |
| **The worst case is \|Aut\|-governed, not "exponential"** | **Proposition 1 verified with 0 violations on 16,370 cohort + 664 constructed graphs**, plus the controlled ladder experiment (12 cells, ρ = +0.892, p = 0.0064) with a flat search-free null arm | §3.2.3, §5.3 → **S7** |
| **The costed operations are the right ones** | Instrumented mirror **byte-identical to the frozen reference on 178,886 (graph, start) pairs, 0 mismatches**; `core/` unmodified | §3.2.3 → **S7** |
| **A fitted exponent is not a complexity result** | Three arms of one method on one cohort fit α = 2.04 / 3.15 / 17.43 | §3.2.3, §5.3 → **S7** |
| **Automorphism detection is the fix** (future work) | Corollary 3, plus nauty's canonical form measured **not** degrading with \|Aut\| (ρ ≈ −0.61, 18 of 20 cells) | §5.3, §7 → **S7** |
| **A finer invariant is not the fix** | 1-WL attains the orbit ceiling on **99.939 %** of the cohort; no invariant can go below it (Prop. 1) | §7 → **S7** |
| No sequential-model experiment | **Not justified by an experiment.** Conceded once in §6.3; the contribution stands on the theorem (§2.3) | §6.3 |

---

## 7. The family taxonomy

Four design points, distinguished by **where canonicity comes from**. Definable without looking at a
single result, which is what makes scoping a sentence to a family principled rather than fitted.

| Family | Members (after P4) | Canonicity |
|---|---|---|
| **Canonical code** | **IsalGraph**, gSpan min-DFS, AGM CAM | intrinsic — the code *is* the canonical form |
| **Canonicalised serialisation** | nauty-graph6, nauty-sparse6 | outsourced — canonicalise, then serialise |
| **Feature map** | WL subtree | invariant, but neither complete nor invertible |
| **Trivial baseline** | `\|n_i − n_j\|` | not a representation; declared as a baseline |

`design.py` already carries this as `Family` and draws it, so a reader can check the scope of any
sentence against the figure. **The scope of every compactness claim is "canonical code", and we are
first in it.**

Two facts to state once and not repeat: `agm_cam` refuses above `n = 12` by its own scope guard and
completes on **6.15 %** of Protein, so any pooled win rate against it is a small-graph statement;
`min_dfs` has a completion floor of 0.9478.

---

## 8. Supplementary architecture

**A separate `supplementary.pdf`, never `\appendix`** (P3). Every section below is cited from the main
text by number — Elsevier requires it, and it is what makes the main text's one-clause justifications
legitimate rather than hand-waving.

| § | Contents | Cited from | Source |
|---|---|---|---|
| **S1** | Cost model derivation; the exact-GED census and its censoring; the three-level zero-value guard and why the per-pair form is wrong | §4.2 | T-03, `statistics.md` D6 |
| **S2** | **The GED bound bake-off** — 12 methods × 5 datasets, tightness, certification rate, cost gate, the determinism finding, the `BIPARTITE` disclosure and the `BP_BEAM_DET` sensitivity arm, literature verification | §4.2, §4.6 | T-27 |
| **S3** | **Bracket behaviour at scale** — absolute and relative width vs `n` on all 10 datasets, by size and density stratum, certification rates, realised cost | §4.2, §5.4 | T-05 |
| **S4** | **Distance admissibility** — the six criteria, every attempted cell including the failures, the selection rule and its repair, F3 measurements | §2.1, §4.3 | T-04a |
| **S5** | **Bit accounting** — four conventions, what each decides, the discordance count, the full per-stratum IUT | §4.4, §5.2 | `t06_bit_convention.py` |
| **S6** | **Statistical protocol in full** — bootstrap, Mantel, MRM with VIF and exclusions, the pre-registered family and its enumeration, the pair-accounting ladder per dataset, BH tables | §4.5, §5.4, §5.5 | T-02, T-06 |
| **S7** | **Algorithm listings and the complexity campaign** — Algorithm 3's full listing, the ladder construction and its three families, the per-ladder Spearman and completion tables, the fitted scaling exponents, the invariant-resolution figure, the real-cohort encode census and the greedy ablation | §3.2.3, §3.3.2, §5.3, §7 | **T-13**, T-06 |
| **S8** | **Per-dataset result grids** — full ρ tables both bounds, per-competitor per-stratum head-to-heads, the within-`n` profiles | §5.2, §5.4 | T-06 |
| **S9** | **Worked examples** — S2G and G2S traces for the exhaustive and pruned canonical forms, four panels | §3.2 | T-09 |
| **S10** | Reproduction — commands, seeds, build hashes, artifact manifest | §4.6 | all |
| **S11** | **Proofs** — Theorem 2.12's proof in full, as re-verified by T-22. The **statement** and a 68-word sketch stay in §3.3.3; this is the worked argument | §3.3.3 | T-22 |

> **S9 is the relief valve.** The worked-example figure is the only artifact in the whole inventory
> that **no reviewer requested** ([manuscript](manuscript.md) §3.2 item 11). Putting it in
> supplementary from the start, rather than cutting it under time pressure at the end, costs nothing
> and removes it from the risk register.

---

## 9. Consequence of P4 — the one hole, and its one-sentence fix

Removing raw `adjacency`, `graph6` and `sparse6` entirely is **defensible and favourable**, and both
halves need stating.

**Defensible**: they are excluded by T-04a §3.4's rule — *no representation reaches a results table on
a distance that fails the admissibility criteria* — which was **fixed before any bit count or
correlation existed**. All three fail isomorphism-invariance at 1/50 relabellings. It is a rule, not
a selection.

**Favourable, and worth knowing**: it removes `sparse6`, which T-06 identified as the **single most
compact representation, beating even nauty-sparse6**; and `adjacency`, which out-correlated IsalGraph
on 3 of 5 datasets *despite* failing invariance. **Apply the rule; state the rule; do not argue from
the benefit.** The order matters — a rule stated after its consequence reads as reverse-engineered.

**The hole**: R1.2 asks what the representation buys in **uniqueness**, and the F3 = 1/50 measurement
was the *measured* answer to "what does canonicalisation buy you". With the rows gone, that answer has
no home.

**The fix, and it is one sentence in §2.1** (already written into §4's brief):

> *"graph6 and sparse6 serialise one specific vertex ordering, so an edit distance between them
> measures node ordering rather than structure unless a canonical labelling is applied first; we
> therefore evaluate them only in their nauty-canonicalised forms (S4)."*

Zero table rows, zero competitor columns, R1.2's conceptual question answered with a measurement
behind it. **If this is cut, R1.2b's `uniqueness` axis becomes an assertion.**

**Unchanged by P4**: `sparse6_nauty` stays, and so does the concession that it dominates IsalGraph on
both axes above `n = 20`. Nothing removes it and nothing should try.

---

## 10. Artifact inventory — revised 2026-08-26 (v1.2)

**Status**: `MUST` = a named demand with no substitute · `STRONG` = carries a hypothesis ·
`CUT-n` = position in the pre-declared cut order.
**Source**: a path means **the PDF exists today**; `BUILD` means the content is measured and locked
but the artifact is not made.

> ### The selection rule this revision applies, and it is the PI's
>
> **A reviewer asked for it → it stays in the main text. Nobody asked → it is a float only if it
> carries a claim no prose can.** Everything else goes to supplementary. That rule is what removed
> the two result *tables* below and what kept the per-dataset block, and it is worth stating in the
> response letter, because it explains the shape of the whole paper in one sentence.

| # | Artifact | § | p | Source | Status | Demand |
|---|---|---|---:|---|---|---|
| Tab. 1 | Instruction set *(kept from the submitted paper)* | 3.1 | 0.5 | submitted | CUT-3 | — |
| **Tab. 2** | **Representation comparison** — R1.2b's five axes, every cell measured | 2.2 | 0.8 | `T-06/tab_representation_properties` | **MUST** | **AE.3**, R1.2b, R3.1b |
| **Tab. 3** | **[28] / [29] inherited · modified · new** — 9 components, 3/5/1 | 2.3 | **0.67** | `T-07/artifacts/tab3_prior_work_delta.tex` | **MUST** | R3.1a(i), R3.7b |
| **Tab. 4** | **Datasets** — 10 rows, both suites, label column, discarded side | 4.1 | 1.2 | **BUILD** (`data.md` §1) | **MUST** | AE.1, AE.4b, E1, E6 |
| Tab. 5 | **Payload bits per stored byte** | 4.4 | 0.3 | `T-06/tab_bit_overhead` | CUT-1 | R3.6a |
| **Fig. 1** | **Information content** vs `n`, with a coding-overhead inset | 5.2 | 0.8 | `T-06/fig4_information_content` | STRONG | AE.4a, H2 |
| **Fig. 2** | 🔴 **REPLACED 2026-08-30 — the within-`n` correlation under BOTH references.** **(a)** the WL kernel, one panel, exact at every `n`; a rule; **(b)** the previous figure entire — exact GED at `n ≤ 12` and the bracket small-multiples above. Shared ρ axis, one legend, per-panel `x`. **The reader sees one variable change.** Rendered 7.03 × 4.48 in against the old 7.03 × 4.38, so the page budget is unchanged. **Carries moves 5–6 — the within-`n` instrument applied to both references, and the head-to-head under WL where the nauty result is visible.** ⚠ It does **not** carry move 4: the size null is `ρ(\|n_i − n_j\|, d_ref)`, and inside a stratum `n_i = n_j` that argument is identically zero **by construction** — which is exactly why the within-`n` view is the size-controlled one. The null is therefore structurally absent from this figure and from the profile behind it, and no rebuild adds it. **Move 4 is the inline table's job alone** | 5.4 | 0.9 | `T-28/fig_rho_vs_size_wl_vs_ged` | STRONG | H3, AE.1, R3.5b |
| — | 🔴 **`fig1_rho_vs_size.pdf` must be REGENERATED regardless** — its `wl_subtree` series carries the `_wl_counts` defect (§5.4 move 5). If Fig. 2 stays the GED-only figure, it is still wrong until rebuilt | 5.4 | — | `6b89b4f` | **MUST** | — |
| **Fig. 3** | **The cost law** — encode time vs \|Aut(G)\| on one ladder, with `n`, `m` and the degree sequence held fixed | 5.3 | 0.7 | `T-13/fig_t13_main` | STRONG | **R3.7d**, H4 |
| **Alg. 1** | **S2G — the interpreter.** Inlined, not floated (E7) | 3.1 | 0.6 | submitted | STRONG | — |
| **Alg. 2** | **Greedy G2S**, with the `C`/`c` guards **and** duplicate checks rewritten to match the implementation | 3.2.2 | 0.9 | submitted, corrected | **MUST** | **R3.4a** |
| | *floats subtotal* | | **7.4** | | | |
| — | *(inline, 5 rows)* **per-dataset ρ against exact GED** — dataset, ρ, size null, excess | 5.4 | 0.25 | `rho_table.json` | **MUST** | **R3.5b** |
| — | 🔴 **NEW *(inline, 6 rows)* — the same five datasets, one column changed.** Pairs, then ρ / size null / excess-with-CI under **exact GED** and again under the **WL kernel**, closing on `1 of 5` against `5 of 5`. **This is §5.4's load-bearing object**: it satisfies R3.5b's per-dataset demand and carries move 4 in one table. The `exact` half reproduces the row above it to four decimals — say so, it is what licenses reading the two halves as one experiment | 5.4 | 0.3 | `T-28/data/t28_bootstrap_verdicts.json` | **MUST** | **R3.5b**, AE.1, H3 |
| — | *(inline, 3 rows)* the IAM Letter LOW/MED/HIGH control | 5.4 | 0.2 | T-06 | STRONG | R1.3a, R1.3c |
| — | *(inline, 4 rows)* confirmatory rejections split by row and direction | 5.5 | 0.2 | T-06 | **MUST** | AE.4c |
| | **Total** | | **8.05** | | | |

> ### 🔴 v1.3, 2026-08-26 — the three §3 figures leave the body and become the graphical abstract
>
> **The canonical search tree and the S2G/G2S worked example are merged into one figure that
> serves as the graphical abstract.** They come out of the inventory: **−1.2 p** of floats, and
> **§3 drops 11.6 → 10.4 p**. The graphical abstract is submitted separately through Editorial
> Manager and is **not typeset into the manuscript**, so it costs no pages at all.
>
> **This also closes an open item rather than adding one.** `graphical_abtract.pdf` was a
> **blocked** artifact — its panel (b) printed `Wins: 99.6 %`, `β = 0.537`, `R² = 0.947` and
> `14,108×`, every one retired by T-06, and regenerating half of it would have made the stale half
> look freshly checked. Replacing it outright with a figure built from the frozen running example
> retires that risk instead of deferring it. **T-24's inherited problem is solved.**

> ⚠ **Sum the column before quoting it. This table has been wrong twice.** v1.1 printed **7.9**
> against rows summing to **8.9**; v1.3's first draft printed **6.5/7.15** against **7.4/8.05**.
> Both were caught by adding the column, and nothing else catches them.
>
> **Alg. 1 and Alg. 2 are new *rows*, not new pages** — both are in the submitted manuscript and
> both already sit inside §3's 10.4 p. They are listed here so the §10.4 decision is visible in the
> inventory rather than buried in prose. The v1.3 delta against v1.2 is **−1.2 p**, which is the
> three figures leaving, and the §3 section budget is the authority.

### 10.0 What moved out, and what protects the demand it served

| Removed | To | What now carries the demand |
|---|---|---|
| **Head-to-head table** (was Tab. 6, 0.8 p) — compactness and correlation against each comparator | **S8** | **AE.4a / R1.1** are carried by §4.3 (the comparator set and how each distance was chosen), **Tab. 2** (their properties), **Fig. 1** (compactness, quantitative) and **Fig. 2** (correlation, quantitative). The four numbers move into §5.2 prose and **must all appear**: 112 of 112 against min-DFS, median **+215 bits**, the both-bounds correlation verdicts, and the nauty-sparse6 concession. **The concession is the one that cannot be dropped** — it is what makes the min-DFS result read as a finding rather than a selection |
| **Per-dataset ρ table** (was Tab. 7, 0.6 p) | **inline, 0.25 p** | **R3.5b asked for this in as many words** — *"dataset-level correlations treated as the primary evidence"* — so under the PI's rule it stays in the main text. Five Suite-1 rows against **exact** GED is the minimum that honours "primary", and it is also where the concession lives: below the size null on **4 of 5**. Full 25 records → **S8** |

> ### ⚠ The one thing this costs, and it needs a deliberate answer
>
> **R3.7c asked for the schematic in a named section** — *"Section 2.3 could also benefit from a
> small schematic illustrating the canonical search space"*. A graphical abstract is not Section 2.3:
> it is not typeset into the body, it carries no figure number, and **`\ref` cannot point at it**.
> A reviewer checking whether the schematic was added to the canonicalisation section finds no
> figure there.
>
> **The modal protects us and the placement does not.** *"Could benefit from"* is a suggestion, and
> [manuscript](manuscript.md) §3.2 already ranks R3.7c tenth of eleven in the cut order — so
> declining it outright would have been defensible. **Silently relocating it to an artifact the
> reviewer will not read as Section 2.3 is a weaker position than either keeping it or cutting it**,
> because it looks like coverage without being it.
>
> **Recommended, and it costs zero main-text pages: put the merged figure in the supplement as
> Fig. S1 as well.** §3.3.1 then carries a real pointer — *"the search space is drawn in Fig. S1,
> reproduced as the graphical abstract"* — the response letter names a numbered artifact, and the
> graphical abstract still does its own job. One line in §3.3.1, one page of supplement, nothing in
> the budget.

**Pre-declared cut order** — decided now, not under deadline pressure:
1. **Tab. 5** → **S5**. One clause survives: *"the realised convention charges the adjacency triangle
   7.50 payload bits per stored byte and the instruction string 3.17 (Table S5)."* **−0.3**
2. **Alg. 1** → **S7**. The interpreter's semantics are given by Table 1 and the §3.1 prose; the
   listing is the formal restatement. **−0.6** *(see §10.4 — this is the contingency, not the plan)*
3. **Tab. 1** → **S7**. The alphabet reads as a nine-item inline list. **−0.5**

**Never cut**: **Alg. 2** (R3.4a names specific lines of it), Tab. 2 (the Area Editor endorsed it in
their own voice), Tab. 3, Tab. 4, and all three inline blocks.

> ### On appending the [28]/[29] delta to the comparison table — asked and answered, 2026-08-26
>
> **Decided: keep them separate.** The merge is a category error and it is worth recording why, so
> the question does not come back. **Tab. 2's rows are representations being compared**;
> **Tab. 3's rows are components of our method attributed to a source.** R3.1a(i) asks *"which
> components are inherited, modified, or genuinely new"* — a rows-as-representations table cannot
> express *"the CDLL architecture is inherited from [29] and extended to unbounded degree"*, because
> that sentence is about a component, not about a representation. A property table says what each
> thing **does**; the delta table says where each piece of ours **came from**. R3 asked for the
> second, and R3 checked thirteen of thirteen checkable claims in round one.

> 🔴 **Two keys in `design.py` name different objects than their words suggest, and §5 reads both.**
> `isalgraph_exhaustive` is the **hybrid with pruned fallback** (α = 4.71); `isalgraph_canonical` is
> the **true exhaustive form** (α = 17.43). Reading `tab_t13_scaling_exponent.tex` by key name
> attributes 17.43 to the wrong arm — a sixfold error on the number that carries §10.1's whole
> argument. §10.1's mapping below is right **because it names mathematical forms, not registry
> keys**; keep it that way.

> ## 🔴 The production upper bound is **BIPARTITE**, not IPFP. Settled 2026-08-26.
>
> **T-27 selected it** (*"Upper bound: `BIPARTITE` — wins 5 of 5 by elimination"*, 2026-08-13) and
> **T-05's production run used it** (*"UB — upper bound, primary (BIPARTITE)"*, with `BP_BEAM_DET`
> as the disclosed sensitivity arm). T-06 consumed T-05's matrices. **Every "IPFP" as the production
> UB predates T-27's selection and is stale** — including in `approx_ged.md` §§1/4/6, the repo
> `CLAUDE.md`, and T-06's own reproduction-parameters table, which inherited it.
> **R3.5b's answer names this bound. Fix it before the letter quotes it.**
>
> **Both looseness ratios are correct and they compare different things** — a third instance of the
> same trap. BIPARTITE's mean relative error is 1.0946: **6.7× the next-loosest** (`BP_BEAM_DET`,
> 0.1624) and **13.0× the tightest** (`IPFP_MS`, 0.0841). Say which comparison is meant, every time.

### 10.1 What R3.4c gets instead of a float

`T-13/tab_t13_scaling_exponent` **dissolves R3.4c in three numbers**, and three numbers do not need a
table. Fitted `T ~ n^α` by OLS over completed encodings, bootstrap CI, **on one constructed cohort**:

> **greedy α = 2.04 [1.88, 2.20] · pruned canonical α = 3.15 [2.18, 4.01] · exhaustive canonical
> α = 17.43 [14.67, 19.35]** — beside min-DFS 2.08, nauty-graph6 3.24, AGM CAM 4.89.

**Three exponents for three arms of one method on one cohort is the whole argument.** A fitted
exponent records how cost happens to scale over a cohort in which |Aut| co-varies with `n`; it is not
a complexity claim, which is exactly why `n^{4.9}` and `n^{9.0}` could coexist in the submitted
manuscript without either being wrong. **Do not reconcile the two figures — dissolve the question.**
That is R3.7d's three-way separation delivered as one argument rather than three paragraphs.
Full table → **S7**.

### 10.2 Supplementary artifacts — all rendered, none to build

| Artifact | S | What it carries |
|---|---|---|
| `T27_{lower,upper}_bound`, `T27_{lower,upper}_cd` | **S2** | The bake-off. The upper-bound panel (a) **is the disclosure**: BIPARTITE runs 0.00 → **2.19** relative error across `n = 4→12`, a slope ten times any alternative's |
| `fig_71_width_vs_n_within_dataset`, `fig_72_certification_rate`, `fig_73_width_by_stratum`, `fig_71_slope_forest`, `fig_74_realised_cost` | **S3** | Bracket behaviour on all 10 datasets; certification 28.5 % → 0.03 % |
| **`tab_representation_summary`** | **S8** | **The master record** — every representation × every property × bits at two anchor sizes × median Δρ in three bands. Landscape, `\scriptsize`. This is what a reviewer recomputing our claims reads, and it is why Tab. 2 can stay small |
| **`tab_representation_headtohead`** | **S8** | **Moved out of the main text 2026-08-26** (§10.0). Compactness under both bit conventions and correlation under both bounds, per comparator. The main text keeps its four numbers in prose |
| `fig2_rho_by_representation`, `fig3_absolute_scale` | **S8** | Per-dataset ρ detail; absolute distance against the GED bracket drawn as a band |
| Full 25-record ρ table | **S8** | The main text prints 5 Suite-1 exact rows inline |
| `fig_t13_cost_law_default` | **S7** | 21 ladder panels × 13 representations, censored rows drawn as arrows |
| **`fig_t13_resolution`** | **S7** | **Proposition 1 drawn**: invariant classes against orbit count with `y = x` as a *proven* ceiling, and the deficit below it. This is why the remedy is nauty-style automorphism detection and not a finer invariant (Corollary 3) — it makes the future-work statement a conclusion rather than an apology |
| `tab_t13_ladder_spearman`, `tab_t13_completion`, `tab_t13_scaling_exponent` | **S7** | 300, 285 and 25 lines |
| **The merged explanatory figure** — search tree + S2G + G2S | **S9** | **Reproduced as Fig. S1**, so §3.3.1 and §3.2 can cite a numbered artifact; the same file is the graphical abstract. See §10's ⚠ box |
| `fig_worked_example_{s2g,g2s}_pruned` | **S9** | The pruned form's traces |
| **Algorithm 3** — pruned canonicalisation, full listing | **S7** | §3.3.2's definition plus four lines of prose carry what the reader needs |

### 10.3 The merged explanatory figure — ✅ REBUILT 2026-08-26

**Superseded in placement, not in content.** The search tree, the S2G panel and the G2S panel are
now **one figure serving as the graphical abstract** (§10) rather than three body figures. The work
recorded below was done first and still applies — it is what makes the merged figure legible, and
the same constraints bind whatever the figure is used for.

**The schematic answers R3.7c as drawn**: six start-node subtrees, branching at `V`/`v`, forced steps
dashed where nothing branches, the canonical path highlighted. Everything below was presentation.

> 🔴 **Why it could not ship as it was.** `search_tree.py` drew 5.5–6.5 pt labels into a **7.0 in**
> figure. Placed at `width=\textwidth` in a **4.72 in** text block, everything scaled by **0.674**
> and the labels reached the page at **3.7–4.4 pt** — unreadable, on the one figure a reviewer
> explicitly asked for. **Point sizes inside a figure are absolute**, so the only way a declared
> size is the printed size is for the render width to equal the placement width.

| # | Change | Where |
|---|---|---|
| 1 | **`PATREC_TEXT_WIDTH_INCHES = 4.72`** — a named constant with its derivation (letterpaper less the 4.8 cm margins at `main.tex:11`), used by the default, the caller and the test | `style.py` |
| 2 | Rendered at **4.72 × 2.6 in**, so nothing is scaled | `figures.py`, `search_tree.py` |
| 3 | **Two-line axes title deleted** — it was caption text baked into the image | `search_tree.py` |
| 4 | Row labels to **Title Case** | `search_tree.py` |
| 5 | `label_fontsize` 5.5 → **6.0**, row labels 6 → **6.5**, node markers 74 → **62** pt² | `search_tree.py` |
| 6 | **Legend 4-across → 2×2**, and capitalised. Four entries overflowed 4.72 in and were **clipped at both ends**. Only the leading word is capitalised: `V` and `v` are distinct instructions, so title-casing the whole label would rename half the alphabet | `search_tree.py` |
| 7 | **Layout rebuilt** — the tree now spans the full width, and the legend and the source graph share the band beneath it, legend left and graph right at equal height. Placing the graph *beside* the tree cost the leaf row a fifth of its width, and the leaf row sets how small every label has to be | `search_tree.py` |
| 8 | Left `xlim` margin 2.6 → **3.8** data units — it holds the row labels, and "Start Node" clipped at the narrower width | `search_tree.py` |
| 9 | GridSpec `top` 0.88 → **0.985** when there is no suptitle — matplotlib reserves that band for a title that no longer exists | `search_tree.py` |
| 10 | **Leaf nodes are labelled.** Truncated and terminal nodes carried an empty label, so the bottom row read as a row of blank markers. The depth budget truncates what comes *after* a node, which says nothing about how it was reached | `search_tree.py` |
| 11 | **Edge labels moved to 0.70 of the way to the child.** Sibling midpoints sit half a child-spacing apart, close enough that two two-character segments (`pv` against `pv`) overlapped illegibly. Siblings diverge toward their children, so the shift separates them by ~40 % without moving a label off its own edge | `search_tree.py` |
| 12 | Height 2.6 → **3.0 in** to give the legend/graph band room | `figures.py` |
| 13 | **Root labels `$v_k$` → plain `k`.** Every other mark in the figure *set* names a node of `G` by its integer — the interior and leaf labels, the inset, and all four worked-example panels (`draw_state_graph` writes `str(node)`). A root drawn `$v_3$` beside a graph drawn `3` made the reader translate between two notations for one object; the start-node role is already carried by the fill colour and the row label | `search_tree.py` |

**Verified**: `ruff` clean, `mypy --strict` clean on 80 files, `tests/viz/` **137 passed**.
The archived `canonical_search_tree.{pdf,png}` are regenerated at **4.72 × 3.0 in**.

**Captions written**, one `.caption.tex` per figure beside the PDFs, compile-checked with
`amsmath`/`amssymb` at 0 errors: `canonical_search_tree`, and all four worked-example panels.

> ⚠ **One test did assert the old width** — `test_worked_example.py::test_search_tree_draws_every_start_node`,
> not in `test_search_tree.py` where it would be looked for. It now asserts
> `PATREC_TEXT_WIDTH_INCHES` **and says why**, so the render-width invariant is enforced rather than
> incidental.

**Unchanged, deliberately**: `max_roots`, the palette, the legend content and the running example.
The six subtrees are the point, and the example was selected by enumeration (`|Aut(G)| = 1`, greedy
attains `w*_G` from one start node and not the others).

> ⚠ **The worked-example panels still carry the width defect.** They are drawn **7.0 × 2.84 in** by
> the same code family, with the same 5.5–6.5 pt labels. **A graphical abstract is displayed at a
> fixed small size on the publisher's page, so this matters more there, not less** — and
> `PATREC_TEXT_WIDTH_INCHES` exists now. Fix them before the merged figure is finalised.

### 10.4 The algorithms — they fit, and two of the three stay

**Answered from the budget, not by preference.** With the three §3 figures gone the manuscript lands
at **≈ 32.4 p**, leaving **≈ 2.6 p** against the ceiling. Algorithms 1 and 2 cost ≈ 1.5 p together
and are already inside §3's 10.4. **They fit.**

| | Disposition | Why |
|---|---|---|
| **Alg. 2 — greedy G2S** | **Main text. Not negotiable.** | R3.4a names *"lines 24 to 30"* and asks us to verify them against the implementation. The correction is the answer to the comment, and an answer that lives in a non-typeset attachment is the same defect §10.0 avoided for R3.5b. **If only one algorithm survives, it is this one.** |
| **Alg. 1 — S2G interpreter** | **Main text**, first out under pressure | No reviewer asked, but it carries *"every string over Σ decodes to a valid graph"* — an abstract-level claim — and with the worked example gone it is now the **only** concrete account of execution in the body. That argues for keeping it, not against |
| **Alg. 3 — pruned canonicalisation** | **S7**, as already planned | The longest of the three, and §3.3.2's definition plus four lines of prose carry what the reader needs. No demand attaches to the listing |

> **Removing the worked example makes the pseudocode *more* load-bearing, not less.** The figure was
> the intuitive account of execution and the listing is the formal one; §3 was going to carry both.
> Dropping both would leave the instruction set explained only in prose — and R3 read this
> manuscript closely enough to check thirteen of thirteen checkable claims. **Keep Alg. 1 unless the
> page count actually forces the cut**, and if it does, cut it *after* Tab. 5.

**Order of operations, and it has bitten this project before**: run **E7's float fix first**
(`\floatpagefraction` / `\textfraction` at `main.tex:66–67` currently push all three algorithms onto
dedicated pages 33–35, *after the bibliography*), **then** measure, **then** decide. Trimming before
E7 measures a document that does not exist.

### 10.5 Build risks — three are silent-failure traps of one family

0. 🔴 **A `\label` duplicated across the article and the supplement resolves to the wrong float
   and the build stays clean.** Found 2026-08-31 by the paper-writing session while placing the new
   correlation table. `tab:representation-headtohead` now exists in **both** documents — the article's
   §5.4 table and an older compactness+GED float the supplement's `s08` still inputs — so a
   `\ref{tab:representation-headtohead}` written in `s13` silently pointed at `s08`'s table. **No
   warning, no undefined reference, exit 0.** `\ref` cannot cross documents, so within the supplement
   LaTeX resolved it to the nearest definition and was satisfied.
   **The rule**: a table moved between the two documents keeps its label only if the label is
   *retired* on the other side. Where a cross-document pointer is genuinely wanted, use `\supp{N}`,
   which is prose and cannot mis-resolve. Related and louder: `\suppshort` is article-only and
   **fails** in the supplement, which is the behaviour to prefer — a macro that errors is safer than
   a label that resolves.

1. 🔴 **`figsize` does not govern the emitted width of a benchmark figure, and every body figure is
   mis-scaled.** Two causes, found 2026-08-26, and the second is the one that matters.
   **(a)** `design.text_width()` returned `IEEE_TEXT_WIDTH_INCHES` = 7.0 — the two-column IEEE print
   area, wrong for a single-column 4.74 in manuscript. Same root cause as `search_tree.py`.
   **Fixed**: it now returns `PATREC_TEXT_WIDTH_INCHES = 4.7382`, wired through `plotting_styles`.
   **(b)** `plotting_styles.save_figure` writes with **`bbox_inches="tight"`**, so the emitted PDF
   width is whatever the *content* needs and the declared `figsize` is advisory. Fixing (a) moved the
   information-content figure 390 → 326 pt and did **nothing** for `rho_vs_size`, which is
   content-bound at 506 pt. **As actually placed** — the scale depends on the `\includegraphics`
   option, not on the file alone:

   | figure | rendered | placement | scale |
   |---|---:|---|---:|
   | information content | 325.8 pt | `width=\textwidth` | 1.047 — fine |
   | cost law | 227.1 pt | **natural size, no `width=`** | **1.000** |
   | rho vs size | 506.3 pt | `width=\textwidth` | **0.674** — 5.5 pt labels land at 3.7 pt |

   **So the defect costs exactly one figure today.** The cost-law figure was placed at natural size
   deliberately, precisely because `\textwidth` would have upscaled it 1.5×; the per-figure `width=`
   fallback is already in force where it was needed. **The pipeline fix is still the right one and is
   not blocking.**

   `isalgraph.viz.style.save_figure` does **not** use tight bbox, which is why the search-tree fix
   worked there and why the two halves of the codebase behave differently.

   > **Tested, and width tuning alone cannot fix this — do not retry it.** The cost-law figure uses
   > `design.column_width()` (3.39 in, the IEEE *single-column* constant), which is why it is
   > enlarged 1.50×. Switching it to `text_width()` was tried: tight bbox reflowed the result to
   > **487 pt**, so the figure went from 1.50× enlarged to **0.70× shrunk** — worse, because 1.50×
   > is ugly and 0.70× is illegible. **Reverted; the staged file is the original.** Under tight
   > bbox the emitted width is content-driven and the declared `figsize` cannot control it.
   >
   > **The real fix is a pipeline change**: make tight bbox opt-out in
   > `plotting_styles.save_figure` (e.g. `tight: bool = True`) and have the paper figures pass
   > `tight=False`, **plus** `constrained_layout` or an explicit `subplots_adjust` so content fits
   > the declared box — without that second half, turning tight off clips the labels. That is a
   > coordinated change across every benchmark figure and it is **not** to be attempted while
   > section agents are running against staged artifacts.
2. 🔴 **`adjacency` has two unrelated jobs in the information-content figure, and removing it the
   obvious way fails silently.** It is a plotted series *and* the sole source of `(n, m)` for the
   information-theoretic floor — `data.unlabeled_floor()` skips every cell whose representation is
   not `adjacency`. Filtering the cells out of the input directory therefore kills the floor **and**
   the coding-overhead inset; and because the caption is generated it simply stops mentioning the
   inset, so figure and caption stay consistent with each other while both lose a panel.
   **Correct surgery: drop it from the plotted registry, leave the cells on disk.** Verified —
   floor and inset survive. Done; `article/information_content.pdf` is regenerated without it.

   > 🔴 **The generated `.caption.tex` must not ship, and it is the same failure mode one level up.**
   > The regenerated caption disagreed with its own figure **three times**: it named the *realised
   > bytes* convention while the axis reads *Entropy Bound* (§4.4 makes entropy primary, so a reader
   > taking the caption at its word reads the compactness result off the demoted column); it said the
   > exhaustive arm was "not drawn here" while **IsalGraph (exh.)** is in the legend and on the plot;
   > and it carried **two** `\ref{tab:representation-summary}`, a label that now lives in the
   > supplement and would compile undefined. Passing `convention="entropy"` did **not** prevent the
   > first. **Verified independently and removed from `article/`.** Captions for the body figures are
   > hand-written and checked against the rendered axis, not generated — a caption that adapts itself
   > stays internally plausible while drifting from the plot, which is exactly how it survives review.
   It stays in the **cost-law** figure, where it is not a comparator but one of the five
   **search-free null arms** whose flatness licenses reading the |Aut| slope as symmetry. The
   caption must say "null arm", not "competitor".
> **Two "N of 10" statistics in T-05 are not interchangeable, and both are right.** **6 of 10** is
> the count of datasets where the absolute and relative bracket slopes **disagree in sign** (§7.1b's
> `same sign?` column). **4 of 10** is the count of datasets spanning enough `n` to carry an
> **unconfounded** slope — `aids_iam`, `coil_del`, `mutagenicity`, `protein`. This file's "6 of 10"
> for sign disagreement is **correct and must not be "corrected" to 4**. Same family as the
> certification-versus-zero-distance conflation above: two figures over the same denominator that
> look interchangeable and are not.

3. 🔴 **A stale `main.bbl` in the working directory shadows the correct one, and the build reports
   phantom undefined citations.** With `-output-directory`, TeX still searches the **working
   directory first**, so an out-of-date `main.bbl` beside the sources wins over the fresh one bibtex
   just wrote into the out directory. **Reproduced: 22 undefined citations, every one a key that
   exists in `cas-refs.bib`.** The tell is `(./main.bbl` in the log — a leading `./` means it read
   the shadowing copy.

   > **The first diagnosis was wrong and the workaround it implied is worse than useless.** It looked
   > as though bibtex was never run in the out directory; running it there *does* produce a correct
   > `.bbl` that pdflatex then ignores, so the workaround appears to fail and invites more of the
   > same. **The fix is to refresh the shared `main.bbl`**, after which a plain build reports zero
   > undefined citations with no manual step.

   The danger is not the noise: an agent adding references concludes its verified DOIs are broken and
   "fixes" a non-problem. **Never diagnose a citation from a build whose log shows `(./main.bbl`.**
   Third member of the same family as the caption and the floor/inset traps — a build that stays
   internally consistent while being wrong about the outside world.
4. ⚠ **Two resolution numbers exist and must never be mixed.** `fig_t13_resolution` reports 1-WL at
   the orbit ceiling on **39 of 132** graphs and the triplet key on **23 of 132** — those are the
   *constructed ladder* graphs, built to be adversarially symmetric. Over the **16,370 cohort**
   graphs the same two quantities are **99.939 %** and **41.869 %**. Both are correct; quoting either
   with the other's population is a defect.
5. 🔴 **Every table float needs `\linespread{1}` inside it.** The `review` class option sets
   `\@blstr{1.5}` (`elsarticle.cls:73`), stretching every table line by 1.5×. **Measured, not
   estimated**: Tab. 3 is 368.4 pt = 0.67 p with the reset and **over a full page** without it. Put
   it after `\footnotesize` in all five tables. Every page figure in §10 assumes it.
6. **Two artifacts do not exist yet** — Tab. 4 (`data.md` §1) and the inline
   per-dataset block (`rho_table.json`). All have their content measured and locked; none needs a
   computation.
   **Tab. 4 is on the critical path**: §4.1 cannot be drafted around a table whose columns are still
   undecided.

**Superseded**: the "not regenerated, by decision" note on the graphical abstract. It is now
*replaced* by the merged explanatory figure (§10), so the four retired numbers are gone rather than
deferred.

---

## 11. Coverage — every demand has a section

Cross-checked against [demands](demands.md). **A blank cell is a hole.**

| Demand | § | Demand | § |
|---|---|---|---|
| AE.1 size impact | 4.1, 4.2, 5.2, 5.3, 5.4, 6.3 | R3.3a scope | abstract, 1, 6.3 |
| AE.2 related work | **2.1** | R3.3b/c directedness | **3.3.3** |
| AE.3 comparison | **2.2 / Tab. 2** | R3.4a Alg. 2 | **3.2.2** |
| AE.4a benchmark models | 4.3, Tab. 2, Fig. 1, Fig. 2 → S8 | R3.4b `P(M)` | **3.2.3** |
| AE.4b labels | 4.1 / Tab. 4, 6.3 | R3.4c exponents | **3.2.3, 5.3** |
| AE.4c analysis | 4.5, 5.5 → S6 | R3.5a exclusions | 4.5 → S6 |
| AE.5 rationale | 2.3 | R3.5b per-dataset | 4.5, **5.4 inline** |
| R1.1 unfair comparison | **5.3 / Fig. 3**, 4.3, 5.2 prose → S8 | R3.5c bootstrap | 4.5 → S6 |
| R1.2a AGM, gSpan | **2.1** | R3.6a "standard construction" | 4.4, 2.2 |
| R1.2b five axes | **2.2 / Tab. 2** | R3.6b "strongly correlates" | abstract, 5.4, 7 |
| R1.3a density | 4.1, 5.4 *(inline)* | R3.7a limitations | **6.3** |
| R1.3b label premise | **5.4 item 8** | R3.7b delta subsection | **2.3** |
| R1.3c limitation impact | **6.3 item 4** | R3.7c schematic | **graphical abstract + S9** ⚠ |
| R1.3d labels future work | **7** | R3.7d three-way separation | **3.2.3, 5.3** |
| R3.1a(i) delta | **2.3 / Tab. 3** | R3.7e broad statements | 1, abstract, 7 |
| R3.1a(ii) sufficiency | **2.3 closing ¶** | EiC.a1–a4, EiC.b | refs, 2.1 |
| R3.1b "too absolute" | 1, 2.2 | EiC.c ≤ 35 p | §3 budget |
| R3.2 sequential model | **DECLINED** — 6.3 item 3, abstract | E1–E13 | per [corrections](corrections.md) |

---

## 12. Risks

1. **§3 does not shrink by 2.5 p.** The largest schedule risk in this file. **Mitigation**: E7 first,
   then the itemised cuts in §3's table in that order, measuring the page count at every commit.
2. **The supplementary/appendix distinction is wrong.** If a separate file *does* count, the
   pre-declared cut order in §10 is what executes. **The separate-file format is safe under both
   outcomes**, which is why P3 does not wait on the query.
3. **R3.2's decline is rejected by the reviewer.** The mitigation is §2.3's sufficiency paragraph and
   §6.3 item 3, conceded plainly. There is no second line of defence, and adding a token sequential
   experiment now would be worse than the concession — R3 would measure it.
4. **A retracted number reappears.** Five results in T-06 were retracted after being promoted, some
   twice. **Every number in the draft is checked against `T-06-article-notes.md` §10 before it is
   typed**, not after.
5. **A scoped claim loses its scope in editing.** The commonest failure mode and the one R3 checks
   for. §5's frozen wordings exist for this; use them verbatim rather than paraphrasing.
6. **The requested figure is illegible at final size.** §10.3 item 1 — the T-09 figures are drawn
   7.0 in wide into a 4.72 in text block. **Check Fig. 1 before anything else in §10**: it is the one
   figure a reviewer explicitly asked for, and shipping it unreadable is worse than not shipping it.
