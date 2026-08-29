# intro — Section 1, Introduction

## What the section now says

The adjacency matrix is the standard form and it fails on three counts: `Θ(n²)` space, no
sequence model reads it, and it is tied to a node ordering. That third failure is named
correctly — the matrix is permutation-*equivariant* (`M → P M Pᵀ`), and what it lacks is a
distinguished representative among its up-to-`n!` matrices. Deciding isomorphism needs
invariance. From there the section motivates a string, and then narrows to the actual move:
IsalGraph encodes a graph as a **program**, not a description — nine instructions driving a
virtual machine, every prefix itself a valid program that executes to a subgraph, so the
representation is generative as well as descriptive.

Then the four properties, stated once and named: **validity, reversibility, canonicity,
compactness**. Locality is named as a fifth and explicitly demoted — we measure it rather than
claim it — which is how the submitted paper's "structure-preserving" desideratum survives
without being asserted. The absolute claim is replaced by "we do not claim that no other
representation has the four", pointing at the comparison table.

Scope follows, positively: decoding total, encoding partial, connected undirected or directed
with a root reaching every node, and canonicity within a fixed directedness class.

The section then states H1–H4 with their verdicts and headline numbers, and closes with two
sentences saying two of the four are refuted and neither refutation is softened. Roadmap names
Sections 2–7.

## Number provenance — every number I wrote

| Number as printed | Source file | Where in it |
|---|---|---|
| `Θ(n²)` space | `original/introduction.tex` | line 14, "occupies $O(N^2)$ space regardless of graph sparsity" |
| `n!` adjacency matrices | — | count of permutations of `n` labels; arithmetic, not a measurement |
| nine instructions / nine-instruction alphabet | `original/introduction.tex` line 48; `review1/article/tab_instruction_set.tex` | "Nine instructions move the pointers through the CDLL…"; the table lists N,n,P,p,V,v,C,c,W |
| 24,764,422 GED-positive pairs, zero collisions | `plan/prose.md` §5 | **C1**, verbatim |
| above `n ≈ 20` | `plan/prose.md` §5 | **C2**, verbatim |
| 112 of 112 strata | `plan/prose.md` §5 | **C2**, verbatim |
| median +215 bits | `plan/prose.md` §5 | **C2**, verbatim |
| 17 of 25 records | `plan/prose.md` §5 | **C3**, verbatim |
| one undetermined, remaining 7 favour the string | `plan/prose.md` §5 | **C3**, verbatim |
| ρ = 0.71–0.997 | `plan/prose.md` §5 | **C5**, verbatim |
| above `n ≈ 40`, ρ = 0 | `plan/prose.md` §5 | **C6**, verbatim |
| 12 ladder cells | `plan/prose.md` §5 | **C11**, verbatim |
| 71 orders of magnitude | `plan/prose.md` §5 | **C11**, verbatim |
| positive in 11 | `plan/prose.md` §5 | **C11**, verbatim |
| median +0.892 | `plan/prose.md` §5 | **C11**, verbatim |
| sign test p = 0.0064 | `plan/prose.md` §5 | **C11**, verbatim |
| the five search-free representations | `plan/prose.md` §5 | **C11**, verbatim |
| fold-change 1.0–1.1× | `plan/prose.md` §5 | **C11**, verbatim |

All seventeen re-verified by string match against `prose.md` after the humanizer pass. No number
moved.

**One frozen wording is rendered short.** C5's middle clause, *"exceeding 0.96 on seven of ten
Suite-2 datasets"*, is not printed in §1 — the opening and closing clauses are verbatim. §5.4
must carry C5 in full.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| **R3.7e** (equivariance → invariance) | ¶1 | *"The property that fails there is \emph{invariance}: relabelling by a permutation matrix $P$ sends $M_G$ to $P M_G P^{\top}$, so the adjacency matrix is permutation-equivariant, and a graph on $n$ nodes has as many as $n!$ of them with none distinguished."* |
| **R3.1b** ("no existing method satisfies all four" too absolute) | ¶4, four-properties | *"We do not claim that no other representation has the four. Table~\ref{tab:representation-properties} compares \IsalGraph{} with established representations on measured axes, and where a competitor holds a property the table says so."* |
| **B6** (two different four-property sets) | ¶4, four-properties | *"Four properties are asked of a sequential graph encoding, and we use these names for them throughout: \emph{validity} … \emph{reversibility} … \emph{canonicity} … and \emph{compactness} …"* plus *"A fifth property, \emph{locality}, asks that small structural differences produce small differences between strings; we measure it rather than claim it, as H3 below."* |
| **B1 / R3.3a (partial)** | ¶5, scope | *"Encoding is partial, and is defined for connected undirected simple graphs and for directed graphs in which some node reaches every other; a disconnected graph, or a directed graph with no such root, has no \IsalGraph{} string."* preceded by *"Decoding is total: every string over the nine-instruction alphabet executes and yields a graph."* |
| **E5 (partial)** (abstract self-contradiction: correlation asserted, then limited) | H3 item | The correlation claim and its limitation are in the same item: *"We refute this at scale: on $17$ of $25$ records…"* followed by *"On these benchmarks the reference itself is size-dominated…"*. §1 never asserts an unqualified correlation, so the contradiction has no site here. |
| **R3.3a, directedness half** | ¶5, scope | *"Canonicity holds within a fixed directedness class, because a string records the edges it creates but not whether they are directed (Section~\ref{sec:canonicalization})."* — a pointer only; R3.3c's full discharge stays with §3.3.3 / T-22. |

## Measured

- **Section length: 2.361 p.** Measured, not estimated: `\typeout{\thepage/\the\pagetotal/\the\pagegoal}`
  inserted immediately after `\section{Introduction}` and at end of file, compiled, then
  `(3 − 1) + 466.0/550.27614 − 241.81194/497.77611 = 2.361`. Markers removed afterwards; the
  shipped file contains no `\typeout`. Target 2.2 p, tolerance ±0.3 → inside.
  Prose tokens: 816. Submitted §1 for comparison: 988 words, 3.0 p.
- **Compile: clean.** `latexmk -pdf -outdir=<scratch>/build/intro main.tex`, exit 0.
  Zero undefined references, zero undefined citations, zero overfull hbox over 5 pt.
  Whole document 30 p at the time of my last build (peers still writing).
- **Undefined citations left deliberately: none.** No `\cite{TODO-…}` was needed — §1 uses only
  keys already in `cas-refs.bib`: `zhou2020gnn`, `kipf2017gcn`, `vaswani2017attention`,
  `weininger1988smiles`, `krenn2020selfies`. Five citations, each attached to its own clause, so
  EiC.a4's "uncommented citation groups" does not apply.
- **Humanizer:** `scripts/stylometry.py` passes every band except **Hedges (LOW, 1.23 against a
  3–20 target)**. Not corrected, deliberately — raising it means adding hedges to calibrated
  claims, which the wave CONTRACT bans and which the humanizer's own rule 4 bans. Three edits
  came out of the pass and are in the file: a three-clause coordination in ¶1 split into two
  sentences; the agentless *"A fifth is often stated beside them"* replaced by *"A fifth property,
  \emph{locality}, asks that…"*; and — the one that mattered — my invented causal sentence
  *"The cause lies in the benchmarks."* replaced by C5's own opening, *"On these benchmarks the
  reference itself is size-dominated:"*. That was a claim-strength escalation over the frozen
  wording and it is now gone.

## Decisions and assumptions

1. **B6, the four-property set — the decision that binds the conclusions agent.** The submitted
   paper has two incompatible sets: `introduction.tex:33` = {compact, reversible,
   structure-preserving, canonicalisable}; `conclusion.tex:74` = {universal validity,
   reversibility, canonical completeness} — three members, and *universal validity* appears in
   neither the other set nor the intro. I unified them as **validity, reversibility, canonicity,
   compactness**, and named the fifth member of the union, *structure-preserving*, as **locality**,
   demoted to a measured quantity (H3). Nothing from either submitted set is silently dropped.
   The rationale: locality is exactly what H3 refutes, so listing it as a property IsalGraph *has*
   would contradict §5.4 in the same paper — which is the defect B6 exists to stop. **§7 must use
   this set and these names.**
2. **The Contributions enumerate is gone.** prose.md calls H1–H4 "the spine … the contribution
   list", so H1–H4 replaces it rather than sitting beside it. That is most of the −0.8 p.
3. **H4's wording avoids the T-13 red line.** prose.md §5.3 forbids *"governed by |Aut|, not by
   size"*. §1 prints: *"size sets how many frames the encoder emits, and at fixed size the
   branching is governed by $|\Aut(G)|$"*. I did not write "milliseconds" for the |Aut| cost —
   prose.md §2 says it but names no source file, so §1 says only "cheap to compute".
4. **No `changes` markup added.** No peer file carries `\added`/`\replaced` either, so I assume a
   dedicated markup pass owns it. The four sentences that pass review-procedure §2.3's test
   ("a sentence a reviewer can check") are, in file order:
   - ¶1: *"The property that fails there is \emph{invariance}: … permutation-equivariant …"* (R3.7e)
   - ¶4: *"We do not claim that no other representation has the four."* (R3.1b)
   - ¶5: *"Encoding is partial, and is defined for connected undirected simple graphs and for
     directed graphs in which some node reaches every other …"* (B1, R3.3a)
   - ¶5: *"Canonicity holds within a fixed directedness class …"* (R3.3b/c scope)

   A header comment in `01_introduction.tex` repeats this list so the markup pass need not
   re-derive it.
5. **Positioning kept out.** SMILES and SELFIES appear in one half-sentence as motivation for
   strings, with no survey and no comparison; all of that is wave-related's. I dropped my first
   draft's separate cheminformatics paragraph for exactly this reason.
6. **No new bib key.** `refs_added.bib` is untouched.

## For the orchestrator

1. **`main.bbl` in the shared article directory is stale and a `-outdir` build silently uses it.**
   `latexmk -outdir=…` writes `main.aux` into the out directory but does not run bibtex there; LaTeX
   then picks up the source directory's `main.bbl`, which at the time I started held 4 entries.
   My five citations reported as undefined until I ran
   `cd <outdir> && BIBINPUTS=<articledir>: bibtex main` by hand. **Any agent measuring with
   `-outdir` will see phantom undefined citations.** Worth putting in the CONTRACT.
2. **prose.md §4's §1 brief conflates two lists and does not say so.** It asks for "the four
   properties, softened and unified" *and* "the four claims as testable hypotheses" in the same
   section, but the property set and the H-set are not the same four — compactness and locality
   are both, validity and reversibility are properties only, and cost is a hypothesis only. I
   resolved it as decision 1 above. **The conclusions agent needs the same resolution or B6
   reopens in §7.** This is the one thing in the plan I'd call underspecified rather than wrong.
3. **prose.md §4's §1 brief says to point R3.1b at "Tab. 2".** I pointed at
   `\ref{tab:representation-properties}` per the label list in my spawn prompt, which is the same
   artifact. Recording it because the plan's float handles and print numbers do not match.
4. **§5.4 owes C5's middle clause.** §1 prints C5's first and last clauses verbatim but not
   *"exceeding 0.96 on seven of ten Suite-2 datasets"*. §5.4 should carry C5 whole.
5. **Nothing I need.** No blocker, no assumption I could not record.
