# related — Section 2, Related work and positioning

## What the section now says

A graph has no intrinsic vertex order, so every serialisation has to pick one of the `n!`
orderings. §2.1 walks the four answers the literature gives. nauty/Traces pick one by
individualisation–refinement and hand back a permutation, which is why a serialisation becomes
invariant only by running *after* it; graph6 and sparse6 are the serialisations that follow, and
because each writes one specific vertex ordering we evaluate them only in their nauty-canonicalised
forms — the P4 sentence prose.md §9 makes load-bearing, with the 1-in-50 invariance failure behind
it. Frequent-substructure mining builds the key into the code instead (AGM's canonical adjacency
matrix, gSpan's minimum DFS code), and both search over orderings, which Babai locates in the
problem rather than in any encoder. Weisfeiler–Lehman escapes the search by giving up completeness
and invertibility. A separate line computes GED on the graphs directly and returns no reusable
encoding, which is why GED is this paper's reference rather than a competitor.

§2.2 fixes the family taxonomy first — canonical code, canonicalised serialisation, feature map,
declared baseline — stating that the grouping predates any measurement, then walks R1.2b's five
axes verbatim off the frozen table. It concedes in place: IsalGraph rejects disconnected graphs
where four other rows accept them; on scalability it ties rather than leads; downstream learning
reads *not evaluated* for every row including ours.

§2.3 introduces the frozen delta table (nine components, three inherited, five modified, one new),
states the attribution rule and its conservative tie-break, names the single new component as the
isomorphism-invariance theorem with its metric corollary folded into the same row, then the table,
then the frozen sufficiency paragraph, with nothing after it.

## Number provenance — every number I wrote

| Number as printed | Source file | Where in it |
|---|---|---|
| `1 of 50 relabelings` | `tab_representation_properties.tex` | caption: *"every representation that fails does so at $1/50$ relabelings"* |
| `183{,}016` comparisons | `tab_representation_properties.tex` | caption: *"the collision column is the false-isomorphism-certificate rate over 183,016 comparisons"* |
| `$\psi = 0$` | `tab_representation_properties.tex` | ψ column, the CC and CS rows |
| `$0.07$--$0.74$` | `tab_representation_properties.tex` | ψ column, `adjacency` row |
| `$2.5\times 10^{-4}$` | `tab_representation_properties.tex` | collis. column, `WL subtree` row |
| `$n = 12$` | `tab_representation_properties.tex` | $n_{\max}$ column, `AGM CAM` row |
| `fifteen dataset cells` | `tab_representation_properties.tex` | caption: *"compl. its completion floor over fifteen dataset cells"* |
| `$6.15\,\%$` of Protein | `.claude/notes/review/plan/prose.md` | §7: *"completes on 6.15 % of Protein"* |
| `$0.9478$` | `.claude/notes/review/plan/prose.md` | §7: *"`min_dfs` has a completion floor of 0.9478"* |
| `nine components … three, five and one` | `T-07-article-notes.md` §2 + `tab3_prior_work_delta.tex` | *"Nine components: 3 inherited, 5 modified, 1 new"*; the table's three grouped row blocks |

`$n!$` and `$n^{2}$` are notation, not measurements. No other number appears in the section.
Nothing was inferred from another number and nothing came from memory.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| AE.2 | §2.1 (`sec:canon-lit`) | the whole subsection: canonicalisation, serialisation, mining codes, feature maps, direct GED |
| AE.3 | §2.2 (`sec:comparison`) | *"Table~\ref{tab:representation-properties} reports five axes."* plus the axis-by-axis walk |
| AE.5 | §2.3 | the frozen sufficiency paragraph — T-07 records the "rationale" clause as delivered there |
| R1.2a | §2.1 | *"Apriori-based Graph Mining uses the canonical adjacency matrix… `gSpan` uses the minimum DFS code, whose length grows with the edge count rather than with $n^{2}$"* — both cited individually, each with a comment |
| R1.2b | §2.2 | the five axes named verbatim; *"\emph{Downstream learning} reads \emph{not evaluated} for every row including ours."* |
| R3.1a(i) | §2.3 | *"…breaks the present system into nine components and marks each one inherited, modified or new: three, five and one."* |
| R3.1a(ii) | §2.3 | `\input{sufficiency_paragraph}`, closing the section |
| R3.1b | §2.2 | *"The table is also what the four properties of Section~\ref{sec:introduction} are stated against, and on its axes the canonicalised serialisations match \IsalGraph{} and exceed it on disconnected input."* |
| R3.6a | §2.2 | *"Each representation enters the distance experiments under an edit distance on its own emitted string, sharing one alphabet of edit operations…"* and *"We compare against established reversible serialisations rather than against a reference construction of our own"* — the second branch of the reviewer's `or`, taken |
| R3.7b | §2.3 | the dedicated subsection exists |
| EiC.a4 | whole section | 13 `\cite` commands, zero groups, each with its own comment |
| EiC.b | §2.1 | the direct-GED sentence discusses three *Pattern Recognition* entries that were in `cas-refs.bib` and cited nowhere |
| prose.md §9 (P4 hole) | §2.1 | *"…both serialise one specific vertex ordering, so an edit distance between them measures node ordering rather than structure unless a canonical labelling is applied first; we therefore evaluate them only in their `nauty`-canonicalised forms, the raw forms failing our isomorphism-invariance filter at 1 of 50 relabelings (\supp{4})."* |

## Measured

- **Section length: 4.14 p** — prose **2.738 p** measured + floats **1.404 p** (0.671 + 0.733, the
  orchestrator's figures; T-07's independent 368.4 pt for tab3 corroborates 0.671 at
  `\textheight` 550.276 pt).
  Method: `scratchpad/measure_prose.sh` builds an isolated document — preamble plus
  `02_related_work.tex` with both `\input{tab…}` lines removed — so no float placement and no peer
  section can contaminate the span, and reads `\thepage`/`\the\pagetotal` at the section's first
  and last line. In-document spans are unusable here: float pages inflate them.
- **Target 3.05 p, tolerance ±0.30. I am 1.09 p over and I could not close it — see below.**
- Compile: **clean. 0 undefined citations, 0 undefined references, 0 overfull boxes in my file.**
  Document is 30 pages. **The citation figure was taken after running `bibtex` in my out
  directory**, per the coordinator's instruction; before that run the same build reported 22
  undefined citations, none of them real.
- Bibliography: 25 entries emitted, my five among them, all rendering correctly. The Inokuchi DOI
  underscore compiles without escaping under `elsarticle-num.bst`.
- Humanizer: run last, as four surgical edits. Ledger re-checked by diff — **13 `\cite` commands,
  11 `\ref`/`\supp` commands and every number byte-identical before and after.**

## Every `refs_added.bib` entry with its verification

All five verified this session against a primary source that was actually fetched. None
reconstructed from memory. No `TODO-` citation was left anywhere in the section.

| Key | Verified against | Identifier |
|---|---|---|
| `mckay2014practical` | `https://api.crossref.org/works/10.1016/j.jsc.2013.09.003` | doi `10.1016/j.jsc.2013.09.003` |
| `mckay2022graph6` | `https://users.cecs.anu.edu.au/~bdm/data/formats.txt` (live) | url, no DOI exists |
| `yan2002gspan` | `https://api.crossref.org/works/10.1109/ICDM.2002.1184038`; pages from `https://dblp.org/db/conf/icdm/icdm2002.html` | doi `10.1109/ICDM.2002.1184038` |
| `inokuchi2000agm` | `https://api.crossref.org/works/10.1007/3-540-45372-5_2`; LNCS vol. 1910 from `https://dblp.org/db/conf/pkdd/pkdd2000.html` | doi `10.1007/3-540-45372-5_2` |
| `shervashidze2011wl` | `https://www.jmlr.org/papers/v12/shervashidze11a.html` (live) | url; JMLR issues no DOI |

Two notes carried from the verification. The **nauty User's Guide was rejected** as the graph6
citation: `formats.txt` is the normative definition, the Guide only summarises it in §20, and the
Guide's URL is version-tied (`nug29.pdf` → `nug30.pdf`) while `formats.txt` is stable. The Guide's
current version is also **2.9.3 dated December 2025**, not 2024, so any key named `mckay2024nauty`
would carry a wrong year.

## Decisions and assumptions

1. **Kept the direct-GED sentence** citing `Lerouge2017ilp`, `Fischer2015hausdorff` and
   `Riba2021gedgnn`. All three are *Pattern Recognition* entries already sitting in `cas-refs.bib`
   and cited by no section. EiC.b is T-19's demand, not mine, but my scaffold says §2.1 is where
   the additions get *discussed* rather than listed, and the sentence also answers R1.1's
   conceptual half — why GED is our reference and not a competitor. It costs 0.169 p, measured.
2. **Did not restate C2** (the compactness claim) in §2.2. It is a §5 result, and restating it
   would duplicate a number I would then have to keep in sync. §2.2 stays at the level of
   properties and points at `sec:res-compactness`.
3. **Did not write "no row satisfies all four properties."** It is false against the frozen table:
   `nauty-graph6` and `nauty-sparse6` carry every tabulated property, and IsalGraph does not
   (disconnected input). R3.1b is discharged by saying so plainly and pointing at the table.
4. **Efficiency split across two figures.** The frozen table's note maps R1.2b's efficiency axis
   onto `fig:information-content` alone, which is encoded *size*. I point at `fig:cost-law` for
   encoding cost and `fig:information-content` for encoded size, because efficiency has both
   readings and collapsing them would be the imprecise choice.
5. **Wrote "so on this axis it ties"** rather than "which is a tie and not a lead". The second is
   the forbidden *"it is not X, it is Y"* shape. Same content, allowed form. It also keeps distance
   from T-06 §10's red line *"IsalGraph computes everywhere, unlike the competitors."*
6. **Assumed §1 will state four properties** and named them as *"the four properties of
   Section~\ref{sec:introduction}"*. §1's brief requires them; if wave-intro lands on a different
   count, that clause needs one word changed.

## For the orchestrator

**1. The §2 page budget cannot be met, and the plan's own arithmetic shows why.**

The `review` class body sets at ~345 words per page (25 lines, `\@blstr{1.5}`, `\textheight`
550.276 pt). Against that:

- **The frozen sufficiency paragraph is 141 words and measures 0.393 p.** `prose.md` §4 budgets it
  at ~0.1 p. That is a **3.9× error**, measured by building the section with and without it. It is
  the plan's own paragraph, whose word count the plan itself states.
- So the discretionary prose budget is `3.05 − 1.404 (floats) − 0.393 (sufficiency) = 1.25 p`,
  about **430 words**, to carry: six works each cited individually with a comment, the P4 sentence,
  a four-family taxonomy, five axes walked with their measurements, and a nine-component table
  introduction. Those items do not compress below ~800 words.

I wrote the section three times, each shorter: 3.67 p of prose → 3.11 p → **2.738 p**. The last is
at the floor. Measured cut options, if you want them:

| Cut | Section becomes | Cost |
|---|---|---|
| — (as delivered) | **4.142 p** | — |
| A: drop the direct-GED sentence | 3.973 p | EiC.b loses its discussion; 3 PR entries go back to being uncited |
| A + B: also drop the executability pointer | 3.907 p | a reviewer reading the table has no answer to "what does IsalGraph have that this table does not show" until §6.1 |

**Even A+B lands at 3.91 p, still 0.86 p over.** Closing to 3.35 p means dropping a demand —
realistically the §2.1 survey (AE.2/R1.2a) or the five-axis walk (AE.3, endorsed by the Area
Editor in their own voice). I am not willing to make that call; it is a PI decision. **I recommend
re-deriving the §2 budget from the measured 345 words/page rather than trimming §2**, and checking
whether the same conversion error inflates the other section budgets — §1 is 896 source words
against a 2.2 p target, which is also over on this constant.

**2. The bibtex trap, confirmed independently and now fixed in my out directory.** I hit it before
your message arrived and reached the same diagnosis. Two additions to what you sent: the stale
`main.bbl` in the article directory carried **4** bibitems when I looked, not 3, so it is being
rewritten by somebody's build as the wave proceeds — it is a moving target, which makes an
undefined-citation list taken without the bibtex step actively misleading rather than merely
uninformative. After `BIBINPUTS=<article>: bibtex main` in my out directory, the list is empty and
the bbl holds 25 entries.

**3. Not mine, but visible in the build.** `LaTeX Font Warning: Font shape 'OT1/txr/m/scit'
undefined, using 'OT1/txr/m/sc' instead` fires around pages 9–18, at line 635 of a §3 file. Zero
occurrences in my isolated build. Likely `\textsc{}` inside an italic context in `03_method.tex`.
Cosmetic, but it is a shape substitution in a submitted PDF.

**4. `backmatter.tex` has an Overfull \vbox of 75.93 pt.** Pre-existing, not mine, flagged because
nobody in this wave owns that file.

**5. Five entries added to the printed bibliography.** EiC.a1's 35–55 window is T-26/T-08/T-19's to
reconcile. Four of my five are the literal subject of R1.2a and AE.2 and cannot be dropped without
leaving those demands undischarged; `shervashidze2011wl` is the one with slack, though
`weisfeiler1968reduction` alone does not cover the subtree kernel the comparison actually uses.
