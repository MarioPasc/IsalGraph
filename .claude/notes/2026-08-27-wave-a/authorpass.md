# wave-authorpass — the corresponding author's eight comments

Applied 2026-08-27 to
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199/review1/article/`.
Baseline on entry: **39 pages**, clean build. On exit: **42 pages**, clean build.

Plus one mid-turn instruction from the author, logged as item 9.

---

## 1. Abstract rebuilt on the traditional structure

`00_abstract.tex`, whole file rewritten (body at lines 39–68).

Four paragraphs: background and problem / aim and methods / results in H1–H4 order /
conclusion. The previous "five moves" organisation is gone, and the header comment
records the replacement.

**267 → 264 prose tokens after the humanizer pass** (stylometry counter; each math span
counts as one token, so a journal word-counter will differ slightly). Target was 200–250;
I stopped at 264 because every remaining sentence carries a red-line-protected claim and
the next cut would have removed one. First draft was 329.

Red lines checked one at a time and all held: no *"strongly correlates with graph edit
distance"*; no *"language-model-compatible"* and no applications sentence; the |Aut|
wording is the frozen *"Size sets how many frames the encoder emits, and at fixed size the
branching is governed by |Aut(G)|"*; compactness carries **both** its scopes in-sentence
(class and `n ≈ 20`); the R3.3a scope sentence and the R3.3b directedness-class clause are
intact.

## 2. The hypotheses no longer contain results

`01_introduction.tex` lines 89–113 (the `enumerate`) plus the two bracketing sentences.

Every verdict, headline number and conclusion came out of H1–H4. Each item now states the
hypothesis and names the section that answers it, by `\ref` throughout. The H1–H4 spine
stays, because §5.1–5.4 each open by naming their hypothesis and §7 refers back to all four.

The closing tally *"Two of the four hypotheses are refuted"* also came out of §1. It
survives in §7, which is where a summary belongs. What replaces it in §1 is a posture
sentence with no verdict in it: *"Where a hypothesis does not survive, we report the
refutation and do not soften it."*

Header comment at lines 7–40 now inventories which number left §1 and which section still
prints it, so nothing reads as lost.

## 3. Over-engineered counts, and the H3 reframe

### The counts

| Was | Now | Exact count survives in |
|---|---|---|
| `112 of 112 strata` (§5.2, abstract, §1 ×2) | *"in every size stratum we tested above that point"* | caption of `fig:information-content`, `05_results.tex:197` |
| `12 ladder cells … positive in 11` (§5.3, abstract, §1) | *"positive in all but one"* | caption of `fig:cost-law`, `05_results.tex:297–300` |
| `17 of 25 records` (§5.4, §7, abstract, §1) | §5.4 states the **complement**: *"clears its own size null on eight"* of *"the twenty-five records"*; §7 *"on most of the twenty-five records"* | §5.4's own prose, `\supp{6}` |
| `19 of 19` / `17 of 19` fits | *"every identifiable fit"* / *"all but two of them"* | derivable in-sentence: 25 attempted − 4 − 2 = 19 |
| `18 of 20 cells` | *"all but two cells"* | caption of `fig:cost-law` |
| `4 of the 5`, `7 of 10` | spelled out: *"four of the five"*, *"seven of the ten"* | unchanged, count kept |

**What this put at risk and how it is held.** Removing a count can remove a denominator,
and the plan's denominator rule says the denominator ships in the same sentence. Two
protections: (i) every replacement phrase carries the sweep *and* its scope
(*"in every size stratum we tested above that point"* is both), and (ii) the exact count
moved into the figure caption rather than out of the paper. `112 of 112` had been *trimmed
out of* the `information_content` caption on 2026-08-27 precisely because the prose printed
it; I put it back, and left a comment saying the prose no longer carries it so it must not
be trimmed again.

C3's `8 = 1 + 7` decomposition is untouched — it is frozen and amended, and it is what
makes the eight clears honest.

### H3

`05_results.tex:329–331`. Was *"We refute it at scale."* Now:

> H3 claims Levenshtein distance between canonical strings approximates graph edit
> distance. Scoped to small graphs it holds; at scale it does not, and where the one
> becomes the other is the result this subsection reports.

This is the construction §5.2 already uses for H2 (*"Scoped, it holds; unscoped, it
fails"*), so the paper is now internally consistent in how it frames a scoped hypothesis.
§7 matches: *"H3 and H4 fail in the form stated. Levenshtein distance … tracks graph edit
distance at small size and stops doing so as graphs grow."*

**The three protected items are untouched**: the boundary itself, the size-baseline
comparison (*"four of the five Suite-1 datasets, worst by −0.4597 on AIDS"*), and the
field-level statement (*"within a fixed node count above n ≈ 40, none is reliably
distinguishable from ρ = 0"*, scope in-sentence). §5.4 still ends *"H3 is refuted"*.

**This is where the author's comment and the plan disagreed.** `prose.md` §4's brief for
§5.4 says the result is *"conceded first, no bracket argument"* and *"No framing repairs
this and none is attempted."* The author asked for a scoped positive. I took the reading
that satisfies both: the concession is still the second clause of the first sentence, so
nothing is buried, but it is stated as a boundary rather than as a defeat. Nothing was
softened and the word *refuted* still appears twice.

## 4. Comparison table simplified

`tab_representation_properties.tex`, rewritten. Removed: the `ψ` column, the `scalability`
group (`n_max`, `compl.`), and `downstream learning`. Eleven columns → seven. The note is
now a paragraph under the tabular rather than a `\multicolumn` spanning row, because with
seven narrow columns a full-width spanning cell inflates the last column instead of the
table.

**What this put at risk and how it is held.** R1.2 named five axes explicitly and the table
now has columns for two. §2.2's prose (`02_related_work.tex:117–147`) was rewritten into
two paragraphs that name **all five** — uniqueness, expressiveness, computational
efficiency, scalability, downstream learning — and say where each is answered. That walk is
now the discharge of R1.2b, and both the section header comment and the table header
comment say so in those words, with an instruction not to compress it without restoring the
columns. The caption repeats the three pointers.

What the deleted columns conveyed, in one sentence each:
- ψ → *"Measured instead as sensitivity to relabeling, every representation in the table
  scores zero except the raw adjacency string, which scores 0.07–0.74 and leaves the
  running comparison for that reason."*
- downstream learning → *"Downstream learning we do not evaluate, for any representation
  including ours, and Section~\ref{sec:limits} states that as a limitation of this paper
  rather than of the representation."* (verified: §6.3 does carry it)
- scalability → pointed at `sec:res-cost`, with AGM CAM's `n = 12` ceiling / 6.15 % Protein
  floor and min-DFS's 0.9478 kept in prose, because they scope every pooled comparison
  against those two arms.

Nothing else `\ref`s a deleted column. Checked: `\psi` appeared in body prose exactly once,
in the sentence I rewrote.

## 5. Graphical abstract is now a numbered figure

`03_method.tex`. Float at lines 458–478, `\label{fig:worked-example}`, placed in §3.3.1
because that is where Reviewer 3 asked for the search-space schematic. Cited from §3.1.3
(line 129), §3.2.2 (line 189) and twice in §3.3.1 (lines 447, 450). All four
*"Figure S1 of \supp{9}, also the graphical abstract"* indirections are gone.

The caption describes what is actually drawn — I rendered the PDF and looked at it rather
than describing it from the surrounding prose: six start-node subtrees, red branches at
`V` and blue at `v`, the shaded forced region, leaf lengths with the shortest filled, the
highlighted canonical path, and the G2S/S2G panels with the `π`/`σ` pointers and the string
`VVVnvPCPV`.

**One deviation from the author's spec, deliberate.** The file carries a 176 pt footer band
with institution logos, an author byline and a project URL. That belongs to the Editorial
Manager graphical abstract and not to a numbered figure in the body, so the float clips it:
`trim=0 176 0 0,clip`. Measured — the rule sits 174.9 pt above the bottom of a
3764.41 × 1505.2 pt page. Clipped ratio 2.83:1, so at `\textwidth` the graphic is 120 pt of
a 550.28 pt block. **To revert, delete the trim/clip options; nothing else depends on
them.** The author's "2.5:1, about a quarter page" describes the unclipped file.

`frontmatter.tex`: the commented `graphicalabstract` environment is untouched as instructed.
I updated only the comment above it, which claimed the graphic was reproduced as Fig. S1.

## 6. Suite 1 total row

`tab_datasets.tex`. Two total rows now, Suite 1 above Suite 2, each carrying its pair count
and its reference type in a `\multicolumn{5}` over the columns that are empty on a total row
anyway:

```
Suite 1 (n ≤ 12) │ 1 │ │ 5,350  │ │ │ │ 12 │ 3,897,911 pairs   reference GED: exact
Suite 2 (no cap) │ 2 │ │ 16,370 │ │ │ │ 98 │ 21,710,892 pairs  reference GED: proven bracket
```

The exact-versus-bracket distinction is also the **first sentence of the caption**, in bold:
*"the distinction that governs how every result in this paper is read: Suite 1 carries exact
ground-truth graph edit distance and Suite 2 does not."* The note keeps the AIDS (GraphEdX)
769-graph delta, which is the only per-dataset difference between the suites.

## 7. Section 4 now has its formulas

Eight displayed equations, all `\label`ed. Every symbol defined where it first appears, and
every symbol checked against §3 for collision (`Γ`, `W`, `L`, `B`, `K`, `δ`, `i*`, `r_i`,
`s_i` are all free; `Σ` is `\Sig`, §3's own macro).

| Label | Content |
|---|---|
| `eq:cost-model` | `c(e)` and GED as a min over edit paths `Γ(G,H)`; cross-referenced to `def:ged` |
| `eq:bracket` | `GED_LB ≤ GED ≤ GED_UB` |
| `eq:bracket-width` | `W_abs = UB − LB`, `W_rel = (UB − LB)/UB` |
| `eq:bit-conventions` | `L log₂|Σ|` (entropy bound, primary) and `8B` (realised) |
| `eq:counting-floor` | `log₂ C(C(n,2), m) − log₂ n!` |
| `eq:payload` | `L log₂|Σ| / B`, the payload-bits-per-stored-byte ratio |
| `eq:spearman` | Pearson on midranks |
| `eq:size-null` | `ρ_size` and the excess |
| `eq:mrm` | the standardised regression |
| `eq:bh` | Benjamini–Hochberg step-up, `q = 0.05`, within family |

Cited from later sections: `eq:size-null` and `eq:mrm` from §5.4, `eq:counting-floor` from
the `information_content` caption (which used to print the formula inline).

**I did not write any of these from memory.** Four definitions I would otherwise have
guessed were verified against the code first, and one of my assumptions was wrong:

- `W_rel` denominator is **UB**, not the midpoint —
  `benchmarks/real_data/eval_setup/approx_ged_analysis.py:276–316`.
- Spearman is **hand-rolled Pearson-on-midranks**, tie-exact, not `scipy.stats.spearmanr` —
  `benchmarks/real_data/eval_setup/ged_bakeoff_analysis.py:332–352`.
- MRM: response **and** predictors z-scored (ddof=0), **no intercept** because both sides
  are centred — `benchmarks/real_data/eval_stats/association.py:498–510`.
- BH: within each family separately; families of 5, 10, 182, `N_MAX = 197` —
  `benchmarks/real_data/eval_stats/family.py:3–19,146,149`.
- **Corrected assumption**: the payload ratio is the cohort median of
  `8*entropy_bits/realised_bits`, not `entropy_bits/B`. Since `realised_bits = 8B` the two
  agree, and for the instruction string `B = L`, so the ratio is `log₂ 9 = 3.1699` — which
  is exactly the 3.17 §4.4 already printed. That arithmetic is now stated in the text as a
  check a reader can run.

**Measured cost: exactly 2 pages.** Isolated by restoring the pre-change
`04_experimental_design.tex`, building, recording 40, restoring mine, rebuilding to 42, and
confirming the restored file was byte-identical to mine.

## 8. §5.5 deleted

`05_results.tex`. The subsection and its four-row table are gone. What survives is the
closing paragraph of §5.4, lines 487–495:

> H2 and H3 were also tested under a confirmatory family enumerated and frozen before any
> p-value existed, and we report it as it came out. Of its 79 cells, 75 reject at q = 0.05
> under Benjamini–Hochberg. A rejection is against H₀: Δ = 0 and can equally mean
> significantly worse, so the count is readable only split by direction: of the 69
> directional rejections, 35 favour IsalGraph and 34 go against it. The pre-registered layer
> establishes that the differences it tests are real, and not that they favour the
> representation. The enumeration, the per-cell p-values and the Benjamini–Hochberg tables
> are in `\supp{6}`.

**What this put at risk and how it is held.** AE.4c asks for the *analysis* of the
pre-registered results, and §4.5 only states the protocol — so without this paragraph AE.4c
would be discharged nowhere in the paper. The §5 header inventory now says exactly that,
with the instruction that cutting the paragraph un-discharges the demand.

The **split by direction travels in the same sentence as the rejection count**, because a
bare "75 reject" is the reading `T-06-article-notes.md` §10 forbids in as many words. I also
kept the denominator (79 cells) rather than quoting 75 bare. Per-family rows (A1 51/28/23,
A2 1, B1e 18/7/11, B3e 5) are in `\supp{6}` and named in the source comment.

Nothing outside §5.5 referenced `sec:res-prereg`. Checked before deleting.

## 9. Author biographies restored — mid-turn instruction

Not one of the eight. The author sent *"Dont cut the author biographies"* while I was
working.

I had not touched `backmatter.tex` — it was byte-identical to the snapshot. But the
biographies were **already commented out**, by a page-ladder pass earlier the same day
(`backmatter.tex:81`, *"WITHDRAWN AGAIN 2026-08-27, page ladder"*, measured cost 0.721 p).
So the state violated the instruction even though I had not caused it.

Restored, lines 96–115 uncommented. This also discharges **E12**, which the withdrawal note
itself records as *"reinstating them was right in principle"*. Comment updated to say the
block must not be commented out again for page count, and that the acknowledgements are not
a substitute candidate because grant conditions commonly require that text verbatim.

## 10. Formatted-submission preview copy — mid-turn instruction

Author asked for a copy of `article/` under `article_formatted_submission/` on the standard
Elsevier template, "so that we can see how it is looking under the final visualization",
and said not to worry about page count.

Created at
`…/review1/article_formatted_submission/`, **18 pages, A4, 0 errors, 0 undefined
references, 0 undefined citations, 1 overfull box (14.12 pt)**. Class swapped from
`elsarticle[review,times,number]` to `elsarticle[final,5p,times,twocolumn,number]` — `5p`
is Elsevier's final two-column layout — and the custom 4.8 cm-margin `geometry` override
disabled, since it exists to force the review text block and fights the class's own final
geometry.

A bare class swap was not enough to be worth looking at: it produced 10 overfull boxes,
three at exactly 270 pt. Cause is that in two-column, `\textwidth` spans **both** columns
while a `figure`/`table` is one column wide. Fixed by making the four wide tables `table*`
and the three `width=\textwidth` figures `figure*`, and switching the biography minipages
to `\columnwidth`. `complexity_by_aut` stays a single-column `figure` — it is placed at
natural size (227 pt) and fits.

One equation needed splitting: `eq:cost-model` set the `cases` block and the `min` side by
side with `\qquad`, 93 pt too wide for a column. Split into two displays, the second
labelled `eq:ged-min`. **Preview-only label; nothing references it.**

**Graphical abstract**: on the author's follow-up, it is rendered here by elsarticle's own
`graphicalabstract` environment rather than only as a body figure. It occupies page 1 as
Elsevier typesets it, with title and authors above the graphic, and uses the **untrimmed**
file, because the logo/byline/URL band belongs to a graphical abstract. Verified by
rendering page 1. The body figure `fig:worked-example` from item 5 is still present as
well, so the graphic appears twice in this copy — once as the page-1 graphical abstract and
once as the §3.3.1 figure. In `../article/` the `graphicalabstract` environment stays
commented out, exactly as author comment 5 instructed.

`README.md` in that directory lists all seven deltas, marks the copy non-authoritative, and
gives the rsync line to regenerate it.

**Proved layout-only.** Eleven files differ from `../article/`. After normalising
`table*`/`figure*`/`\columnwidth` and stripping comments, seven are byte-identical, and the
remaining four differ only in: the equation split, the enabled `graphicalabstract`, the
removed `geometry` line, and the class options. No claim, number, citation or `\ref` target
differs between the two directories.

---

## Verification

**Build**, from the article directory, `latexmk -pdf -interaction=nonstopmode main.tex`:

| Gate | Result |
|---|---|
| exit status | 0 |
| pages | **42** (baseline 39) |
| undefined citations | 0 |
| undefined references | 0 |
| font warnings | 0 |
| float too large | 0 |
| overfull boxes > 5 pt | 0 |
| `\pending` | 0 |
| LaTeX errors | 0 |

Bibliography was not rebuilt by hand and no citation key was added, so the `-outdir` bibtex
trap in the contract does not apply — this is an in-place build against the correct `.bbl`.

**Numbers.** Two passes, both scripted against the snapshot at
`scratchpad/pre-user-comments/`, comments stripped:

1. Per-file numeric diff. Every removal is accounted for by one of the eight items; every
   addition is either equation notation (`n^{2}`, `\binom{n}{2}`, `log_{2}`, `n(n-1)`), a
   verified constant (182 family size, log₂ 9, the 176 pt trim), or a restored biography
   year.
2. Paper-wide significant-value diff. Every value that lost occurrences still appears at
   least once. **The only two values gone entirely are `0.061` and `0.948`** — both rounded
   cells of the deleted `compl.` column, whose facts are in §2.2 prose at *higher* precision
   (`6.15 %` of Protein, floor `0.9478`).

Spot-confirmed still in place after the humanizer pass: `24,764,422`; `ρ = 0.71–0.997`;
`+0.892` / `p = 0.0064` in both §5.3 prose and the caption; `112 of 112` in the caption;
`79 / 75 / 69 / 35 / 34`; `four of the five` / `twenty-five` / `eight` / `seven`.

**Humanizer** run last, over the abstract and §1's hypotheses only, after the numbers were
checked. Three-item coordinations 3.91 → **0.00**; sentence-length CV 0.401 (ok);
negative parallelism 0. Nominalizations stay flagged HIGH (75.76 vs ≤ 60) and I left them
deliberately: the residue is the paper's fixed terminology — completeness, compactness,
fidelity, isomorphism, canonicalisation, serialisations — and `review-procedure` §5 forbids
synonym cycling in as many words. Hedges and parentheticals read LOW for the same reason:
the house style is short declarative sentences with the number in them, and the skill's own
rule is not to add hedges to a calibrated claim. Changes made: `equality of matrices` →
`comparing two matrices`; the closing triad `complete, invertible and compact` unpacked into
a scoped two-part sentence; the repeated pseudo-cleft in H2/H3 (`… is what Section X
measures`) converted to active `Section X measures …`.

---

## For the orchestrator

- **The one deviation to review is the 176 pt trim** on the graphical-abstract float
  (item 5). Body figures in an Elsevier article do not carry logos, an author byline or a
  promotional URL, so I clipped the band. It also changes the ratio the author quoted, from
  2.5:1 to 2.83:1. One-line revert.
- **Page count is 42 against Pattern Recognition's 35.** Not something this pass could fix:
  the author's own comments are net additive (+2 measured for item 7, +1 for item 5), and
  item 9 restored 0.72 p. The reductions available to me (items 1, 2, 3, 8) were taken and
  are worth about 1 p together. The appendices-count/supplementary-doesn't rule means the
  remaining overrun has to come out of the body or move to `\supp{}`.
- **Author comment 3 versus `prose.md` §4's §5.4 brief** is the only place the author
  contradicted the plan. Resolved in favour of both, as described under item 3 — concession
  still first, tone reframed. Worth a second opinion from whoever owns `prose.md`.
- `0.061` and `0.948` no longer appear as table cells. Deliberate, and their facts are in
  §2.2 prose, but if the frozen comparison table is meant to be reproducible cell-for-cell
  from the artifacts, that mapping is now one step indirect.
