# wave-abstract — Abstract (`00_abstract.tex`)

## What the section now says

The abstract runs the five moves in the order the brief fixes, one claim per move, and stops.

It opens by saying what IsalGraph *is* — a program for a nine-instruction virtual machine — and
immediately fixes the scope that R3.3a caught: decoding is total, encoding is defined for connected
undirected simple graphs and for directed graphs in which some node reaches every other. The
submitted *"any finite, simple graph"* is gone.

It then states completeness with its qualifier in the same sentence — *within a fixed directedness
class* — and backs it with the one unscoped positive number the paper has: zero collisions across
24,764,422 pairs at positive graph edit distance, over ten datasets, up to n = 98.

Compactness carries both its scopes inside the sentence that makes the claim (*among
representations whose canonical form is intrinsic to the code*, *above n ≈ 20*), and the concession
that canonically-labelled edge-list serialisations beat us at scale follows in the next sentence,
not in a later section.

Cost is stated as two facts rather than one: size sets how many frames the encoder emits, and at
fixed size the branching is governed by |Aut(G)|. The controlled ladder result supplies the
evidence. This is the phrasing the T-13 red line requires; *"governed by |Aut|, not by size"* is
never written.

Fidelity closes the abstract on the negative result, unhedged. Node-count difference alone attains
ρ = 0.71–0.997 against the reference and beats the canonical string on 17 of 25 records; the
conclusion drawn is about the benchmark, and it convicts every representation including ours; and
above n ≈ 40 nothing tested is distinguishable from ρ = 0. The submitted closing sentence about
language-model compatibility and downstream applications is deleted, not paraphrased.

## Number provenance — every number I wrote

| Number as printed | Source file | Where in it |
|---|---|---|
| nine-instruction | `03_method.tex` | l. 61, *"a string that a nine-instruction virtual machine"*; alphabet at l. 103 |
| ten datasets | `04_experimental_design.tex` | l. ~114, *"lists the evaluation cohort: ten datasets"* |
| `n = 98` | `04_experimental_design.tex` | l. ~125, *"Suite~2 carries it to $n = 98$"*; retained ceiling at l. 135 |
| `24{,}764{,}422` | `05_results.tex` | l. 106, §5.1 (C1 verbatim). Split 3,424,764 Suite-1 + 21,339,658 Suite-2, ll. 107–109 |
| `n \approx 20` | `05_results.tex` | l. 127–128, §5.2 (also `01_introduction.tex` l. ~78) |
| `112` of `112` | `05_results.tex` | l. 128 and l. 166 |
| `12` ladder cells | `05_results.tex` | l. 200–202, §5.3; repeated in the Fig. caption l. 271 |
| positive in `11` | `05_results.tex` | l. 202 (line-wrapped as `positive` / `in $11$`); caption l. 272 |
| `+0.892` | `05_results.tex` | l. 202 and l. 273 |
| `p = 0.0064` | `05_results.tex` | l. 202 and l. 273 |
| `\rho = 0.71$--$0.997` | `05_results.tex` | l. 339, §5.4 (C5 verbatim) |
| `17` of `25` | `01_introduction.tex` | l. 93, exact wording reused. Same fact in `05_results.tex` l. 304–305, phrased *"falls below its own size null on $17$"* |
| `n \approx 40` | `05_results.tex` | l. 398 and l. 437 (C6) |

Every row was checked by `grep -F` against the file in this session after the final edit. No number
was taken from prose.md, from a sibling agent's draft, or from memory.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| **B1**, **E5** | sentences 1–2 | *"Every string over the alphabet decodes to a graph, and every connected undirected simple graph (or directed graph in which some node reaches every other) encodes to a string that decodes back to it up to isomorphism."* |
| **R3.3a** | sentence 2 | same sentence — replaces the submitted *"any finite, simple graph"* |
| **R3.3b** | sentence 3 | *"Within a fixed directedness class the canonical string is a complete invariant"* |
| **B6** | sentences 4–5 | the compactness claim carries both scopes inline, and the edge-list concession follows immediately |
| **R3.6b** | sentences 8–10 | *"strongly correlates with graph edit distance"* deleted; replaced by the size-dominance result and C5's conclusion |
| **AE.1** | final sentence | *"Within a fixed node count, above n ≈ 40, not one of the representations tested is reliably distinguishable from ρ = 0."* |
| **R3.2**, **B4** | by deletion | *"language-model-compatible … with direct applications in graph similarity search, graph generation, and graph-conditioned language modelling"* removed and not paraphrased |

## Measured

- **Length: 244 words**, counted from the rendered PDF (`pdftotext -f 1 -l 1 | wc -w`), not from the
  source. Under the 250 cap; above the ~200 aim — see *Decisions*. Submitted abstract was 213 words.
- **Compile: clean.** 0 undefined citations, 0 undefined references, both taken **after** the bibtex
  step the CONTRACT's build-trap note prescribes (`BIBINPUTS=… bibtex main` in the outdir, then a
  second latexmk pass). Before that step the build reported the stale-`.bbl` failure the note
  describes.
- **Overfull boxes: 1 in the whole document**, 1.90 pt, `while \output is active` — under the 5 pt
  threshold and not introduced by me (it is in the output routine, not in the abstract block).
- Whole manuscript builds to **41 pages** at this point in the wave. Over the 35-page limit, but the
  other two sections are still live and this is not the abstract's to fix.
- Stylometry (`humanizer/scripts/stylometry.py`): median sentence length 20.5 (band 10–24),
  sentence-length CV 0.419 (≥ 0.40), function-word ratio 0.416 (0.40–0.55), nominalizations 54.3
  (≤ 60), participial clauses 0, transition openers 0, filler phrases 0, em-dash pairs 0, negative
  parallelism 0, flagged adverbs 0.

## Decisions and assumptions

**1. Length: 244, not 200.** Getting to ~200 requires cutting one of three things, and each is a
concession against us or a frozen wording: the edge-list-serialisation concession (9 words), C5's
*"for every representation, ours included"* tail, or the 17-of-25 tally. Cutting a concession to
save words moves the abstract in the one direction the plan forbids. I cut everywhere else instead —
the median-+215-bits figure came out, and move 5 was restructured to name `node-count difference`
once rather than twice — and stopped at 244.

**2. The 17-of-25 tally is a sixth claim and I removed it once, then restored it.** The brief says
*"five moves, in this order, and nothing else"*, and the tally is not among them. I cut it on that
reading. The orchestrator's mid-task message endorsed keeping it (as the safest available handling
of C3, since giving the tally without the 8-record split means the abstract cannot inherit that
register's attribution error), so it is back — **without** *"significantly"*, which the orchestrator
showed has no source and which neither §1 nor §5 uses.

**3. C3's 8-record split is deliberately absent.** The abstract states the loss (17 of 25) and not
the partial win (8 clears). Stating the split correctly costs ~25 words; stating it incorrectly is
the error the register was amended to prevent. Omitting it understates nothing in our favour.

**4. `\Aut` is used inside the abstract.** The abstract carries one math macro, `$|\Aut(G)|$`, plus
`$\rho$` and `$n$`. Elsevier discourages heavy math in abstracts, but the cost claim is not statable
without |Aut(G)| and the submitted abstract already used `$\Sig$` and `$\wstar_G$`.

**5. No `changes` markup**, per the spawn prompt. The abstract is rewritten wholesale, so under
`review-procedure` §2.3 it belongs in the blue version's opening note as a wholly-new section rather
than painted blue. The three sentences a reviewer will actually diff are named in the file header so
the markup pass does not have to re-derive them.

## For the orchestrator

**1. 🔴 `T-13-FRAMING.md` does not exist.** `prose.md` §5 names it twice as authoritative for the
cost red lines (*"Condensed from `T-06-FRAMING.md` §6, `T-06-article-notes.md` §10 and
**`T-13-FRAMING.md` §7**, which remain authoritative"*), and `review-procedure`'s required-reading
table lists `T-13-FRAMING.md` §7 as well. `.claude/notes/review/tasks/` contains only
`T-13-design.md`. The red lines survive in prose.md §5.3's own brief, which is what I used, so
nothing was lost here — but this is the same dead-pointer class as the `docs/references/` path in
CLAUDE.md, and the next agent told to read it will find nothing.

**2. 🔴 prose.md §4's abstract brief contradicts prose.md §5.3's red line.** The brief's move 4 says
*"Cost. Governed by |Aut(G)|, not by `n`."* — which is verbatim the sentence §5.3 flags as **wrong**
(*"size sets the frame count and it matters"*). The brief does append *"Read prose.md 5.3's three red
lines before writing this sentence"*, and my spawn prompt caught it explicitly, but an agent working
from §4 alone writes the banned sentence. §4's move 4 should be restated in the §5.3 form.

**3. 🔴 The thesis paragraph contains a clause its own source contradicts, and it is in our favour.**
prose.md §1 reads *"Its edit distance tracks graph edit distance where structure is what varies and
size is not"*. C9 measures the opposite: holding the generator fixed and **adding structural
distortion**, the size baseline stays flat at ρ ≈ 0.92 while the canonical string falls from 0.93 to
0.67. §5's IAM Letter control table agrees — LOW/MED/HIGH gives 0.9278 / 0.8833 / 0.6660 for the
string against 0.9139 / 0.9146 / 0.9195 for the size null. Where structure is what varies and size
is not, the string gets **worse**, and the baseline does not. I kept the clause out of the abstract.
**wave-discussion and wave-conclusions are the ones at risk**: the thesis paragraph is exactly what
a discussion or conclusion section paraphrases, and this clause reads as a licensed positive.

**4. `frontmatter.tex` carries a TODO addressed to whoever lands the abstract.** It says the four
submitted highlights are to be rewritten against the claim register *"when the abstract lands"*, and
names two as out of scope — *"complete graph invariant"* (needs the directedness-class qualifier,
R3.3b) and *"approximates graph edit distance"* (the R3.6b claim). They are commented out, so they
do not ship as-is and no build is broken. I do not own that file and did not touch it. The abstract
has now landed; someone owns those four lines.

**5. Elsevier highlights and the graphical abstract are both commented out** in `frontmatter.tex`
and submitted separately through Editorial Manager. Flagging only so it is a decision rather than an
oversight.
