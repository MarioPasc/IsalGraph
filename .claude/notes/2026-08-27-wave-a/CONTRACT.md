# Wave A — the shared contract

Every agent in this wave reads this file first. It is the part of your instructions that is
identical across the three sections; your spawn prompt carries the part that is yours alone.

You write one section of a journal manuscript under major revision. You are not writing a draft
for someone to polish; what you produce is what ships to Pattern Recognition.

## The two things that get a paper rejected here, and they are both yours to prevent

**1. A number that is not traceable to a named file.** Five results in this project were retracted
after being promoted into prose, some twice. Every number you type must be copied from a file you
opened in this session, and you record where it came from. No number from memory. No number
inferred from another number. No number from a sibling agent's draft.

**2. A claim that loses its scope.** *"Most compact of the canonical codes"* is fair. *"Most
compact"*, with the qualifier moved to a later section, is not — and that difference is exactly
what Reviewer 3 checks for. A scoped claim carries its scope **in the same sentence**, always.

## Before you write a word

1. Invoke `/review-procedure`. It is the scoping contract: order of work, markup, prose style.
2. Read `.claude/notes/review/plan/prose.md` (**v1.4**) — §4's brief for *your* section, §5's claim
   register and its red-line table, and §2's hypothesis spine.
3. Read `.claude/notes/review/tasks/T-06-article-notes.md` §10, **What is NOT claimable**, in full.
4. Read whatever else your spawn prompt names as required reading.

## Rules that bind you

- **You own exactly one file.** Everything else in the manuscript is read-only, including the
  tables, the figures, `cas-refs.bib`, and your peers' section files. If you believe a file you do
  not own is wrong, message the orchestrator; do not edit it.
- **Never write a literal float number.** Not "Table 2", not "Figure 1". Always `\ref{...}` against
  the labels your spawn prompt lists. Print order is not the plan's numbering and the two do not
  match.
- **Answer the reviewer before volunteering anything.** Where a self-found defect touches a comment,
  satisfy the comment first on its own terms; the extra finding follows as *"in addressing this we
  also found…"*. Same content, opposite impression.
- **Concede where we lose, in the same subsection as the win.** Never in a later one.
- **The frozen wordings in prose.md §5 are used verbatim.** They were measured, argued, and in
  several cases retracted before reaching that form. Do not paraphrase them.
- **The supplement is a separate PDF, never `\appendix`.** Cite it with the `\supp{n}` macro.
- You cannot ask the user anything. If you are blocked or must assume something, record the
  assumption, message the orchestrator, and keep going.

## Prose style

Formal, concise, academic *we*, active voice. Short declarative sentences with the number in them.

Do not write: *"it is not X, it is Y"* or its variants; rule-of-three lists used for rhythm;
significance inflation (*groundbreaking*, *novel*, *pivotal*, *crucially*, *remarkably*); synonym
cycling — the canonical string is the canonical string in every sentence; vague attribution
(*studies show*); nominalisation (*"we perform an evaluation of"* → *"we evaluate"*); copula
avoidance (*leverages*, *utilises*, *facilitates* → *uses*, *reduces*, *extends*); stacked em-dash
asides; hedges that carry no information.

When your section is drafted and its numbers are verified, invoke `/humanizer` over it. Run it
**last**, after the numbers are checked, so a rewrite cannot move one. It changes no claim, number,
citation or equation.

## Definition of done

1. Your file compiles as part of the whole manuscript, into **your own output directory**.
2. No undefined reference, no undefined citation you did not deliberately leave and log, no overfull
   box over 5 pt introduced by you.
3. Your section is within ~0.3 p of its page target, **measured**, not estimated.
4. Every demand your spawn prompt assigns you is discharged by text on the page.
5. Your work log is written.
6. You have sent the orchestrator your final report.

## Your work log — write it, it is not optional

`/home/mpascual/research/code/IsalGraph/.claude/notes/2026-08-27-wave-a/<your-slug>.md`

```markdown
# <slug> — <section name>

## What the section now says
<8-15 lines: the argument in order, not a list of edits>

## Number provenance — every number I wrote
| Number as printed | Source file | Where in it |
|---|---|---|
(one row per number. A row you cannot fill is a number that must come out.)

## Demands discharged
| Demand | Where it landed (label or subsection) | The sentence that discharges it |

## Measured
- section length: X.XX p (how you measured it)
- compile: clean / N undefined citations, listed below

## Decisions and assumptions
<anything you had to decide without being told; anything you assumed>

## For the orchestrator
<what you could not do, what you need, what you think is wrong in the plan>
```

---

## Build trap: `-outdir` never runs bibtex

*Found by wave-intro, 2026-08-27; reproduced by the orchestrator.*

`latexmk -pdf -outdir=<dir>` writes `main.aux` into your out directory but **does not run bibtex
there** — no `.bbl` and no `.blg` appear in it. LaTeX then falls back to the stale shared
`main.bbl` in the article directory, which holds the handful of entries from the empty scaffold.
A build in that state reported **22 undefined citations**, every one of them a key that exists in
`cas-refs.bib`.

**Do not "fix" a citation key that reports undefined until you have run:**

```
cd <your outdir>
BIBINPUTS=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199/review1/article: bibtex main
cd /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199/review1/article
latexmk -pdf -interaction=nonstopmode -outdir=<your outdir> main.tex
```

What survives that is a real missing key; what disappears was never missing. Two consequences for
your report: state that your undefined-citation list was taken **after** a bibtex run, and note
that a page count taken against a 3-entry bibliography is short by most of the reference list.

## Frozen by Section 1, and it binds every later section

The **four properties** are `{validity, reversibility, canonicity, compactness}`. The submitted
"structure-preserving" is renamed **locality** and is *not* in the set: it is a measured quantity,
carried by **H3**, because listing it as a property IsalGraph *has* would contradict §5.4 in the
same paper — which is the exact defect B6 exists to stop. Use that set and those names.
