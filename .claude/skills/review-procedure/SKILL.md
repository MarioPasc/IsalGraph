---
name: review-procedure
description: |
  Scoping contract for the agents writing the IsalGraph revision (Pattern Recognition
  PR-D-26-03293): how to write the revised manuscript, how to mark it with the `changes`
  package so one source produces both a clean article and a blue-highlighted summary of
  changes, and how to write the response-to-reviewers letter afterwards. Fixes the order
  (article first, answers second), the markup convention, the response format, and the
  prose style. Triggers on "write the revised article", "draft section X of the revision",
  "write the response to R1.2", "answer the reviewer", "summary of changes", "blue version",
  "review-procedure", or any request to produce revision text for this manuscript.
---

# Review procedure — IsalGraph, Pattern Recognition major revision

**Not to be confused with `review-answer`**, which is the IsalSR/TPAMI skill and points at a
different repository, a different letter and a different directory layout. This one governs
IsalGraph.

---

## 0. Read before writing anything

| File | What you get from it |
|---|---|
| `.claude/notes/review/plan/prose.md` | **The architecture.** Section map, the H1–H4 spine, per-section briefs, the page budget, the artifact inventory, and the frozen claim register |
| `.claude/notes/review/plan/demands.md` | Every reviewer demand → decision → ticket → artifact. **The coverage contract** |
| `.claude/notes/review/source/mail.txt` | The decision letter. **Reviewer comments are quoted from here verbatim and never edited** |
| `.claude/notes/review/tasks/T-06-article-notes.md` §10 | **What is NOT claimable.** Read this before typing any number |
| `…/results/reports/T-13-complexity/T-13-FRAMING.md` §7 | The complexity red lines. **Note the path — this one is under `results/reports/`, not `.claude/notes/review/tasks/` like the others.** An agent sent to the tasks directory reports the file missing and drafts without the red lines |

Working tree: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199/review1/`

```
review1/article/              LaTeX sources + figures. THE ONLY COPY OF THE BODY TEXT.
review1/summary_of_changes/   a thin wrapper main.tex -> the blue version
review1/response/             response_to_reviewers.tex
review1/supplementary/        supplementary.pdf sources -- a SEPARATE document, never \appendix
```

---

## 1. The order is fixed: manuscript first, letter second

**Write the revised article. Then write the answers.** Not the reverse, and not in parallel.

The letter's job is to point at what the manuscript now says. Every response ends by naming a
section, a table or a definition, and quoting the sentence that landed there. You cannot write
that pointer before the text exists, and a letter drafted first turns into a promise the
manuscript then has to keep — which is how a response letter ends up describing a paper nobody
wrote.

Each response is drafted as its section closes, and the letter is assembled at the end. An empty
response for a demand in `demands.md` is a visible hole; a response written from intent rather
than from the page is an invisible one.

---

## 2. The `changes` package — one source, two PDFs

Reference implementation: `…/isalsr/article/journal/69c1637a28a81fea2badda9a/article/paper/main.tex`.

### 2.1 The preamble line

```latex
%% Blue version (summary of changes): marks visible.
\usepackage[commandnameprefix=ifneeded]{changes}

%% Clean version (the article as submitted): `final' strips every mark.
\usepackage[final,commandnameprefix=ifneeded]{changes}
```

Two things that are not optional:

- **`commandnameprefix=ifneeded` is required here**, because `main.tex` already loads
  `comment`, which owns `\comment`. Without it the build breaks on a name clash; with it only
  the clashing command is renamed and `\added` keeps its plain name.
- **Load `changes` after `hyperref`.**

### 2.2 One source, two wrappers

`review1/article/` holds every body file. `review1/summary_of_changes/main.tex` is a short
wrapper that sets the non-`final` option and `\input`s the same bodies by relative path.

**Never copy the body text into both directories.** Two copies drift, and the version that
drifts is always the one the reviewer reads.

### 2.3 🔴 What to mark, and this is the rule that decides whether the blue version is worth anything

**Mark the sentence a reviewer can check. Do not paint new sections blue.**

Sections 3.1 through 5 are rewritten wholesale. Wrapping all of that in `\added` produces a
document that is blue end to end and therefore says nothing — the reviewer cannot find the
change they asked for inside it. IsalSR marked a few dozen spans across the whole paper, not
thousands, and that is the target.

| Mark | Leave unmarked |
|---|---|
| A corrected number, exponent or count | A section that is new in its entirety |
| The clause that scopes a claim (`above n ≈ 20`, `within a fixed directedness class`) | Renumbered floats, moved text, reformatting |
| A restated theorem hypothesis | Prose rewritten for flow with no change of claim |
| A conceded limitation | Anything a reviewer did not ask about and no defect touches |
| The sentence answering a named comment | |

Sections that are new as a whole are declared **once**, in a short note at the top of the blue
version, listing them by number. That is more useful to a reviewer than a blue section and it
costs three lines.

### 2.4 Build check

Both PDFs must compile from the same sources before any response is written. If the clean
version does not build, the blue one is not evidence of anything.

---

## 3. Writing the manuscript

Follow `prose.md`. Beyond it, four standing rules:

1. **A scoped claim carries its scope in the same sentence.** *"Most compact of the canonical
   codes"* is fair. *"Most compact"*, with the qualifier moved to a limitations section, is not,
   and the difference is what a reviewer checks for.
2. **Every number is checked against its source file before it is typed**, not after.
   **A ratio carries its denominator in the same sentence** — three separate figures in this
   project have two correct values describing different comparisons, and the plan carried one
   without naming which. Five
   results in T-06 were retracted after being promoted; `T-06-article-notes.md` §10 lists them.
3. **Answer the reviewer before volunteering anything.** Where a self-found defect touches a
   comment, the comment is satisfied first on its own terms and the extra finding follows as
   *"in addressing this we also found…"*. Same content, opposite impression.
4. **The supplement is a separate PDF, never `\appendix`.** Pattern Recognition's page limit
   names appendices explicitly; a separately-uploaded file is outside it. Every supplementary
   section must be cited from the main text.

---

## 4. Writing the answers

Reference: `…/isalsr/…/reviews/response_to_reviewers.tex`. Lift its `rcomment` and `response`
environments; they are already the right shape.

### 4.1 Format

```latex
\begin{rcomment}{R1.2}
<verbatim from mail.txt, never edited>
\end{rcomment}

\begin{response}
<one paragraph>
\end{response}
```

**One paragraph per comment. No figures. No tables.** If an answer needs a figure, the figure
belongs in the manuscript or the supplement and the response points at it.

### 4.2 The anatomy of a response

Three moves, in this order, and then stop.

1. **Concede or agree, in one clause.** *"We agree."* *"The reviewer is right that we invoked a
   routine we never defined."* No preamble, no restating the comment back.
2. **Give the number or the fact.** The measurement that settles it. If the answer is a
   decline, say so plainly and give the reason once.
3. **Name where it landed.** Section, definition, table or algorithm number, and where useful
   the new sentence quoted in italics. This is what makes the letter checkable in the same
   direction the reviewer checked us.

Target length: **80–160 words.** A response past 200 words is usually restating the manuscript
instead of pointing at it.

### 4.3 What a response must never do

- Restate the comment before answering it.
- Thank the reviewer inside the response block. Thanks go once, in the letter's opening.
- Promise future work in place of an answer.
- Report a favourable number without the scope that qualifies it.
- Quote a result more favourably than it came out. The pre-registered layer is reported
  unchanged; softening one sentence forfeits the protection for all of them.
- Claim a change the files do not yet contain. If a claim is ahead of the sources, mark it
  `\pending{...}` so it renders red and cannot ship silently.

### 4.4 Declines

One comment is declined by decision (R3.2, the sequential-model experiment). A decline is
written as: what was asked, the decision, the reason in one sentence, and where the limitation
is now stated in the manuscript. **No defence, no compensating promise.** The concession is
already carried by §6.3 and the sufficiency paragraph in §2.3.

---

## 5. Prose style — the contract

Formal, concise, scientific. Academic *we*, active voice.

**Do not write:**

- **"It is not X, it is Y"** and its variants — *"this is not a caveat, it is a finding"*,
  *"not merely A but B"*. State Y.
- **Rule-of-three lists** used for rhythm rather than content.
- **Significance inflation**: *groundbreaking*, *novel*, *pivotal*, *crucially*, *remarkably*.
- **Synonym cycling.** Introduce a term once and reuse it. The canonical string is the canonical
  string in every sentence.
- **Vague attribution**: *studies show*, *it is widely known*.
- **Nominalisation**: *"we perform an evaluation of"* → *"we evaluate"*.
- **Copula avoidance**: *leverages*, *utilises*, *facilitates* → *uses*, *reduces*, *extends*.
- **Em-dash asides stacked** two or three to a paragraph.
- Hedges that carry no information: *arguably*, *it could be argued*, *to some extent*.

**Do write:** short declarative sentences with the number in them. *"Node-count difference alone
attains ρ = 0.71–0.997 against ground-truth GED."* Quantify rather than characterise.

**Do not over-detail.** A response is not a methods section. Give the number that settles the
question and the pointer, and let the manuscript carry the derivation.

---

## 6. Before you call it done

- [ ] Both PDFs compile from the same sources.
- [ ] Page count of the clean version is **≤ 35**. Measure it; do not estimate.
- [ ] Every reviewer comment in `demands.md` has a response, and every response names a location.
- [ ] Every number in the letter matches its source file, checked against
      `T-06-article-notes.md` §10 (in `.claude/notes/review/tasks/`) and `T-13-FRAMING.md` §7 (in `results/reports/T-13-complexity/`).
- [ ] Every reviewer comment is quoted verbatim from `mail.txt`.
- [ ] No `\pending{}` survives.
- [ ] Self-found defects appear after the reviewer answers, never before.
- [ ] The blue version marks checkable sentences, not whole sections.

**Then invoke `/humanizer` over everything you wrote** — the manuscript sections and the
response letter both. It does not change a claim, a number, a citation or an equation; it
removes the register that reads as machine-written. Run it last, after the numbers are verified,
so a rewrite cannot move one.
