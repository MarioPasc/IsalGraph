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

---

## 7. Handoff — where the revision stands, and what will waste your time

**Written 2026-08-31, at the close of the WL-integration and compression waves.** Everything below
was measured, not estimated. Re-measure before trusting any number in this section; it is a
snapshot, and this project has shipped a stale figure before.

### 7.1 State, and the one decision that is not yours

| artifact | pages | gates at close |
|---|---:|---|
| `review1/article` | **51** | exit 0 · 0 undefined · 0 float-too-large · 0 overfull > 5 pt · **46 cited** |
| `review1/supplementary` | **59** | exit 0 · 0 undefined |
| `review1/response` | **15** | exit 0 · **1** `\pending{}` — the page count, open by decision |

🔴 **51 pages against a hard 35, and compression is exhausted.** Every section measured its own
claim-free headroom with `\iffalse` or a clean-room rebuild; the total is **~0.3 p**. §1, §2, §6,
§7, the abstract and the back matter are each **measured page-neutral** — cutting prose there moves
no page because the section already shares its last page with the next one. **Do not re-derive
this.** The gap closes in §3, §4 and §5 or it does not close.

`review1/editor_query_DRAFT.md` is drafted, unsent, and **explicitly the authors' decision, not
ours**. Its page placeholder is now 51. The plan's own rule: *if the ladder runs out, the manuscript
asks the Area Editor before it drops a demand.* It has run out.

### 7.2 Read these two files before you type a number

- `.claude/notes/2026-08-30-article-wl/VERIFIED-NUMBERS.md` — every WL, spectral, sign-test and
  edit-path figure **recomputed from the primary artifacts**, with the derivation stated. **Where
  it disagrees with `prose.md`'s claim register, it wins**, and it records which claims failed.
- `.claude/notes/2026-08-30-article-wl/CONTRACT.md` — the frozen narrative for the two distance
  references, the red lines, and the compression directive. Corrected five times in one session;
  each correction is dated and reasoned in place.

### 7.3 The claim register is not trustworthy on its own

Four frozen claims carrying specific numbers were audited against primary data. **Two were wrong.**

| | verdict |
|---|---|
| **C13** | ✅ sound — but it printed `25.5×–108.6×` with no statement of what varied. It is **node count**, n = 6 to n = 10 |
| **C14** | 🔴 **wrong twice** — two medians on different denominators (write **0.914 → 0.428**, Suite-1 throughout), and its direction claim is only readable where the pair sets match |
| **C15** | ✅ verified exactly, cell by cell. Write it as frozen |
| **C17** | 🔴 **counts reproduce from no primary file.** `"+0.148"` and `"one and zero"` exist only in `prose.md`. Rewritten on measured values |

**Recompute any claim-register wording before using it.** The register is a drafting aid, not a
source.

### 7.4 Four traps that produce true numbers answering the wrong question

The project's own rule — *a ratio ships with its denominator* — was violated four times in one day,
each time by a different agent, each time with every individual number correct.

1. **Two campaigns, two pair sets.** The exact-GED column comes from the main T-06 campaign and the
   WL column from T-28. On `aids` and `linux` those are **different pair sets** (131,148 vs 295,296;
   1,685 vs 3,916), so the *direction* of the reference swap flips depending on which exact-GED
   measurement you read. The three IAM Letter rows are on identical pair sets. **Read a direction
   only where the pair sets match, and say so.**
2. **Point estimate vs paired bootstrap.** `t28_probe_point_estimates.json` is over each cell's own
   pairs; `t28_bootstrap_verdicts.json` is over the **intersected** set. They coincide only where
   every arm completes. **Never put both in one table row** — the manuscript shipped that defect.
3. **Do not derive the size null.** There is an authoritative field: `size_null_rho` in the probe
   file, `size_null.point` in `ged/data/rho_table.json`. Deriving it as `ρ − excess` mixes the two
   conventions above and is wrong on `aids` and `linux`.
4. **Cell counts vs dataset counts.** The WL bootstrap covers **14 cells**, not 14 datasets: four
   are Suite-1/Suite-2 duplicates and `suite2/mutagenicity` timed out. Distinct datasets = 10, and
   `wl/REPORT.md` §3.1 quotes medians over **11**. Say *cells*.

### 7.5 Measurement hazards — each one cost an agent real time

- 🔴 **A page count taken after `git restore` of the build artifacts is wrong.** This repo *tracks*
  `main.pdf`, `main.aux`, `main.log`, `main.fls`, `main.fdb_latexmk`. latexmk reuses the restored
  fingerprints and stops before convergence — it reported **49 for sources that build to 50**.
  **Measure only after `latexmk -C && rm -f main.bbl main.blg`.**
- **The `rtk` proxy truncates `grep -c` output.** It under-reported the bibliography by two entries.
  Use `/usr/bin/grep` for any count you intend to write down.
- **`summary_of_changes` must be built with `make`, not bare `latexmk`** — without its `TEXINPUTS`
  it dies on `preamble.tex` and poisons its own `.fdb_latexmk`, so the next `make` reports
  "Nothing to do" and exits non-zero on the stale error.
- **BibTeX does not treat `%` as a comment character.** A literal `@inproceedings` inside a comment
  breaks the bibliography.

### 7.6 Dead ends — measured, do not retry

- **Caption leading.** Single-spacing captions saves **nothing**. `\usepackage[font={stretch=1}]
  {caption}` *costs* a page and adds a float overflow by fighting elsarticle's own `\@makecaption`;
  a hand-patched `\@makecaption` changes the count by zero. Captions are already `\footnotesize`
  and `\raggedbottom` absorbs the slack.
- **Body leading is not available at all.** `\documentclass[review,…]` sets `\@blstr{1.5}` and the
  35-page limit is stated for a *double-spaced* manuscript, so the document is already **below**
  what the Guide asks. Tightening it breaches the submission format.
- **The author biographies are not a page source.** `backmatter.tex` carries the corresponding
  author's own instruction not to cut them, reinstating them is demand E12, and the user declined
  that cut when offered it. Worth ~1 p; **not available**.

### 7.7 The cross-reference layer is fragile and silent

Nothing here fails a build. All of it fails a reader.

- **`amsthm` shares one counter across §3.** Adding or removing any theorem-like environment
  renumbers everything after it. The response letter hard-codes those numbers in a macro block
  (`\thmInvariant`, `\propInvarianceFloor`, …) because **the letter cannot `\ref` into the
  article**. After any §3 edit, re-derive the block from `main.aux`:
  `grep -oE '\\newlabel\{[^}]*\}\{\{[^}]*\}' main.aux`.
- **Same for section numbers.** Inserting §4.3 pushed four subsection macros down by one; all four
  were stale and none broke a build.
- **The supplement hard-codes main-text float numbers** for the same reason. Two had rotted: S9's
  "Table 1" for the instruction set (it is Table 2) and S6's "Table 1" for the dataset grid (it is
  Table 3). **Sweep for these after adding any float**, and re-derive from `main.aux`.
- **The body now contains no algorithm float.** All four listings live in S9. `\algGreedy` expands
  to `S2`, not `1`. Do not send a listing to S7 — `s07_algorithms_complexity.tex:25` explains why.

### 7.8 A float can delete printed text without failing the build

`fig:edit-path` overflowed the text block by 147 pt, and the overflow was **not whitespace**: the
caption's closing sentence appeared in **no PDF ever built**, while the build exited 0. Detect it
with `pdftotext main.pdf - | grep <a phrase from the end of the caption>`, and treat any
`Float too large for page` in `main.log` as a possible text-loss event rather than a layout nit.
That figure is 215 × 515 pt, a 2.4:1 portrait; at any legible width it is a dedicated page.
**A wider-aspect re-render is the only real fix and it is a figure-generation task.**

### 7.9 Open items

1. **The page decision** (§7.1). Nothing else should be cut until it is made.
2. **Bibliography recency.** 46 entries clears EiC.a's floor of 35, but the most recent non-self
   entry is **2024**, and the EiC attached a delay consequence. Two or three 2025–2026
   pattern-recognition papers on graph matching or edit distance close both EiC answers. Verify
   every DOI against Crossref and record what was checked in a comment above the entry — that
   convention is already in `refs_added.bib`. **Both trees need the entry**: the article and the
   supplement have separate `refs_added.bib` files.
3. **Two supplementary promises are unmet**, flagged rather than papered over: `05_results.tex:653`
   points at Mantel tests printed nowhere, and `05:831` at per-cell *p*-values that are still
   absent (the enumeration itself landed).
4. **The `changes`-package blue version has never been produced for this content.** §2.3 fixes what
   to mark; the wave deliberately added no markup, and a note at the top of the blue version should
   declare the wholesale-new sections once rather than painting them blue.
