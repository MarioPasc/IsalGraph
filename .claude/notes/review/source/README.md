# PR-D-26-03293 — Major Revision: source notes

Reference documentation for agents and authors working the IsalGraph revision.

The directory holds two kinds of file and **they must not be mixed**. The *evidence* files record
what the reviewers said and what is verifiably true in the sources, and propose nothing. The
*response* files carry the author decisions, the experiment designs and the schedule. When the two
disagree, the evidence files win and the response file is wrong.

Manuscript: *Representation of Graphs by Sequences of Instructions* (IsalGraph).
Journal: **Pattern Recognition** (Elsevier).
Decision received **2026-08-10**. Revision due **2026-08-31** (21 days).

Manuscript source root:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`
Decision letter: `.claude/notes/review/source/mail.txt`

## Files in this directory

### Evidence — what was said and what is verifiably true

| File | Content |
|------|---------|
| `mail.txt` | The decision letter as received. Source of every verbatim quotation in this package. |
| `00-editor-and-decision.md` | Editor-in-Chief and Area Editor comments, submission mechanics, deadline, required elements. |
| `reviewer-1.md` | R1 — 3 numbered comments (R1.1–R1.3). |
| `reviewer-3.md` | R3 — 7 numbered comments (R3.1–R3.7), several with sub-parts. |
| `manuscript-map.md` | Where every section, theorem, definition, algorithm, table and figure lives; cross-walk between the numbers reviewers cited and the actual `\label`s. |
| `verified-discrepancies.md` | Every factual claim by a reviewer, checked against the `.tex` sources and the code, with `file:line`, marked CORRECT / INCORRECT / PARTIALLY CORRECT. Plus discrepancies no reviewer caught. |
| `codebase-pointers.md` | Where the implementations, datasets and result files are, for anyone who has to re-run or re-measure. |

### Audit history

| File | Content |
|------|---------|
| `gap-audit.md` | The 2026-08-11 coverage audit that produced the traceability matrix: 10 unowned demands and 16 flawed or infeasible locked decisions, with severities and evidence. |
| `../../audit-2026-08-11b/` | The over-scope and integrity re-audit — demand inventory rebuilt from `mail.txt` alone, four voice slices, 24 integrity defects — **and `third-auditor.md`, the audit of that audit.** |

### The response moved — see [`../plan/`](../plan/)

**This directory no longer holds the plan.** As of 2026-08-12 the response documents
(`plan.md`, `data.md`, `statistics.md`, `competitors.md`, `labels.md`, `manuscript.md`) were
refactored into one file per edge of the proposal under **[`../plan/`](../plan/)**, and removed from
here. Git history holds the originals.

**Start at [`../plan/README.md`](../plan/README.md)**, or go straight to
[`../plan/tickets.md`](../plan/tickets.md), which names per ticket exactly which files to read.

The evidence/response separation stated above still governs: when the two disagree, **the evidence
files in this directory win** and the plan file is wrong.

## There is no Reviewer #2

The decision letter contains **Reviewer #1** and **Reviewer #3** only, plus the **Area Editor**
and the **Editor-in-Chief**. There is no `reviewer-2.md` and none is missing. The letter's own
prose says "**both** the reviewers" (`mail.txt:55`, `:59`, `:66`), confirming two reviewers, not
three. Reviewer #3's numbering is an Editorial Manager artefact — a third reviewer was invited and
their report is not part of this decision.

## Who said what

Pattern Recognition uses no structured review form, so there are no per-criterion ratings to
tabulate. What the letter does carry:

| Source | Length | Register | Substance |
|---|---|---|---|
| Editor-in-Chief (Zoran Duric) | 3 lettered items, `mail.txt:124–130` | Checklist, enforced pre-acceptance | Bibliography size and composition, page limit, source files |
| Area Editor | 6 paragraphs, `mail.txt:52–70` | Substantive, agenda-setting | Graph size, related work, side-by-side comparison, experiment design |
| Reviewer #1 | 3 numbered comments, `mail.txt:73–79` | Constructive; opens "The paper is interesting" | Unfair runtime baseline, missing canonicalisation related work, label loss on AIDS |
| Reviewer #3 | 7 numbered comments, `mail.txt:83–116` | Detailed and technically accurate; opens by naming the strengths | Novelty delta, no sequential-model experiment, scope overclaims, algorithm/complexity defects, statistics, interpretation, limitations |

**What is already conceded, verbatim.** Useful because it bounds what is *not* in dispute:

- Area Editor, `mail.txt:55`: "this paper presents a potentially useful contributions for
  representing graphs as sequences for use in analysis and sequential processing."
- R1, `mail.txt:73`: "The paper is interesting as it opens up new research directions in
  sequential graph-string representations."
- R3, `mail.txt:83`: "The manuscript's main strength is its extension of prior instruction-based
  representations to generic graphs through a sparse CDLL construction and relabeling-invariant
  canonicalization. Its deterministic decoding, reversibility, complete-invariant claim, open
  implementation, and speed-quality trade-off are also valuable. The overall objective of
  developing a reversible sequential representation of graph topology is clear."

No reviewer disputes the core construction. Every comment is about scope of claims, missing
comparison, missing measurement, or defects in how the work is described.

## Comment taxonomy

All 14 reviewer/editor comments grouped by **the kind of work each implies**, not by who raised it.
Nothing here prescribes a response. Identifiers are defined in the per-reviewer files.

**New measurement / experiment required**
- R3.2 — no sequential-model (Transformer/LSTM) experiment anywhere, despite LM compatibility being
  a stated motivation. The single heaviest request in the round.
- R1.1 / AE.4 — runtime is compared only against exact GED; no method addressing a similar problem
  setting is run as a comparator.
- R1.3 / AE.4 — the AIDS degradation is attributed to density, but label loss is an uncontrolled
  confound; no experiment separates them.
- R3.5a — number of pairs removed by the `GED > 0` / `Lev > 0` filter is never reported per dataset.
- R3.5c — the bootstrap of Section 4.3 is never described and is not shown to operate at graph level.
- AE.1 / R3.7 — behaviour at graph sizes beyond ~12 nodes is not characterised.

**Theory / formal writing**
- R3.3 — `S2G` determinism depends on the `directed` flag, which is not part of the string.
  The complete-invariant theorem is therefore stated more broadly than it holds.
- R3.4b — whether `P(M)` is recomputed or precomputed is unstated, and pair scanning, pointer
  walking, neighbour checks and backtracking are absent from the complexity discussion.
- R3.7 — theoretical worst case, search behaviour and empirical fit are conflated and must be
  separated.

**Framing / honesty of claims**
- R3.1 — "no existing method satisfies all four properties" is too absolute without a systematic
  comparison.
- R3.6b — "strongly correlates" with GED is not uniform: `rho ~ 0.43` (LINUX), `~ 0.35` (AIDS).
- R3.6a — the "GED standard construction" is an author-defined reference model, not an established
  baseline; compactness claims hold only relative to it.
- R3.7 — "any finite simple graph" / "arbitrary graphs" / "super-polynomial" / adjacency-matrix
  permutation equivariance all need narrowing or correcting.

**Factual corrections (all checked — see `verified-discrepancies.md`)**
- R3.4c — `n^{9.0}` in the conclusion versus `n^{4.9}` in the results; and a degree-4.9 polynomial
  described as "super-polynomial".
- R3.4a — Algorithm 2's `C`/`c` guards contradict Table 1. The **manuscript pseudocode is wrong**;
  the implementation is correct.
- R3.3 — the `directed` flag is external metadata, confirmed empirically and more strongly than
  the reviewer alleged.
- R3.5b — IAM uses uniform edit costs; LINUX and AIDS use topology-only costs. The aggregate figure
  pools all three.

**Related work**
- R1.2 / AE.2 — AGM canonical adjacency matrices and gSpan DFS codes are not discussed.
- R3.1 / AE.3 — a detailed side-by-side comparison against IsalChem [29] and the preprint [28],
  identifying inherited / modified / new components, is requested by both.
- R3.7 — a dedicated subsection comparing IsalGraph with IsalChem and the previous graph
  instruction method.
- EiC.a / EiC.b — recent pattern-recognition work must be cited; arXiv citations replaced.

**Prose / presentation**
- R3.7 — a schematic of the canonical search space in Section 2.3.
- AE.4 — "more detailed and rigorous analysis ... and in the associated analysis of the results."

## Read this before scoping any response — the R1.3 lesson

**What happened, 2026-08-11.** Reviewer 1's comment 3 devotes most of its word count to node and edge
labels, so the first scoping pass read it as a demand for a label experiment and designed one: a
label-aware GED arm under a second cost model, with its own results subsection, tables and page
budget. That was wrong on three counts, and each was checkable in minutes. First, **the volume of
text about a topic is not the size of the ask about it** — the operative request sits in the
comment's last two sentences ("*A more thorough **discussion** of this limitation, along with its
impact on the reported results, would strengthen the paper*"), and it asks for prose. Second, **the
manuscript already satisfied two of the three asks**: the limitation is discussed at
`conclusion.tex:70–71` and the future-work direction at `:81`, so the only genuine gap was the
"impact on the reported results" clause — one paragraph joining §5 to §4.

> ⚠ **This lesson has a defect of its own, found 2026-08-12.** Both sentences it recommends pointing
> at are **false**: `conclusion.tex:70` and `:81` claim labels are "present in all five benchmark
> datasets", which is **wrong for LINUX** (self-found defect **E6**). Answering R1.3 by quoting them
> would quote a false sentence to a reviewer. **Fix E6 first** ([`../plan/corrections.md`](../plan/corrections.md)),
> then point at the corrected text. The lesson's *reasoning* survives — the manuscript did already
> discuss the limitation — but the citation does not.

Third, **the comment's
opening sentence names the real complaint** — "*the discussion of the experimental results is rather
overlooked*" — and the target is the **density attribution**, with labels serving as the illustration
that we never examined it; the label hypothesis itself is refutable by argument alone, because for
AIDS both sides of the correlation are topology-only. Cost of the first plan: ~1 day plus a page we
do not have. Cost of the correct response: one paragraph, one table column, and two arguments that
were free. **The failure mode is designing work before checking what the comment asks for, what the
manuscript already does, and whether an argument would do the job of an experiment.**

### The six-question test — run it on every comment

1. **Where is the ask?** Usually the final sentence, in the imperative or subjunctive. The rest is
   justification. Quote the operative clause verbatim before scoping anything.
2. **What is the modal?** "should be described" / "please report" = requirement. "would strengthen" /
   "could benefit from" / "would be helpful" = suggestion. R3.2 and R3.7c are suggestions; R3.5a and
   R3.5c are requirements. Do not spend a requirement's budget on a suggestion.
3. **Does the manuscript already do it?** Check `verified-discrepancies.md` and the sources *before*
   designing. Two of R1.3's three asks were already in the submitted text.
4. **Is there an explicit `or`?** R3.6a offers "either narrow the claim accordingly **or** include
   comparisons with established reversible graph serializations". One branch is free.
5. **Can an argument replace a measurement?** A causal hypothesis about an existing number can often
   be refuted logically, or with a counter-example already in our own tables — as IAM's ρ ≈ 0.93
   under class-defining attribute loss refutes the label hypothesis for AIDS at zero cost.
6. **Who else is this work for?** If a measurement survives only because of one soft comment, it is a
   candidate for the cut list. If it has a second customer (as the collision count does — R1.2's
   expressiveness axis and the AE.3 table), it stays.

### Where the same error is most likely — audit these first

| Comment | What it literally asks | What we planned | Check |
|---|---|---|---|
| **R3.5b** | interpret Figure 3 cautiously; treat **dataset-level correlations as primary** | recompute **every** GED under one cost model — **1,000–1,650 core-hours**, the single largest item in the revision | `statistics.md` D5 alone (per-dataset primary, pooled demoted) satisfies the literal ask. The recompute is a deliberate decision to **retire** the objection rather than caveat it, and it is also driven by F2 and by the new cohort — but the cheap branch must be recorded as the fallback if T-03 fails |
| **R3.6a** | "**either** narrow the claim **or** include comparisons with established reversible serializations" | build six competitor backends | Narrowing is free. The competitors survive because **R1.1 and AE.4a demand them independently** — but R3.6a alone would not justify them |
| **`statistics.md` D4 (MRM)** | *nobody asked for this* | promoted to the **confirmatory** family | Justified as pre-empting a size-confound attack, and it is cheap. Confirm it stays cheap |
| **`statistics.md` §7 symmetry stratification** | *nobody asked for this*, stated openly in the file | new stratification variable | Cheap (nauty is vendored anyway) and it converts a limitation into a characterisation. Keep, but it belongs on the cut list |
| **AE.1 / Suite 2** | "How graph size might impact the presented results **should be clear**" | 5 → 10 datasets, 40 M pairs | Requirement modal, and the extension costs ~1.3 core-hours. Proportionate — verify the *page* cost, not the compute cost |

**Twenty days.** Every hour spent over-satisfying a suggestion is an hour not spent on T-20, which
rewrites five sections and had no owner at all until the coverage audit.

---

## Hard constraints on the revision

From the Editor-in-Chief, `mail.txt:124–130`. The EiC states these will be checked before approval
independently of the reviewers' verdict.

| Constraint | Source | Current state |
|---|---|---|
| Bibliography **35–55 items** in the final version | `mail.txt:126` | **43 items** — compliant, 12 slots of headroom |
| Replace arXiv citations with peer-reviewed versions | `mail.txt:126` | **1 genuine arXiv-only** citation; 5 more *display* an arXiv id — see below |
| No uncommented citation groups ("In prior work [1,2,3,4,5,6]") | `mail.txt:126` | **1 two-key group**, `\cite{garey1979,Zeng:2009}` at `methodology.tex:803` |
| Cite RECENT pattern-recognition work, not only the PR journal | `mail.txt:128` | Not audited here — flagged as an open item |
| **35 pages max**, double-spaced single column | `mail.txt:130` | `main.pdf` is **exactly 35 pages** — at the ceiling |
| Upload **source files**, not PDF, at resubmission | `mail.txt:22` | LaTeX sources present in the manuscript root |

### How the bibliography was counted

`cas-refs.bib` defines **56** entries, but `elsarticle-num` prints only what is cited, and two keys
are cited solely from commented-out LaTeX. The printed bibliography is the set of keys reached by
an uncommented `\cite` in `{introduction, methodology, computational_experiments, results,
conclusion, main}.tex`:

```bash
cd /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199
for f in main.tex introduction.tex methodology.tex computational_experiments.tex \
         results.tex conclusion.tex; do sed 's/%.*//' "$f"; done \
  | grep -oh '\\cite{[^}]*}' | sed 's/\\cite{//;s/}//' | tr ',' '\n' \
  | sed 's/ //g' | sort -u | grep -v '^$' | wc -l
# -> 43
```

**43 cited, 56 defined, 13 dead entries in the `.bib`.** Adding references in response to R1.2,
R3.1 and EiC.b can consume up to 12 slots before breaching the 55-item ceiling.

### The arXiv-citation constraint is narrower than it looks

Six cited entries carry an arXiv identifier, but five already name a peer-reviewed venue and merely
print the id in a `note` field:

| Key | Venue in the `.bib` | Genuinely arXiv-only? |
|---|---|---|
| `kipf2017gcn` | ICLR 2017 | No — `note = {arXiv:1609.02907}` |
| `velickovic2018gat` | ICLR 2018 | No — `note = {arXiv:1710.10903}` |
| `xu2019powerful` | ICLR 2019 | No — `note = {arXiv:1810.00826}` |
| `fey2019pyg` | ICLR Workshop 2019 | No — `note = {arXiv:1903.02428}` |
| `jain2024graphedx` | NeurIPS 37, 2024 | No — `note = {arXiv:2409.17687}` |
| **`lopezrubio2025isalgraph`** | `journal = {arXiv}, volume = {2512.10429v2}` | **Yes** |

So exactly one reference is an arXiv-only citation — and it is reference **[28]**, the authors' own
prior preprint, which R3.1 discusses at length. The other five will *render* as arXiv citations
because of the `note` field even though the venue is peer-reviewed.

### Page budget is the binding constraint on every additive response

`main.pdf` is 35 of 35 pages. The manuscript already carries content commented out explicitly to
reach that limit: the acknowledgements (`main.tex:175–177`), the AI declaration (`main.tex:198–202`),
both author biographies (`main.tex:225–245`), the algorithm-overview figure
(`methodology.tex:378–420`), the shortest-path figure (`methodology.tex:835–860`), and the whole
neighbourhood-structure subsection (`results.tex:253–327`). Any response that adds a figure,
a table, a comparison subsection or a new experiment competes with material already cut.
