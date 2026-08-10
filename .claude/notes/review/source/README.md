# PR-D-26-03293 — Major Revision: source notes

Reference documentation for agents and authors working the IsalGraph revision.
These files record **what the reviewers said and what is verifiably true in the sources**.
They do not contain proposed answers — drafting the response letter is a separate task.

Manuscript: *Representation of Graphs by Sequences of Instructions* (IsalGraph).
Journal: **Pattern Recognition** (Elsevier).
Decision received **2026-08-10**. Revision due **2026-08-31** (21 days).

Manuscript source root:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`
Decision letter: `.claude/notes/review/source/mail.txt`

## Files in this directory

| File | Content |
|------|---------|
| `mail.txt` | The decision letter as received. Source of every verbatim quotation in this package. |
| `00-editor-and-decision.md` | Editor-in-Chief and Area Editor comments, submission mechanics, deadline, required elements. |
| `reviewer-1.md` | R1 — 3 numbered comments (R1.1–R1.3). |
| `reviewer-3.md` | R3 — 7 numbered comments (R3.1–R3.7), several with sub-parts. |
| `manuscript-map.md` | Where every section, theorem, definition, algorithm, table and figure lives; cross-walk between the numbers reviewers cited and the actual `\label`s. |
| `verified-discrepancies.md` | Every factual claim by a reviewer, checked against the `.tex` sources and the code, with `file:line`, marked CORRECT / INCORRECT / PARTIALLY CORRECT. Plus discrepancies no reviewer caught. |
| `codebase-pointers.md` | Where the implementations, datasets and result files are, for anyone who has to re-run or re-measure. |

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
