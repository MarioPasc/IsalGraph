# Editor decision and submission mechanics

**Manuscript**: PR-D-26-03293 — "Representation of Graphs by Sequences of Instructions"
**Journal**: Pattern Recognition (Elsevier), Editorial Manager
**Decision**: Major Revision
**Date received**: 2026-08-10 08:23:48 +0000 (`mail.txt:2`)
**Deadline**: **2026-08-31** (`mail.txt:20`) — 21 days
**Editor-in-Chief**: Zoran Duric, PhD (`mail.txt:42–44`)
**Addressed to**: Ezequiel López-Rubio, elr@uma.es (`mail.txt:5`, `:14`)
**CC**: Mario Pascual-Gonzalez, mpascual@uma.es (`mail.txt:8`)
**Journal mailbox for queries**: patcog@elsevier.com — "please contact only the Journal mailbox"
(`mail.txt:24`)

## Decision, verbatim

> The reviewers and editors handling your paper have recommended that your paper undergo major
> revision. If you care to revise it, we will reconsider it for publication.
> Your revised manuscript is due by Aug 31, 2026 Kindly advise if you decide not to resubmit your
> paper.

`mail.txt:18–20`.

## Submission mechanics

> When submitting your revised manuscript, please ensure that you upload the source files (e.g.
> Word, Latex). Uploading a PDF file at this stage will create delays should your manuscript be
> finally accepted for publication. If your revised submission does not include source files, we
> will contact you to request them.

`mail.txt:22`. **Source files, not PDF.** The manuscript is LaTeX (`elsarticle`, `cas-*` support
files present in the manuscript root); there is no Word version.

Unlike TPAMI, Pattern Recognition's letter states **no requirement for a clean, unhighlighted main
file** and **no separate "Summary of Changes" designation**. Neither appears anywhere in
`mail.txt`. It also does **not** explicitly demand a point-by-point response document — though the
Area Editor's "Please address these concerns thoroughly" (`mail.txt:67`) and standard Elsevier
practice both imply one. Recorded as an inference, not as a quoted requirement.

Optional, declined by default: AudioSlides (`mail.txt:34`) and the Research Elements journals
(`mail.txt:36–37`, carries an Article Publishing Charge).

## Area Editor comments

The Area Editor's remarks are **first-class comments**, not preamble. They raise four distinct
issues and explicitly amplify specific reviewer points. Verbatim, `mail.txt:52–70`.

> Thank you for submitting your work to Pattern Recognition.
> Both the reviewers and myself believe that this paper presents a potentially useful contributions
> for representing graphs as sequences for use in analysis and sequential processing.
> However, the reviewers identify a number of details that require clarification and additional
> analysis.
> As a result, I am recommending that the paper undergo a major revision to address these issues.

### AE.1 — Graph size and its impact on the results

> For example, the graphs studied are relatively small, but in many real-world machine learning
> applications graphs may be quite large, and both reviewers have questions about differences seen
> for the graph data sets studied, and their potential impact on the results.
> How graph size might impact the presented results should be clear.

**Type**: new measurement + framing.
**Verifiably true.** Every real-world dataset is filtered to `n <= 12`
(`experiments/paper_pipeline/config.yaml:40`, `steps.eval_setup.n_max: 12`; stated in the
manuscript at `computational_experiments.tex:47` and `:53`). Synthetic encoding reaches `n = 50`
greedy / `n = 20` canonical only (`config.yaml:66–67`). The manuscript concedes the ceiling at
`results.tex:251` ("feasible for graphs up to approximately 12 nodes") and `conclusion.tex:68`.

Amplifies **R3.7** (small graphs) and touches **R1.3** / **R3.5b** (differences between the
datasets). Note the AE reads "differences seen for the graph data sets studied" as a *shared*
concern of both reviewers — R1.3 attributes the AIDS drop to label loss, R3.5b to heterogeneous
GED cost models. Those are two different explanations for the same observation.

### AE.2 — Related-work positioning and additional references

> For related work, the reviewers point out that the work needs to be more solidly framed within
> the context of previous work in this area, and that additional references are needed to capture
> the specific contributions of the work in the paper.

**Type**: related work.
**Verifiably true** that the manuscript has no related-work section at all: the section list is
Introduction / Methodology / Computational experiments / Results / Conclusion (`main.tex:158–170`).
All positioning is compressed into `introduction.tex:11–33`. Amplifies **R1.2** and **R3.1**.

Interacts directly with **EiC.a** (35–55 bibliography items, currently 43) and **EiC.b** (cite
recent pattern-recognition work).

### AE.3 — Side-by-side comparison of graph representations

> Related to this, reviewer 3 has asked for a detailed side-by-side comparison of existing graph
> representations with the proposed one, which fairly and completely identifies the properties,
> strengths, and limitations of each -- this will help focus the presentation of work in the paper,
> and clarify the contribution of the work.

**Type**: related work / framing. The AE singles this one request out and endorses it explicitly,
which raises its weight above a normal reviewer comment. Corresponds to **R3.1** and **R3.7**.

The target of the comparison is the absolute claim at `introduction.tex:33` — "No existing method
is simultaneously compact, reversible, structure-preserving, and canonicalisable for arbitrary
graphs" — restated at `conclusion.tex:74`.

### AE.4 — Experiment design and analysis of results

> Both reviewers also ask for a more detailed and rigorous analysis in the experiment designs,
> including the choice of benchmark models, differences in information and structure in the graph
> datasets used (e.g., fully labeled, vs. partially-labeled), and in the associated analysis of the
> results.
> Please address these concerns thoroughly, as they will strongly influence the potential impact of
> the work and citation of the paper if it is accepted for publication after the revision.

**Type**: new measurement + analysis. Three named sub-issues:

1. **"choice of benchmark models"** — corresponds to **R1.1** (GED is the only comparator) and
   **R3.6a** (the GED standard construction is author-defined). Relevant fact: a
   Weisfeiler–Lehman subtree-kernel baseline is **already implemented and already computed** by
   the pipeline (`benchmarks/real_data/eval_setup/wl_kernel_computer.py`;
   `experiments/paper_pipeline/config.yaml:32`, `distance_metrics: [levenshtein, wl_kernel]`;
   `:34`, `wl_kernel.n_iter: 5`) and is **never rendered into any figure or table** — `grep -rn
   wl_kernel benchmarks/real_data/eval_visualizations/` returns nothing. Recorded as a fact about
   the repository, not as a proposal.
2. **"fully labeled vs. partially-labeled"** — corresponds to **R1.3**. See
   `verified-discrepancies.md` D7: the manuscript, R1 and the code give three mutually inconsistent
   accounts of which datasets carry labels.
3. **"associated analysis of the results"** — corresponds to **R3.5** and **R3.6**.

The closing sentence, "they will strongly influence the potential impact of the work and citation
of the paper", is the AE's statement of priority. AE.4 is where the AE puts the most weight.

### AE closing

> There are additional comments from the reviewers that should also be addressed in the revised
> paper -- please check their comments carefully when preparing your revision.
> We look forward to receiving a revised version of your paper, and best of luck with preparing
> your revision.

No comment is dismissed as optional.

## Editor-in-Chief checklist

> EiC: While you are revising your paper, here is a list of points worth checking, which we find
> author's overlook. I will check that these are adhered to before your paper is approved for
> publication, assuming the revision satisfies the Associate Editor and Reviewers.

`mail.txt:124`. These are enforced **independently** of the reviewers.

### EiC.a — Bibliography

> a) Take a careful look at your bibliography and they cover the state of the art. Missing
> references from last and current year most probably would mean you are missing the state of the
> art and the revision process can be delayed being asked to update it. Please do not make
> excessive citation to arXiv papers, but substitute them with their peer-reviewed versions, or
> papers from a single conference series. Do not cite large groups of papers without individually
> commenting on them. So we discourage " In prior work [1,2,3,4,5,6] ...". Your bibliography in the
> final version after the revision still should be between 35-55 items.

`mail.txt:126`. Four separable requirements. Current state, counted as documented in `README.md`:

| Requirement | State | Evidence |
|---|---|---|
| 35–55 items | **43** — compliant | 43 keys reached by an uncommented `\cite` |
| Cover last/current year | Newest cited works are 2024 (`khoshraftar2024survey`, `ju2024comprehensive`, `jain2024graphedx`) — **nothing from 2025 or 2026 except the authors' own [28] and [29]** | citation-order extraction |
| No excessive arXiv | 1 genuine arXiv-only entry, `lopezrubio2025isalgraph` = ref **[28]**; 5 further entries print an arXiv id from a `note` field despite naming a peer-reviewed venue | `cas-refs.bib` |
| No uncommented citation groups | 1 two-key group, `\cite{garey1979,Zeng:2009}` | `methodology.tex:803` |

The "cover last and current year" item is the weakest position: the decision is dated 2026 and the
bibliography's most recent third-party citations are from 2024.

Note on the four-way grouping at `introduction.tex:31`: it cites `Liu2023glam`, `Chen2023pagm`,
`Lan2023aednet` and `Bai2021hypergraph` as four *separate* `\cite` commands, each attached to its
own descriptive phrase ("joint learning-and-matching frameworks", "position-aware structure
embeddings", "neural subgraph matching", "hypergraph convolution"). That is individual commenting,
so it does not violate the letter of EiC.a — recorded because it is the passage most likely to be
mistaken for a violation.

### EiC.b — Pattern Recognition readership

> b) Please make sure the revised version is relevant to the readership of the Pattern Recognition
> field. To this end, please make sure you cite RECENT work from the field of pattern recognition
> not only the Pattern Recognition journal.

`mail.txt:128`. **I did not audit venue composition of the bibliography** — establishing which of
the 43 references count as "pattern recognition" is a judgement call, not a lookup. Flagged as an
open item. What is checkable: the graph-matching cluster at `introduction.tex:31–32`
(`Liu2023glam`, `Chen2023pagm`, `Lan2023aednet`, `Bai2021hypergraph`, `Fuchs2022matchinggraphs`) is
the existing pattern-recognition anchor, and it is 2021–2023.

### EiC.c — Page limit and format

> c) Although the revision could lead to extending your article, it still can not exceed the page
> limits or violate the format, i.e. double spaced SINGLE column with a maximum of 35 pages for a
> regular paper and 40 pages for a review.

`mail.txt:130`. This is a **regular paper**: 35 pages.

| Item | Value | Evidence |
|---|---|---|
| `main.pdf` page count | **35** | `pdfinfo main.pdf` -> `Pages: 35` |
| Format | double-spaced, single column | `\documentclass[review,times,number]{elsarticle}`, `main.tex:6` — the `review` option produces double-spaced single-column output |
| Geometry | letterpaper, 4.3/4.8 cm margins | `main.tex:11` |

**The manuscript is exactly at the ceiling.** The EiC permits extension only up to 35 pages, and
there is nothing left. Content already removed to reach 35 pages, each marked in the source as cut
for that reason:

| Removed content | Location |
|---|---|
| Acknowledgements (funders, SCBI, NVIDIA) | `main.tex:175–177` |
| Generative-AI declaration | `main.tex:198–202` — comment says "will be included in final version" |
| Both author biographies and photos | `main.tex:225–245` |
| Graphical abstract, Highlights | `main.tex:129–141` — submitted separately via Editorial Manager |
| `fig_algorithm_overview` (S2G/G2S side-by-side trace) | `methodology.tex:378–420` |
| `fig_shortest_path_comparison` | `methodology.tex:835–860` |
| Three GED-mitigation citations (ILP, Hausdorff) | `methodology.tex:804–808` |
| Entire "Neighbourhood Structure" results subsection + figure | `results.tex:253–327` |
| Critical semantic note on V/v pointer immobility | `methodology.tex:117–124` |

The AI declaration at `main.tex:198–202` is a live compliance risk independent of the reviewers:
it is commented out, and Elsevier requires it when generative AI was used in manuscript
preparation.

## What the letter does not contain

Checked so nobody looks for them:

- **No structured review form.** No per-criterion ratings, no significance/soundness/readability
  boxes, no reviewer-suggested-references field. The TPAMI-style "ratings at a glance" table has no
  analogue here.
- **No attached reviewer files.** No review says "see the attached file". All comments are inline.
- **No submission URL** in the letter body; resubmission goes through Editorial Manager.
- **No confidential comments to the editor** are reproduced.
- **No Reviewer #2.** See `README.md`.
- The two placeholders `%ATTACH_FOR_REVIEWER_DEEP_LINK INSTRUCTIONS%` and
  `%REVIEW_QUESTIONS_AND_RESPONSES%` (`mail.txt:120`, `:122`) are unexpanded Editorial Manager
  merge fields, not redacted content. `%REVIEW_QUESTIONS_AND_RESPONSES%` expanding to nothing is
  consistent with the journal using no structured form.
