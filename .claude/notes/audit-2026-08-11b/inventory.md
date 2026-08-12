# Demand inventory — audit-2026-08-11b

Built by the orchestrator from `.claude/notes/review/source/mail.txt` **only**, before any agent was
spawned. No verdicts here. Every clause is verbatim; ellipses mark elision inside a quoted sentence
and nothing else.

**Manuscript**: PR-D-26-03293, *Representation of Graphs by Sequences of Instructions*, Pattern
Recognition (Elsevier). Major revision, **due 2026-08-31** (20 days from 2026-08-11).

**Hard constraints that make scope expensive**
- **35 pages max**, double-spaced single column (`mail.txt:130`). `main.pdf` is **exactly 35**.
- Bibliography **35–55 items** (`mail.txt:126`). Currently **43 printed** → 12 slots of headroom.
- Source files, not PDF, at resubmission (`mail.txt:22`).

## ID scheme

IDs follow **`plan.md` §0.5**, not the letter's clause order, so the audit reconciles into the
existing traceability matrix without renumbering. Two consequences an auditor must act on:

1. The letter's `:126` states its clauses in the order *recency → arXiv → citation groups → 35–55*.
   §0.5 numbers them `EiC.a2 → a3 → a4 → a1`. The mapping below is authoritative.
2. **The decomposition itself is under audit.** If a row here corresponds to no imperative in the
   letter, or if a letter clause has no row, that is a finding — an invented demand is over-scope at
   the root, and an unmapped clause is a coverage hole. Two rows are already flagged: **M3** and
   **R1.3b** are marked `INFERRED` / `PREMISE` because the letter contains no imperative for them.

## Voices

| Voice | Lines | Owner |
|---|---|---|
| Editor-in-Chief (Zoran Duric) | `:124–130` | `audit-editors` |
| Area Editor | `:52–70` | `audit-editors` |
| Submission mechanics | `:20`, `:22` | `audit-editors` |
| Reviewer #1 | `:73–79` | `audit-r1` |
| Reviewer #3 | `:83–116` | `audit-r3` |

There is **no Reviewer #2**. The letter says "Both the reviewers" at `:55`, `:59`, `:66`. R3's
numbering is an Editorial Manager artefact.

**Priority statement, `mail.txt:67`** — bears on every AE row and on how the whole wave is weighted:

> "Please address these concerns thoroughly, as they will strongly influence the potential impact of
> the work and citation of the paper if it is accepted for publication after the revision."

The antecedent of "these concerns" is `:59–66` — AE.1 through AE.4c. It is a **weighting statement
over the AE's own agenda**, not a general instruction.

---

## Submission mechanics

| ID | Line | Operative clause (verbatim) |
|---|---|---|
| **M1** | `:22` | "When submitting your revised manuscript, please ensure that you upload the source files (e.g. Word, Latex). Uploading a PDF file at this stage will create delays should your manuscript be finally accepted for publication." |
| **M2** | `:20` | "Your revised manuscript is due by Aug 31, 2026 Kindly advise if you decide not to resubmit your paper." |
| **M3** | `:67` | `INFERRED — no imperative in the letter.` §0.5 records M3 as "Point-by-point response (inferred, `:67`)". The text at `:67` is the priority statement quoted above; it does not request a point-by-point response. Audit whether the row is justified on other grounds (journal convention) and whether `:67`'s real content — the AE priority weighting — is captured anywhere. |

## Editor-in-Chief (`:124–130`)

Preamble, `:124`: "here is a list of points worth checking, which we find author's overlook. **I will
check that these are adhered to before your paper is approved for publication**, assuming the
revision satisfies the Associate Editor and Reviewers." → every EiC row is a **compliance item
enforced independently of the reviewers**; rubric §4 guard 3 applies.

| ID | Line | Operative clause (verbatim) |
|---|---|---|
| **EiC.a1** | `:126` | "Your bibliography in the final version after the revision still should be between 35-55 items." |
| **EiC.a2** | `:126` | "Take a careful look at your bibliography and they cover the state of the art. Missing references from last and current year most probably would mean you are missing the state of the art and the revision process can be delayed being asked to update it." |
| **EiC.a3** | `:126` | "Please do not make excessive citation to arXiv papers, but substitute them with their peer-reviewed versions, or papers from a single conference series." |
| **EiC.a4** | `:126` | "Do not cite large groups of papers without individually commenting on them. So we discourage \" In prior work [1,2,3,4,5,6] …\"." |
| **EiC.b** | `:128` | "Please make sure the revised version is relevant to the readership of the Pattern Recognition field. To this end, please make sure you cite RECENT work from the field of pattern recognition not only the Pattern Recognition journal." |
| **EiC.c** | `:130` | "Although the revision could lead to extending your article, it still can not exceed the page limits or violate the format, i.e. double spaced SINGLE column with a maximum of 35 pages for a regular paper and 40 pages for a review." |

## Area Editor (`:52–70`)

| ID | Line | Operative clause (verbatim) |
|---|---|---|
| **AE.1** | `:59–60` | "For example, the graphs studied are relatively small, but in many real-world machine learning applications graphs may be quite large, and both reviewers have questions about differences seen for the graph data sets studied, and their potential impact on the results. **How graph size might impact the presented results should be clear.**" |
| **AE.2** | `:62` | "For related work, the reviewers point out that the work needs to be more solidly framed within the context of previous work in this area, and that additional references are needed to capture the specific contributions of the work in the paper." |
| **AE.3** | `:63–64` | "Related to this, reviewer 3 has asked for a detailed side-by-side comparison of existing graph representations with the proposed one, which fairly and completely identifies the properties, strengths, and limitations of each -- this will help focus the presentation of work in the paper, and clarify the contribution of the work." |
| **AE.4a** | `:66` | "Both reviewers also ask for a more detailed and rigorous analysis in the experiment designs, including **the choice of benchmark models**, …" |
| **AE.4b** | `:66` | "… **differences in information and structure in the graph datasets used (e.g., fully labeled, vs. partially-labeled)**, …" |
| **AE.4c** | `:66` | "… **and in the associated analysis of the results**." |
| **AE.5** | `:69` | "There are additional comments from the reviewers that should also be addressed in the revised paper -- please check their comments carefully when preparing your revision." — **no row in §0.5.** Catch-all with a requirement modal. Audit whether it is genuinely subsumed by the R1/R3 rows or whether it reaches un-numbered reviewer content (e.g. R3's un-numbered preamble at `:83`, R1's un-numbered opening at `:73`). |

**Note on AE.4**: the sentence is a single ask with three enumerated objects; §0.5 splits it a/b/c.
The split is retained. Note also its premise — "Both reviewers also ask for" — which makes AE.4 an
**amplification**, so each of a/b/c must be deduplicated against the R1/R3 row it amplifies rather
than counted as an independent demand.

## Reviewer #1 (`:73–79`)

Opening, `:73`: "The paper is interesting as it opens up new research directions in sequential
graph-string representations."

| ID | Line | Operative clause (verbatim) |
|---|---|---|
| **R1.1** | `:75` | "A more informative evaluation would compare the proposed methods against alternative approaches that address a similar problem setting." |
| **R1.2a** | `:77` | "In particular, the paper does not adequately position itself with respect to existing graph canonicalization methods. For example, canonical adjacency matrix representations used in Apriori-based Graph Mining (AGM) and depth-first search (DFS) codes employed by gSpan are not discussed." |
| **R1.2b** | `:77` | "It would be helpful for the authors to clarify how the proposed approach differs conceptually from these existing representations and what advantages it offers in comparison. Specifically, does the proposed graph-string representation provide benefits in terms of uniqueness, expressiveness, computational efficiency, scalability, or downstream learning performance? A more thorough comparison with established graph canonicalization techniques would help better contextualize the contribution and novelty of the work." |
| **R1.3a** | `:79` | "As such, it is unclear whether edge density alone is sufficient to explain the observed decline in performance." |
| **R1.3b** | `:79` | `PREMISE — no imperative.` "Consequently, the performance degradation on AIDS may come from the loss of label information rather than structural complexity alone." §0.5 records this as a demand ("Label loss is the uncontrolled confound") and assigns it T-18 Tier 0–1. Audit whether a declarative "may come from" licenses the work booked against it. |
| **R1.3c** | `:79` | "A more thorough discussion of this limitation, along with its impact on the reported results, would strengthen the paper." |
| **R1.3d** | `:79` | "Especially if incorporating label information could be applicable and a promising direction for future work." |

**R1.3's opening sentence, `:79`**: "Moreover, the discussion of the experimental results is rather
overlooked." This names the complaint the rest of the comment illustrates. It is the sentence the
README's R1.3 lesson turns on — see `.claude/notes/review/source/README.md:136–154`.

## Reviewer #3 (`:83–116`)

Opening, `:83`, states the strengths and then: "However, the rationale, novelty, methodological
details, and interpretation of the results require further clarification."

| ID | Line | Operative clause (verbatim) |
|---|---|---|
| **R3.1a** | `:86` | "The paper should provide a detailed side-by-side comparison that identifies which components are inherited, modified, or genuinely new, **and explain why the combined extension constitutes a sufficiently substantive contribution.**" (two clauses in one sentence — enumerate both) |
| **R3.1b** | `:86` | "The statement that \"no existing method satisfies all four properties\" is also too absolute without a systematic comparison." |
| **R3.2** | `:89` | "A downstream experiment using the new canonical representation with a sequential model, such as a Transformer as in [28] or an LSTM model as in [29], would substantially strengthen the paper's contribution." |
| **R3.3a** | `:92` | "Broad claims such as \"any finite simple graph\" and \"arbitrary graphs\" should be narrowed." |
| **R3.3b** | `:92` | "Theorem 2.12 also states that S2G is deterministic given both the string and the `directed` flag." |
| **R3.3c** | `:92` | "Please clarify whether this flag is part of the serialized representation or external metadata, since the string alone does not determine whether the decoded graph is directed or undirected." |
| **R3.4a** | `:95` | "In Algorithm 2, lines 24 to 30, the directed-edge conditions for 'C' and 'c' appear inconsistent with Table 1. … **Please verify these conditions against the implementation.**" |
| **R3.4b** | `:97` | "Please state whether these ordered lists are recomputed at each iteration or precomputed, and account for pair scanning, pointer walking, neighbor checks, and canonical backtracking in the theoretical complexity discussion." |
| **R3.4c** | `:99` | "Section 4.2 reports a canonical empirical fit of (T~n^(4.9)), but the Conclusion later refers to (T~ n^(9.0)) and describes the behavior as \"super-polynomial.\" **These statements should be reconciled.** The fitted (n^(4.9)) curve is polynomial, although the underlying backtracking procedure may have exponential worst-case complexity." |
| **R3.5a** | `:102` | "Please justify these exclusions and report the number of removed pairs for each dataset." |
| **R3.5b** | `:104` | "Because the datasets also differ substantially in density and size, the aggregated results in Figure 3 should be interpreted cautiously, with dataset-level correlations treated as the primary evidence." |
| **R3.5c** | `:106` | "The bootstrap procedure mentioned in Section 4.3 should be described and should operate at the graph level rather than the pair level." — preceded at `:106` by the bounding concession "**This does not invalidate Spearman's p as a descriptive measure**, but it could underestimate uncertainty and produce overly small p-values." |
| **R3.6a** | `:109` | "The authors should **either** narrow the claim accordingly **or** include comparisons with established reversible graph serializations." |
| **R3.6b** | `:111` | "The abstract and conclusion should reflect this density-dependent behavior." |
| **R3.7a** | `:114` | "However, the manuscript should also emphasize that the practical graph sizes used for the real-world evaluation are small, generally no more than approximately 12 nodes; that the canonical method is computationally expensive and may have exponential worst-case backtracking complexity; and that no sequential model or downstream pattern-recognition task is evaluated." (three items) |
| **R3.7b** | `:116` | "The manuscript would benefit from a dedicated subsection comparing the current work with IsalChem and the previous graph instruction method." |
| **R3.7c** | `:116` | "Section 2.3 could also benefit from a small schematic illustrating the canonical search space: different starting nodes and alternative uninserted-neighbor choices form the search branches, whereas displacement ordering and the priority (V\\succ v\\succ C\\succ c) remain fixed." |
| **R3.7d** | `:116` | "The paper should also clearly separate theoretical complexity, worst-case search behavior, and empirical runtime scaling, …" |
| **R3.7e** | `:116` | "… and revise broad statements concerning adjacency-matrix permutation equivariance, arbitrary graph support, universal strong GED correlation, and super-polynomial empirical scaling." |

---

## Totals

| Voice | Demands |
|---|---|
| Mechanics | 3 (one `INFERRED`) |
| Editor-in-Chief | 6 |
| Area Editor | 7 (one absent from §0.5) |
| Reviewer #1 | 7 (one `PREMISE`) |
| Reviewer #3 | 18 |
| **Total** | **41** |

§0.5 carries **40** rows plus 14 self-found-defect rows. The delta is **AE.5**, which has no row.

## Changelog

| Date | Change |
|---|---|
| 2026-08-11 | Created for audit-2026-08-11b. Built from `mail.txt` by the orchestrator at Phase 1. Adopts `plan.md` §0.5 IDs; adds **AE.5**; flags **M3** and **R1.3b** as demands with no imperative in the letter. |
