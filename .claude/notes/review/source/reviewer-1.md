# Reviewer 1

Three numbered comments. Constructive register; no dispute with the construction itself.
Source: `mail.txt:73–79`.

## Opening assessment (verbatim)

> Reviewer #1: The paper is interesting as it opens up new research directions in sequential
> graph-string representations. Overall, the paper raises the following questions and concerns:

`mail.txt:73`. That is the whole of the preamble — R1 concedes novelty of direction and nothing
else. No strengths are enumerated, unlike R3.

R1's three comments are sequenced as one argument: the runtime comparison is against the wrong
thing (R1.1), because the right things are the graph-canonicalisation methods that were never
discussed (R1.2), and separately the results discussion misattributes a cause (R1.3). The Area
Editor amplifies all three under AE.2 and AE.4.

---

## R1.1 — The GED runtime comparison is not a fair comparison

**Verbatim:**

> 1. In Figure 2, Section 4.2., the authors compare the empirical runtime of their proposed methods
> against the per-pair computation time of GED. However, this comparison appears somewhat unfair,
> as the objectives and underlying assumptions of the two approaches differ significantly. A more
> informative evaluation would compare the proposed methods against alternative approaches that
> address a similar problem setting.

**Type**: new measurement / baseline choice. Amplified by **AE.4** ("choice of benchmark models").

**Where it lands.** The reviewer's reference resolves exactly:

| Reviewer's reference | Actual object | Location |
|---|---|---|
| "Section 4.2." | Section 4 = Results; 4.2 = *Empirical Time Complexity* | `results.tex:69` |
| "Figure 2" | `fig:empirical-complexity` (`fig_complexity_ratio_combined.pdf`) | `results.tex:92–99` |

**What the figure actually does.** `results.tex:83` — the panel shows encoding time for the three
IsalGraph methods "with exact GED per-pair computation time overlaid for reference". The fitted
exponents, `results.tex:86–90`: `alpha = 3.1` Greedy-rnd(v0), `4.5` Greedy-Min, `4.9` Canonical
(Pruned), `10.2` GED per pair, over `n = 3`–`20`.

**The reviewer's objection is structurally correct and the manuscript half-concedes it.**
`results.tex:122` already says of the GED fit: "this fit is descriptive and should not be
interpreted as an asymptotic bound." But the comparison is not merely descriptive in the rest of
the paper — it is load-bearing:

- `results.tex:230–240`, Section 4.4, converts it into headline speedups: "All three encoding
  methods outperform exact GED at every tested graph size", "48x at n = 3 to over 14,000x at
  n = 11", and attributes the growth to `n^{10.2}` vs `n^{3.1}`–`n^{4.9}`.
- `conclusion.tex:75`: "The Levenshtein distance on IsalGraph strings provides a polynomial-cost
  proxy for the NP-hard graph edit distance".

**Why the objects differ.** IsalGraph encoding is a *per-graph* cost paid once; GED is a *per-pair*
cost. The manuscript compares a per-graph cost curve against a per-pair cost curve on the same
axes. The pipeline speedup of Section 4.4 does amortise correctly
(`benchmarks/real_data/eval_computational/eval_computational.py::_compute_amortized`), but
Figure 2 itself does not — which is precisely the figure R1 names.

**Existing asset, recorded as fact.** The repository already computes a Weisfeiler–Lehman
subtree-kernel distance over the same datasets and never reports it:

| Item | Location |
|---|---|
| Implementation | `benchmarks/real_data/eval_setup/wl_kernel_computer.py` |
| Enabled in the pipeline | `experiments/paper_pipeline/config.yaml:32` — `distance_metrics: [levenshtein, wl_kernel]` |
| Parameterised | `config.yaml:34` — `wl_kernel.n_iter: 5` |
| Rendered into a paper artifact | **Never** — `grep -rn wl_kernel benchmarks/real_data/eval_visualizations/` returns nothing |

WL is a graph-similarity method addressing a similar problem setting to IsalGraph's Levenshtein
proxy. Its existence and its absence from the manuscript are both facts; nothing here proposes
what to do about either.

---

## R1.2 — Related work omits graph canonicalisation (AGM, gSpan)

**Verbatim:**

> 2. In line with the previous comment, the related work discussion appears incomplete. In
> particular, the paper does not adequately position itself with respect to existing graph
> canonicalization methods. For example, canonical adjacency matrix representations used in
> Apriori-based Graph Mining (AGM) and depth-first search (DFS) codes employed by gSpan are not
> discussed. It would be helpful for the authors to clarify how the proposed approach differs
> conceptually from these existing representations and what advantages it offers in comparison.
> Specifically, does the proposed graph-string representation provide benefits in terms of
> uniqueness, expressiveness, computational efficiency, scalability, or downstream learning
> performance? A more thorough comparison with established graph canonicalization techniques would
> help better contextualize the contribution and novelty of the work.

**Type**: related work. Amplified by **AE.2**; overlaps **R3.1** and **AE.3**.

**Verified: there is no related-work section.** `main.tex:158–170` inputs, in order,
`introduction`, `methodology`, Computational experiments, Results, `conclusion`. All positioning
lives in `introduction.tex:11–33`.

**Verified: neither AGM nor gSpan is cited.** Neither string appears in any `.tex` file, and
`cas-refs.bib` contains no entry for either. The two canonical works R1 has in mind are Inokuchi
et al. (AGM, PKDD 2000) and Yan & Han (gSpan, ICDM 2002).

**What the introduction covers instead** — grouped by family:

| Family | Cited at |
|---|---|
| Adjacency matrix and spectral/GNN methods | `introduction.tex:11–17` |
| String encodings for molecules — SMILES, SELFIES | `introduction.tex:19–22` |
| Knowledge-graph embedding | `introduction.tex:23` |
| Shallow embeddings — DeepWalk, node2vec | `introduction.tex:26` |
| MPNN / GNN / WL expressivity | `introduction.tex:27` |
| Graph transformers, GraphRNN, JT-VAE, GraphVAE | `introduction.tex:28–30` |
| Deep graph matching | `introduction.tex:31–32` |

**The gap R1 identifies is real and specific.** Every family above is either a *learned* or a
*non-canonical* representation. The one thing IsalGraph claims that none of them claims — a
canonical form that is a complete invariant — is exactly the property whose prior art is missing.
The claim at `introduction.tex:33`, "No existing method is simultaneously compact, reversible,
structure-preserving, and canonicalisable for arbitrary graphs", is asserted against a survey that
excludes the canonicalisation literature. R1.2 and R3.1's "too absolute without a systematic
comparison" are the same objection reached from two directions.

Also uncited: `nauty`/Traces (McKay & Piperno) and Babai's quasipolynomial graph-isomorphism
result, neither named by R1 but in the same missing family. The manuscript's only isomorphism
citations are the WL test (`weisfeiler1968reduction`, `introduction.tex:27`) and NP-hardness of GED
(`garey1979`, `Zeng:2009`, `methodology.tex:803`).

**The five axes R1 asks the comparison to address**, and where the manuscript currently has
something to say:

| Axis | Manuscript evidence |
|---|---|
| Uniqueness | Theorem 2.12, `methodology.tex:628–637` — complete invariant |
| Expressiveness | `main.tex:106–108` "any finite, simple graph" — contested by R3.3 |
| Computational efficiency | `results.tex:86–90` — `n^{4.9}` canonical |
| Scalability | `results.tex:251` — "up to approximately 12 nodes"; contested by AE.1, R3.7 |
| Downstream learning performance | **Nothing.** No downstream task is evaluated — this is R3.2 |

The last row is where R1.2 and R3.2 converge: R1 asks whether the representation helps downstream
learning, and no experiment in the paper can answer.

---

## R1.3 — The AIDS degradation may be label loss, not density

**Verbatim:**

> 3. Moreover, the discussion of the experimental results is rather overlooked. The authors
> attribute the performance degradation observed on the AIDS dataset to increased edge density.
> However, the edge density of AIDS remains relatively modest compared to many other graph benchmark
> datasets and real-world networks. As such, it is unclear whether edge density alone is sufficient
> to explain the observed decline in performance. More importantly, graph size and edge density may
> not be the primary limitations of the proposed approach. The current method discards node and edge
> labels, which are often critical for accurately characterizing graph similarity. For example, AIDS
> is the only dataset among the evaluated benchmarks that contains node and edge labels, whereas IAM
> and LINUX are unlabeled. Consequently, the performance degradation on AIDS may come from the loss
> of label information rather than structural complexity alone. This issue is particularly relevant
> for molecular graphs. Considering only the graph topology may be insufficient, as different atom
> types can exhibit similar valencies and occupy analogous structural positions. As a result, two
> molecules may share an identical topological structure while differing substantially in their
> chemical composition. Since the proposed representation does not preserve such label information,
> it may be unable to distinguish between these cases. A more thorough discussion of this
> limitation, along with its impact on the reported results, would strengthen the paper.
> Especially if incorporating label information could be applicable and a promising direction for
> future work.

**Type**: new measurement + framing. Amplified by **AE.4** ("fully labeled, vs. partially-labeled")
and **AE.1**. This is the single strongest criticism in the review: it proposes an uncontrolled
confound for the paper's central negative result.

### Where the density attribution is made

| Location | Text |
|---|---|
| `results.tex:196–202` | "higher mean edge counts coincide with lower `rho` values ... consistent with the sequential nature of the G2S traversal: as edge density grows, a single depth-first pass captures a diminishing fraction of the graph's connectivity" |
| `conclusion.tex:30–36` | "degrades monotonically as graph density increases ... for dense graphs, a single depth-first-like pass captures a diminishing fraction of all pairwise connectivity" |
| `conclusion.tex:69` | "degrades substantially as density increases: Spearman `rho` drops from 0.934 ... to 0.349 on the densest benchmark (AIDS)" |

### The reviewer is right that density is not established

**The manuscript never computes density.** It reports mean *edge count* `m_bar` and calls it density.
Table `tab:information-content` (`results.tex:36–38`) has three property rows — Graphs, Pairs,
`m_bar` — and **no mean node count**. Grepping the uncommented sources for a reported mean node count
(`\bar{n}`, `\bar{N}`, "mean node", "nodes per graph") returns nothing. Density
`2m/(n(n-1))` is therefore not computable from anything the paper prints, yet "density" carries the
whole causal argument. See `verified-discrepancies.md` E1.

R1's own premise — "the edge density of AIDS remains relatively modest" — is likewise not
checkable from the manuscript. Both the reviewer and the authors are reasoning about a quantity the
paper does not report.

### The label claim: R1 is partially correct

R1 asserts "AIDS is the only dataset among the evaluated benchmarks that contains node and edge
labels, whereas IAM and LINUX are unlabeled." Checked against the loaders:

| Dataset | Attributes in the source data | Stripped by | Verdict on R1's claim |
|---|---|---|---|
| IAM Letter LOW/MED/HIGH | **`(x, y)` node coordinates** | `benchmarks/real_data/eval_setup/iam_letter_loader.py:4`, `:60` — "Node attributes (x, y coordinates) are stripped" | R1 is **wrong** that IAM is unlabeled; it carries continuous geometric attributes |
| LINUX | none | `graphedx_loader.py:82` `_strip_node_attributes` (no-op in effect) | R1 correct |
| AIDS | atom and bond types | `graphedx_loader.py:82–88` — "We strip everything for topology-only analysis" | R1 correct |

So R1 is **PARTIALLY CORRECT**: AIDS is the only dataset with discrete *categorical* labels, which
is the substantive point and it stands. But IAM is not unlabeled — it loses `(x, y)` coordinates,
and IAM Letter graphs are geometric objects whose class identity depends on those coordinates.

**And the manuscript contradicts both R1 and itself.** `conclusion.tex:70` and `conclusion.tex:81`
both state that node and edge labels are "present in all five benchmark datasets" — which is false
for LINUX. Meanwhile `computational_experiments.tex:30` says "In all cases, node and edge
attributes are discarded". Three sources, three different accounts. Recorded as **D7** in
`verified-discrepancies.md`.

### What the manuscript already concedes

- `conclusion.tex:70–71` (Limitations): "The current formulation operates on graph topology only;
  node and edge labels ... are discarded during encoding. Extending IsalGraph to attributed graphs
  is therefore a prerequisite for applications in domains such as molecular chemistry and program
  analysis, where labels carry essential semantic information."
- `conclusion.tex:81` (Theoretical challenges): "A second open problem is the incorporation of node
  and edge labels".

So the *limitation* is already stated, and R1's closing request ("a promising direction for future
work") is already satisfied. What is **not** present anywhere is R1's actual argument: that label
loss, rather than density, may **cause the AIDS result**. The limitation is filed under future work;
it is never connected to the interpretation of the AIDS number.

### The confound is real and no experiment separates it

AIDS is simultaneously (a) the densest benchmark by `m_bar`, (b) the only one losing categorical
labels, and (c) the worst-correlating. Confirmed in code:
`benchmarks/real_data/eval_setup/graphedx_loader.py::_strip_node_attributes` removes atom and bond
types for AIDS; `experiments/README.md:150–152` records the same conclusion independently. The GED
ground truth for AIDS is the GraphEdX **topology-only** matrix
(`computational_experiments.tex:52`), so the reference distance also ignores labels — which
complicates R1's causal story but does not dissolve it, because two molecules with identical
topology and different atoms are at topology-GED 0 while being chemically distinct.

**I could not verify** whether the AIDS `rho = 0.349` would improve under a label-preserving GED,
because no label-aware GED matrix exists in the pipeline and the GraphEdX release used here is the
topology-only variant.

### Cross-references

- **R3.7** independently lists "the absence of node and edge attributes" among the limitations the
  manuscript should emphasise more.
- **R3.5b** offers the competing explanation for the same aggregate pattern: heterogeneous GED cost
  models across dataset families.
- **AE.1** treats "differences seen for the graph data sets studied" as a shared concern of both
  reviewers.
