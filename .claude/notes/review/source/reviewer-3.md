# Reviewer 3

Seven numbered comments, several with distinct sub-parts. Source: `mail.txt:83–116`.
The most detailed and the most technically accurate of the two reports. Every factual claim R3
makes about the manuscript checks out (see `verified-discrepancies.md`); every section and figure
number R3 cites resolves exactly.

Sub-identifiers (R3.4a, R3.4b, ...) are used where one numbered comment contains separable requests.

## Opening assessment (verbatim)

> Reviewer #3: The manuscript's main strength is its extension of prior instruction-based
> representations to generic graphs through a sparse CDLL construction and relabeling-invariant
> canonicalization. Its deterministic decoding, reversibility, complete-invariant claim, open
> implementation, and speed-quality trade-off are also valuable. The overall objective of developing
> a reversible sequential representation of graph topology is clear. However, the rationale,
> novelty, methodological details, and interpretation of the results require further clarification.

`mail.txt:83`. Note the precision: R3 says "complete-invariant **claim**", not "complete
invariant". Read alongside R3.3, that hedge looks deliberate.

---

## R3.1 — Relationship to prior work and novelty

**Verbatim:**

> 1. Relationship to prior work and novelty
> The manuscript appears to combine and extend two closely related prior frameworks. The earlier
> preprint [28] already introduced a universally valid, reversible, compact instruction-string
> representation of ordered adjacency matrices, together with locality claims and a
> Transformer-based classification experiment. IsalChem [29] already introduced the
> circular-list/two-pointer virtual-machine architecture, incremental graph construction, exhaustive
> shortest-then-lexicographic normalization, and Levenshtein-based locality. The principal new
> contribution of the present manuscript appears to be the redesign for generic graph topology and,
> most importantly, the graph-isomorphism-invariant canonicalization result. The paper should
> provide a detailed side-by-side comparison that identifies which components are inherited,
> modified, or genuinely new, and explain why the combined extension constitutes a sufficiently
> substantive contribution. The statement that "no existing method satisfies all four properties" is
> also too absolute without a systematic comparison.

**Type**: related work + framing. Explicitly endorsed by the Area Editor as **AE.3**. Restated by
R3 at the end of R3.7 as a request for a dedicated subsection.

**Reference resolution — confirmed.** `elsarticle-num` numbers by order of first citation. Extracted
from the uncommented sources in document order:

| Reviewer's ref | Key | Work |
|---|---|---|
| **[28]** | `lopezrubio2025isalgraph` | López-Rubio (2025), "Representation of the structure of graphs by sequences of instructions", arXiv:2512.10429v2 |
| **[29]** | `ThurnhoferHemsi:2025` | IsalChem |

Both are cited exactly once each, adjacent, at `introduction.tex:52–53`. Reference [28] is the
**only genuinely arXiv-only citation in the bibliography** — see `README.md`, which matters because
EiC.a asks for arXiv citations to be replaced with peer-reviewed versions and this one has none.

**What the manuscript currently says about the delta** — the entirety of it, `introduction.tex:52–53`:

> Our previous work \cite{lopezrubio2025isalgraph} is substantially different from \IsalGraph{}
> because the older approach requires a fixed ordering of the nodes and does not employ a circular
> doubly linked list of nodes.
> Also, our previous IsalChem methodology \cite{ThurnhoferHemsi:2025} is designed for chemical
> molecules only, while the current \IsalGraph{} methodology is devoted to general graphs.

Two sentences, both asserting difference, neither identifying what is inherited. R3's request for
an inherited / modified / new decomposition is unanswerable from the current text.

**R3's characterisation of the delta is accurate.** The two genuinely new items R3 names —
"redesign for generic graph topology" and "the graph-isomorphism-invariant canonicalization result"
— match the manuscript's own contribution list at `introduction.tex:56–62`, whose item 2 is the
complete-invariant claim (Theorem 2.12, `methodology.tex:628`).

**The "too absolute" statement.** R3 does not give a line reference. It resolves to
`introduction.tex:33`:

> No existing method is simultaneously compact, reversible, structure-preserving, and
> canonicalisable for arbitrary graphs.

and its restatement at `conclusion.tex:74`:

> No prior sequential encoding of arbitrary finite simple graphs provides universal validity,
> reversibility, and canonical completeness together; \IsalGraph{} is the first to do so, to the
> best of our knowledge.

The four properties are enumerated at `introduction.tex:38–43` as (i) compact, (ii) reversible,
(iii) structure-preserving, (iv) canonicalisable. Note the two sentences list **different** property
sets — `introduction.tex:33` has compact/reversible/structure-preserving/canonicalisable;
`conclusion.tex:74` has universal-validity/reversibility/canonical-completeness. Neither is
supported by a comparison table, and `conclusion.tex:74` additionally inherits the "arbitrary"
scope problem of R3.3.

**Cross-reference**: R1.2 reaches the same objection from the canonicalisation literature (AGM,
gSpan) rather than from the sibling projects.

---

## R3.2 — No sequential-model evaluation

**Verbatim:**

> 2. Sequential-model evaluation
> Although compatibility with Transformers and language models is repeatedly presented as an
> important motivation and application of IsalGraph, the manuscript does not evaluate the proposed
> representation with any sequential learning model. This is particularly notable because both
> closely related prior works contain sequence-model experiments. The current experiments establish
> that the output is a sequence, but not that it provides practical benefits for Transformers,
> recurrent models, graph classification, or graph generation. A downstream experiment using the new
> canonical representation with a sequential model, such as a Transformer as in [28] or an LSTM
> model as in [29], would substantially strengthen the paper's contribution.

**Type**: new experiment. The heaviest request in the round, and the one least compatible with the
21-day deadline and the 35-page ceiling.

**Verified: the motivation is repeated and the experiment is absent.**

Where LM compatibility is claimed:

| Location | Claim |
|---|---|
| `main.tex:122–126` (abstract) | "a compact, isomorphism-invariant, and **language-model-compatible** sequential encoding ... with direct applications in graph similarity search, graph generation, and graph-conditioned language modelling" |
| `introduction.tex:35–37` | "particularly appealing in the current era of large language models" |
| `conclusion.tex:76` | "it can be consumed directly by transformer and language model architectures, which may enable graph-conditioned tasks such as molecule captioning or knowledge-graph question answering" |
| `conclusion.tex:88–95` | Future work: autoregressive graph generation, graph-conditioned language modelling |

Where the experiments are declared — `computational_experiments.tex:3–11`, four objectives:
(i) message length vs a GED construction model; (ii) empirical time complexity on synthetic graphs;
(iii) Levenshtein/GED agreement; (iv) speed–quality trade-off. **None involves a learned model.**
The results sections are `res-info-content`, `res-complexity`, `res-correlation`, `res-tradeoff`
(`results.tex:5, 70, 129, 210`). No classifier, no generator, no sequence model anywhere.

R3's framing is exact: the experiments "establish that the output is a sequence, but not that it
provides practical benefits". The abstract's word "language-model-compatible" is a claim about
*format*, and the paper proves format compatibility trivially and nothing beyond it.

**R3's premise about the prior works.** R3 states [28] contains "a Transformer-based classification
experiment" and [29] "an LSTM model". I verified neither — the preprint PDF is at
`docs/references/2512_10429v2.pdf` in the repo but I did not open it, and IsalChem is not in the
repo. **Unverified.** R3's own characterisation is the only evidence in this package, and R3 has
been accurate on everything checkable.

**Cross-reference**: R1.2's last axis, "downstream learning performance", is the same gap.
R3.7 lists it again as a limitation to state explicitly.

---

## R3.3 — Scope of the representation

**Verbatim:**

> 3. Scope of the representation
> Broad claims such as "any finite simple graph" and "arbitrary graphs" should be narrowed. The
> current G2S algorithm is defined for connected graphs, and directed graphs additionally require a
> starting node from which all other nodes are reachable. Theorem 2.12 also states that S2G is
> deterministic given both the string and the `directed` flag. Please clarify whether this flag is
> part of the serialized representation or external metadata, since the string alone does not
> determine whether the decoded graph is directed or undirected.

**Type**: theory / framing / factual correction. **Correct on every count, and the true situation
is worse than alleged.**

### R3.3a — Connectivity and reachability

**Verified.** The precondition is real and stated inside the methodology but not in the claims:

| Location | Text |
|---|---|
| `methodology.tex:277` | Algorithm 2 `\Require Connected graph G = (V,E)` |
| `methodology.tex:281` | "Verify all nodes are reachable from `v_0`" |
| `methodology.tex:352–359` | Remark 2.5 (Reachability precondition): "G2S can only encode nodes that are reachable from `v_0` via directed outgoing edges. The algorithm raises an error if any node is unreachable" |
| `methodology.tex:558` | Algorithm 3 iterates "each `v in V` such that `v` can reach all nodes of `G`" |
| `computational_experiments.tex:32–33` | "Only connected graphs are retained, since the G2S algorithm ... requires a connected input" |
| `graph_to_string.py:305–340` (`_check_reachability`) | raises `ValueError` unless every node is reachable from `initial_node` |

The broad claims that ignore it: `main.tex:106–108` ("any finite, simple graph"), `main.tex:114`
("any connected graph" — the abstract contradicts itself two sentences apart),
`introduction.tex:33` and `:45–46` ("arbitrary graphs"), `conclusion.tex:74` ("arbitrary finite
simple graphs"), `conclusion.tex:17–19`. Enumerated in `verified-discrepancies.md` D3.

### R3.3b — "Theorem 2.12 also states..." — PARTIALLY CORRECT, and worth stating precisely

R3 attributes to Theorem 2.12 a statement that is **not in the theorem**. Theorem 2.12
(`methodology.tex:628–637`) reads in full:

> Let `G = (V_G, E_G)` and `H = (V_H, E_H)` be finite, simple, connected graphs. Then
> `w*_G = w*_H  <=>  G ~ H`.

No mention of `S2G`, determinism, or the `directed` flag. The sentence R3 is referring to is in the
**proof**, `methodology.tex:643–644`:

> The decoder S2G is a deterministic function of `w` **and the directed flag**, so
> `G' := S2G(w)` is uniquely defined.

**This makes R3's underlying point stronger, not weaker.** The theorem's *statement* quantifies over
graphs with no directedness qualifier at all, while its *proof* silently relies on the flag being
fixed. A precondition that appears only inside a proof is not a precondition of the theorem as
stated. See `verified-discrepancies.md` D4.

### R3.3c — Is the flag part of the representation? — external metadata, confirmed empirically

**The flag is a constructor argument, not a symbol in the string.**
`src/isalgraph/core/string_to_graph.py:57–61`:

```
def __init__(self, input_string: str, directed_graph: bool | None = None,
             *, directed: bool | None = None, ...)
```

`SparseGraph.__init__(self, max_nodes: int, directed_graph: bool)` likewise. Nothing in
`Sigma = {N,n,P,p,V,v,C,c,W}` encodes directedness.

**Verified empirically, and the consequence is stronger than R3 alleges.** R3 asks only for
clarification. In fact the complete-invariant theorem **fails across directedness classes**.

Reproduction (`isalgraph-cpp` env, engine reported as `cpp`):

```python
from isalgraph import canonical_string
from isalgraph.core.sparse_graph import SparseGraph

def make(n, edges, directed):
    g = SparseGraph(n, directed)
    for _ in range(n): g.add_node()
    for u, v in edges: g.add_edge(u, v)
    return g

canonical_string(make(2, [(0,1)], False))   # -> 'V'
canonical_string(make(2, [(0,1)], True))    # -> 'V'
```

**A single undirected edge and a single directed arc both canonicalise to `"V"`.**

Full sweep over `n = 2..4`, enumerating every edge subset that leaves all nodes reachable from some
root, canonicalising each under both flags:

| n | undirected graphs | distinct strings | directed graphs | distinct strings | strings claimed by **both** classes |
|---|---|---|---|---|---|
| 2 | 1 | 1 | 3 | 2 | 1 |
| 3 | 4 | 2 | 51 | 12 | 2 |
| 4 | 38 | 6 | 3,614 | 185 | 5 |
| **total** | 43 | **9** | 3,668 | **199** | **8** |

Two readings, both checked:

1. **Within a directedness class the invariant holds.** The 9 distinct undirected strings match
   exactly the 9 connected undirected graphs up to isomorphism on `n = 2..4` (1 + 2 + 6). No
   collisions.
2. **Across classes it does not.** **8 of those 9 undirected canonical strings are also produced by
   some directed graph.** Every non-trivial undirected canonical string in this range is ambiguous:
   `V`, `VV`, `VVV`, `VVnv`, `VVPnC`, `VVVPnC`, `VVVPnCPC`, `VVVPnCnCPpC`.

So Theorem 2.12 is true **within a fixed directedness class** and false as stated, since its
statement ranges over "finite, simple, connected graphs" without fixing the class.

**Note on a figure in the project's own documentation.** `.claude/CLAUDE.md` invariant 6 records
this as "63 of 441 small graphs collide". I ran the sweep myself and could not reproduce those
particular totals under the protocol above — my enumeration yields 43 undirected / 3,668 directed
graph instances and 9 / 199 distinct strings. The 63-of-441 figure presumably counts a different
population (a different reachability rule, or graph instances rather than distinct strings). **The
phenomenon is confirmed; the specific counts 63/441 are not reproduced here and should not be
quoted without re-deriving them.**

---

## R3.4 — Algorithmic details and complexity

Three separable requests in one numbered comment.

### R3.4a — Algorithm 2 `C`/`c` conditions contradict Table 1

**Verbatim:**

> In Algorithm 2, lines 24 to 30, the directed-edge conditions for 'C' and 'c' appear inconsistent
> with Table 1. Table 1 defines 'C' as primary to secondary and 'c' as secondary to primary, whereas
> Algorithm 2 appears to test the opposite input-edge directions. Please verify these conditions
> against the implementation.

**Type**: factual correction. **The reviewer is CORRECT about the manuscript. The implementation is
right and the pseudocode is wrong.**

**Line reference resolves exactly.** Counting `algorithmic` line numbers in Algorithm 2 (with the
two `\LineComment` lines numbered), lines **24–27** are the `C` branch and **28–30** the `c` branch
— precisely the range R3 cites. Source: `methodology.tex:321–336`.

**Table 1** (`methodology.tex:102–107`):

| Instr | Table 1 semantics |
|---|---|
| `C` | Edge insertion (primary -> secondary): add `(val(pi_1), val(pi_2))` |
| `c` | Edge insertion (secondary -> primary): add `(val(pi_2), val(pi_1))` |

**Algorithm 2** (`methodology.tex:321–336`):

| Branch | Input-edge guard | Duplicate check | Edge emitted |
|---|---|---|---|
| `C`, `:321–325` | `(v~_2, v~_1) in E` — secondary -> primary | `(val(l~_2), val(l~_1)) not in E(G_out)` | `add_edge(val(l~_1), val(l~_2))` — primary -> secondary |
| `c`, `:330–334` | `(v~_1, v~_2) in E` — primary -> secondary | `(val(l~_1), val(l~_2)) not in E(G_out)` | `add_edge(val(l~_2), val(l~_1))` — secondary -> primary |

**Both branches are internally inconsistent**: in each, the guard and the duplicate check test one
direction while the emitted edge goes the other way. R3 spotted the guard; the duplicate check is
reversed too and R3 did not mention it.

**The implementation is correct.** `src/isalgraph/core/graph_to_string.py:208–221` (`C`) and
`:223–238` (`c`):

```python
# -- C: edge primary -> secondary? --
if tent_sec_in in self._input_graph.neighbors(tent_pri_in) and \
   tent_sec_out not in self._output_graph.neighbors(tent_pri_out):
    self._output_graph.add_edge(tent_pri_out, tent_sec_out)
...
# -- c: edge secondary -> primary? (directed only) --
if (self._input_graph.directed()
        and tent_pri_in in self._input_graph.neighbors(tent_sec_in)
        and tent_pri_out not in self._output_graph.neighbors(tent_sec_out)):
    self._output_graph.add_edge(tent_sec_out, tent_pri_out)
```

Guard, duplicate check and emitted edge all agree, and both agree with Table 1. Same structure in
`canonical.py` and `canonical_pruned.py`.

**Consequence**: a transcription defect confined to the pseudocode. No reported result depends on
it. R3's closing instruction — "verify these conditions against the implementation" — resolves in
favour of the implementation.

### R3.4b — Is `P(M)` recomputed or precomputed?

**Verbatim:**

> Section 2.2.1 defines P(M) as containing ((2M+1)^2=\Theta(M^2)) ordered displacement pairs. Please
> state whether these ordered lists are recomputed at each iteration or precomputed, and account for
> pair scanning, pointer walking, neighbor checks, and canonical backtracking in the theoretical
> complexity discussion.

**Type**: theory / documentation gap.

**Reference resolves exactly**: Section 2.2.1 = *Pair Generation and Cost Ordering*,
`methodology.tex:223`; `P(M)` is Definition 2.4, `methodology.tex:233–242`. R3's arithmetic is
correct: `a, b in {-M,...,M}` gives `|P(M)| = (2M+1)^2 = Theta(M^2)`.

**Definite answer: RECOMPUTED, at every frame, everywhere in the Python reference.**

| Call site | Enclosing scope | Frequency |
|---|---|---|
| `src/isalgraph/core/canonical.py:223` | `_step(...)`, defined at `:202` — the recursive step | once per recursion frame |
| `src/isalgraph/core/canonical_pruned.py:226` | `_pruned_step(...)`, defined at `:204` | once per recursion frame |
| `src/isalgraph/core/graph_to_string.py:155` | inside the `while` loop opened at `:140` | once per loop iteration |

`generate_pairs_sorted_by_sum` (`graph_to_string.py:41`) constructs all `(2m+1)^2` pairs and sorts
them: `Theta(m^2 log m)` per frame. Nothing is memoised anywhere in the Python reference.

*(The pre-verified notes handed to me cited `graph_to_string.py:153`; the actual call is at
**:155**. Corrected here after re-checking.)*

**The complexity discussion R3 asks for does not exist.** The manuscript's only complexity
statements are: the `O(N+E)` triplet cost (`methodology.tex:498–501`), the qualitative "exponential
worst-case complexity in the product of neighbour-choice counts"
(`methodology.tex:477–480`), the `O(|w_1| |w_2|)` Levenshtein cost
(`methodology.tex:783`), and the empirical fits in `results.tex`. Pair scanning, pointer walking,
neighbour checks and backtracking are never costed. There is **no theoretical complexity bound for
G2S or for canonicalisation anywhere in the paper** — only the empirical exponents, which is exactly
the conflation R3.7 objects to.

**Directly relevant fact.** A C++ engine added after submission memoises the pair list per distinct
`m`, and the A/B measurement isolates the cost of the recomputation R3 is asking about:
25.5x at `n = 6`, 41.9x at `n = 8`, 57.6x at `n = 9`, 108.6x at `n = 10`
(`docs/engineering/CPP_OPTIMIZATION_LOG.md:84–87`; `:91` — "This single change accounts for most of
the speedup"). So the answer to R3's question is also the largest single constant factor in the
implementation. See `codebase-pointers.md`.

### R3.4c — `n^{4.9}` versus `n^{9.0}`, and "super-polynomial"

**Verbatim:**

> Section 4.2 reports a canonical empirical fit of (T~n^(4.9)), but the Conclusion later refers to
> (T~ n^(9.0)) and describes the behavior as "super-polynomial." These statements should be
> reconciled. The fitted (n^(4.9)) curve is polynomial, although the underlying backtracking
> procedure may have exponential worst-case complexity.

**Type**: factual correction. **CORRECT, and there are two independent defects.** Full evidence in
`verified-discrepancies.md` D1 and D2. Summary:

1. **`n^{9.0}` has no source.** `results.tex` reports exactly four exponents — 3.1, 4.5, 4.9
   (`results.tex:86–88`) and 10.2 for GED per pair (`:89`). `conclusion.tex:50` says "reducing
   computational cost from `T ~ n^{9.0}` to `T ~ n^{4.5}`". No `9.0` appears anywhere else in the
   manuscript. `conclusion.tex:50` and `conclusion.tex:68` also disagree with each other about the
   canonical exponent (4.5 vs 4.9).
2. **A degree-4.9 polynomial is not super-polynomial.** `conclusion.tex:80`: "The super-polynomial
   scaling of the canonical method calls for pruning strategies..." — a category error, exactly as
   R3 says.

R3's own diagnosis of the fix is precise and correct: the fit is polynomial; the worst case may be
exponential; those are different statements about different objects. Generating code:
`benchmarks/real_data/eval_visualizations/fig_empirical_complexity.py::_fit_polynomial`
(OLS on log-log).

---

## R3.5 — Statistical analysis

Three separable requests.

### R3.5a — Justify the exclusions and report how many pairs were removed

**Verbatim:**

> 5. Statistical analysis
> Section 3.2.5 excludes pairs with GED=0, Levenshtein distance=0, or non-finite GED. Please justify
> these exclusions and report the number of removed pairs for each dataset.

**Reference resolves exactly**: Section 3.2.5 = *Correlation Analysis*,
`computational_experiments.tex:200`. The filter, `computational_experiments.tex:203–205`:

> Agreement between Levenshtein and GED distance matrices is quantified over all valid pairs with
> `GED > 0` and `d_Lev > 0` (isomorphic and non-finite pairs excluded).

**Verified: no justification is given and no removal count is reported, for any dataset.**

**And the numbers that *are* reported do not reconcile** — a defect neither reviewer states outright.
Table `tab:information-content` (`results.tex:37`) gives per-dataset pair counts summing to
**3,897,911**, while `results.tex:187` and the caption at `:182` report the aggregate as
**3,424,764**. The gap is **473,147 pairs, 12.1% of the raw total**, silently dropped between the
table and the figure with no per-dataset breakdown. LINUX is worse: `results.tex:37` reports 3,916
pairs (= C(89,2), so raw), while `conclusion.tex:46` says "89 graphs, 1,685 valid pairs" — **57.0%
of LINUX pairs are filtered out**, and the results table shows the pre-filter number. This is
`verified-discrepancies.md` D8 and E2, and it is precisely the quantity R3 asks for.

### R3.5b — Heterogeneous GED cost models make the aggregate figure hard to read

**Verbatim:**

> IAM uses uniform node/edge edit costs, while LINUX and AIDS use topology-only costs with zero
> node-operation cost. Because the datasets also differ substantially in density and size, the
> aggregated results in Figure 3 should be interpreted cautiously, with dataset-level correlations
> treated as the primary evidence.

**CORRECT.** Both halves verified.

The manuscript states the heterogeneity itself, `computational_experiments.tex:55–56`:

> The IAM subsets use uniform unit costs, while LINUX and AIDS use topology-only costs (zero for
> node operations, unit for edge operations).

and in the dataset paragraphs at `:41–42` (IAM: NetworkX A*, uniform unit costs, node substitution
0) and `:48` (LINUX: "Precomputed exact GED matrices from GraphEdX with topology-only costs").
Confirmed in code: `benchmarks/real_data/eval_setup/ged_computer.py` for IAM,
`graphedx_loader.py` for LINUX/AIDS. `experiments/README.md:144–148` reaches the same conclusion
independently and in the same terms.

"Figure 3" resolves to `fig:heatmap-correlation-ged-lev` (`results.tex:179–185`), which pools all
five datasets into one joint distribution — so R3's characterisation of what the figure does is
right.

R3's recommendation ("dataset-level correlations treated as the primary evidence") is already
half-satisfied structurally: Table `tab:performance-summary` (`results.tex:138–157`) is per-dataset.
What is not satisfied is the *reading* — `results.tex:187–190` draws a conclusion from the pooled
OLS slopes, and `conclusion.tex:38–41` promotes the pooled `beta in [0.78, 0.82]` to a headline.

### R3.5c — Pair-level dependence and the undescribed bootstrap

**Verbatim:**

> The asymptotic Spearman test also appears to treat all pairwise distances as independent
> observations. However, graph pairs share graph instances; for example, (d(G_1,G_2)) and
> (d(G_1,G_3)) both depend on (G_1). Thus, although IAM LOW contains 1,180 graphs, the test appears
> to use up to 695,610 dependent pairs as its sample size. This does not invalidate Spearman's p as
> a descriptive measure, but it could underestimate uncertainty and produce overly small p-values.
> The bootstrap procedure mentioned in Section 4.3 should be described and should operate at the
> graph level rather than the pair level.

**CORRECT on every checkable point.**

- **Arithmetic**: `C(1180, 2) = 695,610` (confirmed), matching `results.tex:36–37` (1,180 graphs / 695,610
  pairs).
- **The test is the asymptotic one**, `computational_experiments.tex:208–209`: "Statistical
  significance is assessed at the `0.001` level via the asymptotic Spearman test (`n > 1,600` pairs
  in all datasets)." The parenthetical confirms R3's reading that the pair count *is* the sample
  size. (Incidentally, `n > 1,600` only holds because LINUX has 1,685 *valid* pairs — a number that
  appears in `conclusion.tex:46` and nowhere in the results section.)
- **"Section 4.3"** resolves to *Correlation with Graph Edit Distance*, `results.tex:128`. The only
  bootstrap mention in the manuscript is a parenthetical, `results.tex:175–176`: "the difference is
  not statistically meaningful (bootstrap 95% CIs overlap substantially)." **That is the whole
  description.** No resampling unit, no replicate count, no CI method. R3's "should be described" is
  a statement of fact about the manuscript.
- **The procedure exists in code but is pair-level**:
  `benchmarks/real_data/eval_correlation/correlation_metrics.py::bootstrap_correlation`, configured
  at `experiments/paper_pipeline/config.yaml:48–49` with `n_bootstrap: 10000` and
  `n_permutations: 9999`. A `mantel_test` is also implemented and configured — the Mantel
  permutation test is the standard remedy for exactly the dependence R3 describes — and, like the WL
  kernel, **it is never reported in the manuscript**.

---

## R3.6 — Interpretation of the results

Two separable claims.

### R3.6a — The "GED standard construction" is not established as a baseline

**Verbatim:**

> 6. Interpretation of the results
> The interpretation and conclusions are only partially supported by the data. The message-length
> comparison in Section 3.2.3 uses an author-defined "GED standard construction" based on explicit
> node and edge insertions. This is a reasonable simple reference model, but the manuscript does not
> establish that it is a standard or competitive graph-encoding baseline. Therefore, the reported
> results demonstrate compactness only relative to this construction model. The authors should
> either narrow the claim accordingly or include comparisons with established reversible graph
> serializations.

**CORRECT.** Reference resolves exactly: Section 3.2.3 = *Message Length Analysis*,
`computational_experiments.tex:141`.

The model is defined at `computational_experiments.tex:162–176`:

> `B_GED(G) = (N - 1 + M) + 2 M ceil(log_2 N)` bits

with one type bit per operation and `2 ceil(log_2 N)` endpoint-addressing bits per edge. It carries
**no citation** — the paragraph is headed "GED standard construction" and the word "standard"
appears in the heading, the caption of `fig:message_length_scatter` (`results.tex:16`), and the
table caption (`results.tex:24–27`), always unsupported. `experiments/README.md:105–106` independently
flags the derived "53%–74% of the bits" range as arithmetic done in the text, emitted by no script.

The claims that inherit the limitation: `results.tex:11` ("shorter bit representations for
98.8%–99.6% of graphs"), `results.tex:65–66` ("require 53%–74% of the bits needed by the GED
construction model" — this one is correctly scoped), and the abstract's implicit "compact"
(`main.tex:106`, `:122`). R3's "narrow the claim accordingly" is aimed at the unscoped instances.

### R3.6b — "Strongly correlates" is not uniform

**Verbatim:**

> The description that IsalGraph distance "strongly correlates" with GED is supported on the sparse
> IAM datasets but not uniformly across all benchmarks. Correlations of approximately (0.43) on
> LINUX and (0.35) on AIDS are weak to moderate. The abstract and conclusion should reflect this
> density-dependent behavior.

**CORRECT, and R3's numbers are exact.** Table `tab:performance-summary` (`results.tex:151`):
Canonical (Pruned) `rho = 0.433` on LINUX and `0.349` on AIDS. R3 quotes 0.43 and 0.35.

The two unqualified statements R3 names:

| Location | Text |
|---|---|
| `main.tex:120–122` (abstract) | "show that the Levenshtein distance between IsalGraph strings **strongly correlates** with graph edit distance (GED)" |
| `conclusion.tex:24–26` (property iv) | "*Metric locality.* The Levenshtein distance between IsalGraph strings **correlates strongly** with graph edit distance on real-world graph benchmarks." |

Both are unconditional. The results section is already properly conditional —
`results.tex:203–206`: "On sparse graphs (`m_bar <= 4.56`, `rho >= 0.682`) ... on denser graphs
(`rho ~ 0.35`), domain-specific validation is advisable." So the mismatch is between the results
section and the abstract/conclusion, the same pattern as R3.4c.

Note `conclusion.tex:37` also asserts "even on the densest datasets the correlation remains
statistically significant" — true (`results.tex:136`, all fifteen significant at 0.001) but a
significance claim standing in for an effect-size claim, on sample sizes R3.5c has just argued are
inflated by dependence.

**Partially self-corrected already**: `methodology.tex:819` states the locality property as "high on
sparse graphs and moderate on denser graphs" — the correct framing exists in the methodology and
does not propagate to the abstract or conclusion.

---

## R3.7 — Limitations and presentation

**Verbatim:**

> 7. Limitations and presentation
> The authors acknowledge the scalability of canonicalization, reduced GED correlation on denser
> graphs, and the absence of node and edge attributes. However, the manuscript should also emphasize
> that the practical graph sizes used for the real-world evaluation are small, generally no more than
> approximately 12 nodes; that the canonical method is computationally expensive and may have
> exponential worst-case backtracking complexity; and that no sequential model or downstream
> pattern-recognition task is evaluated.
>
> The manuscript would benefit from a dedicated subsection comparing the current work with IsalChem
> and the previous graph instruction method. Section 2.3 could also benefit from a small schematic
> illustrating the canonical search space: different starting nodes and alternative uninserted-neighbor
> choices form the search branches, whereas displacement ordering and the priority
> (V\succ v\succ C\succ c) remain fixed. The paper should also clearly separate theoretical complexity,
> worst-case search behavior, and empirical runtime scaling, and revise broad statements concerning
> adjacency-matrix permutation equivariance, arbitrary graph support, universal strong GED correlation,
> and super-polynomial empirical scaling.

**Type**: framing + presentation. A summary comment that re-raises R3.2, R3.3, R3.4c and R3.6b, plus
three new items.

### What R3 credits as already acknowledged — verified

| Acknowledged limitation | Location |
|---|---|
| Scalability of canonicalisation | `conclusion.tex:68` |
| Reduced GED correlation on denser graphs | `conclusion.tex:69` |
| Absence of node/edge attributes | `conclusion.tex:70–71` |

R3 has read the Limitations paragraph accurately.

### R3.7a — Three limitations to add

| Requested | Verifiably true? | Evidence |
|---|---|---|
| Real-world graphs are "no more than approximately 12 nodes" | **Yes** | `config.yaml:40` `n_max: 12`; `computational_experiments.tex:47`, `:53`; already partly stated at `results.tex:251` and `conclusion.tex:68` |
| Canonical method may have exponential worst-case backtracking | **Yes, and the manuscript already says so once** | `methodology.tex:477–480`: "exponential worst-case complexity in the product of neighbour-choice counts". It does not appear in the Limitations paragraph, where `conclusion.tex:80` instead says "super-polynomial" — the R3.4c error |
| No sequential model or downstream pattern-recognition task is evaluated | **Yes** | See R3.2. Not stated as a limitation anywhere |

The middle row is the interesting one: the correct statement exists in the methodology and the
incorrect one in the conclusion, with no cross-reference between them.

### R3.7b — Dedicated comparison subsection

Restates **R3.1**, endorsed as **AE.3**. Competes directly with the 35-page ceiling.

### R3.7c — Schematic of the canonical search space in Section 2.3

**Reference resolves exactly**: Section 2.3 = *Canonicalization*, `methodology.tex:421`.

R3's description of the search space is **correct and matches the manuscript exactly**. Remark 2.8
(`methodology.tex:462–470`) states precisely what R3 says:

> The priority order `V > v > C > c` and the minimum-displacement pair ordering `P(M)` ... are
> intrinsic to the algorithm definition and are *not* branched over. Only the identity of the
> uninserted neighbour chosen at each `V`/`v` step contributes to the search space.

R3 has additionally inferred the starting-node branch, which is in Definition 2.7
(`methodology.tex:452`, union over `v in V`) and Algorithm 3 (`methodology.tex:558`). So the
requested figure would render Remark 2.8 + Definition 2.7, both already written.

**Relevant fact**: a renderer for exactly this figure now exists in the repository —
`src/isalgraph/viz/search_tree.py`, `canonical_search_tree_figure`, described in `.claude/CLAUDE.md`
as "canonical search-space schematic (Reviewer 3)". Recorded as an existing asset; nothing here
proposes using it.

### R3.7d — Separate theoretical complexity, worst case, and empirical scaling

Restates **R3.4b** and **R3.4c**. As established under R3.4b, the manuscript has no theoretical
complexity bound for G2S or canonicalisation at all, so the three-way separation R3 asks for
currently has only two of its three terms.

### R3.7e — Four broad statements to revise

| Statement | Where | Verdict |
|---|---|---|
| Adjacency-matrix **permutation equivariance** | `introduction.tex:16` | **R3 is right and this is a genuine technical error** — see below |
| Arbitrary graph support | `main.tex:106`, `introduction.tex:33`, `:45`, `conclusion.tex:74` | Correct — R3.3a, `verified-discrepancies.md` D3 |
| Universal strong GED correlation | `main.tex:121`, `conclusion.tex:25` | Correct — R3.6b, D6 |
| Super-polynomial empirical scaling | `conclusion.tex:80` | Correct — R3.4c, D2 |

**On permutation equivariance.** `introduction.tex:16` says of the adjacency matrix:

> Last but not least, it breaks permutation equivariance because its meaning depends on the
> arbitrary ordering assigned to the nodes.

This is wrong as stated. The adjacency matrix *is* permutation equivariant: relabelling nodes by a
permutation `P` maps `M` to `P M P^T`, which is the defining property of equivariance. What it is
not is permutation **invariant** — `M` is not a function of the isomorphism class. The manuscript
uses "equivariance" where it means "invariance", and the sentence as written asserts the opposite of
the truth. R3 flags it without spelling out why; the correction is a one-word substitution.
Recorded as `verified-discrepancies.md` E4.
