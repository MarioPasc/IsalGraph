# Verified discrepancies

Every factual claim made by a reviewer, checked line by line against the `.tex` sources and the
implementation, plus discrepancies found during the same pass that no reviewer raised.

Manuscript paths are relative to
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`.
Code paths are relative to `/home/mpascual/research/code/IsalGraph`.

**Verdicts**: **CORRECT** = the reviewer's description of the source is accurate.
**PARTIALLY CORRECT** = the substance holds but a detail is wrong.
**INCORRECT** = the description does not match the source.
**UNVERIFIED** = I could not check it, with the reason stated.

Nothing in this file proposes a fix.

**Provenance.** A set of pre-checked findings was handed to me by the orchestrator. I re-derived
every one of them from the sources rather than copying, and the re-check changed three things:
the pair-recomputation call site is `graph_to_string.py:155`, not `:153` (D12); R3's Theorem 2.12
attribution is to the *proof*, not the statement, which downgrades it to PARTIALLY CORRECT (D4);
and I could not reproduce the "63 of 441" directedness-collision counts recorded in
`.claude/CLAUDE.md`, though the phenomenon itself replicates and is stronger than stated (D3b).

---

## Part 1 — Reviewer-raised

### D1 · `n^{9.0}` appears nowhere in the results — **CORRECT** (R3.4c)

R3: *"Section 4.2 reports a canonical empirical fit of `T~n^(4.9)`, but the Conclusion later refers
to `T~n^(9.0)`."*

Every exponent printed in the manuscript:

| Location | Text | Quantity |
|---|---|---|
| `results.tex:86` | `alpha = 3.1` | Greedy-rnd(`v_0`) |
| `results.tex:87` | `alpha = 4.5` | Greedy-Min |
| `results.tex:88` | `alpha = 4.9` | **Canonical (Pruned)** |
| `results.tex:89` | `alpha = 10.2` | GED **per pair** — not an IsalGraph method |
| `results.tex:107` | "The Canonical (Pruned) method's exponent `alpha = 4.9`" | Canonical |
| `results.tex:119`, `:237` | `T ~ n^{10.2}` | GED per pair |
| `results.tex:239` | "`T ~ n^{3.1}` to `T ~ n^{4.9}`" | the IsalGraph pipeline's span |
| `conclusion.tex:50` | "reducing computational cost from **`T ~ n^{9.0}`** to `T ~ n^{4.5}`" | — |
| `conclusion.tex:68` | canonical "scales empirically as `T ~ n^{4.9}`" | Canonical |

**`9.0` occurs exactly once in the manuscript and has no source.** The reported set is
{3.1, 4.5, 4.9, 10.2}.

**A second, independent inconsistency inside `conclusion.tex` that R3 did not name.**
`conclusion.tex:50` and `conclusion.tex:68` describe the same quantity — the cost of the canonical
method — and disagree:

- `:50` frames the Greedy-min trade-off as "reducing computational cost from `n^{9.0}` to `n^{4.5}`",
  i.e. canonical = 9.0, greedy-min = 4.5.
- `:68` says canonical "scales empirically as `T ~ n^{4.9}`".

So the conclusion assigns the canonical method two different exponents four paragraphs apart, and
neither of the pair `(9.0, 4.5)` matches the results table's `(4.9, 4.5)`.

Generating code:
`benchmarks/real_data/eval_visualizations/fig_empirical_complexity.py::_fit_polynomial` (OLS on
log-log). `experiments/README.md:120` records the emitted values as 3.1 / 4.5 / 4.9 / 10.2 over
`n = 3`–`20`, matching `results.tex` and not `conclusion.tex:50`.

---

### D2 · A degree-4.9 polynomial called "super-polynomial" — **CORRECT** (R3.4c)

`conclusion.tex:80`:

> The **super-polynomial** scaling of the canonical method calls for pruning strategies,
> symmetry-breaking heuristics, or polynomial-time approximations that preserve the completeness
> guarantee.

`n^{4.9}` is polynomial by definition. R3's diagnosis is exact: *"The fitted `n^(4.9)` curve is
polynomial, although the underlying backtracking procedure may have exponential worst-case
complexity."*

**The manuscript already contains the correct statement elsewhere** and does not connect the two.
`methodology.tex:477–480`:

> This exhaustive search is a complete invariant but has **exponential worst-case complexity** in
> the product of neighbour-choice counts, limiting its practicality to graphs of at most 10--12
> nodes for dense instances.

Three distinct objects are conflated across the document — exponential worst case
(`methodology.tex:477`), an empirical fit of `n^{4.9}` over `n = 3`–`20` (`results.tex:88`), and
"super-polynomial" (`conclusion.tex:80`) which is neither. This is the conflation R3.7d asks to be
resolved.

---

### D3 · Scope: "any finite simple graph" vs connectivity and reachability — **CORRECT** (R3.3a)

**D3a — the preconditions are real.**

| Location | Precondition |
|---|---|
| `methodology.tex:277` | Algorithm 2: `\Require Connected graph G = (V,E)` |
| `methodology.tex:281` | "Verify all nodes are reachable from `v_0`" |
| `methodology.tex:352–359` | Remark 2.5: "G2S can only encode nodes that are reachable from `v_0` via directed outgoing edges. The algorithm raises an error if any node is unreachable from the chosen starting node." |
| `methodology.tex:558` | Algorithm 3 ranges over "each `v in V` such that `v` can reach all nodes of `G`" |
| `methodology.tex:630` | Theorem 2.12 is stated for "finite, simple, **connected** graphs" |
| `computational_experiments.tex:32–33` | "Only connected graphs are retained, since the G2S algorithm ... requires a connected input" |
| `graph_to_string.py:305–340` | `_check_reachability` raises `ValueError` unless every node is reachable from `initial_node` |

**The broad claims that omit them:**

| Location | Text |
|---|---|
| `main.tex:106–108` (abstract) | "representing the structure of **any finite, simple graph**" |
| `introduction.tex:33` | "canonicalisable for **arbitrary graphs**" |
| `introduction.tex:45–46` | "a methodology for sequential representation of graphs satisfying all four desiderata" |
| `conclusion.tex:12–13` | "Every string over the alphabet ... decodes to a valid finite simple graph" — this one is true, it is the decode direction |
| `conclusion.tex:74` | "No prior sequential encoding of **arbitrary finite simple graphs** provides universal validity, reversibility, and canonical completeness together" |

The asymmetry matters: **S2G** really is total (every string decodes), and the manuscript is right
about that. **G2S** is partial (connected input, and for directed graphs a root of a spanning
out-tree). The abstract's "any finite, simple graph" conflates the two directions.

**D3b — directedness is not in the string, and the invariant fails across classes.**
**CORRECT, and materially stronger than R3 alleged.**

R3 asks only for clarification of "whether this flag is part of the serialized representation or
external metadata". It is external: `src/isalgraph/core/string_to_graph.py:57–61` takes
`directed_graph` / `directed` as constructor arguments, and `SparseGraph.__init__(self,
max_nodes: int, directed_graph: bool)` likewise. No symbol of
`Sigma = {N,n,P,p,V,v,C,c,W}` encodes directedness.

Verified empirically, `isalgraph-cpp` environment, `isalgraph.engine() == "cpp"`:

```python
from isalgraph import canonical_string
from isalgraph.core.sparse_graph import SparseGraph
def make(n, edges, directed):
    g = SparseGraph(n, directed)
    for _ in range(n): g.add_node()
    for u, v in edges: g.add_edge(u, v)
    return g
canonical_string(make(2, [(0,1)], False))  # 'V'
canonical_string(make(2, [(0,1)], True))   # 'V'
```

Full sweep, `n = 2..4`, all edge subsets leaving every node reachable from some root:

| n | undirected graphs | distinct strings | directed graphs | distinct strings | strings claimed by both |
|---|---|---|---|---|---|
| 2 | 1 | 1 | 3 | 2 | 1 |
| 3 | 4 | 2 | 51 | 12 | 2 |
| 4 | 38 | 6 | 3,614 | 185 | 5 |
| **total** | 43 | **9** | 3,668 | **199** | **8** |

- **Within** the undirected class the invariant holds perfectly: 9 distinct strings for the 9
  connected undirected graphs up to isomorphism on `n = 2..4` (1 + 2 + 6), zero collisions.
- **Across** classes it fails: **8 of the 9** undirected canonical strings are also produced by some
  directed graph — `V`, `VV`, `VVV`, `VVnv`, `VVPnC`, `VVVPnC`, `VVVPnCPC`, `VVVPnCnCPpC`.

Theorem 2.12 (`methodology.tex:628–637`) quantifies over "finite, simple, connected graphs" with no
directedness qualifier, so **as stated it is false**; it is true within a fixed directedness class.

*Caveat.* `.claude/CLAUDE.md` invariant 6 records this as "63 of 441 small graphs collide". Under
the protocol above I get 43 / 3,668 graph instances and 9 / 199 distinct strings, so I **could not
reproduce 63/441** — presumably a different enumeration (different reachability rule, or counting
graph instances rather than distinct strings). The phenomenon replicates; those two specific
numbers should be re-derived before being quoted.

---

### D4 · "Theorem 2.12 also states that S2G is deterministic given ... the `directed` flag" — **PARTIALLY CORRECT** (R3.3b)

The claim is attributed to the wrong object. Theorem 2.12 (`methodology.tex:628–637`) reads, in
full:

> Let `G = (V_G, E_G)` and `H = (V_H, E_H)` be finite, simple, connected graphs. Then
> `w*_G = w*_H  <=>  G ~ H`.

No mention of `S2G`, determinism, or the flag. The sentence R3 means is in the **proof**,
`methodology.tex:643–644`:

> The decoder S2G is a deterministic function of `w` and the directed flag, so `G' := S2G(w)` is
> uniquely defined.

**The correction strengthens R3's point.** A precondition that appears only inside a proof is not a
precondition of the theorem. The theorem statement ranges over graphs without fixing directedness,
while its proof silently assumes the class is fixed — which is exactly the gap D3b exhibits
empirically.

---

### D5 · Algorithm 2's `C`/`c` guards contradict Table 1 — **CORRECT about the manuscript; the implementation is right** (R3.4a)

R3: *"In Algorithm 2, lines 24 to 30, the directed-edge conditions for 'C' and 'c' appear
inconsistent with Table 1 ... Please verify these conditions against the implementation."*

The line reference resolves exactly to `methodology.tex:321–336` — see `manuscript-map.md`.

**Table 1**, `methodology.tex:102–107`:

| Instr | Semantics |
|---|---|
| `C` | Edge insertion (primary -> secondary): add `(val(pi_1), val(pi_2))` |
| `c` | Edge insertion (secondary -> primary): add `(val(pi_2), val(pi_1))` |

**Algorithm 2**, `methodology.tex:321–336`:

| Branch | Input-edge guard | Duplicate check | Edge emitted | Consistent? |
|---|---|---|---|---|
| `C` (`:321–325`) | `(v~_2, v~_1) in E` — **secondary -> primary** | `(val(l~_2), val(l~_1)) not in E(G_out)` | `add_edge(val(l~_1), val(l~_2))` — primary -> secondary | **no** |
| `c` (`:330–334`) | `(v~_1, v~_2) in E` — **primary -> secondary** | `(val(l~_1), val(l~_2)) not in E(G_out)` | `add_edge(val(l~_2), val(l~_1))` — secondary -> primary | **no** |

Both branches test one direction and write the other. R3 flagged the input-edge guard; **the
duplicate check is reversed as well**, which R3 did not mention.

**The implementation is correct and matches Table 1.** `src/isalgraph/core/graph_to_string.py`:

```python
# :208-212 -- C: edge primary -> secondary?
if tent_sec_in in self._input_graph.neighbors(tent_pri_in) and \
   tent_sec_out not in self._output_graph.neighbors(tent_pri_out):
    self._output_graph.add_edge(tent_pri_out, tent_sec_out)

# :223-229 -- c: edge secondary -> primary? (directed only)
if (self._input_graph.directed()
        and tent_pri_in in self._input_graph.neighbors(tent_sec_in)
        and tent_pri_out not in self._output_graph.neighbors(tent_sec_out)):
    self._output_graph.add_edge(tent_sec_out, tent_pri_out)
```

Guard, duplicate check and emitted edge all agree with each other and with Table 1. The same
structure appears in `canonical.py` and `canonical_pruned.py`.

**Conclusion: a transcription defect confined to the pseudocode.** No reported result depends on
it, and the reviewer's instruction to check against the implementation resolves in the
implementation's favour. This is the cheapest substantive correction in the whole review.

---

### D6 · "Strongly correlates" is not uniform — **CORRECT** (R3.6b)

R3's quoted values are exact. `results.tex:151`, Canonical (Pruned) row: `rho = 0.433` on LINUX,
`0.349` on AIDS. R3 says "approximately 0.43" and "0.35".

Unqualified claims:

| Location | Text |
|---|---|
| `main.tex:120–122` (abstract) | "the Levenshtein distance between IsalGraph strings **strongly correlates** with graph edit distance (GED)" |
| `conclusion.tex:24–26` | "*Metric locality.* The Levenshtein distance between IsalGraph strings **correlates strongly** with graph edit distance on real-world graph benchmarks." |

Correctly conditional elsewhere, so the defect is a propagation failure rather than a
misunderstanding:

- `results.tex:203–206`: "On sparse graphs (`m_bar <= 4.56`, `rho >= 0.682`) ... on denser graphs
  (`rho ~ 0.35`), domain-specific validation is advisable."
- `methodology.tex:819`: "the Spearman rank correlation between `d_IsalGraph` and GED is high on
  sparse graphs and **moderate on denser graphs**."

Related: `conclusion.tex:37` substitutes a significance claim for an effect-size claim ("even on the
densest datasets the correlation remains statistically significant") on sample sizes that D10 shows
are inflated by dependence.

---

### D7 · "AIDS is the only dataset ... IAM and LINUX are unlabeled" — **PARTIALLY CORRECT** (R1.3)

Three sources give three different accounts of which datasets carry labels.

| Source | Claim |
|---|---|
| **R1** (`mail.txt:79`) | "AIDS is the only dataset among the evaluated benchmarks that contains node and edge labels, whereas IAM and LINUX are unlabeled" |
| **`conclusion.tex:70`** | "node and edge labels, **present in all five benchmark datasets**, are discarded during encoding" (restated at `:81`) |
| **`computational_experiments.tex:30`** | "In all cases, node and edge attributes are discarded" |

Ground truth from the loaders:

| Dataset | Attributes in the source | Stripped by |
|---|---|---|
| IAM Letter LOW/MED/HIGH | **`(x, y)` node coordinates** | `benchmarks/real_data/eval_setup/iam_letter_loader.py:4`, `:60` — "Node attributes (x, y coordinates) are stripped" |
| LINUX | **none** | `graphedx_loader.py:82` `_strip_node_attributes` (no-op in effect) |
| AIDS | **atom and bond types** | `graphedx_loader.py:82–88` — "We strip everything for topology-only analysis" |

- **R1 is right** that AIDS is the only dataset with discrete categorical labels, and that LINUX is
  unlabeled. That is the substantive point and it stands.
- **R1 is wrong** that IAM is unlabeled: IAM Letter graphs carry continuous `(x, y)` coordinates,
  and those coordinates are what distinguishes the 15 letter classes. Discarding them is a real
  information loss, just not a categorical-label loss.
- **The manuscript is wrong** at `conclusion.tex:70` and `:81`: labels are *not* present in all five
  datasets — LINUX has none. No reviewer caught this; see E6.

R1's causal hypothesis — that the AIDS degradation comes from label loss rather than density — is
consistent with the code and is **not ruled out by any experiment in the paper**. Confirmed
independently at `experiments/README.md:150–152`.

---

### D8 · Filter exclusions unjustified and unquantified — **CORRECT** (R3.5a)

The filter, `computational_experiments.tex:203–205`:

> Agreement between Levenshtein and GED distance matrices is quantified over all valid pairs with
> `GED > 0` and `d_Lev > 0` (isomorphic and non-finite pairs excluded).

**No justification and no per-dataset removal count appears anywhere.** R3's request is a statement
of fact about the manuscript. What makes it worse is E2: the two pair totals the manuscript *does*
print differ by 473,147 and are never reconciled.

---

### D9 · IAM uses uniform costs; LINUX and AIDS topology-only — **CORRECT** (R3.5b)

The manuscript states it itself, `computational_experiments.tex:55–56`:

> The IAM subsets use uniform unit costs, while LINUX and AIDS use topology-only costs (zero for
> node operations, unit for edge operations).

Detail at `:41–42` (IAM: NetworkX A*, unit node/edge insert-delete, node substitution 0) and `:48`
(LINUX: "Precomputed exact GED matrices from GraphEdX with topology-only costs"). Confirmed in
code: `benchmarks/real_data/eval_setup/ged_computer.py` (IAM) and `graphedx_loader.py`
(LINUX/AIDS). `experiments/README.md:144–148` reaches the same conclusion independently.

R3's characterisation of Figure 3 is also correct: `fig:heatmap-correlation-ged-lev`
(`results.tex:179–185`) pools all five datasets into one joint distribution, and `results.tex:187–190`
plus `conclusion.tex:38–41` draw conclusions from the pooled fit.

---

### D10 · "1,180 graphs ... up to 695,610 dependent pairs" — **CORRECT** (R3.5c)

`C(1180, 2) = 695,610` (confirmed), matching `results.tex:36–37`.

The independence assumption is explicit in the manuscript, `computational_experiments.tex:208–209`:

> Statistical significance is assessed at the `0.001` level via the **asymptotic Spearman test**
> (`n > 1,600` pairs in all datasets).

The parenthetical confirms R3's reading that the **pair count is the sample size**. R3's inference
about the consequence ("could underestimate uncertainty and produce overly small p-values") is the
standard consequence of treating dependent observations as independent, and their concession that
it "does not invalidate Spearman's `p` as a descriptive measure" is correct.

Aside: the `n > 1,600` claim only holds via LINUX's 1,685 *valid* pairs — a number that appears
once, in `conclusion.tex:46`, and never in the results section. See E2.

---

### D11 · The Section 4.3 bootstrap is never described — **CORRECT** (R3.5c)

The **only** bootstrap mention in the entire manuscript is a parenthetical, `results.tex:175–176`:

> but the difference is not statistically meaningful (bootstrap 95% CIs overlap substantially)

No resampling unit, no replicate count, no CI construction method, no reference.

The procedure exists in code and is **pair-level**, exactly as R3 suspects:
`benchmarks/real_data/eval_correlation/correlation_metrics.py::bootstrap_correlation`, configured
at `experiments/paper_pipeline/config.yaml:48–49` with `n_bootstrap: 10000`, `n_permutations: 9999`.
A `mantel_test` is implemented in the same module and configured in the same block — the Mantel
permutation test being the standard remedy for precisely the dependence R3 describes — and it is
never reported. See E10.

---

### D12 · `P(M)` has `(2M+1)^2 = Theta(M^2)` pairs; recomputed or precomputed? — **CORRECT**; answer is RECOMPUTED (R3.4b)

R3's arithmetic is right. Definition 2.4, `methodology.tex:233–242`, defines
`P(M) = {(a,b) | a, b in {-M,...,M}}`, so `|P(M)| = (2M+1)^2`.

**Definite answer: recomputed at every frame, in every code path.**

| Call site | Enclosing scope | Frequency |
|---|---|---|
| `src/isalgraph/core/canonical.py:223` | `_step(...)`, defined `:202` — the recursive step | once per recursion frame |
| `src/isalgraph/core/canonical_pruned.py:226` | `_pruned_step(...)`, defined `:204` | once per recursion frame |
| `src/isalgraph/core/graph_to_string.py:155` | inside the `while` loop opened at `:140` | once per loop iteration |

`generate_pairs_sorted_by_sum` (`graph_to_string.py:41`) builds all `(2m+1)^2` pairs and sorts them:
`Theta(m^2 log m)` per frame. Nothing is memoised in the Python reference.

*(The pre-verified notes cited `graph_to_string.py:153`; the call is at **:155**. Corrected after
re-checking.)*

**The complexity accounting R3 asks for does not exist.** The manuscript's only complexity
statements are `O(N+E)` per-node triplet cost (`methodology.tex:498–501`), the qualitative
"exponential worst-case" (`:477–480`), and `O(|w_1||w_2|)` for Levenshtein (`:783`). **There is no
theoretical complexity bound for G2S or for canonicalisation anywhere in the paper** — only
empirical fits. Pair scanning, pointer walking, neighbour checks and backtracking are never costed.

Related measurement, post-submission: the C++ engine memoises the pair list per distinct `m`, and
the A/B isolates the cost of exactly this recomputation — 25.5x at `n = 6`, 41.9x at `n = 8`,
57.6x at `n = 9`, 108.6x at `n = 10` (`docs/engineering/CPP_OPTIMIZATION_LOG.md:84–87`; `:91`
"This single change accounts for most of the speedup").

---

### D13 · The "GED standard construction" is author-defined — **CORRECT** (R3.6a)

`computational_experiments.tex:162–176` defines
`B_GED(G) = (N - 1 + M) + 2 M ceil(log_2 N)` bits, under the heading "GED standard construction",
with **no citation**. The word "standard" recurs in the paragraph heading, in
`fig:message_length_scatter`'s caption (`results.tex:16`) and in Table 2's caption
(`results.tex:24–27`), never supported.

`experiments/README.md:105–106` independently flags the derived "53%–74% of the bits" figure as
arithmetic performed in the text, emitted by no script.

R3's scoping is correct: the comparison establishes compactness relative to this model only. Note
`results.tex:65–66` is already properly scoped ("53%--74% of the bits needed by the GED construction
model"); the unscoped instances are `results.tex:11` and the abstract's "compact"
(`main.tex:106`, `:122`).

---

### D14 · No sequential-model experiment — **CORRECT** (R3.2)

The four declared objectives, `computational_experiments.tex:3–11`: message length; empirical time
complexity; Levenshtein/GED agreement; speed–quality trade-off. The four results subsections
(`results.tex:5, 70, 129, 210`) match. **No classifier, generator, Transformer, LSTM or any other
learned model appears anywhere in the manuscript.**

Against the motivation: `main.tex:122–126` ("language-model-compatible ... direct applications in
graph similarity search, graph generation, and graph-conditioned language modelling"),
`introduction.tex:35–37`, `conclusion.tex:76`, `conclusion.tex:88–95`.

---

### D15 · AGM and gSpan not discussed — **CORRECT** (R1.2)

Neither string appears in any `.tex` file; `cas-refs.bib` has no entry for either. More broadly,
**there is no related-work section** (`main.tex:158–170`), and the survey at `introduction.tex:11–33`
covers adjacency matrices, SMILES/SELFIES, KG embedding, shallow embeddings, MPNN/GNN, graph
transformers, GraphRNN, VAEs and deep graph matching — every one of which is either learned or
non-canonical. The canonicalisation literature that IsalGraph's central claim competes with is
absent. `nauty`/Traces and Babai's quasipolynomial result are likewise uncited, though R1 does not
name them.

---

### D16 · The GED runtime comparison is not like-for-like — **CORRECT** (R1.1)

`results.tex:83` overlays "exact GED per-pair computation time" on a plot of **per-graph** encoding
time. `results.tex:230–240` then converts the comparison into headline speedups (48x to 14,000x).

The manuscript half-concedes at `results.tex:122` ("this fit is descriptive and should not be
interpreted as an asymptotic bound") but the comparison is load-bearing at `results.tex:230–240` and
`conclusion.tex:75`. The pipeline-level speedup of Section 4.4 does amortise encoding across pairs
(`eval_computational.py::_compute_amortized`); the figure R1 names does not.

---

### D17 · Real-world graphs are "no more than approximately 12 nodes" — **CORRECT** (R3.7, AE.1)

`experiments/paper_pipeline/config.yaml:40` — `steps.eval_setup.n_max: 12`. Stated in the manuscript
at `computational_experiments.tex:47` and `:53` ("After filtering (`<=`12 nodes)"), and conceded at
`results.tex:251` and `conclusion.tex:68`. Synthetic encoding reaches `n = 50` greedy / `n = 20`
canonical (`config.yaml:66–67`; `computational_experiments.tex:76–79`).

---

### D18 · "The edge density of AIDS remains relatively modest" — **NOT CHECKABLE FROM THE MANUSCRIPT** (R1.3)

R1 reasons about edge *density*. The manuscript reports mean edge *count* `m_bar` and never reports a
node count or a density. See **E1** — this is the reviewers' shared blind spot: R1 argues density is
too modest to explain the result, the authors argue density explains it, and neither side has the
number.

---

### D19 · "[28] ... Transformer-based classification experiment ... [29] ... an LSTM model" — **UNVERIFIED** (R3.1, R3.2)

Reference resolution **is** verified: [28] = `lopezrubio2025isalgraph` (arXiv:2512.10429v2),
[29] = `ThurnhoferHemsi:2025`, both cited at `introduction.tex:52–53`.

The *content* claims are not. The preprint PDF is in the repo at
`docs/references/2512_10429v2.pdf` and I did not open it; IsalChem is not in the repo at all. R3's
characterisation is the only evidence in this package. R3 has been accurate on every other
checkable claim, so the prior is favourable — but this is the one load-bearing R3 assertion that
nothing here confirms.

---

### D20 · Adjacency matrices "break permutation equivariance" — **CORRECT, and a genuine technical error** (R3.7e)

`introduction.tex:16`:

> Last but not least, it breaks permutation equivariance because its meaning depends on the
> arbitrary ordering assigned to the nodes.

The adjacency matrix **is** permutation equivariant: relabelling by a permutation matrix `P` sends
`M` to `P M P^T`, which is the definition of equivariance. What it is not is permutation
**invariant** — `M` is not a function of the isomorphism class. The sentence as written asserts the
opposite of the truth, and the property IsalGraph actually supplies is invariance
(Theorem 2.12), not equivariance. A one-word substitution. R3 flags it without explaining why.

---

## Part 2 — Not raised by any reviewer

Found during the same source pass. Each is the same class of defect the reviewers already
penalised, so each is a live round-2 risk.

### E1 · Density is never computed, and no node count is ever reported

The paper's central explanatory mechanism is graph **density**, and the paper cannot compute it from
anything it prints.

- Table 2's property block (`results.tex:36–38`) has exactly three rows: Graphs, Pairs, `m_bar` (mean
  edge count). **No mean node count.**
- Grepping the uncommented sources for `\bar{n}`, `\bar{N}`, "mean node", "nodes per graph" returns
  nothing.
- Density `2m/(n(n-1))` is therefore not derivable from the manuscript.

Yet: `results.tex:200` ("as edge density grows"), `conclusion.tex:31` ("as graph density
increases"), `conclusion.tex:69` ("degrades substantially as density increases ... the densest
benchmark"), and the figure named `fig_aggregated_density_correlation.pdf` all rest on it. `m_bar` is
being used as a synonym for density throughout, and the two differ whenever node counts differ —
which they do, since only the `<= 12` ceiling is shared. This is the substrate of R1.3 and AE.1,
and neither reviewer states it outright.

### E2 · The two pair totals do not reconcile — 473,147 pairs unaccounted

| Source | Value |
|---|---|
| Table 2 per-dataset pairs, `results.tex:37` | 695,610 + 784,378 + 2,118,711 + 3,916 + 295,296 = **3,897,911** |
| `results.tex:182` (caption) and `:187` | "aggregates all **3,424,764** pairs" |
| Difference | **473,147 — 12.1% of the raw total, never mentioned** |

The gap is the `GED > 0` / `Lev > 0` filter of `computational_experiments.tex:203–205`, so Table 2
reports **pre-filter** counts while Figure 3 reports **post-filter** counts, with no indication that
they are different quantities.

**LINUX is the extreme case.** `results.tex:37` gives 3,916 pairs — exactly `C(89,2)`, hence raw.
`conclusion.tex:46` gives "89 graphs, **1,685 valid pairs**". **2,231 LINUX pairs, 57.0%, are
filtered out**, and the results section shows only the pre-filter number.
`conclusion.tex:46` is the sole place in the manuscript where a valid-pair count appears.

This is precisely what R3.5a asks for and did not get, and it is worse than R3 realised: the numbers
needed to answer already exist in the manuscript and contradict each other.

### E3 · The fit range excludes most of the greedy data, or the stated range is wrong

`results.tex:86–90` reports all four exponents as "(fitted over the range `n = 3`--`20`)".
But `computational_experiments.tex:76–79` generates greedy data over `n in {3,...,50}` and
`results.tex:116–117` states "Both greedy variants scale to graphs with 50 nodes within the
600-second timeout" — corroborated by `config.yaml:66` (`max_n_greedy: 50`).

So either the Greedy-rnd and Greedy-Min exponents (3.1 and 4.5) discard 30 of 48 available node
counts, or the parenthetical range is wrong for those two rows. The manuscript does not say which.
Since the `n^{4.5}` value feeds `conclusion.tex:50`'s already-broken comparison (D1), this is worth
pinning down.

### E4 · A third, incompatible node range in the speedup analysis

`results.tex:231`: "All three encoding methods outperform exact GED at every tested graph size
(`n = 3` to **11** nodes)", and `results.tex:233` quotes "over 14,000x at `n = 11`". Elsewhere the
ranges are `n = 3`–`20` (fits), `<= 12` (real datasets, `config.yaml:40`), and `<= 50` (synthetic
greedy). Four different ranges across one results section, none cross-referenced.

### E5 · The abstract contradicts itself two sentences apart

`main.tex:106–108`: "a method for representing the structure of **any finite, simple graph**".
`main.tex:114`: "a greedy *GraphToString* algorithm encodes **any connected graph**".

Both in the same abstract. R3.3 objects to the first; neither reviewer notes that the second
already states the correct scope eight lines later.

### E6 · The conclusion asserts labels are present in all five datasets — false for LINUX

`conclusion.tex:70`: "node and edge labels, **present in all five benchmark datasets**, are
discarded during encoding." Repeated at `conclusion.tex:81`: "node and edge labels, which are
present in all five datasets used here".

LINUX program-flow graphs carry no node or edge attributes
(`benchmarks/real_data/eval_setup/graphedx_loader.py:82–88`). The statement is false, and it is
the mirror image of R1's own error (D7): R1 says only AIDS is labelled, the conclusion says all five
are, and the truth is AIDS (categorical) plus IAM (continuous coordinates). Any response to R1.3
that quotes `conclusion.tex:70` will be quoting a false sentence.

### E7 · All three algorithms are typeset after the references, on pages 33–35

`main.tex:66–67` sets `\floatpagefraction{1}` and `\textfraction{.001}`; combined with `[ht]`
placement, LaTeX defers every `algorithm` float to the end of the document. Algorithm 1 lands on
page 33, Algorithm 2 on page **34**, Algorithm 3 on page 35 — after the bibliography. Algorithm 2 is
described on pages 7–9 and printed 26 pages later.

Not raised by either reviewer, but it is the reading experience in which R3 was asked to check
Algorithm 2 against Table 1 (page 7) — and it interacts with EiC.c, since the algorithms occupy
three of the 35 pages almost entirely as float pages.

### E8 · A draft self-correction is printed in Example 2.3

`methodology.tex:200–202`, page 7 of the PDF:

> \texttt{v}: add node $2$; add edge $(0,2)$; CDLL $= [0,1,2]$ (inserted after $\pi_2=\ell_0$, so
> after~$0$ but before~$1$ in circular order **--- actually $[0,2,1]$**); $\pi_2$ still on $\ell_0$.

The example states the CDLL contents, then corrects itself mid-parenthesis. The following step
(`:205`) uses `[0,2,1]`, so the correction is the right value and the initial `[0,1,2]` is simply
wrong. A visible editing artifact in a worked example, in the printed PDF.

### E9 · 13 dead bibliography entries

`cas-refs.bib` defines 56 entries; 43 are reached by an uncommented `\cite`. Two of the 45 keys
cited anywhere are cited **only** from commented-out LaTeX (`methodology.tex:804–808`, the
ILP/Hausdorff GED-mitigation sentence cut for the page limit). The remaining unused entries are
inert. Harmless for compilation, but relevant to EiC.a: the gap between "56 in the file" and "43
printed" is an easy source of a miscount when someone checks the 35–55 constraint.

### E10 · Two computed baselines are never reported

Both are configured, executed by the pipeline, and absent from the manuscript:

| Asset | Implementation | Configured at | In the paper? |
|---|---|---|---|
| Weisfeiler–Lehman subtree kernel distance | `benchmarks/real_data/eval_setup/wl_kernel_computer.py` | `config.yaml:32` `distance_metrics: [levenshtein, wl_kernel]`; `:34` `n_iter: 5` | **No** — `grep -rn wl_kernel benchmarks/real_data/eval_visualizations/` returns nothing |
| Mantel permutation test | `eval_correlation/correlation_metrics.py::mantel_test` | `config.yaml:49` `n_permutations: 9999` | **No** |

WL is a graph-similarity baseline in the same problem setting R1.1 asks about; Mantel is the
standard test for the matrix-dependence problem R3.5c raises. Recorded as facts about the
repository. Nothing here proposes using either.

### E11 · The generative-AI declaration is commented out

`main.tex:198–202`, with the comment "commented out to meet the 35-page limit — will be included in
final version". Elsevier requires the declaration when generative AI was used in manuscript
preparation, and the commented text states Claude Code was used for code generation, testing and
literature search. This is a compliance item independent of the reviewers, and it competes for the
same page budget as every reviewer request.

### E12 · Two orphaned figure PDFs, and a misspelt filename

`fig_shortest_path_comparison.pdf` and `fig_neighborhood_topology.pdf` sit in the manuscript
directory with their `\includegraphics` calls commented out (`methodology.tex:835–860`,
`results.tex:280–288`). Relevant to R3.7c, which requests a *new* schematic while two rendered
figures are already cut for space. Separately, the graphical abstract is
`graphical_abtract.pdf` — misspelt, and referenced under that spelling at `main.tex:131`
(commented out).

---

## Aggregate view

| Class | Items |
|---|---|
| Reviewer claims fully **CORRECT** | D1, D2, D3, D5, D6, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D20 (16) |
| **PARTIALLY CORRECT** | D4 (attributed to the theorem; belongs to its proof), D7 (IAM is not unlabeled) |
| **INCORRECT** | none |
| **UNVERIFIED** | D18 (not checkable from the manuscript), D19 (prior-work content not read) |
| Found independently | E1–E12 |
| Concentrated in | `conclusion.tex` (D1, D2, D6, E5, E6), `results.tex` numbers (E1–E4), the abstract (D3, D6, E5) |
| Root cause of D1, D2, D6, E5 | The abstract and conclusion were not resynchronised with `results.tex` after the numbers changed. Every one of these defects is a claim in the framing sections that the results section already states correctly. |
| Root cause of E1, E2 | Dataset properties are reported as `m_bar` and raw pair counts, while the analysis operates on density and filtered pairs. Neither transformation is documented. |

**Not one reviewer factual claim about the manuscript is wrong.** The two downgrades are an
attribution slip (D4) and an over-general statement about IAM (D7), and in both cases the
reviewer's substantive point survives intact. Anyone drafting the response should assume every R3
assertion checks out until proven otherwise — that was the outcome for all 13 of R3's checkable
claims.
