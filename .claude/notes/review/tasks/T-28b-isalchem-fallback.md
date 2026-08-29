# T-28b — IsalChem p.7949 Fingerprint-Metric Fallback

**Opened** 2026-08-29. **Owner** agent `t28b-isalchem` (wave `2026-08-29-t28-metrics`).
**Status** ANALYSIS COMPLETE — recommendation: **do not implement**. See §4.

---

## 1. What page 7949 actually contains

The journal page 7949 of Thurnhofer-Hemsi et al. (2025), *J. Chem. Inf. Model.* 65(15):7936–7955
(DOI `10.1021/acs.jcim.5c00354`, PMC12344769) falls inside the subsection titled
**"Molecular Similarity"** under Experiments.

The section defines **seven binary-fingerprint similarity metrics** and reports a single
experiment: for 1,000 molecules sampled from ZINC, each is encoded as an IsalChem string;
successive random edits (Levenshtein steps) are applied; the RDKit default fingerprint is
computed for each string's molecule; and the seven metrics are evaluated between the original
and each edited molecule. The result — that all seven metrics decrease monotonically with
Levenshtein distance — is the paper's empirical validation of §3's theoretical locality claim.

There is no subgraph *repertoire* in the usual graph-kernel sense. The "repertoire" is the
RDKit fingerprint's implicit vocabulary: a Morgan/circular algorithm that encodes, for each
atom, the topological environment up to a given radius as a hashed integer. The bit vector
$V$ has a fixed length $D$ (default 2048), and bit $k$ is set to 1 iff some atom in the
molecule has a hashed circular subgraph that maps to position $k$. The PI's phrase
"repertorio de subgrafos" is an accurate description of this: the fingerprint *is* a
presence/absence encoding over a (hash-implicit) set of circular subgraph patterns.

---

## 2. Metric definitions (verbatim from PMC HTML)

All seven metrics take two binary fingerprint vectors $V_a, V_b \in \{0,1\}^D$ as input.
Notation: $\sum V_i$ = Hamming weight (number of 1-bits in $V_i$); $V_a \cdot V_b$ =
element-wise AND, so $\sum(V_a \cdot V_b) = |A \cap B|$ in set notation; $|V_i|$ = vector
length $D$.

### 2.1 Tanimoto (Jaccard)

$$\operatorname{Tani}(V_a, V_b) = \frac{\sum(V_a \cdot V_b)}{\sum V_a + \sum V_b - \sum(V_a \cdot V_b)}$$

Exactly the Jaccard index $|A \cap B| / |A \cup B|$. Range: $[0, 1]$. **Similarity.**
Extracted from PMC HTML without ambiguity.

### 2.2 Dice (Sørensen–Dice)

$$\operatorname{Dice}(V_a, V_b) = \frac{2 \cdot \sum(V_a \cdot V_b)}{\sum V_a + \sum V_b}$$

Range: $[0, 1]$. **Similarity.** Extracted from PMC HTML without ambiguity.

### 2.3 Cosine

The PMC HTML renders the formula as $V_a \cdot V_b / (|V_a| + |V_b|)$, which is
dimensionally inconsistent (numerator is a vector if $\cdot$ is element-wise, denominator
is a scalar). The canonical chemoinformatics form — and the only one consistent with a
scalar output — is:

$$\operatorname{Coss}(V_a, V_b) = \frac{\sum(V_a \cdot V_b)}{\sqrt{\sum V_a \cdot \sum V_b}}$$

This is the cosine of the angle between $V_a$ and $V_b$ in the binary embedding
$\{0,1\}^D$. Range: $[0, 1]$. **Similarity.** The HTML rendering lost the square-root
structure; the canonical formula is used.

### 2.4 Kulczynski

$$\operatorname{Kulc}(V_a, V_b) = \frac{(\sum V_a + \sum V_b) \cdot \sum(V_a \cdot V_b)}{2 \cdot (\sum V_a)(\sum V_b)}$$

Equivalently $\tfrac{1}{2}\bigl(\tfrac{c}{a} + \tfrac{c}{b}\bigr)$ where
$a = \sum V_a$, $b = \sum V_b$, $c = \sum(V_a \cdot V_b)$.
Range: $[0, 1]$. **Similarity.** Extracted without ambiguity.

### 2.5 McConnaughey

The PMC HTML renders the numerator as $(\sum V_a + \sum V_b)^2 - (\sum V_a)(\sum V_b)$,
which does not match any canonical form. The standard McConnaughey coefficient is:

$$\operatorname{McCo}(V_a, V_b) = \frac{c(a + b) - ab}{ab}$$

where $a = \sum V_a$, $b = \sum V_b$, $c = \sum(V_a \cdot V_b)$. Equivalently
$c/b + c/a - 1$. Range: $[-1, 1]$, value $-1$ when $c = 0$. **Similarity.**
The HTML rendering is discarded in favour of the canonical formula (Todeschini & Consonni,
*Handbook of Molecular Descriptors*, 2000, §15.3).

### 2.6 Russel

$$\operatorname{Russ}(V_a, V_b) = \frac{\sum(V_a \cdot V_b)}{|V_a|}$$

where $|V_a| = D$ = fingerprint length (fixed). Range: $[0, 1]$. **Similarity.**
Extracted without ambiguity.

### 2.7 Sokal

$$\operatorname{Sokal}(V_a, V_b) = \frac{\sum(V_a \cdot V_b)}{2\sum V_a + 2\sum V_b - 3\sum(V_a \cdot V_b)}$$

Range: $[0, 1]$. **Similarity.** Extracted without ambiguity.

---

## 3. Transferability to IsalGraph (unlabelled topology-only)

### 3.1 Structural dependency on labels

All seven metrics operate on a **fingerprint** $V \in \{0,1\}^D$. The fingerprint is an
abstraction of the graph structure computed by an external algorithm. In IsalChem, RDKit's
Morgan algorithm hashes circular subgraphs that include **atom-type labels and bond-order
labels** at every depth level. Removing those labels does not invalidate the metric
formulas — it invalidates the fingerprint. The metrics themselves are label-agnostic; the
fingerprint construction is not.

### 3.2 Per-metric transferability table

| Metric | Transfers? | Condition / Barrier |
|--------|-----------|---------------------|
| Tanimoto | **with adaptation** | Replace RDKit fingerprint with unlabelled graphlet fingerprint; formula unchanged. |
| Dice | **with adaptation** | Same condition as Tanimoto. |
| Cosine | **with adaptation** | Same condition as Tanimoto. |
| Kulczynski | **with adaptation** | Same condition as Tanimoto. |
| McConnaughey | **with adaptation** | Can return negative values — compatible with a dissimilarity comparison only. |
| Russel | **with adaptation** | Requires fixed $D$ (fingerprint dimension). For graphlets, set $D$ = number of graphlet types (up to size $k$). |
| Sokal | **with adaptation** | Same as Tanimoto. |

The **necessary and sufficient adaptation** for every metric is: replace the RDKit
molecular fingerprint with an **unlabelled graphlet fingerprint** — a binary vector of
length $D$ where bit $i$ is 1 iff graphlet type $i$ (from a fixed vocabulary of connected
unlabelled graphs up to $k$ nodes) appears as an induced subgraph in the query graph.

No metric "does not transfer" at the formula level. The barrier is downstream, in the
fingerprint, and it is the same barrier for all seven. See §4 for why that barrier is fatal.

---

## 4. Subgraph repertoire specification — and why none of them clears the size null

### 4.1 Candidate repertoires

For unlabelled graphs, the fingerprint vocabulary candidates are:

| Candidate | Sizes $k$ | Count of types | Notes |
|-----------|-----------|----------------|-------|
| Connected graphlets | $k \le 3$ | 2 types | Path/triangle. Too few — degenerates on sparse graphs (triangle count $= 0$). |
| Connected graphlets | $k \le 4$ | 6 types | Adds 4-path, 4-cycle, fork, diamond. Tractable. |
| Connected graphlets | $k \le 5$ | 21 types | Adds all 5-node connected graphs. Still tractable on $n \le 98$. |
| Frequent subgraphs (gSpan) | data-driven | variable | Requires a support threshold; not reproducible without the training set. |
| WL subtrees | depth $h$ | variable | Already computed as a comparison arm; would not add a new reference. |

Graphlets up to $k = 5$ (21 types) is the natural choice: finite, canonically enumerable
(no hash collision), and well studied (Pržulj, *Bioinformatics* 23(2):e177–e183, 2007).

**Containment check cost.** Subgraph isomorphism is NP-complete in general but tractable
here: for $k \le 5$ and host graphs with $n \le 98$, an exhaustive check over all
$\binom{n}{k}$ induced subsets costs $O\bigl(k!\binom{n}{k}\bigr) \approx 120 \binom{98}{5}
\approx 10^{10}$ operations in the worst case. That is not tractable in a production run
with millions of pairs. **Practicable alternative:** the orbit-based O4 algorithm
(Milenkovic & Pržulj) counts graphlets up to $k = 5$ in $O(n \cdot d^{k-1})$ time where
$d$ is max-degree; on the IAM graphs ($d \le 20$, $n \le 98$) this is $\sim 10^7$ per
graph, feasible but not fast (minutes per cohort).

### 4.2 Size-domination analysis — the decisive criterion

**Claim.** Every graphlet-fingerprint-based similarity metric from §2 will be strongly
size-dominated on the IsalGraph cohorts, with a size null at least as high as the GED
size null on the same cohort.

**Mechanism.** Let $f(G)$ denote the number of 1-bits in the graphlet fingerprint of $G$.
For connected graphs of $n$ nodes and average degree $d$:

- The number of $k$-node connected induced subgraphs scales as $\Theta\bigl(n \cdot d^{k-1}\bigr)$
  for $k \ge 2$ in sparse graphs ($d = O(1)$), i.e., **linearly in $n$**.
- Therefore $f(G) \approx \alpha \cdot n + \beta$ for some dataset-dependent $\alpha, \beta > 0$
  (not all 21 graphlet types are present in every graph — sparse graphs miss dense motifs —
  but the dominant contribution is linear in $n$).
- Tanimoto similarity: $\operatorname{Tani}(G_i, G_j) = |A \cap B| / |A \cup B|$.
  For two graphs of very different sizes, $|A \cap B| \le f(G_{\min})$ while
  $|A \cup B| \ge f(G_{\max})$. So $\operatorname{Tani}(G_i, G_j) \le f(G_{\min})/f(G_{\max})
  \approx n_{\min}/n_{\max}$.
- This bounds Tanimoto from above by a **ratio of node counts** — a size proxy even stricter
  than the absolute difference $|n_i - n_j|$.

**Quantitative projection for COIL-DEL** ($n_i \in [2, 98]$, size null $\rho = 0.9971$
against GED). The graphlet fingerprint similarity can only range from $n_{\min}/n_{\max}$
(worst case) to 1 (identity). Spearman correlation between $n_{\min}/n_{\max}$ and GED is
mechanically high when GED itself tracks size. On COIL-DEL, any graphlet-based Tanimoto
will inherit the same size signal — the rank correlation with $|n_i - n_j|$ will be at
least as high as GED's $\rho = 0.9971$.

**Frequency normalization does not rescue the metrics.** If we switch from
presence/absence (bit) to frequency fingerprints $f_i = c_i / \sum_j c_j$ where $c_i$ =
count of graphlet type $i$, then:
1. The formula for Tanimoto (etc.) must be re-interpreted for real-valued vectors, which
   changes the metric from what the paper defines.
2. Even so, the frequency distribution of graphlet types in sparse random graphs
   concentrates on paths and stars regardless of $n$, so two graphs of different sizes
   will have nearly identical frequency vectors — collapsing similarity to near 1.0 for
   all pairs, not just same-size ones. This is the opposite of size-domination but equally
   uninformative.
3. The only normalisation that would escape both failure modes is a **within-size-class**
   normalisation, which replicates the `equal_n` view already present in T-06 — it doesn't
   add a new result.

**Letter LOW/MED exception.** These two cohorts have $n \in \{3, \ldots, 9\}$ with most
graphs at 4–5 nodes. Size variation is small and the GED size null is lower (exact figures
are cohort-specific but Letter LOW is known to have $\rho_{\text{GED}} \approx 0.3$–$0.5$).
A graphlet fingerprint *might* carry structural signal beyond size on these cohorts.
However:
- Letter LOW/MED are precisely the cohorts where IsalGraph already performs acceptably
  (the paper's §5.4 concern is the bulk-loss finding, dominated by the large/heterogeneous
  cohorts).
- A metric that only helps on the already-acceptable cohorts does not address the reviewer's
  concern.
- Implementing graphlets for Letter LOW/MED alone would be scientifically cherry-picked
  and would not survive review scrutiny.

### 4.3 Recommendation

**Do not implement any of the seven metrics.** The route is foreclosed by size-domination,
not by difficulty of implementation. The argument in one sentence:

> The graphlet fingerprint $f(G) \propto n$ on sparse graphs, so every metric from §2
> degrades to a size-ratio proxy; on the cohorts where size already dominates GED
> ($\rho \ge 0.99$), the fingerprint metric will be equally dominated and will not clear
> the mandatory size null; on the cohorts where size is less dominant (Letter LOW/MED),
> IsalGraph already performs acceptably.

If the primary tracks (WL kernel, spectral ESD) also fail, the honest conclusion to report
in the manuscript is that GED on these cohorts is intrinsically size-dominated and no
currently computable graph similarity achieves independence from that confound — which is
itself a shareable finding about the difficulty of the benchmark.

---

## 5. Self-contained forward reference

If a future revision of the manuscript attempts to use fingerprint-based metrics, the
minimum viable approach is:

1. Use graphlets up to $k = 5$ (21 types) via O4 or an equivalent algorithm.
2. Report $\rho(|n_i - n_j|, 1 - \operatorname{Tani})$ alongside every result.
3. Restrict claims to cohorts where the size null is below the metric's size null.
4. Use the `equal_n` view as the primary analysis and `all_pairs` as secondary only.
5. Cite Pržulj (2007) for graphlet vocabulary and Todeschini & Consonni (2000) for metric
   definitions.

None of this has been implemented. It is recorded here as a specification so the option
is not re-evaluated from scratch if needed.

---

## 6. Sources

- Thurnhofer-Hemsi K. et al. (2025). Representation of Molecules by Sequences of
  Instructions. *J. Chem. Inf. Model.* **65**(15):7936–7955. DOI `10.1021/acs.jcim.5c00354`.
  PMC12344769. Open access CC BY. Molecular Similarity section on journal p.7949.
- Pržulj N. (2007). Biological network comparison using graphlet degree distribution.
  *Bioinformatics* **23**(2):e177–e183. DOI `10.1093/bioinformatics/btl301`.
- Todeschini R., Consonni V. (2000). *Handbook of Molecular Descriptors*. Wiley-VCH. §15.3
  (McConnaughey coefficient canonical form).
- Wilson R.C., Zhu P. (2008). A study of graph spectra for comparing graphs and trees.
  *Pattern Recognition* **41**(9):2833–2841. (context: spectral alternative, T-28 primary).
