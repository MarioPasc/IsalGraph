# T-27 — literature table for the GED bound bake-off

**Owner**: track C (`t27-literature`), wave `2026-08-13-t27-bakeoff` · **Base commit** `4a31817`
**Serves**: the two literature tables in the T-27 deliverable (one per bound end), and the
bibliography slots the Editor-in-Chief checks.

This file is the *citation* half of T-27. The *measurement* half is track A's campaign and track B's
analysis. The last column of §5 — "does our measurement agree?" — is deliberately left empty; the
orchestrator fills it once the campaign closes.

---

## 0. What "verified" means in this file

Every bibliographic field below was read off a publisher page, a DOI resolver record, or an
authoritative full text — **never recalled**. Each row names the URL actually read. Where a field
could not be seen, it is marked **UNVERIFIED** together with what was tried.

Two source classes are used and are **not interchangeable**:

| Class | What it licenses |
|---|---|
| **Primary** — the paper that introduces the method | attribution, the authors' own complexity and tightness claims |
| **Secondary** — Blumenthal et al., *VLDB J* 2020, the survey that GEDLIB implements from | a uniform restatement of complexity in one notation, and the cross-method dominance results |

Where the two disagree, both are reported. The survey is *also* the correct citation for any
statement of the form "method X is never tighter than method Y", because those statements are
survey-level results, not claims the individual method papers make about each other.

**Notation** (survey's, used throughout): `n = max{|V^G|,|V^H|}`;
`Δ_min = min{maxdeg(G), maxdeg(H)}`; `Δ_max = max{maxdeg(G), maxdeg(H)}`;
`I` = iteration cap; `K` = beam size / number of initial solutions; `UB` = the initial upper bound.

---

## 1. Master citation table

Ten entries: nine methods (several share a source paper) plus the library.

| # | Serves | Citation | DOI | Verified against |
|---|---|---|---|---|
| C1 | `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT` | D. B. Blumenthal and J. Gamper, "Improved Lower Bounds for Graph Edit Distance", *IEEE Transactions on Knowledge and Data Engineering* **30**(3):503–516, 2018. | `10.1109/TKDE.2017.2772243` | Crossref record via `api.crossref.org/works/10.1109/TKDE.2017.2772243` (publisher-deposited: title, both authors, venue, 30/3, 503–516, 2018, IEEE) |
| C2 | `STAR`, `REFINE` | Z. Zeng, A. K. H. Tung, J. Wang, J. Feng and L. Zhou, "Comparing Stars: On Approximating Graph Edit Distance", *Proceedings of the VLDB Endowment* **2**(1):25–36, 2009. | `10.14778/1687627.1687631` | Crossref record (title deposited in truncated form "Comparing stars"; ACM/VLDB, 2/1, 25–36, 2009) **and** the open-access full text `http://www.vldb.org/pvldb/vol2/vldb09-568.pdf`, whose title page gives the full title and the five authors in this order |
| C3 | `HED` | A. Fischer, C. Y. Suen, V. Frinken, K. Riesen and H. Bunke, "Approximation of graph edit distance based on Hausdorff matching", ***Pattern Recognition*** **48**(2):331–343, 2015. | `10.1016/j.patcog.2014.07.015` | Crossref record (all five authors, venue, 48/2, 331–343, 2015, Elsevier). Cross-confirmed as the method GEDLIB implements by `src/methods/hed.hpp` `@details`, which names this DOI. |
| C4 | `IPFP` | S. Bougleux, L. Brun, V. Carletti, P. Foggia, B. Gaüzère and M. Vento, "Graph edit distance as a quadratic assignment problem", ***Pattern Recognition Letters*** **87**:38–46, 2017. | `10.1016/j.patrec.2016.10.001` | Crossref record (all six authors in this order, venue, vol 87, 38–46, 2017, Elsevier). Cross-confirmed by `src/methods/ipfp.hpp` `@details`, which names this DOI as the method it implements. |
| C5 | `BIPARTITE` | K. Riesen and H. Bunke, "Approximate graph edit distance computation by means of bipartite graph matching", *Image and Vision Computing* **27**(7):950–959, 2009. | `10.1016/j.imavis.2008.04.004` | Crossref record (both authors, venue, 27/7, 950–959, 2009, Elsevier). Cross-confirmed by `src/methods/bipartite.hpp` `@details`. |
| C6 | `BP_BEAM` | K. Riesen, A. Fischer and H. Bunke, "Combining Bipartite Graph Matching and Beam Search for Graph Edit Distance Approximation", *Artificial Neural Networks in Pattern Recognition (ANNPR 2014)*, Lecture Notes in Computer Science, pp. 117–128, Springer, 2014. | `10.1007/978-3-319-11656-3_11` | Crossref record (three authors, book-chapter, LNCS, 117–128, 2014, Springer). **LNCS volume number UNVERIFIED** — see §7. |
| C7 | library | D. B. Blumenthal, S. Bougleux, J. Gamper and L. Brun, "GEDLIB: A C++ Library for Graph Edit Distance Computation", in D. Conte, J.-Y. Ramel, P. Foggia (eds.), *Graph-Based Representations in Pattern Recognition (GbRPR 2019)*, Lecture Notes in Computer Science **11510**:14–24, Springer, Cham, 2019. | `10.1007/978-3-030-20081-7_2` | Springer chapter page `https://link.springer.com/chapter/10.1007/978-3-030-20081-7_2` — its "Cite this paper" block gives the four authors, editors, LNCS vol 11510, pp. 14–24, Springer Cham, print ISBN 978-3-030-20080-0 |
| C8 | complexity + dominance, all nine | D. B. Blumenthal, N. Boria, J. Gamper, S. Bougleux and L. Brun, "Comparing heuristics for graph edit distance computation", *The VLDB Journal* **29**(1):419–458, 2020. | `10.1007/s00778-019-00544-1` | Crossref record (five authors, venue, vol 29, 419–458; Crossref `issued` year is **2019**, online-first — the issue year is 2020) **and** the authors' open-access PDF `https://bougleux.users.greyc.fr/articles/vldbj19comparing-heuristics-ged.pdf`, which is the copy every §2–§4 section number below refers to |
| C9 | `REFINE` variant `K-REFINE` (not used; see §3.2) | N. Boria, D. B. Blumenthal, S. Bougleux and L. Brun, "Improved local search for graph edit distance", *Pattern Recognition Letters* **129**:19–25, 2020. | `10.1016/j.patrec.2019.10.028` | Crossref record (four authors, venue, vol 129, 19–25, 2020, Elsevier) |
| C10 | `BP_BEAM` variant `IBP-Beam` (not used; see §3.4) | M. Ferrer, F. Serratosa and K. Riesen, "A First Step Towards Exact Graph Edit Distance Using Bipartite Graph Matching", *Graph-Based Representations in Pattern Recognition (GbRPR 2015)*, Lecture Notes in Computer Science, pp. 77–86, Springer, 2015. | `10.1007/978-3-319-18224-7_8` | Crossref record (three authors, book-chapter, LNCS, 77–86, 2015, Springer). LNCS volume **UNVERIFIED**. |

**Three of the ten are in the *Pattern Recognition* family** — C3 in *Pattern Recognition* itself,
C4 and C9 in *Pattern Recognition Letters*. See §6.2: this materially strengthens the
Editor-in-Chief venue-fit argument, and C4 is a new instance of it that the plan did not know it had.

### 1.1 What GEDLIB itself says each method implements

Read from the `@details` block of each method header in `dbblumenthal/gedlib`, `master`, at
`https://raw.githubusercontent.com/dbblumenthal/gedlib/master/src/methods/<file>`. This is the
library's own attribution and is the authority for "which paper does the code we ran implement".

| GEDLIB method | Header file | Attributed by GEDLIB to |
|---|---|---|
| `BRANCH` | `branch.hpp` | Blumenthal & Gamper, `10.1109/TKDE.2017.2772243` (C1) |
| `BRANCH_FAST` | `branch_fast.hpp` | same (C1) |
| `BRANCH_TIGHT` | `branch_tight.hpp` | same (C1) |
| `STAR` | `star.hpp` | Zeng et al., `10.14778/1687627.1687631` (C2) |
| `HED` | `hed.hpp` | Fischer et al., `10.1016/j.patcog.2014.07.015` (C3) |
| `IPFP` | `ipfp.hpp` | Bougleux et al. (C4), **plus** three extensions: QAPE (`10.1109/ICPR.2016.7899881`), m-IPFP (Daller et al., `10.5220/0006599901490158`), C-QAP (Blumenthal, Daller, Bougleux, Brun, ICPR 2018) |
| `BIPARTITE` | `bipartite.hpp` | Riesen & Bunke, `10.1016/j.imavis.2008.04.004` (C5) |
| `REFINE` | `refine.hpp` | Zeng et al., `10.14778/1687627.1687631` (C2) — "Implements the methods Refine and K-Refine suggested in:" |
| `BP_BEAM` | `bp_beam.hpp` | Riesen, Fischer & Bunke, `10.1007/978-3-319-11656-3_11` (C6), plus IBP-Beam (C10) |

Note the GEDLIB header for `REFINE` attributes **both** Refine and K-Refine to Zeng et al. The
survey (C8, Table 4) attributes `REFINE` to *two* references — Zeng et al. and Boria et al. — and
`K-REFINE` to **Boria et al. only**. GEDLIB's header is imprecise on K-Refine. Since our
configuration leaves `--max-swap-size` at its default of 2, we run plain **Refine**, so the correct
single citation for what we measure is **C2 (Zeng et al.)**; C9 is the correct citation only if
`--max-swap-size > 2` is ever set.

---

## 2. Lower-bound end — per method

### 2.1 `BRANCH`

| Field | Value |
|---|---|
| **Citation** | C1 — Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018, `10.1109/TKDE.2017.2772243` |
| **Complexity** | LSAPE instance built in `O(\|V^G\|\|V^H\| Δ_min Δ_max²)`; the LSAPE solve is cubic on top. C8 §5.2.3. |
| **Proof status** | **Proven lower bound.** C8 §5.2.3: the construction "ensures that ineq. (2) with ξ :≡ 1 holds for all graphs G, H ∈ G and all edit cost functions c_V and c_E, which implies that BRANCH also computes a lower bound for GED". Holds for **arbitrary** edit costs — no precondition our cost model could violate. |
| **Extra structure** | C8 §5.2.3: for fixed **metric** edit costs the BRANCH lower bound is a **pseudo-metric** on the graph collection — symmetric, non-negative, triangle inequality, and 0 on isomorphic pairs. This is the property `approx_ged.md` §2 reason 3 relies on. |
| **Claimed tightness** | **Proven**: BRANCH ≥ BRANCH_FAST and BRANCH ≥ HED (see §5.1 P1, P2). **Empirical**: in C8's own six-dataset benchmark BRANCH was *not* the tightest lower bound — ADJ-IP, then BRANCH-TIGHT, then F2 were (C8 §9.5, Fig. 10b). |
| **Determinism** | **Deterministic.** `LSAPEBasedMethod`; no randomised option in its base class (`--threads`, `--lsape-model`, `--greedy-method`, `--optimal`, `--centrality-method`, `--centrality-weight`, `--max-num-solutions`). `branch.hpp`: "Does not support any options except for the ones supported by ged::LSAPEBasedMethod." |

### 2.2 `BRANCH_FAST`

| Field | Value |
|---|---|
| **Citation** | C1 (same paper as BRANCH) |
| **Complexity** | `O(max{\|V^G\|,\|V^H\|} Δ_max log Δ_max + \|V^G\|\|V^H\| Δ_min Δ_max)` to build C — one factor of `Δ_max` cheaper than BRANCH, obtained by pre-sorting incident edge labels. C8 §5.2.4. |
| **Proof status** | **Proven lower bound.** C8 §5.2.4: "As ineq. (2) with ξ :≡ 1 holds for each input, BRANCH-FAST returns an upper and a lower bound for GED." Also a pseudo-metric under metric edit costs. |
| **Claimed tightness** | **Proven, and stronger than the brief anticipated.** C8 §5.2.4, verbatim: *"it can be shown that BRANCH-FAST's lower bound is never tighter than the one computed by BRANCH. **For constant edge edit costs, BRANCH and BRANCH-FAST are equivalent.**"* Under D6 the edge edit costs *are* constant (ins = del = 1, sub = 0, label-independent), so on our campaign the relation is **equality, not inequality**. See §5.1 P1. |
| **Determinism** | **Deterministic.** `LSAPEBasedMethod` + one option `--sort-method STD\|COUNTING` (default `COUNTING`), which selects a sorting algorithm, not a random choice. |

### 2.3 `BRANCH_TIGHT`

| Field | Value |
|---|---|
| **Citation** | C1 (same paper) |
| **Complexity** | Anytime/iterative. C8 §8.2: preprocessing `O(N³ Δ_max²)`, one iteration `O(N² Δ_max³ + N³)`, overall `O(N³ Δ_max² + I(N² Δ_max³ + N³))`, where `N = max{\|V^G\|,\|V^H\|}` after dummy-node padding and `I` is the iteration cap. **Exponents reconstructed from a two-column PDF text extraction — see §7.** |
| **Proof status** | **Proven lower bound at every iteration** (this is what "anytime" means here). Padding to `\|V^G\| = \|V^H\| = N` is done with isolated dummy nodes; for metric edit costs the padding is the cheaper one-sided variant. C8 §8.2. |
| **Claimed tightness** | **Empirical**, and favourable: C8 §9.5 / Fig. 10b ranks BRANCH-TIGHT the **second tightest lower bound** of all heuristics surveyed, behind only the ILP-based ADJ-IP, and (§9.5, Fig. 10d) joint-best on the classification coefficient. C8 does **not** prove BRANCH-TIGHT ≥ BRANCH; treat the ordering as empirical. See §5.1 P3. |
| **Determinism** | **Deterministic given its options**, but the options change the *answer*, not just the runtime: `--iterations` (default **20**), `--time-limit` (default 0 = none), `--range`, `--epsilon`, `--regularize NAIVE\|K-FACTOR` (default `NAIVE`), `--threads` (default 1), `--upper-bound NO\|FIRST\|LAST\|BEST` (default `BEST`). No random source. **Pin `--iterations` explicitly**: the default is a truncation, so the reported bound depends on it, and a run that hits `--time-limit` would become wall-clock dependent. |

### 2.4 `STAR`

| Field | Value |
|---|---|
| **Citation** | C2 — Zeng, Tung, Wang, Feng & Zhou, *PVLDB* 2(1):25–36, 2009, `10.14778/1687627.1687631` |
| **Complexity** | C2 §4.2 / p. 421 of the OA PDF: the mapping distance μ and the lower bound `L_m` are both obtained via the Hungarian algorithm, "the complexity of Hungarian algorithm is O(n³)", with an `O(n log n)` sort for the star representations — so **`O(n³)`**, as `approx_ged.md` §2 states. C8 §5.2.6 restates the *construction* cost as `O(max{\|V^G\|,\|V^H\|} Δ_max log Δ_max + \|V^G\|\|V^H\| Δ_min)`, the same as BRANCH-CONST; the cubic LSAPE solve dominates. |
| **Proof status** | **Proven lower bound, with an explicit proof.** C2 **Lemma 4.2** establishes `μ(g₁,g₂) ≤ max{4, max{δ(g₁), δ(g₂)} + 1} · λ(g₁,g₂)` where λ is the true GED and δ the maximum degree; the text immediately after ("Based on Lemma 4.2, μ provides a lower bound L_m of λ") defines the bound as the scaled quantity. C8 §5.2.6 gives the same scaling as `ξ(G,H,c_V,c_E) := 1 / max{4, Δ_max + 1}`. |
| **⚠ Precondition** | **STAR is the only method in the bake-off with a cost-model precondition.** C8 §5.2.6 and Table 4: STAR "requires uniform edit cost functions c_V and c_E and ignores the edge labels of the input graphs" — Table 4's limitation column reads *"ignores edge labels, uniform c_V and c_E"*. Uniform means a single constant `C` with `c_V(α,α') = c_E(β,β') = C` for **all** α ≠ α′ **including ε**. D6 sets substitutions to **0** and insertions/deletions to **1**, which is uniform **only when the graphs carry a single node label and a single edge label** — which is our case, since our graphs are unlabeled and track A attaches a constant dummy label. **If any dataset in the campaign carries genuinely distinct labels, STAR's Lemma 4.2 guarantee does not apply and its "lower bound" may exceed the exact GED.** Track A's per-pair `LB ≤ exact` assertion (CONTRACTS §7 M4) is what catches this; a STAR-only violation means the precondition, not the harness. |
| **Claimed tightness** | The scaling divisor `max{4, Δ_max + 1}` grows with maximum degree, so STAR's bound degrades on dense graphs by construction — this is visible in the bound's definition, not merely observed. C8 offers no proven ordering of STAR against BRANCH. **Empirical**: C8 §9.5, Fig. 10 places STAR mid-field, well behind BRANCH-TIGHT and ADJ-IP. See §5.1 P4. |
| **Determinism** | **Deterministic.** `LSAPEBasedMethod` + `--sort-method STD\|COUNTING` (default `COUNTING`). |

### 2.5 `HED`

| Field | Value |
|---|---|
| **Citation** | C3 — Fischer, Suen, Frinken, Riesen & Bunke, ***Pattern Recognition*** 48(2):331–343, 2015, `10.1016/j.patcog.2014.07.015` |
| **Complexity** | C8 §8.1: HED builds **the same LSAPE instance C as BRANCH**, so the construction is `O(\|V^G\|\|V^H\| Δ_min Δ_max²)`, and since HED then only sums row and column minima instead of solving the assignment, "the overall runtime complexity of HED is `O(\|V^G\|\|V^H\| Δ_min Δ_max²)`" — i.e. **HED avoids the cubic LSAPE solve**, which is the whole point of the method. Fischer et al.'s own statement of the complexity — reported in the literature as quadratic in the number of nodes — is **UNVERIFIED** pending full-text access; see §7. |
| **Proof status** | **Proven lower bound.** C8 §8.1 gives the bound as `LB := 0.5·Σ_i min_k c_{i,k} + 0.5·Σ_k min_i c_{i,k}` and states `LB ≤ LSAPE(C)`, i.e. HED relaxes the assignment constraint of a bound that is already valid. Whether Fischer et al. state it as a numbered theorem in the primary source is **UNVERIFIED**. |
| **Upper bound: none, by design** | C8 §8.1, verbatim: *"in general, LB does not correspond to a feasible LSAPE solution, because of which **HED does not compute an upper bound for GED**."* C8 Table 4 lists HED as `upper bound: no`. **GEDLIB returning `get_upper_bound() = inf` is therefore correct behaviour, not a defect** — see §6.1 correction 3. |
| **Why HED returns half- and quarter-integers** | Two independent halvings compose. (i) BRANCH's LSAPE instance already carries edge costs at **½** — C8 §5.2.3 defines `c_{i,k} := c_V(u_i,v_k) + 0.5·C^{i,k}(π^{i,k})`, because each edge is charged at both of its endpoints. (ii) HED then halves again, taking `0.5·(row minima) + 0.5·(column minima)` (C8 §8.1). The granularity is therefore **0.25**, which is exactly why track A observes 0.50, 1.25 and 1.75 where every other method returns integers. This is arithmetic of the published definition, not a numerical artefact. |
| **Claimed tightness** | **Proven, and it is the loose end of the lower-bound family.** C8 §8.1: `LB ≤ LSAPE(C)` "implies that **HED's lower bound is never tighter than the lower bound computed by BRANCH**". This is a *survey-level* theorem, so cite C8 for it, not C3. See §5.1 P2. |
| **Determinism** | **Deterministic.** `hed.hpp` exposes only `--threads` (default 1), `--lsape-model` (default `ECBP`, and inert unless edge-set distances are `OPTIMAL`), and `--edge-set-distances OPTIMAL\|HED` (default `HED`). No randomness anywhere: HED is a closed-form sum over row and column minima, not a search. Confirmed from GEDLIB's option table rather than inferred from the method family, per the orchestrator's request. |
| **⚠ Configuration** | The **default** `--edge-set-distances HED` is degenerate under D6: our free edge substitution makes the default edge-set distance identically zero, which is why the smoke test read `LB = 0.00`. Track A established that `--edge-set-distances OPTIMAL` yields a genuine non-degenerate bound. **The measured HED cell must record this option in its `meta.options`** (CONTRACTS §4), because the two settings are different estimators and a reader cannot tell them apart from the number alone. |

---

## 3. Upper-bound end — per method

A structural note that governs three of the four rows. `IPFP`, `REFINE` and `BP_BEAM` are all
instantiations of GEDLIB's **LS-GED** paradigm (`ged::LSBasedMethod`), which means they are local
searches *starting from an initial node map*. The initial map is produced by a separate method,
selected by `--initialization-method`, **whose GEDLIB default is `RANDOM`** with
`--randomness REAL` (hardware entropy). See §3.5 — this is the single most consequential finding in
this file for the harness.

### 3.1 `IPFP`

| Field | Value |
|---|---|
| **Citation** | C4 — Bougleux, Brun, Carletti, Foggia, Gaüzère & Vento, ***Pattern Recognition Letters*** **87**:38–46, 2017, `10.1016/j.patrec.2016.10.001`. **This resolves the plan's bare "Bougleux et al., 2017"** — see §6.2. |
| **Complexity** | C8 §7.2.5, per-step: populate `C_k` `O(k\|V^G\|\|V^H\| max{\|V^G\|,\|V^H\|})`; solve LSAPE `O(min{·}² max{·})`; update UB `O(max{·}²)`; step width `α_{k+1}` analytically in `O(\|V^G\|\|V^H\|)`; final integral projection `O(min{·}² max{·})`. **Overall `O(I² \|V^G\|\|V^H\| max{\|V^G\|,\|V^H\|})`** for iteration cap `I` (GEDLIB default `--iterations 100`). |
| **Proof status** | **Proven upper bound, by feasibility — not by an approximation ratio.** C8 §7.2.5: IPFP is a Frank–Wolfe method that converges to a possibly *fractional* local minimum, which "is projected to the closest integral solution X̂ and the upper bound `UB := min{UB, c(X̂)}` is returned". The returned value is the cost of an actual node map, hence an actual edit path, hence an upper bound. **No proven approximation ratio** is claimed. Whether C4 states this as a numbered proposition is pending — see §7. |
| **Claimed tightness** | **Empirical, and the strongest empirical claim in the whole family.** C8 §9.6, verbatim: *"The LS-GED instantiation **IPFP** was Pareto optimal on all datasets, as it **always computed the tightest upper bound**."* Six benchmark datasets, C8's own reimplementation. This is an observation on their benchmark, **not** a proven dominance — nothing forbids another method beating IPFP on a given pair. See §5.2 P5. |
| **Determinism** | **NOT deterministic under GEDLIB defaults.** Two compounding sources: (i) `LSBasedMethod`'s `--initialization-method` defaults to `RANDOM` with `--randomness REAL`; (ii) C8 §7.3.1's **MULTI-START** extension exists precisely because "the quality of the local optimum highly depends on the initialization of the method, which is a general drawback of local search methods", and C8 §9.6/Fig. 12 reports that the tightest upper bounds on all six datasets came from configurations **using** MULTI-START. **This is the direct explanation of the anomaly in the T-27 brief** — `IPFP` returning UB 3.00 on a 4-node instance of true GED 1.00, and disagreeing across machines: a random initial node map on a 4-node graph can land in a poor local optimum, and `--randomness REAL` makes the draw irreproducible. **Pin `--initialization-method` to a deterministic method (e.g. `BRANCH_FAST` or `BIPARTITE`), set `--randomness PSEUDO`, and fix `--initial-solutions`, or the IPFP column is not reproducible.** |

### 3.2 `REFINE`

| Field | Value |
|---|---|
| **Citation** | C2 — Zeng et al., *PVLDB* 2(1):25–36, 2009. **The method is Zeng's, not GEDLIB's** — see §6.3. In C2's own notation it is ρ, "the refined suboptimal value" of the mapping distance τ. C8 Table 4 additionally credits C9 (Boria et al.) for the modern treatment; `K-REFINE`, which we do **not** use, is C9's alone. |
| **Complexity** | C8 §7.2.1: one iteration examines all `O((\|V^G\|+\|V^H\|)²)` swaps (4-cycles in the auxiliary bipartite map graph) at `O(Δ_max)` each → one iteration `O((\|V^G\|+\|V^H\|)² Δ_max)`; since the induced upper bound strictly improves each iteration, overall **`O(UB·(\|V^G\|+\|V^H\|)² Δ_max)` for integral edit costs**, `UB` the initial upper bound. C2's own analysis of ρ gives "at most `O(n⁶)`" — a much weaker bound; **prefer C8's**, and if the manuscript quotes a complexity for REFINE it should cite C8 §7.2.1 rather than C2. |
| **Proof status** | **Proven upper bound, by monotone local search.** Every candidate is a node map, so every value is the cost of a realisable edit path; and REFINE only accepts a swap that *reduces* the induced cost, so its output is `≤` the cost of its initial node map. Termination is guaranteed for integral edit costs (D6 is integral). |
| **Claimed tightness** | **Empirical.** C8 §9.6: "The instantiation REFINE of LS-GED performed well, too, as it was Pareto optimal on all datasets **except for GREC**" and, aggregated over datasets, "NODE, IPFP, and REFINE achieve" the best joint upper-bound scores (§9.6, Fig. 12). Second to IPFP, not equal to it. See §5.2 P6. |
| **Determinism** | **NOT deterministic under GEDLIB defaults** — same `LSBasedMethod` `--initialization-method RANDOM` / `--randomness REAL` inheritance as IPFP. REFINE's own options are `--max-swap-size` (default **2**, i.e. plain Refine; > 2 switches to K-Refine and changes the citation to C9), `--naive` (default FALSE), `--add-dummy-assignment` (default TRUE). **Proven relative to its own initialiser**: `REFINE(π₀) ≤ UB(π₀)` always — so pinning `--initialization-method BIPARTITE` makes `REFINE ≤ BIPARTITE` a *proven*, per-pair checkable relation. See §5.2 P8. |

### 3.3 `BIPARTITE`

| Field | Value |
|---|---|
| **Citation** | C5 — Riesen & Bunke, *Image and Vision Computing* **27**(7):950–959, 2009, `10.1016/j.imavis.2008.04.004` |
| **Complexity** | Cubic in the number of nodes, from the Hungarian/Munkres solve on the `(\|V^G\|+1)×(\|V^H\|+1)` cost matrix. C8 §5.2.2 gives the *construction* cost as `O(\|V^G\|\|V^H\| Δ_min Δ_max²)`; the cubic assignment solve dominates for sparse graphs. C5's own statement of `O(n³)` and its section number is pending — see §7. |
| **Proof status** | **Proven upper bound only — and provably *not* a lower bound.** C8 §5.2.2, verbatim: "This construction does **not** guarantee that ineq. (2) holds, which implies that **BP only returns an upper bound for GED**." This is the precise formal difference between `BIPARTITE` and `BRANCH`: BRANCH is BP with the edge costs halved (C8 §5.2.3, "The only modification is that the edge costs in the construction of the LSAPE instance C are divided by 2"), and that halving is exactly what buys the lower-bound guarantee. **This is why calling `get_lower_bound()` on `BIPARTITE` is meaningless and returns 0.00.** |
| **Claimed tightness** | **Empirical, and it is the loose reference point.** C8 §9.6 / Fig. 11: LSAPE-GED instantiations show "the largest variations" and their upper bounds "greatly depend on the intrinsic difficulty of the datasets", while LS-GED (IPFP, REFINE) supplies the tightest. Consistent with our own prior measurement of +135 % on LINUX (`approx_ged.md` §2). No proven ratio bounds BP's overestimate. |
| **Determinism** | **Deterministic.** `LSAPEBasedMethod`; `bipartite.hpp`: "Does not support any options except for the ones supported by ged::LSAPEBasedMethod." Caveat: the LSAPE solution is not unique in general, so ties are broken by the solver (`--lsape-model`, default `ECBP`); the *value* is deterministic for a fixed solver, which is what we report. |

### 3.4 `BP_BEAM`

| Field | Value |
|---|---|
| **Citation** | C6 — **Riesen, Fischer & Bunke, ANNPR 2014**, LNCS pp. 117–128, `10.1007/978-3-319-11656-3_11`. **This corrects the plan's "Neuhaus & Riesen"** — see §6.4. The `IBP-Beam` extension (`--num-orderings > 1`, default 1, so **off** in our configuration) is C10, Ferrer, Serratosa & Riesen 2015. |
| **Complexity** | C8 §7.2.3: at most `1 + K(\|π\|−1) = O(\|V^G\|+\|V^H\|)` tree nodes are extracted from the priority queue, and constructing the children of each extracted inner node costs `O((\|V^G\|+\|V^H\|) Δ_max)`. The final composed expression was **truncated in our text extraction** — see §7 — but the shape is quadratic in `\|V^G\|+\|V^H\|` and linear in `Δ_max` and the beam size `K`. |
| **Proof status** | **Proven upper bound, by construction.** C8 §7.2.3: BP-BEAM "constructs an output node map π′ with `c(P_{π′}) ≤ c(P_π)`" — it starts from an initial node map, explores only swaps of it, and returns the cheapest map encountered, so the result is always the cost of a realisable edit path and never worse than its own starting point. No approximation ratio. |
| **Claimed tightness** | **Empirical, and unfavourable in C8's benchmark.** BP-BEAM is not among the Pareto-optimal upper-bound heuristics C8 §9.6 names (IPFP on all datasets; NODE on five of six; REFINE on all but GREC). Whether C6 itself claims BP-Beam dominates BP is pending — see §7 — but note that any such claim is only a *proven* dominance when the beam is initialised **from BP**, which is not GEDLIB's default. See §5.2 P7, P8. |
| **Determinism** | **NOT deterministic — doubly so, and one source is in the published algorithm itself.** (i) It inherits `LSBasedMethod`'s `--initialization-method RANDOM` / `--randomness REAL`. (ii) Independently, C8 §7.2.3 states that BP-BEAM "starts by producing a **random ordering** `((u_s, v_s))_{s=1}^{|π|}` of the node assignments contained in π" — the beam search explores swaps in that order, so the result depends on it. **Randomness is intrinsic to the published method, not only to GEDLIB's defaults.** GEDLIB's `--beam-size` default is **5**. `IBP-Beam` (C10) exists precisely to average over several orderings. Pinning `--randomness PSEUDO` and a seed is mandatory for this cell; even then the reported value is one draw from a distribution, and that should be said in the table caption. |

### 3.5 The `LSBasedMethod` default — the harness-critical finding

Read from `https://raw.githubusercontent.com/dbblumenthal/gedlib/master/src/methods/ls_based_method.hpp`,
`@details` option table. Every LS-GED method — **`IPFP`, `REFINE`, `BP_BEAM`** — inherits it.

| Option | Default | Consequence |
|---|---|---|
| `--initialization-method` | **`RANDOM`** | the local search starts from a **random node map**, not from a bipartite assignment |
| `--randomness` | **`REAL`** | real (hardware) randomness, so the draw is **not reproducible across runs or machines** |
| `--initial-solutions` | 1 | one restart; C8 §7.3.1's MULTI-START uses `K > 1` |
| `--ratio-runs-from-initial-solutions` | 1 | — |
| `--num-randpost-loops` | 0 | RANDPOST (C8 §7.3.2) off by default |
| `--lower-bound-method` | `NONE` | no independent termination bound |
| `--threads` | 1 | — |

**This explains the brief's `IPFP` anomaly exactly**: UB 3.00 on a 4-node instance of true GED 1.00,
irreproducible across machines. It is not a build difference and not a GEDLIB bug — it is the
documented default. Three consequences for the campaign:

1. The three LS-GED cells must set `--initialization-method` to a **deterministic** method and
   `--randomness PSEUDO`, and record the exact option string in `meta.options`
   (CONTRACTS §4) and in the determinism probe (CONTRACTS §6).
2. Choosing `--initialization-method BIPARTITE` converts three empirical comparisons into
   **proven per-pair inequalities** (§5.2 P8), which is free validation of the harness.
3. Any published claim that IPFP is the tightest upper bound (C8 §9.6) was measured **with**
   MULTI-START. A single-start IPFP is a weaker configuration than the one that won C8's benchmark,
   and the T-27 write-up must say which one it measured.

---

## 4. GEDLIB itself

| Field | Value |
|---|---|
| **Citation** | C7 — Blumenthal, Bougleux, Gamper & Brun, "GEDLIB: A C++ Library for Graph Edit Distance Computation", in Conte, Ramel & Foggia (eds.), *Graph-Based Representations in Pattern Recognition* (GbRPR 2019), LNCS **11510**:14–24, Springer Cham, 2019, `10.1007/978-3-030-20081-7_2` |
| **Verified against** | the Springer chapter page's "Cite this paper" block and its bibliographic footer (print ISBN 978-3-030-20080-0, online ISBN 978-3-030-20081-7, published 16 May 2019) |
| **Why it is the right engine to cite** | Three of the four GEDLIB authors are authors of C1 (Blumenthal, Gamper) and C4/C8 (Bougleux, Brun). The library is the **reference implementation by the authors of the bounds we report**, which is the argument that makes our numbers defensible to R3.5b. |
| **Relationship to C8** | C8 is the survey whose taxonomy GEDLIB implements; C8 §9's experiments were run *in* GEDLIB. Citing both is correct and not redundant: C7 is the software, C8 is the method comparison. |
| **What we actually ran** | not `dbblumenthal/gedlib` directly but the maintained fork bundled by `jajupmochi/graphkit-learn` (see `gedlib.md` §2). The method sources are the same files quoted throughout this document. **Record the `graphkit-learn` commit SHA in `meta.gedlib_commit`** (CONTRACTS §4) — that, not the paper, is what pins the numbers. |

---

## 5. Published tightness ordering — testable predictions

Each prediction is labelled **PROVEN** or **EMPIRICAL**, and that label decides how to read a
disagreement:

- A **PROVEN** prediction must hold on **every one of the 3.9 M pairs**. A single violation is a
  **bug in our harness** (or, for P4, a violated precondition) — never a finding, and never
  something to report as a result.
- An **EMPIRICAL** prediction is an observation someone else made on *their* benchmark. Ours may
  disagree. Disagreement is then a **finding**, and at 3.9 M pairs across five datasets it is a
  finding worth reporting, because it is measured at a scale the original claim was not.

The final column is left empty on purpose; the orchestrator fills it after track B's analysis.

### 5.1 Lower-bound end

| # | Prediction | Kind | Source | Our measurement agrees? |
|---|---|---|---|---|
| **P1** | **`BRANCH` = `BRANCH_FAST` exactly, on every pair.** The general result is `BRANCH ≥ BRANCH_FAST`, but C8 §5.2.4 adds: "For constant edge edit costs, BRANCH and BRANCH-FAST are equivalent." D6 has constant edge edit costs, so the inequality collapses to equality for us. | **PROVEN** (equality, under D6's precondition) | C8 §5.2.4 | |
| **P2** | **`BRANCH` ≥ `HED` on every pair.** HED sums row and column minima of the *same* LSAPE instance BRANCH solves, and `LB ≤ LSAPE(C)`. | **PROVEN** | C8 §8.1 | |
| **P3** | `BRANCH_TIGHT` ≥ `BRANCH` on average. C8 ranks BRANCH-TIGHT second tightest overall and BRANCH mid-field. | **EMPIRICAL** | C8 §9.5, Fig. 10b | |
| **P4** | `STAR` ≤ exact GED on every pair, and `STAR` degrades as `Δ_max` grows (its divisor is `max{4, Δ_max+1}`). | **PROVEN**, *conditional on uniform edit costs* (C2 Lemma 4.2). If our labels are non-trivial the precondition fails and validity is not guaranteed — see §2.4. | C2 Lemma 4.2; C8 §5.2.6 | |
| **P5** | No ordering between `STAR` and `BRANCH` is proven in either direction. | — (absence of a claim) | C8 | |

**P1 is the cheapest validation of the whole harness**, as the brief anticipated — and it is
stronger than expected. The brief asked whether `BRANCH ≥ BRANCH_FAST` is proven; it is, but under
our own cost model the literature gives **equality**, which is a far sharper test: an exact
elementwise match over 3.9 M pairs, with **zero** tolerance. Run it first. If P1 fails, nothing
downstream is trustworthy and the cause is upstream of any tightness question. P2 is the natural
second gate, and P1+P2 together exercise the lower-bound accessor on three of the five LB cells.

### 5.2 Upper-bound end

| # | Prediction | Kind | Source | Our measurement agrees? |
|---|---|---|---|---|
| **P6** | **`IPFP` is the tightest upper bound**, on every dataset. C8's single strongest empirical statement. | **EMPIRICAL** | C8 §9.6 ("Pareto optimal on all datasets, as it always computed the tightest upper bound") | |
| **P7** | `REFINE` is second — Pareto optimal on all datasets except GREC. | **EMPIRICAL** | C8 §9.6 | |
| **P8** | `BIPARTITE` is the loosest of the four. | **EMPIRICAL** | C8 §9.6, Fig. 11; and our own +135 % on LINUX (`approx_ged.md` §2) | |
| **P9** | Every UB method returns a value **≥ exact GED** on every pair — all four construct a realisable edit path. | **PROVEN** | C8 §5.2.2 (BP), §7.2.1 (REFINE), §7.2.3 (BP-BEAM), §7.2.5 (IPFP) | |
| **P10** | **If** `--initialization-method BIPARTITE` is pinned, then `REFINE ≤ BIPARTITE` and `BP_BEAM ≤ BIPARTITE` on every pair, since both are monotone local searches over their initial node map. | **PROVEN**, *conditional on that option being set* | C8 §7.2.1, §7.2.3 | |

P6 is the prediction most likely to break on our data, and breaking it is the more interesting
outcome. C8 measured six IAM-style benchmarks with **MULTI-START enabled**; we measure five
datasets including LINUX and AIDS, under a cost model with free substitutions, and — unless the
harness pins it — with a **single random start**. A single-start IPFP losing to REFINE or BIPARTITE
on our data would not contradict C8; it would show that IPFP's published advantage is contingent on
multi-start, which is a legitimate, citable finding. **The T-27 write-up must therefore state the
IPFP configuration prominently**, or a reviewer will read a bare "IPFP was not tightest" as
contradicting C8 when it does not.

P10 is worth taking deliberately: it costs one option string and converts two of the four
upper-bound cells into self-checking columns.

### 5.3 Cross-end

| # | Prediction | Kind | Source |
|---|---|---|---|
| **P11** | `LB ≤ exact ≤ UB` for every method and every certified pair. | **PROVEN** (P4's precondition aside) | all of the above |
| **P12** | The best LB and best UB bracket the exact value within a few per cent on small graphs — C8 §9.7 reports the gap "at most 4.23 % and only 1.99 % on average" over six datasets. | **EMPIRICAL** | C8 §9.7 | |

P12 is a useful sanity scale for the T-27 figure, but note it is *their* datasets and *their* per-dataset
cost models; our +135 % BIPARTITE result on LINUX already shows our corpus is harder, so do not
present 4.23 % as an expectation.

---

## 6. Corrections found to the plan

Recorded here, not applied — `approx_ged.md` and `gedlib.md` are the orchestrator's, per the brief.

### 6.1 `approx_ged.md` §5, "References to cite" — three defects

1. **"Bougleux et al., 2017 (IPFP — our UB)" has no venue, volume or pages.** Correct and complete
   reference is **C4**: *Pattern Recognition Letters* **87**:38–46, 2017, `10.1016/j.patrec.2016.10.001`.
   It cannot ship in a *Pattern Recognition* revision in its current form.
2. **"Zeng et al., *VLDB* 2009" should be "*Proceedings of the VLDB Endowment* **2**(1):25–36, 2009",
   DOI `10.14778/1687627.1687631`.** "VLDB 2009" reads as the conference; the paper is in the
   *journal* PVLDB. §5 also lists Zeng only against `STAR`, but the same paper is the source of
   **`REFINE`** — one citation, two methods (§6.3).
3. **The GEDLIB entry lacks its LNCS volume**: "Blumenthal et al., GbRPR 2019" → **LNCS 11510:14–24**,
   eds. Conte, Ramel & Foggia, Springer Cham, `10.1007/978-3-030-20081-7_2`.
4. **`approx_ged.md` §2 lists `BP_BEAM` as "Neuhaus & Riesen"** with no year or venue. This is
   **wrong** — see §6.4.
5. **§5 omits C8 entirely.** Every complexity figure in `approx_ged.md` §2's tables, and both
   dominance claims in its reason 1 (`BED ≥ LED`, `BED ≥ HED`), come from the *VLDB Journal* survey,
   not from C1. C8 must be in the bibliography or those numbers are uncited.

Also, `approx_ged.md` §2 gives BRANCH_FAST `O(n²Δ² + n³)` and BRANCH `O(n²Δ³ + n³)`. C8's
statements are `O(max{|V^G|,|V^H|}Δ_max log Δ_max + |V^G||V^H|Δ_min Δ_max)` and
`O(|V^G||V^H|Δ_min Δ_max²)` respectively (plus the cubic solve). The plan's forms are a defensible
simplification with `n = |V|`, `Δ = Δ_max`, but they **drop the `Δ_max log Δ_max` sorting term** and
collapse `Δ_min`/`Δ_max`. If a complexity appears in the manuscript, quote C8's form and cite
C8 §5.2.3/§5.2.4.

### 6.2 `IPFP` is in a *Pattern Recognition* family journal — a second EiC.b slot

The brief asked to flag this prominently if true. **It is true.** C4 is in ***Pattern Recognition
Letters*** **87**:38–46, 2017. So is C9 (*PRL* **129**:19–25, 2020), the modern REFINE reference.
Together with C3 (HED, in ***Pattern Recognition*** **48**(2):331–343, 2015), the revision can cite
**three** Elsevier *Pattern Recognition*-family papers as the direct sources of its GED bounds, of
which one is the manuscript's own venue.

This changes the shape of the EiC.b argument. `approx_ged.md` §5 currently treats HED as the single
"venue fit" citation and — before track A's fix — HED carried no number. Now HED carries a
measurement **and** IPFP, the currently-nominated primary upper bound, is itself PRL. The venue-fit
case no longer depends on one method.

### 6.3 `REFINE` is Zeng's method, not GEDLIB's

The plan's "Zeng et al. 2009 / GEDLIB" is ambiguous, and the ambiguity resolves in Zeng's favour.
GEDLIB's `refine.hpp` states plainly: "Implements the methods Refine and K-Refine suggested in:
Z. Zeng, A. K. H. Tung, J. Wang, J. Feng, and L. Zhou: 'Comparing stars: On approximating graph
edit distance'". So **`REFINE` and `STAR` come from the same paper** — C2 — and the plan should
cite it once for both. Two refinements:

- C8 Table 4 credits `REFINE` to **two** references, Zeng et al. **and** Boria et al. (C9), the
  latter being the modern local-search treatment whose complexity analysis §3.2 quotes.
- C8 Table 4 credits **`K-REFINE` to Boria et al. alone**. GEDLIB's header, which lumps K-Refine in
  with Zeng, is imprecise. Immaterial for us — `--max-swap-size` defaults to 2 — but it must not be
  copied into the bibliography.

### 6.4 `BP_BEAM` is Riesen, Fischer & Bunke 2014 — not "Neuhaus & Riesen"

`approx_ged.md` §2 lists `BP_BEAM` as "Neuhaus & Riesen" with no year or venue. GEDLIB's
`bp_beam.hpp` attributes the implemented method to **K. Riesen, A. Fischer and H. Bunke**,
"Combining bipartite graph matching and beam search for graph edit distance approximation",
`10.1007/978-3-319-11656-3_11` (ANNPR 2014) — C6 — and C8 Table 4 lists `BP-BEAM` under reference
[56], the same work. The attribution in the plan is **wrong** and must be replaced. (Beam-search
GED does trace back to earlier Neuhaus/Riesen/Bunke work, but that is not the algorithm GEDLIB
implements under this name, and citing it would misattribute the code we ran.)

### 6.5 `gedlib.md` §5 — HED is resolved, and one line is a misdiagnosis

`gedlib.md` §5 says HED is "**unresolved, not usable until diagnosed**, most likely it needs
explicit method options", and the capability matrix flags `get_upper_bound() = inf` as suspect.
Two separate corrections, both confirmed against C8 §8.1 and GEDLIB's `hed.hpp`:

1. **`UB = inf` is correct behaviour, not a defect.** C8 §8.1: HED's bound "does not correspond to a
   feasible LSAPE solution, because of which HED does not compute an upper bound for GED", and C8
   Table 4 lists HED as `upper bound: no`. HED is a **lower-bound-only** method by construction.
   `gedlib.md` §4 "Trap 2" should move `HED` into the lower-bound row of its capability table rather
   than listing it as an anomaly.
2. **The "needs explicit method options" guess was right, but for a reason worth writing down.**
   The option is `--edge-set-distances OPTIMAL` (default `HED`), and the reason the default
   degenerates is D6-specific: free edge substitution makes the default edge-set distance
   identically zero. This is a *cost-model interaction*, not a library defect, and it would not
   reproduce under the IAM per-dataset cost models.

`gedlib.md` §5's `HED` row should therefore read: **LB** capability, `0.00` under default options
**with the reason**, non-degenerate under `--edge-set-distances OPTIMAL`, `UB = inf` **by design**.

### 6.6 One item for the harness, from the literature rather than from testing

`STAR` carries a cost-model **precondition** — uniform `c_V` and `c_E` (C8 §5.2.6, Table 4) — that
no other method in the bake-off has, and D6 satisfies it only because our graphs are effectively
unlabeled. This belongs in the T-27 write-up next to the STAR row, and `statistics.md` D6 should
note it, because D6 is otherwise presented as universally applicable.

---

## 7. UNVERIFIED — what could not be checked, and what was tried

| Item | Status | What was tried |
|---|---|---|
| **C6 LNCS volume number** | **UNVERIFIED** | Crossref returns `container-title: ["Lecture Notes in Computer Science", "Advanced Information Systems Engineering"]` with `volume: null` and ISBNs `9783642387081` / `9783642387098`, which belong to a **different** book — the Crossref record for this chapter is polluted. Page range (117–128), year, authors and DOI are solid. A `read-paper` agent was dispatched to the Springer chapter page to read the volume from the book front matter. Until it is confirmed, **cite C6 without a volume number** rather than guessing one. |
| **C10 LNCS volume number** | **UNVERIFIED** | Same Crossref limitation. C10 is a footnote only (IBP-Beam is off by default), so this is low priority. |
| **C3 (HED) — Fischer et al.'s own complexity statement and theorem number** | **UNVERIFIED** | Elsevier paywall. The complexity reported in §2.5 is **C8 §8.1's restatement**, which is safe to cite as such. A `read-paper` agent was dispatched to look for an author preprint. Do **not** write "Fischer et al. prove HED is a lower bound in Theorem N" until the theorem number is seen. |
| **C4 (IPFP) — proposition number for the upper-bound guarantee; whether it is a special issue** | **UNVERIFIED** | Same; a `read-paper` agent was dispatched to HAL and the authors' pages, where C4 is likely open access. §3.1's proof-status entry is C8 §7.2.5's restatement. |
| **C5 (BIPARTITE) — Riesen & Bunke's own `O(n³)` statement and its section number** | **UNVERIFIED** | Elsevier paywall; agent dispatched. C8 §5.2.2 covers the construction cost and the "upper bound only" result, which is the load-bearing claim. |
| **C6 — whether the paper claims BP-Beam always dominates BP** | **UNVERIFIED** | Agent dispatched. Note §3.4: even if claimed, the dominance is conditional on initialising *from BP*, which GEDLIB does not do by default, so it cannot be quoted unconditionally regardless of what C6 says. |
| **`BRANCH_TIGHT` complexity exponents** | **partially verified** | Read from `pdftotext` output of C8's two-column PDF, where superscripts detach from their base. Reconstructed as `O(N³Δ_max² + I(N²Δ_max³ + N³))`. The *shape* is certain; the exponents should be confirmed against the typeset PDF before appearing in the manuscript. |
| **`BP_BEAM` composed complexity** | **partially verified** | C8 §7.2.3's per-component costs are quoted verbatim in §3.4; the final composed expression was truncated by the extraction. |
| **C1 (TKDE 2018) full text** | **not read** | No open-access copy located (no arXiv record for the title; searches returned only citing works). All C1-derived statements in this file are taken from **C8**, the survey by the same first author, and are attributed to C8 rather than to C1. The C1 bibliographic record itself is fully verified via Crossref. |

Nothing in this file has been filled in from memory. Where only C8 was readable, the claim is
attributed to C8.

---

## 8. Ready-to-use BibTeX keys

Suggested keys, for whoever writes the `.bib`. Fields are exactly as verified above; **do not add a
volume to `riesen2014bpbeam` until §7 clears it**.

```
blumenthal2018branch     C1   IEEE TKDE 30(3):503--516, 2018
zeng2009stars            C2   PVLDB 2(1):25--36, 2009            [STAR and REFINE]
fischer2015hed           C3   Pattern Recognition 48(2):331--343, 2015
bougleux2017ipfp         C4   Pattern Recognition Letters 87:38--46, 2017
riesen2009bipartite      C5   Image and Vision Computing 27(7):950--959, 2009
riesen2014bpbeam         C6   ANNPR 2014, LNCS, pp. 117--128     [volume UNVERIFIED]
blumenthal2019gedlib     C7   GbRPR 2019, LNCS 11510:14--24
blumenthal2020comparing  C8   The VLDB Journal 29(1):419--458, 2020
boria2020krefine         C9   Pattern Recognition Letters 129:19--25, 2020   [only if --max-swap-size > 2]
ferrer2015ibpbeam        C10  GbRPR 2015, LNCS, pp. 77--86       [only if --num-orderings > 1]
```
