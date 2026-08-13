# Competitors — one file per representation, backed by measurement

**Owner**: T-04 · **Gates**: T-04a, T-06, T-17
**Parent**: [competitors](../competitors.md) — the *decision* file. This folder is the *evidence*
behind it. Where the two disagree, §4 below says which wins and who owns the fix.

**All numbers on this page were measured on the local workstation on 2026-08-13.** The scripts and
their raw output are preserved in **[`scratch/`](scratch/)** beside this file — `min_dfs.py`,
`agm_cam.py`, `backends.py`, `validate_min_dfs.py`, `test_kavin.py`, `probe.py`, `sweep.py`,
`scale.py`, `ceiling.py`, `isal_ceiling.py`, `stability.py`, plus every `.out` and `.json` these
tables are read from. **Nothing touches `src/`**; §5 says what must be ported and what must not.

Environment: `networkx` 3.6.1, `pynauty` 2.8.8.1, `grakel` 0.1.8, `numpy` 1.26.4, gcc 12.2.0,
`isalgraph.engine() == "cpp"`.

> **Two samplers, and they are not interchangeable.** The cohort table uses rejection sampling on
> `G(n, m)` conditioned on connectivity — uniform, but it does not terminate at `m ≈ n` above
> `n ≈ 30`. The ceiling table uses a random spanning tree plus uniform extra edges, which
> terminates but over-weights tree-like structure. Rows are labelled with which one produced them.

> ⚠ **The graphs are synthetic.** The IAM corpus is not on this workstation, so every profile is
> `G(n, m)` at T-01's **measured per-dataset `n̄` and `m̄`** ([data](../data.md) §1.2). Orderings
> that depend on graph *family* — Letter graphs are near-planar geometric graphs, not random — are
> **not** settled here. **T-04a on the real 200-graph sample is what settles them.** Everything that
> is a property rather than a distribution (canonicity, completeness, reversibility, feasibility
> ceilings, API behaviour) transfers unchanged.

---

## 1. The files

| File | Competitor | Verdict |
|---|---|---|
| [graph6](graph6.md) | McKay's packed adjacency serialisation | **RUN — negative control** |
| [sparse6](sparse6.md) | McKay's edge-list serialisation | **RUN** — the compactness rival |
| [nauty](nauty.md) | canonical labelling → graph6 (+ the bliss/Traces cut) | **RUN** — the fair canonical serialisation |
| [adjacency-matrix](adjacency-matrix.md) | raw upper triangle | **RUN — reference point** |
| [agm](agm.md) | AGM canonical adjacency-matrix code (CAM) | **RUN, Suite 1 only** — ceiling at `n ≈ 14` |
| [gspan-mdfsc](gspan-mdfsc.md) | gSpan minimum DFS code | **RUN** — the real competitor |
| [wl-subtree-kernel](wl-subtree-kernel.md) | Weisfeiler–Lehman subtree kernel | **RUN, Claim B only** |

Each file answers the same five questions in the same order: reproducibility, representation,
distance, Claim A fit, scope alignment — then a summary table and integration notes.

---

## 2. The pool is not seven arbitrary methods

Stating this in the paper is most of what AE.3 is asking for, and it costs a paragraph.

**Family I — the `n²` serialisations.** All four emit *the same bit sequence*: the strict upper
triangle of the adjacency matrix, read column-wise. They differ on exactly two orthogonal choices.

| | raw bits | 6-bit ASCII packing |
|---|---|---|
| **incident labelling** | [adjacency matrix](adjacency-matrix.md) | [graph6](graph6.md) |
| **nauty canonical labelling** | — | [nauty→graph6](nauty.md) |
| **lex-min labelling** | [AGM CAM](agm.md) | — |

Consequences that must reach the paper:

- **All four have the same Claim A bit count**, `n(n−1)/2`, up to header and padding.
  **Print one `n²` row with a footnote, not four identical columns.**
- The pool therefore isolates **canonicity as a variable at fixed format**: graph6 vs
  nauty→graph6 changes only the labelling, and separation moves 1.00 → 0.83 while invariance moves
  0/40 → 40/40. That subtraction *is* R1.2's uniqueness answer.

**Family II — the mining-literature canonical forms.** Jiang, Coenen & Zito
(*Knowledge Engineering Review* 28(1):75–105, 2013) classify every frequent-subgraph miner's
representation into **CAM** (AGM, FSG, FFSM) and **M-DFSC** (gSpan). R1.2 named one of each.
Covering both with a measured row turns "we cite AGM and gSpan" into "we compared against both
canonical-representation families in the mining literature".

**Outliers.** [sparse6](sparse6.md) is the only non-canonical format whose length scales with `m`,
which makes it IsalGraph's compactness rival. [WL](wl-subtree-kernel.md) is not a serialisation at
all and enters Claim B only.

---

## 3. The master table

Separation `= median Levenshtein on one-edit pairs / median on random same-`n` pairs`, 120 + 120
pairs, `n ∈ [6,12]`. **Lower is better**; 1.00 means a one-edit pair is indistinguishable from an
unrelated one. F3 = isomorphism invariance, 40 graphs × 25 genuine relabellings.

| Representation | Reproducible? | F3 | Complete invariant | Primary distance | **Sep.** | Claim A bits | Feasibility ceiling |
|---|---|---|---|---|---:|---|---|
| [graph6](graph6.md) | **trivial** (`networkx`) | **0/40** | no | *none admissible* | **1.00** | `6(1+⌈n(n−1)/12⌉)` | none |
| [sparse6](sparse6.md) | **trivial** (`networkx`) | **0/40** | no | *none admissible* | 0.88 | `6·len`, scales with `m` | none |
| [adjacency](adjacency-matrix.md) | **trivial** (none) | **0/40** | no | *none admissible* | 0.92 | `n(n−1)/2` | none |
| [nauty→graph6](nauty.md) | `pip install pynauty`, **from-source build verified** | **40/40** | **yes** | padded Hamming | 0.83 | = graph6 | none observed |
| [AGM CAM](agm.md) | **no package — we wrote it**, validated vs brute force on 327 graphs | **40/40** | **yes** | padded Hamming | **0.50** | = adjacency | **`n ≈ 14`** |
| [min-DFS code](gspan-mdfsc.md) | **3 repos tested, all 3 rejected — we wrote it**, validated vs exhaustive enumeration | **40/40** | **yes** | Levenshtein (tuple) | **0.32 / 0.38** | `m·2⌈log₂ n⌉` | none to `n = 98` (124 ms, Python) |
| [WL subtree](wl-subtree-kernel.md) | `grakel` **already installed** | **40/40** | **no** — `d(K₃,₃, prism) = 0` | kernel (**pseudometric**) | — | **none** (not reversible) | none |
| *IsalGraph canonical* | — | 40/40 | yes | Levenshtein | *0.69* | `L log₂ 9` | **`n ≈ 50`, and COIL-DEL at `n = 22`** |
| *IsalGraph pruned* | — | 40/40 | yes | Levenshtein | *0.73* | `L log₂ 9` | none to `n = 70` |
| ~~bliss / Traces~~ | — | — | — | — | — | — | **CUT**, see [nauty](nauty.md) §8 |

### The two orderings that decide the paper

**Message length** — entropy-bound bits, median over Suite-2 profiles:

| Profile | `n` | `m` | adjacency | graph6 | sparse6 | min-DFS | **IsalGraph pruned** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 4 | 3 | **6** | 12 | 24 | 12 | 13 |
| LINUX | 9 | 8 | **36** | 42 | 66 | 64 | 38 |
| AIDS (IAM) | 14 | 15 | 91 | 102 | 102 | 120 | **82** |
| COIL-DEL | 22 | 54 | **231** | 240 | 348 | 540 | 418 |
| Mutagenicity | 29 | 30 | 406 | 414 | 222 | 300 | **181** |
| Protein | 32 | 61 | 496 | 504 | **396** | 610 | 533 |
Ceiling sweep to Suite 2's `n_max = 98` (spanning-tree sampler, see §6):

| `n` | `m` | `m/n` | adjacency | graph6 | sparse6 | min-DFS | **IsalGraph pruned** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 31 | 1.03 | 435 | 444 | 234 | 310 | **187** |
| 30 | 60 | 2.00 | 435 | 444 | **384** | 600 | 561 |
| 50 | 52 | 1.04 | 1225 | 1236 | 426 | 624 | **352** |
| 50 | 100 | 2.00 | 1225 | 1236 | **744** | 1200 | 1024 |
| 70 | 73 | 1.04 | 2415 | 2442 | 714 | 1022 | **539** |
| **98** | **102–103** | **1.04** | 4753 | 4782 | 978 | 1428 | **888** |
| 98 | 196 | 2.00 | 4753 | 4782 | **1698** | 2744 | — |

> **IsalGraph's bit cost is governed by `m`; adjacency, graph6, nauty→graph6 and AGM are governed
> by `n²`.** So IsalGraph loses below `n ≈ 14` (where `n(n−1)/2` is simply small) and on dense
> graphs, and wins on large sparse ones — the regime AE.1 asked us to extend into. At `n = 98,
> m ≈ n` it needs **888 bits against adjacency's 4,753 and graph6's 4,782**. It beats **min-DFS on
> 9 of 10 cohort profiles and at every ceiling profile**, and beats **sparse6 whenever `m/n ≲ 1.5`**,
> losing to it at `m/n ≈ 2`.

Encode time at `n = 98, m = 102`, median of 3: **sparse6 0.31 ms · IsalGraph pruned 0.95 ms ·
graph6 2.11 ms · nauty→graph6 2.10 ms · min-DFS 124 ms (pure Python) · AGM does not terminate.**

**GED tracking** — separation, best to worst:
**min-DFS 0.32 · AGM 0.50 · IsalGraph 0.69–0.73 · nauty 0.83 · sparse6 0.88 · adjacency 0.92 ·
graph6 1.00.**

> **The minimum DFS code tracks a unit edit more than twice as tightly as IsalGraph on this test.**
> On synthetic `G(n,m)` at `n ≤ 12`, not on IAM. It is a prior, not a verdict — but it is a strong
> one and T-04a must resolve it before the paper claims either direction.

---

## 4. What this folder changes in the plan

Each item names the file that must be edited and the ticket that owns it.

| # | Finding | Edit | Owner |
|---|---|---|---|
| **1** | **[competitors](../competitors.md) §4 outcome 3 is inverted.** It pre-commits "sparse6 beats IsalGraph on bits for sparse graphs". Measured, IsalGraph **wins** at `m/n ≈ 1` (Mutagenicity 181 vs 222) and loses at `m/n ≈ 2` (Protein 533 vs 396) | rewrite the pre-commitment as the `m`-versus-`n²` statement in §3 above | T-04 → T-20 |
| **2** | **AGM's canonical code is not computable above `n ≈ 14`** — exact 5/5 to `n = 11`, 3/5 at `n = 14`, 0/5 at `n ≥ 20` under a 300k-node budget. **AGM runs on Suite 1 only** | [agm](agm.md) §2.2; §2's "1 d, derive from nauty labelling" is wrong on both halves | T-04, T-17 |
| **3** | **[preregistration](../preregistration.md) §5's reduction rule has no case for a representation computable on one suite and not the other.** AGM keeps 5 B1e rows and loses 10 B1a rows; the rule only has `−15` and `−15−10` | add the case; `N_max = 182` depends on it | **T-02's owner** |
| **4** | **`canonical_string` times out** at COIL-DEL (`n=22, m=54`) and from `n ≈ 50` upward; `pruned_canonical_string` is 0.15 ms at `n = 70`. **Suite 2 must use the pruned variant.** `timeout_s` *is* honoured — it raises `CanonicalizationTimeoutError` at exactly 5.00 s | record in [data](../data.md) / T-06's plan | T-06 |
| **5** | **min-DFS separation 0.32 beats IsalGraph's 0.73.** Claim B's framing must concede the axis or T-04a must overturn it on real data | [gspan-mdfsc](gspan-mdfsc.md) §5 | T-04a → T-20 |
| **6** | **The adjacency matrix beats IsalGraph on 7 of 10 profiles.** The submitted comparison was against `B_GED` only, which everything beats. **R3.6a applies to us harder than the reviewer knew** | [adjacency-matrix](adjacency-matrix.md) §4 | T-20 |
| **7** | **The four `n²` members share one Claim A number.** Four identical columns read as a copy-paste error | one row + footnote | T-17 |
| **8** | **The gSpan vendoring plan is superseded. Three repositories tested, three rejected**: `LasseRegin/gSpan` does not run on numpy ≥ 1.24 and its `G2DFS` is not minimal; `betterenvi`'s `_is_min` is private; **`kaviniitm/DFSCode` claims exactly this, builds, and is wrong on 50 % of 6-node graphs and not isomorphism-invariant (46/90)**. **Vendor nothing; cite Yan & Han.** Effort **2–3 d → ~1 d** | [competitors](../competitors.md) §2 risk paragraph | T-04, [schedule](../schedule.md) |
| **9** | **The competitor runtime figure must be language-matched.** Timing a pure-Python min-DFS against the C++ engine on one axis reproduces R1.1's own complaint inside our answer to it | [gspan-mdfsc](gspan-mdfsc.md) §5 | T-06, Fig. 2 |
| **10** | **`grakel`'s `n_iter = k` equals our `h = k−1`.** Verified: `grakel(3)` ≡ `ours(2)` = 5.830952 exactly. **E10's existing WL numbers must be re-checked against whichever convention produced them** | [wl-subtree-kernel](wl-subtree-kernel.md) §1 | T-06 |
| **11** | **`nx.relabel_nodes(copy=True)` preserves insertion order**, so any F3 test built on it is void — order-dependent formats look invariant. Every measurement here rebuilds the copy with a fresh insertion order | method note for T-04a's F3 | T-04a |
| **12** | **bliss / Traces stay cut, and the counter-case has expired.** The `pynauty` from-source build was rehearsed under gcc 12.2.0 and succeeded | [nauty](nauty.md) §8; decision S-g | — |

---

## 5. What to port, and what not to

**Port**: `min_dfs.py` **with `validate_min_dfs.py`** — the brute-force oracle is the value; and
`agm_cam.py` with its 327-graph brute-force check. Both become `src/isalgraph/competitors/` backends
plus `tests/unit/`.

**Port as a gate, not a backend**: `test_kavin.py`. It is the acceptance test any third-party
minimum-DFS implementation must pass before adoption — and its **K2 check needs no oracle**, only
relabellings, so it applies to any candidate canonical backend at all.

**Do not port**: `backends.py`'s subprocess bridge to the conda env (a scratchpad hack),
`sweep.py`/`scale.py`'s synthetic-profile generators (T-04a uses the real 200-graph stratified
sample), and **none of the three gSpan / DFS-code repositories**.

**Add as a fixture**: `K_{3,3}` vs the triangular prism. WL distance is **0.0000**; every canonical
backend separates them. Two lines, and it catches a broken canonical backend instantly.

**Watch for**, in order of how quietly they fail:

1. Counting the adjacency matrix as `len('1010…') * 8` — inflates it 8× and hands us a baseline we
   beat for free ([adjacency-matrix](adjacency-matrix.md) §7).
2. Inverting `pynauty.canon_label` — produces a deterministic wrong labelling that **passes F3**
   ([nauty](nauty.md) §1).
3. Fitting the WL kernel per batch rather than per dataset — makes the distance matrix depend on
   batching order ([wl-subtree-kernel](wl-subtree-kernel.md) §7).
4. Mixing min-DFS character-level and tuple-level Levenshtein — a 2× difference
   ([gspan-mdfsc](gspan-mdfsc.md) §3).
5. Returning AGM's incumbent instead of raising when the budget runs out — puts a non-canonical code
   in a column headed canonical ([agm](agm.md) §7).
6. **Accepting a third-party canonical backend on a single example.** `kaviniitm/DFSCode` agrees
   with the oracle on the running example, on every path and on every cycle, and is wrong on half
   of all 6-node graphs ([gspan-mdfsc](gspan-mdfsc.md) §1.3). **Run K2 — invariance under
   relabelling — before anything else; it needs no oracle and it catches this class outright.**

---

## 6. Still open — what only the real cohort can answer

- **F5 for every admissible cell**, on T-04a's 200-graph stratified sample against exact GED. The
  separation figures here are a prior from synthetic graphs, not the result.
- **Whether min-DFS's advantage survives on IAM Letter**, where `n̄ ≈ 4` and the graphs are
  near-planar and geometric rather than `G(n,m)`.
- **Whether AGM's `n ≈ 14` ceiling moves** under orbit pruning from `pynauty.autgrp`. It will move;
  it will not reach `n = 32`.
- **The realised-bytes column** for every method — measured here only for the running example.
- **`pruned_canonical_string`'s own ceiling** above `n = 70` at high density: 0.15 ms at
  `n = 70, m = 73`, but the dense profiles are unmeasured.
