# gSpan minimum DFS code (M-DFSC)

**Verdict**: **RUN — and it is the real competitor.** [competitors](../competitors.md) §2 calls it
"the single most important comparator" and that is correct. The day-1 risk it flags is **confirmed:
neither public repository exposes the minimum DFS code of a single graph**, so we wrote it (~150
lines, exhaustively validated, §1). It is canonical, complete, cheap, and it has the **best GED
separation in the pool — 0.32 against IsalGraph's 0.73.**

> **Read §5 before scheduling T-06.** On the two axes the paper claims, the measured split is:
> **IsalGraph wins compactness on 9 of 10 cohort profiles; the minimum DFS code wins
> edit-distance tracking by more than 2×.** That is a defensible contribution, but it is not the
> contribution the manuscript currently states.

**Role**: [competitors](../competitors.md) §2 row 6 · serves **R1.2a** (named by R1 **by name**),
**R1.1**, **AE.4a**, **AE.3**
**Evidence**: measured on this workstation, 2026-08-13,
`scratchpad/competitors/{min_dfs.py,validate_min_dfs.py,probe,sweep,scale,ceiling,stability}.py`.
Cross-refs: [agm](agm.md) (the other mining-literature family), [README](README.md) §3.

---

## 1. Reproducibility — the day-1 risk, resolved

[competitors](../competitors.md) §2: *"`LasseRegin/gSpan` is a frequent-subgraph miner; the minimum
DFS code of one graph is an internal sub-component and may not be exposed. **Verify on day 1 of
T-04.**"* **Verified. It is not exposed in either gSpan repository, and the one standalone tool that
claims to expose it is wrong** (§1.3).

**Three candidates tested, three rejected:**

| Repository | Last push | Outcome |
|---|---|---|
| `LasseRegin/gSpan` | 2018-04-10 | does not run on numpy ≥ 1.24; `G2DFS` is not the minimum code; no tie branching (§1.1) |
| `betterenvi/gSpan` | 2020-07-12 | correct `_is_min`, but **private**, needs a miner + graph database + `min_support` (§1.2) |
| `kaviniitm/DFSCode` | 2017-01-11 | builds and claims exactly this, but **wrong on 50 % of 6-node graphs and not isomorphism-invariant** (§1.3) |

### 1.1 `LasseRegin/gSpan` — cloned and run, 2026-08-13

Last push **2018-04-10**. Three independent blockers:

| Finding | Evidence |
|---|---|
| **Does not run on a modern numpy** | `read_data` calls `np.array` on a ragged list → `ValueError: setting an array element with a sequence` on numpy 1.26.4. Needs a monkeypatch to load a single file |
| **`G2DFS(G)` is not the minimum DFS code** | It reads the graph's *insertion order*. Fed the same graph with the edges listed differently: `[(0,1),(1,2),(2,3),(3,0),(3,4),(4,5),(5,3)]` vs `[(4,5),(5,3),(3,0),(0,1),(1,2),(2,3),(3,4)]`. **Not isomorphism-invariant** |
| **`is_canonical(C)` only tests, and greedily** | It returns a bool; on the running example's own code it returns `False`. Internally it builds `C*` by taking `get_minimum_DFS` of the extension set at each step — **with no tie branching**. For unlabelled graphs every label is equal, so **every step is a tie**, and a greedy no-branch construction is not guaranteed to reach the minimum |

There is no `min_dfs_code(G)` and no function that would become one without a rewrite.

### 1.2 `betterenvi/gSpan` — cloned and inspected, 2026-08-13

Last push **2020-07-12** — the better-maintained of the two, and the one to vendor *from* if we
vendor anything. It has the right machinery: `DFSedge`, `DFScode`, `build_rmpath`,
`_get_backward_edge`, `_get_forward_rmpath_edges`, and a correct `_is_min` with `project_is_min`
tie handling. **But `_is_min` is a private method of the `gSpan` class**, reachable only through a
constructed miner with a graph database, a `min_support`, and a `_read_graphs` pass over a file in
gSpan's text format. There is no single-graph entry point.

Extracting it is possible; it means lifting ~120 lines out of a class that carries mining state.
That is not cheaper than writing the construction directly, and it is harder to validate.

### 1.3 `kaviniitm/DFSCode` — the one that claims exactly what we need, and fails

Raised 2026-08-13. Last commit **2017-01-11**; one file, `DfsCode.cpp`, 637 lines, plus a
two-line build script. Its README says, in its own words:

> "This C++ project generates the minimum DFS Code for a given graph."

That is precisely the missing entry point, so it deserved a real test rather than a glance.
**It builds cleanly** (`g++ -O2`, no dependencies, no warnings), reads a simple text format, and on
the running example returns `<0,1,a,e,a><1,2,a,e,a><2,0,a,e,a><2,3,a,e,a><3,4,a,e,a><4,5,a,e,a>
<5,2,a,e,a>` — **identical to ours**. On that one graph it looks right.

**It is not.** Differential test against the validated oracle in §1.4, three checks:

| Check | Result |
|---|---|
| **K1** — every connected isomorphism class, `n ≤ 6` | `n=4`: **1 / 6** wrong · `n=5`: **7 / 21** wrong · `n=6`: **56 / 112** wrong |
| **K2** — isomorphism invariance of the tool alone, no oracle involved | **46 of 90 graphs are not invariant**; one 6-node graph produced **6 distinct codes from 13 relabellings** |
| **K3** — random graphs, `n = 7 … 40`, both densities | **89 / 112** disagree; every profile from `n = 15` up disagrees on 6–8 of 8 |

**K2 is dispositive on its own.** It makes no reference to our implementation: a canonical form that
returns different codes for different labellings of the same graph is not a canonical form. Nothing
downstream — no distance, no F3 row, no comparison — can be built on it.

**Worked counterexample** (the smallest failure, `n = 4`): `G` = `K₄` minus edge `(2,3)`, i.e.
edges `01, 02, 03, 12, 13`.

```
DFSCode  <0,1><1,2><2,0><2,3><3,1>
ours     <0,1><1,2><2,0><2,3><3,0>      <-- smaller, and realisable
```

Both open with the same triangle and the same forward edge, so the codes are decided by the final
backward edge. Under the DFS lexicographic order backward edges sort by **increasing** target
index, so `(3,0) ≺ (3,1)` and the tool's answer is not minimal. It is realisable: map DFS indices
`0,1,2 → g0, g3, g1` (a triangle: `g0g3`, `g3g1`, `g1g0` all present), then `(2,3) → g1g2`, and the
backward edge `(3,0) → g2g0` exists. **The tool returns a valid DFS code that is not the minimum
one.**

**Why it fails is visible in its own README**, which describes an iterative-deepening tie-break:
explore each tied branch to a lookahead depth `D ∈ {1,2,4,8,…}`, take the branch with the unique
smallest prefix, double `D` when undecided, and — if still tied at `D = m` — assume "all the options
which give the minimum are symmetric" and pick any. Two problems. The lookahead is itself the same
greedy procedure, so the prefixes it compares are not sub-minima; and the symmetry assumption is
asserted, not proved. For **unlabelled** graphs, where every vertex and edge label is equal and
therefore *every* step is a tie, both problems fire on essentially every graph — which is exactly
the 50 % failure rate measured at `n = 6`.

> **Verdict: do not use, and do not cite as an implementation.** It is a good-faith 2017 student
> project, correct on symmetric-free labelled inputs and wrong on ours. The finding is worth keeping
> in the folder because "we found a repository that claims to do this and tested it" is a stronger
> answer to R1.2a than "no implementation exists", and because the next person to search GitHub for
> `minimum DFS code` will find it first.
>
> **Reusable lesson**: it agreed with us on the running example and on every path and cycle. A
> single-example check would have adopted it. The exhaustive oracle is what caught it.

### 1.4 What we did instead, and how it was validated

`scratchpad/competitors/min_dfs.py` — ~150 lines implementing the standard construction with
**correct tie branching**: hold the set of embeddings that realise the current minimal prefix, take
the globally minimal rightmost-path extension, keep only the embeddings achieving it.

| Check | Result |
|---|---|
| **V1 — exhaustive brute force**: compare against the lexicographic minimum over **every valid DFS traversal** | **agrees on all 30 connected isomorphism classes with `n ≤ 5`** (1, 2, 6, 21 at `n` = 2…5) |
| **V3 — complete invariant**: distinct codes per `n` | **1 / 2 / 6 / 21 / 112** at `n` = 2…6 — exactly the number of connected graphs on `n` nodes (OEIS A001349). **No collisions** |
| **V2 — isomorphism invariance** | **4,440 relabellings across 6 ≤ `n` ≤ 10, 0 mismatches** |
| Reversibility, `code → graph` | isomorphic in every case |
| F3 sweep, 40 graphs × 25 relabellings | **40 / 40 invariant** |
| 200 relabellings of the running example | **1 distinct code** |

**Effort: well inside [competitors](../competitors.md) §2's 2–3 day budget** — the implementation and
its validation together were hours, not days, because the brute-force oracle is only a few lines
once the DFS lexicographic order is written down. **Cut the estimate to 1 day** for
`src/isalgraph/competitors/` integration plus tests.

Cite: Yan & Han, *gSpan: Graph-Based Substructure Pattern Mining*, **ICDM 2002**, 721–724,
doi:10.1109/ICDM.2002.1184038.

---

## 2. What the representation looks like

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`.

```
min-DFS code (G)              (0,1) (1,2) (2,0) (2,3) (3,4) (4,5) (5,2)     7 tuples = m
min-DFS code (G relabelled)   identical                                     <-- INVARIANT
min-DFS code (H = G−edge 0,3) (0,1) (1,2) (2,0) (2,3) (3,4) (4,5)           6 tuples
```

A DFS code is a sequence of 5-tuples `(i, j, l_i, l_ij, l_j)` with `i, j` DFS discovery indices.
Our corpus is topology-only, so every label is constant and the tuple degenerates to `(i, j)`.
Backward edges precede forward edges; forward extensions prefer the deepest point of the rightmost
path; the minimum over all traversals is the canonical form.

| Property | min-DFS code | IsalGraph pruned |
|---|---|---|
| Reversible | **yes**, up to isomorphism | yes, up to isomorphism |
| Isomorphism-invariant | **yes** (40/40) | **yes** (40/40) |
| Complete invariant | **yes** (112/112 at `n = 6`) | yes, within a directedness class |
| **Length** | **exactly `m` tuples — deterministic** | data-dependent |
| **Alphabet** | index pairs; **grows as `O(n²)`** | **fixed, `\|Σ\| = 9`** |
| Requires connectivity | **yes** | **yes** (`DisconnectedGraphError`) |
| Represents an edgeless graph | **no** — no tuple carries an isolated vertex | n/a |
| Encode cost (our pure-Python impl.) | 1.8–25 ms, `n = 20…50` | 0.01–0.15 ms, `n = 20…70` (C++) |
| Worst case | exponential in the number of tied embeddings | exponential in density |

**The two structural differences that survive scrutiny:**

1. **Length is a function of `m` alone.** `|code| = m`, always. IsalGraph's `L` is a search outcome.
   This makes the min-DFS code's bit accounting exact and its Levenshtein distance bounded by
   `max(m₁, m₂)` — a property no other pool member has.
2. **The alphabet grows with `n`.** A min-DFS code is a string over index pairs, so `|Σ| = O(n²)`;
   IsalGraph's is a string over 9 construction operations regardless of `n`. This is the one
   *conceptual* difference R1.2 asks about that is not a matter of degree.

---

## 3. Which distance does it accept?

**Levenshtein, and the serialisation convention changes the answer.** Hamming is undefined for
**0 %** of pairs at the tuple level (lengths differ whenever `m` differs) — measured 0.0 % defined.

Measured, 120 one-edit pairs (unit GED = 1, edited copy randomly relabelled) vs 120 random same-`n`
pairs, `n ∈ [6,12]`:

| Convention | median length | median Lev, GED = 1 | median Lev, random | **separation** |
|---|---:|---:|---:|---:|
| **character-level** `'0-1 1-2 2-0 …'` | 52 | 6.0 | 19.0 | **0.32** |
| **tuple-level**, one symbol per DFS edge | 13 | 3.0 | 8.0 | **0.38** |
| *(IsalGraph pruned, for reference)* | 24 | 11.0 | 15.0 | *0.73* |
| *(AGM CAM, for reference)* | 36 | 4.5 | 9.0 | *0.50* |

> **Both conventions beat every other representation in the pool, by roughly 2×.** The min-DFS code
> is the most edit-distance-compatible representation we measured — which is exactly the property
> the manuscript claims as IsalGraph's.

### 3.1 On the real cohort, against certified exact GED — the prior holds, with a wider margin

Spearman ρ of tuple-level Levenshtein against T-03's **certified exact GED** (D6 unit costs),
200-graph sample per dataset, certified pairs only:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **min-DFS code** | **0.972** | **0.965** | **0.842** | **0.653** | **0.551** |
| IsalGraph pruned | 0.925 | 0.916 | 0.683 | 0.474 | 0.255 |
| **margin** | **+0.047** | **+0.049** | **+0.159** | **+0.179** | **+0.296** |
| *(size null `\|n₁−n₂\|`)* | *0.899* | *0.909* | *0.926* | *0.713* | *0.799* |

Restricted to **equal-`n`** pairs, where the size channel is constant and the comparison is pure
structure:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **min-DFS code** | **0.996** | **0.980** | **0.806** | **0.540** | **0.442** |
| IsalGraph pruned | 0.981 | 0.961 | 0.628 | 0.397 | 0.250 |
| nauty→graph6 | 0.974 | 0.969 | 0.682 | 0.261 | 0.186 |
| adjacency | 0.565 | 0.429 | 0.424 | 0.300 | 0.243 |

> **The minimum DFS code wins every column of both tables.** The margin widens with graph size —
> +0.047 at `n̄ = 4.07` to +0.296 at `n̄ = 10.56` — which is the direction that matters, because
> AE.1 asks the paper to go *up* in size.
>
> Note also that **min-DFS is the only representation in the pool that clears the size null**
> (+0.073 on Letter LOW, +0.056 on Letter MED) — see [README](README.md) §4.1, finding 1. On
> Letter HIGH, LINUX and AIDS **nothing clears it**, min-DFS included.
>
> **Fix the convention before T-06.** Character-level charges 4 edits for one deleted tuple
> (`' 5-2'`) and is an artefact of ASCII framing; tuple-level charges 1 and is the semantically
> correct unit. **Use tuple-level as primary** — it is the like-for-like comparison against
> IsalGraph, whose symbols are also single operations — **and report character-level in the
> supplementary grid**, since the two disagree and a reader will ask.

Primary distance by [competitors](../competitors.md) §3.4: **Levenshtein, tuple-level** (F1 100 %,
F2 metric, F3 40/40, F4 non-degenerate; F6 is the tiebreak and F5 is never allowed to be).

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes, and it is the only competitor besides sparse6 whose bit cost scales with `m`.**

| Convention | Value |
|---|---|
| Entropy bound | **`m · 2⌈log₂ n⌉`** bits — `m` tuples, two indices each, fixed width |
| Realised bytes | the character serialisation, `8 · len` — **inflated**, report it as such |

> ⚠ **The entropy bound above is an over-estimate and a reviewer can say so.** DFS indices are not
> uniform on `[0, n)`: a forward extension always introduces index `max + 1`, and a backward edge
> targets a vertex on the rightmost path. A tighter bound exists. **Report `m · 2⌈log₂ n⌉`, state
> that it is a fixed-width upper bound, and say why we did not tighten it** — the same fixed-width
> convention is applied to `B_GED`'s `2M⌈log₂ N⌉` endpoint addressing
> ([statistics](../statistics.md) §2), so tightening one and not the other would be the exact
> asymmetry R3.6a objects to. Consistency is the defence; silence is not.

Measured entropy-bound bits, median over Suite-2 profiles:

| Profile | `n` | `m` | min-DFS | sparse6 | adjacency | **IsalGraph pruned** | winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Letter LOW | 4 | 3 | 12 | 24 | **6** | 13 | adjacency |
| Letter HIGH | 5 | 5 | 30 | 36 | **10** | 25 | adjacency |
| LINUX | 9 | 8 | 64 | 66 | **36** | 38 | adjacency |
| AIDS (GraphEdX) | 11 | 11 | 88 | 78 | **55** | 60 | adjacency |
| GREC | 11 | 12 | 96 | 84 | **55** | 70 | adjacency |
| AIDS (IAM) | 14 | 15 | 120 | 102 | 91 | **82** | **IsalGraph** |
| COIL-DEL | 22 | 54 | 540 | 348 | **231** | 418 | adjacency |
| Mutagenicity | 29 | 30 | 300 | 222 | 406 | **181** | **IsalGraph** |
| Protein | 32 | 61 | 610 | **396** | 496 | 533 | sparse6 |

Ceiling sweep to Suite 2's `n_max = 98` (**different sampler** — spanning tree plus uniform extra
edges; rejection sampling on `G(n,m)` does not terminate at `m ≈ n` above `n ≈ 30`):

| `n` | `m` | `m/n` | min-DFS | sparse6 | adjacency | **IsalGraph pruned** | winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 21 | 1.05 | 210 | 156 | 190 | **136** | **IsalGraph** |
| 30 | 31 | 1.03 | 310 | 234 | 435 | **187** | **IsalGraph** |
| 30 | 60 | 2.00 | 600 | **384** | 435 | 561 | sparse6 |
| 50 | 52 | 1.04 | 624 | 426 | 1225 | **352** | **IsalGraph** |
| 50 | 100 | 2.00 | 1200 | **744** | 1225 | 1024 | sparse6 |
| 70 | 73 | 1.04 | 1022 | 714 | 2415 | **539** | **IsalGraph** |
| 70 | 140 | 2.00 | 1960 | **1218** | 2415 | — | sparse6 |
| **98** | **102–103** | **1.04** | 1428 | 978 | 4753 | **888** | **IsalGraph** |
| 98 | 196 | 2.00 | 2744 | **1698** | 4753 | — | sparse6 |

**IsalGraph is shorter than the min-DFS code on every profile except Letter LOW (13 vs 12 bits, a
tie at four nodes), and the margin widens with `n`: 82 vs 120 at `n = 14`, 888 vs 1428 at
`n = 98`.** This is the clearest quantitative win in the folder and it should lead the Claim A
discussion. The `m/n ≈ 2` rows are where sparse6 takes over — from **both** of us.

---

## 5. Scope alignment and IsalGraph's advantage — the honest audit

**Fully aligned. R1 named gSpan by name**, it is the M-DFSC family representative in Jiang, Coenen
& Zito's two-family taxonomy, it solves the same problem (canonical representation of an unlabelled
graph as a string), and it admits the same edit distance. There is no competitor in the pool closer
to IsalGraph's problem setting.

| Axis (R1.2b) | Winner | Measurement |
|---|---|---|
| Uniqueness | **tie** | both 40/40 invariant, both complete invariants |
| Expressiveness | **tie** | both require connectivity; neither represents isolated vertices |
| **Computational efficiency** | **unresolved — see below** | IsalGraph (C++) 0.95 ms at `n = 98`; min-DFS (our pure Python) **124 ms** at `n = 98`. **Not a fair comparison** |
| Scalability | **unresolved**, same reason | |
| Downstream learning | **not evaluated** | R1.2b's fifth axis |
| **Message length** | **IsalGraph** | shorter on **9 of 10** cohort profiles (§4) |
| **Edit-distance compatibility** | **min-DFS code**, by >2× | separation 0.38 vs 0.73 (§3) |
| Alphabet | **IsalGraph** | fixed `\|Σ\| = 9` vs `O(n²)` |

> ⚠ **The runtime row is exactly the error R1.1 reported, reproduced inside our own comparison.**
> R1.1's complaint is that Fig. 2 compares things whose "objectives and underlying assumptions
> differ significantly". Putting a hand-written pure-Python min-DFS encoder beside a tuned C++
> engine and printing both on one axis would be the same category error with a different pair of
> objects. **Either both arms are Python or both are compiled**, and the figure says which.
> Cheapest honest option: time IsalGraph's **Python reference** (`isalgraph.core.canonical`)
> against our Python min-DFS, and report the C++ speedup separately as an engineering result.
>
> **This is a plan-level instruction for T-06 and Fig. 2, not a note.**

### Does IsalGraph provide a foundational advantage over the min-DFS code?

**Partly, and the honest answer is narrower than the manuscript's.**

- **Yes on message length** — on the **real** cohort IsalGraph is strictly shorter than the min-DFS
  code on **71.5 %** of Letter LOW graphs, **60.0 %** of Letter HIGH, **98.9 %** of LINUX,
  **99.6 %** of AIDS and **96.2 %** of GREC. Claim A survives against the strongest competitor, and
  the margin grows with `n`. This is IsalGraph's one clean win over gSpan.
- **Yes on the alphabet** — a fixed 9-symbol operational alphabet versus an index-pair alphabet that
  grows as `O(n²)`. This is the conceptual difference R1.2 asks for, and it is a property, not a
  measurement, so it costs nothing to state. **Do not extend it into a claim about sequence models**
  — the plan declines that experiment ([demands](../demands.md)), and the argument does not need it.
- **No on edit-distance compatibility** — the axis the manuscript leads with. **On the real cohort,
  against certified exact GED, the min-DFS code wins all five Suite-1 datasets, by +0.047 to
  +0.296, in both the all-pairs and the equal-`n` view.** This is no longer a synthetic prior.
- **Unresolved on efficiency and scalability** until the runtime comparison is made fair.

**The claim that survives**: *IsalGraph encodes a graph in fewer bits than the minimum DFS code over
a fixed 9-symbol alphabet, at comparable canonicity; the minimum DFS code's Levenshtein distance
tracks unit edits more tightly.* Two representations, two different strengths, both measured. A
reviewer who has read gSpan will find that far more credible than a claim of dominance — and R1,
who named gSpan unprompted, has read gSpan.

~~**Caveat on generality**: the separation figures come from `G(n, m)` random graphs, not IAM. The
ordering may differ there.~~ **Resolved 2026-08-13 — it does not.** §3.1 runs the comparison on the
real cohort against certified exact GED and the ordering is the same, with a larger margin. What
remains open is Suite 2, which has no GED reference until T-05.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Not from any of the three candidates.** `LasseRegin` is broken on modern numpy and its `G2DFS` is not minimal; `betterenvi`'s `_is_min` is private and needs a miner; **`kaviniitm/DFSCode` builds and claims exactly this but is wrong on 50 % of 6-node graphs and is not isomorphism-invariant.** **We wrote it**, validated against exhaustive brute force + 4,440 relabellings |
| 2 | Representation | `m` DFS tuples, **deterministic length**, complete invariant (112/112 at `n = 6`), alphabet grows as `O(n²)` |
| 3 | Distance | **Levenshtein, tuple-level.** **ρ vs certified exact GED = 0.55–0.97, best in the pool on all five Suite-1 datasets** |
| 4 | Claim A? | **Yes**, `m · 2⌈log₂ n⌉` bits. **IsalGraph is shorter on 60–100 % of real graphs** |
| 5 | Scope | **In, and it is the closest competitor there is.** Named by R1; M-DFSC family representative |
| — | IsalGraph advantage | **Bits yes, alphabet yes, GED tracking no (loses by +0.047 to +0.296 on real data), runtime unresolved.** State all four |

---

## 7. For the integration agent

- **Port `scratchpad/competitors/min_dfs.py` and `validate_min_dfs.py` together.** The brute-force
  oracle is the whole value of the port; without it the backend is 150 lines of unverified graph
  theory. Keep V1/V2/V3 as `tests/unit/test_min_dfs.py`, marked slow.
- **Do not vendor `LasseRegin/gSpan`.** [competitors](../competitors.md) §2's plan to vendor it is
  superseded: it does not run and its `G2DFS` is not the minimum code. Cite Yan & Han; vendor
  nothing.
- **Do not adopt `kaviniitm/DFSCode` either**, and keep `test_kavin.py` — if anyone proposes a
  third-party minimum-DFS implementation later, that script is the acceptance test it has to pass.
  Any candidate must clear **K2 (isomorphism invariance) before anything else**; K2 needs no oracle
  and it is where this one died.
- **Fix the serialisation convention in the backend, not in the analysis.** `encode()` returns the
  tuple-level symbol string; expose the character rendering as a separate method for figures.
  Mixing them produced a 2× difference in measured Levenshtein above.
- ⚠ **The budget must be on MEMORY, not just time.** The construction holds every embedding that
  realises the current minimal prefix, and that set is worst-case exponential in the number of ties.
  The first Suite-2 run was **OOM-killed (exit 137)** partway through Mutagenicity (`n_max = 97`) —
  not slow, *killed*. `min_dfs_code` now takes `max_projections` and raises
  `MinDfsBudgetExceeded`; the validation suite was re-run after the change and still passes
  exhaustively. At a 50,000-embedding cap the cost is **24/400 Mutagenicity graphs** and zero
  elsewhere in the cohort.
- Timing, real cohort: 0.05 ms (Letter) · 0.76 ms (AIDS) · 1.0 ms (GREC) · 3.2 ms (AIDS-IAM) ·
  **60 ms (Protein)** · **68 ms (COIL-DEL)** · **1,182 ms (Mutagenicity)** per graph, pure Python.
  It is by far the slowest backend at Suite-2 scale, and the Mutagenicity figure is dominated by the
  24 graphs that run to the cap before failing. **Profile before porting to C++** — the cost is tie
  branching, which a port does not remove.
- Disconnected graphs raise `ValueError` by construction. Suite 1 and Suite 2 are both
  `require_connected = True` ([data](../data.md) §1), so **this never fires on the cohort** — but it
  must still be a documented row in the AE.3 table, because AGM, graph6 and sparse6 handle it and
  IsalGraph does not either.
