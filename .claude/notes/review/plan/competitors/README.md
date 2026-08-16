# Competitors — one file per representation, backed by measurement

**Owner**: T-04 · **Gates**: T-04a, T-06, T-17
**Parent**: [competitors](../competitors.md) — the *decision* file. This folder is the *evidence*
behind it. Where the two disagree, §5 says which wins and who owns the fix.

**Measured on the local workstation, 2026-08-13, on the real cohort.** Suite 1 graphs come from
`data/exported/<ds>.npz` and their **certified exact GED** from T-03's
`extended_merged_exact_ged/computed/<ds>.npz` under the D6 unit cost model `[1,1,0,1,1,0]`;
`graph_ids` alignment between the two is asserted, not assumed. Suite 2 comes from the repo's own
T-01 `iam_gxl_loader` against `IAM_Database/extracted`, under the locked filter
(`min_nodes = 2`, `require_connected`, cxl enumeration) — it reproduces T-01's retained counts
(GREC 650, AIDS-IAM 1,811).

Scripts and raw output: **[`scratch/`](scratch/)**. Environment: `networkx` 3.6.1, `pynauty` 2.8.8.1,
`grakel` 0.1.8, `rapidfuzz` 3.14.5, `numpy` 1.26.4, gcc 12.2.0, `isalgraph.engine() == "cpp"`.

> **ρ here is descriptive and does not select anything.** [competitors](../competitors.md) §3.4's
> rule is **F5-blind by construction** — primary distances are chosen on F1–F4 with F6 as the
> tiebreak — and nothing in this folder changes that. T-04a re-runs the grid under its own
> pre-declared protocol. What these numbers are for is deciding **what the paper can claim**.

---

## 1. The files

| File | Competitor | Verdict |
|---|---|---|
| [graph6](graph6.md) | McKay's packed adjacency serialisation | **RUN — negative control** |
| [sparse6](sparse6.md) | McKay's edge-list serialisation | **RUN** — the compactness rival |
| [nauty](nauty.md) | canonical labelling → graph6 (+ the bliss/Traces cut) | **RUN** — the fair canonical serialisation |
| [adjacency-matrix](adjacency-matrix.md) | raw upper triangle | **RUN — reference point**, and it is not the pushover it looks |
| [agm](agm.md) | AGM canonical adjacency-matrix code (CAM) | **RUN, Suite 1 only** — up to 98 % failure on Suite 2 |
| [gspan-mdfsc](gspan-mdfsc.md) | gSpan minimum DFS code | **RUN** — and it beats IsalGraph on every dataset |
| [wl-subtree-kernel](wl-subtree-kernel.md) | Weisfeiler–Lehman subtree kernel | **RUN, Claim B only** |

Each answers the same five questions in the same order: reproducibility, representation, distance,
Claim A fit, scope alignment — then a summary table and integration notes.

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

- **All four have the same Claim A bit count**, `n(n−1)/2`, up to header and padding. **Print one
  `n²` row with a footnote, not four identical columns.**
- The pool therefore isolates **canonicity as a variable at fixed format**. Measured on Letter LOW,
  graph6 → nauty→graph6 moves F3 invariance from **4/50 to 50/50** and equal-`n` ρ from **0.539 to
  0.974** with the format held constant. That subtraction *is* R1.2's uniqueness answer.

**Family II — the mining-literature canonical forms.** Jiang, Coenen & Zito
(*Knowledge Engineering Review* 28(1):75–105, 2013) classify every frequent-subgraph miner's
representation into **CAM** (AGM, FSG, FFSM) and **M-DFSC** (gSpan). R1.2 named one of each.

**Outliers.** [sparse6](sparse6.md) is the only non-canonical format whose length scales with `m`.
[WL](wl-subtree-kernel.md) is not a serialisation at all and enters Claim B only.

---

## 3. The master table

F3 = isomorphism invariance, 50 real graphs × 20 genuine relabellings, per dataset.

> **What F3 measures for a non-canonical format, established 2026-08-15 by T-04 and proved, not
> sampled.** Over *every* connected graph on `n = 2…6`, exactly **5** are invariant under **every**
> relabelling, and **all 5 are complete graphs** — the strict upper triangle is relabelling-invariant
> iff the adjacency matrix is constant off-diagonal. So `0–6 / 50` for `adjacency`, `graph6` and
> `sparse6` is **the count of complete graphs in the draw**, not a sampling artefact of the 20-draw
> harness: exhaustive enumeration over all `n!` relabellings returns the same counts. That is why the
> Letter sets (many `K₂`/`K₃`) score 4–9/50 while LINUX, AIDS and GREC score **0/50**. Put this in
> the F3 caption; it turns an incidental-looking number into a statement about the cohort.

ρ = Spearman of Levenshtein against **certified exact GED**, 200-graph sample per dataset.
**The ρ column below inherits §4.1's three-draw provenance — see the block there.**

> ## ⚠ The "Primary distance" column is SUPERSEDED 2026-08-16 by T-04a, which ran the grid
>
> It was set by inspection during the scout. **Measured, `levenshtein` is the primary distance for
> every admissible representation and `padded_hamming` is the primary distance for none.** Where both
> passed F1–F4 the F6 tie-break went to `levenshtein` by **68×** on `nauty→graph6` (0.0010 vs 0.0704
> ms/pair) and **8.6×** on AGM CAM. The rows below reading "padded Hamming" are wrong.
>
> *"none admissible"* for `graph6`, `sparse6` and `adjacency` **is confirmed** — F3 = 1/50 on the
> frozen 200-graph draw. That gives **`k = 3`** for [preregistration](../preregistration.md) §5.
>
> Authority: `experiments/metric-admissibility/results/grid_200.json` on the external drive, and
> [T-04a design](../../tasks/T-04a-design.md).

| Representation | Reproducible? | F3 (real) | Complete invariant | Primary distance | ρ range, Suite 1 | Claim A bits | Ceiling |
|---|---|---|---|---|---|---|---|
| [graph6](graph6.md) | **trivial** (`networkx`) | **0–6 / 50** | no | *none admissible* | 0.46–0.69 | `6(1+⌈n(n−1)/12⌉)` | none |
| [sparse6](sparse6.md) | **trivial** (`networkx`) | **0–6 / 50** | no | *none admissible* | 0.52–0.75 | `6·len`, scales with `m` | none |
| [adjacency](adjacency-matrix.md) | **trivial** (none) | **0–6 / 50** | no | *none admissible* | **0.75–0.87** | `n(n−1)/2` | none |
| [nauty→graph6](nauty.md) | `pip install pynauty`, from-source build verified | **50/50** | **yes** | padded Hamming | 0.46–0.68 | = graph6 | none observed |
| [AGM CAM](agm.md) | **no package — we wrote it**, validated vs brute force on 327 graphs | **50/50** | **yes** | padded Hamming | **0.80–0.92** | = adjacency | **24 % fail at GREC** |
| [min-DFS code](gspan-mdfsc.md) | **3 repos tested, 3 rejected — we wrote it** | **50/50** | **yes** | Levenshtein (tuple) | **0.55–0.97** | `m·2⌈log₂ n⌉` | **24/400 Mutagenicity**; needs a *memory* cap |
| [WL subtree](wl-subtree-kernel.md) | `grakel` **already installed** | **50/50** | **no** | kernel (**pseudometric**) | 0.46–0.90 | **none** | none |
| *IsalGraph pruned* | — | 50/50 | yes | Levenshtein | *0.26–0.93* | `L log₂ 9` | `canonical` unusable on Suite 2; **`pruned` 24/400 Mutagenicity, 4/400 Protein** |
| ~~bliss / Traces~~ | — | — | — | — | — | — | **CUT**, [nauty](nauty.md) §8 |

---

## 4. The three orderings that decide the paper

### 4.1 ρ against certified exact GED — per dataset, real

> ## ⚠ SUPERSEDED 2026-08-15 by T-04 — this table is a composite of three draws
>
> **It does not match this folder's own raw output.** The rows below come from three different
> scripts, each with its own `Random(42)` stream and therefore its own 200-graph sample: most rows
> from `scratch/real_size_null.py`, the **AGM** row from `scratch/real_suite1.py`, the **WL** row
> from `scratch/real_wl.py`. Against `real_suite1.out` the printed values differ by up to **0.074**
> (AIDS IsalGraph 0.255 printed, **0.3288** logged; AIDS min-DFS 0.551 printed, **0.6131** logged).
> LINUX is the control: `N = 89 < 200`, so all three scripts draw the same set and all three agree.
>
> This is finding 14 — ρ moving up to 0.07 between draws — appearing *inside* the table rather than
> beside it.
>
> **The shipped code is not the problem.** `python -m isalgraph.competitors.reproduce --mode
> artefacts` replays each script's stream and reproduces its raw artefact on **all five datasets and
> all forty cells with delta exactly `0.00e+00`** — bit-for-bit, not to a tolerance.
>
> **Use instead**: `.claude/notes/2026-08-14-t04-competitors/corrected_rho_table.json`, from
> `reproduce --mode table` — one script, one seed-42 draw per dataset, one convention (column-wise
> adjacency, shared-vocabulary WL at `h = 2`), eleven rows including the null and both views.
> **T-06, T-17 and T-20 quote that file, not this table.**
>
> **What changes materially** (see [T-04-article-notes](../../tasks/T-04-article-notes.md) §1):
> IsalGraph clears the size null on **one** of five datasets, not two — Letter MED's `+0.007`
> becomes `−0.044`, a margin an order of magnitude below the between-draw variability finding 14
> records. **What survives**: min-DFS beats IsalGraph on all five; AGM beats it wherever computable;
> the null dominates; the equal-`n` canonical/non-canonical gap on Letter LOW is if anything wider.

200-graph sample, seed 42, certified-exact pairs only. Levenshtein throughout; WL uses its kernel
distance. **Bold = best in column.**

| Representation | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **NULL: `\|n₁−n₂\|` alone** | *0.899* | *0.909* | *0.926* | *0.713* | *0.799* |
| adjacency | 0.873 | 0.850 | 0.839 | **0.754** | 0.787 |
| graph6 | 0.691 | 0.681 | 0.670 | 0.507 | 0.456 |
| sparse6 | 0.748 | 0.703 | 0.654 | 0.559 | 0.515 |
| nauty→graph6 | 0.677 | 0.663 | 0.639 | 0.538 | 0.460 |
| AGM CAM | 0.911 | 0.920 | **0.892** | 0.798 | *(3/769 fail)* |
| **min-DFS code** | **0.972** | **0.965** | 0.842 | 0.653 | 0.551 |
| WL subtree (h = 2) | 0.895 | 0.869 | 0.580 | 0.573 | 0.459 |
| **IsalGraph pruned** | 0.925 | 0.916 | 0.683 | 0.474 | 0.255 |

> ### ⚠ Two findings here, and both are existential for Claim B as currently written.
>
> **(a) The trivial predictor beats IsalGraph on four of five datasets.**
> `ρ(|n₁ − n₂|, GED)` — count the nodes, subtract, no representation at all — scores **0.71 to
> 0.93**. IsalGraph clears it on Letter LOW (+0.026) and Letter MED (+0.007) and falls **below** it
> on Letter HIGH (−0.243), LINUX (−0.239) and AIDS (−0.544).
>
> The manuscript's headline "ρ ≈ 0.93 on sparse IAM" reproduces (0.925 on Letter LOW) — but it is
> 0.026 above a baseline that needs no method. **Any ρ this paper prints must appear beside the
> size null**, or the first reviewer to compute it has an unanswerable objection. R3.6b already
> pushed on "strongly correlates"; this is the sharper version of the same push.
>
> **(b) gSpan's minimum DFS code beats IsalGraph on all five datasets** — by +0.047, +0.049,
> +0.159, +0.179 and +0.296. The synthetic prior in the first pass predicted this; the real cohort
> confirms it with a wider margin. AGM CAM also beats IsalGraph on three of the four where it is
> computable.

### 4.2 The equal-`n` restriction — where canonicity actually shows up

> ## ⚠ SUPERSEDED 2026-08-15 by T-04 — same three-draw provenance as §4.1
>
> Replaced by the `equal_n` view of `corrected_rho_table.json`. **The argument survives and gets
> stronger**: on Letter LOW the canonical representations score 0.97–1.00 against the non-canonical
> 0.54–0.61, a gap of **0.42–0.46**, and that subtraction is still the answer to R1.2's uniqueness
> axis. **What does not survive is "min-DFS still wins every column"**: on the single-draw table
> `isalgraph_canonical` takes Letter LOW (**0.9987** vs min-DFS 0.9956) and `wl_subtree` takes AIDS
> (**0.4332** vs 0.3993). IsalGraph leading a column is a result the paper may use, and it appears
> only in this view.


Restricting to pairs with `n₁ = n₂` (22–26 % of pairs) removes the size channel entirely, so the
null is constant and every column is pure structure. **This is the comparison the paper should
lead with.**

| Representation | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| adjacency | 0.565 | 0.429 | 0.424 | 0.300 | 0.243 |
| graph6 | 0.539 | 0.430 | 0.447 | 0.286 | 0.171 |
| sparse6 | 0.559 | 0.425 | 0.448 | 0.255 | 0.155 |
| nauty→graph6 | **0.974** | 0.969 | 0.682 | 0.261 | 0.186 |
| **min-DFS code** | **0.996** | **0.980** | **0.806** | **0.540** | **0.442** |
| **IsalGraph pruned** | 0.981 | 0.961 | 0.628 | 0.397 | 0.250 |

> **The canonical/non-canonical split is stark and it vindicates the pool design.** On Letter LOW
> the canonical representations score 0.97–1.00 and the non-canonical ones 0.54–0.57 — a gap of
> **0.42** that the all-pairs view hides completely, because the size channel floats everyone.
> **This is the number that answers R1.2's uniqueness axis**, and it is a far better argument than
> anything in the current draft.
>
> It does not rescue the ordering: **min-DFS still wins every column**, and IsalGraph trails it
> everywhere.

### 4.3 Claim A — message length, real per-graph bit counts

Median entropy-bound bits, all retained graphs per dataset (Suite 2: 400-graph sample).

> ## ⚠ TWO CORRECTIONS 2026-08-15 by T-04
>
> **1. The `sparse6` column is 6 bits high on every row.** It was computed as `6·len(code)`
> *including* the `':'` prefix; the design note then froze `6·len(wire) − 6`, deliberately, because
> the prefix is framing and not payload. Both are defensible and they differ by exactly one
> character. **The frozen convention wins**, so every sparse6 entry drops by 6: Letter LOW
> 24.0 → **18.0**, Protein 390.0 → **384.0**, Mutagenicity 168.0 → **162.0**. Track A's tests assert
> both conventions so the delta is provably the prefix and nothing else. [sparse6](sparse6.md) §4 is
> self-contradictory on this and follows the frozen rule.
>
> **2. The five Suite-2 rows are not exactly reproducible, and the five Suite-1 rows are.** Suite 1
> is the full retained cohort, so it is deterministic — track A reproduced `adjacency` and `graph6`
> **10/10**. Suite 2 is a *400-graph draw* taken by `scratch/real_suite2.py` from the raw IAM GXL
> tree, **which is no longer on this workstation**; the cohort was recovered as exported `.npz` from
> Picasso and enumerates in a different order. Coarse statistics survive the change of draw, finer
> ones do not: min-DFS's `m·2⌈log₂ n⌉` gives Protein **620.0 against 615.0**, and AGM's AIDS-IAM
> ceiling **80.25 % against 82 %**. Both are sample differences, not algorithm differences — GREC
> reproduces exactly at **76.00 %** with the same code and call. **Quote the Suite-2 rows with their
> draw, or requote them from a run whose sample is recorded.**


| Dataset | `n̄` | `m̄` | adjacency / AGM | graph6 | sparse6 | min-DFS | **IsalGraph pruned** | GED constr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 4.07 | 3.07 | **6.0** | 12.0 | 24.0 | 12.0 | 12.7 | — |
| Letter MED | 4.11 | 3.17 | **6.0** | 12.0 | 24.0 | 12.0 | 12.7 | — |
| Letter HIGH | 4.58 | 4.56 | **10.0** | 18.0 | 36.0 | 24.0 | 25.4 | — |
| LINUX | 8.71 | 8.35 | **36.0** | 42.0 | 60.0 | 64.0 | 41.2 | — |
| AIDS | 10.56 | 10.70 | **55.0** | 66.0 | 72.0 | 88.0 | 57.1 | — |
| GREC | 11.54 | 12.59 | 55.0 | 66.0 | 78.0 | 96.0 | 72.9 | 118.0 |
| AIDS-IAM | 13.63 | 14.05 | 55.0 | 66.0 | 72.0 | 88.0 | **60.2** | 109.0 |
| AIDS-IAM, **mean** | | | 135.9 | 144.4 | 93.8 | 128.2 | **85.3** | 154.9 |
| COIL-DEL | 21.30 | 53.48 | **153.0** | 162.0 | 282.0 | 450.0 | 332.8 | 512.0 |
| **Mutagenicity** | 27.91 | 28.87 | 300.0 | 306.0 | 168.0 | 250.0 | **147.4** | 310.0 |
| Protein | 31.88 | 61.81 | 465.0 | 474.0 | **390.0** | 615.0 | 467.6 | 705.5 |

> **The AIDS-IAM row is where the `m`-scaling story finally appears in real data, and it appears in
> the mean rather than the median.** On the *typical* graph (median) the adjacency matrix still wins,
> 55.0 to 60.2. On the *mean*, which the `n_max = 85` tail dominates, IsalGraph wins outright:
> **85.3 bits against adjacency's 135.9 and graph6's 144.4.** Same dataset, opposite ordering,
> depending on which statistic is printed.
>
> **Print both, and say which is which.** [statistics](../statistics.md) §3 already forbids a mean
> bit count without dispersion; this is the concrete reason. A median-only Claim A table understates
> IsalGraph on exactly the large sparse graphs AE.1 asked us to reach; a mean-only table overstates
> it on the typical case.

**Percentage of graphs on which IsalGraph pruned is strictly shorter:**

| Dataset | vs adjacency | vs graph6 | vs sparse6 | vs min-DFS | vs GED constr. |
|---|---:|---:|---:|---:|---:|
| Letter LOW | **0.0 %** | 57.4 % | 100 % | 71.5 % | — |
| Letter HIGH | **0.0 %** | 19.9 % | 99.0 % | 60.0 % | — |
| LINUX | 15.7 % | 59.6 % | 100 % | 98.9 % | — |
| AIDS | 29.9 % | 63.2 % | 89.9 % | 99.6 % | — |
| GREC | 23.0 % | 32.2 % | 89.5 % | 96.2 % | **100 %** |
| AIDS-IAM | 35.2 % | 65.2 % | 81.2 % | 99.5 % | **100 %** |
| COIL-DEL | 5.2 % | 5.2 % | **5.8 %** | 94.0 % | 99.5 % |
| **Mutagenicity** | **96.8 %** | **100 %** | **69.9 %** | **100 %** | **100 %** |
| Protein | 45.7 % | 48.5 % | **2.8 %** | 98.5 % | 99.7 % |

> ### Claim A resolved: it is governed by `m/n`, and the crossover is inside Suite 2
>
> | Dataset | `m/n` | IsalGraph vs the `n²` formats | IsalGraph vs sparse6 |
> |---|---:|---|---|
> | Letter LOW–HIGH | 0.75–1.00 | **loses** (0.0 %) | wins (99–100 %) |
> | LINUX, AIDS, GREC, AIDS-IAM | 0.89–1.09 | loses (16–35 %) | wins (81–100 %) |
> | **Mutagenicity** | **1.03** | **wins (96.8 %)** | **wins (69.9 %)** |
> | Protein | 1.94 | ties (45.7 %) | **loses (2.8 %)** |
> | COIL-DEL | 2.51 | **loses (5.2 %)** | **loses (5.8 %)** |
>
> **Mutagenicity is the dataset where IsalGraph wins outright** — 147.4 bits against adjacency's
> 300.0, graph6's 306.0, sparse6's 168.0, min-DFS's 250.0 and `B_GED`'s 310.0. It is large
> (`n̄ = 27.9`, `n_max = 97`) *and* sparse (`m/n = 1.03`), which is exactly the regime the
> `m`-scaling argument predicts and exactly the regime AE.1 asked the paper to reach.
>
> **Size alone is not enough.** Protein (`n̄ = 31.9`) is larger than Mutagenicity and IsalGraph only
> ties there; COIL-DEL (`n̄ = 21.3`) is smaller and IsalGraph loses badly. **`m/n`, not `n`, is the
> variable.** The paper should say so and give this table.

> **IsalGraph is never the most compact representation on Suite 1.** The adjacency matrix — and AGM,
> which has the same bit count — wins every dataset, and on the three Letter sets IsalGraph is
> shorter on **0.0 %** of graphs. It beats sparse6 (89–100 %) and min-DFS (60–100 %) comfortably,
> and beats the author-defined `B_GED` construction on **100 %** of GREC.
>
> **R3.6a's "narrow the claim accordingly" applies to us harder than the reviewer knew.** The
> claim that survives: *IsalGraph is shorter than every other **string** serialisation and than the
> explicit-construction reference model; the raw adjacency matrix is shorter at these sizes, and
> the crossover is at `n ≈ 14` and low density.*

---

## 5. What this folder changes in the plan

| # | Finding | Edit | Owner |
|---|---|---|---|
| **1** | **The size null is unowned and it dominates.** `ρ(\|n₁−n₂\|, GED)` = 0.71–0.93; IsalGraph beats it on 2 of 5 datasets by ≤ 0.03. **Every printed ρ needs the null beside it, and the equal-`n` restriction (§4.2) should be the primary comparison** | new row in [statistics](../statistics.md) §4; a null column in Tab. 3 | **T-02's owner**, T-06, T-20 |
| **2** | **min-DFS beats IsalGraph on ρ on all five Suite-1 datasets**, all-pairs and equal-`n`. AGM beats it on 3 of 4 | Claim B's framing must concede the axis | T-17, T-20 |
| **3** | **IsalGraph is shorter on 0.0 % of Letter graphs vs the adjacency matrix**, and never wins Claim A on Suite 1 | [adjacency-matrix](adjacency-matrix.md) §4 | T-20 |
| **4** | **[competitors](../competitors.md) §4 outcome 3 is inverted — and Claim A resolves on `m/n`.** §4 predicts sparse6 beating IsalGraph *on sparse graphs*; measured, IsalGraph beats sparse6 on **69.9 %** of the sparsest large dataset (Mutagenicity, `m/n = 1.03`) and loses on the dense ones (Protein **2.8 %**, COIL-DEL **5.8 %**). **Mutagenicity is where IsalGraph wins everything** — 147.4 bits vs adjacency 300.0, sparse6 168.0, min-DFS 250.0. Restate the pre-commitment in terms of `m/n`, not size | rewrite the pre-commitment | T-04 → T-20 |
| **5** | **AGM collapses across Suite 2**: 3/769 fail on Suite-1 AIDS (`n ≤ 12`), then **24 % GREC · 18 % AIDS-IAM · 46 % COIL-DEL · 90 % Protein · 98 % Mutagenicity**, at 173 → 2,743 ms/graph. The ceiling sits exactly at Suite 1's edge and is driven by the **tail**, not the mean | [agm](agm.md) §2.2b | T-04, T-17 |
| **6** | **[preregistration](../preregistration.md) §5's reduction rule has no case for a representation computable on one suite and not the other.** AGM keeps 5 B1e rows, loses 10 B1a | add the case; `N_max = 182` depends on it | **T-02's owner** |
| **7** | **`canonical_string` is fine on Suite 1 (0 failures) and unusable on Suite 2** — at a 2 s budget it times out on **207/400 COIL-DEL**, **118/400 Mutagenicity**, **300/400 Protein**. **Suite 2 must use `pruned`, and the two are not interchangeable** (different strings, different bit counts) | T-06's plan | T-06 |
| **7b** | ⚠ **`pruned_canonical_string` has a ceiling too, and this correction matters**: 0 failures through AIDS-IAM, then **24/400 on Mutagenicity** (149 ms/graph) and **4/400 on Protein** (66 ms/graph) at a 2 s budget. **An earlier note in this folder said `pruned` was fine to `n = 98`; on real graphs it is not.** T-06 needs a per-graph budget and a recorded-failure path, not an assumption of success | [gspan-mdfsc](gspan-mdfsc.md) §7, T-06 | **T-06** |
| **7c** | **The min-DFS backend needs a MEMORY budget, not just a time budget.** The first Suite-2 run was **OOM-killed** (exit 137) on Mutagenicity: the construction holds every embedding realising the current minimal prefix, and at `n = 92` that set grows without bound. A `max_projections` cap now raises `MinDfsBudgetExceeded`; at 50,000 it costs **24/400 Mutagenicity** failures and 0 elsewhere. Validation re-run after the change: still exhaustively correct | [gspan-mdfsc](gspan-mdfsc.md) §7 | T-04 |
| **8** | **WL's incompleteness fires on the real cohort**: ~1 LINUX pair and ~6 AIDS pairs get kernel distance 0 at GED > 0. On Letter its zero-set matches the isomorphic set exactly | [wl-subtree-kernel](wl-subtree-kernel.md) §2 | T-17 |
| **9** | **The four `n²` members share one Claim A number.** Four identical columns read as a copy-paste error | one row + footnote | T-17 |
| **10** | **The gSpan vendoring plan is superseded. Three repositories tested, three rejected**: `LasseRegin/gSpan` (broken on numpy ≥ 1.24, `G2DFS` not minimal), `betterenvi` (`_is_min` private), **`kaviniitm/DFSCode` (builds, claims exactly this, not isomorphism-invariant — 46/90)**. Vendor nothing. Effort **2–3 d → ~1 d** | [competitors](../competitors.md) §2 | T-04, [schedule](../schedule.md) |
| **11** | **The competitor runtime figure must be language-matched.** Timing a pure-Python min-DFS against the C++ engine reproduces R1.1's own complaint inside our answer to it | [gspan-mdfsc](gspan-mdfsc.md) §5 | T-06, Fig. 2 |
| **12** | ~~**`grakel`'s `n_iter = k` equals our `h = k−1`** (verified: `grakel(3)` ≡ `ours(2)` = 5.830952)~~ **CORRECTED 2026-08-15 by T-04: there is no off-by-one.** `grakel(n_iter=k) ≡ ours(h=k)`, from the source and confirmed by arithmetic; `grakel(n_iter=2) = 5.830952` and `grakel(n_iter=3) = 7.211103`. The off-by-one was in **our own** `scratch/backends.py::wl_features`, which compresses colours per graph per round so rounds ≥ 2 are cross-graph incomparable. **Frozen `h = 2` means `n_iter = 2`**, and `wl_kernel_computer.py`'s `n_iter = 5` is `h = 5`, not `h = 4`. §4.1's WL row moves (Letter LOW 0.895 → 0.7792). **E10's existing WL numbers must still be re-checked** | [wl-subtree-kernel](wl-subtree-kernel.md) §1 | T-06 |
| **13** | ~~**`nx.relabel_nodes(copy=True)` preserves insertion order**, so any F3 test built on it is void~~ **REFUTED 2026-08-16 by T-04a for the shipped backends.** The finding was true of the scout's code and is **false of `src/isalgraph/competitors/`**: the `n²` family's `normalised()` reads `sorted(nodes)`, so those backends are insertion-order invariant and shuffling insertion or edge order moves the encoding **0/50** times. Both relabellers land on the automorphism rate instead (38 and 26 per 5,000 against 27.8 expected). **`fixtures.shuffled_copy` is still the right relabeller** — it is a genuine relabelling rather than a reordering — but the stated *reason* is wrong, and a test asserting the old claim fails | correct the reason; keep the relabeller | T-04a ✅ |
| **16** | **The separation ratio ψ is measured, and it reaches 1.** `ψ = E[d(G,π(G))] / E[d(G,H)]` is **0.0000 on all eleven draws** for the seven canonical representations and **0.07–1.15** for the three excluded ones, all 33 intervals excluding 0 — **graph6 on LINUX 1.003 [0.953, 1.054], sparse6 on AIDS 1.148 [1.111, 1.187]**. At ψ ≥ 1 the distance between two relabellings of one graph exceeds the distance between two different graphs. **ψ must be quoted with its metric**: under `padded_hamming` pooled `adjacency` ψ is 0.988 against `levenshtein`'s 0.072, a 14× difference. Also: the invariant set of the `n²` family is **exactly `{K_n}`**, now verified over all 995 connected graphs to `n = 7` by full `n!` enumeration | new row in the AE.3 table; §3's F3 caption | **T-17**, T-20 |
| **14** | **ρ moved by up to 0.07 between two independent 200-graph draws on AIDS** (0.329 vs 0.255). Direct support for [statistics](../statistics.md) D2: effective sample size is governed by **graphs**, not pairs | evidence for D2 | T-02 |
| **15** | **bliss / Traces stay cut** — the `pynauty` from-source build was rehearsed under gcc 12.2.0 and succeeded | decision S-g | — |

---

## 6. What to port, and what not to

**Port**: `min_dfs.py` **with `validate_min_dfs.py`**, and `agm_cam.py` with its 327-graph brute-force
check. The oracles are the value; without them the backends are unverified graph theory.

**Port as a gate, not a backend**: `test_kavin.py`. Any third-party canonical backend must clear its
**K2 (invariance) check before anything else** — K2 needs no oracle and it is where `kaviniitm`
died.

**Port as an analysis, not a backend**: `real_size_null.py`. Finding 1 is not a competitor property
and it will not surface from a competitor harness.

**Do not port**: `backends.py`'s subprocess bridge to the conda env, the synthetic-profile
generators in `sweep.py`/`scale.py` (superseded by the real cohort), and none of the three
gSpan / DFS-code repositories.

**Add as fixtures**: `K₃,₃` vs the triangular prism (WL distance 0, every canonical backend
separates them); and the running example `C₄(0,1,2,3) + K₃(3,4,5)`.

**Watch for**, in order of how quietly they fail:

1. Counting the adjacency matrix as `len('1010…') * 8` — inflates it 8× and hands us a baseline we
   beat for free ([adjacency-matrix](adjacency-matrix.md) §7).
2. ~~Inverting `pynauty.canon_label` — a deterministic wrong labelling that **passes F3**~~
   **CORRECTED 2026-08-15 by T-04**: inverting `canon_label` **fails F3 loudly** (non-invariant on
   every connected trial), and the prescribed guard `nx.is_isomorphic(G, relabelled)` **can never
   fire** — any bijective relabelling is isomorphic by construction. This trap is loud, not quiet.
   ([nauty](nauty.md) §1).
3. Fitting the WL kernel per batch rather than per dataset — makes the distance matrix depend on
   batching order ([wl-subtree-kernel](wl-subtree-kernel.md) §7).
4. Mixing min-DFS character-level and tuple-level Levenshtein — a 2× difference
   ([gspan-mdfsc](gspan-mdfsc.md) §3).
5. Returning AGM's incumbent instead of raising when the budget runs out — puts a non-canonical code
   in a column headed canonical ([agm](agm.md) §7).
6. **Accepting a third-party canonical backend on a single example** — `kaviniitm/DFSCode` agrees
   on the running example and is wrong on half of all 6-node graphs.
7. **Reporting ρ without the size null.** Every number in §4.1 looks respectable until the null row
   is added.

---

## 7. Still open

- ~~Suite 2 Claim A is partial.~~ **Complete, all ten datasets** — see §4.3. The last three rows
  (COIL-DEL, Mutagenicity, Protein) were measured with two deliberate trims, both documented in
  `scratch/real_suite2b.py`: AGM on a 50-graph subsample (failure rate only, no bit row) and
  `canonical_string` at a 2 s budget (failure rate only, no bit row — its bit counts would be
  conditioned on the graphs fast enough to finish, which is a biased sample). Every other
  representation ran on the full 400-graph sample.
- **Suite 2 ρ** does not exist at all — there is no GED reference above `n = 12` until T-05 runs.
  §4.1 and §4.2 are Suite 1 only. **Mutagenicity is the row to watch**: it is the one dataset where
  IsalGraph wins Claim A outright, and whether it also clears the size null on ρ is unknown.
- **Graph-level bootstrap CIs** on every ρ in §4.1 (D2). Finding 14 says they will be wide.
- ~~**Whether IsalGraph clears the size null anywhere in Suite 2.**~~ **ANSWERED 2026-08-16 by T-04a,
  and the answer is that the data does not determine it.** Against the **lower** bound
  (`BRANCH_FAST`) IsalGraph fails the size null on **5 of 5** Suite-2 datasets (−0.082 to −0.295);
  against the **upper** bound (`BIPARTITE`) it clears it on **5 of 5** (+0.027 to +0.383). Every one
  of those ten differences excludes zero under a paired graph-level bootstrap. **The verdict flips
  with the end of the bracket on all five datasets**, and GED lies between them.

  The mechanism is measured: `ρ(|n₁−n₂|, LB)` is **0.960–0.998**, so the lower bound very nearly *is*
  the size null and no representation can beat it — the comparison is degenerate by construction,
  not a fact about IsalGraph. `ρ(|n₁−n₂|, UB)` is 0.460–0.754, and on Suite 1 where truth exists
  `ρ(|n₁−n₂|, exact)` is 0.713–0.920, **between** the two arms. The bracket is valid; neither arm
  alone is a stand-in for the truth. **This is the empirical case for
  [approx_ged](../approx_ged.md) §4's no-interpolation rule** — a midpoint would have produced one
  confident answer to a question the data leaves open, five times. **Inherits: T-06, T-20.**
- **Whether AGM's GREC ceiling moves** under orbit pruning from `pynauty.autgrp`. It will move; it
  will not reach `n = 98`.
- **The realised-bytes column** for every method — measured only for the running example.
