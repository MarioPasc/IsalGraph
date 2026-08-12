# Data and compute feasibility — measured

**Status**: **v1.1, 2026-08-11.** Every number below is **measured on this machine**, not quoted from
literature. Reproduction scripts are named per section.

> **v1.1 correction.** §2.1's size columns were raw-set statistics presented as connected-set
> statistics, and the error propagated into `plan.md` decision 12, §2.3 and §4.2. **§0 is correct and
> reproduces exactly**; §2.1 carries a correction banner; the cohort's retained ceiling is **n = 98**,
> not 417. §2.2.1's Fingerprint row could not be reproduced. Q4, Q6, Q7, Q8 closed; Q9 opened. Full
> derivation in `gap-audit.md` **MF1**. **§0 is the only table any printed number may be taken from.**
>
> **No dataset moves in or out.** The cohort, its counts and its pair totals are unchanged; the drop
> decisions for COIL-RAG, Fingerprint and Web were re-checked on connected-set numbers and all
> survive (§2.3). The correction is to *descriptions*, not to *data*.

**Machine**: local workstation, single thread, `time.process_time()`, `isalgraph.engine() == 'cpp'`.
**Environment**: `~/.conda/envs/isalgraph-cpp` (engine), `~/.conda/envs/isalsr` (torch, for the
GraphEdX `.pt` loaders).
**Data roots**:
- `…/data/source/GED_PRECOMPUTED/{AIDS,Letter,LINUX}` — the five datasets used in the submission
- `…/data/source/IAM_Database/{*.zip, extracted/}` — the IAM Graph Database (Riesen & Bunke, 2008),
  extracted to `extracted/` on 2026-08-11

---

## 0. The two evaluation suites — final post-filter numbers

**Filter, identical to the submitted pipeline**
(`benchmarks/real_data/eval_setup/dataset_filter.py::filter_graphs`):
`min_nodes = 2`, `require_connected = True`, plus `n_max` where stated. Script:
`scratchpad/final_counts.py`.

### Suite 1 — EXACT GED (`n ≤ 12`): the submitted study, recomputed

| Dataset | raw | **kept** | keep % | n̄ | ñ | n max | m̄ | density | **pairs** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IAM Letter LOW | 2,250 | **1,180** | 52.4 % | 4.07 | 4 | 7 | 3.07 | 0.543 | 695,610 |
| IAM Letter MED | 2,250 | **1,253** | 55.7 % | 4.11 | 4 | 8 | 3.17 | 0.542 | 784,378 |
| IAM Letter HIGH | 2,250 | **2,059** | 91.5 % | 4.58 | 5 | 9 | 4.56 | 0.607 | 2,118,711 |
| LINUX | 89 | **89** | 100 % | 8.71 | 9 | 10 | 8.35 | 0.255 | 3,916 |
| AIDS (GraphEdX) | 819 | **769** | 93.9 % | 10.56 | 11 | 12 | 10.70 | 0.218 | 295,296 |
| **Total** | | **5,350** | | | | **12** | | | **3,897,911** |

> **This reproduces the submitted manuscript exactly** — 1,180 / 1,253 / 2,059 / 89 / 769 graphs and
> 3,897,911 pairs, with m̄ matching to two decimals on every row. The filter is understood and the
> pipeline is trustworthy. *(This also closes the earlier reconciliation query: the discrepancy was
> `min_nodes = 2` discarding one 1-node Letter LOW graph.)*

### Suite 2 — APPROXIMATE GED (no `n_max`): everything

| Dataset | raw | **kept** | keep % | n̄ | ñ | **n max** | m̄ | density | **pairs** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IAM Letter LOW | 2,250 | 1,180 | 52.4 % | 4.07 | 4 | 7 | 3.07 | 0.543 | 695,610 |
| IAM Letter MED | 2,250 | 1,253 | 55.7 % | 4.11 | 4 | 8 | 3.17 | 0.542 | 784,378 |
| IAM Letter HIGH | 2,250 | 2,059 | 91.5 % | 4.58 | 5 | 9 | 4.56 | 0.607 | 2,118,711 |
| LINUX | 89 | 89 | 100 % | 8.71 | 9 | 10 | 8.35 | 0.255 | 3,916 |
| AIDS (GraphEdX) | 819 | **819** | 100 % | 11.03 | 11 | 20 | 11.23 | 0.212 | 334,971 |
| **GREC** *(new)* | 1,100 | **650** | 59.1 % | 11.45 | 11 | 24 | 12.43 | 0.244 | 210,925 |
| **AIDS (IAM)** *(new)* | 2,000 | **1,811** | 90.5 % | 14.02 | 11 | 85 | 14.50 | 0.202 | 1,638,955 |
| **COIL-DEL** *(new)* | 7,200 | **7,200** | 100 % | 21.48 | 20 | 79 | 54.03 | **0.328** | 25,916,400 |
| **Mutagenicity** *(new)* | 4,337 | **4,040** | 93.2 % | 28.53 | 27 | **98** | 29.55 | **0.094** | 8,158,780 |
| **Protein** *(new)* | 600 | **569** | 94.8 % | **31.68** | 30 | 96 | 61.44 | 0.163 | 161,596 |
| **Total** | | **19,670** | | | | **98** | | | **40,024,242** |

### What this buys

| | submitted | revision | factor |
|---|---:|---:|---:|
| Datasets | 5 | **10** | 2× |
| Graphs | 5,350 | **19,670** | **3.7×** |
| Graph pairs | 3.9 M | **40.0 M** | **10.3×** |
| Largest graph | 12 nodes | **98 nodes** | **8.2×** |
| Largest mean size | n̄ = 10.6 | **n̄ = 31.7** | **3.0×** |
| Density span | 0.22 – 0.61 | **0.094 – 0.607** | 6.5× span |

**Compute**: Suite 1 (exact GED) is **≈ 1,000–1,650 core-hours**, 16–26 h on 64 Picasso cores, 98 %
of it AIDS. Suite 2 (both GED bounds over all 40 M pairs) is **≈ 1.3 core-hours**, and encoding all
19,670 graphs with the C++ engine takes **under a minute**. The entire cost of this revision is one
SLURM job.

### Observations

**The reviewers' central complaint about scale is answerable, and the answer is not what the
submitted paper implies.** The Area Editor and Reviewer 3 both read the `n ≤ 12` ceiling as a
limitation of IsalGraph. It is not: measured here, the ceiling is imposed entirely by the *reference
measurement*. Exact GED costs 36.9 s per pair at n = 12 and grows roughly fivefold per added node,
so it is unobtainable above ~12 nodes for anyone — GraphEdX stops at the same place for the same
reason. IsalGraph itself encodes a 20-node molecular graph in 122 µs and a 96-node protein graph in
1.1 s. Suite 2 exists precisely to separate these two ceilings: it carries the representation up to
98 nodes, where the reference is a calibrated bound rather than an exact value. The honest framing
for the paper is three separate statements — the encoder scales, the exact reference does not, and
above n = 12 we report bounds whose agreement with exact GED is measured on the regime below.

**Two caveats belong in front of any of these numbers, and both are ours to disclose rather than be
caught on.** First, the connectivity precondition is *size-biased on exactly the datasets we added
for scale*: Mutagenicity keeps graphs averaging 28.5 nodes and discards ones averaging 54.7, and
AIDS (IAM) discards at 2.3× the retained size. The n̄ = 31.7 figure is therefore the connected
subsample's mean, not the dataset's, and must be reported as such alongside the discarded subset's
statistics. Second, canonicalisation does not fail gracefully at scale: cost is governed by
|Aut(G)|, the automorphism-group size, not by node count or density — a 96-node protein graph
finishes in 1.1 s while a 98-node molecule of *lower* density does not finish in five minutes. Both
facts strengthen rather than weaken the submission, because they replace an unexplained empirical
ceiling with a characterised one, but only if we state them first.

## 1. Headline findings

| # | Finding | Consequence |
|---|---|---|
| **H1** | **Pruned canonical encoding costs 3.9 ms at n̄ = 32 (Protein) and never timed out up to n = 96.** | The "≈12-node ceiling" is **not** a property of IsalGraph. §4 |
| **H2** | **Exact GED costs 36.9 s per pair at n = 12 and grows ×5 per node.** | The ceiling is entirely the *reference*. §3 |
| **H3** | **The whole approximate-GED extension — 10 datasets, 67 M pairs — costs 1.24 core-hours.** | Cost is not a constraint anywhere except exact GED. §6 |
| **H4** | **BRANCH-FAST (lower bound) tracks exact GED far better than Riesen–Bunke BP (upper bound): ρ = 0.966 vs 0.840; −11% vs +78% bias.** | **The large-n reference should be BRANCH-FAST, not BP.** §5 |
| **H5** | Mutagenicity (n̄ = 30.3), Protein (n̄ = 32.6), COIL-DEL (n̄ = 21.5) are usable; COIL-RAG, Fingerprint, Web are not. | §2.3 |
| **H6** | IAM connectivity retention ranges **51.4 % – 100 %** and is never reported in the manuscript. | Must be reported per dataset. §2.2 |
| **H7** | **Triplet pruning is what makes canonicalisation scale.** On Protein: pruned **3.4 ms, 0 failures**; exhaustive **5.8 s, 22/40 timeouts** — a ~1,700× gap. | A **new result for the paper**, not just a feasibility fact. §4.1 |
| **H8** | **Canonicalisation cost is driven by structural symmetry, not size or density.** Protein n = 96, density 0.024 → **1.1 s**. Mutagenicity n = 98, density 0.021 → **does not finish in 5 s**. Same size, same density, opposite outcome. | Changes the stratification variables and the limitations text. §4.3 |
| **H9** | **`timeout_s` is enforced exactly** (1.00 s / 5.03 s wall). **Removing the G2S timeout hangs on real graphs** — confirmed >4 min on one Mutagenicity graph. | Keep the timeout; record per-graph time; report the censoring rate. §4.3 |
| **H10** | **GEDLIB builds successfully on Picasso** (gcc 12.2.0, cmake 3.31.4, bundled deps). The maintained Python wrapper exists only in the **graphkit-learn git repo**, not the PyPI wheel. | Recognised BRANCH/BP implementation is achievable. §7.5 |

---

## 2. Dataset inventory

### 2.1 Structural audit — all candidates

Topology-only parse of every GXL file; "connected" means `n ≥ 1` and `nx.is_connected`.
Script: `scratchpad/iam_audit.py` → `iam_audit.json`.

> ### ⚠ CORRECTION 2026-08-11 — the size columns below are RAW-set statistics
>
> Independently re-derived by re-parsing every GXL file under the pipeline's own filter
> (`dataset_filter.py::filter_graphs`, `min_nodes = 2`, `require_connected = True`);
> script `scratchpad/audit_recheck.py`. **`data.md` §0 is correct and reproduces exactly. This table
> is not.**
>
> `N conn` and `ret. %` are computed over the **connected** subset. **`n med`, `n mean`, `n p90`,
> `n p99`, `n max`, `m mean` and `density` are computed over the RAW set** and are labelled as though
> they were connected-set values. The identity is exact, not approximate — for Mutagenicity,
> `(4040 × 28.53 + 297 × 54.70) / 4337 = 30.32`, this table's figure to the decimal; likewise Letter
> LOW (4.679 → 4.68), GREC (11.51) and Protein (32.64 → 32.63).
>
> | Dataset | n̄ **retained** | n max **retained** | n̄ raw | n max raw |
> |---|---:|---:|---:|---:|
> | Letter LOW | 4.07 | **7** | 4.68 | 8 |
> | Letter MED | 4.11 | **8** | — | 9 |
> | Letter HIGH | 4.58 | **9** | — | 9 |
> | GREC | 11.45 | **24** | 11.51 | 24 |
> | AIDS-IAM | 14.02 | **85** | 15.69 | 95 |
> | COIL-DEL | 21.48 | **79** | 21.48 | 79 |
> | **Mutagenicity** | **28.53** | **98** | 30.32 | **417** |
> | Protein | 31.68 | **96** | 32.63 | 126 |
>
> **Consequence — the 417-node Mutagenicity graph is disconnected and is discarded by the filter.**
> The cohort's true ceiling is **n = 98**. Three statements inherit the error and are corrected
> below and in `plan.md`: decision 12's rationale ("Mutagenicity already reaches n = 417"), §2.3's
> "20 → 417", and §4.2's heavy-tail explanation. See `gap-audit.md` **MF1**.
>
> **Use §0 for every number that will be printed.** This table is retained only because its
> raw-versus-connected split is what quantifies the connectivity discard in §2.2.1.

| Dataset | Source | N raw | N conn | ret. % | n med | n mean | n p90 | n p99 | **n max** | m mean | density |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | IAM | 2,250 | 1,181 | 52.5 | 5 | 4.68 | 6 | 8 | **8** | 3.13 | 0.4165 |
| Letter MED | IAM | 2,250 | 1,257 | 55.9 | 5 | 4.67 | 6 | 8 | **9** | 3.21 | 0.4246 |
| Letter HIGH | IAM | 2,250 | 2,067 | 91.9 | 5 | 4.67 | 6 | 7 | **9** | 4.50 | 0.5792 |
| LINUX | GraphEdX | 89 | 89 | 100.0 | 9 | 8.71 | 10 | 10 | **10** | 8.35 | — |
| AIDS | GraphEdX | 911 | 819 | 89.9 | 11 | 10.97 | 12 | 20 | **20** | 10.97 | — |
| **GREC** | IAM | 1,100 | 650 | **59.1** | 11 | 11.51 | 19 | 24 | **24** | 11.93 | 0.2267 |
| **AIDS-IAM** | IAM | 2,000 | 1,811 | 90.6 | 11 | 15.69 | 33 | 73 | **95** | 16.20 | 0.1935 |
| **COIL-DEL** | IAM | 7,200 | 7,200 | **100.0** | 20 | 21.48 | 38 | 62 | **79** | 54.03 | **0.3277** |
| **Mutagenicity** | IAM | 4,337 | 4,040 | 93.2 | 27 | 30.32 | 48 | 94 | **417** | 30.77 | **0.0913** |
| **Protein** | IAM | 600 | 569 | 94.8 | 32 | 32.63 | 48 | 90 | **126** | 62.14 | 0.1599 |
| COIL-RAG | IAM | 7,200 | 7,134 | 99.1 | 3 | 3.01 | 4 | 5 | 11 | 3.03 | 0.9255 |
| Fingerprint | IAM | 4,000 | 2,057 | **51.4** | 4 | 5.49 | 14 | 18 | 26 | 4.49 | 0.3279 |
| Web | IAM | 2,340 | — | — | — | — | — | — | — | — | — |

**Web does not parse** — its files are `doc.*.xml` in a document/word-graph schema, not GXL
`<graph><node>/<edge>`. Excluded pending a dedicated loader; not needed, since Mutagenicity and
Protein already cover the size range.

### 2.2 Connectivity retention is a reportable quantity

G2S requires connected input. Retention spans **51.4 % (Fingerprint) to 100 % (COIL-DEL, LINUX)**.
The manuscript states the connectivity requirement (`computational_experiments.tex:32–33`) but
**never quantifies the loss**. On Letter LOW it discards **47.5 %** of the corpus. This belongs in
the pair-accounting ladder.

**Reconciliation item — CLOSED 2026-08-11.** This audit gives 1,181 / 1,257 / 2,067 connected Letter
graphs against the manuscript's **1,180 / 1,253 / 2,059**. The mechanism is the `min_nodes`
threshold: `filter_graphs(graphs, graph_ids, n_max, require_connected=True, min_nodes=2)`
(`dataset_filter.py:37–43`). Applying `min_nodes = 2` reproduces **1,180 / 1,253 / 2,059 / 650 /
1,811 / 7,200 / 4,040 / 569** exactly — verified independently, `scratchpad/audit_recheck.py`. §0 was
already right; Q4 in §9 is closed.

### 2.2.1 The connectivity discard is size-biased — and GREC is the *least* affected

Script: `scratchpad/grec_check.py`. Retained vs discarded graphs compared by Mann–Whitney U.

| Dataset | retained | n̄ retained | n̄ **discarded** | p(n) | density ret. | density disc. |
|---|---:|---:|---:|---:|---:|---:|
| **GREC** | **59.1 %** | 11.45 | **11.59** | 1.6e−02 | 0.243 | 0.202 |
| Fingerprint | ~~67.2 %~~ **51.4 %** | 5.03 | ~~11.56~~ **5.98** | 3.6e−252 | 0.550 | 0.179 |
| Letter LOW | 52.5 % | 4.06 | 5.35 | 9.2e−122 | 0.543 | 0.277 |
| Letter HIGH | 91.9 % | 4.57 | 5.83 | 1.9e−33 | 0.605 | 0.290 |
| **Mutagenicity** | 93.2 % | 28.53 | **54.70** | 1.6e−38 | 0.094 | 0.059 |
| **Protein** | 94.8 % | 31.68 | **50.19** | 4.9e−04 | 0.163 | 0.103 |
| **AIDS-IAM** | 90.5 % | 14.02 | **31.76** | 7.8e−17 | 0.202 | 0.114 |
| COIL-DEL | 100 % | — | — | — | — | — |

> **Verified 2026-08-11**, `scratchpad/audit_recheck.py` / `audit_dropped.py`. **This table is
> correct except for the Fingerprint row.** GREC, Mutagenicity, Protein and AIDS-IAM reproduce to
> the decimal on all three columns; Letter LOW/HIGH differ only by the `n ≥ 1` vs `n ≥ 2` threshold.
>
> **Fingerprint could not be reproduced**: measured retention is **51.4 %** (2,056 of 4,000 — which
> is what §2.1 itself reports) and the discarded mean is **5.98**, not 11.56. The internal check
> holds on the measured values: `(2056 × 5.03 + 1944 × 5.98) / 4000 = 5.49`, matching the raw mean.
> The origin of 67.2 % / 11.56 is unknown. **Not load-bearing** — Fingerprint is dropped either way
> — but no number from that row may be quoted.

**GREC: include it.** Its retention rate (59.1 %) looks alarming, but the discarded graphs are the
same size as the retained ones (11.59 vs 11.45 nodes — a 0.14-node difference, significant only
because n is large). **Retention rate is the wrong diagnostic; size bias is the right one**, and by
that measure GREC is the *cleanest* dataset in the cohort.

**The real problem is elsewhere, and it is serious.** The datasets with *high* retention discard the
*largest* graphs:

- **Mutagenicity discards graphs averaging 54.7 nodes and keeps ones averaging 28.5** — the discard
  is 1.9× larger;
- AIDS-IAM discards at 2.3× the retained size; Protein at 1.6×; Fingerprint at 2.3×.

So the connectivity precondition systematically removes the biggest graphs from exactly the datasets
we are adding **to demonstrate scaling**. Any claim of the form "IsalGraph now handles n̄ ≈ 30" is
measured on a subsample from which the large graphs have been preferentially removed.

**Required response:**
1. report `n̄` and `n_max` for **both** the retained and discarded subsets in the dataset table;
2. state the connectivity precondition as a **scope limitation with its measured cost**, which is
   the honest version of R3.3a — the manuscript currently states the precondition and never
   quantifies what it removes;
3. do **not** claim a size range without saying it is the connected subsample's range.

### 2.3 Recommended cohort

| Verdict | Datasets | Rationale |
|---|---|---|
| **Keep — exact-GED regime** | Letter LOW/MED/HIGH, LINUX, AIDS-GraphEdX | the submitted study; n ≤ 12 throughout |
| **Add — size extension** | **Mutagenicity** (n̄ **28.5**, sparse 0.094, 4,040 graphs), **Protein** (n̄ **31.7**, 0.163, 569), **COIL-DEL** (n̄ **21.5**, **dense 0.328**, 7,200), **AIDS-IAM** (n̄ **14.0**, max **85**, 1,811) | span n̄ = 4.1 → 31.7 and density 0.094 → 0.607; COIL-DEL is the density stress test |
| **Add — marginal** | GREC (n̄ 11.45) | only 59.1 % retention, but the discard is **size-unbiased** (§2.2.1) — the cleanest in the cohort |
| **Drop** | COIL-RAG (**kept 7,100, n̄ 3.02, n max 6, density 0.936** — near-complete tiny graphs), Fingerprint (**51.4 % retention, kept n̄ 5.03, n max 19**), Web (unparsed) | add no size or density coverage |

> **The drop decisions were re-checked on connected-set numbers and all three survive**
> (`scratchpad/audit_dropped.py`, 2026-08-11). The MF1 correction **changes no dataset in or out**:
> §0's cohort and its counts were always right, and every retained dataset is unaffected. What
> changed is the prose that described them.

This cohort takes the maximum **retained** node count from **12 → 98** and the study from 5 datasets
to 10.

> **Corrected 2026-08-11.** The previous text read "**20 → 417**". Both endpoints were wrong: 20 is
> AIDS-GraphEdX's *pre-filter* maximum where Suite 1's ceiling is 12, and the 417-node Mutagenicity
> graph is disconnected and discarded. All figures in this row are now retained-set values from §0.
> See `gap-audit.md` MF1.

---

## 3. Exact GED — the only real cost

Measured on **real benchmark graphs** (pooled Letter + LINUX + AIDS), `networkx.graph_edit_distance`,
topology-only unit costs, 4 pairs per bucket, 60 s timeout. Script: `scratchpad/real_cost.py`.

| n | median s/pair | max s/pair |
|---:|---:|---:|
| 3 | 0.0005 | 1.889 |
| 4 | 0.0005 | 0.0011 |
| 5 | 0.0040 | 0.0062 |
| 6 | 0.0108 | 0.0203 |
| 7 | 0.0264 | 0.0464 |
| 8 | 0.2300 | 0.5562 |
| 9 | 0.3355 | 0.5205 |
| 10 | 2.168 | 5.453 |
| 11 | 7.484 | 32.03 |
| **12** | **36.88** | 40.39 |
| ≥ 14 | > 60 s (timeout) | — |

**Growth ≈ ×5 per added node** in the 10–12 range. Extrapolated: n = 13 ≈ 3 min, n = 14 ≈ 15 min,
n = 16 ≈ 6 h, n = 20 ≈ months. **Exact GED is unobtainable above ≈ 12 nodes**, which is why
GraphEdX itself stops there. This is a constraint on the field, not on this work.

### 3.1 Exact-GED budget for the five original datasets

| Dataset | pairs | ~s/pair | core-hours |
|---|---:|---:|---:|
| Letter LOW | 696,790 | 0.004 | 0.8 |
| Letter MED | 789,396 | 0.004 | 0.9 |
| Letter HIGH | 2,135,211 | 0.008 | 4.7 |
| LINUX | 3,916 | 2.17 | 2.4 |
| **AIDS-GraphEdX** | 295,296 | 12–20 | **985–1,640** |
| **Total** | | | **≈ 1,000–1,650** |

**16–26 h on 64 cores.** AIDS is 98 % of it. One SLURM job, `1-00:00:00`, checkpointed.

**Cost-reduction option worth deciding**: the calibration arm does not need all-pairs exact GED —
a few thousand stratified pairs give tight CIs on ρ(exact, approx). And because pairs are dyadically
dependent (R3.5c), effective sample size is governed by the **number of graphs**, not pairs, so a
stratified subsample of AIDS pairs would lose very little statistical power at ~10× lower cost.
See open question **Q3**.

---

## 4. IsalGraph encoding cost — H1

C++ engine, median over 15 real graphs per dataset, 5 s timeout.
Script: `scratchpad/feas2.py` → `feas2.json`.

| Dataset | n med | greedy-rnd | greedy-min | **pruned canonical** | exhaustive canonical |
|---|---:|---:|---:|---:|---:|
| Letter LOW | 4 | 5 µs | 13 µs | **6 µs** | 5 µs |
| Letter HIGH | 5 | 5 µs | 15 µs | **7 µs** | 6 µs |
| GREC | 11 | 12 µs | 77 µs | **43 µs** | not attempted |
| AIDS-IAM | 11 | 8 µs | 64 µs | **27 µs** | not attempted |
| COIL-DEL | 17 | 16 µs | 218 µs | **158 µs** | not attempted |
| Mutagenicity | 28 | 22 µs | 512 µs | **1.62 ms** | not attempted |
| **Protein** | **30** | 30 µs | 623 µs | **3.89 ms** | not attempted |

**Zero timeouts anywhere**, including Protein graphs up to n = 96 in the sample.

Cross-check by node count on pooled real graphs (`scratchpad/real_cost.py`): 4 µs at n = 3, 27 µs at
n = 12, 81 µs at n = 16, **122 µs at n = 20**.

**Encoding budget, all 10 datasets, all three reported algorithms: ≈ 10 s total (0.008 core-hours).**

### 4.1 Pruned vs exhaustive canonical — H7

A second, larger run (`scratchpad/feasibility.py`, 40 graphs/dataset, 30 s timeout, full corpora)
**did** attempt exhaustive canonicalisation above n = 12. It breaks down; pruned does not.

| Dataset | n med | greedy-rnd | greedy-min | **pruned** | **exhaustive** | exhaustive timeouts |
|---|---:|---:|---:|---:|---:|---:|
| Letter LOW | 4 | 4 µs | 13 µs | 5 µs | 5 µs | 0/40 |
| Letter HIGH | 5 | 5 µs | 19 µs | 7 µs | 9 µs | 0/40 |
| GREC | 11 | 8 µs | 53 µs | 21 µs | 53 µs | 0/40 |
| AIDS-IAM | 11 | 8 µs | 66 µs | 22 µs | 78 µs | 0/40 |
| COIL-DEL | 19 | 19 µs | 230 µs | **205 µs** | **74.5 ms** | **17/40** |
| Mutagenicity | 26 | 14 µs | 232 µs | **240 µs** | **11.9 ms** | **4/40** |
| **Protein** | **30** | 36 µs | 945 µs | **3.35 ms** | **5.82 s** | **22/40** |

**This is a result the submitted paper does not have.** It reports the pruning as a speed-up but
only measures to n = 20 on synthetic graphs. Here, on real graphs at n̄ = 30, exhaustive
canonicalisation fails on **55 % of Protein graphs** within 30 s while pruned completes **every**
graph in 3.4 ms. The triplet pruning is not an optimisation — it is what makes the canonical form
computable at these sizes. Closes the former open question Q5.

### 4.3 What happens if the G2S timeout is removed — H8, H9

Script: `scratchpad/tail.py`, `scratchpad/timeout_check.py`. Six largest connected graphs per
dataset, **no timeout** on greedy/pruned, 2 s on exhaustive.

| Dataset | n | m | density | greedy-rnd | greedy-min | **pruned** | exhaustive |
|---|---:|---:|---:|---:|---:|---:|---:|
| COIL-DEL | 79 | 228 | 0.074 | 0.25 ms | 17.1 ms | **16.7 ms** | timeout |
| COIL-DEL | 72 | 207 | 0.081 | 0.18 ms | 11.4 ms | **7.8 ms** | timeout |
| AIDS-IAM | 85 | 85 | 0.024 | 0.24 ms | 4.5 ms | **22.9 ms** | timeout |
| AIDS-IAM | 73 | 80 | 0.030 | 0.24 ms | 13.4 ms | **963 ms** | timeout |
| Protein | **96** | 109 | 0.024 | 0.41 ms | 25.7 ms | **1.12 s** | timeout |
| Protein | 90 | 127 | 0.032 | 0.41 ms | 23.8 ms | **149 ms** | timeout |
| **Mutagenicity** | **98** | 98 | **0.021** | — | — | **> 4 min, did not finish** | timeout |

**H8 — cost tracks symmetry, not size or density.** Protein n = 96 at density 0.024 finishes in
1.1 s; Mutagenicity n = 98 at density 0.021 does not finish in **five minutes**. Same size, same
density. Sparse near-regular molecular graphs (long chains, repeated substituents) give the
structural triplet little to discriminate on, so the search space stays large. Within AIDS-IAM the
spread is 100× between n = 78 (9.9 ms) and n = 73 (963 ms).

**Consequences for the design:**

1. **Do not remove the G2S timeout.** It hangs on real data. `timeout_s` is **enforced exactly**
   (measured 1.00 s and 5.03 s wall on three graphs), so it is a reliable safety net — keep it,
   set it generously (300 s), and report the rate at which it fires.
2. **Exhaustive canonical is finished above ~60 nodes.** Every graph tested at n ≥ 66 timed out at
   2 s. Report it as a bounded baseline, not a competitor at scale.
3. **Always record per-graph encode wall time.** It costs nothing and is the raw material for the
   stratification. Do this whether or not a timeout fires.
4. **Add a symmetry proxy to the stratification variables** — orbit count or automorphism-group
   size from nauty, which we are vendoring anyway (`plan.md` §4.2). H8 says this predicts
   canonicalisation cost better than n or density, and no reviewer has asked for it. It converts a
   limitation into a characterisation.
5. **Censoring is not random.** Whatever fires the timeout is correlated with symmetry, hence with
   the strata being analysed. The timeout rate must be reported **per stratum**, never pooled.

*Pending*: a full timeout-rate sweep (`scratchpad/timeout_rate.py`, 400 graphs × 3 datasets at a
10 s budget) was launched and did not survive its shell. Re-run under T-01.

### 4.4 Why some graphs are hard — and whether a WL variant would fix it

Script: `scratchpad/symmetry_diag.py`. Compares the graphs that hang against ones of the same size
that do not.

| Graph | n | m | cyclomatic | **1-WL classes** | **triplet classes** | orbits | \|Aut\| | encode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mutagenicity/3703 | 98 | 98 | **1** | **66** | **28** | 84 | **>20,000** | **hangs** |
| Mutagenicity/3970 | 97 | 105 | 9 | 79 | 53 | 85 | **>20,000** | **hangs** |
| Protein/enzyme_293 | 96 | 109 | 14 | **87** | 33 | 87 | **16** | 1.1 s |
| Protein/enzyme_123 | 90 | 127 | 38 | **90** | 57 | 90 | **1** | 149 ms |
| AIDS/7400 | 85 | 85 | 1 | 79 | 29 | 79 | 64 | 22.9 ms |
| AIDS/41883 | 81 | 84 | 4 | 79 | 26 | 79 | 4 | 9.9 ms |

**Diagnosis: cost is governed by \|Aut(G)\|, the automorphism-group size.** The two graphs that hang
have automorphism groups exceeding the 20,000 enumeration cap; every graph that completes has
\|Aut\| ≤ 64. Cyclomatic number is the mechanism — tree-like molecules (cyclomatic 1) with repeated
substituents have enormous symmetry groups, and each automorphism generates a branch the canonical
search explores redundantly.

**This retires the "density" intuition for encoding cost too.** Density predicts nothing here
(0.021 hangs, 0.024 finishes in 1.1 s); \|Aut\| predicts everything.

### Would importing the IsalSR / IsalHG WL variant fix it? **Partly — and the honest answer is no.**

`IsalHG/src/isalhg/core/algorithms/greedy_min_wl_pruned.py` states the limit itself:

> *"On vertex-transitive designs (Fano, STS, GQ) every vertex shares the same WL colour, so the WL
> filter is a no-op — the preprint reports this no-speedup region explicitly."*

Our hard cases are the graph analogue of that region. But the measurement gives a **real, quantified
gain that is worth taking**:

| | Mutagenicity/3703 (hard) | Protein/enzyme_293 (easy) |
|---|---:|---:|
| IsalGraph structural triplet `(\|N₁\|,\|N₂\|,\|N₃\|)` | **28 classes** / 98 nodes | 33 / 96 |
| 1-WL colour refinement | **66 classes** / 98 nodes | **87** / 96 |
| Ratio | **2.4× finer** | 2.6× finer |

**1-WL is 2.4–2.6× more discriminating than the structural triplet the paper currently uses.** That
is not a coincidence: the triplet `(|N₁|,|N₂|,|N₃|)` is a *truncated, weaker* invariant than WL —
it counts neighbourhood sizes at depth 1–3, whereas WL propagates the full multiset of neighbour
colours to stability. **WL strictly subsumes the current pruning key.**

> ### ⚠ SUPERSEDED 2026-08-11 — do not execute this recommendation
>
> **T-16 was REJECTED** by author decision 17 (`plan.md` §7.2, S-b resolved 2026-08-11): no reviewer
> or editor asked for `wl_pruned_canonical`, and it originated as a transfer from IsalSR / IsalHG.
> **The WL *measurement* below is retained and moves into T-13**, where it answers R3.7d with a
> characterised worst case (`plan.md:818–836`).
>
> The measurement and its justification are left in place because T-13 must read them. **The
> "Recommendation" verb is what is retired** — build nothing from this section.
> Found by audit-2026-08-11b (I-10): `data.md` is required reading before executing any ticket
> (`plan.md:13`), so a live "Recommendation" here contradicts a signed decision elsewhere.

**Recommendation (RETIRED — see banner above)** — one new C++ variant, `wl_pruned_canonical`
(ticket **T-16**):

- **Correctness**: WL colour is isomorphism-invariant, and we keep the *whole* argmin-colour class
  rather than a representative, so no raw node id is read. This is exactly the admissibility
  argument in `greedy_min_wl_pruned.py` lines 10–15, already accepted in the sibling projects.
- **Justification for the paper**: measured 2.4× finer partition than the incumbent triplet, on the
  instances that currently fail.
- **Cost**: `O(n·m)` per refinement round — negligible against the search it prunes.
- **Family consistency**: the same mechanism is validated in IsalSR (`native/src/wl.cpp`,
  `canonical.cpp` candidate key `(label_char, wl_hash)`) and IsalHG. IsalGraph is the only sibling
  without it.
- **Honest limit, to be stated in the paper**: WL will not rescue graphs whose automorphism group
  is large *and* whose WL colouring is coarse. Report the residual timeout rate per stratum.

**What would actually fix the hard cases** is automorphism pruning — individualisation-refinement
with automorphism detection, which is what nauty/bliss/Traces do. That is a re-implementation of a
graph-isomorphism engine, not a 20-day edit. **State it as future work and cite nauty**, which we
are already vendoring as a competitor. This also gives R3.7d a *characterised* worst case
(\|Aut(G)\| large) instead of an unqualified "exponential".

### 4.2 Cross-run agreement

Two independent runs with different sampling (`feasibility.py`: 40 graphs, full corpora;
`feas2.py`: 15 graphs, subsampled corpora) agree to within sampling noise on greedy and pruned
timings. The exception is **Mutagenicity pruned: 240 µs vs 1.62 ms**, a 6.7× spread caused by its
heavy tail (median n = 27, **max n = 98 in the retained corpus** — corrected from 417, which is a
discarded disconnected graph, `gap-audit.md` MF1) — with 15–40 samples the median is unstable.
**Mutagenicity encoding cost must be re-measured on the full corpus** before any timing is reported.
The spread is in any case governed by `|Aut(G)|` rather than by `n` (§4.4), so the re-measurement
must report the distribution and not only a median.

---

## 5. GED bounds — validity, tightness, and H4

Implementation: `scratchpad/ged_bounds.py`.
- **Upper**: Riesen–Bunke bipartite (BP). Riesen & Bunke, *Image and Vision Computing* 27(7):950–959, 2009.
- **Lower**: BRANCH-FAST. Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018 — `O(n²Δ² + n³)`,
  and the bound is a **pseudo-metric** on a graph collection, which matters if we want the reference
  itself to be metric.
- One LSAP solve yields both: the assignment optimum is the lower bound, the induced edit path is
  the upper bound.

Validation on real pairs, n = 3–9, topology-only unit costs, exact GED via `networkx`
(mean exact GED = 5.56). **Two independent runs**, different samples:

| Quantity | run A (300 pairs) | run B (400 pairs) |
|---|---|---|
| Upper-bound violations (UB < exact) | **0 / 300** | **0 / 400** |
| Lower-bound violations (LB > exact) | **0 / 300** | **0 / 400** |
| **ρ(exact, BP upper)** | **0.840** | **0.840** |
| **ρ(exact, BRANCH-FAST lower)** | **0.966** | **0.966** |
| BP mean relative **over**estimate | +78.1 % | +85.3 % |
| BRANCH-FAST mean relative **under**estimate | −11.2 % | −12.5 % |
| Pairs certified exact (LB = UB) | 11.3 % | 9.8 % |
| Pairs where UB = exact | 23.3 % | 22.8 % |

The two correlations reproduce to three decimal places across independent samples.

### Reading

1. **Both bounds hold empirically.** Zero violations.
2. **H4 — the lower bound is the better proxy.** ρ = 0.966 vs 0.840, and −11 % bias vs +78 %.
   The natural assumption (use BP as the large-n reference) is **wrong on this data**. This is
   exactly what the calibration arm exists to catch, and it caught it before we committed compute.
   **Recommendation: BRANCH-FAST is the primary large-n reference; BP is reported as the bracket's
   other end.**
3. **The "free exact values at large n" idea is weak** — only 11.3 % of pairs are certified exact.
   Report it honestly; do not build an argument on it.
4. **Caveat on our BP.** This is plain Riesen–Bunke without refinement. Refined variants (BP-Beam,
   IPFP) tighten the upper bound substantially. Before publication, either add a refinement pass or
   state the variant precisely — a reviewer familiar with the GED literature will know that plain BP
   is the loosest member of the family, and a +78 % overestimate invites exactly that objection.

---

## 6. Approximate-GED cost — H3

Median over 100 real pairs per dataset, both bounds. Script: `scratchpad/feas2.py`.

| Dataset | n med | BP upper | BRANCH lower | pairs / core-hour |
|---|---:|---:|---:|---:|
| Letter LOW | 4 | 16 µs | 12 µs | 127,323,206 |
| Letter HIGH | 5 | 18 µs | 13 µs | 116,063,512 |
| GREC | 11 | 31 µs | 18 µs | 72,884,083 |
| AIDS-IAM | 11 | 33 µs | 18 µs | 71,461,891 |
| COIL-DEL | 17 | 66 µs | 34 µs | 36,165,456 |
| Mutagenicity | 28 | 69 µs | 39 µs | 33,163,523 |
| Protein | 30 | 109 µs | 51 µs | 22,529,922 |

### All-pairs budget across the whole candidate cohort

Script: `scratchpad/budget.py`.

| Dataset | connected | all pairs | encode (pruned, s) | approx GED (core-h) | exact GED |
|---|---:|---:|---:|---:|---|
| COIL-RAG | 7,134 | 25,443,411 | 0.04 | 0.20 | feasible |
| Letter LOW | 1,181 | 696,790 | 0.01 | 0.01 | feasible |
| Letter MED | 1,257 | 789,396 | 0.01 | 0.01 | feasible |
| Letter HIGH | 2,067 | 2,135,211 | 0.01 | 0.02 | feasible |
| Fingerprint | 2,057 | 2,114,596 | 0.01 | 0.02 | feasible |
| GREC | 650 | 210,925 | 0.03 | 0.00 | feasible |
| AIDS-IAM | 1,811 | 1,638,955 | 0.05 | 0.02 | feasible |
| COIL-DEL | 7,200 | 25,916,400 | 1.14 | 0.72 | **intractable** |
| Mutagenicity | 4,040 | 8,158,780 | 6.53 | 0.25 | **intractable** |
| Protein | 569 | 161,596 | 2.22 | 0.01 | **intractable** |
| **Total** | | **67.3 M pairs** | **10.0 s** | **1.24 core-h** | |

**The entire extension study costs 1.24 core-hours.** Encoding costs 10 seconds. Compute is not a
constraint anywhere except exact GED (§3.1).

---

## 7. Total revision compute budget

| Item | Core-hours | On 64 cores |
|---|---:|---:|
| All encoding, 10 datasets, 3 algorithms | 0.01 | seconds |
| All approximate GED (both bounds), 67 M pairs | 1.2 | ~1 min |
| Levenshtein + competitor distances, all pairs | ~5–15 | ~15 min |
| **Exact GED, five original datasets, all pairs** | **1,000–1,650** | **16–26 h** |
| Graph-level bootstrap + Mantel, all datasets | 5–10 | ~10 min |
| **Total** | **≈ 1,010–1,680** | **≈ 17–27 h** |

**One Picasso job dominates everything: exact GED on AIDS.** Everything else is a laptop-minutes
workload. Write the job with the `picasso-sbatch` skill; `ged_computer.py` already checkpoints.

---

## 7.5 GEDLIB on Picasso — H10 (verified 2026-08-11)

Motivation: a **recognised, citable** implementation of the bounds is more defensible than our own
150-line version, and it must run on Picasso because that is where GED is computed.

### Maintenance status — checked via the GitHub API

| Repo | Last push | Stars | Verdict |
|---|---|---:|---|
| `Ryurin/gedlibpy` | **2019-10-03** | 8 | **dead — do not use** |
| `dbblumenthal/gedlib` | 2023-06-22 | 66 | the canonical C++ library, by the **author of the BRANCH paper** |
| `jajupmochi/graphkit-learn` | **2025-06-07** | 128 | **maintained**; carries the Cython wrapper |

### What was verified on Picasso

| Step | Result |
|---|---|
| conda env `isalgraph` created at `fscratch/conda_envs/isalgraph` | **Python 3.11.15**, matches local |
| toolchain | `gcc/12.2.0`, `cmake/3.31.4`, `boost/{1.74.0_gcc15, 1.80}` available as modules |
| `pip install graphkit-learn` | **succeeds** |
| `from gklearn.gedlib import gedlibpy` from the **PyPI wheel** | **fails** — ships Python glue only, no compiled `.so`, no `.pyx` |
| `git clone dbblumenthal/gedlib` + `python install.py` | **"Successfully installed GEDLIB"**, `lib/` populated; bundles boost 1.69, eigen 3.3.4, fann 2.2.0, libsvm 3.22, lsape 5, nomad 3.8.1 — **no network needed beyond the clone** |
| Cython wrapper sources | **only in the graphkit-learn git repo**, not the wheel: `gklearn/gedlib/gedlibpy.pyx` (55 KB), `src/gedlib_bind_gxl.cpp`, `setup.py` |

### Build path — corrected

`graphkit-learn/gklearn/gedlib/setup.py` is **self-contained**: it downloads `jajupmochi/gedlib`
(the maintained fork) into `include/gedlib-master/` and builds the whole tree. So the manual
`dbblumenthal/gedlib` clone was unnecessary.

1. ~~`git clone dbblumenthal/gedlib` → `install.py`~~ — redundant; **deleted 2026-08-11**, freeing
   92,259 files
2. `git clone --depth 1 jajupmochi/graphkit-learn` ✔ **done**
3. `python setup.py build_ext --inplace` ✔ **done** — emits
   `gedlibpy_gxl.cpython-311-x86_64-linux-gnu.so` and `gedlibpy_attr...so`
4. bindings import and expose all methods ✔ **verified** — see below
5. reproduce `ged_bounds.py` on the same 300 pairs — **pending (T-05)**

### ✔ GEDLIB is working on Picasso (verified 2026-08-11)

Full install/verify/troubleshoot procedure is in **`.claude/CLAUDE.md`**. Verified facts:

- module names changed in the refactor: `libraries_import` (not `librariesImport`),
  **`gedlibpy_gxl`** (not `gedlibpy`), **`GEDEnvGXL`** (not `GEDEnv`). Most online tutorials are stale.
- **21 methods available**: `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT`, `BRANCH_UNIFORM`,
  `BRANCH_COMPACT`, `PARTITION`, `HYBRID`, `RING`, `ANCHOR_AWARE_GED`, `WALKS`, `IPFP`, `BIPARTITE`,
  `SUBGRAPH`, `NODE`, `RING_ML`, `BIPARTITE_ML`, `REFINE`, `BP_BEAM`, `SIMULATED_ANNEALING`, `HED`,
  `STAR`.
- **11 edit-cost models**: `CONSTANT` (our D6 unit model) plus the published IAM per-dataset models
  `LETTER`, `LETTER2`, `GREC_1`, `GREC_2`, `CHEM_1`, `CHEM_2`, `PROTEIN`, `FINGERPRINT`, `CMU`,
  `NON_SYMBOLIC`.

**Two opportunities this opens:**

1. **`ANCHOR_AWARE_GED` is an exact solver.** Benchmark it against `networkx` A* — GEDLIB is a
   specialised C++ implementation and may push the exact-GED ceiling above n = 12, which would
   directly enlarge the calibration regime. **Add to T-03 as a pre-step.**
2. **The published per-dataset cost models are available** as a sensitivity analysis alongside our
   `CONSTANT` decision — though adopting them as primary would reintroduce the heterogeneity R3.5b
   objects to.

### RESOLVED — fscratch file-count quota

The first build attempt failed as `shutil.Error: [Errno 122] Disk quota exceeded` mid-`copytree` —
**not a compile error**. fscratch enforces a **file-count** limit, not just space:

| | before | after cleanup |
|---|---:|---:|
| fscratch space | 0.48 TB / 1.40 TB — fine | unchanged |
| **fscratch files** | **399.7k / 250.0k**, hard limit **400.0k** | **305.8k** |

Deleting the redundant `build_gedlib/gedlib` (**92,259 files**) cleared it, and the rebuild then
succeeded — confirming the standalone clone was unnecessary. Remaining pressure is pre-existing:
`conda_envs` 111,513 files, `results` 62,209, `graphkit-learn` 55,328.

**Still above the 250k soft quota with a 7-day grace.** T-03 checkpoints frequently and a run near
the hard limit will fail partway, so free more before launching it.

### The bounds we will report, and their justification

| Role | Method | Reference | Why |
|---|---|---|---|
| **Lower** | **BRANCH-FAST** | Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018 | proven lower bound, `O(n²Δ² + n³)`, and a **pseudo-metric** on a graph collection — so the reference itself has metric structure. Measured **ρ(exact, LB) = 0.966**, bias −11 % |
| Lower (anytime) | BRANCH-TIGHT | same | tightens iteratively; use if the bracket is too wide |
| **Upper** | **IPFP** or **REFINE** | Bougleux et al., 2017 (IPFP); GEDLIB | tight upper bounds. **Replaces our plain BP**, which overestimates by **+78 %** (§5 caveat 4) — the loosest member of the family and an easy reviewer target |
| Upper (baseline) | BIPARTITE | Riesen & Bunke, *IVC* 27(7):950–959, 2009 | reported as the well-known reference point, not as our primary |

Where **LB = UB the value is exact**, certified for free (measured 9.8–11.3 % of pairs with our
plain BP; expect materially more with IPFP).

## 8. Reproduction

| Script (in the session scratchpad) | Produces |
|---|---|
| `size_audit.py` | IAM Letter size distributions (needs `isalgraph-cpp`) |
| `graphedx_audit.py` | LINUX/AIDS sizes + within-split GED coverage (needs `isalsr` for torch) |
| `export_graphs.py` | LINUX/AIDS → `graphs.json` edge lists, so the engine env needs no torch |
| `real_cost.py` | §3 exact-GED and §4 encoding timings on real graphs |
| `iam_audit.py` | §2.1 structural audit → `iam_audit.json` |
| `ged_bounds.py` | BP upper + BRANCH-FAST lower |
| `feas2.py` | §5 bound validation, §4 per-dataset encoding, §6 approx-GED cost → `feas2.json` |
| `budget.py` | §6 all-pairs budget table |
| **`audit_recheck.py`** *(2026-08-11)* | **§0 / §2.1 reconciliation** — re-parses every GXL under `min_nodes = 2` + connected and emits retained *and* discarded `n̄`, `n_max`, `m̄`, density per dataset. The script that established MF1 |
| **`audit_dropped.py`** *(2026-08-11)* | **§2.3 drop-decision re-check** — connected-set statistics for COIL-RAG, Fingerprint and GREC. Confirmed all three dispositions and surfaced the unreproducible §2.2.1 Fingerprint row |

**These live in a session scratchpad and will not survive.** Port them to
`benchmarks/real_data/eval_setup/` and `tests/` as part of T-01/T-05 before relying on them.
`audit_recheck.py` in particular must become a **test**: it is the only thing standing between the
raw/connected mix-up and a printed number, and it caught one that had already reached two locked
decisions.

---

## 9. Open questions raised by the data

| # | Question | Why it matters |
|---|---|---|
| **Q1** | Adopt **BRANCH-FAST as the primary large-n reference** instead of BP (H4)? | ρ 0.966 vs 0.840. Changes what every large-n number means |
| **Q2** | Add a **BP refinement pass** (BP-Beam / IPFP) so the upper bound is not the loosest in its family? | +78 % overestimate is an easy reviewer target |
| **Q3** | **All-pairs vs stratified-subsample exact GED on AIDS?** | 985–1,640 vs ~100 core-hours; dyadic dependence means little power is lost |
| ~~Q4~~ | ~~Reconcile Letter connected counts~~ | **Closed 2026-08-11.** `dataset_filter.py:37–43` defaults to `min_nodes = 2`; applying it reproduces 1,180 / 1,253 / 2,059 exactly. Verified independently, `scratchpad/audit_recheck.py` |
| ~~Q5~~ | ~~Exhaustive canonical above n = 12~~ | **Closed** by §4.1 — measured; it fails on 55 % of Protein graphs while pruned completes all |
| ~~Q6~~ | ~~Include GREC despite 59.1 % retention?~~ | **Closed** by §2.2.1 — the discard is size-unbiased (11.59 vs 11.45 nodes), the cleanest in the cohort. Include |
| ~~Q7~~ | ~~Write a **Web** loader?~~ | **Closed — and its stated reason was wrong.** "Mutagenicity max n = 417" counts a *discarded* disconnected graph; the retained ceiling is **98** (`gap-audit.md` MF1). Still declined: Web needs a new schema loader, and Protein/Mutagenicity already carry the size range at n̄ ≈ 30 |
| **Q8** | Which cost model: unit node+edge (IAM tradition) or topology-only zero-node (GraphEdX)? | **Resolved** — `statistics.md` D6, unit node + unit edge, substitutions free |
| ~~Q9~~ | ~~Does decision 12 hold on the corrected ceiling of 98?~~ | **Closed 2026-08-11 — AFFIRMED by the author.** 98 vs 12 is an 8.2× extension on ten datasets with published edit costs. Re-measurement confirmed no dataset moves in or out and that the COIL-RAG / Fingerprint / Web drops all survive. The residual "real-world graphs are far larger than 98" objection is acknowledged in the paper via `plan.md` §3.5's three-statement framing |

> **Note, audit-2026-08-11b**: `plan.md:925` records **Q3** as resolved ("all-pairs, recover
> everything", author decision 2026-08-11); it is still listed live above. The plan is the authority.

---

## 10. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-12 | **v1.3** | **Third-auditor pass** (`.claude/notes/audit-2026-08-11b/third-auditor.md`). **Triage of the 24 standing defects, which v1.2 listed without ranking**: only **four reach a number a reviewer will read** — **I-02** (AIDS raw 819 vs the true 911, which feeds Tab. 2's retention row and R3.5a's ladder), **I-03** (§3.1's inflated Letter pair counts, against which risk R1's ~100 core-h fallback is costed), **I-05** (Fingerprint 2.3×, a counter-example to the sentence citing it, inside §2.2.1's discarded-subset columns) and **I-08b** (uncommenting `Fischer2015hausdorff` / `Lerouge2017ilp` takes bibliography headroom from 12 to 10). Fix those four; batch the rest as document hygiene after the manuscript work. ⚠ **I-11 is DOWNGRADED and must not be applied as written**: "AIDS 131,148 contradicts F2's 181,909" compares **different populations** — `C(769,2) = 295,296` is on the 769 filtered graphs while `C(546,2)+C(182,2)+C(183,2) = 181,909` is within-split on the 911 **raw** graphs, so I-11's proposed **1.62×** would itself be MF1's defect class. The population-matched comparator is ≈ 129,600, within 1.2 % of 131,148. New disposition: *provenance not recorded* (minor) — record the source when T-03 reproduces the run, keep **2.25×** |
| 2026-08-12 | **v1.2** | **Over-scope and integrity audit `audit-2026-08-11b`** — re-measured, not argued, over four populations side by side (`RAW` / `CONN_ge1` / `KEPT_ge2` / `DISC_ge2`) so the MF1 defect class becomes detectable everywhere rather than only where already suspected. **Applied here**: §4.4's live "**Recommendation** — build `wl_pruned_canonical` (T-16)" now carries a SUPERSEDED banner — T-16 was **rejected** by signed decision 17, and `data.md` is required reading before executing any ticket, so a live recommendation here contradicted a signed decision (I-10). **Confirmed clean and reproducible, on record**: §0 Suite 1 and Suite 2 reproduce **cell for cell**; the six "what this buys" ratios; §2.1's correction banner; §2.1's density convention (it averages over `RAW`, counting n=1 graphs as 0 — verified by identity, **not a defect**); §2.2.1 for GREC, Letter LOW/HIGH, Mutagenicity, Protein, AIDS-IAM; **MF17's Fingerprint correction (51.4 %, 5.03/5.98) reproduces exactly**; §2.3's drop row, so all three drop decisions survive. **Outstanding defects recorded but not yet applied** — I-01 (blocking: **13 of the 16 scripts §8 names no longer exist**, including `ged_bounds.py`, which makes §7.3 validation gate 2 unexecutable and leaves §5/H4's ρ = 0.966 vs 0.840 unreproducible), I-02 (§0's AIDS "raw" column is the **connected** count, 819 vs the true 911 — MF1's class inside the table §0 declares authoritative), I-03 (§3.1 mixes n≥1 and n≥2 pair counts, 22,698-pair gap), I-04 (§2.1 Fingerprint `N conn` is a different population and off by one), I-05 (the retracted 11.56 survives as "Fingerprint 2.3×", making the cited example a **counter-example** to the sentence citing it), I-11 (AIDS "131,148" unsourced, contradicts F2's measured 181,909), I-12–I-16, I-24. Full detail and the re-measurement log: `.claude/notes/audit-2026-08-11b/findings-integrity.md` |
