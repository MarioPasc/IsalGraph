# Data — cohort, filter and measured counts

**Owner**: T-01 · **Serves**: AE.1, AE.4b, R1.3a, R3.5a, E1, F1
**Status**: LOCKED (decision 12, re-affirmed 2026-08-11 on the corrected ceiling).
**Rule**: §1's tables are the **only** source a printed number may be taken from.

Related: [exact_ged](exact_ged.md) · [approx_ged](approx_ged.md) · [statistics](statistics.md) ·
[decisions](decisions.md) · [tickets](tickets.md)

---

## 1. The two suites

**Filter**, identical to the submitted pipeline
(`benchmarks/real_data/eval_setup/dataset_filter.py::filter_graphs`):
`min_nodes = 2`, `require_connected = True`, plus `n_max` where stated.
**Splits are merged** (decision 3) — GED is symmetric and carries no train/test semantics.

### Suite 1 — exact GED, `n ≤ 12`

| Dataset | raw | **kept** | keep % | n̄ | ñ | n max | m̄ | density | **pairs** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IAM Letter LOW | 2,250 | **1,180** | 52.4 % | 4.07 | 4 | 7 | 3.07 | 0.543 | 695,610 |
| IAM Letter MED | 2,250 | **1,253** | 55.7 % | 4.11 | 4 | 8 | 3.17 | 0.542 | 784,378 |
| IAM Letter HIGH | 2,250 | **2,059** | 91.5 % | 4.58 | 5 | 9 | 4.56 | 0.607 | 2,118,711 |
| LINUX | 89 | **89** | 100 % | 8.71 | 9 | 10 | 8.35 | 0.255 | 3,916 |
| AIDS (GraphEdX) | **911** | **769** | 84.4 % | 10.56 | 11 | 12 | 10.70 | 0.218 | 295,296 |
| **Total** | | **5,350** | | | | **12** | | | **3,897,911** |

Reproduces the submitted manuscript **exactly** — graph counts, 3,897,911 pairs, m̄ to two decimals.
Every pair count is `C(kept, 2)`.

> **AIDS raw = 911, not 819** (audit I-02). 819 is the *connected* count; `graphs.json` is already
> connectivity-filtered. Suite 1 retention is **84.4 %** (769/911), not 93.9 %. This feeds the
> dataset table's retention column and the first rung of R3.5a's pair-accounting ladder.

### Suite 2 — proven bracket, no `n_max`

| Dataset | raw | **kept** | keep % | n̄ | ñ | **n max** | m̄ | density | **pairs** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW / MED / HIGH | | 1,180 / 1,253 / 2,059 | | 4.07 / 4.11 / 4.58 | | 7 / 8 / 9 | | 0.543 / 0.542 / 0.607 | 695,610 / 784,378 / 2,118,711 |
| LINUX | 89 | 89 | 100 % | 8.71 | 9 | 10 | 8.35 | 0.255 | 3,916 |
| AIDS (GraphEdX) | 911 | **819** | 89.9 % | 11.03 | 11 | 20 | 11.23 | 0.212 | 334,971 |
| **GREC** *(new)* | 1,100 | **650** | 59.1 % | 11.45 | 11 | 24 | 12.43 | 0.244 | 210,925 |
| **AIDS (IAM)** *(new)* | 2,000 | **1,811** | 90.5 % | 14.02 | 11 | 85 | 14.50 | 0.202 | 1,638,955 |
| **COIL-DEL** *(new)* | 7,200 | **7,200** | 100 % | 21.48 | 20 | 79 | 54.03 | **0.328** | 25,916,400 |
| **Mutagenicity** *(new)* | 4,337 | **4,040** | 93.2 % | 28.53 | 27 | **98** | 29.55 | **0.094** | 8,158,780 |
| **Protein** *(new)* | 600 | **569** | 94.8 % | **31.68** | 30 | 96 | 61.44 | 0.163 | 161,596 |
| **Total** | | **19,670** | | | | **98** | | | **40,024,242** |

### What the extension buys

| | submitted | revision | factor |
|---|---:|---:|---:|
| Datasets | 5 | **10** | 2× |
| Graphs | 5,350 | **19,670** | **3.7×** |
| Pairs | 3.9 M | **40.0 M** | **10.3×** |
| Largest graph | 12 nodes | **98 nodes** | **8.2×** |
| Largest mean size | n̄ = 10.6 | **n̄ = 31.7** | **3.0×** |
| Density span | 0.22–0.61 | **0.094–0.607** | 6.5× |

---

## 2. Why this cohort and not another

**The extension is the IAM Graph Database itself** (Riesen & Bunke, SSPR 2008) — we already use IAM
Letter, it is *the* pattern-recognition GED benchmark, and it ships published edit costs. Staying
inside one benchmark family is far easier to defend than mixing sources.

| Rejected | Reason |
|---|---|
| **TUDataset** (MUTAG, IMDB-BINARY, PROTEINS) | the IAM family already spans n̄ = 4 → 32 and density 0.09 → 0.61; no gap left to fill |
| **`cs.cornell.edu/~arb/data/`** (Benson) | hypergraphs and simplicial complexes — **IsalHG's** domain. A simple-graph paper drawing from it reads as a dataset grab |
| COIL-RAG | n̄ = 3.0, density 0.93 — degenerate |
| Fingerprint | 51.4 % retention, n̄ = 5.03 — nothing the Letter sets do not already cover |
| Web | different XML schema, does not parse |

All three drop decisions were re-verified on connected-set numbers and survive.

---

## 3. Two disclosures we make first rather than be caught on

1. **The connectivity discard is size-biased on exactly the datasets added for scale.** Mutagenicity
   keeps graphs averaging 28.5 nodes and discards ones averaging 54.7 (**1.92×**); AIDS-IAM discards
   at **2.27×**; Protein at **1.58×**. So `n̄ = 31.7` is the *connected subsample's* mean.
   **Report retained and discarded `n̄` / `n_max` side by side.**
2. **The retained ceiling is 98, not 417.** The 417-node Mutagenicity graph is **disconnected** and
   never enters the study. Quoting a raw-set maximum to justify a cohort defined on the connected
   subset is a category error. 98 vs 12 is still an 8.2× extension.

Residual objection to acknowledge in the paper: real-world machine-learning graphs, which is what
AE.1 asks about, are routinely far larger than 98 nodes. The honest framing is the three-way split in
[approx_ged](approx_ged.md) §1 — the encoder has no 12-node ceiling, **exact GED does**, and that is
a constraint on the field rather than on this work.

---

## 4. Encoding cost — the bottleneck moved

Real benchmark graphs, `engine() == 'cpp'`, single thread, `process_time`:

| n | exact GED / pair (nx A*) | pruned-canonical encode / graph | ratio |
|---:|---:|---:|---:|
| 5 | 4.0 ms | 6 µs | 6.7 × 10² |
| 9 | 336 ms | 16 µs | 2.1 × 10⁴ |
| 11 | 7.48 s | 21 µs | 3.6 × 10⁵ |
| 12 | **36.9 s** | 27 µs | 1.4 × 10⁶ |
| 20 | *(intractable)* | **122 µs** | — |
| n̄ = 32 (Protein) | — | 3.9 ms | — |
| 96 (Protein) | — | 1.1 s | — |

Exact GED grows **≈ 5× per added node** near n = 12; encoding ≈ 1.15× per node.
"The canonical encoder is the bottleneck" was true under pure Python and **is no longer true**.

### Three measured findings that become paper results

- **Triplet pruning is what makes canonicalisation scale.** On Protein: pruned **3.4 ms, 0
  failures**; exhaustive **5.8 s, 22/40 timeouts** — a ~1,700× gap.
- **Cost is governed by `|Aut(G)|`, not size or density.** Protein `n = 96`, density 0.024 → **1.1 s**.
  Mutagenicity `n = 98`, density 0.021 → **does not finish in 5 minutes**. Same size, same density,
  opposite outcome; the failing graph has `|Aut| > 20,000`. This is the *characterised* worst case
  R3.7d asks for — see [corrections](corrections.md) and T-13.
- **1-WL is 2.4–2.6× finer than the incumbent pruning key and strictly subsumes it.** Mutagenicity/3703:
  28 triplet classes vs 66 WL colours. Reported as a *measurement* in the complexity section — **not**
  implemented (T-16 rejected, [decisions](decisions.md) 17).

### Compute totals

| | Core-hours | On 64 cores |
|---|---:|---|
| Encode all 19,670 graphs, C++ engine | < 0.01 | seconds |
| Levenshtein, all pairs | 1–2 | ~2 min |
| WL kernel (not accelerated) | 2–4 | ~5 min |
| Bootstrap + Mantel ([statistics](statistics.md) D15) | **40–80** | ~1 h |
| **Suite 2, both GED bounds, 40 M pairs** | **≈ 1.05** | minutes |
| **Suite 1, exact GED** | **≈ 1,000–1,650** | 16–26 h |

**All new compute is GED, and 98 % of that is AIDS.** No pair subsampling is needed anywhere except
Suite 1 — see [exact_ged](exact_ged.md) §3.

---

## 5. Known defects in the measured record — fix these four, batch the rest

From `.claude/notes/audit-2026-08-11b/findings-integrity.md`, triaged in `third-auditor.md` §7.
**Only these reach a number a reviewer will read:**

| ID | Defect | Where it surfaces |
|---|---|---|
| **I-02** | AIDS raw is the connected count (819 vs 911) | dataset-table retention column; R3.5a ladder rung 1. **Corrected in §1 above** |
| **I-03** | §3.1's Letter pair counts use the n≥1 population, inflated by 22,698 | risk R1's ~100 core-h fallback is costed against it — re-cost from [exact_ged](exact_ged.md) §2 |
| **I-05** | "Fingerprint discards at 2.3×" is computed from a retracted mean; true 1.19× | Fingerprint is a **counter-example** to the sentence citing it. Use Mutagenicity 1.92× / AIDS-IAM 2.27× / Protein 1.58× |
| **I-08b** | `Fischer2015hausdorff` and `Lerouge2017ilp` are cited only from commented-out LaTeX | uncommenting either takes bibliography headroom 12 → 10. See [compliance](compliance.md) |

> ⚠ **I-11 must NOT be applied.** It reports "AIDS 131,148 contradicts F2's 181,909" and offers
> **1.62×**. The two count **different populations**: `C(769,2) = 295,296` is on the 769 *filtered*
> graphs, `C(546,2)+C(182,2)+C(183,2) = 181,909` is within-split on the 911 *raw* graphs. The
> population-matched comparator is ≈ 129,600 — **within 1.2 % of 131,148**. Substituting 181,909
> would itself be the population-mixing defect. **Keep 2.25×**; record 131,148's provenance when T-03
> reproduces the run.

Everything else (I-04, I-06, I-07, I-09-consequence, I-10, I-12…I-25) is internal document hygiene:
one batched pass after the manuscript work, not before it.

---

## 6. Reproduction

**13 of the 16 measurement scripts named in the v1.x record no longer exist**, including
`ged_bounds.py`. Surviving: `export_graphs.py`, `audit_recheck.py`, `audit_dropped.py`,
`final_counts.py`, `gedlib_api.py`. T-01 ports what survives into `tests/`; everything else must be
re-derived if challenged. Consequences for the GED validation gates: [exact_ged](exact_ged.md) §4.

**Machine**: local workstation, single thread, `time.process_time()`, `isalgraph.engine() == 'cpp'`.
**Env**: `~/.conda/envs/isalgraph-cpp`. **Data roots**: `…/data/source/GED_PRECOMPUTED/{AIDS,Letter,LINUX}`
and `…/data/source/IAM_Database/extracted/`.
