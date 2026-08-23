# Data — cohort, filter and measured counts

**Owner**: T-01 · **Serves**: AE.1, AE.4b, R1.3a, R3.5a, E1, F1
**Status**: **LOCKED and VERIFIED. T-01 closed 2026-08-13** — every number below is re-derived from
tracked code (`cohort_audit.py`, `iam_gxl_loader.py`, 34 unit tests). See §7 RESULT.
**Rule**: §1's tables are the **only** source a printed number may be taken from.

> **Provenance, measured 2026-08-13.** Suite 1 reproduces `export_graphs.py`'s asserted counts
> **exactly** — 1,180 / 1,253 / 2,059 / 89 / 769 and 3,897,911 pairs — which is the check that the
> new code is right. Nine of the ten Suite-2 rows reproduce the previously recorded values to the
> last printed decimal, **including all three discard ratios**. **One row changed: COIL-DEL**, see
> §1.3. Reproduce with:
>
> ```bash
> ~/.conda/envs/isalgraph-cpp/bin/python -m benchmarks.real_data.eval_setup.cohort_audit
> ```
>
> ## ⚠ CORRECTED 2026-08-15 (T-05) — **that command can no longer re-derive two of the ten rows.**
> The cohort numbers are unaffected; the *reproduction* is.
>
> `cohort_audit.py:254` and `export_graphs.py:430` both resolve GraphEdX as
> `<source>/GED_PRECOMPUTED/<NAME>`. The real path is `<source>/GED_PRECOMPUTED/**datasets**/<NAME>`,
> and `<source>/GED_PRECOMPUTED/LINUX` does not exist — orchestrator-verified. Because IAM now lives
> under `APPROX_GED/datasets/IAM_Database/extracted` while GraphEdX lives under
> `GED_PRECOMPUTED/datasets`, **no single `--source` value makes either module resolve both roots.**
>
> **What this changes**: decision 22's tracked reproduction script — whose entire purpose is that
> "what it measures becomes the table" — **cannot re-derive the LINUX and AIDS-GraphEdX rows on the
> current tree without a path fix**. Neither file was patched by T-05: both are frozen T-01/T-03
> artifacts and patching them would re-open a closed ticket's certified output.
>
> **What survives**: every count in §1. T-05's own exporter takes **two roots** and reproduced all
> ten rows exactly (16,370 graphs, 21,710,892 pairs, exit 0) on 2026-08-15, so the numbers are
> confirmed — by a different program than the one this line names.
>
> ## ✅ DISCHARGED — by T-05 itself, not by T-06. Verified 2026-08-17; **strike this item, do not re-implement it.**
>
> `benchmarks/real_data/eval_setup/data_roots.py` already exists and both call sites go through it: a probing resolver with an environment override and its own tests. Verified live that **one `--source` resolves both trees**, so the “no single `--source` value” sentence above is **false of the current tree** and only the *history* it records is still accurate.
>
> The frozen T-01/T-03 artifacts were never patched — the resolver was added beside them, which is why the concern about re-opening a closed ticket’s certified output does not arise. **T-06 inherited a debt that had already been paid**, and nearly re-implemented it.

Related: [exact_ged](exact_ged.md) · [approx_ged](approx_ged.md) · [statistics](statistics.md) ·
[preregistration](preregistration.md) · [decisions](decisions.md) · [tickets](tickets.md)

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
| Letter LOW | 2,250 | 1,180 | 52.4 % | 4.07 | 4 | 7 | 3.07 | 0.543 | 695,610 |
| Letter MED | 2,250 | 1,253 | 55.7 % | 4.11 | 4 | 8 | 3.17 | 0.542 | 784,378 |
| Letter HIGH | 2,250 | 2,059 | 91.5 % | 4.58 | 5 | 9 | 4.56 | 0.607 | 2,118,711 |
| LINUX | 89 | 89 | 100 % | 8.71 | 9 | 10 | 8.35 | 0.255 | 3,916 |
| AIDS (GraphEdX) | 911 | **819** | 89.9 % | 11.03 | 11 | 20 | 11.23 | 0.212 | 334,971 |
| **GREC** *(new)* | 1,100 | **650** | 59.1 % | 11.45 | 11 | 24 | 12.43 | 0.243 | 210,925 |
| **AIDS (IAM)** *(new)* | 2,000 | **1,811** | 90.5 % | 14.02 | 11 | 85 | 14.50 | 0.202 | 1,638,955 |
| **COIL-DEL** *(new)* | **3,900** | **3,900** | 100 % | 21.54 | 20 | 77 | 54.24 | **0.328** | **7,603,050** |
| **Mutagenicity** *(new)* | 4,337 | **4,040** | 93.2 % | 28.53 | 27 | **98** | 29.55 | **0.094** | 8,158,780 |
| **Protein** *(new)* | 600 | **569** | 94.8 % | **31.68** | 30 | 96 | 61.44 | 0.163 | 161,596 |
| **Total** | | **16,370** | | | | **98** | | | **21,710,892** |

**Density is the mean of per-graph `2m/(n(n−1))`**, not `2m̄/(n̄(n̄−1))`. The two differ by 10–27 %
on this cohort — AIDS (IAM) is 0.202 against 0.159 — so the convention has to travel with the number.
Both are emitted by `cohort_audit.py`; the table has always reported the former.

### 1.3 COIL-DEL — the one row that changed, and why

**3,900, not 7,200.** The IAM `COIL-DEL/data` directory holds **7,200 `.gxl` files**, but the split
index files (`train.cxl`, `valid.cxl`, `test.cxl`) name **3,900** of them — 2,400 / 500 / 1,000. The
previously recorded row enumerated the *directory*; running `cohort_audit.py --enumeration directory`
reproduces the old figures exactly (19,670 graphs, 40,024,242 pairs), which settles the provenance.

Two measurements decided it, and the decision is **signed** ([decisions](decisions.md) 27):

- the 3,900 indexed graphs are **exactly class-balanced — 100 classes × 39 graphs**;
- the other **3,300 carry no class label at all**, because no split index lists them.

Decision 3 merges *splits*; the union of the splits is 3,900. Adding 3,300 unlabelled graphs is not
merging splits, and a cohort in which 46 % of the largest dataset has no class would contradict
"COIL-DEL, 100 classes" wherever the paper says it ([decisions](decisions.md) §3, [labels](labels.md)).

### What the extension buys

| | submitted | revision | factor |
|---|---:|---:|---:|
| Datasets | 5 | **10** | 2× |
| Graphs | 5,350 | **16,370** | **3.1×** |
| Pairs | 3,897,911 | **21,710,892** | **5.6×** |
| Largest graph | 12 nodes | **98 nodes** | **8.2×** |
| Largest mean size | n̄ = 10.56 | **n̄ = 31.68** | **3.0×** |
| Density span | 0.218–0.607 | **0.094–0.607** | 6.5× |

> ⚠ **CORRECTED 2026-08-13 (T-01).** The graph and pair factors were **3.7×** and **10.3×**; both
> carried COIL-DEL's directory enumeration. The **size** claims are untouched — `n_max = 98`,
> `n̄ = 31.68` and the density span are all set by Mutagenicity, Protein and Letter HIGH, none of
> which changed. **AE.1's evidence is unaffected; only the pair-count headline moves.**
> ~~Graphs 19,670 (3.7×) · Pairs 40.0 M (10.3×)~~

### 1.4 The discarded side — measured, per dataset and per reason

`data.md` promised this table in §3 and no script ever emitted it. Suite-2 filter (`min_nodes = 2`,
`require_connected`, no `n_max`), so every discard here is a **connectivity** discard except where noted.

| Dataset | discarded | disc. n̄ | disc. ñ | disc. n max | disc. m̄ | disc. density | **bias** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 1,070 | 5.35 | 5 | 8 | 3.20 | 0.277 | **1.32×** |
| Letter MED | 997 | 5.38 | 5 | 9 | 3.25 | 0.278 | **1.31×** |
| Letter HIGH | 191 | 5.62 | 6 | 9 | 3.90 | 0.290 | **1.23×** |
| LINUX | 0 | — | — | — | — | — | — |
| AIDS (GraphEdX) | 92 | 10.46 | 11 | 19 | 8.68 | 0.176 | **0.95×** |
| GREC | 450 | 11.59 | 9 | 24 | 11.22 | 0.202 | **1.01×** |
| AIDS (IAM) | 189 | 31.76 | 12 | 95 | 32.42 | 0.114 | **2.27×** |
| COIL-DEL | 0 | — | — | — | — | — | — |
| Mutagenicity | 297 | 54.70 | 42 | **417** | 47.35 | 0.059 | **1.92×** |
| Protein | 31 | 50.19 | 40 | 126 | 75.00 | 0.103 | **1.58×** |

`bias` = discarded `n̄` / retained `n̄`. Letter LOW/MED/HIGH also lose 1 / 4 / 8 graphs to
`min_nodes = 2`; Suite 1 additionally drops 51 AIDS (GraphEdX) graphs to `n_max = 12`, at `n̄ = 18.2`.

**All three previously quoted ratios reproduce exactly** (Mutagenicity 1.92×, AIDS-IAM 2.27×,
Protein 1.58×), as does GREC's size-neutral discard ([decisions](decisions.md) §5 called it "the
cleanest in the cohort" — measured 1.01×).

### 1.5 Label content — measured, for AE.4b and E6

Attribute names present in the source GXL and **discarded** by the topology-only loader:

| Dataset | node attributes | edge attributes | kind |
|---|---|---|---|
| Letter LOW/MED/HIGH | `x`, `y` | — | continuous |
| LINUX | **none** | **none** | **unlabelled** |
| AIDS (GraphEdX) | stripped upstream | — | (topology-only release) |
| GREC | `x`, `y`, `type` | `frequency`, `type0`, `angle0` | continuous + categorical |
| AIDS (IAM) | `symbol`, `chem`, `charge`, `x`, `y` | `valence` | categorical + continuous |
| COIL-DEL | `x`, `y` | `valence` | continuous |
| Mutagenicity | `chem` | `valence` | categorical |
| Protein | `type`, `aaLength`, `sequence` | `frequency`, `type0`, `distance0`, `type1`, `distance1` | categorical |

**This settles E6 by measurement**: `conclusion.tex:70` and `:81` claim labels are "present in all
five benchmark datasets", and **LINUX carries no node or edge attribute at all**. It also supplies
T-18's Tier-0 label column ([labels](labels.md) C3) without a separate measurement.

---

## 2. Why this cohort and not another

**The extension is the IAM Graph Database itself** (Riesen & Bunke, SSPR 2008) — we already use IAM
Letter, it is *the* pattern-recognition GED benchmark, and it ships published edit costs. Staying
inside one benchmark family is far easier to defend than mixing sources.

| Rejected | Reason | Status 2026-08-13 |
|---|---|---|
| **TUDataset** (MUTAG, IMDB-BINARY, PROTEINS) | the IAM family already spans n̄ = 4 → 32 and density 0.09 → 0.61; no gap left to fill | argument, not measurable here |
| **`cs.cornell.edu/~arb/data/`** (Benson) | hypergraphs and simplicial complexes — **IsalHG's** domain. A simple-graph paper drawing from it reads as a dataset grab | argument |
| COIL-RAG | ~~n̄ = 3.0, density 0.93~~ → **measured n̄ = 3.02, density 0.935**, `n_max = 6` — degenerate | **confirmed** |
| Fingerprint | ~~51.4 % retention, n̄ = 5.03~~ → **measured 52.4 % (1,467 / 2,799), n̄ = 4.96** — nothing the Letter sets do not already cover | **confirmed**, numbers corrected |
| Web | different XML schema, does not parse | **confirmed** — the directory holds `doc.*.xml` and **zero** `.gxl` files |

All three drop decisions are now **measured rather than asserted** (`results/cohort_audit/rejected.json`),
under the same CXL enumeration as the cohort. Web is confirmed by the loader raising, which is why it
has no spec in `iam_gxl_loader.IAM_DATASETS`.

---

## 3. Two disclosures we make first rather than be caught on

1. **The connectivity discard is size-biased across the cohort — not only on the datasets added for
   scale.** Mutagenicity keeps graphs averaging 28.5 nodes and discards ones averaging 54.7
   (**1.92×**); AIDS-IAM discards at **2.27×**; Protein at **1.58×**. So `n̄ = 31.7` is the *connected
   subsample's* mean. **Report retained and discarded `n̄` / `n_max` side by side** — §1.4 is that
   table.

   > ⚠ **EXTENDED 2026-08-13 (T-01) — the scope of this disclosure was too narrow.**
   > The three IAM Letter sets discard at **1.32× / 1.31× / 1.23×**, and Letter LOW keeps only
   > **52.4 %** of its graphs. Letter is **4,492 of Suite 1's 5,350 graphs (84 %)** and carries the
   > paper's strongest correlation (ρ ≈ 0.93). So the size-biased discard is a property of the
   > **submitted** cohort as much as of the extension, and a disclosure naming only Mutagenicity,
   > AIDS-IAM and Protein would understate it in the direction that flatters us.
   >
   > Two datasets run the other way and both are worth stating: **AIDS (GraphEdX) discards at 0.95×**
   > — the only reversed case in the cohort — and **GREC at 1.01×**, effectively unbiased.
   >
   > ~~The connectivity discard is size-biased on exactly the datasets added for scale.~~
2. **The retained ceiling is 98, not 417.** The 417-node Mutagenicity graph is **disconnected** and
   never enters the study. Quoting a raw-set maximum to justify a cohort defined on the connected
   subset is a category error. 98 vs 12 is still an 8.2× extension.
   **Confirmed 2026-08-13**: Mutagenicity's *discarded* set has `n_max = 417` and its retained set
   has `n_max = 98` (§1.4). The 417-node graph is in the discarded column, by measurement.

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
| Encode all **16,370** graphs, C++ engine | < 0.01 | seconds |
| Levenshtein, all pairs | 1–2 | ~2 min |
| WL kernel (not accelerated) | 2–4 | ~5 min |
| Bootstrap + Mantel ([statistics](statistics.md) D15) | **40–80** | ~1 h |
| **Suite 2, both GED bounds, 21.7 M pairs** | ~~≈ 0.57~~ → **≈ 2,140 realised** (T-05, done) | see note |
| **Suite 1, exact GED** | ~~≈ 1,000–1,650~~ → **≈ 2,081 measured** (T-03, done) | 16–26 h |

**All new compute is GED.** No pair subsampling is needed anywhere except Suite 1 —
see [exact_ged](exact_ged.md) §3.

> **Updated 2026-08-13 (T-01):** graph and pair counts follow §1's COIL-DEL correction (16,370 graphs,
> 21,710,892 Suite-2 pairs). Suite 1's figure is now T-03's **measured** ≈ 2,081 core-hours rather
> than the estimate. ~~19,670 graphs · 40 M pairs · ≈ 1.05 core-h~~

> ## ⚠ CORRECTED 2026-08-15 (T-05) — the Suite-2 bounds row was wrong by ~3,750×, and the row it
> sat next to explains why nobody caught it
>
> **≈ 0.57 core-h** came from "~100 µs/pair", a rate that predates T-27 and **was never measured**.
> The realised cost of the three full-cohort roles was **≈ 2,140 core-hours** — comparable to
> Suite 1's exact GED, not to "minutes". Two independent errors compounded:
>
> | | |
> |---|---|
> | **The rate was wrong.** | T-27 measured `BRANCH_FAST` at 285 µs/pair at n̄ = 29.51, not 100 µs, and its probe used **160 graphs with `25 ≤ n ≤ 35`** while Suite 2 reaches `n = 98`. Per-pair cost scales roughly as `max(n₁,n₂)³`. |
> | **The parallelisation was pathological.** | The process pool used by the first campaigns is **negative-scaling**: on identical work, 1 worker took 36 core-s, 4 → 212, 15 → 928, 32 → 5,260. Three campaigns hit a 12 h wallclock after 8 of 10 datasets. `--workers 1` is the measured optimum and the array is the only correct fan-out. |
>
> **What this changes**: no Suite-2 GED figure in this table may be quoted from a rate that was not
> measured on Suite-2-sized graphs. The ~2,140 core-h is realised, not projected.
>
> **What survives**: "All new compute is GED" — still true, and now more so. The Levenshtein, WL and
> bootstrap rows are untouched by this and remain estimates.
>
> ⚠ **The `seconds_matrix` in the published files is *in-worker solver time*, not job wall time**, so
> it under-reports job consumption and is **not comparable across datasets** (they ran at different
> worker counts). See [T-05 article notes](../tasks/T-05-article-notes.md).

---

## 5. Known defects in the measured record — fix these four, batch the rest

From `.claude/notes/audit-2026-08-11b/findings-integrity.md`, triaged in `third-auditor.md` §7.
**Only these reach a number a reviewer will read:**

| ID | Defect | Where it surfaces |
|---|---|---|
| **I-02** | AIDS raw is the connected count (819 vs 911) | dataset-table retention column; R3.5a ladder rung 1. **Corrected in §1 above** |
| **I-03** | §3.1's Letter pair counts use the n≥1 population, inflated by 22,698 | risk R1's ~100 core-h fallback is costed against it — re-cost from [exact_ged](exact_ged.md) §2 |
| **I-05** | "Fingerprint discards at 2.3×" is computed from a retracted mean; true 1.19× | **CLOSED 2026-08-13 — 1.19× reproduces exactly** (1,332 discarded at n̄ 5.92 against 1,467 kept at n̄ 4.96). Fingerprint is a **counter-example** to the sentence citing it. Use Mutagenicity 1.92× / AIDS-IAM 2.27× / Protein 1.58×, all three now measured in §1.4 |
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

> ## ⚠ CORRECTED 2026-08-13 (T-01) — four of the five "surviving" scripts do not exist
>
> Searched the working tree **and all of git history** (`git log --all --diff-filter=A`).
>
> | Script | Status |
> |---|---|
> | `export_graphs.py` | **exists** — `benchmarks/real_data/eval_setup/`, with `tests/unit/test_export_graphs.py` (22 KB). Already ported |
> | `audit_recheck.py` | **absent from the tree and from every commit that ever existed** |
> | `audit_dropped.py` | **absent** — this is the script that produced the discard ratios in §3 |
> | `final_counts.py` | **absent** — this is the script that produced §1 |
> | `gedlib_api.py` | **absent locally**; it was a Picasso-side smoke script, never in this repository |
>
> So the loss is **15 of 16, not 13**, and "T-01 ports what survives" reduces to a no-op — the one
> survivor was ported already. **T-01's real job is re-derivation, not porting.**
>
> **The exposure is larger than the script count.** `export_graphs.py` hardcodes **Suite 1 only** —
> five `DatasetSpec` rows with asserted counts, `FILTER_N_MAX`, `assert_cohort`. Grep finds **zero
> occurrences of GREC, Mutagenicity, COIL-DEL or Protein anywhere** in `benchmarks/`, `src/` or
> `tests/`, and there is no IAM-GXL loader. **The entire Suite-2 half of §1** — 19,670 graphs,
> 40,024,242 pairs, `n_max = 98`, the density span, and every discard ratio in §3 — **has no
> reproducing code.** That is the revision's headline extension claim (3.7× graphs, 10.3× pairs,
> 8.2× size).
>
> **Resolution — DONE 2026-08-13.** `benchmarks/real_data/eval_setup/cohort_audit.py` and
> `iam_gxl_loader.py`, with 34 unit tests, re-derive all ten datasets plus the two auditable rejected
> ones, retained **and** discarded, per discard reason. **Suite 1 reproduces `export_graphs.py`
> exactly; nine of ten Suite-2 rows reproduce their recorded values; COIL-DEL changed** (§1.3).
> Design: `.claude/notes/review/tasks/T-01-design.md`. Result: §7.

~~**13 of the 16 measurement scripts named in the v1.x record no longer exist.** Surviving:
`export_graphs.py`, `audit_recheck.py`, `audit_dropped.py`, `final_counts.py`, `gedlib_api.py`.
T-01 ports what survives into `tests/`; everything else must be re-derived if challenged.~~

**`ged_bounds.py` was rewritten 2026-08-12 and is no longer among the missing** — it now lives at
`benchmarks/real_data/eval_setup/ged_bounds.py` with `validate_ged_bounds.py` and 35 unit tests,
**tracked in the repository rather than in a scratchpad**, which is the failure mode that lost the
other twelve. Any measurement a locked decision rests on belongs in `benchmarks/` or `tests/`, never
in `scratchpad/`. Result: [exact_ged](exact_ged.md) §4.

> **The re-measurement showed the loss was not cosmetic.** Of the six figures the retired
> `ged_bounds.py` had produced, **none reproduced**, and all six were more flattering than the truth.
> Treat any surviving number whose script is gone as unverified until re-derived, not as
> presumptively correct.

**Machine**: local workstation, single thread, `time.process_time()`, `isalgraph.engine() == 'cpp'`.
**Env**: `~/.conda/envs/isalgraph-cpp`. **Data roots**: `…/data/source/GED_PRECOMPUTED/{AIDS,Letter,LINUX}`
and `…/data/source/IAM_Database/extracted/`.

---

## 7. RESULT — T-01, closed 2026-08-13

| Artifact | Path |
|---|---|
| IAM GXL/CXL loader, both enumeration policies | `benchmarks/real_data/eval_setup/iam_gxl_loader.py` |
| Cohort audit — both suites, retained and discarded, per reason | `benchmarks/real_data/eval_setup/cohort_audit.py` |
| Unit tests — **34 passing** | `tests/unit/test_iam_gxl_loader.py`, `tests/unit/test_cohort_audit.py` |
| Machine-readable output | `results/cohort_audit/{suite1,suite2,rejected}.json` + `cohort_table.md` |

**Acceptance criteria, all met.** Suite 1 reproduces the locked cohort exactly (1,180 / 1,253 / 2,059
/ 89 / 769; **3,897,911** pairs), which is the self-check on the new code. Suite 2 is **16,370 graphs
/ 21,710,892 pairs / `n_max = 98`**. Full suite **1,101 passed / 271 skipped** with the engine; ruff
clean; nothing in `scratchpad/`.

**What changed, and what did not.** One row of ten: COIL-DEL, 7,200 → **3,900** (§1.3). Nine rows
reproduce their recorded `n̄`, `ñ`, `n_max`, `m̄` and density to the last printed decimal, and all
three quoted discard ratios reproduce exactly. **This is a materially better outcome than T-25's**,
where none of six figures reproduced — the lost scripts were right about almost everything, and the
one thing they got wrong was a definitional choice rather than an arithmetic error.

**Four findings carried:**

1. **The size-biased discard is cohort-wide, not confined to the new datasets** — Letter discards at
   1.23–1.32× and is 84 % of Suite 1 (§3 disclosure 1).
2. **Label content measured per dataset**, and **LINUX carries no attributes at all** — E6 settled by
   measurement (§1.5), feeding T-18 Tier 0 at no extra cost.
3. **The density convention matters**: mean-of-densities vs density-of-means differ by up to 27 % on
   this cohort (§1).
4. **I-05 closed** — Fingerprint's 1.19× reproduces; the rejected-dataset table in §2 is now measured
   rather than asserted, Web included.

**Standing requests answered.** §3 disclosure 1's "report retained and discarded side by side" is now
§1.4. §3 disclosure 2's claim about the 417-node graph is confirmed. §2's drop reasons are measured.

**Debt carried, and it is not T-01's**: I-03's Letter pair-count defect is costed against
[exact_ged](exact_ged.md) §2; I-11 remains a do-not-apply; the ladder's later rungs
(`GED-available → GED > 0 → Lev > 0 → analysed`) need T-05 and T-06.
Article notes: `.claude/notes/review/tasks/T-01-article-notes.md`.
