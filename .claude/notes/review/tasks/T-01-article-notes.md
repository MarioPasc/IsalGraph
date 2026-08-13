# T-01 — article notes

**For**: T-20 (manuscript), T-14 (letter), T-18 (labels), T-12 (claim scoping), T-13 (complexity)
**Ordered by consequence.** Source: [data](../plan/data.md) §1–§3 and §7, `results/cohort_audit/`.

---

## A. Changes what the paper may claim

### A1 — The cohort is 16,370 graphs and 21,710,892 pairs, not 19,670 / 40,024,242
**Owner**: T-20 → §3.1 dataset table, abstract, §1 · **Lands**: every place a cohort size is printed

COIL-DEL contributes **3,900** graphs, the IAM benchmark's own definition (100 classes × 39, split
2,400 / 500 / 1,000), not the 7,200 `.gxl` files that ship in the same directory. The extension is
therefore **3.1× graphs** and **5.6× pairs** over the submitted study, not 3.7× and 10.3×.

**What is unchanged, and this is the part that matters for AE.1**: `n_max = 98`, largest mean size
`n̄ = 31.68` (Protein), density span **0.094–0.607**. Those are set by Mutagenicity, Protein and
Letter HIGH. **The graph-size argument the Area Editor asked for is untouched**; only the pair count
moves. Say the cohort size plainly and let the size evidence carry AE.1.

### A2 — LINUX carries no labels, measured
**Owner**: T-12 (E6), T-18 (C3) · **Lands**: `conclusion.tex:70`, `conclusion.tex:81`, §3.1 table

The submitted text claims node and edge labels are "present in all five benchmark datasets". Parsing
the sources shows **LINUX has no node attribute and no edge attribute**. E6 is now settled by
measurement rather than by inspection of prose. The full label-content column ([data](../plan/data.md)
§1.5) answers **AE.4b** directly and cost nothing beyond the audit that was already running.

### A3 — The size-biased discard is a property of the whole cohort, including the submitted one
**Owner**: T-20 → §3.1 and §5 limitations; T-12 → claim scoping · **Lands**: one paragraph + §1.4 table

The planned disclosure named only the datasets added for scale. Measured, the three IAM Letter sets
discard at **1.32× / 1.31× / 1.23×**, and Letter is **4,492 of Suite 1's 5,350 graphs (84 %)** — the
part of the study carrying the strongest reported correlation (ρ ≈ 0.93). Two datasets run the other
way: **AIDS (GraphEdX) at 0.95×** and **GREC at 1.01×**.

Disclosing this as a cohort-wide property is both more accurate and stronger: it converts "we added
big datasets and quietly lost the big graphs" into a measured, two-directional characterisation that
already applied to the submitted study.

---

## B. Reporting obligations

### B1 — Retained and discarded, side by side, per discard reason
**Owner**: T-20 → §3.1 or supplementary · **Lands**: [data](../plan/data.md) §1.4, printed as-is

Ten rows: discarded count, `n̄`, `ñ`, `n_max`, `m̄`, density, and the bias ratio. This is the first
rung of **R3.5a**'s pair-accounting ladder and the table §3 disclosure 1 has been promising.

### B2 — State the density convention
**Owner**: T-20 → §3.1 caption · **Lands**: one clause

Reported density is the **mean of per-graph `2m/(n(n−1))`**, not `2m̄/(n̄(n̄−1))`. On this cohort the
two differ by 10–27 % — AIDS (IAM) is 0.202 against 0.159. A reader who recomputes from `n̄` and `m̄`
in the same table will get the other number and conclude we made an error.

### B3 — The rejected datasets are measured, not asserted
**Owner**: T-20 → §3.1 or the response letter · **Lands**: [data](../plan/data.md) §2

COIL-RAG `n̄ = 3.02`, density **0.935**, `n_max = 6`. Fingerprint retains **52.4 %** at `n̄ = 4.96`.
Web ships `doc.*.xml` and **zero** `.gxl` files. Each rejection reason is now a measurement, which is
what makes the cohort choice defensible rather than a preference.

### B4 — The 417-node graph, confirmed in the discarded column
**Owner**: T-20, T-13 · **Lands**: §3.1, §4.2

Mutagenicity's discarded set has `n_max = 417`; its retained set has `n_max = 98`. Quoting 417 as a
cohort ceiling would be a category error, and the measurement now says so directly.

---

## C. Reproduction parameters

| Parameter | Value |
|---|---|
| Filter | `min_nodes = 2`, `require_connected = True`; `n_max = 12` (Suite 1), none (Suite 2) |
| Splits | merged (decision 3) |
| Enumeration | **split index (CXL union)** — decision 27 |
| Command | `python -m benchmarks.real_data.eval_setup.cohort_audit` |
| Code | `cohort_audit.py`, `iam_gxl_loader.py`; parsing delegated to `iam_letter_loader.parse_gxl` |
| Tests | `tests/unit/test_cohort_audit.py`, `tests/unit/test_iam_gxl_loader.py` — 34 passing |
| Output | `results/cohort_audit/{suite1,suite2,rejected}.json`, `cohort_table.md` |
| Data root | `…/data/source/IAM_Database/extracted/`, `…/data/source/GED_PRECOMPUTED/` |
| Env | `~/.conda/envs/isalgraph-cpp`, Python 3.11.15 |
| Self-check | Suite 1 reproduces `export_graphs.py` exactly: 3,897,911 pairs |

---

## D. What is NOT claimable

1. **Do not print 19,670 graphs, 40,024,242 pairs, 3.7× or 10.3×.** Superseded by decision 27.
2. **Do not print COIL-DEL `n_max = 79`, `n̄ = 21.48` or `m̄ = 54.03`.** Those belong to the 7,200-file
   enumeration. The measured values are **77 / 21.54 / 54.24**.
3. **Do not quote Fingerprint's discard ratio as evidence of size-biased discarding.** It is
   **1.19×** — a counter-example to the sentence that cited it, not support for it. Use Mutagenicity
   1.92× / AIDS-IAM 2.27× / Protein 1.58×.
4. **Do not say the connectivity discard affects only the datasets added for scale.** Letter discards
   at 1.23–1.32×, AIDS (GraphEdX) at 0.95×, GREC at 1.01×.
5. **Do not recompute density from the table's `n̄` and `m̄`.** Different convention; see B2.
6. **Do not claim the cohort's raw graph counts equal the files on disk.** They do for nine datasets
   and **not** for COIL-DEL (3,900 indexed of 7,200 present) or Fingerprint (2,799 of 4,000).
7. **The audit measures topology only.** §1.5 records which attributes were *discarded*; it makes no
   claim about label-aware distances. That is [labels](../plan/labels.md) Tier 2, and S-d is open.
8. **`n_max = 98` is a retained-set property under `require_connected`.** It is not the largest graph
   in IAM, and the paper must not imply the encoder was tested to the raw ceiling.
