# results — Section 5, Results

## What the section now says

§5.1 opens on the one unscoped positive: H1 is a complete invariant and the cohort
tests it by census, zero collisions over 24,764,422 GED-positive pairs, with the
Suite-2 `LB > 0` gap conceded in the same sentence that reports it.
§5.2 states compactness scoped — first among codes whose canonical form is intrinsic,
above n ≈ 20 — shows the advantage *growing* with size, and concedes in the same
subsection that canonically-labelled edge-list serialisations win at scale. The
exhaustive arm is declared with both its numbers, including the one where it does not
overtake.
§5.3 replaces H4 rather than defending it: the controlled ladder result with its flat
null arm, the statement that other canonical codes obey the same law, the concession
that the one family implementing automorphism detection escapes it and completes
94.7 % against our 55.3 %, the head-to-head with both halves, and R3.4c dissolved into
three exponents for three arms of one method. The 1-WL retraction is disclosed here,
after the reviewer answers, never before them.
§5.4 concedes first and without a bracket argument, then pivots: the benchmarks
themselves are size-dominated, which is why the within-`n` instrument exists and why
the field-level statement (nothing distinguishable from ρ = 0 above n ≈ 40) is the
contribution rather than an excuse.
§5.5 reports the pre-registered family unchanged, split by family and direction, and
says out loud that on the fidelity row it goes against us 11 to 7.

Every subsection opens by naming its hypothesis and closes with one sentence of
interpretation.

## Number provenance — every number I wrote

Path aliases: **`$T06`** = `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute`;
**`$T13`** = `.../results/reports/T-13-complexity`;
**`$PLAN`** = `/home/mpascual/research/code/IsalGraph/.claude/notes/review/plan`;
**`$TASKS`** = `/home/mpascual/research/code/IsalGraph/.claude/notes/review/tasks`.

### §5.1
| Number as printed | Source file | Where in it |
|---|---|---|
| 24,764,422 | `$T06/REPORT.md` | :190, collision census |
| 3,424,764 | `$T06/T-06-FRAMING.md` | :22 |
| 21,339,658 | `$T06/T-06-FRAMING.md` | :24 (sums to 24,764,422) |
| 101 | `$T06/T-06-FRAMING.md` | :25 (fallback graphs, all Mutagenicity) |

### §5.2
| Number as printed | Source file | Where in it |
|---|---|---|
| 112 of 112 | `$T06/T-06-FRAMING.md` | :41, :1129 |
| +215 bits | `$PLAN/prose.md` | :665 (C2). File value +214.8, `$T06/T-06-FRAMING.md:313` |
| 10 against / 9 for | `$TASKS/T-06-article-notes.md` | §10, A1 pooled row |
| 20.4 % → 45.6 % | `$T06/T-06-FRAMING.md` | :71 |
| −1.2 → +242.1 bits | `$T06/T-06-FRAMING.md` | :72 |
| 8–12 %, n = 13–20 | `$TASKS/T-06-POSITIONING.md` | :345 |
| 114.1 vs 144.0, 20.8 % | `$TASKS/T-06-FILES.md` | :232 |
| 342.4 vs 336.0; 6.3 of 12.7; 96.8 % | `$TASKS/T-06-article-notes.md` | :212 |

### §5.3
| Number as printed | Source file | Where in it |
|---|---|---|
| 12 cells, 11 positive, +0.892, p = 0.0064 | `$T13/REPORT.md` | :138 |
| 71 orders of magnitude | `$T13/T-13-FRAMING.md` | :20 |
| 1.0–1.1× null arm | `$T13/REPORT.md` | :143 |
| +0.189 vs +0.326 | `$T13/REPORT.md` | :28–29 |
| +0.686, p = 0.041 | `$T13/REPORT.md` | :140 |
| p = 0.18, 57.6 % | `$T13/REPORT.md` :141; `$T13/T-13-FRAMING.md` :173 | |
| ρ ≈ −0.61, 18 of 20 | `$T13/REPORT.md` | :174 (graph6 −0.601, sparse6 −0.619) |
| 94.7 % vs 55.3 % | `$T13/REPORT.md` | :175, :184 |
| 3.39×, 42 of 66 | `$T13/REPORT.md` | :189–190 |
| 83 of 132 vs 73 | `$T13/REPORT.md` | :190 |
| 0 % / 21.85 % / 100 %, 3,703 graphs, 300 s | `$TASKS/T-06-FRAMING.md` | :150 |
| ≈5× per node, ≈1.15× | `$PLAN/data.md` | :237 |
| α 2.04 [1.88,2.20] / 3.15 [2.18,4.01] / 17.43 [14.67,19.35]; 2.08, 3.24, 4.89 | `$T13/tables/tab_t13_scaling_exponent.tex` | rows 13, 11, 12, 14, 16, 15 — **read by mathematical form, verified against the completion column** |
| 1.0952; 99.939 %; 16,370; 41.869 % | `$PLAN/prose.md` :703–707; `$T13/T-13-FRAMING.md` :176–177 | |

### §5.4
| Number as printed | Source file | Where in it |
|---|---|---|
| 4 of 5; −0.4597 | `$T06/REPORT.md` | :59 |
| 17 of 25; 8 clear; 1 exact + 7 UB | `$T06/REPORT.md` | :48, :53–57 table |
| **Inline block 1** — all 15 cells | `$T06/data/rho_table.json` | `rows[]`, suite1 / exact / all_pairs / isalgraph_pruned |
| 7 of 10 inversions | `$T06/T-06-FRAMING.md` :515; `$T06/REPORT.md` :61 | |
| ρ = 0.71–0.997; 0.96 on 7 of 10; 0.9971; ≈0.92 | `$T06/T-06-FRAMING.md` | :465–474, :468–469 |
| p = 0.041, 0.041, 0.012 | `$T06/REPORT.md` | :80, :85, :90 |
| 19 of 19; 17 of 19; 25 attempted | `$T06/REPORT.md` | :182; :150–175 |
| VIF 18.1 / 16.2 | `$T06/data/collinearity.json` | `aids_iam`, `coil_del`, Δn entries |
| 4 excluded + 2 further; 2 M subsample | `$T06/REPORT.md` | :178–180 |
| **Inline block 2** — ρ columns | `$T06/data/rho_table.json` | suite1 / exact |
| **Inline block 2** — β columns | `$T06/REPORT.md` | :155–157 (D4 table) |
| **Inline block 2** — mean n, mean m; 49 % | `$PLAN/data.md` :58–60; `$T06/T-06-FRAMING.md` :676–677 | |
| 0.93 → 0.67 | `$T06/data/rho_table.json` | Letter LOW → HIGH |
| 21,710,892 | `$T06/PROVENANCE.md` | :32 |
| 8,158,780; +0.7700/+0.7395; +0.8806/+0.8636; +0.6449/+0.6095; +0.7253/+0.6528; 0.017–0.073 | `$T06/T-06-FRAMING.md` | :118–122 |

### §5.5
| Number as printed | Source file | Where in it |
|---|---|---|
| N_actual = 79; 79 with p-values | `$T06/T-06-FRAMING.md` :1084–1087; `$T06/PROVENANCE.md` :19 | |
| 75 at q = 0.05; 69 directional; 35 / 34 | `$T06/REPORT.md` | :212–223 |
| **Inline block 3** — 51/28/23, 1, 18/7/11, 5 | `$T06/REPORT.md` | :212–223 (dup. `T-06-FRAMING.md:1096–1102`) |

**Rows I could not fill, and what I did.** One: prose.md §5.4 move 1 says the 17
below-null records are *"every one of them significantly"*. No source states it —
`$T06/REPORT.md:48` and `$T06/T-06-FRAMING.md:1142` give the count and the percentage
only, and the LOSS/tie verdicts in the adjacent table are against the *best competitor*,
not against the size null. The adverb is dropped; the count is printed plain. Dropping
it makes our own negative result marginally weaker, which is why it is flagged rather
than quietly kept.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| R1.1 | §5.3 ¶1 | *"Encoding cost is a per-graph quantity and graph-edit-distance cost a per-pair one, and the two no longer share an axis…"* |
| R1.3a | §5.4 ¶8 + Letter block | *"Density does not account for it either, measured within AIDS across its own density strata."* |
| R1.3b | §5.4 ¶8, leading its paragraph | *"The degradation on AIDS does not come from discarded labels. Both sides of the correlation are topology-only…"* |
| R1.3c | §5.4 ¶7 | *"The representation stops paying its way between LOW and MED distortion…"* + pointer to `sec:limits` |
| R3.4c | §5.3 ¶6 | *"Three arms of one method on one cohort give three exponents, which is why a fitted exponent decides nothing about complexity…"* |
| R3.5b | §5.4 inline block 1 + ¶3 | five Suite-1 per-dataset rows against exact GED, primary; *"On these benchmarks the reference itself is size-dominated…"* |
| R3.6b | §5.2 ¶1, §5.4 throughout | *"Scoped, it holds; unscoped, it fails."* Every compactness and fidelity claim carries its scope in-sentence |
| R3.7d | §5.3 ¶6 | *"The complexity statement is the bound of Section~\ref{sec:cost-model} together with the \|Aut(G)\| characterisation."* |
| AE.1 | §5.4 entire | size null, within-`n` instrument, n ≈ 40 statement, exact/bracket split |
| AE.4a | §5.2, §5.3, §5.4 | six competitors named and reported per graph and per stratum |
| AE.4c | §5.5 entire | the pre-registered family, unchanged, split by direction |
| E3 / E4 | §5.3 ¶6 | exponents state their basis (*"over completed encodings on one constructed cohort"*) instead of a declared node range |
| E10 | §5.3, §5.4 ¶6 | WL kernel reported in both; Mantel tests named and pointed at `\supp{6}` |

## Measured

- **section length: 8.33 p** against a 6.10 p target — **+2.23 p**. Measured, not
  estimated: a `\pdfsavepos` instrument at the section bounds wrote page + ypos at
  shipout, length = Δpage + Δy/`\textheight` (550.27614 pt). START p23 y=9269941 sp,
  END p32 y=33374936 sp. The instrument is removed; the method is recorded in the file
  header so the next agent re-measures rather than estimating.
- prose 2,140 raw words. The header's 249 w/p constant does **not** apply to prose this
  number-dense: measured density is ≈450 raw words per prose page, so the 730-word
  allowance is ≈1,300 raw words in this register. I am over even on that reading.
- compile: **clean.** 0 undefined citations, 0 undefined references, 0 errors,
  0 font warnings, 0 overfull boxes over 5 pt. Bibtex was run in the outdir per the
  CONTRACT trap before any citation was judged.
- whole document: **37 pages** (baseline 27 before this section's prose and before
  wave-design's §4 edits).

## Decisions and assumptions

1. **The 17-of-25 split in prose.md C3 is inverted against its own source.** C3 and
   `01_introduction.tex:93–95` both read *"one record is undetermined; the remaining 7
   favour the string."* `$T06/REPORT.md:53–57` gives exact 4 below / 1 clears, lb 10/0,
   ub 3/7, and `:61` says *"All 10 lb records fall below their null and all 7 ub records
   clear it — on the same pairs. The verdict inverts across the bracket."* So the **1**
   is the genuine clear (IAM Letter LOW, against ground truth) and the **7** are the
   undetermined ones. Source wins: §5.4 reads *"Of the 8 that clear it, one clears
   against exact graph edit distance, and the other seven clear only under the upper
   bound of the bracket and fall below the lower bound on the same pairs."* Counts
   unchanged; attribution corrected. **§1 still carries the inverted version and I do
   not own that file.**
2. **MRM exclusions restated to match the file.** prose.md says *"two more"*;
   `$T06/REPORT.md:180` says **4 fits on 2 datasets** fail the bootstrap criterion, of
   which 2 (`coil_del`) were already excluded for collinearity. Net is 2, so §5.4 prints
   *"Of the 25 fits attempted, four on two datasets are excluded as unidentifiable …
   and a further two on Mutagenicity …"* — 25 − 4 − 2 = 19, which is the printed
   denominator.
3. **The censored-pair sensitivity is Mutagenicity's 8,158,780 pairs, not 21.7 M.**
   The two were adjacent in the brief and are different cohorts. §5.4 attributes each
   explicitly.
4. **Registry-key trap cleared by direct read**, not by trusting a key name.
   `tab_t13_scaling_exponent.tex` rows cross-checked against the completion column:
   plain `IsalGraph` = pruned (α 3.15, 73 pts, 55 %  → the 73 and the 55.3 % of the
   head-to-head), `†` = hybrid (4.71, 42 % → the 57.6 % censored), `*` = true exhaustive
   (17.43). The three exponents are named by mathematical form.
5. **Figure placement changed `[!t]` → `[!tb]`** on all three floats. Captions, widths
   and `\includegraphics` options untouched. This recovered the dedicated float page
   `rho_vs_size` had taken.
6. **No `changes` markup**, per instruction. Header carries the reviewer-checkable
   sentence list instead.
7. §5.3 points forward to `sec:conclusion` for the 1-WL headroom result, as prose.md
   §5.3 directs. That is a cross-section dependency I cannot verify from my own file.

## For the orchestrator

- **§5 overruns by 2.23 p and I did not absorb it, per instruction.** There is nothing
  left to move: the MRM detail already points at `\supp{6}`, the per-competitor and
  per-dataset grids already point at `\supp{8}`, the three inline blocks are marked
  never-cut, and the two figures cannot leave because §2 cites both by `\ref`. What
  remains is claims and their scope clauses. The overrun is the eight moves of §5.4 and
  the thirteen of §5.3 meeting a budget written before either count was known.
- **The document is at 37 pages against EiC.c's hard ≤ 35.** Baseline was 27. This needs
  arbitration above my section.
- **Two plan defects, both listed above:** C3's inverted 1-vs-7 attribution, which also
  ships in `01_introduction.tex:93–95`; and prose.md's *"two more"* MRM exclusions
  against the file's *"4 fits on 2 datasets"*.
- **One unsourced qualifier removed:** *"every one of them significantly"* on the 17
  below-null records. If a source exists I will put it back.
