# design — Section 4, Experimental design

File owned and written: `04_experimental_design.tex`. Nothing else was touched.

## What the section now says

§4 defines the instruments, one subsection per instrument, and every subsection states a decision,
gives its justifying experiment in a clause, and points at the supplementary section that carries it.

**4.1** introduces the cohort table and attributes the size ceiling to the reference: exact graph
edit distance costs 36.9 s per pair at n = 12 and grows about fivefold per node, so no public
benchmark supplies it further; the encoder has no comparable limit and Suite 2 reaches n = 98. The
density convention travels with the number. Two disclosures are made rather than caught on — the
connectivity discard is size-biased across the whole cohort including the submitted five, and the
retained ceiling is 98 and not 417 — and LINUX is stated to carry no attribute at all. The residual
objection, that real graphs are larger than 98 nodes, is conceded here and not only in §6.

**4.2** fixes one cost model against `\ref{cor:metric}`, splits the reference into an exact regime
below n = 12 computed by `networkx` A* (so the bound library is never compared against itself) and a
proven bracket above it, and then volunteers the adverse fact: the selected upper bound is the
loosest of the seven measured, by 6.7×, primary only because a frozen cost gate eliminates every
tighter method, degrading fastest in exactly the direction AE.1 asks about. Bracket width is reported
on both scales because they disagree, and the three-level bound guard is stated with the reason the
per-pair form of it is wrong.

**4.3** names the five comparators and the trivial baseline, states that each distance is chosen by
measurement against six criteria fixed in advance with correlation against the reference forbidden
from entering the choice, then states the rule's consequence — three representations excluded — and
concedes that the consequence favours us.

**4.4** reports both bit conventions with the entropy bound primary, and prints the reason they
differ. **4.5** describes the graph-level cluster bootstrap R3.5c asked for, demotes the pooled
analysis, states the frozen confirmatory family, and discloses the 473,147-pair gap *after* the
ladder that answers R3.5a. **4.6** says neither the compiled encoder nor GEDLIB existed at
submission, which is what licenses recomputing every reference distance.

## Number provenance — every number I wrote

| Number as printed | Source file | Where in it |
|---|---|---|
| 16,370 · 21,710,892 · 5,350 · 3,897,911 | `plan/data.md` | §1.1, §1.2 totals |
| n ≥ 2, connected; Suite 1 n ≤ 12 | `plan/data.md` | §1 filter |
| 36.9 s at n = 12; ≈ 5× per added node | `plan/data.md` | §4 table + closing line |
| n = 98; 8.2× | `plan/data.md` | §1.2, §3 item 2 |
| 10–27 % (density convention) | `plan/data.md` | §1.2 note |
| 417 (discarded) | `plan/data.md` | §1.4 Mutagenicity row |
| LINUX: no node or edge attribute | `plan/data.md` | §1.5 table |
| 3,836,827 of 3,897,911; 1.57 % censored | `plan/exact_ged.md` | §7 RESULT table |
| 60 s per-pair exact budget | `tasks/T-27-article-notes.md` | §8 |
| 12 methods × 5 datasets | `tasks/T-27-design.md` | grid, 60 `.npz` cells |
| 46,774,932 evaluations, 0 violations | `tasks/T-27-article-notes.md` | §5 row 3 |
| BRANCH = BRANCH_FAST on all 3,836,827 | `tasks/T-27-article-notes.md` | §1 |
| loosest of seven, by 6.7× | `tasks/T-27-article-notes.md` | §5 row 6 |
| cost gate < 1 ms/pair at n̄ = 30 | `tasks/T-27-article-notes.md` | §3 |
| +0.294 per node on AIDS; +0.029 to +0.055 | `tasks/T-27-article-notes.md` | §3, "error compounds with size" |
| 2 of 5; BP_BEAM_DET 0 of 5; misses gate by 17 % | `tasks/T-27-article-notes.md` | §3 |
| absolute 10 of 10; relative falls in 6 | `tasks/T-05-article-notes.md` | §1 slope-sign row |
| 949; 28.46 % Letter MED; 0.03 % Mutagenicity | `tasks/T-05-article-notes.md` | §4 |
| 306,768 pairs at exact distance 0 | `tasks/T-27-design.md` | §19 row 3, §40 total |
| library returns 0.00 on a wrong accessor | `tasks/T-27-article-notes.md` | §2 / project CLAUDE.md trap 2 |
| 1 of 50 relabelings | `tab_representation_properties.tex` caption; `02_related_work.tex` | already printed in §2.1 |
| 3 of 5; ρ = 0.83–0.93 | `tasks/T-04a-article-notes.md` | §4a |
| six criteria, F5 forbidden, import-closure test | `tasks/T-04a-article-notes.md` | §4 |
| 7.50 and 3.17 payload bits per stored byte | `plan/prose.md` | §4 (§4.4 brief), §10 CUT-1 — frozen clause, used verbatim |
| three even-handed rules agree; `inflated` flag | `tasks/t06_bit_convention.py` | module docstring, four arms |
| 2,000 replicates, percentile, seed 42 | `plan/statistics.md` | D2 |
| 89 graphs; 3,916 pairs | `plan/data.md` | §1.1 LINUX row |
| 197 tests; three families; BH q = 0.05 | `plan/statistics.md` | §9 |
| 473,147 pairs; 12.1 % | `source/verified-discrepancies.md` | E2 |
| 43.0 % LINUX; 44.4 % AIDS within-split | `tasks/T-03-article-notes.md` | §4 |
| 91.5–93.6 % vs 0.0000 % | `tasks/T-27-article-notes.md` | §2 |
| 300 s per-graph encode budget | `plan/statistics.md` | D14 |
| `-march=x86-64-v3` | project `CLAUDE.md` | C++ engine build flags |
| 14 regression fits, byte-identical | `plan/prose.md` | §4 (§4.6 brief) |

No number was inferred from another number, taken from memory, or read from a peer's section file.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| AE.1 | `sec:datasets` | *"The size ceiling belongs to the reference, not to the encoding."* + the 36.9 s / n = 98 / 8.2× sentence |
| AE.1 | `sec:reference-ged` | *"The upper bound is the loosest of the seven we measured, by 6.7×, and we state that in the main text."* and the +0.294-per-node clause |
| AE.1 | `sec:reference-ged` | *"the absolute gap widens with n in 10 of the 10 datasets while the relative width falls in 6 … so we report the absolute gap first"* |
| AE.4a | `sec:representations` | *"Each representation's distance is selected by measurement rather than assigned by inspection, against six criteria fixed in advance"* |
| AE.4b | `sec:datasets` | *"Labels are counted and discarded, since this representation and the reference distance are computed on topology alone"* |
| AE.4c | `sec:stats` | *"A confirmatory family of at most 197 tests was enumerated and frozen before any p-value existed"* |
| R1.1 (partial) | `sec:representations` | the five comparators named and grouped by the families of `\ref{sec:comparison}` |
| R1.3a | `sec:datasets` | *"Density is the mean of the per-graph 2m/(n(n−1)) and not the ratio of means, which differ by 10–27 % here"* |
| R3.5a | `sec:datasets`, `sec:stats` | *"the connectivity discard is size-biased across the whole cohort, the five submitted datasets included"*; *"A pair-accounting ladder runs per dataset from raw graphs through connected, reference-available, positive-distance and positive-Levenshtein to analysed pairs, with each rung's loss attributed"* |
| R3.5b | `sec:reference-ged`, `sec:stats` | *"One cost model applies to every dataset…"*; *"Per-dataset correlations are primary and a pooled figure is secondary"* |
| R3.5c | `sec:stats` | *"Uncertainty comes from a graph-level cluster bootstrap… LINUX contributes 89 graphs and not 3,916"* |
| R3.6a | `sec:bits` | the two conventions, entropy bound primary, and the frozen 7.50 / 3.17 clause |
| E1 | `sec:datasets` | *"the submitted paper reported neither density nor node count"* |
| E2 / F2 | `sec:stats` | *"In building it we also found that the submitted study's own two pair totals differ by 473,147 pairs…"* — third sentence of the ladder paragraph, never the opening move |
| E6 | `sec:datasets` | *"LINUX carries none at all, which corrects a statement the submitted conclusion makes twice"* |

Every row of `prose.md` §6 that names §4.2–§4.6 leaves a pointer on the page: S1 (cost model, exact
census, the guard), S2 (bake-off, `BRANCH_FAST` equivalence, `BIPARTITE` disclosure, `BP_BEAM_DET`
arm, pinned options), S3 (bracket at scale), S4 (admissibility), S5 (bit conventions), S6 (bootstrap,
per-dataset primary, confirmatory family, ladder), S10 (reproduction).

## Measured

- **Section length: 4.81 p — prose 4.176 p measured, plus `tab_datasets` at the orchestrator's
  0.632 p. Target 4.05 p. I am +0.76 p over and I report it rather than close it by dropping a
  demand.**
  Method: `scratchpad/measure.sh` builds preamble + this section alone under `[review,times,number]`
  with `\input{tab_datasets}` stripped, and probes `\thepage`/`\the\pagetotal` immediately after the
  `\section` line and at end of file. **The harness reproduces wave-related's independently reported
  2.738 p for §2 exactly**, which is what makes the figure comparable across the wave.
  `scratchpad/measure_subs.sh` gives the per-subsection split below.
- Per subsection, measured against the brief's own sub-allocations:

  | | allowance | measured | Δ |
  |---|---:|---:|---:|
  | 4.1 datasets | 0.568 (1.2 − table) | 0.721 | +0.15 |
  | 4.2 reference GED | 1.000 | 1.174 | +0.17 |
  | 4.3 representations | 0.900 | 0.776 | **−0.12** |
  | 4.4 bit accounting | 0.054 (0.5 − Tab. 5) | 0.406 | **+0.35** |
  | 4.5 statistical protocol | 0.700 | 0.681 | −0.02 |
  | 4.6 implementation | 0.200 | 0.417 | **+0.22** |
  | section head | — | 0.038 | +0.04 |
  | **total prose** | **3.42** | **4.214** | **+0.79** |

  (4.214 is measured from *before* the `\section` line; 4.176 from after it. Same measurement.)
- **Density is not a constant and I did not convert with one.** Measured on my own file: 362 words
  per page under my counter, against 324 on §2 under the same counter. The header's 249 w/p figure
  does not reproduce under any counting convention I could construct, and 850 words would have been
  roughly 2.3 p, not 3.42.
- **Compile: clean.** `latexmk -pdf -outdir=<scratch>/build/design main.tex`, exit 0. **0 undefined
  citations, 0 undefined references, 0 font warnings, 1 overfull hbox and it is under 5 pt.** The
  citation figure was taken after a `bibtex` run in my out directory, per the CONTRACT's build trap.
  Whole document 37 p at my last build, with §5–§7 still being written.
- **My section cites nothing.** No `\cite` command, therefore no `\cite{TODO-…}` anywhere. §4 is a
  design section; every external attribution it needs (nauty, graph6, gSpan, AGM, WL) is already made
  in §2, and repeating the keys here would have cost EiC.a1 slots for nothing.
- **Humanizer: run last, after the numbers were verified.** `stylometry.py` passes every band except
  **Hedges (LOW, 1.59 against a 3–20 target)**, and I did not correct it: raising it means adding
  hedges to calibrated claims, which the wave CONTRACT bans and which the humanizer's own rule 4
  bans. wave-intro reached the same conclusion independently. Six edits came out of the pass, all
  passive→active where *we* is the true agent (*"Five representations are compared"* → *"We compare
  five representations"*, and four more) plus one meta-phrase repair (*"and that belongs in the main
  text"* → *"and we state that in the main text"*). **Ledger re-checked by diff: 98 numeric tokens
  and 18 `\ref`/`\supp` commands byte-identical before and after.**

## Decisions and assumptions

1. **`\ref{sec:canon-lit}` is not on my allowed label list**, though §2 defines it. I used
   `\ref{sec:related}` for the 1-of-50 pointer instead. If the orchestrator wants the tighter
   pointer, one word changes.
2. **The table caption already carries most of §4.1's disclosure content** — the size-bias ratios,
   98-vs-417, the density convention, LINUX. I did not repeat the ratios in prose; the prose asserts
   each property once and the table supplies the numbers. That is why §4.1 is 0.721 p and not 1.1 p.
3. **I did not print the per-matrix guard's 0.99 threshold.** The three levels are named and `\supp{1}`
   carries the value. Cut for space.
4. **I named AIDS on the +0.294 slope**, which the brief does not. The slope is an AIDS measurement
   (T-27 §3, n = 4→12), not a cohort property, and T-27's own not-claimable list is explicit that a
   number of this kind must travel with its population.
5. **I did not write `prose.md` §4.2's "~10× faster in n than any alternative".** It holds against
   `IPFP_MS` (0.294/0.029 = 10.1) and `BRANCH_FAST` (8.2) but not against `BP_BEAM_MS`
   (0.294/0.055 = 5.3), so *"any alternative"* is false as written. I printed the three slopes
   instead, which is stronger and cannot be checked wrong.
6. **I did not claim `sparse6` is the most compact representation** in the exclusion paragraph, though
   `prose.md` §9 offers it. It is a compactness result, it belongs to §5, and borrowing it here would
   have been a number taken from a peer's territory.
7. **`n = 17` is not in this section.** T-05 §3 moved the measured exact-GED ceiling from 12 to 17
   under a 1,200 s budget, but Suite 1 is defined at n ≤ 12 under T-03's 60 s budget and the cohort
   table says so. Introducing 17 here would put two ceilings on one page with no room to explain the
   budget difference. It belongs to §5 or §6 with R3.7a's "with its cause".

## For the orchestrator

**Two places where a source contradicts the plan. The source wins in both, and both are recorded in
the file header.**

1. 🔴 **The §4.2 brief's zero-distance sentence is wrong, and it is wrong in a checkable way.** Both
   `prose.md` §4.2 and my file's own TODO block say *"28.05 % of IAM Letter LOW pairs are certified
   isomorphic"*. `T-05-article-notes.md` §4 measures **28.05 % as Letter LOW's certification rate —
   the fraction of pairs where LB == UB** — which is a completely different statistic from the
   fraction at distance zero. The project `CLAUDE.md` states both correctly and separately. I wrote
   the zero-distance claim from `T-27-design.md` instead: **306,768 Suite-1 pairs at exact GED 0**.
   The same conflation appears in `CLAUDE.md`'s corrected-2026-08-15 block, which reads *"28.05 % of
   IAM Letter LOW pairs are certified with LB == UB"* — correct there — so the error entered at
   `prose.md`. **Worth fixing in `prose.md` before the response letter quotes it**, because the
   letter will make the same claim to a reviewer who can check it.
2. **`prose.md` §6 and `T-05-article-notes.md` §1 disagree on one count.** §6 says the absolute and
   relative bracket measures *"disagree in sign on 6 of 10 datasets"*; T-05 §1's correction box says
   *"in 4 of 10 datasets the two measures carry opposite signs"*. T-05's own slope-sign table gives
   absolute positive in 10 of 10 and relative negative in 6 of 10, which implies **6** disagreements,
   so §6 looks right and T-05's box looks like the slip — but I did not print either count. The page
   states only the two measured tallies, which are identical in both sources.

**The overrun, and what buying it back costs.** I am +0.76 p on the section. Two thirds of it is
structural rather than verbosity:

- **§4.4 is +0.35 p against an allowance of 0.054 p.** When Tab. 5 moved to the supplement (CUT-1)
  the table's 0.446 p left the section budget with it, but §4.4's *prose* brief did not shrink: it
  still specifies both conventions, the reason they differ, the frozen 7.50/3.17 clause, the
  "you chose the convention that flatters you" objection and its answer, and the `bits.py` point.
  That content does not fit in 0.054 p. **Either §4.4 gets ~0.35 p of real allowance, or you tell me
  which two of its five sentences to drop.** My recommendation if you must cut: drop the
  `bits.py`-already-flags-min-DFS sentence (−0.09 p); it is the least checkable of the five.
- **§4.6 is +0.22 p against 0.2 p** for five named reproducibility items. Cutting the determinism
  check saves 0.08 p and cutting the pinned-options sentence saves 0.07 p, but the second is a
  `prose.md` §6 row with an S2 pointer and I would not drop it.
- The remaining +0.19 p is spread over §4.1 and §4.2 and is genuine content: §4.2 alone carries five
  decisions, and the `BIPARTITE` disclosure that AE.1 turns on is 0.26 p of it. **§4.3 came in 0.12 p
  under**, which is already netted into the totals above.

**One thing to check that is not mine.** The document is **37 pages** at my last build with §5, §6
and §7 still being written, against EiC.c's hard 35. That is your number to manage, but it moved
while I was working and you may want it earlier than the end of the wave.
