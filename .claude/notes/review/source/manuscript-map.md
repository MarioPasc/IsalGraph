# Manuscript map and numbering cross-walk

Everything a reviewer reference points to, resolved to a file, a line and a page.

Manuscript root:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`
All paths below are relative to it. Page numbers are from the submitted `main.pdf` (35 pages),
extracted with `pdftotext main.pdf`.

**The map is confirmed against the reviewers' own citations.** R1 cites "Figure 2, Section 4.2.";
R3 cites Sections 2.2.1, 3.2.3, 3.2.5, 4.2, 2.3, Theorem 2.12, Table 1, Figure 3, Algorithm 2
lines 24–30, and references [28]/[29]. **Every one resolves correctly under this map**, which is
the evidence that the numbering below matches the PDF the reviewers actually read.

## On-disk layout

```
69b82c5859ed47c5468ca199/
├── main.tex                        frontmatter, abstract, keywords, \input order, back matter
├── introduction.tex                Section 1
├── methodology.tex                 Section 2   (860 lines — the largest file)
├── computational_experiments.tex   Section 3   (\section declared in main.tex:162)
├── results.tex                     Section 4   (\section declared in main.tex:166)
├── conclusion.tex                  Section 5
├── cas-refs.bib                    56 entries defined, 43 cited
├── main.pdf                        the submitted PDF, 35 pages
├── elsarticle.cls, elsarticle-num.bst, cas-*.{cls,sty}   Elsevier support files
├── fig_*.pdf                       6 figures present; 4 used, 2 orphaned (see below)
├── graphical_abtract.pdf           submitted separately via Editorial Manager (sic, misspelt)
└── EzequielLopez.pdf, MarioPascual.jpg   author photos, currently commented out
```

`main.tex:158–170` fixes the input order: `introduction`, `methodology`, then a `\section` +
`\input` pair for computational experiments and for results, then `conclusion`.
Note the asymmetry: `introduction.tex`, `methodology.tex` and `conclusion.tex` declare their own
`\section`; `computational_experiments.tex` and `results.tex` do not — theirs live in `main.tex`.

## Sections

| # | Title | Label | File:line | Page |
|---|-------|-------|-----------|------|
| 1 | Introduction | `sec:introduction` | `introduction.tex:1` | 2 |
| 2 | Methodology | `sec:method` | `methodology.tex:15` | 5 |
| 2.1 | Instruction Set and String Execution | `sec:instructions` | `methodology.tex:28` | 5 |
| 2.1.1 | Interpreter State | — | `methodology.tex:32` | 5 |
| 2.1.2 | The Instruction Alphabet | — | `methodology.tex:70` | 6 |
| 2.1.3 | The StringToGraph Algorithm | `sec:s2g` | `methodology.tex:131` | 7 |
| **2.2** | **Graph-to-String Conversion** | `sec:g2s` | `methodology.tex:212` | 7 |
| **2.2.1** | **Pair Generation and Cost Ordering** | — | `methodology.tex:223` | 8 | ← **R3.4b** |
| 2.2.2 | Algorithm Description | — | `methodology.tex:244` | 8 |
| **2.3** | **Canonicalization** | `sec:canonicalization` | `methodology.tex:421` | 9 | ← **R3.7c** |
| 2.3.1 | Canonical Strings | — | `methodology.tex:434` | 9 |
| 2.3.2 | Canonicalization via Structural-Triplet Pruning | — | `methodology.tex:472` | 10 |
| 2.3.3 | Formal Proof of Graph Invariance | — | `methodology.tex:623` | 12 |
| 2.4 | Topological Structure | `sec:topology` | `methodology.tex:757` | 14 |
| 2.4.1 | The String Distance | — | `methodology.tex:768` | 14 |
| 2.4.2 | Graph Edit Distance | — | `methodology.tex:791` | 14 |
| 2.4.3 | The Locality Property | — | `methodology.tex:812` | 15 |
| 3 | Computational experiments | `sec:Computational-experiments` | `main.tex:162` | 15 |
| 3.1 | Benchmark Datasets | `sec:datasets` | `computational_experiments.tex:14` | 16 |
| 3.1.1 | Real-World Graph Collections | — | `computational_experiments.tex:25` | 16 |
| 3.1.2 | Synthetic Graph Families | — | `computational_experiments.tex:59` | 16 |
| 3.2 | Evaluation Protocol | `sec:protocol` | `computational_experiments.tex:90` | 17 |
| 3.2.1 | Encoding Methods Under Comparison | `sec:enc-methods` | `computational_experiments.tex:103` | 17 |
| 3.2.2 | Distance Computation | — | `computational_experiments.tex:129` | 18 |
| **3.2.3** | **Message Length Analysis** | `sec:info-content` | `computational_experiments.tex:141` | 18 | ← **R3.6a** |
| 3.2.4 | Complexity and Speedup Measurement | `sec:complexity` | `computational_experiments.tex:191` | 19 |
| **3.2.5** | **Correlation Analysis** | `sec:corr-analysis` | `computational_experiments.tex:200` | 19 | ← **R3.5a** |
| 3.3 | Implementation | `sec:implementation` | `computational_experiments.tex:234` | 19 |
| 4 | Results | `sec:results` | `main.tex:166` | 20 |
| 4.1 | Message Length | `sec:res-info-content` | `results.tex:4` | 20 |
| **4.2** | **Empirical Time Complexity** | `sec:res-complexity` | `results.tex:69` | 21 | ← **R1.1**, **R3.4c** |
| **4.3** | **Correlation with Graph Edit Distance** | `sec:res-correlation` | `results.tex:128` | 22 | ← **R3.5c** ("Section 4.3") |
| 4.4 | Speed–Quality Trade-off | `sec:res-tradeoff` | `results.tex:209` | 24 |
| 5 | Conclusion | `sec:Conclusion` | `conclusion.tex:1` | 25 |

There is **no Related Work section**. All positioning is in `introduction.tex:11–33`. This is the
structural fact behind AE.2, R1.2 and R3.1.

## Numbered environments

`main.tex:35–47` declares `theorem` numbered `[section]`, with `lemma`, `proposition`, `corollary`,
`definition`, `example`, `conjecture` and `remark` all sharing that counter. Hence the single
continuous 2.1 ... 2.15 run. **Every numbered environment in the manuscript is in Section 2** —
Sections 1, 3, 4 and 5 contain none.

| # | Kind | Label | Line | Page |
|---|------|-------|------|------|
| 2.1 | Definition — Interpreter state | `def:state` | `methodology.tex:37` | 5 |
| 2.2 | Remark — `C`/`c` identical for undirected graphs | *(unlabeled)* | `methodology.tex:185` | 7 |
| 2.3 | Example — Decoding `VvNV` | `ex:decode` | `methodology.tex:192` | 7 |
| **2.4** | **Definition — Sorted displacement pairs `P(M)`** | `def:pairs` | `methodology.tex:233` | 8 | ← **R3.4b** |
| 2.5 | Remark — Reachability precondition | `rem:reachability` | `methodology.tex:352` | 9 | ← **R3.3a** |
| 2.6 | Remark — String length decomposition | `rem:length` | `methodology.tex:361` | 9 |
| 2.7 | Definition — Exhaustive canonical string `w^+_G` | `def:exhaustive-canonical` | `methodology.tex:436` | 10 |
| **2.8** | **Remark — What is and is not searched over** | `rem:search-space` | `methodology.tex:462` | 10 | ← **R3.7c** |
| 2.9 | Definition — Structural triplet `tau(v)` | `def:triplet` | `methodology.tex:487` | 11 |
| 2.10 | Definition — Pruned canonical string `w*_G` | `def:pruned-canonical` | `methodology.tex:525` | 11 |
| 2.11 | Remark — Pruned vs. exhaustive string length | `rem:pruned-length` | `methodology.tex:614` | 12 |
| **2.12** | **Theorem — Pruned canonical string is a complete graph invariant** | `thm:invariant` | `methodology.tex:628` | 12 | ← **R3.3b** |
| 2.13 | Corollary — Graph distance is an isomorphism-invariant metric | `cor:metric` | `methodology.tex:728` | 13 |
| 2.14 | Definition — Levenshtein distance on IsalGraph strings | `def:levenshtein` | `methodology.tex:770` | 14 |
| 2.15 | Definition — Graph Edit Distance | `def:ged` | `methodology.tex:793` | 14 |

**Theorem 2.12 is the only theorem in the manuscript.** Its proof runs `methodology.tex:639–726`
in three steps (triplet invariance, execution-path bijection, string-set equality). The sentence
R3.3b attributes to the theorem statement is in the proof, at `methodology.tex:643–644`.

## Algorithms

| # | Title | Label | Source line | **PDF page** |
|---|-------|-------|-------------|--------------|
| 1 | `S2G(w, directed)`: StringToGraph | `alg:s2g` | `methodology.tex:137–183` | **33** |
| **2** | **`G2S(G, v_0)`: GraphToString (greedy)** | `alg:g2s` | `methodology.tex:273–344` | **34** ← **R3.4a** |
| 3 | `PCAN(G)`: Pruned Canonical String | `alg:pcan` | `methodology.tex:547–612` | **35** |

**All three algorithms float to the last three pages of the document, after the references.**
This is caused by `\def\floatpagefraction{1}` and `\def\textfraction{.001}` at `main.tex:66–67`
combined with `[ht]` placement: LaTeX defers each algorithm to a float page and they land at the
end. Algorithm 2 is discussed on pages 7–9 and typeset on page 34 — a 26-page separation.
R3 nonetheless found the `C`/`c` defect in it. Worth knowing before anyone tries to reproduce the
reviewers' reading experience.

### R3.4a's "lines 24 to 30" — resolved

`algorithmic[1]` numbers each `\State`, `\If`, `\ElsIf`, `\For`, `\While`, `\Return` and
`\LineComment`. Counting from `\LineComment{Initialise state}` at `methodology.tex:280`:

| Alg. line | Source line | Content |
|---|---|---|
| 24 | `methodology.tex:321` | `\ElsIf` opening the **`C`** branch — the input-edge guard |
| 25 | `methodology.tex:324` | `add_edge(val(l~_1), val(l~_2))` |
| 26 | `methodology.tex:326` | append moves + `C` |
| 27 | `methodology.tex:328` | update pointers, `break` |
| 28 | `methodology.tex:330` | `\ElsIf` opening the **`c`** branch — the input-edge guard |
| 29 | `methodology.tex:333` | `add_edge(val(l~_2), val(l~_1))` |
| 30 | `methodology.tex:335` | append moves + `c` |

R3's range covers exactly the two branches, with no slack. Diff against Table 1 in
`verified-discrepancies.md` D5.

## Tables

| # | Object | Label | Line | Page | Generated by |
|---|--------|-------|------|------|--------------|
| **1** | **The IsalGraph instruction set** | `tab:instructions` | `methodology.tex:79–115` | 7 | **hand-written LaTeX** (`experiments/README.md:81`) |
| 2 | Dataset properties, OLS slope `beta`, median compression ratio | `tab:information-content` | `results.tex:22–50` | 20–21 | `eval_visualizations/fig_message_length.py::generate_information_content_table` |
| 3 | Spearman `rho` between GED and Levenshtein distance | `tab:performance-summary` | `results.tex:138–157` | 22–23 | `eval_visualizations/table_performance_summary.py::generate_performance_table` |

R3.4a's "Table 1" is `tab:instructions`. Its `C`/`c` rows are `methodology.tex:102–107`.

Tables 2 and 3 were **hand-edited after generation** — `experiments/README.md:101–104` records that
the printed captions and column sets differ from what the scripts emit, so the LaTeX is not
byte-reproducible from the pipeline. Values are traceable; the markup is not.

## Figures

All four figures in the submitted PDF are in `results.tex`. Both `methodology.tex` figures are
commented out, so figure numbering runs entirely through Section 4.

| # | Object | Label | Line | Page | PDF file |
|---|--------|-------|------|------|----------|
| 1 | Message length comparison | `fig:message_length_scatter` | `results.tex:13–20` | 20 | `fig_message_length_scatter_log.pdf` |
| **2** | **Empirical time complexity + compression ratio** | `fig:empirical-complexity` | `results.tex:92–99` | 21–22 | `fig_complexity_ratio_combined.pdf` ← **R1.1** |
| **3** | **Aggregated GED/Levenshtein joint distribution** | `fig:heatmap-correlation-ged-lev` | `results.tex:179–185` | 23 | `fig_aggregated_density_correlation.pdf` ← **R3.5b** |
| 4 | Computational–quality trade-off | `fig:g2s-method-comparison` | `results.tex:221–228` | 24 | `fig_composite_method_tradeoff_v2.pdf` |

**Commented out of the submitted PDF** (each marked in the source as cut for the page limit):

| Object | Label | Where | PDF present in the directory? |
|---|---|---|---|
| Algorithm overview (S2G/G2S side-by-side trace) | `fig:algorithm_overview` | `methodology.tex:378–420` | no `fig_algorithm_overview.pdf` on disk |
| Shortest-path comparison | `fig:shortest_path_comparison` | `methodology.tex:835–860` | **yes** — `fig_shortest_path_comparison.pdf`, orphaned |
| Neighbourhood topology | `fig:neighborhood-topology` | `results.tex:280–288` | **yes** — `fig_neighborhood_topology.pdf`, orphaned |

Two orphaned figure PDFs sit in the manuscript directory unreferenced. Relevant to R3.7c, which
asks for a *new* schematic while two existing figures are already cut for space.

## Equations

| Tag | Label | Line | Content |
|---|---|---|---|
| — | `eq:isal-bits` | `computational_experiments.tex:157–160` | `B_Isal(w) = L log_2 9` |
| — | `eq:ged-bits` | `computational_experiments.tex:172–176` | `B_GED(G) = (N-1+M) + 2M ceil(log_2 N)` ← **R3.6a** |
| — | `eq:compression-ratio` | `computational_experiments.tex:182–185` | `r(G) = B_GED / B_Isal` |
| — | `eq:canonical` | `methodology.tex:444–455` | exhaustive canonical string `w^+_G` |
| — | `eq:pruning` | `methodology.tex:509–518` | pruned candidate set `C_pruned` |
| — | `eq:pruned-canonical` | `methodology.tex:532–543` | pruned canonical string `w*_G` |
| — | `eq:triplet-equiv` | `methodology.tex:664–668` | `tau_G(v) = tau_H(phi(v))` |

## Reference-number cross-walk

`elsarticle-num` numbers by order of first citation. **43 numbered references.** The two R3 cites:

| Ref | Key | Work | Cited at |
|---|---|---|---|
| **[28]** | `lopezrubio2025isalgraph` | López-Rubio (2025), arXiv:2512.10429v2 — the earlier preprint | `introduction.tex:52` |
| **[29]** | `ThurnhoferHemsi:2025` | IsalChem | `introduction.tex:53` |

Both are cited exactly once, in adjacent sentences, and nowhere else in the manuscript. [28] is the
only genuinely arXiv-only reference in the bibliography (see `README.md`).

Reproduce the numbering with:

```bash
python3 - <<'EOF'
import re
order, seen = [], set()
for f in ['introduction.tex','methodology.tex','computational_experiments.tex',
          'results.tex','conclusion.tex','main.tex']:
    for line in open(f):
        line = re.sub(r'(?<!\\)%.*', '', line)
        for m in re.finditer(r'\\cite\{([^}]*)\}', line):
            for k in (x.strip() for x in m.group(1).split(',')):
                if k and k not in seen:
                    seen.add(k); order.append(k)
for i, k in enumerate(order, 1): print(i, k)
EOF
```

## Where each reviewer comment lands

| ID | Primary location | Page |
|---|---|---|
| AE.1 | `computational_experiments.tex:47`, `:53`; `results.tex:251`; `conclusion.tex:68` | 16, 24, 25 |
| AE.2 | `introduction.tex:11–33` (no related-work section exists) | 2–3 |
| AE.3 / R3.1 | `introduction.tex:33`, `:52–53`; `conclusion.tex:74` | 3, 26 |
| AE.4 | `results.tex:69–126`; `computational_experiments.tex:141–189` | 18, 21 |
| EiC.a | `cas-refs.bib`; `methodology.tex:803` | 27+ |
| EiC.c | whole document — 35 of 35 pages | — |
| R1.1 | `results.tex:69–126` (Fig. 2 / Sec. 4.2) | 21–22 |
| R1.2 | `introduction.tex:11–33` | 2–3 |
| R1.3 | `results.tex:192–206`; `conclusion.tex:30–36`, `:69–71` | 23, 25–26 |
| R3.1 | `introduction.tex:33`, `:52–53`; `conclusion.tex:74` | 3, 26 |
| R3.2 | `main.tex:122–126`; `conclusion.tex:76`, `:88–95`; absent from `computational_experiments.tex:3–11` | 1, 15, 26 |
| R3.3 | `main.tex:106–108`, `:114`; `methodology.tex:352–359`, `:628–644`; `conclusion.tex:74` | 1, 9, 12, 26 |
| R3.4a | `methodology.tex:102–107` (Table 1) vs `:321–336` (Alg. 2) | 7 vs **34** |
| R3.4b | `methodology.tex:223–242` | 8 |
| R3.4c | `results.tex:86–90` vs `conclusion.tex:50`, `:68`, `:80` | 21 vs 25–26 |
| R3.5a | `computational_experiments.tex:200–209`; `results.tex:36–37`, `:182–187` | 19, 20, 23 |
| R3.5b | `computational_experiments.tex:41–56`; `results.tex:179–190` | 16, 23 |
| R3.5c | `computational_experiments.tex:208–209`; `results.tex:175–176` | 19, 23 |
| R3.6a | `computational_experiments.tex:162–176`; `results.tex:11`, `:65–66` | 18, 20 |
| R3.6b | `main.tex:120–122`; `conclusion.tex:24–26` | 1, 25 |
| R3.7 | `conclusion.tex:67–71`, `:80`; `introduction.tex:16`; `methodology.tex:421`, `:462–470` | 2, 9–10, 25–26 |

## Build

```bash
cd /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The directory is its own git repository (`.git/` present), separate from the code repo.

`main.tex:65` has `\linenumbers` **commented out** while `lineno` is loaded at `:12`. The submitted
PDF therefore carries **no line numbers**, which is why both reviewers cite sections, figures and
algorithm-internal line numbers rather than manuscript line numbers.
