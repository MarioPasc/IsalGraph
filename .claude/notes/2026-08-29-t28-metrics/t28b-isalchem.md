# Work Log — T-28b: IsalChem p.7949 Fallback

**Agent** `t28b-isalchem`. **Wave** `2026-08-29-t28-metrics`.
**Branch** `worktree-agent-a47dd4c3e16b042b3`.

---

## 2026-08-29 — session 1

### Context secured

Checked archive at
`/home/mpascual/research/data/isalgraph_archive/results/reports/T-07-prior-work-delta/sources/`.
Contains: `29.txt` (75 KB PMC full text), `29_bioc.json` (135 KB BioC JSON, formulas lost in
extraction), `28.txt` (65 KB), `2512_10429v2.pdf` (IsalGraph arXiv preprint — NOT IsalChem).
No IsalChem PDF in archive; fetched from PMC instead.

Fetched PMC HTML: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12344769/` — 73 sections, 114 KB
indexed. Formulas recovered in full from HTML.

Required reads completed before writing:
- `.claude/notes/review/tasks/T-28-design.md` — §9 defines T-28b scope.
- `docs/worklogs/T-28-metric-distances.md` — fast probe findings.
- `data.md` — cohort statistics.
- IsalChem paper (PMC HTML, full Molecular Similarity section).

### What page 7949 contains

Journal page 7949 is the "Molecular Similarity" subsection of Experiments. It defines seven
fingerprint similarity metrics: Tanimoto, Dice, Cosine, Kulczynski, McConnaughey, Russel,
Sokal. All operate on binary fingerprint vectors computed by RDKit's Morgan algorithm.
No explicit graphlet repertoire is named — the "repertoire" is the hash-implicit circular
subgraph vocabulary of RDKit fingerprints.

The text extraction (`29.txt`) lost all formulas. The BioC JSON was corrupt/empty for those
passages. The PMC HTML rendering recovered them.

### Formula extraction quality

Tanimoto, Dice, Kulczynski, Russel, Sokal: extracted without ambiguity.

Cosine: HTML rendered as `Va·Vb / (|Va| + |Vb|)` which is dimensionally inconsistent.
Canonical form $\sum(V_a \cdot V_b) / \sqrt{\sum V_a \cdot \sum V_b}$ used with a note.

McConnaughey: HTML numerator parsed as $(\sum V_a + \sum V_b)^2 - (\sum V_a)(\sum V_b)$
which does not match any known coefficient. Canonical form $c(a+b)-ab)/ab$ used with a note.

### Core analytical finding

ALL seven metrics are defined on binary fingerprint vectors. For unlabelled topology-only
graphs, the adaptation is: replace RDKit fingerprints with graphlet fingerprints
(presence/absence of connected induced subgraphs up to $k$ nodes). The metric formulas are
unchanged.

**The metrics do not transfer cleanly because of size domination, not label dependency.**

Mechanism: graphlet counts scale linearly with $n$ on sparse graphs
($f(G) \propto n$). Tanimoto $\le n_{\min}/n_{\max}$ for any two graphs — a pure size ratio.
On COIL-DEL (size null $\rho = 0.9971$) the graphlet-fingerprint Tanimoto will achieve
similar or worse size null. On Letter LOW/MED where size variation is small, IsalGraph
already performs acceptably — this metric adds no new win.

Frequency normalization: collapses to near-1 for all pairs (sparse graphs have nearly
identical graphlet-type distributions regardless of size). Not a solution.

### Decisions

1. **Do not implement.** Size-domination forecloses the route on every cohort where the
   paper needs help. Documented in `T-28b-isalchem-fallback.md` §4.3.

2. **Repertoire specification written anyway** (§4 of the fallback doc) as insurance and
   as a forward reference if needed by a future revision. Graphlets up to $k = 5$ (21 types),
   O4 algorithm, with the specific tractability caveats for COIL-DEL.

3. **No code written.** Consistent with §3 of the mission brief: implementation is warranted
   only if §2 analysis is solid and the metric can clear the size null.

4. **Honest conclusion preserved.** If both primary tracks (WL, spectral ESD) fail, the
   correct manuscript claim is that GED on these cohorts is intrinsically size-dominated.
   The IsalChem fallback route does not change that conclusion.

### Files created

- `.claude/notes/review/tasks/T-28b-isalchem-fallback.md` — main analysis (deliverables 1+2).
- `.claude/notes/2026-08-29-t28-metrics/t28b-isalchem.md` — this file.

No code written; no test files created.

### Status

- [x] Required reading completed.
- [x] p.7949 content identified and formulas documented.
- [x] Per-metric label-transferability verdict written.
- [x] Subgraph repertoire specified (graphlets up to k=5, O4 algorithm).
- [x] Size-domination analysis completed; recommendation: do not implement.
- [x] Files written and committed.
