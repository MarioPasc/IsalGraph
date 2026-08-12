# Editor-in-Chief compliance — bibliography, venue, pages, package

**Owners**: T-08, T-19 (bibliography), **T-26** (the two arithmetics), T-15 (pages), T-24 (package)
**Serves**: EiC.a1–a4, EiC.b, EiC.c, M1
**Status**: **pass/fail, checked independently of the reviewers.**

> `mail.txt:124`: "here is a list of points worth checking … **I will check that these are adhered to
> before your paper is approved for publication**, assuming the revision satisfies the Associate
> Editor and Reviewers."
>
> **These do not negotiate.** A revision that satisfies every reviewer and misses one of these is not
> approved. Two of them are arithmetic, and **neither reconciled before this plan** — that is what
> T-26 exists for.

Related: [manuscript](manuscript.md) (page budget) · [approx_ged](approx_ged.md) (which references) ·
[demands](demands.md)

---

## 1. Measured baseline

Independently re-derived 2026-08-12 from `*.tex` × `cas-refs.bib`:

| Quantity | Value |
|---|---|
| `@`-entries defined in `cas-refs.bib` | **56** |
| **Cited with comments stripped — the printed bibliography** | **43** |
| Cited including commented-out LaTeX | 45 |
| The two extras | `Fischer2015hausdorff`, `Lerouge2017ilp` — cited **only** from `methodology.tex:805–806`, commented out |
| **Dead entries** (defined, never cited uncommented) | **13** |
| **Headroom to the 55 ceiling** | **12 slots** |
| `main.pdf` | **35 pages of 35** |

`elsarticle-num` prints only keys reached by an **uncommented** `\cite`, which is why 43 is the
operative number and why **retiring a dead `.bib` entry frees nothing** — the 13 dead entries never
occupied a slot. **The only real relief is removing existing citations from the text.**

---

## 2. EiC.a1 — bibliography 35–55 items

**The allocations did not fit: 16–17 requested against 12 available.**

| Purpose | Originally requested |
|---|---:|
| AGM, gSpan, nauty/Traces, bliss, graph6, Babai | 5–6 |
| GED approximation: Riesen–Bunke 2009, Fischer 2015, Blumenthal | 3 |
| Datasets: IAM Graph Database (Riesen & Bunke, SSPR 2008) | 1 |
| GEDLIB (Blumenthal et al., GbRPR 2019) | 1 |
| **Recency (2025–26)** — T-19 requires **≥ 6**, not the 2–3 originally booked | **6** |
| **Total** | **16–17 against 12** |

### The allocation that fits

| Purpose | Slots | Note |
|---|---:|---|
| AGM (Inokuchi 2000), gSpan (Yan & Han 2002), nauty/Traces (McKay & Piperno 2014), Babai 2016 | **4** | graph6/sparse6 cite the nauty manual — same key. **Cutting the bliss/Traces backends ([competitors](competitors.md)) removes the bliss citation too** |
| Riesen–Bunke 2009 (BIPARTITE), **Fischer 2015** (*Pattern Recognition*, HED), Blumenthal & Gamper **2018** (*IEEE TKDE* 30(3):503–516) | **3** | the earlier "Blumenthal 2020" and the TKDE 2018 entry are **not the same paper** — reconcile |
| IAM Graph Database (SSPR 2008) | **1** | |
| GEDLIB (GbRPR 2019) | **1** | |
| Recency / PR-venue additions, 2025–26 | **4** | ≥ 3 at PR-field venues other than the PR journal; **self-citations excluded** |
| **Total** | **13** | **one over — drop Babai, or one GED-approximation entry** |

> **Budget for I-08b.** Fischer 2015 is *in the allocation above*, so plan for **12 with Fischer
> counted**, not 12 plus Fischer. If `Lerouge2017ilp` is also uncommented — plausible, since the
> revision expands the GED discussion — headroom falls to **11**.

**Options considered**: (A) accept 55 as the working ceiling and spend all 12, dropping the weakest
4–5 planned additions — ~0 effort, but arrives at the ceiling with no margin for round 2; (B) remove
4–5 existing citations from the text — 0.5 d, keeps every planned addition, but touches introduction
prose late and each removal must be checked against R1.2a/AE.2, which *added* those obligations;
(C) re-scope T-19 from ≥ 6 to ≥ 4. **Recommendation: A + C, with the allocation above.**

---

## 3. EiC.a2 — cover last and current year

**Measured**: of 43 printed references, **5 postdate 2023** — three from 2024
(`jain2024graphedx`, `ju2024comprehensive`, `khoshraftar2024survey`) and **two from 2025, both group
self-citations** (`lopezrubio2025isalgraph` = [28], `ThurnhoferHemsi:2025` = [29]).

> **Zero third-party references after 2024. Zero from 2026.**

**T-19's criterion must exclude self-citation**, or it can be satisfied without adding a single
external reference. Acceptance: **≥ 4 additions dated 2025–2026, self-citations excluded, no year gap
after 2024.**

---

## 4. EiC.b — recent *pattern-recognition* work, not only the PR journal

**Measured venue coverage of the 43 printed references:**

| Venue | Count | Years |
|---|---:|---|
| *Pattern Recognition* (journal) | **6** | 2021, 2021, 2022, 2023, 2023, 2023 |
| *Pattern Recognition Letters* | 1 | **1983** |
| SSPR / S+SSPR | 1 | 2008 |
| **CVPR / ICCV / ECCV / ICPR / TPAMI / IJCV** | **0** | — |

> **No pattern-recognition-venue reference after 2023** — which is exactly what `:128` prohibits, and
> the PR-field coverage is six citations of the *Pattern Recognition journal*, which is exactly the
> narrowness `:128` names ("not only the Pattern Recognition journal").

**T-19's b-half had no threshold** — "a stated PR-community share" fixes no value, so no outcome could
fail it. **Locked criterion: ≥ 3 of the additions at pattern-recognition venues other than the
*Pattern Recognition* journal, ≥ 1 dated 2025–26.**

**Fischer et al. 2015** (*Pattern Recognition* 48(2):331–343) satisfies **venue**, contributes
**nothing** to recency, and **is not currently in the printed bibliography at all** — it is cited only
from commented-out LaTeX. Uncommenting it is a venue win and a slot cost. Both halves of EiC.a/b are
needed and they are different work.

---

## 5. EiC.a3 — arXiv citations

**[28] is and will remain arXiv-only** — "substitute arXiv citations with their peer-reviewed
versions" **cannot be satisfied** for it. Response: state that plainly in one sentence.

Then reduce the *visible* arXiv footprint: **strip the `note = {arXiv:…}` fields from the five entries
that already name ICLR / NeurIPS venues** — `kipf2017gcn`, `velickovic2018gat`, `xu2019powerful`,
`fey2019pyg`, `jain2024graphedx`. **Rendered arXiv count: 6 → 1.**

---

## 6. EiC.a4 — no uncommented citation groups

Comment `\cite{garey1979,Zeng:2009}` (`methodology.tex:803`) individually. `Zeng:2009` is the STAR
lower bound, which [approx_ged](approx_ged.md) discusses anyway — **so the fix is free**.

**The four-way group at `introduction.tex:31` is already individually commented — do not "fix" it.**

---

## 7. EiC.c — 35 pages

`main.pdf` is **exactly 35 of 35** and the revision adds ≈ 12–13 gross against ≈ 4.75 recoverable.
Full arithmetic, the supplementary query and the pre-declared cut ranking:
[manuscript](manuscript.md) §2–§3. **T-26 re-derives the budget as deltas, not gross sizes, after
T-08 and T-19 and before T-15.**

---

## 8. Submission package — every item verified before upload

| # | Item | State today | Owner |
|---|---|---|---|
| 1 | **LaTeX source files, not PDF** (`mail.txt:22`) | sources present, package never assembled | T-24 |
| 2 | Main PDF, **≤ 35 pages**, double-spaced single column | 35 / 35 | T-15 |
| 3 | Response letter, parts 0–5 | not started | T-14 |
| 4 | **Generative-AI declaration** (E11) | **commented out**, `main.tex:198–202` | T-24 |
| 5 | **Author biographies + photos** — *Pattern Recognition* requires them | **commented out**, `main.tex:225–245` | T-24 |
| 6 | Acknowledgements (funders, SCBI, NVIDIA) | **commented out**, `main.tex:175–177` | T-24 |
| 7 | Highlights, updated for the scoped claims | not started | T-24 |
| 8 | Graphical abstract — updated; **filename misspelt** `graphical_abtract.pdf` (E12) | stale | T-24 |
| 9 | Declaration of competing interest | absent | T-24 |
| 10 | Data availability statement — GEDLIB, IAM, GraphEdX provenance | absent | T-21, T-24 |
| 11 | Artifact updated: competitor backends, GEDLIB pin, new datasets, library versions | not started | T-21 |
| 12 | Bibliography 35–55; arXiv `note` stripped from five entries; `\cite{garey1979,Zeng:2009}` split | 43 cited, 13 dead (E9) | T-08, T-19 |

**Two reproducibility traps that belong in item 11, not only in internal notes:**

- the `.so` **does not rsync** — the C++ extension is built on the cluster as part of environment setup;
- build flags are `-march=x86-64-v3`, **never** `-march=native` — Picasso is heterogeneous and
  `native` yields SIGILL on a fraction of nodes, which reads like flaky hardware rather than a build
  fault.
