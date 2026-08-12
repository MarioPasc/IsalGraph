# Revision plan — PR-D-26-03293 (IsalGraph)

**Status**: **v0.9, 2026-08-12.** **GED implementation LOCKED** (§7.3); cohort locked from measured
counts (`data.md` §0); statistics locked (`statistics.md` v2.2); coverage audited (§0.5,
`gap-audit.md`), **re-audited in both directions** (`.claude/notes/audit-2026-08-11b/`) and that
re-audit **itself audited** (`.claude/notes/audit-2026-08-11b/third-auditor.md`);
**T-16 rejected**, competitor distances handed to a measured selection (`competitors.md`), labels
tiered and referred to the PI (`labels.md` §0).
**Decision**: Major Revision, Pattern Recognition. **Due 2026-08-31 (19 days remaining).**

> ⚠ **Three things this plan asserts are not true, found 2026-08-11b by re-measurement. Read before
> executing anything.** (i) **Validation gate 2 cannot be run** — `ged_bounds.py` and 12 other named
> scripts do not exist (§7.3 banner, T-25, **S-e**). (ii) **The board is 93.5 days, not 76.5**, and
> the declared critical path is 27.5 days serial against a 19-day window (§7 banner, **S-f**).
> (iii) **§0.5 carried two demands the letter does not make and missed one it does** (§0.5 banner).
> Coverage is otherwise sound: **41 letter demands across 42 matrix rows, all owned** — the one
> genuine hole (R3.1a(ii)) is now on T-07.
>
> ⚠ **And one thing the 2026-08-11b audit itself asserts is not true** (`third-auditor.md`, 2026-08-12):
> its verdict that **T-03's all-pairs AIDS census is proportionate** does not survive the plan's own
> `statistics.md` **D2**. The census's drivers are real but its *dose* is not: 769 AIDS graphs are the
> independent unit either way, so the census buys coverage, not precision — **~900–1,550 core-hours
> and 2–5 days of the long pole for no statistical gain.** See §7.1 and **S-f option E**.
**Mode**: full recompute. Lock data → lock methods → lock statistics → recompute → write.
Page budget is **not** a constraint during drafting; trim at the end — under the ordering and
priority rules in `manuscript.md` §3, which are now part of the lock.

**Companion documents** — read these before executing any ticket:
- **`data.md`** (v1.2) — measured dataset inventory, exact/approximate GED cost, encoding cost,
  compute budget. All numbers measured 2026-08-11, not quoted. **§0 is the only table a printed
  number may be taken from**; §2.1's size columns are raw-set values and carry a correction banner.
  Its §0 AIDS "raw" column is the *connected* count (I-02) — use 911, not 819.
- **`statistics.md`** (v2.2) — the locked statistical protocol (D1–D15): Mantel, graph-level
  bootstrap, partial Mantel/MRM, Friedman/CD, multiplicity, calibration ladder, censoring rule,
  resampling budget.
- **`labels.md`** (v2.0) — the R1.3 / AE.4b response, **tiered and awaiting a PI decision on effort**
  (§0, due 2026-08-18). Establishes that the manuscript never claimed label handling and that a
  labelled variant is a different paper.
- **`competitors.md`** — competitor backends and the **T-04a metric-feasibility experiment** that
  selects each representation's distance by measurement rather than by assertion.
- **`manuscript.md`** — section-by-section rewrite map, artifact inventory, **page budget**, response
  letter architecture, submission and compliance package.
- **`gap-audit.md`** — the coverage audit that produced §0.5 and tickets T-17…T-24: 10 unowned
  demands, 16 flawed or infeasible locked decisions, with severities.
- **`.claude/notes/audit-2026-08-11b/third-auditor.md`** — the audit **of** that re-audit. Read §1
  before submitting T-03 and §2 before applying any `data.md` integrity fix: **I-11 must not be
  applied as written**, and only 4 of the 24 integrity defects reach a printed number.

---

## 0. Author decisions taken

| # | Decision | Section |
|---|---|---|
| 1 | Re-execute everything with the C++ engine | §2 |
| 2 | Competitors enter the experiments, vendored **as backends in the IsalHG `iso_backends` style** | §4 |
| 3 | **Merge all splits.** GED is symmetric and carries no train/test semantics | §3.2 |
| 4 | **Exact GED for `n <= 12`; approximate GED above it, on larger real datasets** — replaces the controlled-edit cohort | §3.3 |
| 5 | **Decline the sequential-model experiment**; defer to future work, downgrade LM claims to conjecture | §6 |
| 6 | Render an S2G/G2S example figure as in IsalSR / IsalHG | T-09 |
| 7 | Ignore the page budget while drafting | T-15 |
| 8 | gSpan vendored from `github.com/LasseRegin/gSpan` | §4.2 |
| 9 | IsalChem source at `github.com/icai-uma/IsalChem`; ~~paper unavailable~~ — **CORRECTED 2026-08-11b: [29] is fully published.** `cas-refs.bib` gives `ThurnhoferHemsi:2025`, *Representation of Molecules by Sequences of Instructions*, **J. Chem. Inf. Model. 65(15):7936–7955, 2025**, and it is already cited in the manuscript. T-07 was budgeted 1–4 days partly on the premise that only source code was available; **the paper is obtainable, which should make T-07 cheaper and D19 ([29]'s LSTM experiment) directly resolvable** rather than inferred from code | T-07 |
| 10 | **[28] is and will remain arXiv-only** | §5.3 |
| **11** | **GED comes from GEDLIB, not our own code.** Exact = `ANCHOR_AWARE_GED`; proven lower bound = `BRANCH_FAST`; proven upper bound = `IPFP`. Verified working on Picasso 2026-08-11 | **§7.3 — authoritative** |
| **12** | **Cohort locked** to the IAM Graph Database; **TUDataset dropped** — the cohort reaches **n = 98 retained**, an 8.2× extension of the submitted ceiling. Rationale corrected 2026-08-11 (the 417-node graph is disconnected and discarded, `gap-audit.md` MF1); re-measurement confirmed **no dataset moves in or out**. **RE-AFFIRMED by the author 2026-08-11 on the corrected number** | `data.md` §0 |
| **13** | **Labels are tiered and referred to the PI.** The manuscript never claimed label handling, R1.3 asks for a discussion, and a labelled variant is a different paper. Tier 0 (rebuttal, label column, **E6 fix**, future work) is not optional; Tiers 2–3 need an effort decision by **2026-08-18** | `labels.md` §0 |
| **17** | **T-16 rejected** — no reviewer asked for `wl_pruned_canonical`; the WL *measurement* moves into T-13 | §7.2 |
| **18** | **Competitor distances are selected by measurement (T-04a), not by assertion**, with the rule fixed in advance and ties broken on cost — never on correlation with GED | `competitors.md` §2.4 |
| **19** | **T-09 and T-10 merged**; both figures double as the refreshed **graphical abstract**, which is submitted separately and costs no manuscript pages | §7 |
| **14** | **Encoding-censored graphs are analysed, not dropped** — greedy-min fallback plus a complete-case sensitivity arm | `statistics.md` D14 |
| **15** | **No marked-up manuscript.** The response letter's per-comment pointers are the change map | `manuscript.md` §4.4 |
| **16** | **Query patcog@elsevier.com on day 1**: does supplementary material count toward the 35-page limit? The whole page strategy branches on the answer | `manuscript.md` §3.2 |

> **§7.3 is the single source of truth for every GED computation.** Anything in §3 that predates it
> is superseded and has been struck through below.

**Three decisions still need sign-off** and are marked in place, not buried (full list in §11):
decision 12's re-affirmation on its corrected rationale (`data.md` Q9 — a wording decision, since
re-measurement showed no dataset moves); **the PI's choice of `labels.md` tier** (**2026-08-18**);
and disclosure of the self-found defects E1–E12 to the reviewers (`manuscript.md` §4.3,
**2026-08-20**).

---

## 0.5 Traceability matrix — every demand, locked to a decision, a ticket and an artifact

**This is the coverage contract.** One row per numbered demand in `mail.txt`, plus every self-found
defect. A row with no ticket is a hole. Derived in `gap-audit.md`; **re-audited independently in
`.claude/notes/audit-2026-08-11b/`, which did not inherit `gap-audit.md`'s verdicts.**

> **Audit 2026-08-11b — what changed here.** The demand inventory was rebuilt from `mail.txt` alone
> (`.claude/notes/audit-2026-08-11b/inventory.md`, 41 demands). Against it, this matrix carried **two
> rows the letter does not contain** and **missed one it does**:
> - **R1.3b** and **M3** were booked as demands; neither has an imperative in the letter. R1.3b is a
>   premise ("may come from"), M3 is inferred from `:67`, which is the Area Editor's **priority
>   statement**, not a request for a point-by-point response. Both rows are retained — a
>   point-by-point letter is journal convention and the R1.3b work has other owners — but they are
>   now labelled so no budget is allocated to them directly.
> - **AE.5** (`mail.txt:69`, "There are additional comments from the reviewers that **should** also be
>   addressed") had no row. Added below.
> - The claim "there are none" was also inaccurate on its own terms: **M2** and **R3.2** carry
>   section pointers (`§12`, `§6`) rather than tickets (I-18). R3.2's owner is legitimately a
>   *decision*; M2's is a calendar. The contract wording now admits both.
>
> `gap-audit.md` was treated as an object under audit, not as evidence. Three of its own findings —
> **MF12, MF13, MF17** — have no downstream owner anywhere in the corpus (I-25), and `data.md`
> §2.2.1's banner implementing MF17 carries no MF pointer while the neighbouring MF1 corrections all
> do.

Legend — **✓** covered before this audit · **NEW** owner created by this audit · **FIX** a locked
decision that was wrong and has been corrected.

### Editor-in-Chief and submission mechanics

| ID | Demand (`mail.txt`) | Locked decision / experiment | Ticket | Manuscript artifact | |
|---|---|---|---|---|---|
| M1 | Upload **source files**, not PDF (`:22`) | assemble the LaTeX package | **T-24** | — | **NEW** |
| M2 | Due **2026-08-31** (`:20`) | dated calendar with gates | §12 | — | **NEW** |
| M3 | Point-by-point response — **the deliverable is right, its cited authority is not** | **MISMATCHED, audit-2026-08-11b.** `:67` is the Area Editor's **priority statement**, and it requests no response document; the letter contains no imperative for one. T-14 is justified independently by Editorial Manager convention (`00-editor-and-decision.md:34–35` reached this separately), so the ticket stands — but the citation must change. **Second defect: `:67`'s actual content — "Please address these concerns thoroughly, as they will strongly influence the potential impact of the work", a weighting over AE.1–AE.4c — appears nowhere in this matrix.** It is the sentence that converts AE.4a and AE.4b from soft to requirement. Fix ≈ 15 min | T-14 | response letter | **FIX** |
| EiC.a1 | Bibliography **35–55** items (`:126`) | 43 printed → ≤ 55, **12 slots**. **The allocations do not fit.** §5.4 sums 5–6 + 3 + 1 + 1 + 2–3 = 12–14 ("over by ~2"), but **T-19 requires ≥ 6 from 2025–2026**, not 2–3, so the true sum is **16–17 against 12 — over by 4–5**. No ticket sees it: T-08 owns the table, T-19 owns the ≥ 6. And §5.4's stated relief — "retire a dead citation" — **frees nothing**: `elsarticle-num` prints only *cited* entries, so the 13 dead entries never consumed a slot. The only real relief is removing existing citations from the text, which no ticket contemplates | **T-26** (reconcile), T-08, T-19 | `cas-refs.bib` | **FIX — UNDER-blocking, audit-2026-08-11b** |
| EiC.a2 | Cover **last and current year** (`:126`) | ≥ 6 refs from 2025–2026; protocol + acceptance criteria. **Measured baseline, audit-2026-08-11b**: of 43 printed refs only **5** postdate 2023 — 3 from 2024, **2 from 2025, and both 2025 entries are group self-citations** (`lopezrubio2025isalgraph` = [28], `ThurnhoferHemsi:2025` = [29]). **Zero third-party refs after 2024; zero from 2026.** T-19's "≥ 6 from 2025–2026" must therefore exclude self-citation, or it can be satisfied without adding a single external reference | **T-19** | related work | **FIX** — baseline was unmeasured |
| EiC.a3 | No excessive arXiv (`:126`) | strip `note = {arXiv:…}` from the 5 peer-reviewed entries: 6 → 1 | T-08 | `cas-refs.bib` | ✓ §5.3 |
| EiC.a4 | No uncommented citation groups (`:126`) | comment `\cite{garey1979,Zeng:2009}` individually; **do not "fix"** `introduction.tex:31` | T-08 | `methodology.tex:803` | ✓ §5.4 |
| EiC.b | Cite **recent pattern-recognition** work (`:128`) | venue audit + targeted additions. **Two corrections.** (i) ⚠ **Fischer 2015 is not in the printed bibliography at all** — `Fischer2015hausdorff` is cited only from commented-out LaTeX (`methodology.tex:805–806`), so it satisfies neither venue nor recency and the baseline was off by one (I-08b). (ii) **T-19's b-half has no threshold**: "a stated PR-community share" fixes no value, so no outcome can fail it — T-19 could close with all 6 additions at non-PR venues. **Measured**: PR-field coverage is 6× *Pattern Recognition* journal (2021–2023), PR Letters ×1 (**1983**), SSPR ×1 (2008); **zero** CVPR/ICCV/ECCV/ICPR/TPAMI/IJCV; **no PR-field reference after 2023** — which is exactly what `:128` prohibits. Proposed criterion: **≥ 3 of the 6 at PR-field venues other than the PR journal, ≥ 1 dated 2025–26** | **T-19** | related work | **FIX — UNDER-blocking** |
| EiC.c | **≤ 35 pages** (`:130`) | page budget, priority ranking, supplementary query. **`manuscript.md:65`'s "≈12–13 gross addition" is not derivable and its composition is wrong in both directions**: it **over-counts ≈3.5–4.5** by pricing Figs. 1–4 and Tabs. 1–3 (6.0 p) as new when they are *replacements* (Tab. 1 is "keep"; Fig. 3 "demote" shrinks), and **under-counts ≈4.8–5.8** by omitting §1.x related work, §2.2.x complexity, §3.2 protocol prose, §3.3 implementation, §5 limitations and the Suite 1/2 framing — all committed in `manuscript.md` §1, none in §2's page column. The magnitudes nearly cancel, **but the composition is what the strategy rests on**: the supplementary-relief plan targets calibration / ladder / CD (≈2–2.5 p, all inventory rows), while the ≈5 p of genuine growth is **main-text prose that cannot move to supplementary**. Also: **decision 16's query to patcog@elsevier.com is scheduled Day 1 and nothing records it sent** | **T-26** (re-derive), T-15 | whole document | **FIX — UNDER-blocking** |

### Area Editor

| ID | Demand | Locked decision / experiment | Ticket | Manuscript artifact | |
|---|---|---|---|---|---|
| AE.1 | Graph size impact must be clear (`:59–60`) | Suite 1 (`n ≤ 12`, exact) / Suite 2 (`n ≤ 98`, proven bracket); ceiling attributed to the **reference**, not to IsalGraph; relative bracket width vs `n` | T-01, T-05, T-06 | §3.1, §4 size results | ✓ + **FIX** (`statistics.md` §6.1) |
| AE.2 | Related-work framing + references (`:62`) | new related-work section: canonicalisation literature | T-08 | **§1.x (new)** | ✓ |
| AE.3 | **Side-by-side comparison of existing graph representations** — properties, strengths, limitations of each (`:63–64`) | comparison table as a **paper artifact**, axes from R1.2 | **T-17** | **comparison table (new)** | **NEW** — was mapped only to the [28]/[29] delta |
| AE.4a | Choice of benchmark models (`:66`) | six competitor representations enter three experiments; **each one's distance selected by measurement** (T-04a, `competitors.md`) | T-04, **T-04a**, T-06 | Tabs. 2–3, Fig. 1–2 | ✓ §4.3 + **FIX** |
| AE.4b | **Fully labeled vs partially labeled** (`:66`) | a **label-content column** in the dataset table (Tier 0) | **T-18** | §3.1 dataset table | **NEW** |
| AE.4c | Associated analysis of results (`:66`) | `statistics.md` D1–D15 | T-02, T-06 | §3.2, §4 | ✓ |
| **AE.5** | **"There are additional comments from the reviewers that *should* also be addressed in the revised paper -- please check their comments carefully" (`:69`)** | **Catch-all, requirement modal.** Largely subsumed by the 20 numbered rows, but **not entirely**: R3's un-numbered preamble ends in a requirement-modal sentence — `mail.txt:83`, "the **rationale**, novelty, methodological details, and interpretation of the results **require further clarification**" — naming four targets. Novelty → R3.1, details → R3.4, interpretation → R3.5/R3.6; **"rationale" has no clean owner.** R1's `:73` is pure framing and drops nothing. **Fix ≈ 0, not 0.25 d** (`third-auditor.md` §5.1): "rationale" is housed in the *same* §2.x closing paragraph R3.1a(ii) already buys, so the only marginal work is one verification pass over `:73–116` inside T-14 | **T-14** + **T-07** (rationale, no marginal cost) | response letter, §2.x | **NEW — UNDER-major, audit-2026-08-11b** |

### Reviewer 1

| ID | Demand | Locked decision / experiment | Ticket | Manuscript artifact | |
|---|---|---|---|---|---|
| R1.1 | GED runtime comparison unfair; compare against a similar problem setting (`:75`) | competitor encode-time curves; **per-graph and per-pair costs stop sharing an axis** | T-04, T-06, T-20 | Fig. 2 restructured | ✓ §4.3(c) |
| R1.2a | AGM and gSpan uncited (`:77`) | both cited. **T-08 is the owner** — the ask is *discussion*, ≈0.5 d, and it is satisfied by citation alone. gSpan's vendoring (T-04) serves R1.1/AE.4a, **not** this row: if T-04 slips, R1.2a is still answered | **T-08** (T-04 only enriches it) | §1.x | ✓ + **FIX** — T-04 demoted from owner to enrichment |
| R1.2b | Five axes: uniqueness, expressiveness, efficiency, scalability, **downstream learning** (`:77`) | all five are printed rows; downstream reads **"not evaluated"** — R3.2 is declined and the table says so | **T-17** | comparison table | **NEW** |
| R1.3a | Density insufficient to explain AIDS (`:79`) | true density computed; within-AIDS density stratification, **which can refute** `conclusion.tex:30–36` | T-02, T-06 | §4.x | ✓ §8 |
| R1.3b | **PREMISE, not a demand** — "the performance degradation on AIDS **may come from** the loss of label information" (`:79`). Declarative; the letter contains no imperative here | rebuttal (both sides topology-only) **leads** and is free: `computational_experiments.tex:52` makes AIDS a **topology-only variant** and `:30–31` discards labels, so a variable absent from both sides of the correlation cannot have determined it. **Licenses no work of its own** — every Tier 0–1 item is served by R1.3c, R1.2b or AE.4b, and is verified to be so | served by **R1.3c / R1.2b / AE.4b** — no independent allocation | §4 paragraph + 2 table columns | **FIX** — was booked as a demand |
| R1.3c | Discuss the limitation and its impact (`:79`) | R1.3 asks for a **discussion**, not an experiment — the missing piece is the *connection* between the §5 limitation and the §4 AIDS interpretation | T-18, T-12 | §5 limitations | **NEW** |
| R1.3d | Labels as future work (`:79`) | concrete `Σ × L` extension; **already named as future work at `conclusion.tex:71`, `:81`**; [29] as precedent **conditional on T-07** | T-07, T-12 | §5 future work | ✓ + **NEW** substance |

### Reviewer 3

| ID | Demand | Locked decision / experiment | Ticket | Manuscript artifact | |
|---|---|---|---|---|---|
| R3.1a**(i)** | Inherited / modified / new vs [28], [29] (`:86`) | read both sources; delta table | T-07 | **§2.x delta table (new)** | ✓ |
| **R3.1a(ii)** | **"…and explain why the combined extension constitutes a sufficiently substantive contribution" (`:86`)** — the second conjunct of the same sentence | **one paragraph closing §2.x**: the completeness theorem (`methodology.tex:628–637`) is the new result; generic topology replaces [28]'s fixed node ordering and [29]'s molecular restriction; scope extends to unlabeled, unbounded-degree graphs. Re-orders facts T-07 already gathers — **no new investigation** | **T-07** | **§2.x closing paragraph** | **NEW — audit-2026-08-11b** |
| R3.1b | "No existing method satisfies all four" too absolute (`:86`) | §10 B6; softened and attached to the T-17 table | T-12, T-17 | §1, §5 | ✓ |
| R3.2 | **Sequential-model evaluation** (`:89`) | **DECLINED** (decision 5) + claims come down + contingency with a **2026-08-22 go/no-go** | §6 | abstract, §5 | ✓ + **FIX** (contingency) |
| R3.3a | Narrow "any finite simple graph" / "arbitrary graphs" (`:92`) | §10 B1: undirected **connected**; directed **root reaching all nodes**; S2G total, G2S partial | T-12 | abstract, §1, §5 | ✓ |
| R3.3b | Thm 2.12 and the `directed` flag (`:92`) | restate **within a fixed directedness class**; hypothesis moves from proof to statement; **re-verify all three proof steps and Cor. 2.13** | **T-22** | §2.3.3 | ✓ + **NEW** (proof/corollary audit) |
| R3.3c | Is the flag part of the representation? (`:92`) | **external metadata**; exact witness (`"V"` under both semantics); never quote a rate without its window | T-12, **T-22** | §2.3.3 | ✓ |
| R3.4a | Alg. 2 `C`/`c` vs Table 1 (`:95`) | pseudocode rewritten to match `graph_to_string.py:208–238` — **guards *and* duplicate checks** | T-11 | §2.2, Alg. 2 | ✓ §9 |
| R3.4b | `P(M)` recomputed or precomputed; cost the four operations (`:97`) | **recomputed per frame**; four operations costed; `|Aut(G)|`-governed worst case | T-13 | **§2.2.x (new)** | ✓ |
| R3.4c | `n^{4.9}` vs `n^{9.0}`; "super-polynomial" (`:99`) | all exponents re-derived; three-way separation | T-06, T-13 | §4.2, §5 | ✓ §9 |
| R3.5a | Justify exclusions, report removals per dataset (`:102`) | pair-accounting ladder, per dataset | T-02, T-06 | ladder table | ✓ |
| R3.5b | **Interpret Fig. 3 cautiously; dataset-level correlations are primary** (`:104`) — the row previously read "Heterogeneous cost models", which is the comment's *premise*, not its ask | **`statistics.md` D5 answers the literal clause at ~0 cost** (per-dataset primary, pooled demoted): one caveat at `results.tex:187` + demote `conclusion.tex:38–41`. **D6's full recompute is a separate, deliberate choice** to retire the objection rather than caveat it, and it is driven by **F2** (473,147-pair gap; ~378–630 core-h of AIDS alone, independent of any cost model) and by **Cor. 2.13's pseudometric problem** — *not* by the Suite-2 cohort, which costs 1.24 core-h total. ⚠ **Those drivers set the direction, not the dose** (`third-auditor.md` §1): they justify recomputing AIDS GED under one cost model; they do not justify the **census**. See §7.1's two-stage structure | **D5** (floor, T-02/T-06) · T-03 **stage 1**, T-05 | §3.1, §4.3 | ✓ + **FIX** |
| R3.5c | Pair dependence; describe the bootstrap; graph level (`:106`) | D2 graph-level cluster bootstrap, D3 Mantel; **D15 makes it affordable** | T-02, T-06 | §3.2 | ✓ + **FIX** |
| R3.6a | "GED standard construction" not established (`:109`) | B3 rename + real serializations beside it + shared edit-operation alphabet | T-12, T-17 | §3.2.3, Tab. 2 | ✓ |
| R3.6b | "Strongly correlates" is not uniform (`:111`) | B4 — the results section's conditional framing propagates to abstract and conclusion | T-12 | abstract, §5 | ✓ |
| R3.7a | Three limitations to add (`:114`) | B5 — `n` ceiling **with its cause**, exponential worst case, no sequential/downstream task | T-12 | §5 | ✓ |
| R3.7b | Dedicated comparison subsection (`:116`) | §2.x delta table + §1.x | T-07, T-08 | §2.x | ✓ |
| R3.7c | Canonical search-space schematic (`:116`) | renderer exists: `viz/search_tree.py::canonical_search_tree_figure` (verified). Merged with the S2G/G2S figure; both double as the new graphical abstract | **T-09** | §2.3 figure | ✓ |
| R3.7d | Separate theory / worst case / empirical (`:116`) | three-way separation; `|Aut(G)|` characterisation replaces "exponential" | T-13 | §2.2.x, §4.2 | ✓ |
| R3.7e | Four broad statements (`:116`) | equivariance → **invariance**; + B1, B4, D2 | T-11, T-12 | §1, abstract, §5 | ✓ |

### Self-found defects (no reviewer raised these)

| ID | Defect | Ticket | |
|---|---|---|---|
| E1 | Density never computed; no node count reported | T-01, T-20 | ✓ |
| E2 / F2 | 473,147-pair gap; LINUX 3,916 vs 1,685 — **cause is within-split GED coverage** | T-03 | ✓ |
| E3 | Fits declared `n = 3–20`, greedy data to 50 | T-06, T-20 | ✓ |
| E4 | A fourth node range (`n = 3–11`) | T-20 | ✓ |
| E5 | Abstract self-contradiction (`:106` vs `:114`) | T-12 | ✓ |
| E6 | "Labels present in all five datasets" — **false for LINUX**. Two sites, both verified verbatim: `conclusion.tex:70` and `:81` | **T-12 owns the edit** (was claimed by T-12, T-18 Tier 0 **and** §9/T-11 simultaneously — I-19) | **FIX** |
| E7 | Algorithms float to pp. 33–35 | **T-11** | **FIX** — moved out of T-15; it changes pagination and must precede the trim |
| E8 | Draft self-correction printed in Example 2.3 | T-11 | ✓ |
| E9 | 13 dead bibliography entries | T-08 | **NEW** |
| E10 | WL kernel and Mantel computed, never reported | T-04, T-02 | ✓ |
| E11 | Generative-AI declaration commented out | **T-24** | ✓ |
| E12 | Orphaned figure PDFs; **`graphical_abtract.pdf` misspelt** | **T-24** | **NEW** |
| D19 | [28] Transformer / [29] LSTM claims **unverified** | T-07 | ✓ |
| — | C++ engine and GEDLIB absent from the Implementation section; artifact release | **T-21** | **NEW** |
| — | Picasso `fscratch` file-count quota exceeded, 7-day grace | **T-23** | **NEW — blocking** |

---

## 1. What the data audit found (2026-08-11, measured)

### F1 — The current benchmarks contain no large graphs. The `n <= 12` filter is nearly vacuous.

| Dataset | raw N | **max n** | median n | connected | survives `n<=12` |
|---|---|---|---|---|---|
| IAM Letter LOW | 2,250 | **8** | 5 | 1,180 | **2,250 (100%)** |
| IAM Letter MED | 2,250 | **9** | 5 | 1,253 | **2,250 (100%)** |
| IAM Letter HIGH | 2,250 | **9** | 5 | 2,059 | **2,250 (100%)** |
| LINUX | 89 | **10** | 9 | 89 (100%) | **89 (100%)** |
| AIDS | 911 | **20** | 11 | 819 | 769 |

`n_max: 12` removes **zero** graphs from IAM and LINUX. What removes IAM graphs is the
**connectivity** requirement (2,250 → 1,180 on LOW, 47.6% lost) — stated in the manuscript, never
quantified. This is why §3.3 must bring in new datasets rather than unfilter the old ones.

### F2 — GraphEdX ships GED only *within* train/val/test splits.

| Dataset | splits | within-split pairs | `n_valid_ged_pairs` | all pairs | coverage |
|---|---|---|---|---|---|
| LINUX | 53 / 17 / 19 | 1,378 + 136 + 171 = **1,685** | **1,685** | 3,916 | 43.0% |
| AIDS | 546 / 182 / 183 | 148,785 + 16,471 + 16,653 = **181,909** | **181,909** | 414,505 | 43.9% |

Exact match on both. **This corrects `verified-discrepancies.md` E2**, which attributes the LINUX
3,916 → 1,685 drop to the `GED > 0` / `Lev > 0` filter. It is missing ground truth, not filtering.
The published LINUX ρ = 0.433 and AIDS ρ = 0.349 are within-split figures, undisclosed.

### F3 — Encoding cost has collapsed. GED is the only cost centre.

Real benchmark graphs, `engine() == 'cpp'`, single thread, `process_time`:

| n | exact GED / pair (nx A*) | pruned-canonical encode / graph | ratio |
|---:|---:|---:|---:|
| 5 | 4.0 ms | 6 µs | 6.7 × 10² |
| 9 | 336 ms | 16 µs | 2.1 × 10⁴ |
| 11 | 7.48 s | 21 µs | 3.6 × 10⁵ |
| 12 | **36.9 s** | 27 µs | 1.4 × 10⁶ |
| 20 | *(intractable)* | **122 µs** | — |

Exact GED grows ≈ **5× per added node** near n = 12; encoding ≈ 1.15× per node.
`codebase-pointers.md`'s "the canonical encoder is the bottleneck" was true under pure Python and
is **no longer true**.

---

## 2. Re-run cost with the C++ engine

| Stage | Core-hours | On 64 cores |
|---|---|---|
| All encoding, 4 algorithms, 5 datasets | < 0.01 | seconds |
| Levenshtein, 3.9 M pairs | 1–2 | ~2 min |
| WL kernel (not accelerated) | 2–4 | ~5 min |
| IAM exact GED from scratch, n ≤ 9 | ~13 | ~12 min |
| Graph-level bootstrap + Mantel | 4–8 | ~10 min |
| **Reproduce the submitted study** | **~20–27** | **< 1 h** |

The `2-00:00:00` limit at `config.yaml:42` was sized for pure Python and is now a ~50×
over-provision. **All new compute is GED.**

---

## 3. Data and GED strategy

### 3.1 Principle

Two regimes, reported separately and never mixed:

| Regime | Reference | Role |
|---|---|---|
| **n ≤ 12** (Suite 1) | **exact GED** — GEDLIB `ANCHOR_AWARE_GED`, one cost model | ground truth + **calibration anchor** |
| **n > 12** (Suite 2) | **proven bracket** — GEDLIB `BRANCH_FAST` (lower) and `IPFP` (upper), calibrated on the regime above | size-scaling evidence |

Final post-filter suite composition and counts: **`data.md` §0**. Method assignment: **§7.3**.

### 3.2 Splits are merged (decision 3)

Each dataset becomes a single pool. GED is a symmetric function of two graphs with no train/test
semantics; the correlation study is a global measurement. Consequently:

**Recompute *all* GED ourselves, for every dataset, under one stated cost model.** Do not mix our
values with GraphEdX's.

- retires **R3.5b** outright — the heterogeneous-cost objection disappears rather than being
  caveated;
- retires **F2** — no split-shaped holes;
- gives LINUX 3,916 pairs (from 1,685, **2.3×**) and AIDS 295,296 (from 131,148, **2.25×**), on the
  two datasets where ρ is weakest;

  > ⚠ **Do not "fix" 131,148 to F2's 181,909** (`third-auditor.md` §2). `audit-2026-08-11b` I-11
  > reports the two as contradictory and offers **1.62×**; they count **different populations** and
  > 1.62× would be the population-mismatched ratio — MF1's own defect class. Exactly:
  > `C(769,2) = 295,296` is on the **769 filtered** graphs, while
  > `C(546,2)+C(182,2)+C(183,2) = 181,909` is within-split on the **911 raw** graphs. The
  > population-matched comparator is within-split-and-filtered, ≈ 129,600 under proportional
  > retention — **within 1.2 % of 131,148**. I-11 is downgraded to *provenance not recorded*: record
  > the source when T-03 reproduces the run, and print 2.25×.
- **keeps GraphEdX as a validation arm**: recompute ~500 within-split AIDS pairs and assert exact
  agreement. If they disagree, the cost models differ and everything downstream is suspect. **This
  check runs first and gates T-03.**

| GED job | Pairs | Core-hours | On 64 cores |
|---|---|---|---|
| IAM LOW + MED + HIGH (n ≤ 9) | ~3.6 M | ~13 | 12 min |
| LINUX all-pairs (n ≤ 10) | 3,916 | ~2.4 | 3 min |
| AIDS all-pairs (n ≤ 12) | 295,296 | **1,000–1,600** | **16–26 h** |
| **Total** | | **~1,020–1,620** | **~17–26 h** |

One `cpu` job, 64–128 cores, `1-00:00:00`, checkpointing (`ged_computer.py` already does).
Write it with the **`picasso-sbatch`** skill.

### 3.3 Approximate GED above n = 12 (decision 4) — and the calibration that makes it rigorous

**I agree this beats the controlled-edit cohort**: constructed pairs live in a `k`-ball around a
base graph and are not distributed like real pairs, which is exactly the objection R3 would raise.
Real datasets with a calibrated approximation is the stronger design.

**But it is confounded unless calibrated, and the confound is fatal if missed.** Correlating
Levenshtein against an *approximate* GED at large n mixes two effects: how well Levenshtein tracks
true GED, and how well the approximation tracks true GED. Bipartite GED's error is known to grow
with graph size, so a declining ρ at large n would be uninterpretable. Mandatory protocol:

1. **On n ≤ 12, where exact GED exists**, report all three:
   ρ(Lev, GED_exact), ρ(Lev, GED_approx), ρ(GED_approx, GED_exact),
   plus the mean relative overestimate of the approximation.
2. **State the calibration in the paper.** If ρ(GED_approx, GED_exact) is high and
   ρ(Lev, GED_approx) ≈ ρ(Lev, GED_exact) on the same pairs, the approximation is a validated
   stand-in and the extension to n > 12 is defensible. If not, we report the exact-GED result and
   say the extension is not supportable — that is a legitimate outcome.
3. **Above n = 12**, report ρ(Lev, GED_approx) with the calibration quoted alongside every number.
4. **Bracket it.** Report the **proven** bracket `LB ≤ GED ≤ UB`. Where **LB = UB the value is
   exact**, certified for free, and the certification rate is itself reportable (measured 9.8–11.3 %
   with a plain BP; expect more with `IPFP`).

> ~~**Implementation: write Riesen–Bunke BP ourselves, do not vendor.**~~ **SUPERSEDED 2026-08-11.**
> GEDLIB builds and runs on Picasso (`data.md` §7.5), so the bounds now come from the **reference
> implementation by the authors of the bound we cite**, which is materially more defensible than our
> own 150-line version — and our plain BP measured **+78 % overestimate**, the loosest member of its
> family. **See §7.3 for the locked method assignment.** `scratchpad/ged_bounds.py` is retained
> permanently as an *independent cross-check*, not as a reported source.

References to cite: Blumenthal & Gamper, ***IEEE TKDE*** 30(3):503–516, 2018 (BRANCH / BRANCH-FAST —
**our lower bound**); Bougleux et al., 2017 (IPFP — **our upper bound**); Riesen & Bunke,
*Image and Vision Computing* 27(7):950–959, 2009 (BIPARTITE — the reference point);
Fischer et al., ***Pattern Recognition*** 48(2):331–343, 2015 (Hausdorff GED — venue fit for EiC.b);
Zeng et al., *VLDB* 2009 (STAR — **already in our bibliography**, fixes the `methodology.tex:803`
citation group); Blumenthal et al., GbRPR 2019 (GEDLIB itself); Jain et al., NeurIPS 2024
(already cited).

### 3.4 New datasets — where I diverge from the suggestion

**`cs.cornell.edu/~arb/data/` is Austin Benson's higher-order repository** — hypergraphs, simplicial
complexes, temporal networks. That is **IsalHG's** domain, not IsalGraph's, and a simple-graph paper
drawing from it would look like a dataset grab.

**The extension is the IAM Graph Database itself** (Riesen & Bunke, SSPR 2008) — we already use IAM
Letter, it is *the* pattern-recognition GED benchmark, and it ships published edit costs.

**LOCKED, from measured counts** (`data.md` §0 — filter `min_nodes = 2`, connected):

| Added dataset | kept | n̄ | n max | density | Why |
|---|---:|---:|---:|---:|---|
| **GREC** | 650 | 11.45 | 24 | 0.244 | symbol recognition; its 59.1 % retention is **size-unbiased**, the cleanest discard in the cohort |
| **AIDS (IAM)** | 1,811 | 14.02 | 85 | 0.202 | the full IAM version, far richer than the GraphEdX subset |
| **COIL-DEL** | 7,200 | 21.48 | 79 | **0.328** | 100 % connected; the **density** stress test |
| **Mutagenicity** | 4,040 | 28.53 | **98** | **0.094** | molecular, sparse, largest corpus |
| **Protein** | 569 | **31.68** | 96 | 0.163 | largest mean size, different domain |

**TUDataset (MUTAG, IMDB-BINARY, PROTEINS) is dropped** — the IAM family already spans n̄ = 4 → 32
and density 0.09 → 0.61, the cohort reaches **n = 98 retained** (an 8.2× extension of the submitted
12-node ceiling), and staying inside one benchmark family with published edit costs is far easier to
defend than mixing sources.

> ⚠ **Rationale corrected 2026-08-11 (`gap-audit.md` MF1).** This previously read "Mutagenicity
> reaches n = 417 raw". Verified: **the 417-node Mutagenicity graph is disconnected**, so
> `filter_graphs` discards it and it never enters the study. Quoting a raw-set maximum to justify a
> cohort defined on the connected subset is the same category error `data.md` §2.2.1 warns about for
> `n̄`.
>
> The decision most likely survives — 98 versus 12 is still an 8.2× extension, on ten datasets with
> published edit costs — but **it must be re-affirmed on the corrected number** (`data.md` Q9), and
> the residual objection acknowledged in the paper: real-world machine-learning graphs, which is
> what AE.1 actually asks about, are routinely far larger than 98 nodes. The honest framing is §3.5:
> the encoder has no 12-node ceiling and we demonstrate it to 98; **exact GED has one, and that is a
> constraint on the field rather than on this work.**
**Dropped from IAM**: COIL-RAG (n̄ = 3.0, density 0.93), Fingerprint (51.4 % retention, n̄ = 5.5),
Web (different XML schema, does not parse).

Everything above is **measured, not quoted** — connectivity retention, density and encoding cost
were all verified before committing (`data.md` §§2, 4). Two residual items:

- **pair subsampling** — `BRANCH_FAST`/`IPFP` cost ~100 µs/pair at n̄ = 30, so all 40 M pairs cost
  ≈ 1.3 core-hours and **no subsampling is needed**;
- **the connectivity discard is size-biased** on Mutagenicity, Protein and AIDS-IAM
  (`data.md` §2.2.1) — report retained *and* discarded statistics.

### 3.5 What we say about size

1. **IsalGraph encoding has no ~12-node ceiling** — measured: 122 µs at n = 20, 3.9 ms at n̄ = 32
   (Protein), no timeout to n = 96. The locked cohort carries it to **n = 98**.
2. **Exact GED does** — measured, 36.9 s/pair at n = 12, ×5 per node. No public benchmark supplies
   exact GED beyond this; GraphEdX stops there for the same reason. A constraint on the field.
3. **Above n = 12 the reference is a proven bracket**, `BRANCH_FAST ≤ GED ≤ IPFP`, with its
   agreement to exact GED calibrated on the regime below and quoted alongside every number.
   Levenshtein is correlated against **both ends separately** — no interpolation (§7.3).
4. **Canonicalisation does not fail gracefully** — cost is governed by |Aut(G)|, not size or
   density (`data.md` §4.4). State the characterised worst case, not an unqualified "exponential".

---

## 4. Competitors — architecture and placement

### 4.1 Vendoring: follow IsalHG's `iso_backends`

`IsalHG/src/isalhg/iso_backends/` is the model: an ABC (`base.py`), a **lazy registry** keyed by
name (`registry.py`, with `_LAZY_MODULES` so optional deps import only on request), a
`subprocess_base.py` for external binaries, and `BackendUnavailableError` on failure. This is the
same idiom as IsalGraph's existing `core/backends.py` (`BackendError`, never degrade silently).

**Proposed: `src/isalgraph/competitors/`** with **two** protocols, because IsalHG's `IsoBackend`
answers a different question (fingerprint / are-isomorphic) than we need here:

| Protocol | Methods | Implementations |
|---|---|---|
| `ReprBackend` | `encode(G) -> str\|bytes`, `bit_length(G) -> int`, `distance(a, b) -> float` | graph6, sparse6, nauty-canonical graph6, AGM code, **gSpan min-DFS code**, IsalGraph |
| `GEDBackend` | `ged(G, H) -> float`, `kind: 'exact'\|'upper'\|'lower'` | **GEDLIB** `ANCHOR_AWARE_GED` (exact), `BRANCH_FAST` (lower), `IPFP` (upper), `BIPARTITE` (reference point); `networkx` A* and `ged_bounds.py` as cross-checks — see **§7.3** |

Reuse IsalHG's `IsoBackend` shape verbatim for **nauty / bliss / Traces** — we need canonical
relabelling anyway, to make the graph6 comparison *fair* rather than a strawman.

### 4.2 Competitor set

| Competitor | Reversible | Canonical | String | **Distance** | Effort |
|---|---|---|---|---|---|
| graph6 | yes | only if relabelled | yes | **decided by T-04a** | hours (`nx.to_graph6_bytes`) |
| **sparse6** | yes | only if relabelled | yes | **decided by T-04a** | hours |
| **nauty** canonical labelling | yes | **yes** | via graph6 | **decided by T-04a** | 1 d (`pynauty`, IsalHG has the pattern) |
| bliss / Traces | yes | yes | via graph6 | **decided by T-04a** | 0.5 d each, reuse IsalHG |
| adjacency matrix | yes | no | no | **decided by T-04a** | trivial |
| AGM canonical code | yes | yes | yes | Levenshtein | 1 d, derive from nauty labelling |
| **gSpan minimum DFS code** | yes | **yes** | **yes** | **Levenshtein** | **2–3 d**, vendor `LasseRegin/gSpan` |
| WL subtree kernel | no | — | no | kernel | already computed |

**gSpan's minimum DFS code remains the single most important comparator**: canonical, a string,
edit-distance-comparable, named by R1, same problem setting.

#### The distance column is measured, not asserted — see `competitors.md`

The audit found that **plain Hamming is undefined for most pairs** (`gap-audit.md` MF6): graph6
encodes `n` in its header and packs `n(n−1)/2` bits, so graphs with different node counts give
strings of different length. Node counts run 2–12 in Suite 1 and 2–98 in Suite 2, so equal-`n` pairs
are a small minority, and "Hamming on graph6 correlates poorly" would have recorded an artefact of
undefinedness as a finding — inside the comparison the Area Editor explicitly endorsed.

The first fix proposed a padding convention. **Withdrawn — that was decided by argument when it can
be decided by measurement**, and a distance chosen after seeing which choice flatters IsalGraph is
not a distance we can defend.

> **Locked: T-04a.** Every (representation × distance) cell is attempted on a fixed stratified sample
> of 200 graphs / 19,900 pairs and scored on six criteria — well-defined, metric axioms,
> isomorphism-invariance, non-degeneracy, correlation with exact GED, and cost. The primary distance
> for each representation is **the cheapest that is well-defined, metric, invariant and
> non-degenerate**, with ties broken on cost and **never on correlation with GED**. Full protocol,
> selection rule and the two pre-committed outcomes in **`competitors.md`** §2–§3.

**MF7 — sparse6 is the head-to-head competitor for Claim A, not a make-weight.** sparse6 exists
specifically to encode **sparse** graphs compactly, and that is exactly where IsalGraph claims its
compactness advantage. Two consequences, both locked in `competitors.md` §3–§4: **both bit
conventions** (entropy bound and realised bytes) are reported for every method, because
`B_Isal(w) = L log₂ 9` is an entropy bound and graph6/sparse6 emit bytes; and **a losing result is
pre-committed as publishable** — if sparse6 wins on bits for sparse graphs, the contribution is
stated as *canonical **and** edit-distance-compatible* rather than *shortest*.

**Risk on the vendored gSpan**: `LasseRegin/gSpan` is a *frequent-subgraph miner*. We need the
**minimum DFS code of one graph**, which is an internal sub-component and may not be exposed. If it
is not, extract or reimplement it — budget the same 2–3 days and verify on day 1 of T-04.

### 4.3 Where each experiment gains columns

| Experiment | Gains | Retires |
|---|---|---|
| **(a) Message length**, §3.2.3 / Table 2 / Fig 1 | bit cost for graph6, nauty-graph6, adjacency, AGM, min-DFS | **R3.6a** — we stop calling our own model "standard" and put real serializations beside it |
| **(b) GED proxy**, §3.2.5 / Table 3 / Fig 3 | ρ for Levenshtein-on-min-DFS, Hamming-on-nauty-graph6, WL | **R1.1** (proxy half) |
| **(c) Runtime**, §4.2 / Fig 2 | encode-time curves for min-DFS and nauty | **R1.1** + **D16** — the per-graph/per-pair category error |
| **(d) [28] / [29] delta** | **conceptual table only — no experiment** | R3.1 / AE.3 / R3.7b |

Building an experiment for (d) would be a category error: it asks what we borrowed from our own
prior work, which is answered by reading the sources.

*Stated in advance*: Hamming on non-canonical graph6 should correlate **poorly**, because bit
position is not edit-aligned. That is an informative result isolating why canonical **and**
edit-distance-compatible is the contribution. Report it either way.

---

## 5. Prior-work and bibliography

### 5.1 [29] IsalChem — **published; read the paper, not just the code**

> **CORRECTED 2026-08-12** (`third-auditor.md` X-1). This section previously read "the paper is
> unavailable" and sent T-07 to infer [29]'s contents from source. That contradicts **decision 9**,
> corrected in the same revision and not propagated here.

[29] is **published and already cited**: `ThurnhoferHemsi:2025` — Thurnhofer-Hemsi, García-Aguilar,
Fernández-Rodríguez, López-Rubio, *Representation of Molecules by Sequences of Instructions*,
**J. Chem. Inf. Model. 65(15):7936–7955, 2025** (entry verified in `cas-refs.bib`).

Consequences for T-07: the inherited/modified/new table is written from the **paper**, with
`github.com/icai-uma/IsalChem` as the implementation cross-check rather than the primary source; and
**D19's [29] half — the LSTM experiment — is directly resolvable** by reading it instead of inferred
from the presence of a training script. T-07's 1–4 day range was budgeted partly on the missing-paper
premise and should come down accordingly.

### 5.2 [28] — the preprint

PDF in-repo at `docs/references/2512_10429v2.pdf`. Read it to confirm the Transformer-classification
claim (D19's other half) and to write the delta table.

### 5.3 [28] is permanently arXiv-only (decision 10)

EiC.a's "substitute arXiv citations with their peer-reviewed versions" **cannot be satisfied** for
[28]. Response: state it plainly in one sentence. Then reduce the *visible* arXiv footprint —
**strip the `note = {arXiv:...}` fields from the five entries that already name ICLR / NeurIPS
venues** (`kipf2017gcn`, `velickovic2018gat`, `xu2019powerful`, `fey2019pyg`, `jain2024graphedx`).
That takes the rendered arXiv count from **6 to 1**.

### 5.4 Bibliography budget

43 cited → 55 ceiling = **12 slots**.

| Purpose | Slots |
|---|---|
| AGM, gSpan, nauty/Traces, bliss, graph6, Babai | 5–6 |
| GED approximation: Riesen–Bunke 2009, Fischer 2015 (*Pattern Recognition*), Blumenthal 2020 | 3 |
| Datasets: IAM Graph Database (Riesen & Bunke, SSPR 2008) — **one citation, TUDataset dropped** | 1 |
| GEDLIB itself (Blumenthal et al., GbRPR 2019) | 1 |
| Recent (2025–26) pattern-recognition work — **weakest current position: nothing third-party after 2024** | 2–3 |

Over budget by ~2; drop the weakest additions or retire a dead citation. Also fix the one
uncommented group `\cite{garey1979,Zeng:2009}` (`methodology.tex:803`). The four-way group at
`introduction.tex:31` is already individually commented — **do not "fix" it**.

---

## 6. Sequential model — declined (decision 5)

R3.2's modal is the softest in the report ("*would substantially strengthen*", against "should
provide" / "should be narrowed" / "should be described"), so it is framed as an enhancement. A
credible sequence-model study is a paper, not a subsection.

**The decline is only defensible if the claims come down with it.** Non-negotiable:

| Location | Required change |
|---|---|
| `main.tex:122–126` (abstract) | "language-model-compatible … **with direct applications in** graph similarity search, graph generation, graph-conditioned LM" → format compatibility as a **property**; applications as conjecture ("may enable") or dropped |
| `introduction.tex:35–37` | keep as motivation, explicitly not a result |
| `conclusion.tex:76` | already hedged ("can be consumed", "may enable") — leave |
| `conclusion.tex:88–95` | **expand**: name the Transformer/LSTM study as the designated next step, citing [28] and [29] as templates |
| Limitations | **add R3.7a in substance**: no sequential model and no downstream pattern-recognition task is evaluated |

**Residual risk**: R3 may hold the line in round 2. Mitigation is §4 — we chose the comparison the
Area Editor endorsed (AE.3) over the experiment one reviewer suggested, and the letter should frame
it as exactly that exchange.

### 6.1 The decline is undercut by our own R3.1 table — `gap-audit.md` MF10

**T-07 produces a table documenting that [28] contains a Transformer classification experiment and
[29] contains an LSTM experiment.** That table is the deliverable for R3.1, R3.7b and AE.3. It is
also, in our own words, the clearest possible statement that **both predecessors evaluated a sequence
model and this paper does not** — which is R3.2's exact argument. Nothing in the plan noticed the
interaction.

Two required consequences:

1. **Pre-empt the reading in the delta table itself.** The row exists either way; write it as a
   stated scope decision — *sequence-model evaluation: present in [28] and [29] on their respective
   domains; deliberately out of scope here, where the contribution is the canonicalisation result;
   designated as the next study* — rather than leaving the reader to notice the gap.
2. **A contingency with a date, not a risk paragraph.**

> **Go/no-go: 2026-08-22.** If T-03 has finished and the critical path has slack, run a **minimal**
> sequence-model arm. Otherwise the decline stands and nothing further is discussed.
>
> > **Condition tightened 2026-08-12** (`third-auditor.md` §10). §7's banner establishes there is
> > **no slack** — the critical path is 27.5–28.0 days minimum in a 19-day window — so as written
> > this go/no-go is unreachable and would spend a decision cycle on day 11 concluding what §7
> > already knows. **The contingency is live only if S-f's extension request is granted.** If it is
> > not, the decline stands and 08-22 is struck from the calendar. This costs nothing to decide now
> > and removes a phantom gate from the schedule.
>
> **Fixed minimal scope, decided now so it cannot expand**: character-level Transformer, ≤ 2 M
> parameters, on canonical strings, **graph classification only**, on the datasets that already
> carry class labels — Letter (15 classes), GREC (22), Mutagenicity (2), Protein (6), AIDS (2),
> COIL-DEL (100). Baselines: WL subtree kernel (already computed, E10) and the same model on the
> **competitor** strings from T-04 — which makes it a *representation* comparison rather than a weak
> claim about Transformers. Fixed splits, one seed set, no architecture search, no tuning beyond a
> learning-rate sweep. Reported as a **feasibility demonstration**, explicitly not as a benchmark
> result.
>
> Every ingredient exists: canonical strings for 19,670 graphs across ten labeled datasets, and an
> RTX 4060 trains a model this size on ~10⁴ short strings in minutes. The reason to decline is time,
> not capability, and the contingency exists so that if time appears we can spend it.

---

## 7. Ticket board

| ID | Ticket | Depends | Days | Pri |
|---|---|---|---|---|
| **T-01** | **Data lock**: size/density/connectivity audit tables (**retained *and* discarded**, per `data.md` §2.2.1); `n_max = 12` **retained for Suite 1, dropped for Suite 2**; merge splits; define cohorts; port `audit_recheck.py` into `tests/` | — | 1–2 | **P0** |
| ~~T-01b~~ | ~~New-dataset audit~~ — **DONE 2026-08-11.** IAM Database extracted and audited; n, density, connectivity retention and encoding cost all measured; cohort locked (`data.md` §0, §2.3). Residual: port the scratchpad scripts into `benchmarks/` | T-01 | — | done |
| **T-02** | **Statistics lock**: §8; graph-level bootstrap; Mantel; pair-accounting ladder | T-01 | 2–4 | **P0** |
| **T-03** | **Exact-GED job on Picasso** — full spec in §7.1 | T-01 | 3–8 | **P0 — long pole** |
| **T-04** | **Competitor backends**: `src/isalgraph/competitors/` in the IsalHG idiom; graph6, nauty, bliss/Traces, AGM, **gSpan min-DFS** | — | 3–8 | **P0** |
| **T-05** | **Bounded GED via GEDLIB** (§7.3): wire `BRANCH_FAST` + `IPFP` on Picasso, pass the three validation gates, run the **calibration arm at n ≤ 12**, then all 40 M Suite-2 pairs (~1.3 core-h) | T-01b, T-03 | 5–10 | **P0** |
| **T-06** | **Full recompute**: all experiments, C++ engine, new cohorts, competitor columns, new statistics | T-02..T-05 | 10–14 | **P0** |
| **T-07** | **Read [28] PDF + [29] source**; inherited/modified/new table; resolve D19 | — | 1–4 | **P0** |
| **T-08** | **Related work section** + bibliography to ≤55 (§5.4) | T-07 | 4–10 | P1 |
| **T-09** | **Explanatory figures** (merged, was T-09 + T-10): the **S2G/G2S worked example** via `isalgraph.viz`, and the **canonical search-space schematic** — renderer exists, `viz/search_tree.py::canonical_search_tree_figure` (verified present). **Both are regenerated as the new graphical abstract**, which is submitted separately and therefore costs no manuscript pages | — | 1.5 | P1 |
| ~~T-10~~ | ~~Canonical search-space schematic~~ — **merged into T-09** | — | — | — |
| **T-11** | **Manuscript errors** (§9) | — | 2 | P1 |
| **T-12** | **Claim scoping** (§10) | T-06 | 2 | P1 |
| **T-13** | **Complexity section**: `P(M)` recomputation, four costed operations, three-way separation | — | 2 | P1 |
| **T-14** | **Response letter** | all | 3 | **P0** |
| **T-15** | **Page trim to 35** + supplementary + AI declaration | all | 2 | **P0** |

| ~~T-16~~ | ~~`wl_pruned_canonical` C++ variant~~ — **REJECTED 2026-08-11 (author).** No reviewer or editor asked for it; it originated as a transfer from IsalSR / IsalHG. The **WL measurement** it was motivated by is retained inside **T-13**, where it answers R3.7d. Rationale in §7.2 | — | — | — |

### Tickets added by the 2026-08-11 coverage audit (`gap-audit.md`)

| ID | Ticket | Depends | Days | Pri |
|---|---|---|---|---|
| **T-17** | **AE.3 comparison table as a paper artifact** — existing graph representations vs IsalGraph, with **properties, strengths and limitations of each**, on R1.2's five axes (uniqueness, expressiveness, computational efficiency, scalability, **downstream learning = not evaluated**). Rows populated from T-04's measurements, not asserted. Licenses the softening of `introduction.tex:33` / `conclusion.tex:74` | T-04, T-07 | 2–3 | **P0** — the AE endorsed this one in their own voice |
| **T-04a** | **Metric feasibility** — attempt every (representation × distance) cell on a fixed 200-graph / 19,900-pair stratified sample; select each primary distance by the pre-declared rule. **Must close before any production distance matrix is computed.** Protocol in **`competitors.md`** §2 | T-04 | 0.5–1 | **P0** |
| **T-18** | **Labels** — **tiered, PI decision on effort due 2026-08-18** (`labels.md` §0). Tier 0 (rebuttal, label column, **E6 fix**, future work) is not optional; Tier 1 (topological collision count) recommended; Tier 2 (label-aware GED **logged, not written up**) proposed at 0.3 core-hours; Tier 3 (a results subsection) declined by default | T-05, T-06 | 0.3–1 | **P0** (Tier 0) / P2 (Tier 2) |
| **T-19** | **Bibliography recency and venue audit** — classify all 43 existing references by venue and year; add **≥ 6 from 2025–2026** in pattern recognition, graph matching or graph representation. Acceptance: no year gap after 2024, and a stated PR-community share. Fischer 2015 counts for **venue**, not recency | T-08 | 1–2 | **P0** — EiC checks independently |
| **T-20** | **Manuscript rewrite** — §3.1, §3.2, §3.3, §4, §5, abstract. Section map in **`manuscript.md`** §1. This is the largest single writing task and had no owner | T-06 | 5–7 | **P0** |
| **T-21** | **Implementation, reproducibility and artifact release** — C++ engine and GEDLIB in §3.3; library versions; `-march=x86-64-v3` and the non-rsyncing `.so`; data-availability statement; updated public artifact | T-06 | 1–2 | P1 |
| **T-22** | **Formal-statement audit** — restate Thm 2.12 within a fixed directedness class, move the flag hypothesis from proof to statement, **re-verify all three proof steps**, propagate to **Cor. 2.13** (which `statistics.md` D6 justification 1 leans on), add the directedness-collision regression to `tests/property/` | — | 1–2 | **P0** |
| **T-23** | **Clear the Picasso `fscratch` file-count quota** — 305.8k against a 250k soft quota, 400k hard, **grace expires ≈ 2026-08-18**. T-03 checkpoints frequently and fails partway if it hits the limit | — | 0.5 | **P0 — blocks T-03** |
| **T-24** | **Submission package and Elsevier compliance** — LaTeX source files; AI declaration; author biographies and photos; acknowledgements; highlights; graphical abstract (**fix the `graphical_abtract.pdf` filename**); competing-interest and data-availability statements. Checklist in `manuscript.md` §5 | T-15 | 1 | **P0** |

### Tickets added by the 2026-08-11b over-scope and integrity audit (`.claude/notes/audit-2026-08-11b/`)

| ID | Ticket | Depends | Days | Pri |
|---|---|---|---|---|
| **T-25** | **Restore validation gate 2, or retire it on the record** — `scratchpad/ged_bounds.py` and 12 other measurement scripts **do not exist** (I-01, measured). Gate 2 of three (`§7.3`) is currently unexecutable and it gates **T-03 production**. Also re-establishes the evidence for the "BRANCH-FAST is the primary large-`n` reference" decision (§11 item 4), which is presently unreproducible. Option set and decision in **S-e** | — | **0.1–0.2** (S-e's recommended path C+B; **0.5–1 if the PI chooses option A**, rewriting the script) | **P0 — blocks T-03** |
| **T-26** | **Bibliography-slot and page-budget reconciliation** — the two compliance arithmetics the EiC checks independently, neither of which any existing ticket owns end to end. (a) Reconcile reference allocations: **16–17 requested against 12 slots**, with "retire a dead entry" providing **zero** relief; decide what is cut from the text or accept 55 as the ceiling and re-plan T-19's additions. (b) Re-derive `manuscript.md`'s page budget as **deltas, not gross sizes**, separating movable inventory rows from immovable main-text prose. **Must run after T-08 and T-19 and before T-15** — §12.1 currently has no such gate. Decision in **S-h** | T-08, T-19 | 0.5 | **P0 — EiC pass/fail** |

**Critical path**: **T-23** → T-01 → T-03/T-05 → T-06 → **T-20** → T-15 → T-24, with T-14 accruing
throughout. **T-04 → T-04a → T-17**, T-07 → T-08 → T-19, T-22 and T-13 run in parallel off it.
T-04a gates T-06's distance matrices, so it is on the path for everything downstream of the
competitors even though it is half a day. **T-25 joins T-23 as a day-1 gate on T-03.**

> ### ⚠ The board does not fit the window — audit-2026-08-11b, I-06 / I-07, measured
>
> **§12's "76.5 days" is the v0.5 board.** Re-summing §7's own Days column: T-01…T-15 (excluding
> merged T-10) = 72.5, **+ T-16 (3–4 d) = 76.5** — the quoted figure, which predates both v0.6
> (T-17…T-24) and v0.7 (T-16 rejected), yet is attributed to the audit that created those tickets.
> **The current board is 93.5 days at upper bounds, 54.8 at lower** — understated by 17.0 days
> (22 %). §7's "T-16 removes 3–4 days" and §12.3's relief both read against the wrong baseline: the
> board grew **17.0 days net** in the same revision that returned 3–4.
>
> > **Corrected again 2026-08-12** (`third-auditor.md` §4). `audit-2026-08-11b` published **91.0 /
> > 52.8**, which sums T-01…**T-23** and stops — omitting **T-24 (1 d)**, **T-25 (0.5–1 d)** and
> > **T-26 (0.5 d)**, i.e. the submission package *and the two tickets that audit created in the same
> > revision*. That is I-06's own failure mode. Re-parsed programmatically from §7's Days columns the
> > board is **93.5 upper / 54.8 lower**. Quote these.
>
> **Worse, the declared critical path is serial and does not fit.** T-23 → T-01 → T-03 → T-05 →
> T-06 → T-20 → T-15 → T-24 sums to **27.5 days at lower bounds** (44.5 at upper) against a
> **20-day window** — **28.0 with T-25**, which the same audit makes a day-1 gate on T-03. §12's
> mitigation — "survivable only because most tickets parallelise" — does not apply to a critical
> path, by definition. **The largest single lever on this path is §7.1's two-stage T-03**: it takes
> 985–1,640 core-hours of AIDS census off the path without giving the census up.
>
> §12.1 also allocates **below the board's own minima** on the two largest tickets: **T-05** (5–10 d)
> gets Days 5–8 = **4**; **T-06** (10–14 d) gets Days 8–12 = **5**, half its minimum. The other 18
> tickets are at or above minimum. Risk R1 budgets for T-03 slipping; **nothing budgets for T-06
> receiving 5 of the 10–14 days it needs.**
>
> **This is not a scope finding and no ticket fixes it. Decision required — see S-f.**

**T-16 is rejected**, not deferred — see §7.2. That removes 3–4 days of C++ from the board.

### 7.1 T-03 — recover **all** exact GED (author decision, 2026-08-11)

**Scope**: every pair of every connected graph in the five original datasets. No subsampling, no
split structure, no reliance on GraphEdX's within-split coverage.

| Dataset | connected | **all pairs** | ~s/pair | core-hours |
|---|---:|---:|---:|---:|
| Letter LOW | 1,180 | 695,610 | 0.004 | 0.8 |
| Letter MED | 1,253 | 784,378 | 0.004 | 0.9 |
| Letter HIGH | 2,059 | 2,118,711 | 0.008 | 4.7 |
| LINUX | 89 | 3,916 | 2.17 | 2.4 |
| **AIDS (GraphEdX)** | 769 | **295,296** | 12–20 | **985–1,640** |
| **Total** | | **3.90 M** | | **≈ 1,000–1,650** |

**16–26 h on 64 cores.** Counts are **post**-reconciliation — open question 3 is closed (I-22), and
`third-auditor.md` §8 confirms this table uses the correct `KEPT_ge2` population, unlike `data.md`
§3.1 (I-03).

> ### ⚠ Run this in two stages — `third-auditor.md` §1, 2026-08-12
>
> **The census is disproportionate to every demand that drives it, and this plan already contains
> the proof.** `audit-2026-08-11b` cleared T-03 on the cut-guard test — F2 and Cor. 2.13 are genuine
> non-R3.5b drivers, both verified. But a driver fixes the *direction* of the work, not its *dose*,
> and the dose here is 98 % of the compute budget and the whole critical path.
>
> **The internal contradiction, stated plainly.** `statistics.md` **D2** — locked, and the literal
> answer to R3.5c — resamples **graphs**, not pairs, because pairs are dyadically dependent. §12.2
> risk R1 draws the consequence in this plan's own words: "effective sample size is governed by the
> **number of graphs**, so very little power is lost". AIDS contributes **769 graphs** whether we
> compute 131,148 pairs or all 295,296. **If D2 is right, the census buys no precision. If the census
> buys precision, D2 is wrong and R3.5c is unanswered.** Both cannot hold.
>
> What the census *does* buy, fairly: a pair-accounting ladder that says "all pairs" rather than "a
> stratified sample" (rhetorically stronger against R3.5a); no sampling design to defend; and more
> pairs inside each §8 density stratum — though the independent-unit count per stratum is again the
> number of graphs in it.
>
> **Structure that keeps both.** Do not choose between census and subsample — *order* them:
>
> | Stage | Content | Cost | Role |
> |---|---|---|---|
> | **1** | Stratified sample over **all 769 AIDS graphs**, spanning every density and size stratum §8 needs | **~100 core-h**, hours | **Pre-declared as the reported analysis.** Unblocks T-06 |
> | **2** | The full 295,296-pair census, submitted at the same time, unattended | 985–1,640 core-h | **Supersedes stage 1 only if it lands before the T-20 text freeze** |
>
> The supersession rule is fixed **now**, in advance, so the choice between the two ρ values cannot
> be made after seeing which is more favourable. Letter wording either way: "GED was recomputed under
> a single unit cost model over a stratified sample of N pairs spanning all 769 graphs / over all
> 295,296 pairs."
>
> **Why this beats S-f's options A and D.** A gives the census up; D keeps it on the critical path
> and absorbs a 7.5-day overrun. Staging removes T-03 from the path *and* keeps the census.
> **Cost: one paragraph of protocol in §3.2. Saving: 2–5 elapsed days on the long pole.**
> Recorded as **S-f option E**.
>
> **Re-cost stage 1 from the table above, not from `data.md` §3.1** — I-03 shows §3.1's Letter rows
> use the n≥1 population and are inflated by 22,698 pairs, and risk R1's "~100 core-h" is costed
> against it.

**Configuration** — fixed:
- cost model: **unit node + unit edge, substitutions free** (`statistics.md` D6)
- **GED timeout: unchanged from the submission** (author decision). Record it explicitly and report
  the censoring rate **per stratum** — censoring is symmetry-correlated, never pool it
- non-computable pairs are **interval-censored `[LB, UB]`**, not dropped (`statistics.md` D11)
- checkpoint every 5,000 pairs (`ged_computer.py` already does)
- write the SLURM script with the **`picasso-sbatch`** skill; `cpu` constraint, 64–128 cores,
  `1-00:00:00`, 128 GB

**Gate — run this first, before the main job**: recompute ~500 *within-split* AIDS pairs under
GraphEdX's own topology-only cost model and assert exact agreement with the published matrix. If
they disagree, our solver or our cost model is wrong and everything downstream is invalid. Only
after this passes does the unit-cost production run start.

> **The gate runs under a different cost model from production, and the configuration must be
> written down** (`gap-audit.md` MF14). GraphEdX charges **zero for node operations**, so the gate is
> `edit_cost_constant=[0, 0, 0, 1, 1, 0]` — **not** the D6 production model `[1, 1, 0, 1, 1, 0]`.
> §7.3's production-assignment table has no row for it. Running the gate under the production model
> produces a guaranteed mismatch that looks exactly like a solver bug, which would burn a day at the
> worst possible point on the critical path.
>
> Note also what the gate does *not* establish: agreement under GraphEdX's pseudometric model does
> not validate our metric model. It validates the **solver**. The cost-model change is justified
> separately, by `statistics.md` D6.

**Expected consequence**: LINUX ρ = 0.433 and AIDS ρ = 0.349 will both change — the pair sets grow
2.3× and 2.25×, and the cost model changes. Every downstream number must be re-derived.

### 7.3 GED implementation, and the two **proven** bounds

Everything below runs through **GEDLIB** (verified working on Picasso, `data.md` §7.5). One library,
one cost model, one provenance chain — this is what makes the numbers defensible to R3.5b.

#### Why proof status matters here

The revision reports GED values above the exact-computable range. If we report a *heuristic*
estimate, a reviewer can ask how far it is from the truth and we have no answer. If we report a
**proven bracket** `LB ≤ GED ≤ UB`, the true value is contained by construction and the only open
question is the bracket's width, which we measure. **Every GED number above n = 12 must therefore
come from a method with a published proof, not from a heuristic that merely performs well.**

The asymmetry is worth stating because it drives the choice:

- **Upper bounds are structurally easy.** Any method that returns a *valid edit path* yields a
  proven upper bound — the path's cost is achievable, and GED is the minimum over all paths, so
  path cost ≥ GED. `BIPARTITE`, `IPFP`, `REFINE`, `BP_BEAM` all construct explicit edit paths.
- **Lower bounds need a theorem.** Each requires a proof that no edit path can be cheaper. Only the
  published families qualify.

#### Exact GED (`n ≤ 12`) — the calibration anchor

| Role | Method | Source |
|---|---|---|
| **Primary** | GEDLIB **`ANCHOR_AWARE_GED`** | Blumenthal & Gamper; exact when run to completion |
| **Cross-check** | `networkx.graph_edit_distance` (A*) | the submitted study's solver |

**Benchmark the two before the production run.** GEDLIB is specialised C++ and `networkx` is
Python; if `ANCHOR_AWARE_GED` is materially faster it **raises the exact-GED ceiling above n = 12**
and enlarges the calibration regime — the single cheapest way to strengthen the whole design. They
must agree exactly on a shared sample; disagreement means a cost-model mismatch and blocks T-03.

#### Proven **lower** bound

| Role | Method | Reference | Complexity | Status |
|---|---|---|---|---|
| **Primary** | **`BRANCH_FAST`** | Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018 | `O(n²Δ² + n³)` | **proven LB; pseudo-metric** |
| Tighter | `BRANCH` | same | `O(n²Δ³ + n³)` | proven LB, strictly ≥ BRANCH_FAST |
| Anytime | `BRANCH_TIGHT` | same | iterative | proven LB at every iteration |
| Reported alternative | **`HED`** | Fischer et al., ***Pattern Recognition*** 48(2):331–343, 2015 | `O(n²)` | proven LB |
| Legacy reference | `STAR` | Zeng et al., *VLDB* 2009 | `O(n³)` | proven LB |

Three reasons `BRANCH_FAST` is primary:

1. **It is the tightest family.** The literature ordering is `BED ≥ LED` and `BED ≥ HED` — the
   branch edit distance dominates the linear and Hausdorff bounds. Tighter lower bound → narrower
   bracket → stronger claim.
2. **Measured**: ρ(exact, BRANCH-FAST) = **0.966** with **−11 %** bias, against ρ(exact, BP) = 0.840
   with **+78 %** bias (`data.md` §5, reproduced across two independent samples). The lower bound is
   the better proxy on our data, which is *not* the intuitive expectation and is why we measured it.
3. **It is a pseudo-metric** on a graph collection (proved in the same paper). Corollary 2.13 claims
   the IsalGraph distance is a metric; validating it against a reference with metric structure is
   coherent in a way that validating against an arbitrary heuristic is not.

Report **`HED`** alongside it: it is a published *Pattern Recognition* result on exactly this
problem, which serves **the venue half** of EiC.b.
`STAR` is worth one sentence because **`Zeng:2009` is already in our bibliography** — it is half of
the uncommented two-key group at `methodology.tex:803` that EiC.a flags, so commenting on it
individually fixes that violation at the same time.

> ⚠ **Corrected 2026-08-11 (`gap-audit.md` MF15).** This previously claimed that reporting `HED`
> serves EiC.b "directly". Fischer et al. is *Pattern Recognition* **48(2), 2015**. EiC.a asks for
> references from "**last and current year**", and our weakest position is that **nothing
> third-party postdates 2024**. A 2015 paper satisfies venue fit and contributes **nothing** to
> recency. Both halves are needed and they are different work — **T-19** owns recency.

#### Proven **upper** bound

| Role | Method | Reference | Status |
|---|---|---|---|
| **Primary** | **`IPFP`** | Bougleux et al., 2017 | proven UB (returns a valid edit path) |
| Refinement | `REFINE` | Zeng et al., 2009 / GEDLIB | proven UB; local search on the assignment |
| **Reference point** | **`BIPARTITE`** | Riesen & Bunke, *IVC* 27(7):950–959, 2009 | proven UB; the canonical, widely-cited baseline |
| Alternative | `BP_BEAM` | Neuhaus & Riesen | proven UB |

`BIPARTITE` is reported because it is the comparator every reader knows, **not** because it is good:
our own implementation of it overestimates by **+78 %**, and it is the loosest member of the family.
`IPFP` and `REFINE` handle node and edge assignment simultaneously rather than sequentially and are
substantially tighter. **Select the primary UB by measured tightness on the calibration set**, with
the criterion fixed in advance: the method minimising mean relative overestimate against exact GED,
subject to costing under 1 ms/pair at n̄ = 30.

#### How the bracket is reported — no interpolation

Above n = 12 we hold `LB ≤ GED ≤ UB` and do **not** know where in the bracket the true value lies.
**Do not report a midpoint or any other interpolation** — it would be an unjustified assumption
sitting under every downstream number.

Instead, **correlate Levenshtein against the lower and upper bounds separately and report both ρ.**

> If ρ(Lev, LB) and ρ(Lev, UB) agree, the conclusion is **robust to wherever the true GED lies in
> the bracket**, and that robustness is stated as the result. If they disagree, the bracket is too
> wide to support a claim at that size and we say so.

This needs no assumption about the bracket's interior, it is trivially explainable to a reviewer,
and a disagreement is itself an informative and publishable outcome. Also report:

- **bracket width** `UB − LB`, absolute and relative, per size and density stratum;
- **certification rate** — the fraction with `LB = UB`, where GED is exact for free (measured
  9.8–11.3 % with our plain BP; expect materially more with `IPFP`).

#### Cost-model configuration

GEDLIB **`CONSTANT`** edit costs, set to `statistics.md` D6: node insert = node delete = 1,
edge insert = edge delete = 1, substitutions free. GEDLIB also ships the published per-dataset IAM
models (`LETTER`, `GREC_1/2`, `PROTEIN`, `FINGERPRINT`, `CHEM_1/2`) — these are available as a
**sensitivity analysis** but must not be primary, since per-dataset costs reintroduce exactly the
heterogeneity R3.5b objects to.

#### Which GEDLIB implementation runs which computation — verified on Picasso 2026-08-11

Smoke test `scratchpad/gedlib_api.py`, P₄ (path) vs C₄ (cycle), unit costs, true GED = 1:

| GEDLIB method | `get_lower_bound()` | `get_upper_bound()` | runtime | **capability** |
|---|---:|---:|---:|---|
| `ANCHOR_AWARE_GED` | **1.00** | **1.00** | 0.72 ms | **exact** (LB = UB certifies optimality) |
| **`BRANCH_FAST`** | **1.00** | 1.00 | 0.20 ms | **LB** (+ incidental UB) |
| `BRANCH` | 1.00 | 1.00 | 0.19 ms | LB, tighter, costlier |
| `BRANCH_TIGHT` | 1.00 | 1.00 | 0.55 ms | LB, anytime |
| `STAR` | 1.00 | 1.00 | 0.09 ms | LB (Zeng et al. 2009) |
| `BIPARTITE` | **0.00** | **1.00** | 0.20 ms | **UB only** |
| **`IPFP`** | **0.00** | **1.00** | 0.33 ms | **UB only** |
| `REFINE` | 0.00 | 1.00 | 0.35 ms | UB only |
| `BP_BEAM` | 0.00 | 1.00 | 0.89 ms | UB only |
| `HED` | 0.00 | **inf** | 0.20 ms | **investigate before use** |

> **TRAP — this would silently corrupt an entire GED matrix.** Upper-bound methods return
> **`get_lower_bound() = 0.00`**, and `HED` returns **`get_upper_bound() = inf`**. These are not
> errors and nothing warns you. **Read `get_lower_bound()` only from `BRANCH*`/`STAR`/
> `ANCHOR_AWARE_GED`, and `get_upper_bound()` only from `BIPARTITE`/`IPFP`/`REFINE`/`BP_BEAM`/
> `ANCHOR_AWARE_GED`.** Assert `0 < value < inf` on every read.

**Production assignment:**

| Computation | Suite | Method | Accessor |
|---|---|---|---|
| **Exact GED** — primary reference and calibration anchor | 1 (`n ≤ 12`) | **`ANCHOR_AWARE_GED`** | both; assert `LB == UB` |
| Exact GED — independent cross-check | 1, sample only | `networkx.graph_edit_distance` | — |
| **Proven lower bound** | 1 (calibration) + 2 (all) | **`BRANCH_FAST`** | `get_lower_bound()` |
| Lower bound — tightening, if the bracket is too wide | 2 | `BRANCH_TIGHT` | `get_lower_bound()` |
| **Proven upper bound** | 1 (calibration) + 2 (all) | **`IPFP`** | `get_upper_bound()` |
| Upper bound — literature reference point | 1 + 2 | `BIPARTITE` (Riesen–Bunke) | `get_upper_bound()` |
| Upper bound — refinement arm | 2, if `IPFP` is loose | `REFINE` | `get_upper_bound()` |
| Independent re-implementation cross-check | 1, 300–400 pairs | `scratchpad/ged_bounds.py` | — |

`HED` was earmarked as a *Pattern Recognition*-venue lower bound (Fischer et al. 2015) for EiC.b.
It returned `LB = 0, UB = inf` under the default options, so **it is not usable until that is
diagnosed** — most likely it needs explicit method options. Cite it in the related-work discussion
regardless; only report numbers from it if the accessor issue resolves.

**Invocation pattern** (`gedlibpy_gxl`, verified):

```python
env = gedlibpy_gxl.GEDEnvGXL()
i0 = env.add_nx_graph(g0, "")          # or env.load_GXL_graphs(folder, collection)
i1 = env.add_nx_graph(g1, "")
env.set_edit_cost("CONSTANT", edit_cost_constant=[1, 1, 0, 1, 1, 0])
#                  order: [node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]
env.init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")
env.set_method("BRANCH_FAST", ""); env.init_method()
env.run_method(i0, i1)
lb = env.get_lower_bound(i0, i1)       # valid for BRANCH*/STAR/ANCHOR_AWARE only
```

`edit_cost_constant=[1, 1, 0, 1, 1, 0]` **is** `statistics.md` D6 — unit insert/delete on nodes and
edges, free substitution. `add_nx_graph` requires string-valued node and edge attributes, so attach
a constant dummy label before adding (topology-only is what we want anyway).

#### Validation gates — all three must pass before T-03 production

1. **Bracket validity**: `LB ≤ exact ≤ UB` on every calibration pair. Our own implementation gave
   **0 violations in 400 pairs**; GEDLIB must match. A single violation is a cost-model mismatch.
2. **Cross-implementation agreement**: GEDLIB's `BRANCH_FAST` and `BIPARTITE` must reproduce
   `scratchpad/ged_bounds.py` on the same 300–400 pairs. Disagreement is a bug in one of them and we
   need to know which before either is trusted.

   > ### 🚫 GATE 2 IS NOT EXECUTABLE — `ged_bounds.py` DOES NOT EXIST
   >
   > **audit-2026-08-11b, I-01 (blocking), MEASURED**: `find / -name 'ged_bounds.py'` returns
   > nothing. It was never committed. **13 of the 16 scripts `data.md` §8 names are gone** —
   > `size_audit.py`, `graphedx_audit.py`, `real_cost.py`, `iam_audit.py`, `feas2.py`, `budget.py`,
   > `grec_check.py`, `feasibility.py`, `tail.py`, `timeout_check.py`, `symmetry_diag.py`,
   > `timeout_rate.py` and `ged_bounds.py` itself. Only `export_graphs.py`, `audit_recheck.py`,
   > `audit_dropped.py`, `final_counts.py` and `gedlib_api.py` survive. `data.md:672` predicted this
   > ("they **will not** survive"); T-01's scope (`plan.md:541`) ports only `audit_recheck.py`.
   >
   > **This gate blocks T-03 production, which is the long pole.** Do not quietly drop it — it is the
   > only check that would catch a GEDLIB cost-model misconfiguration, and `.claude/CLAUDE.md` calls
   > it "Cross-check, do not skip". **Owner: T-25. Decision required — see S-e.**
   >
   > **Collateral**: `data.md` §5 / H4 — ρ(exact, LB) = 0.966 vs ρ(exact, UB) = 0.840, and the
   > +78 % / −11 % bias figures — is the evidence for decision "BRANCH-FAST is the primary large-`n`
   > reference" (§11 item 4, `plan.md:673–675`). **That evidence is now unreproducible from any
   > surviving artifact**, as is every timing in `data.md` §§3, 4, 4.1, 4.3, 4.4, 6. The decision may
   > well be right; it is currently unsupported.

3. **Exact-solver agreement**: `ANCHOR_AWARE_GED` and `networkx` A* must agree exactly on a shared
   sample under the same cost model.

Keep `ged_bounds.py` in the repo permanently as the independent cross-check, even once GEDLIB is
the reported source. **(Aspirational, not descriptive — it was never in the repo. See the gate-2
banner above.)**

### 7.2 T-16 — REJECTED (author decision, 2026-08-11)

**`wl_pruned_canonical` is not built.** No reviewer or editor asked for it; it entered the board as a
transfer from IsalSR and IsalHG, which both carry a WL-pruned variant. Nothing in `mail.txt` requires
a new canonicalisation algorithm, and shipping one in revision would be a *new algorithm introduced
during revision* — changing Tables 2 and 3 and Figures 2 and 4, needing its own claim scoping, in a
round whose opening comment (R3.1) asks whether the contribution is substantive enough. It would
invite "is this a contribution or a patch?" at the worst possible moment.

Removing it returns **3–4 days of C++** to the board.

#### What is retained: WL as a measurement, inside T-13

The motivation was a real finding and it stays, as one paragraph and one table row in the complexity
section rather than as an implementation. `data.md` §4.4, already measured:

| Graph | nodes | structural triplet `(\|N₁\|,\|N₂\|,\|N₃\|)` | 1-WL colours | outcome |
|---|---:|---:|---:|---|
| Mutagenicity/3703 | 98 | **28 classes** | **66 classes** | hangs, `\|Aut\| > 20,000` |
| Protein/enzyme_293 | 96 | 33 | 87 | 1.1 s, `\|Aut\| = 16` |

**1-WL is 2.4–2.6× finer than the incumbent pruning key, and strictly subsumes it** — the triplet
counts neighbourhood sizes at depth 1–3, whereas WL propagates the full multiset of neighbour colours
to stability.

This answers **R3.7d**'s request for a *characterised* worst case, which is what the reviewer
actually asked for: cost is governed by `|Aut(G)|`, not by size or density, and the current pruning
key is provably coarser than an available one. That is a stronger and more honest statement than an
unqualified "exponential", and it costs hours instead of days, with no parity re-proof and no new
claim to scope.

**Do not attempt automorphism pruning either.** Individualisation-refinement with automorphism
detection is what nauty/bliss/Traces do and is the actual fix; re-implementing it is a project, not a
revision. State it as future work and cite nauty, which is already being vendored as a competitor
(§4.2) — so the citation is free.

The `wl_pruned_canonical` design notes are preserved in the v0.6 history of this file should the
implementation ever be picked up outside the revision.

---

## 8. Statistics and stratification lock

| Item | Current | Locked | Driver |
|---|---|---|---|
| Correlation | Spearman ρ, asymptotic p, pair-level | ρ + **graph-level bootstrap CI** + **Mantel permutation test** | R3.5c |
| Bootstrap unit | pairs | **graphs** — resample graphs, recompute over induced pairs | R3.5c |
| Reported effect | pooled OLS β as headline | **per-dataset primary**; pooled demoted | R3.5b |
| Pair accounting | one unreconciled number | **raw → connected → GED-available → GED>0 → Lev>0 → analysed**, per dataset | R3.5a, E2, F2 |
| Exclusions | unjustified | justified per stage; `Lev=0 & GED>0` counted and **reported explicitly** | R3.5a |
| Stratification | none | by **node count** and **true density**, within and across datasets | AE.1, R1.3, E1 |
| Dataset properties | graphs, pairs, `m̄` | **+ `n̄`, + density, + connectivity retention** | E1, F1 |
| GED cost model | mixed across datasets | **one model, recomputed throughout** | R3.5b |
| GED reference | exact only, undisclosed gaps | **exact ≤ 12 / calibrated approximate > 12** | AE.1, R3.7 |

### The AIDS question, settled with data

R1.3 attributes the AIDS degradation to label loss. The rebuttal stands: **the GraphEdX GED is
itself topology-only**, so both sides of the correlation are label-blind and a label-loss mechanism
cannot explain that number. But we now also test the authors' *own* density claim:

- report true density per dataset (**currently uncomputable from the paper** — E1);
- **stratify AIDS pairs by density and report ρ within strata**, on 295,296 pairs instead of 131,148
  (**do not restate this as 181,909 → 1.62×** — see the §3.2 banner; the two count different
  populations).

**This can refute `conclusion.tex:30–36`.** If ρ does not recover on sparse AIDS strata, the density
explanation is wrong and must be rewritten. Run it early — **which is an argument for §7.1's stage 1,
not against it**: the stratification needs every stratum *populated*, not *saturated*, and stage 1
delivers it days earlier than the census would.

---

## 9. Manuscript errors (all accepted)

| ID | Defect | Fix |
|---|---|---|
| R3.4a / D5 | Alg. 2 `C`/`c` guards **and** duplicate checks reversed vs Table 1 | rewrite `methodology.tex:321–336` to match `graph_to_string.py:208–238` |
| R3.4c / D1 | `n^{9.0}` at `conclusion.tex:50` has no source; `:50` vs `:68` disagree | all exponents re-derived in T-06 |
| R3.4c / D2 | `n^{4.9}` called "super-polynomial" | three-way separation, T-13 |
| R3.7e / D20 | "breaks permutation equivariance" | → **invariance**. `M → P M Pᵀ` *is* equivariance |
| E1 | density never computed; no node count reported | T-01 |
| E2 / **F2** | 473,147-pair gap; LINUX 3,916 vs 1,685 | **cause: within-split GED coverage.** Fixed by T-03 |
| E3 | fits declared `n = 3–20`, greedy data to 50 | re-derived, T-06 |
| E4 | a fourth node range (`n = 3–11`) | cross-referenced |
| E5 | abstract self-contradiction (`:106` vs `:114`) | §10 B1 |
| E6 | "labels present in all five datasets" — **false for LINUX** | corrected |
| E7 | algorithms float to pp. 33–35, after the references | **T-11, and it must run BEFORE T-15.** Relax `\floatpagefraction{1}` / `\textfraction{.001}` (`main.tex:66–67`) and place the algorithms near their discussion. This **changes pagination** — trimming first measures the wrong document — and it is the single largest page recovery available, up to ~2 pages of near-empty float pages (`gap-audit.md` MF9, `manuscript.md` §3.1) |
| E8 | draft self-correction printed in Example 2.3 | delete; `[0,2,1]` is right |
| E9 | 13 dead entries in `cas-refs.bib` (56 defined, 43 cited) | T-08 — prune, so the 35–55 count cannot be miscounted from the file |
| E11 | generative-AI declaration commented out | **T-24** — restore; Elsevier compliance |
| E12 | two orphaned figure PDFs; **`graphical_abtract.pdf` misspelt**, referenced under that spelling at `main.tex:131` | **T-24** — rename and re-reference; decide reinstatement of the two orphans against the §3 page priority list |

---

## 10. Claim scoping (all accepted)

- **B1** — scope G2S: undirected **connected**; directed **root reaching all nodes**. State the
  asymmetry: S2G total, G2S partial. `main.tex:106–108`, `introduction.tex:33`, `:45–46`,
  `conclusion.tex:74`.
- **B2** — directedness: flag is **external metadata**; **restate Theorem 2.12 within a fixed
  directedness class**; move the "deterministic given `w` and the flag" hypothesis from the proof
  into the statement. Use the **exact witness** (one undirected edge and one directed arc both
  canonicalise to `"V"`); never quote a collision rate without its enumeration window.
- **B3** — "GED **standard** construction" → "explicit-construction reference model"; §4.3(a)
  supplies the real baselines.
- **B4** — propagate the results section's conditional framing to abstract and conclusion. Numbers
  re-derived in T-06.
- **B5** — limitations: `n` ceiling **with its cause** (§3.5); exponential worst case; no sequential
  or downstream task (§6).
- **B6** — unify the four-properties claim; attach it to the §4 table; soften "no existing method".

---

## 11. Open questions

Consolidated. Data-side detail in `data.md` §9, statistics-side in `statistics.md` §10.

### Resolved by author decision or measurement

| # | Item | Resolution |
|---|---|---|
| 1 | Exact-GED scope | **All-pairs, recover everything** (author, 2026-08-11). ~1,000–1,650 core-h, 16–26 h on 64 cores. Applies to the **five original datasets only** — exact GED is unobtainable on the extension cohort |
| 2 | One cost model | **Unit node + unit edge** (`statistics.md` D6). Published GraphEdX values will no longer match ours; stated in the text |
| 4 | Primary large-n reference | **BRANCH-FAST**, ρ(exact, LB) = 0.966 vs ρ(exact, UB) = 0.840 |
| 5 | Refine the upper bound | **GEDLIB supplies IPFP / REFINE / BRANCH_TIGHT** — use them instead of our plain BP |
| 6 | Calibration gate | Rule fixed in `statistics.md` §6 |
| 8 | MRM / partial Mantel | **Confirmatory** (`statistics.md` D4). Run in week 1 |
| 9 | Kendall τ-b | Spearman primary, τ-b as robustness check (`statistics.md` D1) |
| 10 | Exhaustive canonical > n = 12 | Measured — fails on 55 % of Protein graphs; **report the pruned/exhaustive gap as a result** |
| 11 | Bounds implementation | **GEDLIB** (recognised, builds on Picasso — `data.md` §7.5); our `ged_bounds.py` retained as a cross-check |

### Still open

| # | Question | Recommendation |
|---|---|---|
| ~~3~~ | ~~Reconcile Letter counts~~ — **closed**. The pipeline filter is `min_nodes = 2`, `require_connected = True`; applying it reproduces the manuscript **exactly** (1,180 / 1,253 / 2,059 / 89 / 769, 3,897,911 pairs, m̄ to two decimals). See `data.md` §0 | — |
| ~~7~~ | ~~Cohort / GREC~~ — **closed**. Add Mutagenicity, Protein, COIL-DEL, AIDS-IAM **and GREC**; drop COIL-RAG, Fingerprint, Web. GREC's 59.1 % retention is misleading: its discard is **size-unbiased** (11.59 vs 11.45 nodes), the cleanest in the cohort (`data.md` §2.2.1) | — |
| **15** | **The connectivity discard is size-biased on the datasets we added for scaling** — Mutagenicity discards graphs 1.9× larger than it keeps, AIDS-IAM 2.3× (`data.md` §2.2.1). Any "n̄ ≈ 30" claim is on a subsample with the large graphs preferentially removed | report retained **and** discarded `n̄`/`n_max`; state the precondition as a scope limitation with its measured cost. **Now paired with `statistics.md` D14** — the *encoding* discard has the same structure and was unflagged |
| ~~16~~ | ~~Benchmark GEDLIB `ANCHOR_AWARE_GED` against `networkx` A*~~ | **Promoted out of "open"**: it is now a required step of the size-stratified calibration ladder, `statistics.md` §6.1. Every node the exact solver buys widens the regime the large-`n` study is licensed from |
| ~~12~~ | ~~Confirmatory vs exploratory~~ | **Decided**, `statistics.md` §10. Outstanding requirement: the family must be **enumerated and its cardinality frozen** in T-02 before T-06 runs |
| ~~13~~ | ~~G2S timeout~~ | **Decided**: keep at 300 s, record per-graph time, report the rate per stratum — **and `statistics.md` D14 now fixes what the analysis does with a censored graph**, which was the missing half |
| ~~14~~ | ~~Symmetry stratification~~ | **Adopted**, `statistics.md` §7 |

### Awaiting author sign-off

| # | Decision | Due | Where |
|---|---|---|---|
| ~~S-a~~ | ~~Re-affirm decision 12~~ | **RESOLVED 2026-08-11 — affirmed** on the corrected number (n = 98 retained). No dataset moves | §3.4, `data.md` Q9 |
| ~~S-b~~ | ~~T-16 publication status~~ | **RESOLVED 2026-08-11 — rejected** | §7.2 |
| ~~S-c~~ | ~~Disclose E1–E12?~~ | **RESOLVED 2026-08-11 — yes, but the reviewer's comment is answered first and our own findings follow.** Ordering rule in `manuscript.md` §4.3 | `manuscript.md` §4.3 |
| **S-d** | **PI: which `labels.md` tier?** Tier 0 is not optional; Tier 1 recommended; **Tier 2 proposed** (0.5 d + 0.3 core-h, logged not written up); Tier 3 declined by default. The counter-case for Tier 0 alone is stated fairly in that file | **2026-08-18** — before T-06 launches | `labels.md` §0 |
| **S-e** | **Validation gate 2 — restore or retire?** (I-01, blocking) | **2026-08-13** — before T-03 is submitted | §7.3, T-25 |
| **S-f** | **The schedule does not fit** (I-06 / I-07) | **2026-08-13** — the value of every mitigation falls with time | §12 |
| **S-g** | **Two over-scope cuts, 1.0–1.5 d** (audit-2026-08-11b) | **2026-08-14** — before T-04 starts building backends | §7, §12.3 |
| **S-h** | **The bibliography does not fit, and T-19's recency criterion is satisfiable without adding an external reference** | **2026-08-16** — T-19's search strategy depends on it | §0.5 EiC.a1/a2/b, T-26 |

#### S-h — the bibliography arithmetic

**Floor (not optional)**: EiC.a1 and EiC.b are **pass/fail items the Editor-in-Chief checks
independently of the reviewers** (`mail.txt:124`), so rubric §4 guard 3 applies: they do not
negotiate. What needs deciding is *how* they are met, and the current allocation cannot meet them.

Two measured facts, both new:

1. **16–17 slots requested against 12 available**, and the stated relief is illusory — retiring a
   dead `.bib` entry frees nothing, because `elsarticle-num` prints only *cited* keys, so the 13
   dead entries never occupied a slot. The only real relief is **removing existing citations from
   the text**, which no ticket contemplates.
2. **Of 43 printed references, 5 postdate 2023 and both 2025 entries are group self-citations**
   ([28] and [29]). Zero third-party references after 2024, none from 2026, and **no
   pattern-recognition-venue reference after 2023**.

| Option | Effort | What it buys | What it risks |
|---|---|---|---|
| **A — accept 55 as the working ceiling** and spend all 12 slots, dropping the weakest 4–5 planned additions | ~0 | no text surgery; stays compliant | arrives at the ceiling with no margin for round 2, when reviewers routinely ask for more citations |
| **B — remove 4–5 existing citations** from the text to fund the additions | 0.5 d | keeps every planned addition and lands mid-range | touching the introduction's related-work prose late; each removal must be checked against R1.2a/AE.2, which *added* those obligations |
| **C — re-scope T-19 to ≥ 4** rather than ≥ 6 from 2025–26 | ~0 | fits inside 12 with margin | `:126`'s "last and current year" is the EiC's own emphasis; 4 is defensible, 2 would not be |

**Recommendation: A + C together**, with T-19's criterion tightened so it cannot be gamed:
**≥ 4 additions dated 2025–2026, at least 3 of them at pattern-recognition venues other than the
Pattern Recognition journal, and self-citations excluded from the count.** That is the smallest
change that makes both EiC.a2 and EiC.b testable and fits the slots.

> **A + C needs its allocation written out, or it does not visibly fit** (`third-auditor.md` §8):
> 5–6 + 3 + 1 + 1 + 4 is still **14–15 against 12**. One allocation that does fit, and what makes it
> fit:
>
> | Purpose | Slots | Note |
> |---|---:|---|
> | AGM (Inokuchi 2000), gSpan (Yan & Han 2002), nauty/Traces (McKay & Piperno 2014), Babai 2016 | **4** | graph6/sparse6 cite the nauty manual — same key, no extra slot. **If S-g cuts the bliss/Traces backends, the bliss citation goes with them** — that is the S-g ↔ S-h link neither audit drew |
> | Riesen–Bunke 2009 (BIPARTITE), Fischer 2015 (*Pattern Recognition*, HED), Blumenthal & Gamper 2018 (*IEEE TKDE*, BRANCH-FAST) | **3** | §5.4 says "Blumenthal 2020"; §7.3 cites **TKDE 30(3):503–516, 2018**. Reconcile — they are not the same paper |
> | IAM Graph Database (Riesen & Bunke, SSPR 2008) | **1** | |
> | GEDLIB (Blumenthal et al., GbRPR 2019) | **1** | |
> | Recency / PR-venue additions, 2025–26 | **4** | ≥ 3 at PR-field venues other than the PR journal; self-citations excluded |
> | **Total** | **13** | **one over** — drop Babai (cited nowhere load-bearing) **or** one of the three GED-approximation entries |
>
> **Also budget for I-08b**: `Fischer2015hausdorff` and `Lerouge2017ilp` are cited only from
> commented-out LaTeX at `methodology.tex:805–806`. The revision expands the GED discussion, so
> uncommenting either is plausible — and takes the printed count to 44–45 and headroom to **11–10**.
> Fischer 2015 is *in the table above*, so plan for it: the fitting allocation is **12 with Fischer
> counted**, not 12 plus Fischer.

**Counter-case, stated fairly**: option B is the only one that preserves the full related-work
build-out, and R1.2a and AE.2 both explicitly ask for *more* references — cutting existing ones to
fund new ones can read as churn if a reviewer diffs the bibliography. If the supplementary query
(decision 16) comes back saying supplementary material sits outside the page limit, B also becomes
cheaper, because the pressure moves off the main text.

**Owner: PI. Date: 2026-08-16.**

#### S-e — validation gate 2: restore or retire

**Floor (not optional)**: gate 2 currently cites a script that does not exist. Either it is restored
or it is **struck with the reason recorded**. Leaving an unexecutable gate in a locked section is the
one option that is not available, because T-05 will be run by someone who reads §7.3 as a checklist.

| Option | Effort | What it buys | What it risks |
|---|---|---|---|
| **A — rewrite `ged_bounds.py`** (BP + BRANCH-FAST direct, per `.claude/CLAUDE.md`) | **0.5–1 d** | an independent implementation to cross-check GEDLIB; restores the H4 evidence for §11 item 4 | a day off the critical path, which I-07 says has none to give |
| **B — retire gate 2, keep gates 1 and 3** | ~0 | gate 1 (bracket validity `LB ≤ exact ≤ UB`) and gate 3 (`ANCHOR_AWARE_GED` vs `networkx` A*) still run; both are cost-model-sensitive | loses the only *cross-implementation* check; a systematic GEDLIB misconfiguration that respects the bracket would pass 1 and 3 |
| **C — defer to a spot-check** — 20 pairs by hand against `networkx` under the unit model | ~1 h | most of B's coverage at a fraction of A | not the 300–400-pair agreement the gate specifies |

**Recommendation: C, then B.** Gate 1 already catches the failure mode that matters most (a bracket
violation *is* a cost-model mismatch), and `.claude/CLAUDE.md`'s GEDLIB section documents the silent
traps far better than a re-written script would re-derive them. **Counter-case, stated fairly**:
option A is the only one that restores the evidence for the *primary large-`n` reference* decision —
ρ(exact, LB) = 0.966 vs ρ(exact, UB) = 0.840 currently has no surviving artifact behind it, and that
decision determines what the whole Suite-2 size story is measured against. If a reviewer asks how the
lower bound was chosen, C and B leave us citing a number we cannot reproduce.

**Owner: PI. Date: 2026-08-13.**

#### S-f — the schedule does not fit

**Floor (not optional)**: §12 is the plan's feasibility argument and it is built on a stale total.
The number is corrected in §7 regardless of what is decided here. What needs a decision is the
response.

| Option | Effort | What it buys | What it risks |
|---|---|---|---|
| **A — invoke risk R1's subsample now**, not on day 10 | saves ~900–1,550 core-h and 2–5 elapsed days | it is **already pre-approved and costed** (`plan.md` §12.2 R1, ~100 core-h, ~10× cheaper); dyadic dependence means effective sample size tracks the **number of graphs**, so little power is lost | the exact-GED story becomes "stratified sample" rather than "all pairs", which is weaker against R3.5a's pair-accounting ladder |
| **B — request a deadline extension** from patcog@elsevier.com | one email | Elsevier routinely grants 2–4 weeks on major revisions; converts a 27.5-day path into a feasible one | none material; it is asked on day 2, not day 19 |
| **C — cut to fit** | see §12.3 + S-g | keeps the deadline | §12.3's list totals ~2 days against a 7.5-day lower-bound overrun; **it is not large enough to close the gap** |
| **D — accept and absorb** | — | — | T-06 gets 5 of the 10–14 days it needs, on the ticket every downstream artifact depends on |
| **E — stage T-03** (`third-auditor.md` §1) | one paragraph of protocol | **Strictly dominates A.** A stratified stage-1 sample (~100 core-h, hours) is pre-declared as the reported analysis and unblocks T-06 on schedule; the census runs unattended behind it and supersedes stage 1 if it lands before the T-20 freeze. Takes 985–1,640 core-h **off the critical path without giving the census up**. ~2–5 elapsed days | the pre-declared supersession rule must be written before either runs, or the choice between two ρ values becomes outcome-dependent |

**Recommendation: B immediately, with E as the technical structure** (E replaces A, which it
dominates). B is nearly free and is the only option that does not degrade the science; it also pairs
naturally with decision 16's day-1 query to the same mailbox. **E is free** and is the only option
that shortens the critical path without cutting anything. **Counter-case, stated fairly**: an extension request signals the revision is
struggling, and some handling editors read it as a predictor of a weak resubmission. Against that —
`mail.txt:67` says these concerns "will strongly influence the potential impact of the work", which
is an argument for doing them properly rather than quickly. **C alone will not work**, and that is
the finding, not an opinion: §12.3's four items return ~2 days against a 7.5-day minimum overrun.

**Owner: PI (B is Ezequiel's to send). Date: 2026-08-13.**

#### S-g — two over-scope cuts

Both are **sub-ticket** items, so neither appears in §12.3's cut order, which operates on whole
tickets. Neither is on the critical path.

| Item | Effort returned | Recommendation | Counter-case |
|---|---|---|---|
| **bliss / Traces backends** (`plan.md:384`, "0.5 d each") | **1.0 d** | **Cut.** Absent from the `ReprBackend` set at `plan.md:371`; functionally duplicate nauty (all three emit a canonical labelling serialised to graph6, differing in *speed*, not representation); produce no table row; requested by no reviewer or editor. nauty alone discharges the `:374–375` rationale ("we need canonical relabelling anyway, to make the graph6 comparison fair") | they are cheap insurance if `pynauty` fails to build on Picasso, which would otherwise take the graph6 **and** AGM rows down with it |
| **T-09 split** (`plan.md:1002–1006`) | **0.5–0.75 d** if the S2G/G2S half is cut | **Split the cut.** T-09 merges two figures with different justifications: the search-space schematic answers **R3.7c** and its renderer already exists (`viz/search_tree.py`, ~2–3 h, ≈0.25–0.5 p); the S2G/G2S worked example answers **no demand at all** (author decision 6) and is what rubric §4's cut order names first. Bundling them protects the unasked-for figure behind the requested one | both feed the refreshed graphical abstract, which is submitted separately and costs no manuscript pages — so cutting either loses part of that. This argument does **not** distinguish them, and §12.3 currently uses it to protect both |

**Owner: PI. Date: 2026-08-14** — before T-04 starts building backends.

---

## 12. Schedule and risk register

`gap-audit.md` MF11: durations were given as ranges and never laid on a calendar. **Summed at the
upper bound the board is 76.5 days of work in a 20-day window.** That is survivable only because most
tickets parallelise and most of the compute is unattended — but only if the sequencing is explicit.

### 12.1 Calendar with gates

| Window | Gate that must close | Why it is a gate |
|---|---|---|
| **Day 1 — 08-12** | **T-23** quota cleared · **decision 16** query sent to patcog@elsevier.com · T-01 started | T-03 fails partway without the quota; the page strategy branches on the query and latency is not ours to control |
| **Day 2 — 08-13** | **S-e** (validation gate 2) · **S-f** (the schedule does not fit) | **Both gate T-03**, the long pole. S-f's best option — request an extension — loses value every day it is delayed, and its fallback (risk R1's pre-approved subsample) must be chosen *before* T-03 is submitted, not on day 10 |
| **Day 3 — 08-14** | **S-g** (bliss/Traces cut · T-09 split) | T-04 starts building backends; after that the 1.0 d is spent |
| **Day 5 — 08-16** | **S-h** (bibliography arithmetic) | T-19's search strategy branches on it, and T-19 feeds T-08 → T-26 → T-15 |
| **After T-08 + T-19, before T-15** | **T-26** — slot and page-budget reconciliation | The two arithmetics the **EiC checks independently**. There is currently no gate between the tickets that *spend* slots and the trim that discovers the overrun |
| **Days 2–4** | T-01, T-02 closed · **T-03 gate passed** (GraphEdX agreement under `[0,0,0,1,1,0]`, §7.1) · T-03 submitted | T-03 is the long pole: 16–26 h of compute **plus unbudgeted queue time** on a cluster with offline nodes |
| **Days 2–6, parallel** | T-04 → **T-04a** · T-07 · T-22 · T-13 | none depends on T-03. **T-04a gates every production distance matrix**, so it cannot slip past T-06 |
| **Days 5–8** | T-05 calibration arm · **`statistics.md` §5 MRM** · **§8 AIDS density stratification** | both can refute a central claim; a refutation on day 15 has no absorption time |
| **Day 7 — 08-18** | **PI decision on the `labels.md` tier** (§0 of that file) | Tier 2 must be configured into the T-06 run, not bolted on afterwards |
| **Day 9 — 08-20** | **E1–E12 disclosure decision** (`manuscript.md` §4.3) | fixes the letter's structure before assembly |
| **Days 8–12** | T-06 full recompute · T-18 labels · T-17 comparison table · T-08 → T-19 | |
| **Day 11 — 08-22** | **R3.2 contingency go/no-go** (§6.1) | the last date a minimal sequence-model arm could start |
| **Days 12–17** | **T-20 manuscript rewrite** · T-11 (**including E7, before any trim**) · T-12 · T-21 | |
| **Days 17–19** | T-15 page trim · T-14 letter assembly | fragments have accrued since day 2 |
| **Day 20 — 08-31** | **T-24 package uploaded** | |

### 12.2 Risk register

| # | Risk | Trigger | Mitigation, decided now |
|---|---|---|---|
| **R1** | **T-03 does not finish.** 985–1,640 core-hours of it is AIDS alone; queue time is unbudgeted | no result by day 10 | Fall back to the **stratified-subsample exact GED** already costed in `data.md` §3.1 ("Cost-reduction option worth deciding", ~100 core-hours, ~10×cheaper). Because pairs are dyadically dependent, effective sample size is governed by the **number of graphs**, so very little power is lost. **Pre-approved — do not re-litigate it on day 10** |
| **R2** | **The MRM or the AIDS stratification refutes `conclusion.tex:30–36`** | β₁ collapses, or ρ does not recover on sparse AIDS strata | This is a *result*, not a failure, and both `statistics.md` §5 and §8 already fix the interpretation in advance. Reserve days 13–15 for the rewrite. Running these in week 1 exists precisely to buy that time |
| **R3** | **Supplementary material counts toward the 35 pages** | reply from patcog@elsevier.com | `manuscript.md` §3.2's priority ranking. Items 10 (search-space schematic) and 11 (S2G/G2S figure) are the only two no reviewer requires and are cut first — which makes decision 6 an author preference to re-affirm |
| **R4** | **gSpan's minimum DFS code is not exposed** by `LasseRegin/gSpan` — the plan already flags this at §4.2 | day 1 of T-04 | Extract or reimplement within the same 2–3 day budget. If it slips, gSpan is **discussed** in the related-work section and the *running* comparator set drops to nauty-graph6 + sparse6 + AGM. R1.2 is answered by citation and comparison; only the empirical row is lost |
| **R5** | **Page overflow discovered at T-15** | count > 35 on day 17 | Track the page count at **every commit** to the manuscript repository from the moment the first new section lands, not at the end (`manuscript.md` §3.3) |
| **R6** | **Round-2 rejection on R3.2** | out of our control | §6.1: the delta table pre-empts the reading, the contingency exists, and the letter frames the AE.3-over-R3.2 choice as the deliberate exchange it is |

### 12.3 What gets cut, in order

Decided now so it is not decided under pressure on day 17:

T-16 is not on this list: it is **rejected**, not deferred (§7.2), and its 3–4 days are already back.

1. **`labels.md` Tier 2** — the logged label-aware GED arm. Half a day; Tiers 0–1 answer R1.3 without
   it.
2. **The GEDLIB per-dataset cost-model sensitivity arms** — cheap to run, expensive in pages.
3. **Exhaustive-canonical baseline at scale** — `data.md` §4.1 shows it fails on 55 % of Protein
   graphs; report it as a bounded baseline in one row rather than a full arm.
4. **T-09** explanatory figures — the *only* board item no reviewer requires, since R3.7c is a "would
   benefit from" and the S2G/G2S figure is author preference (decision 6). **Cut last among these
   even so**: the renderer already exists, and T-09 also produces the refreshed graphical abstract,
   which is submitted separately and therefore costs no manuscript pages. Cutting it saves ~1.5 days
   and 0.5–1.25 pages but loses the graphical abstract update as well.

> **Restated in the currency that binds — `third-auditor.md` §9.** This list is denominated in
> **days**, but `manuscript.md` §3 measures a **≈ 8-page gap that "cannot be closed by editing"**
> against a document at exactly 35/35. Days are recoverable by an extension (S-f); **pages are
> pass/fail at the EiC (EiC.c) and no extension returns any.** In pages, from `manuscript.md` §2's
> own inventory: **S2G/G2S worked example 0.75 p** (unrequested, author decision 6) ·
> `labels.md` **Tier 3 subsection ≈ 1.0 p** (already declined) · **Tier 1 label/collision table
> 0.75 p** (driver is R1.2/AE.3, not R1.3) · **search-space schematic 0.5 p** (R3.7c, requested —
> cut last). The S2G/G2S figure is ~9 % of the gap and answers no demand; that, not the 0.5–0.75 d,
> is why S-g's split is worth taking.

**The "nothing below this line" claim is false at component granularity, and both prior audits say
so.** `audit-2026-08-11b` falsified it with bliss/Traces (1.0 d, in no `ReprBackend` row, duplicates
nauty). The **next component item after it, if T-04 slips**, is the **AGM `ReprBackend`
implementation** (1 d, §4.2): R1.2a's owner has already been demoted from T-04 to T-08 because "the
ask is *discussion*", and AE.3 is satisfied by a **qualitative** properties row in T-17. **Not
recommended as a cut** — AGM is named by R1 by name and a measured row is more defensible than an
asserted one — but it is the correct next candidate, and naming it is better than declaring the line
uncuttable.

Below the component items, every remaining **ticket** is the sole owner of at least one numbered
reviewer or editor demand (§0.5).

---

## 13. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v0.1 | Initial grouping and disposition |
| 2026-08-11 | v0.2 | Data audit F1–F3. Ordered recompute. Sequential model declined. Competitors into three experiments. E2's cause corrected |
| 2026-08-11 | v0.3 | Splits merged; all GED recomputed under one cost model. Controlled-edit cohort **dropped** in favour of exact ≤12 + calibrated approximate >12. Competitor backends in the IsalHG `iso_backends` idiom. New-dataset cohort proposed (IAM family + TUDataset, not Benson). [28] permanently arXiv-only |
| 2026-08-11 | v0.4 | **`data.md`** and **`statistics.md`** added, all figures measured. Cohort resolved to the IAM Graph Database alone (TUDataset unnecessary — Mutagenicity reaches n = 417). Key measured results: pruned canonical encodes n̄ = 32 in 3.9 ms with no timeout to n = 96; exact GED is 36.9 s/pair at n = 12; the whole approximate-GED extension costs 1.24 core-hours; **BRANCH-FAST lower bound tracks exact GED better than the BP upper bound** (ρ 0.966 vs 0.840). Open questions restructured by decision deadline |
| 2026-08-11 | **v0.5** | **GED implementation LOCKED in §7.3**, and every earlier statement superseded: §3.1 regime table, §3.3 "write BP ourselves" (struck through), §3.3 reference list, §3.4 cohort, §3.5 size story, §4.1 `GEDBackend` row, §5.4 bibliography budget, T-01b (closed), T-05. **Exact = `ANCHOR_AWARE_GED`; proven lower = `BRANCH_FAST`; proven upper = `IPFP` — all via GEDLIB**, verified end-to-end on Picasso, with the accessor-capability matrix and the silent-failure traps documented. Cohort locked from measured post-filter counts; **TUDataset dropped**. §3.5 gains the \|Aut(G)\| finding |
| 2026-08-11 | **v0.6** | **Coverage audited and closed.** New **§0.5 traceability matrix** — every demand in `mail.txt` plus every self-found defect, locked to a decision, a ticket and a manuscript artifact; no row is unowned. **Eight tickets added** (T-17…T-24) for demands that had none: the AE.3 comparison table as a paper artifact, labels, bibliography recency, the **manuscript rewrite** (which had no owner at all), implementation/artifact release, the Thm 2.12 / Cor. 2.13 formal audit, the **blocking Picasso quota**, and the submission package. **Decisions 13–16 added.** Corrections to locked decisions: decision 12's rationale (the 417-node graph is **disconnected**; the retained ceiling is **98** — verified), Hamming undefined for unequal-`n` competitor pairs, sparse6 identified as the head-to-head competitor for Claim A, Fischer 2015 does not satisfy EiC.a's recency, the GraphEdX gate needs its own cost model, E7 moved out of the page trim, R3.2's decline given a dated contingency. New **§12 schedule and risk register**. Companions added: **`gap-audit.md`**, **`labels.md`**, **`manuscript.md`**; `data.md` → v1.1, `statistics.md` → v2.1 |
| 2026-08-12 | **v0.9** | **Third-auditor pass over `audit-2026-08-11b`** (`.claude/notes/audit-2026-08-11b/third-auditor.md`). The wave is substantively sound — its bibliography arithmetic (43 cited / 13 dead / 12 slots), venue audit (6× *Pattern Recognition* 2021–23, zero CVPR/ICCV/ECCV/ICPR/TPAMI/IJCV, nothing PR-field after 2023), 35/35 page count, missing-script finding and R3.1a(ii) coverage hole all reproduce independently. **One major understatement**: it cleared T-03's all-pairs AIDS census on the cut-guard test, which verifies a driver exists but never asks whether the **dose** is proportionate. `statistics.md` **D2** and §12.2 risk R1 already prove it is not — 769 AIDS graphs are the independent unit whether we compute 131,148 pairs or 295,296 — so **T-03 now runs in two stages** (§7.1): a pre-declared stratified stage 1 (~100 core-h) unblocks T-06, the census runs unattended behind it and supersedes stage 1 only if it lands before the T-20 freeze. **~900–1,550 core-hours and 2–5 days off the critical path, with the census kept.** Recorded as **S-f option E**, which dominates option A. **One misdirected finding**: **I-11** must not be applied — 131,148 and 181,909 count different populations and the proposed 1.62× would itself be MF1's defect class; downgraded to *provenance not recorded*, 2.25× stands. **One invented consequence**: I-09's correction (417 → 98) is right but "sizes the heavy-tail strata T-02 must freeze" is not — §6.1 already read 98 in v2.1; banner amended in `statistics.md`. **One arithmetic error**: the corrected board total **91.0 / 52.8 omits T-24, T-25 and T-26** — i.e. I-06's own failure mode — and is **93.5 / 54.8**. **Three effort inflations** trimmed (AE.5 0.25 d → ≈ 0, double-counted with R3.1a(ii)'s paragraph; T-25 booked at S-e's recommended path rather than option A; R3.1a(ii)'s "blocking" label). **Three consistency defects the wave introduced, fixed**: §5.1 still said [29]'s paper was unavailable after decision 9 recorded it published (**verified: JCIM 65(15):7936–7955, 2025**); companion versions still read v1.1/v2.1; the coverage banner still read "40 covered". **Two items neither audit priced**: the cut order is denominated in days when the binding constraint is `manuscript.md` §3's **≈ 8-page gap at 35/35** (restated in pages, §12.3), and the **AGM `ReprBackend`** is named as the next component cut after bliss/Traces. S-h gains a concrete 12-slot allocation; the R3.2 08-22 contingency is tied to S-f's outcome, since §7 establishes there is no slack for it |
| 2026-08-12 | **v0.8** | **Over-scope and integrity audit (`audit-2026-08-11b`), four read-only agents against a demand inventory rebuilt from `mail.txt` alone.** Coverage re-checked without inheriting `gap-audit.md`. **Under-coverage: one hole — R3.1a(ii)**, the second conjunct of R3's strongest-modal sentence ("explain why the combined extension constitutes a sufficiently substantive contribution"), unowned in every document; now on T-07 at ~2 h. **Over-scope: 1.0–1.5 days**, both sub-ticket and therefore invisible to §12.3's whole-ticket cut order — bliss/Traces (1.0 d, duplicates nauty, in no `ReprBackend` row) and the T-09 bundle (**S-g**). **Both rows the README nominated as likely over-scope survive**: R3.5b's recompute has two verified non-R3.5b drivers (F2's 473,147-pair gap; Cor. 2.13's pseudometric problem) and the Suite-2 cohort is *not* one of them at 1.24 core-h — but **D5 answers the literal clause for free and was recorded in no ticket**, so R3.5b now carries a zero-compute floor; R3.6a's expensive branch is owned by AE.4a, not by R3.6a, and the free branch is already taken unconditionally. **Corrections to the matrix**: R1.3b is a *premise* and M3 is *inferred*, both relabelled so no budget attaches; **AE.5 added**; R1.2a's owner demoted from T-04 to T-08; E6's three simultaneous owners resolved to T-12. **Integrity: 24 defects stand, 1 rejected on re-measurement** (the bibliography is **43 cited / 13 dead / 12 slots** — the plan was right; `elsarticle-num` prints only uncommented `\cite` keys). **One blocking**: validation gate 2 is unexecutable (**T-25**, **S-e**), which also leaves the "BRANCH-FAST is the primary large-`n` reference" decision without reproducible evidence. **Two major schedule findings** (**S-f**): the 76.5-day total is the v0.5 board and the critical path does not fit. **Eight of the 24 are one recurring defect** — a statistic computed over one population and printed under another's header, MF1's class, recurring across five documents. Stale numbers corrected in place with their superseded values recorded: `statistics.md:116` (`max n = 417` → **98**, inside the locked protocol), `data.md:438` (live "Recommendation: build T-16", rejected by decision 17) |
| 2026-08-11 | **v0.7** | **Author review of v0.6, with three findings re-verified against the data and the manuscript rather than argued.** (i) **T-16 REJECTED** — no reviewer asked for `wl_pruned_canonical`; it was an IsalSR/IsalHG transfer. The WL *measurement* is retained inside T-13, where it answers R3.7d with a characterised worst case. **3–4 days returned.** (ii) **MF1 blocks nothing** — re-measured: the cohort, its counts and its pair totals are unchanged, and the COIL-RAG / Fingerprint / Web drop decisions all survive on connected-set numbers. A second error surfaced in passing: `data.md` §2.2.1's Fingerprint row (67.2 %, discarded n̄ 11.56) is unreproducible; measured 51.4 % / 5.98. (iii) **`labels.md` rescoped to v2.0 and handed to the PI** — grepping the sources confirms the manuscript **never claimed label handling** (`computational_experiments.tex:30`, `conclusion.tex:70`, `:71`, `:81`), R1.3 asks for a **discussion**, and `conclusion.tex:81` already names the label-aware GED study as future work needing a variant that does not exist. Four costed tiers, PI decision due 2026-08-18. (iv) **T-09 and T-10 merged**, and both figures now double as the refreshed graphical abstract, which costs no manuscript pages. (v) **The padded-Hamming decision is withdrawn** in favour of **T-04a**, a measured metric-feasibility experiment in the new **`competitors.md`**, with the selection rule fixed in advance and ties broken on cost, never on correlation with GED. (vi) `statistics.md` D13–D15 rewritten with worked examples after the author reported them unclear |
