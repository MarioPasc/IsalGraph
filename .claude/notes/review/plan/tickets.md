# Ticket board

**A brief index, not a specification.** Each row names the ticket in one line and lists the files an
agent must read before starting it. The *content* lives in those files — do not duplicate it here.

**Board: 93.5 days upper / 54.8 lower. Critical path: 27.5–28.0 days serial against 19.**
See [schedule](schedule.md).

**Read for every ticket**: [decisions](decisions.md) (do not re-litigate a signed decision) and
[demands](demands.md) (what the ticket is answering, and to whom).

---

## Board

| ID | Ticket | Depends | Days | Pri | **Read first** |
|---|---|---|---|---|---|
| **T-01** | **Data lock** — size/density/connectivity audit tables (retained **and** discarded); `n_max = 12` retained for Suite 1, dropped for Suite 2; merge splits; define cohorts; port surviving scripts into `tests/` | — | 1–2 | **P0** | [data](data.md) |
| **T-02** | **Statistics lock** — graph-level bootstrap, Mantel, pair-accounting ladder, **and the frozen confirmatory family with its cardinality** | T-01 | 2–4 | **P0** | [statistics](statistics.md), [data](data.md) |
| **T-03** | **Exact GED on Picasso** — Suite 1, **two stages**: stratified stage 1 is the reported analysis, census unattended behind it | T-01, T-23, T-25 | 3–8 | **P0 — long pole** | [exact_ged](exact_ged.md), [gedlib](gedlib.md), [statistics](statistics.md) D6/D11 |
| **T-04** | **Competitor backends** — `src/isalgraph/competitors/` in the IsalHG idiom: graph6, sparse6, nauty, AGM, **gSpan min-DFS** | — | 3–8 | **P0** | [competitors](competitors.md) |
| **T-04a** | **Metric feasibility** — every (representation × distance) cell on a fixed 200-graph sample; select each primary distance by the pre-declared rule. **Must close before any production distance matrix** | T-04 | 0.5–1 | **P0** | [competitors](competitors.md) §3 |
| **T-05** | **Bounded GED via GEDLIB** — wire `BRANCH_FAST` + `IPFP`, pass the validation gates, run the **calibration ladder**, then all 40 M Suite-2 pairs | T-01, T-03 | 5–10 | **P0** | [approx_ged](approx_ged.md), [gedlib](gedlib.md), [exact_ged](exact_ged.md) §4 |
| **T-06** | **Full recompute** — all experiments, C++ engine, new cohorts, competitor columns, new statistics | T-02…T-05 | 10–14 | **P0** | [statistics](statistics.md), [data](data.md), [competitors](competitors.md), [labels](labels.md) |
| **T-07** | **Read [28] and [29]**; inherited/modified/new delta table **plus the sufficiency paragraph**; resolve D19 | — | 1–4 | **P0** | [corrections](corrections.md) §4, [decisions](decisions.md) 9 |
| **T-08** | **Related-work section** (§1.x) + bibliography to ≤ 55 | T-07 | 4–10 | P1 | [compliance](compliance.md), [manuscript](manuscript.md) §1 |
| **T-09** | **Explanatory figures** — the canonical search-space schematic (**R3.7c, requested**) and the S2G/G2S worked example (**author preference, first page cut**). Both regenerate the graphical abstract | — | 1.5 | P1 | [manuscript](manuscript.md) §2, [decisions](decisions.md) S-g |
| **T-11** | **Manuscript defects** — Alg. 2, Example 2.3, equivariance→invariance, **and E7's float fix, which must precede any trim** | — | 2 | P1 | [corrections](corrections.md) §3 |
| **T-12** | **Claim scoping** — B1…B6, and E5/E6 | T-06 | 2 | P1 | [corrections](corrections.md) §1 |
| **T-13** | **Complexity section** — `P(M)` recomputation, four costed operations, three-way separation, the `\|Aut(G)\|` worst case | — | 2 | P1 | [corrections](corrections.md) §5, [data](data.md) §4 |
| **T-14** | **Response letter** — assembles the fragments each ticket emits; **not written from scratch at the end** | all | 3 | **P0** | [manuscript](manuscript.md) §4, [demands](demands.md) |
| **T-15** | **Page trim to 35** + supplementary split | all | 2 | **P0** | [manuscript](manuscript.md) §3, [compliance](compliance.md) §7 |
| **T-17** | **AE.3 comparison table** as a paper artifact — properties, strengths, limitations of each, on R1.2's five axes. **Rows populated from T-04's measurements, not asserted** | T-04, T-07 | 2–3 | **P0** — the AE endorsed this in their own voice | [competitors](competitors.md), [demands](demands.md) AE.3/R1.2b |
| **T-18** | **Labels** — tiered; Tier 0 not optional, Tier 1 recommended | T-05, T-06 | 0.3–1 | **P0** (Tier 0) | [labels](labels.md) |
| **T-19** | **Bibliography recency and venue audit** — classify all 43 references; add **≥ 4 from 2025–26**, ≥ 3 at PR venues other than the PR journal, **self-citations excluded** | T-08 | 1–2 | **P0** — EiC checks independently | [compliance](compliance.md) §2–§4 |
| **T-20** | **Manuscript rewrite** — §3.1, §3.2, §3.3, §4, §5, abstract. The largest single writing task | T-06 | 5–7 | **P0** | [manuscript](manuscript.md) §1, [statistics](statistics.md), [data](data.md) |
| **T-21** | **Implementation, reproducibility and artifact release** — C++ engine and GEDLIB in §3.3; versions; the `-march` and non-rsyncing-`.so` traps; data-availability statement | T-06 | 1–2 | P1 | [compliance](compliance.md) §8, [gedlib](gedlib.md) |
| **T-22** | **Formal-statement audit** — restate Thm 2.12 within a fixed directedness class, move the flag hypothesis into the statement, **re-verify all three proof steps**, propagate to **Cor. 2.13** | — | 1–2 | **P0** | [corrections](corrections.md) §2, [statistics](statistics.md) D6 |
| **T-23** | **Clear the Picasso `fscratch` file-count quota** — a **file-count** limit, not a space limit | — | 0.5 | **P0 — blocks T-03** | [gedlib](gedlib.md) §2 |
| **T-24** | **Submission package and Elsevier compliance** — source files, AI declaration, biographies, acknowledgements, highlights, graphical abstract (**fix the misspelt filename**), competing-interest and data-availability statements | T-15 | 1 | **P0** | [compliance](compliance.md) §8 |
| **T-25** | **Restore validation gate 2, or retire it on the record** — `ged_bounds.py` and 12 other named scripts do not exist. Also re-establishes the evidence for "BRANCH-FAST is primary" | — | **0.1–0.2** (recommended path); 0.5–1 if option A | **P0 — blocks T-03** | [exact_ged](exact_ged.md) §4, [decisions](decisions.md) S-e |
| **T-26** | **Bibliography-slot and page-budget reconciliation** — the two arithmetics the EiC checks independently and no other ticket owns end to end. **Runs after T-08 and T-19, before T-15** | T-08, T-19 | 0.5 | **P0 — EiC pass/fail** | [compliance](compliance.md) §2, [manuscript](manuscript.md) §2–§3 |

**Retired**: ~~T-01b~~ (new-dataset audit — done) · ~~T-10~~ (merged into T-09) ·
~~T-16~~ (`wl_pruned_canonical` — **rejected**, [decisions](decisions.md) §2).

---

## Dependency structure

**Critical path** — T-23 → T-01 → T-03 → T-05 → T-06 → T-20 → T-15 → T-24, with T-14 accruing
throughout. **T-25 joins T-23 as a day-1 gate on T-03.**

**Parallel off it** — T-04 → T-04a → T-17 · T-07 → T-08 → T-19 → T-26 · T-22 · T-13 · T-09 · T-11.

**T-04a gates T-06's distance matrices**, so it is on the path for everything downstream of the
competitors even though it is half a day.

**Ordering constraints that cost rework if violated**: [manuscript](manuscript.md) §5.

---

## Response-letter fragments

**Every ticket emits its response fragment when it closes.** T-14 assembles, harmonises the register
and writes part 0 — three days is enough for that and is not enough for writing 41 answers from
scratch. [demands](demands.md) is the index; an empty fragment cell is a visible hole.
