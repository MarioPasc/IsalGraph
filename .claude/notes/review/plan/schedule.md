# Schedule, risk and cut order

**Due 2026-08-31.** Day 1 = 2026-08-12, Day 20 = 2026-08-31.
**Status: the board does not fit the window. Decision S-f is open.**

Related: [tickets](tickets.md) · [decisions](decisions.md) S-f · [manuscript](manuscript.md) §5
(ordering constraints)

---

## 1. The arithmetic, measured

Re-parsed programmatically from [tickets](tickets.md)'s Days column:

| | Value |
|---|---:|
| **Board, upper bound** | **93.5 days** |
| **Board, lower bound** | **54.8 days** |
| **Declared critical path, lower bound** | **27.5 days** (28.0 with T-25) |
| Declared critical path, upper bound | 44.5 days |
| **Window** | **19–20 days** |

**The critical path is serial and does not fit.** T-23 → T-01 → T-03 → T-05 → T-06 → T-20 → T-15 →
T-24 sums to 27.5 days at *lower* bounds. "Survivable because most tickets parallelise" does not
apply to a critical path, by definition.

Two earlier figures were both wrong and both wrong the same way — a total that predated the tickets
added alongside it. **76.5** was the pre-audit board (T-01…T-15 + the now-rejected T-16). **91.0 /
52.8** summed T-01…T-23 and omitted T-24, T-25 and T-26. Quote **93.5 / 54.8**.

**The largest single lever is decision 21** — staging T-03 takes 985–1,640 core-hours of AIDS census
off the path without giving the census up. See [exact_ged](exact_ged.md) §3.

**Two tickets are allocated below their own minimum** in §2 below: **T-05** (5–10 d) gets 4, and
**T-06** (10–14 d) gets 5 — half its minimum, on the ticket every downstream artifact depends on. The
other 18 tickets are at or above minimum. Risk R1 budgets for T-03 slipping; **nothing budgets for
T-06 receiving 5 of the 10–14 days it needs.**

---

## 2. Calendar with gates

| Window | Gate that must close | Why it is a gate |
|---|---|---|
| **Day 1 — 08-12** | **T-23** quota cleared · **decision 16** query sent to patcog@elsevier.com · T-01 started | T-03 fails partway without the quota; the page strategy branches on the query and latency is not ours to control |
| **Day 2 — 08-13** | **S-e** (validation gate 2) · **S-f** (the schedule) | **Both gate T-03**, the long pole. S-f's best option — request an extension — loses value every day it is delayed, and staging T-03 must be chosen *before* submission, not on day 10 |
| **Day 3 — 08-14** | **S-g** (bliss/Traces cut · T-09 split) | T-04 starts building backends; after that the 1.0 d is spent |
| **Days 2–4** | T-01, T-02 closed · **T-03 gate 0 passed** (GraphEdX agreement under `[0,0,0,1,1,0]`) · **T-03 stage 1 submitted** | T-03 is the long pole: compute **plus unbudgeted queue time** on a cluster with offline nodes |
| **Days 2–6, parallel** | T-04 → **T-04a** · T-07 · T-22 · T-13 | none depends on T-03. **T-04a gates every production distance matrix**, so it cannot slip past T-06 |
| **Day 5 — 08-16** | **S-h** (bibliography arithmetic) | T-19's search strategy branches on it, and T-19 feeds T-08 → T-26 → T-15 |
| **Days 5–8** | T-05 calibration arm · **MRM (D4)** · **AIDS density stratification** | **both can refute a central claim**; a refutation on day 15 has no absorption time |
| **Day 7 — 08-18** | **S-d** — the labels tier | Tier 2 must be configured into the T-06 run, not bolted on afterwards |
| **Day 9 — 08-20** | E1–E12 disclosure structure fixed | fixes the letter's structure before assembly |
| **Days 8–12** | **T-06 full recompute** · T-18 · T-17 · T-08 → T-19 | |
| **Day 11 — 08-22** | R3.2 contingency go/no-go — **only if S-f's extension was granted** | otherwise the decline stands and this date is struck |
| **Days 12–17** | **T-20 manuscript rewrite** · T-11 (**including E7, before any trim**) · T-12 · T-21 | |
| **After T-08 + T-19, before T-15** | **T-26** — slot and page-budget reconciliation | the two arithmetics the **EiC checks independently**. There is otherwise no gate between the tickets that *spend* slots and the trim that discovers the overrun |
| **Days 17–19** | T-15 page trim · T-14 letter assembly | fragments have accrued since day 2 |
| **Day 20 — 08-31** | **T-24 package uploaded** | |

---

## 3. What gets cut, in order

Decided now so it is not decided under pressure on day 17.

> **Restated in the currency that binds.** This list is denominated in **days**, but
> [manuscript](manuscript.md) §3 measures a **≈ 8-page gap that "cannot be closed by editing"**
> against a document at exactly 35/35. **Days are recoverable by an extension. Pages are pass/fail at
> the EiC and no extension returns any.**

| # | Item | Days | **Pages** | Note |
|---|---|---|---|---|
| 1 | **[labels](labels.md) Tier 2** — the logged label-aware GED arm | 0.5 | 0 | Tiers 0–1 answer R1.3 without it |
| 2 | **GEDLIB per-dataset cost-model sensitivity arms** | ~0 | ~0.5 | cheap to run, expensive in pages |
| 3 | **Exhaustive-canonical baseline at scale** | ~0.5 | ~0.5 | fails on 55 % of Protein graphs — report it as a bounded baseline in one row rather than a full arm |
| 4 | **S2G/G2S worked example** (half of T-09) | 0.5–0.75 | **0.75** | author preference, **answers no demand**. ~9 % of the page gap |
| 5 | **[labels](labels.md) Tier 1** collision table | ~0.4 | 0.75 | driver is R1.2/AE.3, **not** R1.3 — cut only if pages force it |
| 6 | **Search-space schematic** (other half of T-09) | ~0.3 | 0.5 | **R3.7c, requested** — cut last among these; the renderer already exists |

**The list totals ~2 days against a 7.5-day lower-bound overrun. Cutting to fit will not work** —
that is a finding, not an opinion, and it is why S-f recommends an extension plus staging T-03.

**T-16 is not on this list**: it is **rejected**, not deferred, and its 3–4 days are already back.

### Component items below whole-ticket granularity

Whole-ticket cut orders cannot see these:

- **bliss / Traces backends — 1.0 d**, in no `ReprBackend` row, duplicating nauty. **S-g.**
- **T-22's `tests/property/` directedness-collision regression** — unasked-for, cheap, drops first if
  T-22 overruns. Cost of dropping: hours, no manuscript content.
- **The AGM `ReprBackend`** — 1 d. **Not recommended**; it is the correct next candidate if T-04
  slips. See [competitors](competitors.md) §2.

Below these, every remaining ticket is the sole owner of at least one numbered demand
([demands](demands.md)).

---

## 4. Risk register

| # | Risk | Trigger | Mitigation, decided now |
|---|---|---|---|
| **R1** | **T-03 does not finish.** 985–1,640 core-hours of it is AIDS alone; queue time is unbudgeted | no result by day 10 | **Superseded by decision 21** — stage 1 is the reported analysis and lands in hours, so the census slipping is no longer a failure mode. Re-cost stage 1 from [exact_ged](exact_ged.md) §2, **not** from the superseded pair table that mixes populations |
| **R2** | **The MRM or the AIDS stratification refutes `conclusion.tex:30–36`** | β₁ collapses, or ρ does not recover on sparse AIDS strata | **This is a *result*, not a failure**, and the interpretation is fixed in advance ([statistics](statistics.md) §6, §8). Reserve days 13–15 for the rewrite. Running these in week 1 exists precisely to buy that time |
| **R3** | **Supplementary material counts toward the 35 pages** | reply from patcog@elsevier.com | [manuscript](manuscript.md) §3.2's pre-declared priority ranking. Items 10 and 11 are the only two no reviewer requires and are cut first |
| **R4** | **gSpan's minimum DFS code is not exposed** by `LasseRegin/gSpan` | day 1 of T-04 | Extract or reimplement within the same 2–3 day budget. If it slips, gSpan is **discussed** and the running comparator set drops to nauty-graph6 + sparse6 + AGM. **R1.2 is answered by citation and comparison either way; only the empirical row is lost** |
| **R5** | **Page overflow discovered at T-15** | count > 35 on day 17 | Track the page count at **every commit** from the moment the first new section lands, not at the end |
| **R6** | **Round-2 rejection on R3.2** | out of our control | The delta table pre-empts the reading, the contingency exists, and the letter frames the AE.3-over-R3.2 choice as the deliberate exchange it is |
| **R7** | **`pynauty` fails to build on Picasso** | day 1 of T-04 | Takes the **graph6 and AGM rows down with it** — this is the counter-case for the S-g bliss/Traces cut. Verify the build before signing S-g |
