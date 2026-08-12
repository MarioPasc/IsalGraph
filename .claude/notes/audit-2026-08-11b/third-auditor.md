# Third-auditor review of `audit-2026-08-11b`

**Object under audit**: the wave `audit-2026-08-11b` (`inventory.md`, `findings-r1.md`,
`findings-r3.md`, `findings-editors.md`, `findings-integrity.md`, `summary.md`) **and the edits it
made** to `plan.md` (v0.8), `data.md` (v1.2), `statistics.md` (v2.2).
**Question asked**: did the second auditor **overstate** or **understate** its findings, measured
against `mail.txt` and the manuscript sources?
**Date**: 2026-08-12. **Method**: independent re-derivation. Nothing below is inherited from the
wave's verdicts; where I reproduce one of its numbers I re-measured it myself.

---

## 0. Headline

The wave is **substantively sound and unusually well evidenced**. Every claim I could re-measure
independently reproduced — the bibliography arithmetic, the venue audit, the page count, the missing
scripts, the critical-path sum, and [29]'s publication status. Its two OVER findings are real and its
one coverage hole (R3.1a(ii)) is real.

It nevertheless **understates over-scope in one large place and overstates in five small ones**, and
it introduced **three consistency defects of exactly the class it was hunting**.

| Direction | Item | Magnitude |
|---|---|---|
| **UNDERSTATED** | T-03's all-pairs AIDS census is disproportionate to every demand that drives it, and the plan's own D2 proves it | **~900–1,550 core-h; 2–5 elapsed days off the long pole** |
| **UNDERSTATED** | Only ~4 of the 24 standing integrity defects reach a printed number; the other 20 are internal hygiene and were not triaged | ~0.5 d of misallocated effort |
| **OVERSTATED** | **I-11** — "AIDS 131,148 unsourced, contradicts F2's 181,909". The two count **different populations**; applying this finding would corrupt a correct ratio | would inject an error |
| **OVERSTATED** | **I-09's consequence** — "this is the number that sizes the heavy-tail strata T-02 must freeze". It sizes nothing; §6.1 already used 98 | make-work for T-02 |
| **OVERSTATED** | **AE.5** at 0.25 d — its only unowned clause ("rationale") is the same paragraph as R3.1a(ii) | double-counted |
| **OVERSTATED** | **T-25** carried onto the board at 0.5–1 d while the same file recommends the ~1 h path | +0.9 d on a board declared infeasible |
| **WRONG** | **The corrected board total is itself wrong** — 91.0 / 52.8 omits T-24, T-25 and T-26 | +2.5 d / +2.0 d |

---

## 1. UNDERSTATED — the largest item in the revision is still the largest item

### 1.1 What the wave concluded

`findings-r3.md:115` returns **R3.5b = COVERED, not OVER**, on the cut-guard reasoning that the
1,000–1,650 core-hour recompute has "two verified non-R3.5b drivers": F2's 113,387 missing AIDS pairs
(378–630 core-h) and Cor. 2.13's pseudometric problem. Both drivers are real; I re-derived the
113,387 and the 378–630 and they are correct.

**But driver ≠ dose.** The wave verified that *something* justifies recomputing AIDS GED. It never
asked how *much* AIDS GED that something justifies. Those are different questions, and on this board
the difference is 98 % of the compute budget and the entire critical path.

### 1.2 The plan already contains the proof that the census is unnecessary

This is not an opinion. It is an internal contradiction between two locked decisions:

- **`statistics.md` D2** (locked, and the literal answer to R3.5c) resamples **graphs**, not pairs,
  precisely because pairs are dyadically dependent: `d(G₁,G₂)` and `d(G₁,G₃)` share `G₁`. The
  independent unit is the graph.
- **`plan.md` §12.2 risk R1** states the consequence in the plan's own words: "effective sample size
  is governed by the **number of graphs**, so very little power is lost."

AIDS contributes **769 graphs** whether we compute 131,148 pairs or all 295,296. A graph-level
bootstrap CI is governed by resampling those 769 units; enlarging the induced pair set per replicate
does not add independent information. **If D2 is right, the census buys no precision. If the census
buys precision, D2 is wrong and R3.5c is unanswered.** The plan cannot hold both.

What the census *does* buy, stated fairly: (i) the pair-accounting ladder can say "all pairs" rather
than "a stratified sample", which is rhetorically stronger against R3.5a; (ii) no sampling design to
defend; (iii) more pairs inside each density stratum for §8's AIDS test — though the independent-unit
count per stratum is again the number of graphs in it.

### 1.3 What no reviewer asked for

| Demand | What it actually needs |
|---|---|
| R3.5b (`:104`) | one caveat + demote the pooled β. **D5, ≈ 0.5 h** — the wave established this and it is now the recorded floor |
| R3.5a (`:102`) | *counts* of excluded pairs, not GED values for all of them |
| R3.5c (`:106`) | a graph-level bootstrap — which is indifferent to the pair census |
| R1.3a (`:79`) | within-AIDS density stratification — needs strata populated, not saturated |
| AE.1 (`:59–60`) | the Suite-2 size extension, which costs **1.24 core-h** |

The census is driven by **E2/F2 and D6 — both self-found**. That is a legitimate reason to do work;
it is not a reason to do the maximal version of it under a 19-day clock.

### 1.4 Recommendation — better than either branch of S-f

S-f frames this as *A: subsample now* versus *D: accept and absorb*. Both are worse than decoupling
the analysis from the census:

> **Two-stage T-03.** Stage 1: a stratified sample of AIDS pairs spanning all 769 graphs and all
> density strata (~100 core-h, hours not days), **pre-declared as the reported analysis**. Stage 2:
> the full census runs unattended behind it, and **supersedes stage 1 only if it lands before the
> T-20 text freeze**. The supersession rule is fixed in advance so the choice cannot be made after
> seeing which ρ is more favourable.

This removes T-03 from the critical path without giving up the census, keeps the "all pairs" claim
available if the cluster cooperates, and costs one paragraph of protocol. It is the single largest
schedule lever on the board and it is compatible with S-f option B (the extension request).

**Caveat on costing**: `findings-integrity.md` I-03 shows `data.md` §3.1 — the table risk R1's ~100
core-h fallback is costed against — mixes n≥1 and n≥2 populations. Re-cost stage 1 from
`plan.md` §7.1's table, which I-03 itself certifies as using the correct population.

---

## 2. OVERSTATED — I-11 would inject the error it is trying to remove

**Finding as written**: `plan.md:272`/`:949`'s "AIDS 295,296 (from **131,148**, 2.25×)" is unsourced
and contradicts F2's measured 181,909; "on F2's number the gain is 1.62×".

**Re-derivation** (exact, no measurement needed):

```
C(769, 2)                             = 295,296   ← the printed "after"; population = 769 FILTERED graphs
C(546,2) + C(182,2) + C(183,2)
  = 148,785 + 16,471 + 16,653         = 181,909   ← F2's number; population = 911 RAW graphs, within-split
```

The two numbers are computed over **different populations**. Substituting 181,909 into the ratio
divides a filtered count by an unfiltered one — which is `gap-audit.md` **MF1's defect class**, the
very pattern the wave nominates as its "pattern worth carrying forward".

The population-matched comparator is *within-split pairs restricted to the 769 filtered graphs*.
Under proportional retention (769/911 = 84.4 %) that is
`C(461,2)+C(154,2)+C(154,2) ≈ 129,600` — **within 1.2 % of 131,148**. The printed number is almost
certainly the submitted study's *analysed* AIDS pair count, and it is the right one for that ratio.

**Disposition**: downgrade I-11 from "major, contradicts F2" to **"minor — provenance not recorded"**.
Record the source when the run is reproduced. **Do not substitute 181,909, and do not print 1.62×.**
I-11 is currently listed in `data.md`'s changelog under "outstanding defects recorded but not yet
applied", so applying it as written is a live risk.

---

## 3. OVERSTATED — I-09's correction is right, its consequence is invented

`statistics.md:116` did print `max n = 417` and **98 is correct** — that half stands and I reproduced
it. But the banner the wave wrote asserts:

> "**This is the number that sizes the heavy-tail strata T-02 must freeze before T-06 runs**, so
> re-check the stratum boundaries against 98 rather than 417."

The line sits in **§3, Claim A**, and its whole function is to justify *reporting dispersion*
("Never report a mean bit count without dispersion: length distributions are right-skewed"). It
defines no stratum. The stratification is defined in `plan.md` §8 (by node count and true density)
and `statistics.md` §7; the size-regime boundary lives in **§6.1, which already reads `n ≤ 98`** —
v2.1 wrote it that way before this audit existed. Nothing derives a boundary from the maximum.

Left as written, the banner sends T-02 to re-check boundaries that were never wrong, inside the
document the plan calls locked.

**The real consequence, which the wave missed**: at max 98 rather than 417 the skew is 3.6× the
median, not 15.4×. The sentence's *argument* is weaker than it was, though still true. That is what
should be re-checked — the claim, not the strata.

---

## 4. WRONG — the corrected board total is wrong in the same way as the number it corrects

I-06 correctly shows that §12's "76.5 days" is the v0.5 board. Its replacement, **91.0 upper / 52.8
lower**, is itself short. Parsing §7's own Days columns programmatically:

| | Wave | Re-measured | Delta |
|---|---:|---:|---:|
| Board, upper bound | 91.0 | **93.5** | +2.5 |
| Board, lower bound | 52.8 | **54.8** | +2.0 |

The omissions are **T-24 (1 d)**, **T-25 (0.5–1 d)** and **T-26 (0.5 d)** — the wave summed
T-01…T-23 and stopped, so it excluded the submission package *and the two tickets it created in the
same revision*. This is I-06's own failure mode: a total that predates the tickets added alongside it.

Confirmed unchanged: the declared critical path sums to **27.5 d lower / 44.5 d upper**. Adding T-25,
which the wave itself makes a day-1 gate on T-03, takes the lower bound to **28.0 d** against a
19-day window.

---

## 5. OVERSTATED — three small effort inflations on a board declared infeasible

1. **AE.5 at 0.25 d.** AE.5 (`mail.txt:69`) is a catch-all; the wave's own analysis finds exactly one
   clause not subsumed — "**rationale**" from R3's preamble (`:83`) — and then houses it in "the §2.x
   closing paragraph of R3.1a(ii)". That is the *same paragraph* R3.1a(ii) already buys at ~2 h.
   AE.5's marginal cost is **one verification pass inside T-14, ≈ 0**, not 0.25 d.
2. **T-25 at 0.5–1 d.** S-e recommends **"C, then B"** — a 20-pair spot-check (~1 h) followed by
   retiring gate 2 on the record. The board nevertheless carries T-25 at 0.5–1 d, which is option A.
   Book the recommended path (**0.1–0.2 d**) with A named as the escalation if the PI chooses it.
3. **R3.1a(ii) as "UNDER-blocking".** The coverage finding is right and important — I verified
   `introduction.tex:55–62` asserts three contributions and none argues sufficiency against [28]/[29].
   But a 2-hour paragraph inside a ticket that already exists is not *blocking* in the sense T-23 and
   T-25 are (nothing waits on it). The severity label conflates "high consequence if forgotten" with
   "gates other work". Keep the priority, drop the word.

---

## 6. Consistency defects the wave introduced

All three are instances of "corrections that update a number but not what was derived from it" — the
wave's own **second pattern**.

| # | Defect | Where |
|---|---|---|
| **X-1** | Decision 9 now states [29] is published (**verified: JCIM 65(15):7936–7955, 2025, key `ThurnhoferHemsi:2025`, already cited**), but **§5.1 still reads "the paper is unavailable"** and instructs T-07 to work from source code | `plan.md:51` vs `plan.md:474–480` |
| **X-2** | `data.md` was bumped to v1.2 and `statistics.md` to v2.2, but plan.md's companion list still points at **v1.1** and **v2.1**, and §0's status line still says "statistics locked (`statistics.md` v2.1)" | `plan.md:3`, `:22`, `:24` |
| **X-3** | The v0.8 banner reads "41 demands, **40 covered**, one genuine hole" — the pre-fix state. R3.1a(ii) and AE.5 are now owned, and R3.1a is split in two, so the matrix carries 42 rows and all are covered | `plan.md:15` |

---

## 7. UNDERSTATED — 24 integrity defects, untriaged

The wave records 24 standing defects with severities but no answer to the only question that matters
under a 19-day clock: **which of them can reach a number a reviewer will read?**

| Class | IDs | Action |
|---|---|---|
| **Reaches a printed number — fix** | **I-02** (AIDS raw 819 vs 911 → Tab. 2's retention row and R3.5a's ladder), **I-03** (inflated pair counts → the fallback costing), **I-05** (Fingerprint 2.3× is a counter-example to its own sentence, and §2.2.1 feeds the discarded-subset columns), **I-08b** (uncommenting Fischer/Lerouge takes headroom 12 → 10) | 4 items, ~1 h |
| **Do not apply as written** | **I-11** (§2 above) | re-word only |
| **Internal document hygiene — batch, or leave** | I-04, I-06, I-07, I-09-consequence, I-10, I-12…I-25 | one pass, ≤ 1 h, after the manuscript work |

Without this split the natural reading is "24 defects to fix before executing", which is ~0.5 d of
misallocated effort on a board with none to give.

---

## 8. Where the wave was right, re-measured independently

Recorded so these are not re-litigated:

- **Bibliography**: 43 cited with comments stripped, 45 with them included, the two extras being
  `Fischer2015hausdorff` and `Lerouge2017ilp`; 56 `@`-entries; **13 dead**; **12 slots**. The
  rejection of I-08 was correct, and I-08b is real.
- **EiC.a2 recency**: of 43 printed refs, **5 postdate 2023** (2024 ×3, 2025 ×2) and **both 2025
  entries are group self-citations** (`lopezrubio2025isalgraph`, `ThurnhoferHemsi:2025`). Zero
  third-party after 2024, zero from 2026. T-19's criterion must exclude self-citation — confirmed.
- **EiC.b venues**: *Pattern Recognition* journal ×6 (2021–2023), PR Letters ×1 (**1983**), SSPR ×1
  (2008), **zero** CVPR/ICCV/ECCV/ICPR/TPAMI/IJCV, **no PR-field reference after 2023**. Exactly as
  reported.
- **EiC.c**: `main.pdf` is **35 pages of 35**. The constraint is binding, not notional.
- **I-01**: `find / -name 'ged_bounds.py'` returns nothing. Confirmed.
- **Decision 9 correction**: [29] is published, entry verified in `cas-refs.bib`.
- **S-g bliss/Traces**: absent from the `ReprBackend` set at §4.1, present only in §4.2's effort
  column at "0.5 d each". The 1.0 d cut is real and costs nothing.
- **R3.1a(ii)**: the second conjunct is verbatim in `mail.txt:86` and unowned before this wave.

---

## 9. Two items neither audit priced

1. **Pages, not days, are the binding currency.** `manuscript.md` §3 measures a **≈ 8-page gap that
   "cannot be closed by editing"** against a document at 35/35. The wave's S-g argues the T-09 split
   in **days** (0.5–0.75 d); the S2G/G2S figure's real cost is **0.75 page** from `manuscript.md` §2's
   own inventory. Restate the cut in pages — it is ~9 % of the gap, and it is unrequested work.
2. **The AGM empirical row.** The wave demoted R1.2a's owner from T-04 to T-08 because "the ask is
   *discussion*". The same logic applies one step further: AGM's `ReprBackend` implementation (1 d,
   §4.2) exists to fill a measured column, but R1.2a is answered by citation and AE.3 by a
   *qualitative* properties row in T-17. **Not a recommended cut** — AGM is named by R1 and a measured
   row is more defensible than an asserted one — but it is the correct **next** item after
   bliss/Traces if T-04 slips, and §12.3 should say so rather than declaring nothing below the line
   cuttable.

---

## 10. Changes made to the plan files

| File | Change | Finding |
|---|---|---|
| `plan.md` §7 banner | board total corrected to **93.5 / 54.8**; T-24/T-25/T-26 omission recorded | §4 |
| `plan.md` §7.1 | two-stage T-03 recorded as the recommended structure, with the D2 contradiction stated | §1 |
| `plan.md` §11 S-f | two-stage T-03 added as option **E**, recommended alongside B | §1 |
| `plan.md` §3.2, §8 | 131,148 protected with a provenance note; 1.62× explicitly rejected | §2 |
| `plan.md` §5.1 | rewritten — [29] is published | X-1 |
| `plan.md` §3, §0 | companion versions corrected to v1.2 / v2.2 | X-2 |
| `plan.md` §0.5 AE.5 | effort corrected to ≈ 0, subsumed by R3.1a(ii)'s paragraph | §5.1 |
| `plan.md` §7 T-25 | board cost set to the recommended path | §5.2 |
| `plan.md` §11 S-h | a concrete 12-slot allocation supplied | §8 |
| `plan.md` §12.3 | cut order restated in pages; AGM named as the next component item | §9 |
| `plan.md` §6.1 | R3.2 contingency tied to S-f's outcome | below |
| `statistics.md` §3 | I-09 banner's spurious T-02 consequence removed, correct consequence substituted | §3 |
| `data.md` changelog | I-11's disposition corrected | §2 |

**R3.2 contingency**: §6.1 sets a go/no-go for a minimal Transformer arm on **2026-08-22**,
conditional on "T-03 has finished and the critical path has slack". §7's own banner establishes there
is no slack — the path is 27.5 d minimum in a 19-day window. As written the date is unreachable and
will consume a decision cycle on day 11 to conclude what §7 already knows. Tied to S-f: the
contingency is live **only if the extension is granted**.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-12 | Created. Third-auditor pass over `audit-2026-08-11b`. 1 major understatement (T-03 census), 1 misdirected finding (I-11), 1 invented consequence (I-09), 1 arithmetic error (board total), 3 effort inflations, 3 consistency defects. Wave's bibliography, venue, page, script-absence and coverage findings independently reproduced and confirmed. |
