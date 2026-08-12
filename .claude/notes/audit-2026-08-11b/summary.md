# Audit summary — `audit-2026-08-11b`

**Manuscript**: PR-D-26-03293, *Representation of Graphs by Sequences of Instructions*, Pattern
Recognition (Elsevier). Major revision, **due 2026-08-31**.
**Wave**: `audit-r1`, `audit-r3`, `audit-editors`, `audit-integrity` — four read-only agents, disjoint
slices, all against a demand inventory the orchestrator rebuilt from `mail.txt` before any agent
existed.
**Sources readable**: manuscript **yes** — so no "already satisfied" check downgraded to UNVERIFIED.

## Coverage, before and after

| | Before | After |
|---|---|---|
| Demands in the letter | 40 (per §0.5) | **41** — `inventory.md`, rebuilt from `mail.txt` |
| Rows with an owner | 40 | 41 |
| Genuine holes | 0 claimed | **1 found** (R3.1a(ii)), now owned by T-07 |
| Rows the letter does not contain | not checked | **2** (M3, R1.3b) — retained, relabelled |
| Rows missing from the matrix | not checked | **1** (AE.5) — added |
| Compliance rows verified by measurement | asserted | **all re-derived** |

**UNDER: 5.** One blocking hole (R3.1a(ii)) plus four compliance gaps, three of them
`UNDER-blocking` because the Editor-in-Chief checks them independently of the reviewers
(`mail.txt:124`).
**OVER: 2**, worth **1.0–1.5 days**. Both are *sub-ticket* items, which is why §12.3's whole-ticket
cut order could not see them.
**Integrity: 24 defects stand, 1 rejected** on orchestrator re-measurement.

## What the audit did not find, and that matters

**Both rows the README nominated as likely over-scope survive.** `R3.5b`'s 1,000–1,650 core-hour
recompute has two verified drivers that hold independently of it — F2's 473,147-pair GraphEdX
coverage gap (~378–630 core-h of AIDS alone) and Cor. 2.13's pseudometric problem — and, contrary to
the hypothesis in the wave brief, the large-`n` cohort is **not** one of them: the whole Suite-2
extension costs **1.24 core-hours**. `R3.6a`'s expensive branch is owned by AE.4a, not by R3.6a, and
the free branch is already taken unconditionally. **The plan was right on both, for reasons it had
not fully written down** — and writing them down is the fix.

Twelve of R3's nineteen rows and all seven of R1's are clean COVERED at proportionate cost. The
`labels.md` tiering in particular gets R1.3b exactly right: every Tier 0–1 item has a driver other
than the premise it is filed under.

---

## Decision queue

| # | Decision | Owner | Due | Blocks |
|---|---|---|---|---|
| **S-e** | Validation gate 2 — restore, spot-check, or retire on the record | PI | **2026-08-13** | T-03 production |
| **S-f** | The schedule does not fit — extension, subsample, cut, or absorb | PI (Ezequiel sends) | **2026-08-13** | everything downstream of T-06 |
| **S-g** | Two over-scope cuts: bliss/Traces (1.0 d); split the T-09 bundle | PI | **2026-08-14** | T-04 backend build |
| **S-h** | Bibliography: 16–17 slots requested against 12 | PI | **2026-08-16** | T-19 search strategy |
| **S-d** | `labels.md` tier (pre-existing) — now with the audit's note that **no reviewer asked for Tier 2** | PI | 2026-08-18 | T-06 configuration |

### S-e — validation gate 2 is not executable

`scratchpad/ged_bounds.py` **does not exist**, and neither do 12 of the 16 measurement scripts
`data.md` §8 names. Gate 2 of three requires GEDLIB to reproduce it on 300–400 pairs; it gates T-03
production, the long pole. Collateral: `data.md` §5 / H4 — ρ(exact, LB) = 0.966 vs ρ(exact, UB) =
0.840 — is the evidence for "BRANCH-FAST is the primary large-`n` reference", and that evidence is
now unreproducible from any surviving artifact.

**Recommended: spot-check 20 pairs against `networkx` (~1 h), then retire gate 2 on the record.**
Gate 1 (bracket validity) already catches the failure mode that matters most. **Counter-case**:
rewriting the script (0.5–1 d) is the only option that restores the evidence behind the primary-
reference decision.

### S-f — the board does not fit the window

§12's "76.5 days" is the **v0.5** board — it reproduces exactly as T-01…T-15 plus the now-rejected
T-16, predating the eight tickets the v0.6 audit added, yet it is attributed to that audit. The
current board is **91.0 days** upper / 52.8 lower. Worse, the declared critical path is serial and
sums to **27.5 days at lower bounds** against a 19-day window; §12's "most tickets parallelise"
mitigation does not apply to a critical path. §12.1 also allocates T-05 four days against a 5–10 day
minimum and T-06 five days against 10–14.

**Recommended: request an extension immediately** (nearly free, pairs with decision 16's query to the
same mailbox), **with risk R1's pre-approved subsample as the technical fallback**. **Cutting to fit
will not work**, and that is a finding rather than an opinion: §12.3's four items return ~2 days
against a 7.5-day minimum overrun.

### S-g — the two OVER findings, 1.0–1.5 days

- **bliss/Traces backends, 1.0 d** — absent from the `ReprBackend` set, functionally duplicate nauty
  (all three emit a canonical labelling serialised to graph6, differing in speed not representation),
  produce no table row, requested by nobody. *Counter-case*: cheap insurance if `pynauty` fails to
  build, which would otherwise take the graph6 and AGM rows down with it.
- **T-09 bundle, 0.5–0.75 d** — merges the R3.7c search-space schematic (requested, renderer already
  written, ~2–3 h) with the S2G/G2S worked example (author decision 6, **requested by nobody**).
  Bundling protects the unasked-for figure behind the requested one. *Counter-case*: both feed the
  graphical abstract — but that argument does not distinguish them, and §12.3 currently uses it to
  protect both.

### S-h — the bibliography arithmetic

**16–17 slots requested against 12**, and §5.4's stated relief frees nothing: `elsarticle-num` prints
only *cited* keys, so the 13 dead `.bib` entries never occupied a slot. Separately, **of 43 printed
references only 5 postdate 2023, and both 2025 entries are group self-citations** ([28] and [29]) —
zero third-party references after 2024, none from 2026, and no pattern-recognition-venue reference
after 2023.

**Recommended: accept 55 as the working ceiling, re-scope T-19 to ≥ 4 additions dated 2025–26 with
at least 3 at PR-field venues other than the PR journal, and exclude self-citations from the count.**
As written, T-19's criterion is satisfiable without adding a single external reference.

---

## UNDER findings

| ID | Class | What is missing | Owner now | Effort |
|---|---|---|---|---|
| **R3.1a(ii)** | UNDER-blocking | "…**and explain why the combined extension constitutes a sufficiently substantive contribution**" — the second conjunct of R3's strongest-modal sentence. Unowned in every document; §0.5 quoted only the first conjunct | **T-07** | ~2 h, ≈0.1 p |
| **EiC.a1** | UNDER-blocking | Slot allocations sum to 16–17 against 12; no ticket sees the total | **T-26** | 0.25 d |
| **EiC.b** | UNDER-blocking | "a stated PR-community share" fixes no threshold, so no outcome can fail it | **T-19** | 0.5 d |
| **EiC.c** | UNDER-blocking | The page budget is not derivable — over-counts replacements ≈3.5–4.5 p, omits ≈4.8–5.8 p of committed main-text prose | **T-26** | 3 h |
| **AE.5** | UNDER-major | No row at all. Mostly subsumed, but R3's preamble (`:83`) names **"rationale"**, which has no owner | **T-14** + T-07 | 0.25 d |

**R3.1a(ii) is the one that would have cost most.** `plan.md` §6.1 already notes the delta table will
document that both predecessors ran a sequence model and this paper does not. Delivering the
inherited/modified/new table *without* the sufficiency argument hands R3 the conclusion that the
extension is **less** substantive — the artifact becomes evidence against us.

## Corrections to locked decisions

| What | Was | Now |
|---|---|---|
| `statistics.md:116` | `max n = 417` — a **raw-set** value inside the locked protocol | **98**. Sizes the heavy-tail strata T-02 must freeze before T-06 |
| `data.md:438` | live "**Recommendation** — build `wl_pruned_canonical` (T-16)" | SUPERSEDED banner; T-16 rejected by signed decision 17 |
| `plan.md` §7.3 gate 2 | "GEDLIB must reproduce `ged_bounds.py`" | flagged unexecutable; T-25, S-e |
| `plan.md` §0.5 R1.3b | booked as a demand with a ticket | **PREMISE** — served by R1.3c / R1.2b / AE.4b, no independent allocation |
| `plan.md` §0.5 M3 | authority cited as `:67` | **MISMATCHED** — `:67` is the AE priority statement; T-14 stands on convention instead |
| `plan.md` §0.5 R1.2a | owner T-04 + T-08 | **T-08**; T-04 demoted to enrichment — the ask is discussion |
| `plan.md` §0.5 R3.5b | titled "Heterogeneous cost models" (the premise) | retitled to the ask; **D5 named as the zero-compute floor** |
| `plan.md` §0.5 E6 | claimed by T-12, T-18 and §9/T-11 at once | **T-12** |
| `plan.md` decision 9 | "[29] paper unavailable" | **published** — *JCIM* 65(15):7936–7955, 2025. T-07 gets cheaper; D19 becomes directly resolvable |
| `plan.md` §12 | "76.5 days" | **91.0**; critical path 27.5 d serial vs a 19-day window |

## Rejected finding

**I-08 (bibliography counts) — rejected on orchestrator re-measurement.** `audit-integrity` reported
45 cited / 11 dead / 10 slots. Measured both ways: **43** with comments stripped, 45 with comments
included, the two extras being `Fischer2015hausdorff` and `Lerouge2017ilp`, cited only from
commented-out LaTeX at `methodology.tex:805–806`. `elsarticle-num` prints only uncommented `\cite`
keys, so the plan's **43 / 13 dead / 12 slots** stands. `audit-editors` independently measured 43 and
agrees.

The arbitration surfaced **I-08b**, which does stand: §0.5's EiC.b row reasons about **Fischer 2015**
as an existing printed reference whose only problem is recency. It is not in the printed bibliography
at all.

## The pattern worth carrying forward

**Eight of the 24 standing integrity defects are one defect** — a statistic computed over one
population and printed under another's header. `gap-audit.md` MF1 identified the mechanism and
corrected three call sites; **it recurs across five documents**, including inside `data.md` §0 and
`statistics.md`, the two tables the plan declares authoritative. The generalisation that finds them
all is the four-population table: print every candidate statistic under
`RAW` / `CONN_ge1` / `KEPT_ge2` / `DISC_ge2` and match each against its header, rather than checking
only the sites already suspected. `<scratchpad>/remeasure.py` does this and is reusable.

**Second pattern**: corrections that update a number but not what was derived from it — I-05 (a
ratio still computed from a retracted mean, making the cited example a counter-example to its own
sentence), I-06 (a day sum), I-08b (a slot budget), I-09, I-10, I-17, I-22.

## Wave notes

- **All four agents were blocked from writing their own findings files** by a harness hook. Three
  returned findings as text and the orchestrator persisted them; `audit-editors` reached the file via
  Bash heredoc. `audit-editors` was also stopped mid-write and resumed via `SendMessage` rather than
  re-run, preserving ~16 minutes of verification.
- **Cross-slice dependency resolved**: `audit-r1` made R1.1's verdict conditional on what AE.4a's
  "choice of benchmark models" refers to. `mail.txt:66` names "the graph datasets used" as a
  *separate* clause in the same sentence, so it cannot mean datasets; the orchestrator and
  `audit-editors` reached this independently. **The six competitor backends have a requirement-modal
  owner.**
- **The README's own R1.3 lesson has a defect** (`README.md:136–154`): it recommends answering R1.3
  by pointing at `conclusion.tex:70–71` and `:81`, and **both sentences are false** — they claim
  labels are "present in all five benchmark datasets", which is wrong for LINUX. `labels.md` catches
  this as E6; the README does not, so anyone scoping from the README alone reproduces the error.

## Files

`inventory.md` · `findings-r1.md` · `findings-r3.md` · `findings-editors.md` · `findings-integrity.md`
Reconciled into: `plan.md` (v0.8) · `data.md` (v1.2) · `statistics.md` (v2.2)

## Changelog

| Date | Change |
|---|---|
| 2026-08-12 | Wave complete. 41 demands audited, 5 UNDER, 2 OVER, 24 integrity defects standing, 1 rejected. Five decisions queued (S-d…S-h). |
