# Findings — cross-document integrity · audit-2026-08-11b

**Agent**: `audit-integrity` · **Owns no reviewer demands** · **Date**: 2026-08-11

> **Provenance.** The agent's write was blocked by a harness hook; the orchestrator persisted the
> content. Phase-5 arbitration by the orchestrator is marked `[ORCH-VERIFIED]` or `[ORCH-REJECTED]`.
> **One finding (I-08) was rejected on re-measurement** — see below.

## Defect table

| ID | Severity | Basis | Summary | Orchestrator |
|---|---|---|---|---|
| **I-01** | **blocking** | MEASURED | `ged_bounds.py` gone; 13 of 16 named scripts absent | **CONFIRMED** |
| I-02 | major | MEASURED/INFERRED | §0's AIDS "raw" column is the connected count (819 ≠ 911) | **CONFIRMED** |
| I-03 | major | MEASURED | `data.md` §3.1 mixes n≥1 and n≥2 pair counts | accepted |
| I-04 | major | MEASURED | §2.1 Fingerprint `N conn` wrong population, off by one | accepted |
| I-05 | major | MEASURED | Retracted 11.56 survives as "Fingerprint 2.3×" | accepted |
| I-06 | major | MEASURED | "76.5 days" is the pre-audit board; true 91.0 | accepted |
| I-07 | major | MEASURED | T-05/T-06 allocated below minimum; critical path 27.5 d > 20 d | accepted |
| ~~I-08~~ | ~~major~~ | MEASURED | ~~Bibliography: 45 cited not 43; 11 dead not 13~~ | **REJECTED** — see arbitration |
| I-09 | major | MEASURED | `statistics.md:116` still prints retired `max n = 417` | **CONFIRMED** |
| I-10 | major | MEASURED | `data.md:438` still recommends rejected T-16 | **CONFIRMED** |
| I-11 | major | INFERRED | AIDS "131,148" unsourced, contradicts F2's 181,909 | accepted |
| I-12 | minor | MEASURED | H3's "10 datasets, 67 M pairs" conflates two disjoint tens | accepted |
| I-13 | minor | MEASURED | "1.3 core-hours for 40 M pairs" is a 67 M figure | accepted |
| I-14 | minor | MEASURED | "10 s (0.008 core-hours)" disagrees by 2.9× | accepted |
| I-15 | minor | MEASURED | H8 says 5 seconds; three places say 5 minutes | accepted |
| I-16 | minor | MEASURED | `data.md:373` "n = 78" is n = 81 per §4.4 | accepted |
| I-17 | minor | MEASURED | `plan.md:27` says 16 MFs; gap-audit defines 17 | accepted |
| I-18 | minor | MEASURED | M2 and R3.2 rows carry no ticket | accepted |
| I-19 | minor | MEASURED | E6 owned by T-12, T-18 and §9 simultaneously | accepted |
| I-20 | minor | MEASURED | E12's "two orphaned PDFs" is four | accepted |
| I-21 | minor | MEASURED | F1 mixes raw and connected within one column | accepted |
| I-22 | minor | MEASURED | "pre-reconciliation (open question 3)" inverted, wrong Q | accepted |
| I-23 | minor | MEASURED | `D<n>` namespace shared by two documents | accepted |
| I-24 | minor | MEASURED | `data.md` §4 subsections run 4.1→4.3→4.4→4.2 | accepted |
| I-25 | minor | MEASURED | MF12, MF13, MF17 cited nowhere outside gap-audit | accepted |

**24 defects stand; 1 rejected.** Eight of the 24 are one recurring defect — see **Pattern** below.

---

## Orchestrator arbitration — I-08 REJECTED

`audit-integrity` reported 45 cited keys, 11 dead, 10 slots, against the plan's 43 / 13 / 12. It
noted a caveat — "the count includes any `\cite` in a commented line" — and then carried 45 into the
finding anyway and derived the slot budget from it.

I re-ran the count both ways:

```
comments stripped (sed 's/%.*//'):  43 distinct cited keys
comments included:                  45
difference:  Fischer2015hausdorff, Lerouge2017ilp  — both cited ONLY from commented-out LaTeX
cas-refs.bib:                       56 @-entries
main.pdf:                           35 pages
```

`elsarticle-num` prints only keys reached by an **uncommented** `\cite`. The printed bibliography is
therefore **43**, dead entries **13**, headroom **12 slots**. **The plan's figures are correct and
I-08 does not stand.**

**But the arbitration surfaced a real defect underneath, recorded as I-08b:**

### I-08b — the plan reasons about a reference that is not in the bibliography. Major, MEASURED

`plan.md:85` (§0.5, EiC.b row) reads "⚠ Fischer 2015 satisfies *venue*, **not recency**", treating
`Fischer2015hausdorff` as an existing printed reference whose recency is the only problem. It is
**cited solely from commented-out LaTeX** and is not in the printed bibliography at all. The same
applies to `Lerouge2017ilp`.

Consequences, both live: (i) EiC.b's audit baseline is wrong by one entry; (ii) if either block is
uncommented during the revision — plausible, since both are GED-related and the revision expands GED
discussion — the printed count rises to 45 and **headroom falls from 12 to 10**, against §5.4's own
request of 12–14 slots (`plan.md:470–475`). The overrun the plan states as "~2" becomes **2–4**.
That half of I-08's analysis survives its rejection and belongs to **T-08 / T-19**.

---

## Re-measurement log

**Method.** The two prior scripts were found in *another session's* scratchpad, not in the repo. The
agent generalised them into `remeasure.py`, which re-parses all 10 IAM datasets and emits every
statistic over **four** populations side by side: `RAW`; `CONN_ge1` (connected, n≥1 — §2.1's stated
definition); `KEPT_ge2` (connected, n≥2 — `dataset_filter.py::filter_graphs`, §0); `DISC_ge2`. This
makes the defect class detectable everywhere rather than only where already suspected. Run with
`~/.conda/envs/isalgraph-cpp/bin/python`. Supporting checks: bib-key resolution over
`*.tex` × `cas-refs.bib`; `\includegraphics` vs figures on disk; ticket-day arithmetic;
calendar-vs-board allocation; core-hour re-derivation; `curl` on named repos; existence checks on
every named asset.

**Artifacts kept**: `<scratchpad>/remeasure.py`, `<scratchpad>/remeasure.json`.

### CLEAN results — coverage the orchestrator needs on record

- **`data.md` §0 Suite 1 — every cell EXACT**: 1,180 / 1,253 / 2,059 / 89 / 769; n̄ 4.07 / 4.11 /
  4.58 / 8.71 / 10.56; ñ 4/4/5/9/11; n max 7/8/9/10/12; m̄ 3.07 / 3.17 / 4.56 / 8.35 / 10.70; density
  0.543 / 0.542 / 0.607 / 0.255 / 0.218; pairs 695,610 / 784,378 / 2,118,711 / 3,916 / 295,296;
  totals 5,350 and 3,897,911. Every pair count = C(kept, 2). **§0's authority claim is earned except
  at I-02.**
- **§0 Suite 2 — every cell EXACT**: 19,670 graphs, 40,024,242 pairs; GREC 650 / 11.45 / 24, AIDS-IAM
  1,811 / 14.02 / 85, COIL-DEL 7,200 / 21.48 / 79, Mutagenicity 4,040 / 28.53 / **98**, Protein
  569 / 31.68 / 96.
- **§0 "What this buys"** — all six ratios correct (2×, 3.68×, 10.27×, 8.17×, 3.0×, 6.46×).
- **§2.1 correction banner CONFIRMED**: every size column is a `RAW` value; none reproduces on
  `KEPT_ge2`. The identity `(4040×28.53 + 297×54.70)/4337 = 30.32` holds exactly.
- **§2.1 density convention — not a defect.** It averages over `RAW`, counting n=1 graphs as 0.
  Verified by identity: Fingerprint mean over n≥2 is 0.4285 over 3,061 graphs, and
  0.4285 × 3061/4000 = 0.32791 = the printed 0.3279. COIL-RAG: 0.9299 × 7166/7200 = 0.9255.
- **§2.2.1 — GREC, Letter LOW/HIGH, Mutagenicity, Protein, AIDS-IAM all reproduce.** Letter HIGH's
  discarded n̄ 5.83 is the strictly-disconnected mean: (191×5.62 − 8×1)/183 = 5.822 ✓.
- **§2.2.1's v1.1 Fingerprint correction reproduces exactly**: 51.4 % (2,056/4,000), 5.03 / 5.98,
  0.5505 / 0.1790. **MF17 is sound.**
- **§2.3 drop row reproduces exactly** on `KEPT_ge2`. All three drop decisions survive.
- **`plan.md` §7.1 T-03 table** sums to 993.8–1,648.8 ≈ "1,000–1,650" ✓, and its pair column uses the
  *correct* population (unlike `data.md` §3.1).
- **§12.1 day-to-date mapping internally consistent**: Day 1 = 08-12 ⇒ Day 20 = 08-31 ✓; S-d 08-18 =
  Day 7 ✓; disclosure 08-20 = Day 9 ✓; R3.2 08-22 = Day 11 ✓; T-23 on Day 1, inside the ≈ 08-18
  grace ✓.
- **§0.5 ↔ §7 ticket round-trip clean both ways.** Every matrix ticket exists on the board; the only
  board entries with no matrix row are T-01b (done), T-10 (merged), T-16 (rejected) — all explicitly
  retired.
- **Assets present**: `search_tree.py::canonical_search_tree_figure` ✓ · 16 `benchmarks/` symlinks ✓ ·
  `dataset_filter.py` `min_nodes: int = 2` at line 42 ✓ · `gedlib_api.py` ✓ · `graphical_abtract.pdf`
  present and misspelt ✓ · 56 bib `@`-entries ✓.
- **Repos reachable**: gSpan, IsalChem, graphkit-learn all HTTP 200 ✓.
- **E6 REAL**: `conclusion.tex:70` and `:81` — labels "present in all five benchmark datasets".
  False for LINUX.
- **E12 REAL**: `graphical_abtract.pdf` referenced under that spelling at `main.tex:131`.
- **Correctly retired, verified not stale**: `competitors.md`'s padded Hamming; `Ryurin/gedlibpy`;
  the controlled-edit cohort and TUDataset; "write BP ourselves" (`plan.md:281`, struck with a
  SUPERSEDED banner).

---

## Defects in full

### I-01 — `ged_bounds.py` does not exist; 13 of 16 named scripts are gone. **BLOCKING**, MEASURED

`find / -name 'ged_bounds.py'` returns nothing `[ORCH-VERIFIED — I re-ran it; no result]`.

**Absent**: `ged_bounds.py` (`data.md:666`, `:473`), `size_audit.py` (`:661`), `graphedx_audit.py`
(`:662`), `real_cost.py` (`:664`, `:270`, `:328`), `iam_audit.py` + `iam_audit.json` (`:665`, `:127`),
`feas2.py` + `feas2.json` (`:667`, `:314`, `:516`), `budget.py` (`:668`, `:530`), `grec_check.py`
(`:197`), `feasibility.py` (`:335`), `tail.py` / `timeout_check.py` (`:356`), `symmetry_diag.py`
(`:396`), `timeout_rate.py` (`:391`).
**Present**: `export_graphs.py`, `audit_recheck.py`, `audit_dropped.py`, `final_counts.py`,
`gedlib_api.py`.

`data.md:672` warns they "**will not** survive". They already have not.

**Downstream**: `plan.md:798–800` — **validation gate 2 of three**, which T-05 must pass before T-03
production ("GEDLIB's `BRANCH_FAST` and `BIPARTITE` must reproduce `scratchpad/ged_bounds.py` on the
same 300–400 pairs") — **is not executable**. Also `plan.md:769`, `:285–286`, `:804` ("Keep
`ged_bounds.py` in the repo permanently" — it never was), and `.claude/CLAUDE.md`'s "Cross-check, do
not skip". Further, `data.md` §5 / **H4** — ρ = 0.966 vs 0.840 and the +78 % / −11 % biases, on which
the "BRANCH-FAST is the primary large-n reference" decision rests (`plan.md:673–675`, §11 item 4) —
is now **unreproducible from any surviving artifact**, as is every timing in §§3, 4, 4.1, 4.3, 4.4,
6. `data.md:672–676`'s own porting instruction was not executed; T-01's scope (`plan.md:541`) covers
only `audit_recheck.py`.

### I-02 — §0's AIDS "raw" column is the connected count. Major, MEASURED/INFERRED

`data.md:41` raw **819** / kept 769 / **93.9 %**; `data.md:57` raw **819** / kept 819 / **100 %**.
Versus `data.md:167` (§2.1) N raw **911**, N conn 819, 89.9 %; and `plan.md:167` (F1) raw **911**,
connected 819, survives 769. `[ORCH-VERIFIED — all three lines read directly]`

§2.1 and F1 are correct. Established by: every *other* §0 raw value re-measures as the true raw set
(Letter 2,250 ✓, GREC 1,100 ✓, AIDS-IAM 2,000 ✓, COIL-DEL 7,200 ✓, Mutagenicity 4,337 ✓, Protein
600 ✓); `graphs.json` holds **819** AIDS graphs, i.e. the export is already connectivity-filtered;
two documents independently state 911.

**This is MF1's defect class *inside* §0** — the table `plan.md:15` and `data.md:10` name as the sole
source from which a printed number may be taken.

**Downstream**: Suite 2 prints AIDS retention as 100 % (true 89.9 %) — 92 discarded graphs vanish;
**H6** (`data.md:114`) and `data.md:183` would list AIDS among the 100 % datasets; the R3.5a
pair-accounting ladder (`plan.md:855`) collapses its first rung for AIDS; Suite 1's 93.9 % is
conditional (769/819), not 84.4 % (769/911).

### I-03 — §3.1 mixes two populations in one table. Major, MEASURED

`data.md:294–298` gives Letter 696,790 / 789,396 / 2,135,211 = C(1181,2), C(1257,2), C(2067,2) — the
**n≥1** sets — while its LINUX (3,916) and AIDS (295,296) rows are the **n≥2 kept** sets.
`plan.md:588–592` gives the same table with 695,610 / 784,378 / 2,118,711 — verified correct.
Totals: **3,920,609** vs **3,897,911**, a **22,698-pair gap** against the very number `data.md:44–47`
cites as proof of exact reproduction. Downstream: §3.1's core-hours (0.8 / 0.9 / 4.7) derive from the
inflated counts; `plan.md` §12.2 risk R1's fallback is costed against §3.1.

### I-04 — §2.1 Fingerprint `N conn` is a different population. Major, MEASURED

`data.md:174` prints N conn **2,057**, ret **51.4**. Measured: `CONN_ge1` = **2,995** (74.9 %) — the
column's stated definition at `data.md:126`; `KEPT_ge2` = **2,056** (51.4 %). Every other row of that
column is `CONN_ge1`. **Fingerprint alone carries the n≥2 count and is off by one against it.** The
dataset has 939 single-node graphs, which is why the definitions diverge here and nowhere else.

Downstream: `data.md:538` (§6) "connected 2,057 / 2,114,596 pairs" — should be 2,995 / 4,483,515 or
2,056 / 2,112,540; the 67.3 M total inherits it. `data.md:114` (**H6**) and `:183`'s "51.4 % – 100 %"
range mixes two definitions — under `CONN_ge1` the corpus minimum is **Letter LOW at 52.5 %**.

### I-05 — the retracted value survives as a ratio. Major, MEASURED

`data.md:230`: "AIDS-IAM discards at 2.3×…; Protein at 1.6×; **Fingerprint at 2.3×**." Measured:
AIDS-IAM 31.76/14.02 = **2.27×** ✓; Protein 50.19/31.68 = **1.58×** ✓; **Fingerprint 5.98/5.03 =
1.19×**, not 2.3×. The 2.3× is 11.56/5.03 = 2.30 — from the discarded n̄ that `data.md:214–218`
**strikes eight lines earlier**.

Downstream: `data.md:225–232`'s thesis — "the datasets with *high* retention discard the *largest*
graphs" — cites Fingerprint as evidence. **Fingerprint has the corpus's lowest retention and the
smallest size bias: it is a counter-example to the sentence citing it.** The claim survives on
Mutagenicity (1.92×), AIDS-IAM (2.27×), Protein (1.58×). The banner's "no number from that row may be
quoted" is violated on the next page.

### I-06 — "76.5 days" is the pre-audit board. Major, MEASURED

`plan.md:961`. Summed from §7's own Days column: T-01…T-15 (excluding merged T-10) = **72.5**;
**+ T-16 (3–4 d) = 76.5** ← the quoted figure; + T-17, T-04a, T-18, T-19, T-20, T-21, T-22, T-23 =
+18.5; **current board = 91.0** (lower bound 52.8).

76.5 reproduces exactly as the **v0.5** board — it predates both v0.6 (which added T-17…T-24) and
v0.7 (which rejected T-16), yet is attributed to `gap-audit.md` MF11, the audit that created those
tickets. §12 is the plan's entire feasibility argument; **the workload is understated by 14.5 days
(19 %)**. `plan.md:579` and `:995` read as relief against 76.5 when the board grew 14.5 days net in
the same revision.

### I-07 — the calendar allocates below the board's own minima; the critical path does not fit. Major, MEASURED

**T-05** (`plan.md:546`, 5–10 d) gets Days 5–8 = **4 days**. **T-06** (`:547`, 10–14 d) gets Days
8–12 = **5 days**, half its minimum. T-03 (3–8 d) gets 3. All other 18 tickets are at or above
minimum (checked programmatically).

Separately, the critical path declared at `plan.md:574` (T-23 → T-01 → T-03 → T-05 → T-06 → T-20 →
T-15 → T-24) is serial and sums to **27.5 days at lower bounds**, 44.5 at upper, **against a 20-day
window**. §12's mitigation — "survivable only because most tickets parallelise" — does not apply to a
critical path. Risk R1 budgets for T-03 slipping; **nothing budgets for T-06 getting 5 of 10–14
days.** *Assumption*: windows read as inclusive day ranges.

### I-09 — `statistics.md` still prints `max n = 417`. Major, MEASURED

`statistics.md:116`: "(Mutagenicity median n = 27, **max n = 417**)." `[ORCH-VERIFIED verbatim]`
Measured `KEPT_ge2` n max = **98**; 417 is in `DISC_ge2`. `plan.md:55–56` asserts superseded text
"**has been struck through**"; `data.md:155` lists three corrected sites and this is not among them.
`statistics.md` v2.1 is the **locked** protocol and the 417 sits in its heavy-tail/stratification
justification, which drives the size strata **T-02 must freeze before T-06 runs**.

### I-10 — `data.md` §4.4 still recommends building the rejected T-16. Major, MEASURED

`data.md:438`: "**Recommendation** — one new C++ variant, `wl_pruned_canonical` (ticket **T-16**):"
with a five-point justification at `:440–450`. `[ORCH-VERIFIED verbatim]` Not struck, not
banner-marked. Contradicted by `plan.md:48` (decision 17), `:558`, `:579`, `:807–844` (§7.2), `:952`
(S-b), `:995`, `gap-audit.md:39` (MF8). `plan.md` is correct — decision 17 is dated and
author-signed. **`data.md` is listed at `plan.md:13` as required reading before executing any
ticket, and T-13 must read exactly this section** (`plan.md:818–836` relocates the WL measurement
into it).

### I-11 — "131,148" unsourced and contradicted by F2. Major, INFERRED

`plan.md:242` "AIDS 295,296 (from **131,148**, 2.25×)" and `:869` "on 295,296 pairs instead of
**131,148**", versus `plan.md:178` (F2, measured) AIDS `n_valid_ged_pairs` = **181,909**
(148,785 + 16,471 + 16,653), "Exact match on both". 295,296/131,148 = 2.2517, so "2.25×" is
arithmetically consistent with 131,148 — not a typo. But **131,148 occurs nowhere else**: not in the
other twelve source documents, not in `verified-discrepancies.md` E2, not in any `.tex`. On F2's
number the gain is **1.62×**.

Undetermined which is right without the submitted AIDS result file; F2 is the only *measured* figure
and 131,148 carries no provenance. Marked INFERRED because `graphedx_audit.py`, which produced F2, is
among the scripts lost at I-01. Downstream: §8's "The AIDS question, settled with data" states the
power gain for the density stratification that "**can refute** `conclusion.tex:30–36`" — one of two
week-1 experiments scheduled early precisely to buy absorption time.

### I-12 — H3's "10 datasets, 67 M pairs". Minor, MEASURED

`data.md:111`. The 67.3 M comes from §6 (`:534–544`), whose ten rows **include** COIL-RAG (25.4 M)
and Fingerprint (2.1 M) — both dropped at §2.3 — and **omit** LINUX and AIDS-GraphEdX, both in the
cohort. The cohort's ten total **40,024,242** pairs. Two disjoint tens, one label. Repeated at
`data.md:556` and `plan.md:1020`.

### I-13 — "1.3 core-hours for 40 M pairs" is a 67 M figure. Minor, MEASURED

`data.md:77` and `plan.md:338–339`. Re-derived from §6's own rows restricted to the Suite-2 cohort:
0.01+0.01+0.02+0.00+0.02+0.72+0.25+0.01 = 1.04, plus LINUX and AIDS-GraphEdX (~338 k pairs) ≈ **1.05
core-hours**. The 1.3 is §6's 67.3 M total including dropped COIL-RAG (0.20) and Fingerprint (0.02).
Same class as MF1. Conservative direction (24 % high), but "no subsampling is needed" rests on it.

### I-14 — encoding budget stated two ways. Minor, MEASURED

`data.md:331` "**≈ 10 s total (0.008 core-hours)**": 10 s = 0.00278 core-h; 0.008 core-h = 28.8 s.
`data.md:555` gives 0.01 core-h for the same item; `:78` says "under a minute"; §6's measured encode
column sums to 10.0 s.

### I-15 — H8: 5 seconds vs 5 minutes. Minor, MEASURED

`data.md:116` "does not finish in **5 s**" against `data.md:101` "five minutes", `:367` "**> 4 min,
did not finish**", `:373` "five minutes". §4.3 is the measurement; **H8 understates by ~60×**. H8 is
the finding `plan.md:352–353` (§3.5 item 4) and `:832–836` (T-13, R3.7d) both cite.

### I-16 — n = 78 vs n = 81. Minor, MEASURED

`data.md:373` "between **n = 78** (9.9 ms) and n = 73 (963 ms)". `data.md:406` (§4.4):
`AIDS/41883 | **81** | 84 | … | 9.9 ms`. No n = 78 row exists in §4.3 or §4.4.

### I-17 — MF count. Minor, MEASURED

`plan.md:27` "10 unowned demands, **16** flawed or infeasible locked decisions". `gap-audit.md`
defines **MF1–MF17**. MF17 is the Fingerprint finding `plan.md:1023` itself describes in the v0.7
changelog — the count was left at 16 in the revision that created the 17th.

### I-18 — two §0.5 rows have no ticket. Minor, MEASURED

`plan.md:69`: "A row with no ticket is a hole; **there are none**." **M2** (`:79`, Ticket = "§12") and
**R3.2** (`:117`, Ticket = "§6") carry section pointers, not tickets. R3.2 is a decline so its owner
is genuinely a decision, but the contract as worded does not admit that; M2 (the deadline) has no
owning ticket at all.

### I-19 — E6 has three owners. Minor, MEASURED

`plan.md:144` (§0.5) → **T-12**; `plan.md:566` and `:47` put "**E6 fix**" inside T-18 Tier 0;
`plan.md:889` lists it under §9 manuscript errors, whose owner is T-11. §0.5 is the coverage contract
and names only T-12. **Risk: T-12 and T-18 each assume the other edits the same two sentences.**

### I-20 — E12's orphan count. Minor, MEASURED

`plan.md:894` "**two** orphaned figure PDFs". Measured by differencing `\includegraphics` against
PDFs on disk: **four** — `fig_algorithm_overview_full.pdf`, `fig_empirical_complexity.pdf`,
`fig_message_length_ratio.pdf`, `fig_message_length_scatter.pdf`. Additionally three assets are
referenced **only from commented-out blocks**: `graphical_abtract.pdf` (`main.tex:130–132`),
`EzequielLopez.pdf` (`:229`), `MarioPascual.jpg` (`:240`). E11 flags the commented AI declaration but
**not the commented author biographies**, which T-24 (`plan.md:572`) lists as a compliance item
without recording that they are disabled.

### I-21 — F1 mixes populations. Minor, MEASURED

`plan.md:161–167`. (1) `max n` and `median n` are raw values beside a connected column: Letter LOW
prints max **8**, median **5**; retained are **7** and **4**. (2) The "survives n≤12" column changes
population between rows — Letter shows "2,250 (100 %)" (raw) while AIDS shows 769 (from the connected
819). The section's conclusion is correct either way, and `:170`'s 47.6 % is right.

### I-22 — "pre-reconciliation" note inverted. Minor, MEASURED

`plan.md:595`: "Counts are **pre-reconciliation** (open question 3)." The counts are verified to be
exactly the **post-reconciliation** kept sets matching §0. The reconciliation item is `data.md` Q4 /
`plan.md` §11 item 3, both struck as closed. `data.md` §9's live **Q3** is the unrelated subsampling
question. The note now reads as a caveat on the plan's most-verified table.

### I-23 — `D<n>` namespace collision. Minor, MEASURED

`statistics.md` defines **D1–D15**; `verified-discrepancies.md` defines **D1–D20**. `plan.md` cites
both unqualified: discrepancy namespace at `:881` (D1), `:882` (D2), `:880` (D5), `:883` (D20),
`:430` (D16), `:151`/`:450` (D19); statistics namespace at `:51` (D14), `:126` (D2, D3, D15), `:601`
(D11), `:729` (D6). **D1, D2, D5, D6, D11, D14, D15 exist in both.** Resolvable from context, not
mechanically — and **T-14's response letter cites both families**.

### I-24 — §4 out of order. Minor, MEASURED

File order: §4 (`data.md:311`) → §4.1 (`:333`) → **§4.3** (`:354`) → **§4.4** (`:394`) → **§4.2**
(`:458`). §4.2 carries the MF1 correction and sits after the two sections depending on it.

### I-25 — MF12, MF13, MF17 cited nowhere outside gap-audit. Minor, MEASURED

Cited across the corpus: MF1–MF7, MF9, MF10, MF11, MF14, MF15, MF16. Never cited: MF8 (resolved in
place as decision 17 — acceptable), **MF12, MF13, MF17**. MF17 is the Fingerprint irreproducibility
finding, and `data.md` §2.2.1's banner (`:210–218`) that implements it carries **no MF pointer**,
while the neighbouring MF1 corrections all do. Per the brief, `gap-audit.md` is an object under
audit: **three of its findings have no verifiable downstream owner.**

---

## Pattern

**Eight of the 24 standing defects are one defect** — *a statistic computed over one population,
printed under another's header*: I-02 (§0 AIDS raw), I-03 (§3.1 mixed pair counts), I-04
(Fingerprint `N conn`), I-05 (discard ratio on a retracted mean), I-12 (two disjoint tens), I-13
(67 M core-hours labelled 40 M), I-21 (F1's raw/connected mix), and H6's retention range built from
two definitions.

MF1 identified the mechanism and corrected three call sites; **the mechanism recurs across five
documents.** The generalisation that finds them all is the four-population table: print every
candidate statistic under `RAW` / `CONN_ge1` / `KEPT_ge2` / `DISC_ge2` and match each against its
header, rather than checking only the suspected sites.

**Second pattern — corrections that update a number but not what was derived from it**: I-05 (ratio),
I-06 (day sum), I-08b (slot budget), I-09 (stale copy), I-10 (stale recommendation), I-17 (MF count),
I-22 (stale caveat). In every case the primary value was fixed and a ratio, sum, count or
recommendation computed from it was not.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-11 | Created by `audit-integrity` for audit-2026-08-11b; persisted by the orchestrator after a harness hook blocked the agent's write. **I-08 rejected** on orchestrator re-measurement (43 cited / 13 dead / 12 slots stand); **I-08b added** in its place. `[ORCH-VERIFIED]` stamps added for I-01, I-02, I-09, I-10. |
