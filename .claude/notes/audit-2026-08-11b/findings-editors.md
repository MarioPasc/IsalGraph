# Findings — Editor-in-Chief, Area Editor, submission mechanics · audit `audit-2026-08-11b`

**Agent**: `audit-editors` · **Demands owned**: M1, M2, M3, EiC.a1–a4, EiC.b, EiC.c, AE.1–AE.4c, AE.5
**Date**: 2026-08-11
**Sources readable**: manuscript **yes** (`.tex`, `.bib`, `main.pdf` all opened) · letter **yes** · plan **yes**

---

## Compliance re-derivation

Every number `README.md` / `inventory.md` assert about my rows, re-derived from the sources with my
own commands. **Manuscript root** `$D` =
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`.

| # | Asserted | Source of assertion | Command I ran | Measured | Verdict |
|---|---|---|---|---|---|
| 1 | `main.pdf` is **exactly 35** pages | `README.md:200`, `inventory.md:11` | `pdfinfo $D/main.pdf` | `Pages: 35`; `Page size: 612 x 792 pts (letter)` | **agree** |
| 2 | **43** printed bibliography items | `README.md:196`, `inventory.md:12` | Python: strip first unescaped `%` per line, then `\\(?:cite|citep|citet|citealp|citeauthor|citeyear)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}` over the six `.tex` files, union of keys | **43** distinct keys | **agree** |
| 3 | **56** entries in `cas-refs.bib` | `README.md:205` | `grep -c '^@' $D/cas-refs.bib`, cross-checked with `@(\w+)\s*\{\s*([^,\s]+)\s*,` | **56** unique keys (29 `@article`, 26 `@inproceedings`, 1 `@book`) | **agree** |
| 4 | **13 dead** entries (E9) | `plan.md` §0.5 E9, `manuscript.md:222` | set difference `defined − uncommented-cited` | **13**, of which **2** (`Fischer2015hausdorff`, `Lerouge2017ilp`) are cited *only* from commented LaTeX at `methodology.tex:805–806`; 11 are never cited at all | **agree**, 2/11 split newly recorded |
| 5 | **exactly 1** genuine arXiv-only citation | `README.md:197`, `plan.md:463` | classify each cited entry: `arxiv` in `journal∪booktitle∪note∪eprint∪url∪doi` → 6; of those, `journal`/`booktitle` itself arXiv/CoRR → 1 | **6** display an arXiv id (`kipf2017gcn`, `velickovic2018gat`, `xu2019powerful`, `fey2019pyg`, `jain2024graphedx`, `lopezrubio2025isalgraph`); **1** genuinely arXiv-only (`lopezrubio2025isalgraph`) | **agree** |
| 6 | **exactly 1** uncommented multi-key `\cite` group at `methodology.tex:803` | `README.md:198`, `plan.md:478` | same scan, keeping matches whose brace body splits into ≥ 2 keys | **1**: `methodology.tex:803` — `\cite{garey1979,Zeng:2009}`, **2 keys** | **agree** |
| 7 | `introduction.tex:31` is a "four-way group … already individually commented" | `plan.md:478–479` | read `introduction.tex:26–40` verbatim | **It is not a group at all** — line 31 carries **four separate `\cite{}` commands**, each with its own descriptive phrase. The detector correctly did not flag it | **decision agree, label disagree** — see EiC.a4 |
| 8 | **12 slots** of headroom | `README.md:196`, `plan.md:467` | 55 − 43 | **12** | arithmetic **agree**; sufficiency **disagree** — see EiC.a1 |
| 9 | format "double spaced SINGLE column" | `mail.txt:130` | `grep -n documentclass $D/main.tex` → `\documentclass[review,times,number]{elsarticle}` | `review` = double-spaced single column; US Letter | **compliant** |
| 10 | *(not previously asserted)* refs from **2025–2026** | — | year histogram over the 43 cited | **2** in 2025, **0** in 2026. **Both 2025 entries are group self-citations**: `lopezrubio2025isalgraph` (López-Rubio) and `ThurnhoferHemsi:2025` (Thurnhofer-Hemsi, García-Aguilar, Fernández-Rodriguez, **López-Rubio**). **Zero third-party refs after 2024** | new measurement |
| 11 | *(not previously asserted)* `[28]` / `[29]` identity | — | replay citation order over `main.tex` preamble → the five inputs | `[28]` = **`lopezrubio2025isalgraph`** (arXiv-only); `[29]` = **`ThurnhoferHemsi:2025`**, *J. Chem. Inf. Model.* **65(15):7936–7955, 2025** | new measurement — see Notes |
| 12 | *(not previously asserted)* PR-venue coverage | — | venue classification of the 43 | *Pattern Recognition* journal ×6 (2021, 2021, 2022, 2023, 2023, 2023); *Pattern Recognition Letters* ×1 (**1983**); SSPR ×1 (2008); IEEE TSMC ×1 (1983). **Zero** CVPR/ICCV/ECCV/ICPR/TPAMI/IJCV. **No PR-field reference after 2023** | new measurement — see EiC.b |

**Every number the README asserts about my rows is correct.** Rows 10–12 are new measurements the
README does not contain, and they raise the severity of EiC.a2 and EiC.b.

---

## Verdict table

| ID | Modal | Verdict | One line |
|---|---|---|---|
| M1 | REQUIREMENT | **COVERED** | T-24 owns the LaTeX package; checklist item 1 |
| M2 | REQUIREMENT | **COVERED** | §12.1 calendar, 20 dated days, gates named |
| M3 | *(no imperative)* | **MISMATCHED** | Right deliverable, wrong authority; `:67` unrecorded |
| EiC.a1 | REQUIREMENT | **UNDER-blocking** | Allocations sum to 16–17 against 12 slots |
| EiC.a2 | REQUIREMENT | **COVERED** | T-19 ≥ 6 criterion right; self-citation loophole open |
| EiC.a3 | REQUIREMENT | **COVERED** | 5 of 6 already compliant; note-strip cheap and correct |
| EiC.a4 | REQUIREMENT | **COVERED** | One 2-key group; the `intro:31` exemption is right |
| EiC.b | REQUIREMENT | **UNDER-blocking** | No testable criterion; zero PR-field refs after 2023 |
| EiC.c | REQUIREMENT | **UNDER-blocking** | Budget exists; its arithmetic is not derivable |
| AE.1 | REQUIREMENT | **COVERED** | Owned; the size story has no page line of its own |
| AE.2 | REQUIREMENT | **COVERED** | T-08 new §1.x |
| AE.3 | REQUIREMENT | **COVERED** | T-17 P0 paper artifact; axes deliver properties only |
| AE.4a | REQUIREMENT | **COVERED** | Competitors + decision 18 selection rule |
| AE.4b | REQUIREMENT | **COVERED** | T-18 Tier 0 label column; also fixes E6 |
| AE.4c | REQUIREMENT | **COVERED** | `statistics.md` D1–D15 via T-02/T-06 |
| AE.5 | REQUIREMENT | **UNDER-major** | No row anywhere; R3's `:83` preamble never decomposed |

**No OVER finding in this slice.** Nothing the editors asked for is over-served; the failures are all
under-coverage or mis-derived compliance arithmetic.

---

## M1 — Upload source files, not PDF

**Operative clause** (`mail.txt:22`), verbatim:

> "When submitting your revised manuscript, please ensure that you upload the source files (e.g. Word, Latex)."

**Modal**: "please ensure that you upload" → **REQUIREMENT**.

**Full comment spans** `mail.txt:22`. **What the rest of it is doing**: the following two sentences
state the consequence ("will create delays", "we will contact you to request them") — pure
enforcement language, no additional ask.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| LaTeX sources exist | already satisfied | `$D` contains `main.tex`, `introduction.tex`, `methodology.tex`, `computational_experiments.tex`, `results.tex`, `conclusion.tex`, `cas-refs.bib`; `main.tex:6` = `\documentclass[review,times,number]{elsarticle}` |
| assembled as an upload package | absent | `manuscript.md:211` — "sources present, package never assembled" |

**Residual gap**: assembling and verifying the upload set (sources + figure PDFs + `.bib` + class
files), not producing anything new.

### Q4 — Explicit `or`? No. "(e.g. Word, Latex)" enumerates formats, not branches.

### Q5 — Argument or measurement? Neither applies; this is a mechanical deliverable.

### What the plan currently commits to

**T-24** (`plan.md:572`): "Submission package and Elsevier compliance — LaTeX source files; AI
declaration; author biographies and photos; acknowledgements; highlights; graphical abstract (**fix
the `graphical_abtract.pdf` filename**); competing-interest and data-availability statements.
Checklist in `manuscript.md` §5". 1 day, P0, depends on T-15. `manuscript.md:209–222` is a 12-row
checklist with a per-item "state today" column. Searched: `T-24`, `source file`, `M1`, `package`,
`Elsevier`, `upload`.

**Verified against the letter**: the checklist's Elsevier items that the *letter* names are items 1
(`mail.txt:22`) and 2 (`mail.txt:130`). Items 4–11 (AI declaration, biographies, acknowledgements,
highlights, graphical abstract, competing interest, data availability) come from **Pattern
Recognition author guidelines, not from this letter** — they are correct additions, but they are
unasked-for by `mail.txt` and should be labelled as such per rubric Q6.

### Q6 — Second customer? EiC.c shares item 2 (page count). Items 4–11 have no letter-level customer.

### Verdict: **COVERED**

**Why**: an owner exists (T-24), it is P0, its checklist covers the letter's clause explicitly at
`manuscript.md:211`, and the work is proportionate — assembly, not creation.

**Effort**: current plan **1 day** (`plan.md:572`). Proportionate response **0.5 day**, derived as:
collect 7 source files + `.bib` + `elsarticle` class + 4 figure PDFs (~1 h), recompile from a clean
directory to prove the package is self-contained (~1 h), verify the 12 checklist rows (~1 h), upload
(~0.5 h). The extra half-day in T-24's estimate is the AI declaration, biographies, highlights and
graphical-abstract regeneration, which are guideline items rather than letter items.

**Assumptions made**: that `elsarticle.cls` and the `cas-*` support files travel with the upload.
`manuscript.md:211` says "sources present" without enumerating them; I verified the six `.tex`
files and the `.bib`, not the class files.

---

## M2 — Deadline 2026-08-31

**Operative clause** (`mail.txt:20`), verbatim:

> "Your revised manuscript is due by Aug 31, 2026 Kindly advise if you decide not to resubmit your paper."

**Modal**: "is due by" + "Kindly advise" → **REQUIREMENT** (the second clause is conditional on a
decision not to resubmit, which is not the case).

### Q3 — Already satisfied? Not applicable — this is a schedule constraint, not manuscript content.

### What the plan currently commits to

**`plan.md` §12.1** (`:964–978`): a nine-row calendar from "**Day 1 — 08-12**" to "**Day 20 —
08-31**", each row naming the gate that must close and why it is a gate. §12.2 is a six-row risk
register; §12.3 is a pre-decided cut order. §0.5 row M2 (`plan.md:14`) points at §12.

**The honest number is in the plan itself**, `plan.md:960–961`: "**Summed at the upper bound the
board is 76.5 days of work in a 20-day window.**" That is disclosed, not hidden, and the mitigation
(parallelism + unattended compute) is stated.

### Q6 — Second customer? Everything. The calendar is the schedule for all 41 demands.

### Verdict: **COVERED**

**Why**: the deadline has a dated calendar with gates, a risk register that names the long pole
(T-03, `plan.md:984`), and a pre-decided cut order (`plan.md:991–1009`). The 76.5-vs-20 ratio is
stated openly rather than concealed by optimistic estimates.

**Effort**: zero incremental — the artifact exists.

**One correction for the orchestrator**: `00-editor-and-decision.md:7` records the deadline as
"**2026-08-31** (`mail.txt:20`) — **21 days**"; `inventory.md:8` says **20 days from 2026-08-11**.
2026-08-12 through 2026-08-31 inclusive is **20** days, and §12.1 runs Day 1 = 08-12 to Day 20 =
08-31, which is internally consistent. `00-editor-and-decision.md`'s "21 days" counts 08-11 itself.
Immaterial to the schedule but the two files should agree.

---

## M3 — Point-by-point response

**Operative clause**: **there is none.** `mail.txt:67`, the line §0.5 cites, reads:

> "Please address these concerns thoroughly, as they will strongly influence the potential impact of the work and citation of the paper if it is accepted for publication after the revision."

**Modal**: "Please address these concerns thoroughly" → **REQUIREMENT**, but its object is "these
concerns", whose antecedent is `:59–66` (AE.1–AE.4c). It is a **weighting statement over the Area
Editor's own agenda**. It does not request a response document of any kind.

**What the rest of the letter says about response documents**: nothing. I searched `mail.txt` for
"response", "point-by-point", "summary of changes", "rebuttal", "cover letter" — the only
response-adjacent text is `:22` (source files) and `:24` (query mailbox).

### Q3 — Already satisfied?

Not a manuscript property. The relevant prior finding is at
`00-editor-and-decision.md:34–35`: the letter requires "**no clean unhighlighted main file** and
**no separate 'Summary of Changes' designation**. Neither appears anywhere in `mail.txt`. It also
does **not** explicitly demand a point-by-point response document". That file already reached the
same conclusion I reach; it is corroboration from a prior audit, which I re-derived independently.

### Q4 — Explicit `or`? No.

### Q5 — Argument or measurement? Not applicable.

### What the plan currently commits to

**T-14** ("Response letter", `plan.md:555`, 3 days, P0, depends on all) plus `manuscript.md` §4
(`:134–201`): a six-part architecture (part 0 summary, part 1 Area Editor, part 2 R1, part 3 R3,
part 4 EiC checklist, part 5 self-found E1–E12), a fixed per-comment format ("verbatim quotation →
response → *exact* pointer to the changed location", `manuscript.md:147`), and the fix that
fragments accrue per ticket rather than being written in the last three days
(`manuscript.md:151–156`). §4.4 locks "no marked-up manuscript".

### Q6 — Second customer? Every numbered demand routes through the letter.

### Verdict: **MISMATCHED**

**Why**: the *deliverable* is correct, owned, well-specified, and justified — Elsevier Editorial
Manager provides a "Response to Reviewers" field at revision and a point-by-point response is
universal practice at Pattern Recognition. But §0.5's row cites `:67` as its authority, and `:67`
says something else entirely. Two consequences, both real: (a) the row's justification is
unfalsifiable as written, so a future reader cannot check it; (b) **`:67`'s actual content — a
priority weighting over AE.1–AE.4c — has no home in `plan.md` at all.** The only place the
weighting is operationalised is T-17's priority note (`plan.md:564`, "the AE endorsed this one in
their own voice") and `manuscript.md:108`; AE.1, AE.2, AE.4a/b/c inherit no weighting from it.

**Effort**: current plan **3 days** for T-14 assembly, plus per-ticket fragments. Proportionate
response is the same 3 days — the deliverable is not in question. The **fix costs ~15 minutes**:
re-cite M3 to journal convention / Editorial Manager rather than `:67`, and add `:67` as a header
note over the Area Editor block of §0.5 so the weighting is recorded where the AE rows are read.

**If MISMATCHED — what must change**: §0.5 row M3's authority column. Without the fix, the plan
carries one demand attributed to a line that does not support it, and one priority statement from
the Area Editor that no ticket weighting reflects.

**Assumptions made** (I could not ask): that Pattern Recognition's Editorial Manager instance
exposes a Response to Reviewers field at revision. I did not verify this against Elsevier's
submission system; it is journal convention, not a letter requirement.

---

## EiC.a1 — Bibliography between 35 and 55 items

**Operative clause** (`mail.txt:126`), verbatim:

> "Your bibliography in the final version after the revision still should be between 35-55 items."

**Modal**: "should be between" → **REQUIREMENT**, and by `mail.txt:124` ("**I will check that these
are adhered to before your paper is approved for publication**") a **pass/fail compliance item**
enforced independently of the reviewers. Rubric §4 guard 3 applies.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| ≥ 35 items | already satisfied | 43 printed (re-derivation row 2) |
| ≤ 55 items | already satisfied **today** | 43 printed; 12 slots free |

**Residual gap**: none in the submitted version. The risk is entirely in the **revised** version,
because every additive demand in the plan draws on the same 12 slots.

### Q4 — Explicit `or`? No.

### Q5 — Argument or measurement? Arithmetic. See below.

### What the plan currently commits to

**T-08** ("Related work section + bibliography to ≤55 (§5.4)", `plan.md:549`, 4–10 days, P1) with
the budget at `plan.md:465–479`. Searched: `EiC.a1`, `35-55`, `bibliography`, `12 slots`, `T-08`,
`T-19`, `cite`, `reference` across `plan.md`, `manuscript.md`, `competitors.md`, `statistics.md`,
`data.md`, `labels.md`.

`plan.md:467–479` allocates:

| Purpose | Slots claimed |
|---|---|
| AGM, gSpan, nauty/Traces, bliss, graph6, Babai | 5–6 |
| GED approximation: Riesen–Bunke 2009, Fischer 2015, Blumenthal 2020 | 3 |
| IAM Graph Database (Riesen & Bunke, SSPR 2008) | 1 |
| GEDLIB (Blumenthal et al., GbRPR 2019) | 1 |
| Recent (2025–26) pattern-recognition work | 2–3 |
| **Plan's own total** | **12–14** — "Over budget by ~2" (`plan.md:477`) |

**Two defects in that arithmetic, both verified.**

**Defect 1 — the recency allocation contradicts T-19's own acceptance criterion.** §5.4 allocates
**2–3** slots to recent pattern-recognition work. **T-19** (`plan.md:567`) requires "add **≥ 6 from
2025–2026** in pattern recognition, graph matching or graph representation." Substituting 6 for 2–3:

> 5–6 (canonicalisation) + 3 (GED approx) + 1 (IAM) + 1 (GEDLIB) + **6** (recency) = **16–17**

against **12** available. The plan is over budget by **4–5 slots**, not by ~2. **No single ticket
owner sees this**: T-08 owns §5.4's table, T-19 owns the ≥ 6 criterion, and nothing reconciles them.

**Defect 2 — the stated relief mechanism frees nothing.** `plan.md:477` offers: "drop the weakest
additions **or retire a dead citation**." Retiring a dead citation frees **zero** slots, because
`elsarticle-num` prints only what is cited and the 13 dead entries (re-derivation row 4) were never
printed. The printed count — 43 — is what the EiC counts. Confirmed: 56 defined − 43 cited = 13, and
the bibliography renders 43.

**A third item, smaller**: 2 of the 13 "dead" entries (`Fischer2015hausdorff`, `Lerouge2017ilp`) are
cited from commented LaTeX at `methodology.tex:805–806`. §5.4 wants Fischer 2015 in the
GED-approximation slot. Uncommenting `:805–806` restores **both** at zero `.bib` cost but consumes
**2 printed slots** and adds ~0.15 pages — so the GED-approximation line is 3 new slots whether the
citations are new or restored.

**The only relief that actually works** is **removing existing citations from the text**, which is
contemplated nowhere in the plan. Each removal frees exactly one printed slot and costs one edited
sentence. Candidates visible from the citation-order scan: `[40]` `barabasi1999emergence` and `[41]`
`erdos1959random` (synthetic-graph provenance, likely superseded by the IAM-only cohort under
decision 12), and one of the two Blumenthal citations if GEDLIB and the 2020 bound paper can share a
sentence.

### Q6 — Second customer? EiC.a2, EiC.b, AE.2, R1.2, R3.1, R3.6a all add references against these
same 12 slots. This is the most heavily shared resource in the revision.

### Verdict: **UNDER-blocking**

**Why**: EiC.a1 is a pass/fail item the Editor-in-Chief checks independently (`mail.txt:124`), the
plan's own allocations exceed the ceiling by 4–5 once T-19's criterion is honoured, and the one
relief mechanism the plan names — retiring dead entries — provably frees nothing because dead
entries were never printed. An owner exists for the *ceiling* (T-08) and an owner exists for the
*additions* (T-19), but no owner reconciles them, so the constraint can be breached with every
ticket individually complete.

**Effort**: current plan — the reconciliation is not scheduled anywhere, so **0 days allocated**.
Proportionate response **≈ 0.25 day**, derived as: re-run the 43-key count after T-08 and T-19 land
(5 min, the scan above is scripted); if over 55, remove *k* existing citations by editing one
sentence each (~10 min per citation, so ≤ 1 h for the plausible k = 4–5); recompile and re-count
(~15 min). The cost is not the labour — it is that the check must be **gated after T-08 and T-19
and before T-15**, and `plan.md` §12.1 has no such gate.

**If UNDER — what must exist**: (a) a single reconciliation owner with a hard stop at 55, placed in
§12.1 between "Days 8–12 · T-08 → T-19" and "Days 17–19 · T-15"; (b) correction of `plan.md:477`'s
relief sentence, which is factually wrong; (c) a pre-declared removal list, so the cut is not made
on day 19. Without these, EiC.a1 — a pass/fail item — is breached by roughly 4–5 items.

**Assumptions made**: that T-19's "≥ 6" and §5.4's "2–3" refer to the same set of new references. If
some of the ≥ 6 are meant to be *retitled* existing entries the conflict shrinks, but §5.4:475's
note — "**weakest current position: nothing third-party after 2024**" — and my re-derivation row 10
(zero third-party references after 2024) both establish that the 6 must be genuinely new.

---

## EiC.a2 — Cover last and current year

**Operative clause** (`mail.txt:126`), verbatim:

> "Take a careful look at your bibliography and they cover the state of the art. Missing references from last and current year most probably would mean you are missing the state of the art and the revision process can be delayed being asked to update it."

**Modal**: "Take a careful look" (imperative) + the consequence "the revision process **can be
delayed**" → **REQUIREMENT**, compliance-checked per `mail.txt:124`.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| references from **last year** (2025) | nominally 2, **substantively 0** | `cas-refs.bib`: `lopezrubio2025isalgraph` (López-Rubio, arXiv) and `ThurnhoferHemsi:2025` (Thurnhofer-Hemsi, García-Aguilar, Fernández-Rodriguez, **López-Rubio**, *J. Chem. Inf. Model.* 65(15):7936–7955). **Both are group self-citations** |
| references from **current year** (2026) | absent | year histogram: `{…, '2023': 3, '2024': 3, '2025': 2}` — no 2026 entry exists |
| bibliography "covers the state of the art" | most recent third-party references are **2024** | `jain2024graphedx`, `ju2024comprehensive`, `khoshraftar2024survey` |

**Residual gap**: **zero third-party references from 2025 or 2026.** This is precisely the state the
EiC's sentence describes as evidence of missing the state of the art.

### Q4 — Explicit `or`? No.

### Q5 — Argument or measurement? Neither; the fix is a literature search.

### What the plan currently commits to

**T-19** (`plan.md:567`): "**Bibliography recency and venue audit** — classify all 43 existing
references by venue and year; add **≥ 6 from 2025–2026** in pattern recognition, graph matching or
graph representation. Acceptance: no year gap after 2024, and a stated PR-community share. Fischer
2015 counts for **venue**, not recency." 1–2 days, **P0 — EiC checks independently**. Depends on
T-08. §0.5 row EiC.a2 marks it **NEW**.

### Q6 — Second customer? EiC.b (shares T-19), AE.2 (T-08's related-work section supplies the
citation contexts for the new references), R1.2 (AGM/gSpan framing). Load-bearing; keep.

### Verdict: **COVERED**

**Why**: T-19 exists, is P0, is correctly justified as independently checked, and its numeric
criterion (≥ 6 from 2025–2026, no year gap after 2024) is the right shape for a pass/fail item. The
plan's diagnosis at `plan.md:475` — "weakest current position: **nothing third-party after 2024**" —
matches my measurement exactly.

**Two criterion defects worth fixing, neither large enough to change the verdict:**

1. **"No year gap after 2024" is already satisfiable by self-citation.** The bibliography today has
   two 2025 entries, both by the author group. A checker reading "cover the state of the art" will
   not count them. **The criterion should read "≥ 6 third-party references from 2025–2026, with ≥ 1
   in 2026"**, which is what the clause means and what the current text does not require.
2. **The recency budget conflicts with §5.4** — carried as the EiC.a1 finding; not repeated here.

**Effort**: current plan **1–2 days** (`plan.md:567`). Proportionate response is the same 1–2 days,
derived as: venue+year classification of 43 references (43 lookups × ~1 min ≈ 45 min, and I have
already produced this classification — see re-derivation rows 10 and 12, which T-19 can adopt
directly and save half a day); literature search for 6 recent papers in graph canonicalisation /
graph matching / graph representation at PR-field venues (~4 h); writing them into §1.x with
individual comment per EiC.a4 (~2 h). **The classification half of T-19 is already done by this
audit** — the ticket's remaining work is the search and the writing, ≈ 1 day.

**Assumptions made**: that "last and current year" means 2025 and 2026 relative to the 2026 decision
date. The letter does not state the years.

---

## EiC.a3 — Do not make excessive citation to arXiv papers

**Operative clause** (`mail.txt:126`), verbatim:

> "Please do not make excessive citation to arXiv papers, but substitute them with their peer-reviewed versions, or papers from a single conference series."

**Modal**: "Please do not … but substitute them" → **REQUIREMENT**, compliance-checked.

### Q3 — Already in the manuscript?

This is the row where Q3 changes the answer most, so clause by clause:

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| arXiv citations replaced by peer-reviewed versions | **already satisfied for 5 of 6** | `kipf2017gcn`, `velickovic2018gat`, `xu2019powerful` all carry `booktitle = {International Conference on Learning Representations}`; `fey2019pyg` carries `booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds}`; `jain2024graphedx` carries `booktitle = {Advances in Neural Information Processing Systems}, volume = {37}`. Each additionally carries `note = {arXiv:…}` — a courtesy pointer, not the citation |
| remaining arXiv citation | 1 — **cannot be substituted** | `lopezrubio2025isalgraph`: `journal = {arXiv}, volume = {2512.10429v2}`. This is `[28]`, the paper being extended, and `plan.md:457` (decision 10) records it is and will remain arXiv-only |
| "excessive" | 1 of 43 = **2.3 %** | not excessive by any reading |

**Residual gap**: none substantive. The *displayed* arXiv count is 6, which is what a checker
grepping for "arXiv" sees; the *cited venues* are peer-reviewed for 5 of them.

### Q4 — Explicit `or`? **Yes**: "substitute them with **their peer-reviewed versions, or** papers
from a single conference series." The plan takes branch 1 for the five and neither for `[28]`. Branch
2 (substituting a single conference series) is not applicable — there is no conference version of
`[28]`. No fallback is needed because branch 1 is already satisfied.

### Q5 — Argument or measurement? **Argument, and it is already written.** `plan.md:459–460`:
"EiC.a's 'substitute arXiv citations with their peer-reviewed versions' **cannot be satisfied** for
[28]. Response: state it plainly in one sentence." That is the correct and complete response for the
one irreducible case.

### What the plan currently commits to

**T-08** via `plan.md:461–463`: "**strip the `note = {arXiv:...}` fields from the five entries that
already name ICLR / NeurIPS venues** (`kipf2017gcn`, `velickovic2018gat`, `xu2019powerful`,
`fey2019pyg`, `jain2024graphedx`). That takes the rendered arXiv count from **6 to 1**."
`manuscript.md:222` carries it as checklist item 12. §0.5 row EiC.a3 marked **✓ §5.3**.

I verified the five named keys are exactly the five that carry both a peer-reviewed venue and an
arXiv `note`. The plan's list is correct and complete.

### Q6 — Second customer? None. EiC.a3 is the sole driver.

### Verdict: **COVERED**

**Why**: the substantive compliance already exists for 5 of the 6 entries, the sixth is
irreducible and the plan answers it by disclosure rather than by pretence, and the note-stripping —
though cosmetic — is the right move because the EiC is checking a rendered bibliography, not a
`.bib` file. Proportionate: five line deletions plus one sentence in the response letter.

**Effort**: current plan folded into T-08's 4–10 days. Proportionate response **≈ 10 minutes**,
derived as: 5 `note = {arXiv:…}` lines deleted from `cas-refs.bib` (~2 min), recompile and confirm
the rendered bibliography shows one arXiv string (~5 min), one sentence in response-letter part 4
(~3 min). This is the cheapest row in my slice.

**Assumptions made**: that `elsarticle-num` renders the `note` field. It does for `@inproceedings`
in the standard `elsarticle-num.bst`; I did not open the rendered bibliography pages of `main.pdf`
to confirm the five strings appear. If they do not render, the work is already complete and only the
one-sentence disclosure of `[28]` is needed.

---

## EiC.a4 — Do not cite large groups of papers without individually commenting on them

**Operative clause** (`mail.txt:126`), verbatim:

> "Do not cite large groups of papers without individually commenting on them. So we discourage \" In prior work [1,2,3,4,5,6] …\"."

**Modal**: "Do not cite … without" → **REQUIREMENT**, compliance-checked.

**Reading the clause precisely**: the prohibition is on *uncommented* groups. The EiC's illustration
is a **six-key** group with no commentary. Two tests follow, and the second is the operative one:
(i) how many keys appear together, (ii) **whether each key is discussed**. A group of any size in
which each member is individually characterised satisfies the clause.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| no uncommented multi-key groups | **1 group, 2 keys** | `methodology.tex:803`: "$\mathsf{NP}$-hard to compute~\cite{garey1979,Zeng:2009}." — the two keys share one uncommented citation |
| no "large" groups | **already satisfied** | the largest uncommented group in the document is 2 keys; the EiC's example is 6 |
| individual commentary where several works are cited together | **already satisfied at `introduction.tex:31`** | see below |

**`introduction.tex:31` verbatim**:

> "Graph matching methods based on deep learning --- including joint learning-and-matching frameworks \cite{Liu2023glam}, position-aware structure embeddings \cite{Chen2023pagm}, neural subgraph matching \cite{Lan2023aednet}, and hypergraph convolution \cite{Bai2021hypergraph} --- learn task-specific graph representations but do not produce explicit encodings from which the input graph can be reconstructed."

This is **four separate `\cite{}` commands**, each preceded by its own descriptive phrase ("joint
learning-and-matching frameworks", "position-aware structure embeddings", "neural subgraph
matching", "hypergraph convolution"). It is not a group in the EiC's sense on either test — the keys
are not bracketed together, and every one is individually characterised. My group detector, which
flags `\cite{}` bodies splitting into ≥ 2 keys, correctly returned only `methodology.tex:803`.

**Residual gap**: one two-key citation at `methodology.tex:803`. Whether it even violates the clause
is arguable — 2 keys is not a "large group" — but splitting it is one clause of prose.

### Q4 — Explicit `or`? No.

### Q5 — Argument or measurement? Argument: the response letter can state, with the count, that the
revised manuscript contains no uncommented multi-key citation and that the single two-key case has
been split. That is checkable by the EiC in seconds and is stronger than a claim of compliance.

### What the plan currently commits to

**T-08** via §0.5 row EiC.a4 (`plan.md:19`): "comment `\cite{garey1979,Zeng:2009}` individually;
**do not 'fix'** `introduction.tex:31`", artifact `methodology.tex:803`. Repeated at `plan.md:477–479`.
`manuscript.md:222` carries it as checklist item 12.

**The exemption is correct.** `introduction.tex:31` requires no change; "fixing" it would break four
well-formed individual characterisations into four sentences and cost ~0.1 pages in a document with
zero headroom. The plan reached the right decision.

**One terminology correction.** `plan.md:478–479` calls it "**The four-way group** at
`introduction.tex:31`". It is not a group. A future reader who trusts the label and does not open
the line may conclude the exemption is a judgement call about size — it is not; the line simply is
not an instance of the pattern. Recommend rewording to "the four individually-commented citations at
`introduction.tex:31` — not a group; no change needed."

### Q6 — Second customer? None. EiC.a4 is the sole driver for the `methodology.tex:803` split.

### Verdict: **COVERED**

**Why**: one owner (T-08), one artifact (`methodology.tex:803`), a correct exemption for the only
line that looks like a violation and is not, and a fix proportionate to the ask — one clause, not a
rewrite. The manuscript is arguably already compliant; the split makes compliance visible.

**Effort**: current plan folded into T-08. Proportionate response **≈ 5 minutes**, derived as:
rewrite `methodology.tex:803` from `\cite{garey1979,Zeng:2009}` to two clauses each naming what its
citation contributes (Garey & Johnson for the NP-hardness reduction, Zeng et al. for the graph-edit
-distance-specific hardness result) — one sentence, ~3 min — plus recompile. **Verify with the
scanning command in re-derivation row 6 before upload**; it is scripted and takes seconds.

**Assumptions made**: none. Both lines were read verbatim.

---

## EiC.b — Cite recent work from the pattern-recognition field, not only the PR journal

**Operative clause** (`mail.txt:128`), verbatim:

> "Please make sure the revised version is relevant to the readership of the Pattern Recognition field. To this end, please make sure you cite RECENT work from the field of pattern recognition not only the Pattern Recognition journal."

**Modal**: "please make sure you cite" (twice) → **REQUIREMENT**, compliance-checked per `:124`.

**This is not EiC.a2.** a2 is about **recency** across the whole bibliography; b is about **field
relevance** — pattern-recognition *venues*, and specifically venues **beyond the PR journal itself**.
The capitalised "RECENT" makes recency a modifier on the field constraint, not a substitute for it.
A reference can satisfy a2 and not b (a 2026 chemistry paper), or b and not a2 (a 1983 PR Letters
paper). §0.5 assigns both to T-19, which is fine as ticket bookkeeping but must not collapse the two
criteria.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| cites the pattern-recognition field | partially satisfied | 6 × *Pattern Recognition* journal: `Bai2021hypergraph` (2021), `Riba2021gedgnn` (2021), `Fuchs2022matchinggraphs` (2022), `Chen2023pagm` (2023), `Lan2023aednet` (2023), `Liu2023glam` (2023) |
| **not only** the PR journal | **absent** | the only non-PR-journal pattern-recognition venues are `bunke1983inexact` (*Pattern Recognition Letters*, **1983**), `riesen2008iam` (SSPR, 2008) and `sanfeliu1983ged` (IEEE TSMC, 1983). **Zero** CVPR / ICCV / ECCV / ICPR / TPAMI / IJCV references exist |
| **RECENT** PR-field work | **absent** | the most recent PR-field reference of any venue is **2023**. No PR-field reference in 2024, 2025 or 2026 |

**Residual gap**: the bibliography's pattern-recognition representation is a **single-venue cluster
frozen at 2023**. The clause names exactly this failure mode in its own words ("not only the Pattern
Recognition journal"), which makes this the row an EiC is most likely to notice.

### Q4 — Explicit `or`? No.

### Q5 — Argument or measurement? Neither; this needs a literature search at named venues.

### What the plan currently commits to

**T-19**, shared with EiC.a2 (`plan.md:567`). Its acceptance criterion has two halves:

> "Acceptance: **no year gap after 2024**, and **a stated PR-community share**."

The first half is a testable threshold and belongs to EiC.a2. **The second half is not a criterion.**
"A stated PR-community share" requires that a number be *reported*; it fixes no value that number
must reach, so no outcome fails it. A bibliography with 6 PR-journal papers, none after 2023, and no
other PR-field venue satisfies "a stated PR-community share" by stating "16 %".

§0.5 row EiC.b (`plan.md:20`) additionally carries the correction "⚠ Fischer 2015 satisfies *venue*,
**not recency**" — which is right, and which I confirm: `Fischer2015hausdorff` is *Pattern
Recognition* 48(2), 2015, currently cited only from the commented block at `methodology.tex:806`.
Restoring it adds a PR-journal venue and nothing to recency.

Searched: `EiC.b`, `pattern recognition`, `venue`, `T-19`, `PR-community`, `readership` across
`plan.md`, `manuscript.md`, `gap-audit.md`, `competitors.md`, `statistics.md`.

### Q6 — Second customer? EiC.a2 (same ticket), AE.2 (the new §1.x is where these citations land).

### Verdict: **UNDER-blocking**

**Why**: a ticket exists, but for the EiC.b half it carries **no pass/fail test**, and EiC.b is a
compliance item the Editor-in-Chief checks independently of the reviewers (`mail.txt:124`) — rubric
§4 guard 3. Because §0.5 merges b into a2 under one ticket, T-19 can close with its only numeric
criterion (≥ 6 from 2025–2026, no year gap after 2024) satisfied entirely by **non-pattern-recognition
venues**, leaving the bibliography's PR-field coverage exactly where it is today: one journal,
nothing after 2023. That is the specific outcome `mail.txt:128` prohibits.

I record this as the **weaker of my three blocking findings** — the fix is a criterion edit, not new
work — but it stays blocking because the guard does not distinguish by cost.

**Effort**: current plan **1–2 days** for T-19, jointly with EiC.a2. Proportionate response **adds
≈ 0.5 day**, derived as: the venue classification is already done (re-derivation row 12, adoptable
directly, saving ~45 min); the incremental work is (a) a 20-minute criterion edit, and (b) directing
the ≥ 6 search at PR-field venues **other than** the PR journal — CVPR, ICCV, ECCV, ICPR, TPAMI,
IJCV, PR Letters, S+SSPR, GbRPR — which is the same search a2 already requires, so the marginal cost
is the constraint on where to look, ~2–3 h of extra screening, not a second search.

**If UNDER — what must exist**: T-19's acceptance criterion must be split so both halves are
testable. Concretely, and stated as a target rather than a prescription: **≥ 3 of the ≥ 6 new
references from pattern-recognition venues other than the *Pattern Recognition* journal, and ≥ 1
PR-field reference dated 2025 or 2026.** Without a threshold on the b-half, no ticket outcome can
fail EiC.b, and the demand that the Editor-in-Chief will check by inspection goes unanswered.

**Assumptions made**: that "the field of pattern recognition" includes the major vision and
pattern-recognition conferences (CVPR/ICCV/ECCV/ICPR) and TPAMI/IJCV, and the structural-pattern
-recognition workshops (S+SSPR, GbRPR) that are this paper's nearest community. The letter does not
enumerate venues. This reading is favourable to the paper, since GbRPR and S+SSPR are where the
GEDLIB and IAM literature lives and are therefore natural citation targets.

---

## EiC.c — Maximum 35 pages, double-spaced single column

**Operative clause** (`mail.txt:130`), verbatim:

> "Although the revision could lead to extending your article, it still can not exceed the page limits or violate the format, i.e. double spaced SINGLE column with a maximum of 35 pages for a regular paper and 40 pages for a review."

**Modal**: "can not exceed … or violate" → **REQUIREMENT**, and the hardest one in the letter: it is
pass/fail, mechanically checkable, and its opening subordinate clause ("Although the revision could
lead to extending your article") explicitly anticipates and refuses the excuse this revision would
have to make.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| ≤ 35 pages | **exactly at the ceiling** | `pdfinfo main.pdf` → `Pages: 35`. Zero headroom |
| double-spaced single column | already satisfied | `main.tex:6` — `\documentclass[review,times,number]{elsarticle}`; the `review` option produces double-spaced single-column output |
| format not otherwise violated | already satisfied | `Page size: 612 x 792 pts (letter)` |

**Residual gap**: the format is compliant and needs no work. **The entire risk is the page count**,
and it is a function of everything else the plan adds. EiC.c is the binding constraint on every
additive response in the revision.

### Q4 — Explicit `or`? No. (`plan.md`'s decision 16 creates one — main text vs supplementary — but
that is a strategy the authors are constructing, not a branch the letter offers.)

### Q5 — Argument or measurement? **Measurement, and the plan does not have it yet.** See below.

### What the plan currently commits to

**T-15** ("Page trim to 35 + supplementary + AI declaration", `plan.md:556`, 2 days, **P0**, depends
on all), with the budget at `manuscript.md` §3 (`:69–130`), the artifact inventory at §2 (`:41–65`),
and **decision 16** (`plan.md:53`) — the day-1 query to `patcog@elsevier.com`.

**The apparatus exists and is well-designed in three respects.** (1) `manuscript.md` §2 is a
19-row artifact inventory with a per-row page estimate. (2) §3.2 carries a **pre-declared priority
ranking of 11 additive items**, ordered, with items 10 and 11 identified as "the only two on the
list no reviewer requires" (`manuscript.md:122`). (3) §3.3 corrects decision 7 in the right place:
"Decision 7 — 'ignore the page budget while drafting' — is right for drafting and **dangerous at the
end**", with the instruction to track the page count at every commit (`manuscript.md:127–130`).
Ordering constraint 1 (`manuscript.md:238–239`) correctly sequences the E7 float fix before any trim.
That is a serious page-management apparatus and the plan deserves credit for it.

**But the arithmetic it rests on is not derivable as stated.** `manuscript.md:65` asserts "**Gross
addition ≈ 12–13 pages against 0 pages of headroom**", and `:83` concludes "Recoveries (≈ 4.75) do
not cover additions (≈ 12–13). The gap is ≈ 8 pages". Summing the §2 inventory's page column gives
**12.25**, so the 12–13 figure is the inventory sum. That sum contains two errors of opposite sign.

**Error 1 — seven replacements are counted at full size instead of as deltas.** Inventory rows
Fig. 1 (0.75), Fig. 2 (1.0), Fig. 3 (0.5), Fig. 4 (0.75), Tab. 1 (0.5), Tab. 2 (1.25) and Tab. 3
(1.25) total **6.0 pages**, and their disposition column reads *replace*, *restructure*, *demote*,
*replace*, *keep*, *replace*, *replace*. These artifacts **already occupy space in the 35-page
document**. Only their growth is an addition. Tab. 2 grows (5 → 10 datasets, new columns), Tab. 3
grows (competitor columns, brackets, CIs), Fig. 2 grows (restructured, competitor curves), Fig. 3
**shrinks** (demoted), Tab. 1 is unchanged ("keep"). A defensible net delta for these seven is
**+1.5 to +2.5**, not +6.0 — an **over-count of ≈ 3.5–4.5 pages**.

**Error 2 — every new prose section is missing from the inventory.** §1's rewrite map commits to
sections that appear nowhere in §2's page column: **§1.x Related work (NEW)** (`manuscript.md:24`),
**§2.2.x Complexity (NEW subsection)** (`:26`), the **§3.2 protocol replacement** (`:31` — "the whole
statistical protocol is replaced; pair-accounting ladder; exclusion justifications with counts";
only the ladder *table* is inventoried, at 0.5), **§3.3 Implementation** (`:33` — C++ engine, GEDLIB,
versions, artifact release), the **§5 limitations expansion** (`:37`), and the Suite 1 / Suite 2
description in §3.1 (`:30`). In a double-spaced single-column format these are the *largest* items
in the revision. A conservative estimate: related work 1.5–2.0, complexity 1.0–1.5, protocol prose
1.0, implementation 0.5, limitations 0.5, Suite 1/2 framing 0.3 → **≈ 4.8–5.8 pages, uncounted**.

Net, the two errors roughly cancel in magnitude (12.25 − 4 + 5 ≈ 13) — which is why the ≈ 8-page gap
is not wildly wrong. **But the composition is wrong, and the composition is what the strategy
depends on.**

**Why the composition matters — the supplementary branch is aimed at the wrong pages.**
`manuscript.md:101–104` says that if supplementary does not count, the relief comes from moving "the
calibration tables, the stratified analyses, the pair-accounting ladder, the sensitivity arms and the
full per-dataset result grids" to supplementary. Every one of those is an **artifact-inventory** row:
calibration 0.75 + ladder 0.5 + CD diagram 0.5 ≈ **1.75–2.5 pages of relief**. The ≈ 5 pages of new
**main-text prose** — a related-work section, a complexity subsection, an implementation section, an
expanded limitations paragraph — **cannot be moved to supplementary**; a reviewer asked for them to
be in the paper, and AE.2/AE.3 place them in the argument. So the favourable branch of decision 16
buys ≈ 2 pages against a ≈ 5-page prose growth, not the "the gap closes comfortably" that
`manuscript.md:103` promises.

**Decision 16's status.** `plan.md:53` and `manuscript.md:86–99` require the query on day 1;
`plan.md:968` schedules it for **Day 1 — 08-12**, i.e. tomorrow. It is **not yet sent and not yet
late**. `manuscript.md:88` notes "the plan has never used" the mailbox. `plan.md:986` (risk R3)
correctly makes the whole page strategy branch on the reply. I record it as **on schedule but
unexecuted**, and note that the reply latency is outside the authors' control — which is precisely
`manuscript.md:98`'s argument for sending it on day 1 rather than day 18.

### Q6 — Second customer? Every additive demand in the revision. EiC.c is the constraint that
converts all other over-scope into a compliance failure.

### Verdict: **UNDER-blocking**

**Why**: the page limit is pass/fail, checked by the Editor-in-Chief independently of the reviewers
(`mail.txt:124`), and the document is already at 35/35. An owner (T-15), a budget, a priority
ranking and a day-1 query all exist — but the budget's headline number double-counts seven
replacements as additions and omits ≈ 5 pages of new prose entirely, and the supplementary-relief
branch is targeted at the artifacts rather than at the prose that is the real growth. A 36-page
submission is rejected on format regardless of the science (`manuscript.md:129` says exactly this),
so a page budget that cannot be re-derived from its own inputs is a blocking defect.

**Effort**: current plan **2 days** for T-15 (`plan.md:556`), scheduled Days 17–19 (`plan.md:977`).
Proportionate response **adds ≈ 0.5 day, spent now rather than on day 17**, derived as: measure the
actual page occupancy of the seven existing artifacts in `main.pdf` (7 × ~8 min with a PDF viewer ≈
1 h); estimate the six new prose sections at ~15 min each (1.5 h); rebuild §2's table with two
columns — *current occupancy* and *net delta* — and re-sum (~0.5 h). Total ≈ 3 h. The output is a
budget whose ≈ 8-page gap figure is defensible and whose supplementary branch is aimed at pages that
can actually move.

**If UNDER — what must exist**:

1. **§2's page column split into "current occupancy" and "net delta"** for the seven replacement
   rows, so replacements stop being counted as additions.
2. **Page lines for the six new prose sections**, which are the largest uncounted items and the ones
   that cannot go to supplementary.
3. **The supplementary branch re-costed** against what can actually move — ≈ 2 pages of artifacts,
   not the whole gap.
4. **Decision 16 sent on 08-12 as scheduled.** No change needed; flagged only because nothing in the
   plan records it as sent, and `plan.md:986` makes the entire strategy conditional on the reply.

Without 1–3, the priority ranking at `manuscript.md:108–120` is applied to a wrong total, and the cut
made on day 17 will be either too small (submission over 35) or larger than necessary (a reviewer
demand cut that did not need to be).

**Assumptions made** (I could not ask): the ≈ 4.8–5.8-page prose estimate is mine, derived from the
scope of the six sections in `manuscript.md` §1 and the double-spaced single-column format, not
measured. It should be replaced by T-15's own measurement. My arithmetic conclusion — that the
budget is not derivable as stated — does not depend on the estimate's exact value, only on the fact
that the inventory assigns those sections **zero**.

---

## AE.1 — How graph size impacts the presented results

**Operative clause** (`mail.txt:59–60`), verbatim — the operative sentence is the second:

> "For example, the graphs studied are relatively small, but in many real-world machine learning applications graphs may be quite large, and both reviewers have questions about differences seen for the graph data sets studied, and their potential impact on the results.
> **How graph size might impact the presented results should be clear.**"

**Modal**: "should be clear" → **REQUIREMENT**. Note what it requires: **clarity about an impact**,
not a demonstration at scale. The subject is "the presented results" — the results already in the
paper.

**Full comment spans** `mail.txt:59–60`. **What the rest of it is doing**: `:59` is the premise
(graphs are small; real applications are large) and an attribution to the reviewers. The AE does not
ask for larger experiments; the AE asks that the size dependence of what is reported be legible.
`mail.txt:67` weights this row along with AE.2–AE.4c.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| the size ceiling is stated | **already satisfied** | `conclusion.tex:68` — "The canonical encoding, while formally complete, scales empirically as $T \sim n^{4.9}$, restricting its use to graphs of approximately 12 nodes within a 600-second timeout." |
| encoding cost vs size is reported | **already satisfied** | `results.tex:75` "of producing these encodings scales with graph size"; `:80` "as a function of graph size $n$"; `computational_experiments.tex:23` "for measuring how encoding time scales with the number of nodes" |
| speedup vs size is reported | **already satisfied** | `results.tex:225` "Geometric-mean speedup … **binned by graph size**"; `:231` "at every tested graph size ($n = 3$ to $11$ nodes)"; `:236` "**The speedup grows with graph size**" |
| compression vs size is reported | **already satisfied** | `results.tex:97` "Compression ratio vs.\ number of nodes"; `:125` "remains above unity for all observed graph sizes" |
| **how size affects the ρ(Levenshtein, GED) correlation** — the paper's central claim | **absent** | the correlation is reported per dataset and pooled, never stratified by `n`. The one statement that would have gestured at it is **commented out**: `results.tex:326–327` — "%A systematic study across larger graph families / %would be needed to quantify the generality of this asymmetry" |
| the ceiling is attributed to the right cause | **absent / wrong** | `conclusion.tex:68` attributes the ~12-node ceiling to **the canonical encoding**. The 600 s timeout binds on **exact GED**, the reference, not on IsalGraph |

**Residual gap**: two things, and both are smaller than "run a bigger experiment". (a) The
correlation result is never stratified by size, so the AE's question about "the presented results"
is genuinely unanswered for the paper's headline number. (b) `conclusion.tex:68` misattributes the
size ceiling to IsalGraph when it belongs to the exact-GED reference — a **factual correction**,
rubric §4 guard 1, uncuttable regardless of what else happens.

### Q4 — Explicit `or`? No.

### Q5 — Argument or measurement?

**Partly argument, and the plan already has it.** §0.5 row AE.1 (`plan.md:27`) records "ceiling
attributed to the **reference**, not to IsalGraph". That single reattribution answers the strongest
form of the AE's premise — the study is small because *exact GED* is exponential, not because the
encoding fails — and it costs one corrected sentence at `conclusion.tex:68`.

**The measurement is not thereby unnecessary.** "How graph size might impact the presented results
should be clear" is a requirement about the reported correlation, and a correlation reported at one
size regime cannot be made "clear" as a function of size by argument alone. The plan's Suite 2
(`n ≤ 98`, proven bracket) is the right instrument, and it has three other customers.

### What the plan currently commits to

§0.5 row AE.1 (`plan.md:27`): "Suite 1 (`n ≤ 12`, exact) / Suite 2 (`n ≤ 98`, proven bracket);
ceiling attributed to the **reference**, not to IsalGraph; relative bracket width vs `n`" → tickets
**T-01, T-05, T-06**, artifacts §3.1 and §4 size results. Decision 12 (`plan.md:46`) locks the cohort
at the IAM Graph Database reaching **n = 98 retained, an 8.2× extension**. Ordering: T-01 →
T-03/T-05 → T-06 (`plan.md:574`). Priority rank **7** of 11 (`manuscript.md:115`).

Searched: `AE.1`, `graph size`, `Suite 2`, `n = 98`, `scaling`, `size results`, `T-05`, `T-06`.

### Q6 — Second customer? R3.7a (scalability claims), R3.5b (the GED recompute that Suite 2 rides
on), F1 (the `n ≤ 12` filter is nearly vacuous), E1 (density and node count never reported). Four
customers. Load-bearing; not a cut candidate.

### Verdict: **COVERED**

**Why**: an owner exists across T-01/T-05/T-06, the compute is already justified by four demands,
the reattribution argument is recorded, and the response is proportionate to a requirement modal on
the paper's central claim. The prior audit's finding at `gap-audit.md:329` — "It is the one
measurement that directly answers AE.1" — I re-tested against `results.tex` and `conclusion.tex` and
confirm.

**One gap, reported per my brief's question 7 — the page cost of the size story is unbudgeted.**
`manuscript.md` §3.2 ranks "**Size-scaling results (Suite 2)**" at position 7 of 11
(`manuscript.md:115`), but **§2's artifact inventory contains no row for it.** The size material is
absorbed into Tab. 2 (1.25, replace), Tab. 3 (1.25, replace) and the calibration table (0.75), none
of which is labelled as AE.1's. Consequence: when the day-17 trim reaches rank 7, there is no page
figure to cut *against* — the trimmer would have to cut a row that other demands also own. Every
other ranked item maps to a named inventory row; AE.1's does not.

**Effort**: current plan — Suite 2 is inside T-01 (1–2 d), T-05 (5–10 d) and T-06 (10–14 d), none of
which exists solely for AE.1, so AE.1's **marginal** compute cost is close to zero: the bracket is
computed for the whole cohort anyway, and stratifying ρ by `n` is a `groupby` over results T-06
already produces. **Marginal effort ≈ 0.25 day** (the stratified table and its paragraph), derived
as: one aggregation over existing output (~1 h), one table (~1 h), one paragraph (~30 min).
Proportionate: yes — the AE's requirement is answered by 0.75 pages of new material plus one
corrected sentence, not by a new experimental arm.

**Assumptions made**: that T-06 emits per-`n` correlation output. `statistics.md` is `audit-r3`'s
slice and I did not verify the stratification variable list; if `n` is not among them, AE.1's
deliverable needs an explicit line in `statistics.md` rather than being assumed from the recompute.

---

## AE.2 — Framing within previous work, and additional references

**Operative clause** (`mail.txt:62`), verbatim:

> "For related work, the reviewers point out that the work needs to be more solidly framed within the context of previous work in this area, and that additional references are needed to capture the specific contributions of the work in the paper."

**Modal**: reported speech — "the reviewers point out that the work **needs to be** … and that
additional references **are needed**". No imperative of the AE's own. It becomes a **REQUIREMENT**
through `mail.txt:67` ("Please address these concerns thoroughly"), whose antecedent is `:59–66`.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| a positioning discussion exists | **partially satisfied** | `introduction.tex:26–33` surveys embeddings, MPNNs, graph transformers, GraphRNN, VAEs, deep graph matching, matching-graph formalisms — with individual commentary per work |
| framed against **graph canonicalisation** | **absent** | no citation of AGM, gSpan, nauty/Traces, bliss, graph6/sparse6 or Babai appears anywhere in the 43 (verified against the full citation-order list) |
| the positioning claim is defensible | **overstated** | `introduction.tex:33` — "No existing method is simultaneously compact, reversible, structure-preserving, and canonicalisable for arbitrary graphs." A claim of this form is not supportable while the canonicalisation literature is uncited |

**Residual gap**: the canonicalisation literature — which is where R1.2 says the true comparators
live — and the softening of `introduction.tex:33` that follows from citing it.

### Q4 — Explicit `or`? No. ### Q5 — Argument or measurement? Neither; a literature section.

### What the plan currently commits to

**T-08** ("Related work section + bibliography to ≤55", `plan.md:549`, 4–10 d, P1, depends on T-07),
producing **§1.x Related work (NEW section)** — "canonicalisation literature: AGM, gSpan,
nauty/Traces, bliss, graph6/sparse6, Babai; the AE.3 comparison table" (`manuscript.md:24`), drivers
AE.2, AE.3, R1.2, R3.1. §0.5 row AE.2 marked **✓**. Recovery line `manuscript.md:77` offsets 0.5
pages by restructuring §1.

### Q6 — Second customer? R1.2 (the comment the AE is relaying), AE.3 (the table lives in this
section), R3.1b (softening "no existing method"), EiC.a2/EiC.b (this is where new references land).
Five customers.

### Verdict: **COVERED**

**Why**: T-08 exists, its scope matches the ask (framing + references), it is placed in a named new
section, and its reference additions are budgeted — imperfectly, per EiC.a1, but explicitly. The
priority rank is 5 of 11 (`manuscript.md:113`), above the size results and the label results, which
is consistent with `:67`'s weighting.

**Effort**: current plan **4–10 days** (`plan.md:549`), the widest range on the board. Proportionate
response **≈ 3 days**, derived as: read six canonicalisation sources well enough to characterise
each individually per EiC.a4 (6 × ~2 h = 1.5 d), draft 1.5–2 pages of section (~1 d), fold in the
new-reference screening T-19 shares (~0.5 d). The 10-day upper bound is T-08's dependency on T-07
and its co-ownership of the bibliography, not the section itself.

**Note for the orchestrator**: T-08 is **P1** while its dependents T-17 and T-19 are **P0**
(`plan.md:564`, `:567`), and `plan.md:575` puts T-07 → T-08 → T-19 on a chain. A P1 ticket gating two
P0 tickets is a priority inversion. Not my call to fix, but it belongs in the reconciliation.

---

## AE.3 — Side-by-side comparison of existing graph representations

**Operative clause** (`mail.txt:63–64`), verbatim:

> "Related to this, reviewer 3 has asked for a detailed side-by-side comparison of existing graph representations with the proposed one, which fairly and completely identifies the properties, strengths, and limitations of each --
> this will help focus the presentation of work in the paper, and clarify the contribution of the work."

**Modal**: reported ("reviewer 3 has asked for"), but the AE endorses it **in their own voice** in
the trailing clause — "this **will** help focus the presentation of work in the paper, and clarify
the contribution of the work". Combined with `:67`, a **REQUIREMENT** carrying more weight than R3.1a
alone. This is the one reviewer request the Area Editor singles out and adopts.

**Three deliverables in one sentence**, and they are not the same thing: **properties**, **strengths**,
**limitations** — "of **each**", i.e. per representation, **including the proposed one**.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| a comparison with existing representations | **absent** | no comparison table exists; `introduction.tex:26–33` is a prose survey that never places IsalGraph beside a named alternative on shared axes |
| properties of each | absent | — |
| strengths of each | absent | — |
| limitations of each | absent for others; **present for IsalGraph in prose** | `conclusion.tex:68–71` gives IsalGraph's three limitations (n≈12 ceiling, ρ degradation with density, labels discarded) |
| a claim the comparison would discipline | **present and overstated** | `introduction.tex:33` "No existing method is simultaneously compact, reversible, structure-preserving, and canonicalisable for arbitrary graphs"; `conclusion.tex:74` "No prior sequential encoding … **IsalGraph is the first to do so**, to the best of our knowledge" |

**Residual gap**: the whole artifact. This is the largest genuine hole among the AE rows.

### Q4 — Explicit `or`? No. ### Q5 — Argument or measurement? Both: the property columns are
argued from the literature, the efficiency and scalability columns are measured by T-04.

### What the plan currently commits to

**T-17** (`plan.md:564`), created by the prior audit: "**AE.3 comparison table as a paper artifact**
— existing graph representations vs IsalGraph, with **properties, strengths and limitations of
each**, on R1.2's five axes (uniqueness, expressiveness, computational efficiency, scalability,
**downstream learning = not evaluated**). Rows populated from T-04's measurements, not asserted.
Licenses the softening of `introduction.tex:33` / `conclusion.tex:74`." 2–3 days, **P0 — the AE
endorsed this one in their own voice**. Depends on T-04, T-07.

**Verified against my brief's question 5**, three checks:

1. **Does T-17 exist?** Yes, `plan.md:564`.
2. **Is it a paper artifact, not just a response-letter table?** Yes — `manuscript.md:52` inventories
   it as a **NEW** artifact at **0.75 pages**, and `manuscript.md:24` places it in §1.x. It is also
   priority **1 of 11**, "the Area Editor endorsed it in their own voice. **Non-negotiable**"
   (`manuscript.md:108`). Correct weighting.
3. **Do the axes match `:63–64`?** **Partially — and this is the one substantive defect.** The
   ticket's *prose* names all three ("properties, strengths and limitations of each"), but the *axes
   it specifies* are R1.2's five, and all five are **properties** (uniqueness, expressiveness,
   computational efficiency, scalability, downstream learning). A five-column feature matrix
   delivers clause 1 of 3. **Strengths and limitations per representation are not columns in that
   matrix** and need either two further columns or a per-row prose note. As specified, T-17 would
   deliver a feature matrix and the response letter would have to claim it satisfies "properties,
   strengths, and limitations".

### Q6 — Second customer? R3.1a and R3.7b (the [28]/[29] delta, a **different object** — it
decomposes *our own* prior work, per `gap-audit.md:87–88`, which I re-tested and confirm), R3.1b and
R3.6a (both softenings are licensed by this table), R1.2 (the five axes are R1.2's). Four customers
plus the AE's endorsement.

### Verdict: **COVERED**

**Why**: T-17 exists, is P0, is a paper artifact with a page allocation and a section home, is ranked
first among the additive items, and is populated from measurement rather than assertion. The prior
audit's GAP-1 (`gap-audit.md:78–88`) is genuinely closed — I re-derived it rather than inheriting it,
and the mapping of AE.3 to the [28]/[29] delta table has indeed been separated.

**Criterion sharpening required (does not change the verdict, but must reach T-17's acceptance
criteria):** the artifact must carry, **per representation and including IsalGraph**, an explicit
*limitations* entry. The AE's phrase is "fairly and completely identifies the properties, strengths,
and limitations of **each**"; "fairly" is doing work — a table in which only the competitors have
limitations is the failure mode the word anticipates. At 0.75 pages
(`manuscript.md:52`) a 6–7-row × 5-column matrix plus a limitations column is tight but feasible in
double-spaced single column if the limitations are terse noun phrases.

**Effort**: current plan **2–3 days** (`plan.md:564`). Proportionate response the same 2–3 days;
the sharpening costs **≈ 0.1 day** — a two-line edit to T-17's acceptance criteria adding
"strengths" and "limitations" as required per-row content, plus ~1 h to write the limitation entries
once the rows exist. The strengths and limitations are already known from T-04's measurements and
from the literature read for §1.x; nothing new must be computed.

**Assumptions made**: that "existing graph representations" means the competitor set (graph6/sparse6,
nauty/Traces canonical forms, bliss, AGM canonical adjacency matrix, gSpan min-DFS code) plus SMILES
/ SELFIES as the molecular precedents. `plan.md:545` names the first five; the last two are cited in
`introduction.tex:20–21` and belong in the table.

---

## AE.4a — The choice of benchmark models

**Operative clause** (`mail.txt:66`), verbatim (the full sentence, with a4a's item emphasised):

> "Both reviewers also ask for a more detailed and rigorous analysis in the experiment designs, including **the choice of benchmark models**, differences in information and structure in the graph datasets used (e.g., fully labeled, vs. partially-labeled), and in the associated analysis of the results."

**Modal**: "Both reviewers also ask for" — reported speech, made **REQUIREMENT** by `mail.txt:67`
("Please address these concerns thoroughly"), whose antecedent is `:59–66`.

### The referent — "benchmark models" means comparison methods, not datasets

`audit-r1`'s R1.1 verdict is conditional on this, so it is answered directly. **"The choice of
benchmark models" means the set of representations/methods IsalGraph is compared against.** Three
independent arguments, all from the sentence itself and the reviewer text it relays:

1. **The list already contains a datasets item.** The sentence enumerates three things: "the choice
   of benchmark models", "**differences in information and structure in the graph datasets used**
   (e.g., fully labeled, vs. partially-labeled)", and "the associated analysis of the results".
   Reading item 1 as datasets makes it redundant with item 2 and leaves "benchmark models"
   unexplained.
2. **"Both reviewers also ask for" names the antecedent comments, and both are about comparators.**
   R1.1 (`mail.txt:75`): "this comparison appears somewhat unfair … **A more informative evaluation
   would compare the proposed methods against alternative approaches that address a similar problem
   setting.**" R3.6a asks for comparisons with established reversible graph serializations. Neither
   is about the datasets; both are about what the method is benchmarked *against*.
3. **"in the experiment designs"** frames all three items as design choices. What you compare against
   is the archetypal experiment-design choice; which datasets you use is item 2's subject.

**Consequence for `audit-r1`**: **the six competitor backends have a requirement-modal owner.** R1.1
in isolation carries a suggestion modal ("would compare"). AE.4a relays it and `mail.txt:67` makes it
a requirement. So the competitor set is **not** cuttable as suggestion-only work, and R3.6a's cheap
branch ("either narrow the claim … or include comparisons") does not dispose of it either — R3.6a's
`or` is about a *claim*, while AE.4a is about the *experiment design*. `README.md:179` already
reaches this conclusion ("The competitors survive because **R1.1 and AE.4a demand them
independently**"); I re-derived it from the letter and confirm it.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| a benchmark model set exists | **absent** | the only comparator is exact GED itself; `results.tex:225` compares "the \IsalGraph{} pipeline over exact GED". No alternative representation appears in any experiment |
| the choice is justified | absent | there is no choice to justify |
| comparison is fair | **the specific defect R1.1 names** | `results.tex:231`, `:236` compare a per-graph encoding cost against a per-pair GED cost |

**Residual gap**: the comparator set and a stated rule for selecting it.

### Q4 — Explicit `or`? Not in AE.4a. ### Q5 — Argument or measurement? Measurement: a comparator
set is not arguable into existence. The *fairness* half (per-graph vs per-pair axes) is a
restructuring, not a new experiment.

### What the plan currently commits to

§0.5 row AE.4a (`plan.md:30`): "six competitor representations enter three experiments". **T-04**
(`plan.md:545`, 3–8 d, P0) builds `src/isalgraph/competitors/` — graph6, nauty, bliss/Traces, AGM,
gSpan min-DFS. **T-04a** (`plan.md:565`, 0.5–1 d, P0) is metric feasibility: "attempt every
(representation × distance) cell on a fixed 200-graph / 19,900-pair stratified sample; select each
primary distance by the pre-declared rule. **Must close before any production distance matrix is
computed.**" **Decision 18** (`plan.md:49`): "**Competitor distances are selected by measurement
(T-04a), not by assertion**, with the rule fixed in advance and ties broken on cost — never on
correlation with GED." Protocol in `competitors.md` §2. Risk **R4** (`plan.md:987`) pre-decides the
gSpan fallback.

### Q6 — Second customer? R1.1, R3.6a, AE.3 (T-17's rows are populated from T-04's measurements,
`plan.md:564`), R1.2 (the five axes). Four customers.

### Verdict: **COVERED**

**Why**: the demand is owned by T-04 → T-04a with a P0 priority, the *choice* half — which is what
AE.4a literally asks about — is answered by decision 18's pre-declared, cost-tie-broken selection
rule rather than by post-hoc justification, and the fairness half is owned by the Figure 2
restructuring at `manuscript.md:35`. Ties broken "never on correlation with GED" is the right
guard against the exact charge of a rigged comparator choice.

**Effort**: current plan **3–8 days** (T-04) + **0.5–1 day** (T-04a). Proportionate: yes, and the
range is honest — the 8-day upper bound is the gSpan minimum-DFS-code extraction that risk R4 flags
as possibly unexposed by `LasseRegin/gSpan`. Marginal effort attributable to AE.4a specifically
(as opposed to R1.1/AE.3) is **≈ 0.5 day**: the paragraph stating the selection rule and its
outcome. The backends themselves are shared infrastructure.

**Assumptions made**: none beyond the referent argument above, which is derived from the letter.

---

## AE.4b — Differences in information and structure across the datasets

**Operative clause** (`mail.txt:66`), the second enumerated item, verbatim:

> "…**differences in information and structure in the graph datasets used (e.g., fully labeled, vs. partially-labeled)**…"

**Modal**: REQUIREMENT via `mail.txt:67`.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| structural differences across datasets | **partially satisfied** | Tab. 2 reports dataset properties; but E1 records that **density and node count are never computed**, so the structural axis the AE names is incomplete |
| **label content** per dataset | **absent, and misstated** | `conclusion.tex:70` — "node and edge labels, **present in all five benchmark datasets**, are discarded during encoding"; repeated at `:81` — "which are **present in all five datasets** used here". R1's premise (`mail.txt:79`) is that "AIDS is the only dataset among the evaluated benchmarks that contains node and edge labels, whereas IAM and LINUX are unlabeled" |
| the impact of those differences on results | absent | the AIDS degradation is attributed to density alone (`conclusion.tex:69`) |

**Residual gap**: a label-content column, the correction of a **false statement** in the conclusion,
and one paragraph connecting the two to the AIDS result. This is defect **E6** and rubric §4 guard 1
(factual corrections) applies — it is uncuttable regardless of what happens to the rest of T-18.

### Q4 — Explicit `or`? No. ### Q5 — **Argument, largely.** The label-confound hypothesis is
refutable from the paper's own numbers: IAM is unlabeled and reaches ρ ≈ 0.934 (`conclusion.tex:69`),
so label loss cannot be the mechanism behind AIDS's ρ = 0.349 — both sides of the AIDS correlation
are topology-only. That is a free counter-example already in the manuscript.

### What the plan currently commits to

§0.5 row AE.4b (`plan.md:31`): "a **label-content column** in the dataset table (Tier 0)" → **T-18**
(`plan.md:566`), tiered, **PI decision on effort due 2026-08-18**. "Tier 0 (rebuttal, label column,
**E6 fix**, future work) is **not optional**; Tier 1 (topological collision count) recommended; Tier
2 (label-aware GED **logged, not written up**) proposed at 0.3 core-hours; Tier 3 (a results
subsection) **declined by default**." Artifact §3.1 dataset table + `results.tex` §4.x
(`manuscript.md:36`), 0.75 pages (`manuscript.md:59`).

### Q6 — Second customer? R1.3a/b/c (the same label material), E6 (the factual error). Three
customers for Tier 0.

### Verdict: **COVERED**

**Why**: Tier 0 is marked not optional, it carries the E6 factual correction, and the tiering is the
correct response to the R1.3 over-scoping lesson — the expensive arm (Tier 3, a results subsection)
is declined by default rather than built. The AE's phrase "differences in **information** and
structure" is answered by the label column (information) plus E1's density/node-count restoration
(structure), and both are in the plan.

**Effort**: current plan **0.3–1 day** for Tier 0–1 (`plan.md:566`). Proportionate: yes. Derived
marginal cost of the AE.4b-specific piece: one table column (~30 min), the E6 sentence fix in
`conclusion.tex:70` and `:81` (~10 min), one paragraph connecting label content to the AIDS
interpretation (~1 h) ≈ **0.25 day**.

**One scheduling risk**: the PI tier decision is due **2026-08-18** (`plan.md:972`), day 7, and Tier
2 "must be configured into the T-06 run, not bolted on afterwards". Tier 0 does not depend on that
decision and should not be gated behind it.

---

## AE.4c — The associated analysis of the results

**Operative clause** (`mail.txt:66`), the third enumerated item, verbatim:

> "…**and in the associated analysis of the results.**"

**Modal**: REQUIREMENT via `mail.txt:67`. The head noun is "a more detailed and rigorous analysis",
so the ask is that the analysis of the results be more rigorous — the same complaint R1.3 opens with
("the discussion of the experimental results is rather overlooked", `mail.txt:79`) and R3.5 states
in detail.

### Q3 — Already in the manuscript?

| Clause of the ask | State in the submitted manuscript | Evidence |
|---|---|---|
| results are analysed | partially satisfied | `computational_experiments.tex:90–233` is the protocol; `results.tex` reports it |
| the analysis is rigorous | **the contested point** | `computational_experiments.tex:198` — correlation "is aggregated as the geometric mean, stratified by graph size"; pooled-across-datasets analysis is primary, which is what R3.5b objects to |
| causal attributions are examined | **absent** | `conclusion.tex:69` attributes the AIDS degradation to density with no test of the attribution |

**Residual gap**: owned in full by the statistical protocol replacement.

### Q4 — Explicit `or`? No. ### Q5 — Mixed; this is `audit-r3`'s call on R3.5, not mine.

### What the plan currently commits to

§0.5 row AE.4c (`plan.md:32`): "`statistics.md` D1–D15" → **T-02** (statistics lock, 2–4 d, P0) and
**T-06** (full recompute, 10–14 d, P0), artifacts §3.2 and §4. `manuscript.md:31` maps the whole of
`computational_experiments.tex:90–233` to replacement. Priority rank **4 of 11**
(`manuscript.md:112`).

### Q6 — Second customer? R3.5a/b/c, R1.3, E2/F2, D1–D15. The most heavily shared row in the AE block.

### Verdict: **COVERED**

**Why**: an owner exists, it is P0, and the artifact is a wholesale protocol replacement rather than
a patch. Whether that replacement is **proportionate** — in particular whether recomputing every GED
under one cost model at 1,000–1,650 core-hours is required by R3.5b or is a deliberate over-response
— is `audit-r3`'s determination, and `README.md:178` already flags it for exactly that scrutiny. From
the AE's side the row is covered; I do not double-count the scale question here.

**Effort**: current plan, T-02 2–4 d + T-06 10–14 d, but neither exists solely for AE.4c.
AE.4c-attributable marginal effort ≈ **0**: it is satisfied by whatever R3.5 produces.

---

## AE.5 — The reviewers' additional comments

**Operative clause** (`mail.txt:69`), verbatim:

> "There are additional comments from the reviewers that should also be addressed in the revised paper -- please check their comments carefully when preparing your revision."

**Modal**: "**should also be addressed**" + "**please check** their comments carefully" →
**REQUIREMENT**. Two verbs, both operative: address, and check.

**What the sentence does**: it closes the AE's own enumeration (`:59–66`) and hands the remainder to
the reviewers' text. It is a **completeness obligation**, not a distinct deliverable — which is why
it is easy to skip, and why skipping it is exactly the failure it warns about.

### Q3 — Already in the manuscript? Not applicable; the object is the plan's coverage, not the text.

### What the plan currently commits to

**Nothing. There is no AE.5 row.** Searches run:

- `grep -rn 'AE\.5' .claude/notes/review/source/` → **zero hits outside `inventory.md`**.
- `grep -rn 'additional comments\|:69\b' plan.md manuscript.md gap-audit.md` → one hit, and it is
  unrelated (`manuscript.md:35`, a Figure 2 line).
- `plan.md` §0.5's Area Editor block runs AE.1, AE.2, AE.3, AE.4a, AE.4b, AE.4c (`plan.md:27–32`)
  and stops. `mail.txt:69` has no row, no ticket and no artifact.

### Is it subsumed by the numbered R1/R3 rows?

**Largely, but not entirely.** §0.5 carries 20 reviewer rows (R1.1–R1.3d at `plan.md:38–44`,
R3.1a–R3.7e at `:50–68`), which is a fine-grained decomposition — R3's seven comments become
seventeen rows. Against that, I checked the two un-numbered passages my brief names:

- **R1's opening, `mail.txt:73`**: "The paper is interesting as it opens up new research directions
  in sequential graph-string representations. Overall, the paper raises the following questions and
  concerns:" — pure framing plus a colon introducing the numbered list. **No ask. Nothing dropped.**
- **R3's preamble, `mail.txt:83`**: "The manuscript's main strength is its extension of prior
  instruction-based representations … However, **the rationale, novelty, methodological details, and
  interpretation of the results require further clarification.**" That final sentence is
  **requirement-modal** ("require further clarification") and names **four** targets. Three map
  cleanly onto numbered rows — *novelty* → R3.1, *methodological details* → R3.4, *interpretation of
  the results* → R3.5. **"Rationale" has no clean numbered owner.** R3.7e ("four broad statements")
  is the nearest, but it is about specific overstatements, not about why the method is designed as it
  is. Whether "rationale" is genuinely uncovered is `audit-r3`'s determination — I flag it as the
  one un-numbered requirement-modal sentence that the decomposition may have dropped.

### Q6 — Second customer? By construction, every reviewer row.

### Verdict: **UNDER-major**

**Why**: a requirement-modal Area Editor sentence has no row, no ticket and no mention anywhere in
the plan folder. Its substance is *mostly* discharged by the 20 numbered rows, so this is a process
gap rather than a missing artifact — but the gap is precisely that **nobody has ever asked whether
the numbered decomposition dropped anything**, and one candidate (`:83`'s "rationale") exists.

**Departure from the default, stated explicitly**: my brief sets `UNDER-blocking` as the default for
this slice under rubric §4 guard 3. Guard 3 covers "compliance items the editor checks
independently: page and word limits, reference-count bounds, declarations, source-file formats,
submission mechanics" — the **Editor-in-Chief's** pass/fail list. AE.5 is an Area Editor requirement,
agenda-setting rather than mechanically checked, so guard 3 does not reach it. **UNDER-major.**

**Effort**: current plan **0 days** — no owner. Proportionate response **≈ 0.25 day**, derived as:
one pass re-reading `mail.txt:73–79` (R1, 7 lines) and `:83–116` (R3, 34 lines) against §0.5's 20
reviewer rows, marking any sentence in the imperative or subjunctive that no row claims — ~1.5–2 h
for 41 lines at the care level this audit uses. Output: either a confirmation that the decomposition
is complete, or one or two new rows.

**If UNDER — what must exist**: an **AE.5 row in §0.5** whose decision reads "completeness check:
every imperative/subjunctive sentence in `:73–116` is claimed by a numbered row", with a named owner
and a date before T-14 assembly. Without it, the Area Editor's closing requirement is the one demand
in the letter that nobody has read against the plan, and `mail.txt:69` is the sentence a round-2
Area Editor is most likely to quote back.

**Assumptions made**: that §0.5's 20 reviewer rows are faithful decompositions of the numbered
comments. I did not audit them — that is `audit-r1`'s and `audit-r3`'s slice. My finding is that the
*completeness question itself* has no owner, which is true regardless of how those rows turn out.

---

## Notes for the orchestrator

### Cross-voice overlaps — merge to one row, do not duplicate

- **AE.3 vs R3.1a / R3.7b.** These are **different objects** and must not be merged. AE.3 (`:63–64`)
  asks for existing representations vs IsalGraph on shared axes → **T-17**. R3.1a asks for
  inherited / modified / new versus **our own** [28] and [29] → **T-07**. `manuscript.md:52` and
  `:53` correctly inventory them as two artifacts at 0.75 pages each. What AE.3 adds to R3.1a is the
  Area Editor's endorsement in their own voice, which is why T-17 is P0 and priority 1 of 11.
- **AE.4a vs R1.1.** One demand, two voices. AE.4a is decisive for `audit-r1`: it converts R1.1's
  suggestion modal ("*would* compare") into a requirement via `:67`. **The six competitor backends
  are requirement-modal work.** Also note R1.1 has a second half — the per-graph vs per-pair category
  error in Figure 2 — which is a factual restructuring (`manuscript.md:35`) and uncuttable under
  guard 1 independent of the competitors.
- **AE.4b vs R1.3.** One demand, two voices; T-18 Tier 0 serves both. AE.4b is the reason Tier 0 is
  "not optional" rather than a courtesy — R1.3 alone carries a suggestion modal.
- **AE.1 vs R3.7a.** Overlapping but distinct. AE.1 asks how size impacts **the presented results**;
  R3.7a is about scalability **claims**. Suite 2 serves both; the claim-scoping half is nearly free
  and should not wait on the compute.
- **EiC.a2 / EiC.b vs AE.2 / R1.2 / R3.1 / R3.6a.** All draw on the **same 12 bibliography slots**.
  This is the shared resource with the most claimants and no reconciliation owner — see EiC.a1.

### Priority statements bearing on demands outside my slice

- **`mail.txt:67`** — "Please address these concerns thoroughly, as they will **strongly influence
  the potential impact of the work and citation of the paper**". Antecedent `:59–66` = AE.1–AE.4c.
  This is what upgrades AE.2, AE.3, AE.4a/b/c from reported speech to requirements, and by extension
  R1.1, R1.3 and R3.1 where the AE relays them. It is currently mis-cited as M3's authority and
  appears nowhere in §0.5 as a weighting.
- **`mail.txt:124`** — "**I will check that these are adhered to before your paper is approved for
  publication**". Makes every EiC row pass/fail regardless of the reviewers' verdict.
- **`mail.txt:69`** — AE.5, unowned. See above.

### Anything I could not verify, and precisely why

1. Whether `elsarticle-num.bst` renders the `note` field, i.e. whether the five arXiv strings
   actually appear in `main.pdf`'s bibliography. I read the `.bib` and the `.tex`, not the rendered
   pages. If they do not render, EiC.a3 is already fully compliant and the note-strip is unnecessary.
2. Whether the upload package includes `elsarticle.cls` and the `cas-*` support files.
   `manuscript.md:211` says "sources present" without enumerating.
3. Whether T-06 emits per-`n` correlation output, which AE.1's deliverable assumes. `statistics.md`
   is outside my slice.
4. Whether decision 16's query has been sent. Scheduled for **Day 1 = 2026-08-12** (`plan.md:968`);
   nothing records it as sent, and nothing records it as late.
5. My ≈ 4.8–5.8-page estimate for the new prose sections is derived from scope, not measured. The
   EiC.c finding does not depend on its value — only on the inventory assigning those sections zero.

### Factual premises worth recording neutrally

- **`plan.md:446` says of [29]: "the paper is unavailable."** By citation order, **[29] =
  `ThurnhoferHemsi:2025`**, "Representation of Molecules by Sequences of Instructions", *Journal of
  Chemical Information and Modeling* **65(15):7936–7955, 2025**, by Thurnhofer-Hemsi, García-Aguilar,
  Fernández-Rodriguez and López-Rubio — a complete peer-reviewed entry already in `cas-refs.bib` and
  already cited in the manuscript. If [29] is this paper, T-07's premise that only the source code is
  available should be re-checked; the published article would settle the LSTM-experiment half of D19
  directly. This is `audit-r3`'s slice (R3.1a); I raise it because I established the numbering.
- **Both 2025 references are group self-citations.** [28] López-Rubio (arXiv) and [29]
  Thurnhofer-Hemsi et al. incl. López-Rubio. There are **zero third-party references from 2025 or
  2026**. This is not a scoring point against the authors — it is the measurement that makes EiC.a2's
  ≥ 6 criterion necessary rather than precautionary.
- **R1's factual premise at `mail.txt:79`** — "AIDS is the only dataset among the evaluated benchmarks
  that contains node and edge labels, whereas IAM and LINUX are unlabeled" — **contradicts
  `conclusion.tex:70`** ("labels … present in all five benchmark datasets"). The reviewer is right and
  the manuscript is wrong; this is E6. Recorded neutrally: it strengthens the response, because
  correcting it is a factual fix the authors make on their own initiative, and it is the evidentiary
  basis of AE.4b's label column.
