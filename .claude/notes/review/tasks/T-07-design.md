# T-07 — design note

**Ticket**: *Read [28] and [29]; inherited/modified/new delta table plus the sufficiency paragraph;
resolve D19.*
**Serves**: R3.1a(i), R3.1a(ii), R3.7b, AE.5 ("rationale"), D19.
**Depends**: nothing. **Blocks**: T-08 (related-work section), and via §2.3 the whole of new §2.
**Priority**: P0. **Board estimate**: 1–4 h.
**Base commit**: `9048228` (`integration/2026-08-26-t13-complexity`).
**Reading list per the board**: [corrections](../plan/corrections.md) §4,
[decisions](../plan/decisions.md) 9. Also read: [prose](../plan/prose.md) §2 and §10,
[manuscript](../plan/manuscript.md) §2, [demands](../plan/demands.md) R3.1a/AE.5,
[verified-discrepancies](../source/verified-discrepancies.md) D19, `mail.txt:83–90`.

---

## 1. State measured now, not assumed

Seven items were checked against what the plan asserts. **Two differ, and one of them was blocking.**

| # | Plan says | Measured 2026-08-26 | Consequence |
|---|---|---|---|
| 1 | 🔴 **[28]'s PDF is in-repo at `docs/references/2512_10429v2.pdf`** ([corrections](../plan/corrections.md) §4; also `.claude/CLAUDE.md` "Key References") | **`docs/references/` does not exist.** The file was deleted in `7d18f52` *"Initialize github pages site structure"*, together with `docs/references/Idea.pdf` and `docs/original_code_and_files/2512.10429v2.pdf` | **Would have blocked the ticket.** Recovered from `a23acbf` to scratchpad. 12 pp., `arXiv GenPDF`, title *Representation of the structure of graphs by sequences of instructions*, single author E. López-Rubio, 13 Dec 2025. **Two plan files carry a dead path → fix at close.** |
| 2 | 🟡 **[29] is published, read the paper** ([decisions](../plan/decisions.md) 9) — no access route named | **[29] is CC BY open access.** PMC12344769, DOI `10.1021/acs.jcim.5c00354`, PMID 40720985. Full text retrieved via the NCBI BioC REST endpoint | **Better than planned.** The delta's [29] column rests on the paper itself, not on the abstract plus the repo. Decision 9's "D19's [29] half is directly resolvable" is confirmed and is now *fully* resolvable. |
| 3 | `github.com/icai-uma/IsalChem` is the implementation cross-check (decision 9) | Live, last push **2025-12-17**, 33 KB, **2,499 lines** of Python in 6 modules + a notebook. Shallow-cloned | Usable as corroboration. Governance rule frozen in §3. |
| 4 | Delta table lands "§1.x or `methodology.tex`" ([manuscript](../plan/manuscript.md) §2) | **Superseded by [prose](../plan/prose.md) §2**: new top-level **§2 "Related work and positioning"**, subsection **§2.3 "Relation to the authors' prior work"**, ~0.8 p = **Tab. 3 (0.7 p)** + the paragraph. Tab. 3 is on the **never-cut** list | Settled. No question needed. `manuscript.md` §2's "§1.x or methodology.tex" is stale → fix at close. |
| 5 | Table priced at **0.75 p** (`manuscript.md` §2) | `prose.md` §10 prices **Tab. 3 at 0.7 p**, §2.3 at 0.8 p total | Consistent within rounding. Budget: **0.7 p table + 0.1 p paragraph**. |
| 6 | — | `main.tex:6` is `\documentclass[review,times,number]{elsarticle}` — **single column**, review geometry. Existing Table 1 uses 12.5 cm of `p{}` width | A **5-column** table fits without `table*`. This is what made option 1 in §2 affordable. |
| 7 | — | Working tree carries **uncommitted peer work** (T-13 figures, `plan/README.md`, untracked `plan/prose.md`) on a shared checkout | **Commit only T-07 paths, explicitly.** Never `git add -A`. Re-check `HEAD` before each commit ([memory: two sessions share one checkout]). |

**`prose.md` is untracked.** It is the authoritative brief for §2.3 and it is not yet in git. Flag at
close; do not commit it here — it is the peer session's file.

---

## 2. Approach, and what was rejected

**Three read-only extraction agents in one turn, then synthesis by the orchestrator.**

- **Track A** — [28] component inventory, from the recovered PDF. Model: Sonnet.
- **Track B** — [29] component inventory, from the PMC CC-BY full text. Model: Sonnet.
- **Track C** — IsalChem *source* cross-check, blind to the paper. Model: Sonnet.

Each writes exactly one scratchpad file and owns nothing else. The extraction contract is the same
12 components for A and B, so the columns are commensurable; C answers 8 questions targeted at the
four claims [corrections](../plan/corrections.md) §4 attributes to [29].

**A and B were told not to read the successor manuscript, and C was told to read neither paper.**
That is the point of the decomposition: if C's blind reading of the code and B's reading of the paper
agree on the CDLL and the normalisation, the "inherited" cells rest on two independent sources. If
they disagree, that disagreement is a finding.

The **IsalGraph column is the orchestrator's**, from `methodology.tex` and
`src/isalgraph/core/README.md`. It is the column most likely to be over-claimed and it is not
delegated.

### Rejected

| Alternative | Why it lost |
|---|---|
| **Read both papers in the main thread** | ~140 KB ≈ 35 K tokens, affordable, and fidelity is what this ticket needs. Rejected only because the *extraction contract* is worth more than the raw text: a fixed 12-row schema with verbatim quotes and line anchors is directly checkable, and I verify the load-bearing cells by grep anyway. |
| **Git worktrees (`parallel-agents` default)** | **Waived, and the waiver is the point.** No track writes to the repository — all three write to one scratchpad file each, and file ownership is already disjoint. A worktree isolates concurrent *repo* writes; there are none. It would also be actively harmful under the known trap that a worktree cannot import the built C++ extension, and Track C must not run code at all. The rest of `parallel-agents`' discipline — pinned base commit, mandatory numbered acceptance criteria, orchestrator verification against source — is kept in full. |
| **One agent reading both papers** | Cross-contamination. An agent that has read the successor's framing of [28] cannot then read [28] neutrally, and the whole value of the delta is that it is checkable *against the predecessors as published*. |
| **Trust R3's characterisation of [28]/[29] and skip the reading** | That is precisely D19, which the source audit left **UNVERIFIED** because nobody opened the PDF. R3 was right on 13 of 13 checkable claims, so the prior is favourable — but "the prior is favourable" is not a citation, and this table will be checked. |
| **Merge Tab. 3 into Tab. 2** (representation comparison) | Already asked and answered in `prose.md` §10, 2026-08-26: category error. Tab. 2's rows are representations; Tab. 3's rows are components attributed to a source. Not reopened. |

---

## 3. Frozen before reading the inventories

Committed **before** the extraction agents returned, so that no verdict is chosen to fit an outcome.
This is the ticket's only outcome-selecting rule and it is the whole of its integrity.

### 3.1 The attribution rule

For each component *X* of IsalGraph, against predecessor *P* ∈ {[28], [29]}:

- **Inherited from *P***: *P* has a component serving the same role, and IsalGraph's version is
  functionally equivalent up to renaming — on the domain the two share, *P*'s version could be reused
  without changing its interface or its behaviour.
- **Modified from *P***: *P* has a component serving the same role, but IsalGraph's differs in a way
  that changes its **behaviour**, its **domain of applicability**, or its **guarantees**.
  *The cell must name the difference.* A verdict of "modified" with an unnamed difference is invalid.
- **New**: no component of [28] or of [29] serves that role.

### 3.2 Tie-breaks — all resolve toward the conservative reading

1. **Generalisation is modification, never novelty.** A component present in a predecessor for a
   restricted domain (bounded valence, fixed node ordering, labelled atoms) and generalised here is
   **modified**. Over-claiming novelty is exactly what R3.1a is probing; the rule is deliberately
   asymmetric against us.
2. **A theorem is a component.** Stated without proof in a predecessor and proved here → **modified**.
   Neither stated nor proved in either predecessor → **new**.
3. **Both predecessors have it** → the cell names both; the verdict attributes to the one whose form
   IsalGraph's resembles, and says so.
4. **Paper governs, code corroborates.** Where the IsalChem code and the IsalChem paper disagree, the
   **paper** sets the cell, because that is what a reviewer can check. The disagreement is reported in
   the article notes; it is not silently averaged away.
5. **`ABSENT` / `NOT FOUND` prints as `—`.** Never upgraded to a claim, never filled by plausibility.
6. **Every non-`—` predecessor cell carries a line anchor** into `28.txt` / `29.txt`. A cell I cannot
   anchor does not go in the table.

### 3.3 Table shape — **PI decision, 2026-08-26**

**Five columns**: `Component | [28] | [29] | This work | Verdict`. R3.1a(i) asks us to *identify*
which components are inherited, modified or new; an explicit verdict column identifies it, a grouped
table makes the reviewer infer it. Affordable because of measured item 6.

### 3.4 The sequence-model row — **PI decision, 2026-08-26: DROPPED**

🔴 **This overrules [corrections](../plan/corrections.md) §4**, which instructs: *"Pre-empt the reading
inside the table itself. Write the row as a stated scope decision […] rather than leaving the reader
to notice the gap."* Tab. 3 is now **architectural only**. The concession is conceded **once**, in
**§6.3**, per [prose](../plan/prose.md) §2's red line.

**The risk this creates, stated plainly.** `corrections.md` §4's failure mode was *"the artifact
becomes evidence against us"*. Dropping the row does not create that risk — R3 already knows both
predecessors ran sequence models, because R3 is the one who said so — but it does mean **no T-07
artifact discharges the pre-emption.** If §6.3 does not carry it, the demand falls between two
tickets and nothing in the board catches it.

**Mitigation, mandatory at close**: T-07 hands T-14 the *measured* content of both predecessors'
sequence-model experiments (what model, what task, what data, real or synthetic) as a named handoff,
and `corrections.md` §4 is amended to say the pre-emption moved. This is an acceptance criterion, not
a courtesy.

### 3.5 The sufficiency paragraph — inherited red line

120–150 words, closing §2.3. **It stands on the theorem.** It is *not* a defence of R3.2's decline
([prose](../plan/prose.md) §2). It also absorbs AE.5's only unowned clause — "rationale" from R3's
preamble at `mail.txt:83` — at no marginal cost ([demands](../plan/demands.md) AE.5).

---

## 4. Acceptance criteria

| # | Criterion | Proof |
|---|---|---|
| **A1** | [28] read in full and inventoried against the 12-component schema; every non-`—` cell anchored | `inventory_28.md` present, 12 components, orchestrator greps ≥ 6 quotes back to `28.txt` and all hit |
| **A2** | [29] read in full and inventoried against the same schema | `inventory_29.md`, same check against `29.txt` |
| **A3** | IsalChem source cross-checked **blind**, and the four `corrections.md` §4 claims (CDLL, two-pointer VM, validity guarantee, shortest-then-lex normalisation) each given an explicit confirm / contradict / silent verdict | `inventory_29_code.md`; orchestrator greps ≥ 4 code snippets back to the clone and all hit |
| **A4** | **D19 resolved**, both halves, in the two directions: R3's claim *confirmed or refuted*, and *what the experiment actually was* recorded | Article notes carry both verdicts with verbatim quotes; `verified-discrepancies.md` D19 moves off **UNVERIFIED** |
| **A5** | **Tab. 3 built** as a compilable `elsarticle` fragment: 5 columns, every predecessor cell anchored or `—`, every "modified" verdict naming its difference, **no sequence-model row** | `tab3_prior_work_delta.tex`; orchestrator re-checks every cell against the inventories, cell by cell |
| **A6** | Tab. 3 fits **≤ 0.7 p** in review geometry | Measured, not estimated: compile a standalone `elsarticle[review]` harness and read the height |
| **A7** | **Sufficiency paragraph**, 120–150 words, standing on the theorem, no defence of the R3.2 decline, absorbing AE.5's "rationale" | `sufficiency_paragraph.tex`; word count printed; red line checked by re-reading against `prose.md` §2 |
| **A8** | **Every number and claim in both artifacts traceable** to `28.txt`, `29.txt`, the clone, or `methodology.tex` | A per-cell provenance table in the article notes; no unanchored assertion |
| **A9** | **T-14 handoff written** for the dropped sequence-model row, carrying the measured content of both predecessors' experiments | Named section in the article notes + a letter-fragment note |
| **A10** | Results archived | `/media/…/results/reports/T-07-prior-work-delta/` holds sources, inventories, artifacts, and a `REPORT.md` |
| **A11** | Plan defects found by this ticket are recorded for propagation | Dead `docs/references/` path (2 files), stale `manuscript.md` §2 placement, `corrections.md` §4 §3.4 amendment, untracked `prose.md` |
| **A12** | Suite unaffected | This ticket touches no code. `git diff --stat` over T-07's commits shows no file under `src/` or `tests/` |

## 5. Stop and ask

1. **An extraction agent reports that [28] or [29] already contains a completeness / isomorphism-
   invariance theorem.** That would make the paper's single "NEW" row **modified**, and it changes
   what the paper can claim. Escalate before writing a single cell — this is the one outcome that
   damages the contribution rather than the table.
2. **The IsalChem code contradicts the paper on the CDLL or on the normalisation.** Rule 3.2.4 says
   the paper governs, but a contradiction on a component we call *inherited* is a fact the PI must
   see, not a cell I quietly resolve.
3. **Tab. 3 exceeds 0.7 p** and cannot be cut without dropping a component. Pages are the binding
   constraint; the trade is the PI's.
4. **The sufficiency paragraph cannot be written in 150 words without leaning on the R3.2 decline.**
   That would mean the theorem does not in fact carry the argument, which is a finding about the
   contribution, not about the prose.
5. Two failed iteration rounds with any agent.

## 6. Out of scope — named, so it does not creep

- **T-08**'s §2.1 and §2.2, the bibliography, and Tab. 2. T-07 builds §2.3's two artifacts only.
- **T-14**'s §6.3 concession and the R3.2 response.
- **T-12**'s B6 softening of "no existing method satisfies all four" (R3.1b).
- **T-17**'s comparison table.
- Restoring `docs/references/` — recorded as a defect, fixed by whoever owns the repo hygiene ticket.
- Any manuscript edit. T-07 emits fragments; `review-answer` lands them.
