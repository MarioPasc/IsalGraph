---
name: review-close
description: |
  Close a finished revision ticket: write its documentation, propagate what it learned
  into the plan files the *next* ticket reads, strike its board entry, emit its article
  notes and response-letter fragment, and verify its results are where the analysis
  expects them. The closing counterpart to `review-ticket`, which drives a ticket to
  completion but stops at the documentation. Triggers on "close T-0x", "the ticket is
  done, write the docs", "cross out the ticket", "update the docs for T-0x", "write the
  closing docs", "mark the ticket complete", "review-close", "document what the ticket
  found", or any request to standardise how a finished ticket is written up.
---

# Closing a revision ticket

The work is done. What remains is not a summary — it is **propagation**.

## The one rule

> **A finding that contradicts a plan file must be written INTO that file, not only into
> the ticket log.** The log is a record for whoever audits the ticket. The plan files are
> the instruction set for whoever runs the *next* one. A correction that lives only in the
> log is a correction that the next agent will not read, and they will inherit the error
> you just paid to discover.

Everything below follows from that. If you do nothing else, do §3.

Work through the phases in order; each produces a named artifact. Skip a phase only when
the ticket genuinely has nothing for it, and say so rather than leaving a silent gap.

---

## Phase 0 — Inventory what actually changed

Before writing anything, list four things. Most tickets have entries in all four.

| Category | Question | Where it lands |
|---|---|---|
| **Results** | What numbers did the ticket produce? | board entry, plan file, article notes |
| **Corrections** | What did the ticket prove *wrong* — in the plan, in the code, in the submitted study? | the file carrying the wrong claim (§3), plus §4 |
| **Findings** | What is true now that nobody had written down? | article notes (§5) |
| **Debts** | What did the ticket leave unfinished, and who owns it? | board entry, ticket log |

**Separate measured from inherited.** For every number you are about to print, know which
it is: *measured by this ticket*, *inherited from a plan file*, or *predicted and never
checked*. This distinction is the whole content of §6 and it is where tickets go wrong.

---

## Phase 1 — The board entry

Strike the row. Do not merely flip a status word — a struck row reads as closed at a glance
and preserves the original scope beside the outcome.

```markdown
| ~~**T-0x**~~ | ~~Original one-line scope~~ → **DONE <date>.** <the headline result, with
numbers>. <Findings carried, named>. | <deps> | **done** | — | [log](path), [plan](path) |
```

Two things belong in that cell and are routinely omitted:

- **The headline number**, not an adjective. "3,897,911 pairs, 98.43 % certified exact,
  ≈ 2,081 core-h" — not "completed successfully".
- **Findings that other tickets must act on**, named in the cell itself. Someone scanning
  the board for what to do next reads only this row.

**If the ticket invalidated a shared premise, add a warning to the board header too**, not
just the row. A row can be skimmed past; a header block above the table cannot. Name the
tickets that inherit the error.

---

## Phase 2 — Update the plan file the ticket owned

Every ticket owns one or more plan files. Append a **RESULT** section — do not rewrite the
plan's design sections, because the plan's *reasoning* is a historical record of what was
decided before the run, and rewriting it destroys the ability to check whether the design
was followed or quietly adjusted afterwards.

The RESULT section carries: the outcome table, whether pre-declared rules fired as written
(and which branch they took), the artifact paths, and a pointer to the article notes.

**Close any standing request the plan made of the ticket.** Plan files often contain
instructions like "record X's provenance when T-0y reproduces the run". Search for them and
answer them explicitly — an unanswered standing request is invisible debt.

---

## Phase 3 — Propagate corrections to every file that carries the wrong claim

This is the phase that matters and the one that gets skipped.

```bash
# find every plan file asserting the thing you disproved
grep -rn "<the wrong claim, or a distinctive phrase from it>" .claude/notes/review/plan/
```

For each hit, edit that file **in place**, under two rules.

### Strike, don't delete

Leave the wrong text visible with a strikethrough and put the correction immediately above
it. Deleting it means a reader who remembers the old claim finds nothing and assumes they
misremembered; striking it tells them the claim existed, was checked, and failed.

```markdown
> ## ⚠ CORRECTED <date> — the premise below is wrong; the decision survives
>
> <what was asserted> ... **It is not.** <the measurement that refutes it, as a table>
>
> **What this changes**: <consequences, and which tickets inherit them>
> **What survives**: <the parts of the decision that do not depend on the wrong premise>

~~<the original wrong sentence, struck through>~~
```

### Separate the premise from the decision

A decision usually rests on several justifications. When one premise falls, say explicitly
which parts of the decision fall with it and which do not. A correction that reads as "the
whole decision is void" when only its rhetorical framing was wrong causes more rework than
the error did.

> Worked example — T-03 disproved "GraphEdX charges zero for node operations", a premise
> stated in two plan files and used as decision D6's lead justification. The correction
> struck the premise in both files **and stated that D6 itself survives**, because D6's
> real argument is that zero node cost makes GED a pseudometric *in general* — an argument
> about cost models, not about what one dataset shipped. Only the framing "the submission
> mixes two models" became unprintable.

---

## Phase 4 — Retractions, if the ticket contradicted itself

A ticket that ran for hours may have reported a finding early and disproved it late. Say so
in the ticket's own design/decision log, in this shape:

1. **Retain the wrong entry**, marked `⚠ RETRACTED by <n>`. Never edit it into correctness —
   the record of what was believed, and on what evidence, is what makes the correction
   auditable.
2. **Add the retracting entry** with: the measurement, what it retracts, **why the original
   evidence looked convincing**, and the consequences to carry.
3. **Mark every downstream artifact that quoted the retracted claim** as superseded, with
   the corrected numbers inline. Do not silently rewrite a fragment someone may already have
   copied.

### The inherited-premise trap — name it when it happens

The most expensive failure mode in this workflow, because it produces *confirming* evidence:

```
the plan asserts X
  → the ticket configures a gate from X
  → the gate returns a clean, one-sided result
  → the ticket concludes Y about the data
  → the ticket "independently verifies" Y with a second script that also assumes X
  → both are wrong, because X was wrong
```

**A verification that reuses the original assumption is not a verification.** When a result
is one-sided and clean, that is *weaker* evidence than a messy one, because a systematic
offset is exactly what a wrong constant produces.

There is usually a tell in the data that contradicts the premise directly — in the case
above, 77,739 pairs recorded as *equal* whose graphs differed in node count, which is
impossible if one side charges for node operations and the other does not. **Before
concluding something about a dataset, test the premise against the dataset itself**, not
against a second computation that shares the premise.

---

## Phase 5 — Article notes

`.claude/notes/review/tasks/T-0x-article-notes.md`. This is the ticket's real scientific
output and the thing T-20 and T-14 will actually read.

- **Order by consequence**, not chronology. Items that change what the paper may claim go
  first; reporting obligations after.
- **Every item names its owner** (`T-20` manuscript, `T-14` letter, `T-21` reproducibility)
  and *where in the manuscript it lands*.
- **Every number carries its provenance** and its units of measurement.
- **Include the parameters needed to reproduce**: solver and version, cost model, timeouts,
  hardware, total compute. A number whose timeout is unstated is not reportable.
- **End with a "what is NOT claimable" section.** List, explicitly, the things someone might
  reach for and must not: retracted findings, quantities the ticket measured a *mechanism*
  for but never an *incidence* of, and ceilings that looked movable but were not. This
  section prevents more damage than the rest of the file.

**Flag anything that is a property of the measurement setup rather than of the object.** If
a rate depends on hardware, timeout, or sample, that dependence must travel with the number
in the same sentence — otherwise a reviewer fails to reproduce it and is right to.

---

## Phase 6 — The response-letter fragment

Every ticket emits one; an empty fragment is a visible hole in the demand index.

If the ticket already drafted a fragment and later contradicted it, **do not edit it into
correctness**. Append a superseding section: what is retracted, what the rewritten version
should say, and the corrected numbers. Someone may have already lifted the old text.

Keep the fragment's provenance table current — one row per claim, naming the artifact that
produced it. A claim with no source row is a claim that will be cut under page pressure,
because nobody will be able to defend it.

---

## Phase 7 — Results in their canonical places

- **Verify, do not assume.** List the files in every location the ticket was supposed to
  write, and check counts and shapes rather than existence.
- **Delete stale near-duplicates.** A smoke-test output sitting beside the production result
  with a similar name and a slightly different size is a trap that will eventually be read
  by mistake.
- **Keep large binaries out of git**, and gitignore the directory so the next run cannot
  commit them by accident.
- **Leave the canonical locations named** in the plan file and the log, so nobody has to
  search for them.

---

## Phase 8 — Verify, then commit

```bash
git status --porcelain          # clean when you are done
<test command>                  # unchanged or better than the ticket's baseline
grep -rn "<retracted claim>" .claude/notes/review/plan/   # zero live hits
```

Commit the documentation as its own change, with a body that says **what was corrected and
why**, not just what was written. The commit message is where a future reader looks first
when a number surprises them.

---

## Definition of done

1. Board row struck, with the headline number and the findings other tickets must act on.
2. The ticket's plan file carries a RESULT section, and every standing request it made of
   this ticket is answered.
3. **Every plan file asserting a claim this ticket disproved has been corrected in place**,
   with the wrong text struck rather than deleted, and the surviving parts of the affected
   decision named.
4. Retractions recorded with their evidence; superseded artifacts marked, not rewritten.
5. Article notes written, ordered by consequence, with owners, provenance, reproduction
   parameters, and a "not claimable" section.
6. Response-letter fragment emitted or explicitly superseded.
7. Results verified in every canonical location; stale duplicates removed.
8. Tests green, tree clean, no live hits on a retracted claim.

## What "done" does not mean

Not that the ticket's findings are correct — only that they are **traceable**. A reader must
be able to take any number in any artifact and find the measurement that produced it, the
premise it rests on, and whether that premise was measured or inherited. A ticket closed
that way survives being wrong. One closed without it does not.

Templates for each artifact: `references/templates.md`.
