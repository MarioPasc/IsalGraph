# Closing templates

Copy and fill. Every `<…>` is mandatory; delete a section only when the ticket genuinely has
nothing for it, and say so where it would have been.

The worked examples are from **T-03** (exact GED on Picasso, closed 2026-08-13), the first
ticket closed under this protocol.

---

## 1. Board row — `plan/tickets.md`

```markdown
| ~~**T-0x**~~ | ~~<original one-line scope>~~ → **DONE <YYYY-MM-DD>.** <headline result with
numbers>. <N> findings carried: <finding 1, one clause>, <finding 2>, <finding 3> | <deps> |
**done** | — | [log](<path>), [plan](<path>) |
```

**Worked example**

```markdown
| ~~**T-03**~~ | ~~Exact GED on Picasso~~ → **DONE 2026-08-13.** All five Suite-1 datasets:
**3,897,911 pairs, 98.43 % certified exact, 1.57 % interval-censored, ≈ 2,081 core-h.** Both
stages ran and **agree on their 22,051-pair overlap**. Three findings carried: the **exact
solver changed**, **GraphEdX uses UNIT node costs, not zero**, and **censoring is
hardware-dependent** | T-01 | **done** | — | [log](…), [exact_ged](…) §7 |
```

### Board header warning — only when a shared premise fell

Place above the table, not in the row.

```markdown
> ⚠ **T-0x invalidated a premise that <T-0a>, <T-0b> and <T-0c> all read.** <the wrong claim>
> is asserted in <file> §<n> and <file> <decision id>. Measured: <the refuting result>.
> Anything derived from "<the exact phrase to stop using>" needs re-checking before it is
> printed. <What survives> is unaffected. Full record: <path> amendment <n>.
```

---

## 2. RESULT section — appended to the ticket's plan file

```markdown
---

## <n>. RESULT — T-0x closed <YYYY-MM-DD>

<one line: scale, cost, failures>

| <unit> | <col> | <col> | <col> |
|---|---:|---:|---:|
| … | | | |
| **Total** | | | |

<Structural checks that passed.> **<Whether the locked counts reproduced.>**
<Where cost differed from the estimate, and why — attribute it, do not just report it.>

**<Pre-declared rule name>**: <which branch it took and on what basis>. <If a supersession or
fallback rule existed, state which side it landed on and when the decision was made.>

**Artifacts**: <canonical path>, mirrored at <path> and <path>.

### <N> findings the run produced, all of which change something

1. **<finding>** — <one-sentence evidence>. **<what decision or file it changes>**.
2. …

### For the manuscript

`<path to article notes>` collects what belongs in the paper: <one clause per major item>.
```

---

## 3. In-place correction — in the file carrying the wrong claim

```markdown
> ## ⚠ CORRECTED <YYYY-MM-DD> — <one line: what is wrong, what survives>
>
> This section previously said <the claim>. **That is wrong**, and <T-0x> measured it by
> <the method — one clause>:
>
> | <case> | <expected under wrong premise> | <expected under right premise> | observed |
> |---|---:|---:|---:|
> | … | | | |
>
> **<summary of the score>**, and <the systematic pattern that explains the old result>.
>
> **What the old text cost.** <what was configured from it, what was concluded, and that the
> conclusion is retracted.>
>
> **What survives.** <the parts of the decision independent of the premise, and why.>

~~<the original wrong sentence, struck through and left in place>~~
```

**Do not delete the struck line.** A reader who remembers the old claim and finds nothing
concludes they misremembered.

---

## 4. Retraction entry — in the ticket's design/decision log

Insert **above** the entry it retracts; leave that entry in place, marked.

```markdown
- **<date>, amendment <n+1> — ⚠ AMENDMENT <n> IS RETRACTED. <the corrected claim>.**

  <How it surfaced — ideally "found while doing X, by testing the premise instead of assuming
  it".>

  | <case> | <Δ> | <observed> | <model A> | <model B> |
  |---|---:|---:|---:|---:|
  | … | | | | |

  **<the score>**, and <the exact systematic relationship>.

  **<Where the wrong premise is asserted>** — <file> §<n> and <file> <id> — **and every
  conclusion drawn through it is void**, including <your own supposedly independent check, if
  it shared the premise>.

  **What amendment <n> got wrong.** <the mechanism: why the wrong premise produced convincing
  evidence.>

  **What survives, measured like-for-like**: <the corrected comparison and its numbers>.

  **Consequences to carry:**
  1. <consequence> — <owner>
  2. …

  **Amendment <n> below is superseded and retained only as a record of the error.**

- **<date>, amendment <n> — <original title>.** <original text> **⚠ RETRACTED by amendment
  <n+1> — see above.**
```

---

## 5. Article notes — `tasks/T-0x-article-notes.md`

```markdown
# T-0x — what belongs in the article about <topic>

**Written <date> on T-0x's completion.** Everything here is measured, and each item names
where in the manuscript or the letter it lands. Owners: **T-20** (manuscript), **T-14**
(letter), **T-21** (reproducibility)<, others>.

Sorted by consequence. The first <n> change what the paper can claim; the rest are reporting
obligations a reviewer would otherwise raise.

---

## 1. <The item that most changes what is claimable>

<table or measurement>

**Where it goes**: <section of the manuscript, and what must appear beside it>.

**Why it cannot be omitted.** <the bias or failure it prevents — argue it, do not assert it>.

## 2. <…>

…

## <n>. Method and parameters to report (T-21, reproducibility)

- **Solver**: <name, version, and what it is NOT — name the rejected alternative>
- **Cost model**: <exact vector, and the decision id>
- **Budget / timeout**: <value>, <its provenance>. **State it wherever a censored or
  timed-out rate appears.**
- **Hardware**: <CPU model, clock, cores per task, processes per core>
- **Compute**: <core-hours> for <N> units, of which <dominant part> is <share>
- **Determinism**: <what makes the result reproducible independent of scheduling>

## <n+1>. What is *not* claimable from T-0x

Stated so nobody reaches for it later:

- **Not** <a retracted finding> — retracted, see §<n>.
- **Not** <a quantity whose mechanism was shown but whose incidence was never measured>.
- **Not** <a limit that looked movable but was not, and why>.
- **Not** <an analysis whose data exists but whose computation belongs to another ticket>.
```

### The provenance discipline

Every number in the notes is one of three kinds, and the kind must be recoverable:

| Kind | How to write it |
|---|---|
| **Measured by this ticket** | give the artifact that produced it |
| **Inherited from a plan file** | cite the file and section — and check it before relying on it |
| **Predicted, not measured** | mark it and say so; never print it as a result |

### Properties of the setup, not the object

If a number depends on the timeout, the hardware, or the sample, **that dependence travels
in the same sentence**. Worked example: the same LINUX cohort censored 0.13 % on a
workstation and 1.17 % on a cluster node under an identical 60 s budget — a 9× swing from
core speed alone. A censoring rate is a property of *(cohort, timeout, machine)*.

---

## 6. Superseding a response-letter fragment

Append; never edit the original into correctness.

```markdown
---

## ⚠ <date> — THIS FRAGMENT IS SUPERSEDED AND MUST BE REWRITTEN

The paragraph beginning *"<first words>"* asserts a claim that has since been **retracted**.
See `<path>` amendment <n>.

<the correction in two sentences, with the measurement>

**What the rewritten fragment should say instead**, measured over <scope>:

- <corrected claim with numbers>
- <corrected claim with numbers>

**<What still stands>**, on the grounds that were always the real ones and are untouched:
<ground 1 with numbers>, and <ground 2>.

Do not reuse the retracted paragraph. <Where the corrected version already exists.>
```

---

## 7. Closing commit message

```
docs(T-0x): close the ticket, <the correction in five words>, note the article items

Marks T-0x done on the board with its results, and carries its <n> findings
into the plan files that other tickets read.

The important one is a correction: <file> and <file> both assert <claim>,
and T-0x measured the opposite -- <the score>. <What was configured from
the wrong premise, and what that invalidated.> Both files now carry the
correction inline, since <tickets> read them and would otherwise inherit
the error. <What survives> is untouched and says so.

Adds T-0x-article-notes.md: the <n> items that belong in the manuscript,
ordered by consequence, each with its owner and the numbers already
measured -- including what is explicitly NOT claimable.
```

Say **what was corrected and why**, not only what was written. This message is where a
future reader looks first when a number surprises them.
