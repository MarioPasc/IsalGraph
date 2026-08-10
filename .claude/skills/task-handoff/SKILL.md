---
name: task-handoff
description: |
  Log an out-of-scope task to the IsalHG article task ledger
  (docs/article/DEVELOPMENT/) instead of solving it inline. Use it mid-development
  when you discover work that must be done but is NOT part of the task you are
  currently on. Lightweight: it writes one well-formed task file with a live
  timestamp and the context pointers the next agent needs, then you return to your
  current work. Triggers on "found something out of scope", "this needs its own
  task", "log a follow-up task", "hand off a task", "add to the task ledger",
  "out-of-scope but must be addressed", "park this for later", "note this as a
  separate task".
---

# task-handoff — park out-of-scope work as a ledger task

Preserve focus. When you hit work that must happen but is not your current task,
do not scope-creep and do not silently drop it. File it in the ledger and move on.

## When to use

- Mid-implementation you find a bug, missing primitive, refactor, or research
  question that is real but belongs to a *different* task.
- A `task-reader` run surfaces work outside the task it is executing.

Do **not** use it to record progress on your current task (that goes in the
task's own file), or to solve the discovered work now.

## Ledger layout

```
docs/article/DEVELOPMENT/
  README.md              hub — index, status legend, dependency graph
  DECISIONS.md           decisions pending PI
  <SCOPE>/
    README.md            one paragraph: what this scope is for
    OPEN/ IN-PROGRESS/ BLOCKED/ CLOSED/
      T-<id>.md          one task per file
```

Scopes are the milestone prefixes: `T-M0` (seed selection), `T-M1`
(`metric_space` foundation), `T-M2` (HGED), `T-M3` (competitors), `T-M4`
(corpora + scoring), `T-M5` (experiments), `T-M6` (reparent), `T-TA`
(completeness), `T-TB` (stability), `T-DQ` (data questions).

## Steps

1. **Locate the ledger:** `docs/article/DEVELOPMENT/`. If it does not exist, the
   project has no article ledger — say so and stop rather than inventing a
   location.
2. **Get a real timestamp:** run `date '+%Y-%m-%d %H:%M %Z'`. Never fabricate it.
3. **Pick the scope.** The task belongs to an existing scope directory whenever it
   extends one (`ls docs/article/DEVELOPMENT/`). Read that scope's `README.md`
   before deciding — if the task does not fit any scope's paragraph, that is a
   design discussion: say so rather than creating a new scope silently.
4. **Pick the next id:** list the scope's task files across all four status
   directories (`ls docs/article/DEVELOPMENT/<SCOPE>/*/`) and choose a fresh
   suffix (`T-M0c`, `T-TAh`, …). Ids containing `'` are filed with `prime`
   (`T-M4'` → `T-M4prime.md`).
5. **Write** `docs/article/DEVELOPMENT/<SCOPE>/OPEN/T-<id>.md` using the template
   below. Never edit, reorder, or renumber existing task files.
6. **Do not update the hub index by hand** unless the scope's open/closed counts
   in `README.md` become wrong — in which case fix only those cells.
7. **Return** to your original task. Mention in your reply that you parked the
   item as `T-<id>` and give its path.

## File template (match the ledger's existing style)

```markdown
# T-<id> — <concise title>
**Declared:** <output of the date command>
**Status:** OPEN
**Depends on:** T-<id> | —
**Delegation:** agent | orchestrator-only   (omit if you are unsure — the orchestrator judges)
**Why out of scope:** <one line — which task you were on and why this is separate>
**Context to read first:**
- `path/to/file.py::Function` — the symbol involved
- `docs/article/<doc>.md` §N ("title") — the spec/precedent
- `docs/article/DEVELOPMENT/<SCOPE>/<STATUS>/T-<id>.md` — a related task, if any
- `.claude/rules/coding_rules.md` — always
**Description:** <what & why, 1–3 sentences>
**Acceptance:** <the check that would mark it DONE>
**Out of scope here:** <what the eventual worker should NOT touch>
```

## Rules

- **One task per file, one file per task.** The `# T-<id>` H1 is the first line.
- **Context pointers are mandatory.** An entry with no `path::Function` /
  `docs/FOO.md §N` pointers is incomplete — the next agent must be able to start
  cold. Use the same pointer style as existing entries.
- **Keep it lightweight:** a handful of lines. Precision over prose.
- **One entry per discovery.** If you found three things, write three files.
- **Status is always `OPEN`** at handoff time, so the file lands in `<SCOPE>/OPEN/`.
- **`Delegation:`** marks work the `task-orchestrator` must not hand to a subagent:
  `orchestrator-only` for anything irreversible, definitional, or where deciding
  what "correct" means *is* the task (golden regeneration, default flips, freezing
  a definition). Omit the field when unsure; the orchestrator will judge and say so.
- **A task changes status by moving**, with `git mv`, not by being rewritten in
  place. Only its `Status` line and its closing-check output change.
