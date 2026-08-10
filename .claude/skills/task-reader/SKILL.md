---
name: task-reader
description: |
  Pick up and execute a task from the IsalHG article task ledger
  (docs/article/DEVELOPMENT/) under a strict, design-first protocol. Use it
  whenever you are asked to work a ledger task. It enters plan mode, reads the
  task file, its scope README, and every context source the task cites, reads the
  coding rules, reasons step by step, flags decisions instead of defaulting
  silently, asks the user what it needs, routes out-of-scope discoveries to the
  task-handoff skill, then follows Plan -> Test -> Implement -> Verify. Triggers on
  "work on task T-…", "start task T-…", "pick up the task", "address the task",
  "read the task and begin", "implement task T-…", "do the next ledger task",
  "task-reader".
---

# task-reader — execute a ledger task, context-first

You are about to work a declared task. Do not open source files and start
editing. Load the full context first, plan, then implement. Opus follows
instructions literally — read what the task tells you to read; do not fill gaps
from assumption.

## Ledger layout

```
docs/article/DEVELOPMENT/
  README.md              hub — index, status legend, dependency graph, critical path
  DECISIONS.md           decisions pending PI
  <SCOPE>/
    README.md            one paragraph: what this scope is for
    OPEN/ IN-PROGRESS/ BLOCKED/ CLOSED/
      T-<id>.md          one task per file
```

Resolve an id to its file with
`find docs/article/DEVELOPMENT -name 'T-<id>.md'` (ids containing `'` are filed
with `prime`, e.g. `T-M4'` → `T-M4prime.md`). The parent directory names give the
task's scope and status.

## Protocol

1. **Enter plan mode.** This is design-first work; no edits until a plan is
   approved. (If already in plan mode, continue.)
2. **Read the task file in full.** If the user named an id, resolve it as above;
   if not, read `docs/article/DEVELOPMENT/README.md`, take the highest-priority
   task from its **critical path**, and confirm with the user.
3. **Read the scope's `README.md`** (the sibling of the status directories) before
   the task's own context list. It states what the scope exists to achieve, which
   is what makes the task's `Out of scope here` boundary legible.
4. **Read every context source the task cites** — each `docs/article/*.md §N`,
   each `src/isalhg/...::Function`, each proof, paper, or sibling-repo path, and
   each related ledger task it points at. Delegate heavy reads (large files,
   papers) to subagents to keep the main context clean. Additionally **always**
   read:
   - `.claude/rules/coding_rules.md` — the project coding discipline;
   - `docs/article/DEVELOPMENT/DECISIONS.md` — a pending decision may be yours to
     surface, never to silently resolve;
   - the `docs/article/` scope docs the task touches (`PROPOSAL.md`,
     `CODE_DESIGN.md`, and the relevant `theoretical/` or `empirical/`
     breakdown — for geometry/measurement tasks that includes
     `theoretical/geometry.md`, the invariant→consumer spec).

   **Scope era check (v3, D-ART2 2026-07-18).** The article is
   *characterize → exploit*: geometry characterized, applications licensed by
   it, HGED entering only through the E1' discussion figure and Qin ladder
   budgets. Tasks or docs written before the rescope may carry v2 wording
   (correlation study with competitor rows, density sweep, MI, Theorem-B
   "capstone"). Where a task's prose and `PROPOSAL.md` v3 / D-ART2 disagree,
   the latter win — surface the conflict instead of following stale
   instructions.
5. **Think step by step and reason explicitly.** State tensor/shape/type flow,
   invariants, and complexity where relevant. Verify the task's premises against
   the code and the scope docs before building on them (scientific-challenge
   protocol) — if a premise is wrong, say so with the contradicting source.
6. **Flag every decision.** Parameter values, ABC/naming choices, refactor scope,
   distance definitions — surface them as explicit decisions with a
   recommendation, never a silent default. Hold them for the user if they change
   the outcome.
7. **Ask the user** any clarification you need, batched into a single
   `AskUserQuestion`, before finalizing the plan. Do not drip-feed questions.
8. **Route out-of-scope discoveries to `task-handoff`.** If you find work that
   must be done but is outside this task's `Out of scope here` boundary, invoke
   the `task-handoff` skill to file it as a new task, then continue the current
   task. Do not scope-creep.
9. **Plan -> Test -> Implement -> Verify** (`coding_rules.md` §2.3):
   - write the acceptance-criteria checklist and the failing test first;
   - present the plan and `ExitPlanMode` for approval;
   - implement against the task's `Acceptance` field;
   - run the `test-runner` agent (pytest + ruff + mypy); re-run
     `tests/property/` if canonical/seed/`core` code changed.
10. **Close the loop.** On success, set the task's `Status` to `DONE`, append the
    closing-check output to its file, and **`git mv` the file into
    `<SCOPE>/CLOSED/`**. If blocked, set `BLOCKED` and `git mv` it into
    `<SCOPE>/BLOCKED/`, noting what it waits on. Update the scope's open/closed
    counts in `docs/article/DEVELOPMENT/README.md`, and the dependency graph if the
    task changed it.

## Rules

- **Know which environment you are in.** The editable install is path-pinned to
  the main checkout, so `~/.conda/envs/isalhg` always imports the main tree's
  source. If you are working in a git worktree, you MUST clone the env
  (`conda create -y -n isalhg-<TASK> --clone isalhg`), `pip install -e ".[dev]"`
  inside your worktree, and invoke only `~/.conda/envs/isalhg-<TASK>/bin/python`.
  Otherwise your tests silently exercise code you did not write. If you edit
  `core/_native/`, reinstall before testing — a stale `.so` against new bindings
  produces phantom failures.
- Respect the task's `Depends on` — do not start a task whose dependency is not
  `DONE` (i.e. not filed under `CLOSED/`) without flagging it.
- Respect the task's `Delegation:` field if present. `orchestrator-only` means a
  human or the `task-orchestrator` runs it, not a subagent.
- Respect the task's `Out of scope here` — that boundary is deliberate.
- A task file is append-only apart from its `Status` line: never rewrite an
  existing entry's description or acceptance to match what you built.
- Prefer the smaller change; reuse existing functions/utilities named in the
  context sources before writing new code.
- Report outcomes faithfully: failing tests, skipped steps, and open decisions
  stated plainly.
