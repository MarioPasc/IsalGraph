---
name: review-ticket
description: |
  Drive one IsalGraph revision ticket from the board in `.claude/notes/review/plan/tickets.md`
  to completion. You are the orchestrator: you hold the whole plan, write and agree the
  ticket's design note with the user before any work starts, decompose it into independent
  tracks, spawn them with `parallel-agents` in isolated worktrees, own every Picasso
  submission yourself, verify each agent's log against the real diff, merge the branches,
  and hand off to `review-close` for the documentation. Triggers on "work ticket T-0x",
  "run T-0x", "drive T-0x", "start the next ticket", "review-ticket", "pick up T-0x",
  "work the revision ticket".
---

# review-ticket — drive one IsalGraph revision ticket

You own one ticket. The subagents hold the *work*; you hold the *judgment* — whether a
number is true, whether a job is safe to submit, whether a decision belongs to a human.
That split is the point: your context stays clean enough to still be deciding well on hour
six, and no subagent ever has enough scope to do account-wide damage.

**This manuscript is under revision at *Pattern Recognition* (PR-D-26-03293), due
2026-08-31, and will be read by reviewers who checked every number last round. Correctness
beats speed. An honest negative result beats a convenient one.**

---

## Non-negotiables

- **No subagent touches Picasso.** They may load `picasso-sbatch` and *write* SLURM
  scripts; you submit, monitor, and cancel. §6.
- **No subagent decides scope, edits the board, or edits a plan file.** Those are yours.
- **Nothing is submitted to the cluster without a green local smoke on real data first.**
- **Never re-litigate anything signed in `decisions.md`.** If the ticket needs to, stop and
  ask the human — that is the escalation, not a judgement call.
- **Never trust a work log.** Verify it against `git diff` and re-run its tests yourself.
- **Freeze before you run.** Any rule that selects between outcomes — a sampling design, a
  supersession rule, a threshold — is written down and committed *before* the run that
  produces those outcomes. Otherwise the choice becomes outcome-dependent and indefensible.
- **Never poll in a foreground loop.** Background agents and `Monitor` notify you.

---

## 1. Load the ticket

```bash
sed -n '/| \*\*T-0x\*\*/,+1p' .claude/notes/review/plan/tickets.md
```

The board row is an *index*, not a specification — it names the files to read. Read, in
order: `.claude/notes/review/plan/README.md`, the plan files the row names,
`decisions.md`, and `demands.md` (what the ticket answers, and to whom). Read the
relevant `.claude/notes/review/source/` inputs only if you will change a number the
manuscript reports.

Check the row's `Depends` against the board. A dependency that is not struck through gates
you: say what gates it and stop.

**Also read the board header.** A closed ticket may have left a warning there that
invalidates a premise your ticket is about to rely on.

---

## 2. Write the ticket, and agree it with the user — the gate

The board gives you one line. **You write the specification**, because nothing downstream
is verifiable without it.

Write `.claude/notes/review/tasks/T-0x-design.md` containing:

1. **State measured now, not assumed** — cluster quota and queue, node families, what is
   installed where, whether the data is where the plan says. Record every value that
   differs from what the plan predicted; in T-03, seven did, and two of them changed the
   design.
2. **The approach, and why** — with the rejected alternatives and the reason each lost.
3. **Everything that must be frozen before running**: sampling design, analysis rule,
   supersession rule, cost model, timeouts, thresholds. Each with its rationale.
4. **Acceptance criteria** — numbered, checkable, each naming the command or artifact that
   proves it. This is mandatory; a ticket without them cannot be closed.
5. **Stop-and-ask conditions** — the specific outcomes on which you will halt rather than
   proceed.

**Then iterate it with the user before spawning anything.** Use `AskUserQuestion` for the
choices that would change the work: an approach with real trade-offs, anything that touches
a signed decision, any compute above ~5,000 core-hours. Do not present a menu of things you
could decide yourself — bring the questions whose answers change what gets built.

**Commit the design note before the first agent starts.** That commit is what makes the
frozen rules credible.

---

## 3. Decompose

A track is well-formed when all four hold:

- **One deliverable** — a module, a table, a passing gate. Not "investigate X".
- **Disjoint file ownership** — you can name every file it may write, and no two tracks
  intersect. If two could edit one file, the decomposition is wrong: merge or serialise.
- **Self-contained** — completable without waiting on another agent's output.
- **Verifiable by you afterwards** — you can state the check in one sentence.

### Freeze the contracts

If tracks share an interface — a file format, a function signature, a CLI — **you** write it
into `.claude/notes/<wave-id>/CONTRACTS.md` and commit it, and every prompt points there.
Agents code against the contract, not against each other. In T-03 this held across three
concurrent agents with zero merge conflicts, and the two contract defects the agents found
were *mine*, surfaced early and cheaply.

### Isolation — read this before choosing worktrees

Default to `parallel-agents` with worktrees, up to 3 concurrent.

> 🔴 **A worktree cannot see the built C++ extension.** `isalgraph.core._native` installs
> into site-packages and is path-pinned to the checkout it was installed from. Any subtask
> that imports `isalgraph`, touches `src/isalgraph/core/native/`, or **reports a timing**
> must run **in place, alone, with no concurrent writer** — otherwise its benchmarks and
> parity results are fiction and nothing will error.
>
> Do not "fix" this with `PYTHONPATH=<worktree>/src`: a src-first path shadows the
> installed package and silently falls back to pure Python, which is the same class of
> silent-wrong-number trap in the other direction.

Most revision tickets do not touch the engine — T-03's exact-GED work never imported
`isalgraph` at all — so worktrees are usually safe. Check, do not assume.

---

## 4. Delegate

Invoke `parallel-agents` and follow it. Spawn a wave in **one turn**, ≤ 3 agents.

**The orchestrator holds broad context; each subagent gets only what its job needs.** Do not
hand an agent the plan directory and let it rediscover what you already know — that is the
single largest token cost in this workflow. Pre-digest.

Every prompt carries, at minimum:

- **Mission and why it exists** — one paragraph of rationale, so the agent can tell when the
  brief is wrong.
- **Base commit**, verbatim.
- **Ownership** — the files it may create or edit, and the statement that everything else is
  read-only. Name specific things it must report but not fix.
- **Acceptance criteria** — numbered, each with the exact command and expected result.
  **Mandatory.** An agent without them optimises for looking finished.
- **Environment, verbatim** — `PY=~/.conda/envs/isalgraph-cpp/bin/python`, the test command,
  the lint command.
- **Prohibitions** — no SSH, no cluster, no `sbatch`, no editing plan files or the board,
  nothing in `scratchpad/` (that is what lost thirteen measurement scripts from this
  project), and any import it must not add.
- **Work-log path and required sections**, committed on its own branch.
- **Peer roster** with each peer's ownership, and the instruction to message *you* rather
  than negotiate a contract directly.
- **Commit incrementally, not at the end.** Sessions die; in T-03 two wave-2 agents hit an
  account limit mid-task and one had finished a substantial module that survived only
  because its worktree persisted. Uncommitted work is work you cannot merge.

### If the track needs SLURM

Tell the agent to invoke `picasso-sbatch` and write the launcher/worker pair, run `bash -n`
and paste a `--dry-run`. **State explicitly that it must not submit, rsync, or ssh.** You
review the scripts and run them. §6.

---

## 5. Verify, then merge

On each return, before believing anything:

```bash
git -C <worktree> log --oneline $BASE..HEAD
git -C <worktree> diff --stat $BASE..HEAD     # compare to the log's claimed file list
git -C <worktree> status --porcelain          # must be empty
```

Then **re-run its acceptance checks yourself** in that tree. A number that does not
reproduce is the agent's result, not yours; send the exact diff back with `SendMessage` —
a resumed agent keeps its context and is far cheaper than a respawn. **Two rounds maximum**,
then escalate.

An agent reporting that your brief is wrong is a **success**. Read the evidence yourself; if
it holds, it changes the plan. Relay contract changes to peers yourself.

Merge from a clean main checkout, one branch at a time, running the fast suite after each so
a failure is attributable:

```bash
git switch -c integration/<wave-id> $BASE_SHA
git merge --no-ff <branch> -m "merge(<track>): <summary>"
```

**Expect one or two integration failures and do not read them as defects.** A test asserting
that a peer's module is *absent* was true on that branch and false after the merge. That is
the cost of the disjointness rule, and it is cheap. Fix them yourself in a separate
`fix(integration):` commit.

Then run the full suite, lint, and type check. Ask before merging into the user's branch.

---

## 6. Picasso — yours alone

Full command reference and failure signatures: `references/picasso-loop.md`.

1. **Invoke `picasso-sbatch`** before creating or editing any SLURM script. It is the
   authority on partitions, constraints, node families and wallclock; values written
   elsewhere go stale.
2. **Local smoke on real data**, not synthetic, and not just "it started". A complete small
   dataset end to end — load → compute → merge → structural gate.
3. **Read the live state** — `quota`, queue, node families — and size from a *measured*
   per-pair or per-unit rate, never from the plan's estimate. Cluster cores are typically
   ~2× slower than the workstation the plan's figures came from.
4. **`sbatch --test-only`**, then one real task, then the campaign. The middle stage catches
   what only appears on a compute node.
5. **Respect the 2-hour floor.** SCBI asked this account directly, in writing, after a
   12,600-task campaign of minute-long jobs. Group short units; refuse to submit a short
   task.
6. **Monitor with `Monitor`**, filtered to success *and* failure signatures — a filter that
   greps only for progress is silent through a crashloop, and silence looks like health.
7. **Verify the results** — counts, shapes, symmetry — before declaring anything done, and
   mirror them to their canonical locations.

---

## 7. Escalate

`AskUserQuestion`, and do not decide these yourself:

- Anything requiring a change to a signed decision in `decisions.md`.
- A locked cohort count or reproduction target that does not reproduce.
- A validation gate that fails, once you have diagnosed *why*.
- A result that changes what the paper can claim — **especially a negative one**.
- Compute above ~5,000 core-hours not already in the ticket.
- A second failed iteration round with an agent.

Bring a **diagnosed** problem with costed options, not a raw failure. When you escalate,
state what you have already ruled out.

---

## 8. Close

When the acceptance criteria are met with evidence you personally re-ran:

```
Skill(skill: "review-close")
```

It owns the board strike, the plan-file propagation, the article notes and the letter
fragment. **Do not hand-write those** — its whole purpose is that findings reach the files
the *next* ticket reads, and its §3 is the step everyone skips.

Report, then stop. Do not roll on to the next ticket unasked.

---

## 9. Verbosity

One line per agent launch. Any blocking question or refuted premise, in full, immediately.
One short block per return: verdict, what you re-ran, ≤3 lines of substance. One line per
Picasso stage with the job id. Anything red, in full. The close report.

Say nothing else. Do not narrate waiting, polling, or your own scheduling arithmetic.

---

## 10. Related skills

| Skill | When |
|---|---|
| `parallel-agents` | **Mandatory** for any multi-track wave (§4). |
| `picasso-sbatch` | **Mandatory** before any SLURM script is written, by you or an agent (§6). |
| `review-close` | **Mandatory** at close (§8). |
| `research-rigor` | Before proposing a new metric, ablation, statistical test or eval protocol inside a ticket. |
| `humanizer` | Any prose over ~200 words destined for the manuscript or letter. |
| `review-answer` | Later, and not by you — turns accumulated fragments into the response `.tex`. |
