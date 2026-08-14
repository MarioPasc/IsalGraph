# T-04 — prompt for the execution session

Paste the block below into a **fresh** Claude Code session at
`/home/mpascual/research/code/IsalGraph`. Run `/effort xhigh` first.

---

```
/review-ticket T-04

The design note is already written, agreed with me, and is authoritative:
.claude/notes/review/tasks/T-04-design.md

Read it in full before anything else, then read
.claude/notes/review/plan/competitors/README.md (its §5 lists fifteen findings
with owners) and .claude/notes/review/plan/competitors.md. Do NOT re-derive the
design and do NOT re-litigate the three decisions I already signed at design
time; they are recorded in the note's §7, §6 and §3.1 and in its changelog:

  1. The orchestrator owns Picasso. Subagents never ssh, rsync or sbatch. The
     smoke runs in ONE interactive loginexa session (30-min wallclock, no queue,
     so SCBI's 2-hour job floor does not apply), and each agent receives its own
     JSON slice back by SendMessage to close its Picasso criterion.
  2. Three agents in one wave, grouped by shared trap, not seven one-per-file.
  3. size_null is a registered backend AND is hard-excluded from the frozen
     confirmatory family (decision 23, N_max = 197).

Base commit: 152c80d18293d6e699bd36cf301a88a7596c6464 (tree clean).
Wave id: 2026-08-14-t04-competitors

Skip §2 of the review-ticket skill (write and agree the design) — it is done.
Start at §2's last line: commit the design note, then go to §3.

Execute in three waves.

WAVE 0 — you, alone, no agents. Own everything shared:
  - src/isalgraph/competitors/{__init__,base,registry,bits,fixtures,smoke,grid}.py
  - src/isalgraph/competitors/metrics/**
  - src/isalgraph/competitors/backends/{isalgraph_ref,size_null}.py
  - the errors.py additions and the pyproject.toml `competitors` extra
  - .claude/notes/2026-08-14-t04-competitors/CONTRACTS.md
Also: install pynauty==2.8.8.1 into isalgraph-cpp (it is currently only in the
isalhg env — design note §0), build it on Picasso under `module load gcc/12.2.0`,
and re-verify the grakel off-by-one under 0.1.10 (the folder recorded 0.1.8).
Commit CONTRACTS.md before any agent starts.

WAVE 1 — invoke the parallel-agents skill and spawn all three in ONE turn, in
isolated worktrees off the base commit. The agent definitions carry the frozen
conventions, the traps and the acceptance criteria; your spawn prompts add only
the base commit, the wave id, the CONTRACTS.md path and the peer roster.

  competitor-serial     adjacency, graph6, sparse6
  competitor-canonical  nauty->graph6, sparse6-nauty, AGM CAM
  competitor-mining     gSpan min-DFS, WL subtree

Worktrees are safe for this ticket: no track touches core/native/, and the only
component needing the C++ engine — the reference arm — is yours and is timed in
place, alone, in wave 2.

WAVE 2 — you: merge one branch at a time from a clean checkout with the fast
suite after each, then the cross-backend gates that need every backend present
(design note §8 criteria 1, 3-6), the loginexa session, the reference arm's
in-place timings, and the T-04a handoff. Then review-close.

Two things I care about most, in order:

  (a) Acceptance criterion 1, the reproduction gate. The competitors/ folder is
      the evidence base for five plan-level findings that reach a printed number,
      and §0 of the design note shows it was measured in a DIFFERENT environment
      than the shipped code will run in - different grakel, a pynauty that
      isalgraph-cpp does not have. Until src/ reproduces those numbers the plan
      is resting on unreproduced measurements. A mismatch is not a tolerance
      question; it stops the ticket and comes to me.

  (b) F5-blindness being structural rather than procedural (design note §4.5).
      Decision 24's whole defence is that T-04a's exclusion rule could not have
      seen the outcome. grid.py must have no import path to a GED loader, and a
      test must assert it.

Verify every agent's log against the real diff and re-run its acceptance checks
yourself. Escalate to me on any of the seven stop-and-ask conditions in §9.
```
