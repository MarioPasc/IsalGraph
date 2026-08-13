# Wave summary — `2026-08-13-t27-bakeoff`

**Date**: 2026-08-13 · **Ticket**: T-27, GED bound bake-off · **Base**: `4a31817`
**Integration branch**: `integration/2026-08-13-t27-bakeoff` · **Target**: `main`

## The user's prompt, verbatim

> Load /review-ticket for T-27 in .claude/notes/review/plan/tickets.md ; Iterate for as long as you
> need and write the report on the topic in
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/{report_name} with the
> embedded plots and data saved there. Think step by step, reason.

## Agents

| Track | Model | Isolation | Branch | Head | Verdict | Log |
|---|---|---|---|---|---|---|
| **A — harness** | Opus, xhigh | worktree | `worktree-agent-a06572452d2b53f38` | `8e6289a` | **ACCEPT** | [harness.md](harness.md) |
| **B — analysis + figures** | Opus, xhigh | worktree | `worktree-agent-a208cc07b0c6a4c8f` | `c6cc6ac` | **FIXUP** → ACCEPT | [analysis.md](analysis.md) |
| **C — literature** | Opus, high | worktree | `worktree-agent-acea836daa7d77780` | `e76ea7d` | **ACCEPT** | [literature.md](literature.md) |

Spawned in one turn, ≤ 3 concurrent, disjoint file ownership, zero merge conflicts.

## Interventions

| # | What | Outcome |
|---|---|---|
| 1 | **Track A refuted the contract's zero-value rule.** I had written that a bound of exactly `0.0` where `exact > 0` means the wrong accessor and must raise | **My defect.** Verified: C₆ vs 2·C₃ has exact GED 4.0 and all four LB methods return 0.00, all valid — free node *and* edge substitution under D6 makes a degree-preserving assignment cost nothing. Replaced with a capability probe on a pair with differing degree sequences. Would have halted the run on legitimate data |
| 2 | **Track A resolved HED**, which spec §3 had written off | Lower-bound-only by design (`hed.ipp`); default vacuous under D6; `--edge-set-distances OPTIMAL` yields a real bound. Verified on four hand-built pairs. Grid grew by 5 cells. The *Pattern Recognition*-venue citation now carries a measurement |
| 3 | **My `--seed 42`** in the CONTRACTS §6 example does not exist as a GEDLIB token | Corrected; GEDLIB raises on unknown options rather than ignoring them |
| 4 | **Track C found `BRANCH ≡ BRANCH_FAST` under constant edge costs** (survey §5.2.4) | Verified: 3,916/3,916 LINUX pairs exactly equal. Promoted to a harness gate. **Decision 11's choice of BRANCH_FAST is upgraded from 400 LINUX pairs to a theorem plus a census verification** |
| 5 | I added a **second pinned configuration** for local-search UBs (`_MS` / `_DET`) | GEDLIB's `LSBasedMethod` defaults to one random start under `REAL` randomness — the cause of IPFP returning 3.00 on P₄/C₄. Grid → 12 cells × 5 datasets = **60**. `_MS` enters selection, `_DET` is a self-checking companion. Frozen before any tightness result |
| 6 | **Track B could not write its work log** — a `PreToolUse` hook blocks subagents from writing report files under `.claude/notes/` | It tried twice and correctly refused to route around the guard. **I wrote the log myself** and committed it on its branch with the provenance stated in the file. **Decomposition defect — see below** |
| 7 | **I over-estimated the HED bootstrap cost by two orders of magnitude** | Track B refuted it by measuring the counterfactual instead of extrapolating: ~10 %, not 30×. The fix stayed on its own merits (removes a precondition rather than guessing a granularity) |
| 8 | **My `PYTHONPATH=$PWD/src` instruction to track B was wrong** | scikit-build-core registers a `MetaPathFinder`, which outranks `sys.path`, so a worktree's `src` stays invisible. Its `__path__` workaround is test-only and a no-op post-merge |

## Merge record

Merged A → B → C, fast suite after each. **Zero conflicts** — the ownership partition held.

Two integration commits, both my repairs:

- `ffcdf3e` — `write_index` hard-required GEDLIB solely to stamp `gedlib_commit` into metadata, so
  three of track A's tests failed after merge. The index carries ground truth and Levenshtein only,
  nothing GEDLIB computed, so provenance was making an artifact impossible to write. Now records
  `"unavailable"` and degrades. Cell files still fail loudly, because their values do come from GEDLIB.
- `a76e0b7` — track A hard-coded my wave instruction ("no more than 4 processes, two peer agents
  share this workstation") as a permanent CLI ceiling, which blocked the campaign once the agents
  finished. Now `os.cpu_count()`.

## Verification

| Check | Result |
|---|---|
| Base `4a31817`, `tests/unit/` | **726 passed** — exactly the CLAUDE.md reference state |
| Post-merge, without GEDLIB on the path | **872 passed / 44 skipped** |
| Post-merge, with GEDLIB | **915 passed / 1 skipped** |
| `ruff` on all owned files | clean |
| P1 gate (`BRANCH == BRANCH_FAST`) | **passes on all 5 datasets** |
| Dominance gate (`*_DET ≤ BIPARTITE`) | **passes on all 5 datasets** |
| §3.10 cross-check vs `ged_bounds.py` | BRANCH 400/400; BIPARTITE 156/400 value-equal but **passes** — BIPARTITE reports the induced cost of a non-unique argmin, and GEDLIB's node map attains our LSAP optimum 400/400 |
| M4 over 46,774,932 bound evaluations | **0 violations** at `TOL = 1e-9` |

**One false alarm, mine.** My ad-hoc M4 check compared floats exactly and reported 10,623
violations. All were ≤ **3.55e-15**. `BRANCH_TIGHT` is the only iterative, non-integer-combinatorial
cell in the grid and returns e.g. `5.000000000000001`; the harness's `TOL = 1e-9` is correct and
track A's M4 = 0 stands.

## What the decomposition got wrong

1. **The work-log invariant is unenforceable for subagents in this repo.** A `PreToolUse` hook
   blocks them from writing under `.claude/notes/`, so invariant 5 cannot be satisfied as written.
   Track B lost turns discovering this. **Next wave: either exempt the wave-log path in the hook, or
   instruct agents to return the log as text for the orchestrator to write.** Tracks A and C were
   unaffected, which is why it surfaced late and inconsistently.
2. **Two of my three environment instructions were wrong** — `PYTHONPATH=$PWD/src` (defeated by the
   editable install's `MetaPathFinder`) and `--seed 42` (not a GEDLIB option). Both were stated
   "verbatim, do not search for alternatives", which is the right instruction only when the command
   has actually been run in a worktree first. **Next wave: run every pasted command once in a throwaway
   worktree before pasting it.**
3. **Wave-scoped constraints leaked into production code.** My "no more than 4 processes" was about
   three agents sharing a box; track A reasonably made it a permanent CLI invariant. **Next wave:
   label such constraints explicitly as session-scoped in the prompt.**
4. **The agent-definition files were not discovered**, so all three spawned via `general-purpose`
   with `isolation: worktree` and the persona folded into the prompt. Worked, but the frontmatter
   `effort` setting was lost.

## Open follow-ups

- The frozen M7 gate admits only `BIPARTITE` at the upper end (+146 %). **User decision 2026-08-13:
  frozen gate stays primary, tighter methods reported as a disclosed sensitivity arm.**
  `BRANCH_FAST` is the lower-bound primary regardless.
- Plan-file corrections owed to `review-close`: `approx_ged.md` §5 (four citation defects and an
  omitted survey), `gedlib.md` §5 (HED, IPFP determinism, stale timings), `statistics.md` D6
  (`STAR` requires uniform costs).
- The `n ≤ 12` → `n = 98` transfer gap is **narrowed, not closed**, by design §3.12. Bracket width
  vs `n` remains T-05's.
