# Wave summary -- 2026-08-10-cpp-and-viz

**Date:** 2026-08-10
**Orchestrator:** main session (Opus 5, effort max)
**Base commit:** `2f393a1`
**Integration branch:** `integration/2026-08-10-cpp-and-viz`
**Work branch to land on:** `main`

---

## 1. User's original prompt (verbatim, task portion)

```
CONTEXT: This is the IsalGraph project, the first project from the Isal family we
developed. It was sent as a paper to Pattern Recognition (article in
/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199).
The sibling projects are IsalSR (code: /home/mpascual/research/code/IsalSR article:
/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a
Sent to TPAMI) and IsalHG (code: /home/mpascual/research/code/IsalHG preprint:
/media/mpascual/Sandisk2TB/research/ISAL/isalhg/article/preprint/6a31285eea907326ad1128f2, not yet
sent to a journal). IsalGraph, being the first project from the three, carries some code flaws and
things that we have been refining afterwards, for example, is missing a C++ engine, and we did not
know some properties about the isal family when we developed it.

TASK: IsalGraph has recieved a major revision from Pattern Recognition, and we have until the 31st
of august to develop it (21 days). Read the reviewers notes [...]. We need to clear up the codebase
so that the original experiments' routines are clearly defined, we need to develop a C++ engine for
IsalGraph (examples for IsalSR in /home/mpascual/research/code/IsalSR/src/isalsr/core/native and
IsalHG in /home/mpascual/research/code/IsalHG/src/isalhg/core/_native ; some useful docs in
/home/mpascual/research/code/IsalSR/.claude/notes/review/tasks/T01-cpp-core-port.md
/home/mpascual/research/code/IsalSR/.claude/notes/review/tasks/T01-appendix
/home/mpascual/research/code/IsalHG/docs/engineering/CPP_OPTIMIZATION_LOG.md) we also need to
develop a standard visualization tool for the instruction string, CDLL, and graphs, HEAVILY
inspired by IsalHG, check /home/mpascual/research/code/IsalHG/src/isalhg/viz ; When everything is
done regarding the refactorization of the codebase, we must update CLAUDE.md of the project to
point out to sibling projects, article folder, new codepaths for visualization, always use the C++
engine, ... Finally, we need to organize the notes from the reviewers for the revision, we dont
need to declare tasks just yet, but create a structure like the one we had for IsalSR
(/home/mpascual/research/code/IsalSR/.claude/notes/review/source) with the notes for the review
organized and grouped by similarity.

ACCEPTANCE CRITERIA: You have planned the refactorization of the codebase so that the things I have
described are clear, have implemented the C++ engine and tested it w.r.t.: (a) Properties of
IsalGraph (e.g., Roundtrip, Isomorphism deduplication, Canonicality, ...) (b) Reproduce some
partial results from the paper on real data ; You have also implemented the visualization submodule
to help us generate visualizations for IsalGraph easily; You have updated the CLAUDE.md of the
project, and have created the reviewers notes.

HOW TO PROCEED: I propose you the following: Perform the refactorization of the codebase, so that
everything is correctly cleaned up. Then, load /parallel-agents and follow the protocol described
in there, spawning agents Claude Opus 5 to perform in parallel the implementation of the C++ engine
(max effort) and the visualization submodule (xhigh effort). Then, you take care of the merging of
their branches and the CLAUDE.md update. Finally, you spawn a subagent for the review setup md
docs (the original mail is in .claude/notes/review/source/mail.txt). Think step by step, reason,
iterate for as long as you need.
```

---

## 2. Preflight findings

Two blockers that would have wasted the wave, found before spawning:

1. **The `isalgraph` conda environment did not exist on this machine**, despite
   `CLAUDE.md` documenting it. Created it (Python 3.11.15 + nanobind 2.14 +
   scikit-build-core 1.0.3 + ninja), plus an `isalgraph-cpp` clone so the C++
   agent's editable install could not repoint the environment under its peer.
2. **`IsalSR`'s T01 notes record "no git worktrees"** -- an editable install
   resolves to the main checkout, so a worktree's tests silently exercise the
   wrong code. Verified the mechanism here: the install is a plain `.pth`
   holding the main `src` path. Resolved per agent rather than abandoning
   isolation: the viz agent used `PYTHONPATH=$WT/src` (safe, pure Python), the
   C++ agent used a dedicated environment with its own editable install
   (`PYTHONPATH` would have shadowed the extension). Both were required to
   print `isalgraph.__file__` before starting; both did, and both resolved
   inside their own worktree.

Also set `.claude/settings.local.json` `worktree.baseRef: "head"`, without which
worktrees branch from `origin/HEAD` and would have missed the refactor commit.

---

## 3. Pre-wave refactor (commit `2f393a1`)

Baseline before: 449 passed / 271 skipped / **1 failed** -- a Hypothesis
`DeadlineExceeded` flake imposing a 200 ms per-example deadline on an
exponential-time canonical search. Fixed with `deadline=None`. After: 450/271/0.

- `slurm/` -> `experiments/synthetic_suite/`; `experiments/` is now the single
  orchestration root and `benchmarks/` holds the routines.
- `experiments/README.md`: the reproducibility registry -- every figure, table
  and reported number mapped to its generating function, the pipeline DAG,
  dataset provenance, and the code that produces no paper artifact.
- Corrected `paper_outputs`, exported the drifted public API, extended the
  exception hierarchy, declared `networkx` in the `viz` extra.

Three reproducibility defects recorded rather than silently patched: the
`fig_` prefix mismatch on `composite_method_tradeoff_v2`, `paper_outputs`
listing a figure the paper does not use while omitting Table 1's source, and
two tables hand-edited after generation.

---

## 4. Agents

| | `wave-cpp-engine` | `wave-viz` |
|---|---|---|
| Model / effort | Opus 5 / max | Opus 5 / xhigh |
| Branch | `wave/cpp-engine` | `wave/viz` |
| Head | `9331189` | `bb693c2` |
| Files changed | 43 | 40 |
| Log | `.claude/notes/2026-08-10-cpp-and-viz/cpp-engine.md` | `.../viz.md` |
| Verdict | **ACCEPT** | **ACCEPT** (after one RETURN) |
| Duration | ~97 min | ~46 min |

Definition files archived at `.claude/notes/2026-08-10-cpp-and-viz/agents/`.
`.claude/agents/` did not exist at session start, so its contents were not
discoverable as `subagent_type`; both agents were spawned as `general-purpose`
with `model` and `isolation` set explicitly and the system prompt folded into
the delegation. Effort therefore inherited the session's `max` for both, so
`wave-viz` ran hotter than the specified `xhigh`.

### Verification performed by the orchestrator (not taken from the logs)

Both: worktree clean, zero ownership violations, log file list matching
`git diff --name-only`, tests re-run in the worktree, `isalgraph.__file__`
confirmed resolving inside the worktree.

`wave-viz`: 560 passed / 271 skipped / 0 failed, mypy strict clean, ruff clean.
Its two claims were checked independently and both held -- 80 ruff errors in
`benchmarks/` at base *and* in the worktree (zero introduced), and
`INSTRUCTION_COLORS["N"] == "#4477AA"` at `2f393a1`.

`wave-cpp-engine`: 605 passed / 281 skipped / 0 failed with the engine;
`.so` moved aside -> 450 passed / 276 skipped, exactly the baseline, with
`engine()` returning `python` and explicit `backend="cpp"` raising
`BackendError` rather than degrading. `build_info()` reports
`x86-64-v3 / gcc 12.2.0 / build_hash 298fc1188bf1b051`.

---

## 5. Interventions

1. **Exception contract (to `wave-cpp-engine`).** It correctly found that the
   `errors.py` classes I had written did not inherit the builtins that ~30
   tests pin, so dispatching `algorithms/*.py` through the backend would break
   them. Its proposed fix -- `EncodingError(IsalGraphError, ValueError)` -- was
   wrong: `EncodingError`'s descendants straddle the split, since an
   unreachable start node is historically a `ValueError` while "no valid
   operation found" is a `RuntimeError`. Supplied a corrected contract putting
   the builtin on the **leaves** and adding `EncodingStuckError`; validated
   every MRO before sending.
2. **Byte-parity risk (to `wave-cpp-engine`).** Asked it to confirm that the
   canonical search reads the *output* graph's adjacency only for membership,
   never iterating it to choose. It audited and confirmed: the only first-wins
   iteration is `graph_to_string.py:286` on the *input* graph.
3. **IAM surrogate (to `wave-cpp-engine`).** Approved, with the added
   requirement to report the surrogate's realised distribution against the
   IAM LOW targets rather than asserting a match.
4. **RETURN (to `wave-viz`).** Work accepted but the mandatory log was
   uncommitted. Also required the g2s docstring to warn against reading a
   replay trace as an encoder trace.
5. **Withdrew an acceptance criterion (to `wave-viz`).** "ruff clean on
   `benchmarks/`" was my error -- 80 errors were already there at base.

---

## 6. Merge record

Both merges **clean, zero conflicts** -- the ownership partition held.

```
git switch -c integration/2026-08-10-cpp-and-viz 2f393a1
git merge --no-ff wave/cpp-engine    # contracts first
git merge --no-ff wave/viz
```

Integration repairs, commit `9510290` (`fix(integration): ...`):

- Landed the exception mixins per the contract; added `EncodingStuckError`.
- Deleted `wave-cpp-engine`'s `_as_legacy_value_error` shim and **confirmed it
  was a no-op**: `tests/unit/test_algorithms.py` stayed at 41 passed, and the
  10 gated native error tests flipped skip -> pass exactly as predicted
  (total skips 281 -> 271).
- Rewired both `__init__.py` to re-export the dispatching entry points from
  `core.backends`, so `isalgraph.canonical_string` runs on the active engine.
- Fixed `tests/viz/test_import_without_matplotlib.py`: purging
  `isalgraph.core._native` from `sys.modules` re-ran nanobind's module init,
  emitting `RuntimeWarning: type 'Cdll' was already registered!` and leaving
  two type objects for one C++ class. A test artifact, but it masked real
  warnings; the extension is now exempt from the purge.
- Widened the `benchmarks/` per-file-ignores to the naming-convention rules.

### Final verification

| Configuration | Result |
|---|---|
| Engine present | **726 passed / 271 skipped / 0 failed / 0 warnings** |
| `.so` removed | **561 passed / 276 skipped / 0 failed** |
| `ruff check src/ tests/` | clean |
| `mypy --strict src/isalgraph/` | clean, 42 files |

---

## 7. Substantive findings for the revision

1. **The canonical string does not encode directedness.** `wave-cpp-engine`
   raised this; its example (the 3-node path) was wrong -- that gives `'VV'`
   undirected vs `'Vpv'` directed. The phenomenon is real, and the orchestrator
   established the correct minimal witness: a **single undirected edge and a
   single directed arc both canonicalise to `"V"`**, with **63 of 441**
   comparable small graphs colliding. The complete-invariant theorem holds
   within a fixed directedness class, not across. This is direct empirical
   confirmation of Reviewer 3's point 3.
2. **Reviewer 3's complexity objection is correct on both counts.**
   `conclusion.tex:50` cites `n^{9.0}`, which appears nowhere in `results.tex`
   (reported: 3.1, 4.5, 4.9, and 10.2 for GED), and `conclusion.tex:80` calls
   `n^{4.9}` "super-polynomial", a category error.
3. **Reviewer 3's Algorithm 2 objection is the opposite of what they think.**
   The *code* matches Table 1 correctly at `graph_to_string.py:207-236`; the
   defect, if any, is in the manuscript's pseudocode. Cheap to fix.
4. **Answer to "recomputed or precomputed":** the published Python recomputes
   the displacement-pair list at *every* recursion frame. Memoising it is the
   dominant optimisation, worth 25x-109x on its own.
5. **Threading is a negative result** on this workload and is documented as
   such: 4 threads are 1.8x slower at n=6.
6. **The WL-kernel baseline already exists and is already computed**
   (`eval_setup/wl_kernel_computer.py`) but is never reported. Lowest-cost
   substantive response available to Reviewer 1's "compare against alternative
   approaches".

---

## 8. Open follow-ups

| Item | Severity | Detail |
|---|---|---|
| IAM results are a surrogate | **high** | Data is Picasso-only. The surrogate caps at 7 nodes and cannot reach 12: "connected" plus "mean 3.07 edges" forces mean nodes near 3.85. It is parity evidence, not scalability evidence. **Regenerate on Picasso before anything from it reaches the response letter.** |
| 28 ruff errors in `benchmarks/` | low | E501/B007/SIM113/SIM108/F841/I001, all pre-existing. Genuine signals, deliberately not suppressed. |
| `cohort_panel.py` not ported | low | No consumer in this repo. |
| `nx_view`, `alignment_view` untested | low | Absorbed verbatim; import shims only, no rendering tests. |
| networkx backend ghosting | low | Length-matches `ax.collections`; a networkx change could silently disable it. Default backend unaffected. |
| Directedness in the paper pipeline | medium | Check whether any dedup keys on the canonical string alone over a mixed-directedness corpus. |

---

## 9. What the decomposition got wrong

- **An acceptance criterion I wrote was unmeetable** ("ruff clean on
  `benchmarks/`"). I set it without running the check first. Withdrawn
  mid-wave. Verify a gate is currently passable before making it a gate.
- **`errors.py` was under-specified when I froze it.** I added the exception
  classes during the pre-wave refactor without checking what the existing tests
  pinned, which forced a shim in the C++ agent's worktree and an integration
  repair. Reading the ~30 `pytest.raises` sites would have cost minutes.
- **The colour instruction in the viz brief was not followable as written.** I
  specified tying the pointer channel to hue without checking the published
  palette, where uppercase `N` already carries the secondary-pointer blue. The
  agent caught it and produced a better design.
- **`.claude/agents/` did not exist**, so the definition files were inert and
  effort could not be set per agent. Create it before the session that needs it.
- A stray non-English token ("frozen-reference問題") reached the C++ agent's
  prompt. Harmless in context, but prompts should be proofread.

What the decomposition got **right**: the ownership partition produced zero
merge conflicts across 83 changed files and two agents touching the same
package, and the ordered-marshalling instruction in the C++ brief -- the one
insight that made byte-exact greedy parity possible -- worked on first attempt.
