# Track B — analysis and figures · wave `2026-08-13-t27-bakeoff`

> **Authorship note.** This log was written by the **orchestrator**, not by the agent that did the
> work. A `PreToolUse` hook blocks subagents from writing report-style files under
> `.claude/notes/`, so the agent attempted the mandated log twice, was refused, and correctly
> declined to route around the guard via Bash. Content is taken from the agent's two completion
> reports; every number below was **re-verified by the orchestrator** in this worktree before being
> recorded. The blocked-log problem is a defect in the wave setup, not in the agent's work — see
> `summary.md`.

## Mission

Build the aggregation, graph-level bootstrap, significance testing and figures that turn track A's
per-cell GEDLIB measurements into the T-27 method selection.

## Files changed

| Path | Lines |
|---|---:|
| `benchmarks/real_data/eval_setup/ged_bakeoff_analysis.py` | 2,887 |
| `src/isalgraph/viz/bound_bakeoff_view.py` | 571 |
| `tests/unit/test_ged_bakeoff_analysis.py` | 1,043 |
| `tests/viz/test_bound_bakeoff_view.py` | 324 |

Commits: `022a2a4` implementation · `1b79a83` tests · `3f62cf6` orchestrator amendments +
figure layout · `0ac3058` factorized ranking.

## Design decisions

- **Counting-sort midranks.** Every ranked quantity is drawn from a small set, so ranking is
  O(p + K) rather than O(p log p). Exactness is asserted against `scipy.stats.rankdata`.
- **Dense factorization once per cell** (`0ac3058`). `np.unique(..., return_inverse=True)[1]` is
  computed once in a shared `_bootstrap_state` builder; ranks come from the codes, error means from
  the raw edit-operation counts. Ranks are invariant under strictly monotone maps, so Spearman is
  unchanged. This **removes** the integrality precondition rather than widening it — no scaling
  constant appears anywhere in the module.
- **Family size is derived, never literal.** `methods_for_end` yields 10 comparisons for the lower
  end and 6 for the upper; no cardinality is hard-coded.
- **`methods_for_end` vs `cells_for_end`.** The upper end has 7 cells but only 4 selection
  competitors — `BIPARTITE` plus the three `_MS` arms. A test asserts a `_DET` cell can never become
  a selection candidate.
- **Four figures, not two.** The two contracted figures keep exactly their promised panels; design
  §3.8's critical-difference diagram goes to separate `T27_{lower,upper}_cd.{pdf,png}` rather than
  altering the contracted panel count.

## Deviations from the brief, with evidence

1. **`PYTHONPATH=$PWD/src` does not work in a worktree — the brief's test command was wrong.**
   `isalgraph` is editable-installed through scikit-build-core, which registers a `MetaPathFinder`;
   that takes precedence over `sys.path`, so the worktree's `src` stays invisible. The viz test
   extends `isalgraph.viz.__path__` instead, which is a no-op after merge. **Orchestrator-verified**:
   no `sys.path`, `__path__` or `PYTHONPATH` manipulation exists in either production module.
2. **The mandated work log could not be written** — see the authorship note above.

## Evidence — orchestrator re-ran all of it in this worktree

```
$PY -m pytest tests/unit/test_ged_bakeoff_analysis.py tests/viz/test_bound_bakeoff_view.py -q
  98 passed in 1.53s
$PY -m pytest tests/unit/test_ged_bakeoff_analysis.py -q -k "hed or factoriz"
  8 passed, 75 deselected
$PY -m ruff check <all four owned files>          All checks passed!
git status --porcelain                            (empty)
```

Agent-reported, not independently re-run: full suite 1,200 passed / 271 skipped; `mypy --strict`
clean on owned files.

### The HED fast-path episode — and a correction to the orchestrator

The agent flagged that a non-integral bound value would drop `midranks` off its fast path. It could
not know which method would do that; the orchestrator measured it — **`HED --edge-set-distances
OPTIMAL` emits quarter-integers** (`0, 0.25, … 1.75`, 8 distinct values on LINUX), so one of the
twelve cells is non-integral in every dataset.

**The orchestrator then estimated the cost at ~8 hours for Letter HIGH. That was wrong by roughly
two orders of magnitude**, and the agent refuted it by measuring the counterfactual directly rather
than extrapolating — same bootstrap, HED's codes swapped back to raw values:

| configuration | jobs=1 | jobs=8 |
|---|---|---|
| factorized | 0.642 s/rep → **21.4 min** | 0.299 s/rep → **10.0 min** |
| HED unfactorized | 0.712 s/rep → 23.7 min | 0.325 s/rep → 10.8 min |

The 8.5× ranking penalty is real but local: one HED column costs 84.3 ms raw against 9.9 ms
factorized, and that is one of fourteen ranked arrays per replicate, with ranking only part of the
replicate cost. **True penalty ≈ 10 %, i.e. 2.3 minutes serial.** The bootstrap was never at risk of
the 2-hour threshold, before or after.

**The fix stands on its own merits** — it removes a precondition instead of guessing at a
granularity — but the justification is "free robustness", not "averted a budget blowup".

**Use 10.0 min on 8 workers** as the bootstrap figure. The agent's earlier 6.8 min was measured
while the machine was shared and while the fixture still had integral HED.

## Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Unit tests incl. seed-42 reproducibility, induced-pair correctness, one-sided censored validity, `exact == 0` exclusion, hand-checked Holm | **PASS** — re-run by orchestrator |
| 2 | Viz tests incl. import-without-matplotlib | **PASS** — re-run by orchestrator |
| 3 | `ruff` + `mypy --strict` clean on owned files | **PASS** (ruff re-run; mypy agent-reported) |
| 4 | End-to-end on synthetic fixture: 5 schema-valid JSON + figures | **PASS** — agent-reported |
| 5 | Bootstrap timing projection stated | **PASS** — 10.0 min, well under threshold |

## Open issues

- **M7 will report `unevaluated`, never `pass`, unless track A's n≈30 probe exists.** Deliberate and
  left as-is: the frozen gate is "< 1 ms/pair at n̄ = 30" and no such pair exists in the `n ≤ 12`
  bake-off corpus. The report must state the gate was not applied.
- The full-suite count (1,200) is far above CLAUDE.md's reference state of 726 passed / 271 skipped.
  Higher, not lower, so nothing regressed — but the documented reference state appears stale and the
  true post-merge number must be established at integration.
- All analysis so far is against the **synthetic fixture**. Nothing here is evidence about real GED
  bounds until track A's cells exist and the campaign runs.
