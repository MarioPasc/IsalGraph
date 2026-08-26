# track-C-measure — wave `2026-08-26-t13-complexity`

Branch: `worktree-agent-a3dc68c8694abf8d5` · Base: `1ace4f9` · Head: see final commit below.

---

## Mission

Build the controlled-experiment runner for T-13: the CLI that times every representation on
every constructed graph under a frozen budget and timing rule, emits one immutable record per
`(graph, representation, arm)`, and the SLURM launcher/worker pair that fans it out on Picasso.

T-13 must replace the manuscript's unqualified "exponential worst case" with a characterised one
(R3.7d): the cost of a canonical search is governed by `|Aut(G)|`, not by size or density. On the
real IAM cohort the marginal Spearman ρ(log|Aut|, log t) is only +0.189 against +0.326 for log n,
and only the within-fixed-`(n, m)` contrast recovers the effect (+0.655, positive in 12 of 13
strata). An effect that small does not survive a sloppy clock, so the timing rule is the
deliverable as much as the code is.

---

## What I built

### `benchmarks/real_data/eval_t13_complexity/schema.py`

The frozen `t13.1` record, as a frozen slotted dataclass, plus a validator that rejects a record
with a **missing or an extra** field. Both directions are errors: the analysis builds its frame
from `FIELDS`, and a row carrying an undeclared key comes from code that disagrees with this
module about what a measurement is — a second `wall_seconds` beside `seconds`, a second `length`.
Those are precisely the pairs a permissive reader averages together.

Beyond the shape check, the validator rejects status/field combinations the frozen timing rule
cannot emit: `status="ok"` with `repeats=0` (a row whose `seconds` was never measured),
`status="censored"` with a non-null `length_chars` (the signature of a fallback laundering a
censored graph into a completed one), and `fallback_used=True` (see *Decisions*).

Nothing in this module imports `isalgraph`, `networkx` or `numpy` — the schema must be readable
by an analysis process with no C++ engine.

**39 fields**, `SCHEMA_VERSION = "t13.1"`: the 37 of CONTRACTS §5.1 as amended, plus the
orchestrator's `fallback_used` and `budget_spec`, plus `params` (see *Defects found*).

### `benchmarks/real_data/eval_t13_complexity/measure.py`

The runner. The pieces that carry the design:

| Function | What it settles |
|---|---|
| `assert_engine()` | Aborts unless `engine() == "cpp"` **and** `build_hash == 298fc1188bf1b051`. Not a warning. |
| `timed_call()` | The timing rule, implemented once. Clock injected, so both branches are pinnable. |
| `engine_arm()` | Context manager for the ablation arms; restores both toggles to `True` in a `finally`. |
| `run_unit()` | The budget: a subprocess killed at `budget_s + grace`. Never `SIGALRM`. |
| `execute_unit()` | The child half: builds the timed callable, classifies every exception into a status. |
| `shard_of()` / `unit_digest()` | `blake2b`, not `hash()` — Python's string hash is salted per process. |
| `select_ablation_graphs()` | The stratified ablation subsample, fixed in code before any result exists. |
| `canonical_identity_violations()` | The orchestrator's free gate: the two canonical arms must agree at `n <= 12`. |

**One fresh subprocess per work unit.** This reads as overhead and is actually the measurement.
A persistent worker would carry the engine's pair-memoisation cache across units, so a graph's
measured cost would depend on which graphs preceded it in the shard — and the shard order is a
hash. Measured cost of the isolation: **82 ms per unit** including interpreter startup, under
1 % of the design note's 400–700 core-h estimate.

### `benchmarks/real_data/eval_t13_complexity/slurm/launch.sh` · `slurm/worker.sh`

Whole-node CPU array. One array task owns one node `--exclusive` and runs 128 single-threaded
shards concurrently, each pinned to its own core with `taskset`. No `#SBATCH` header in the
worker; every flag is on the `sbatch` command line, following
`experiments/paper_pipeline/launch.sh`. Detail in the *SLURM* section.

### `benchmarks/real_data/eval_t13_complexity/tests/{test_schema.py,test_measure.py}`

124 tests. The field list in `test_schema.py` is **written out by hand** rather than derived from
the dataclass: a test that reads `FIELDS` to check `FIELDS` cannot catch a rename.

### `benchmarks/eval_t13_complexity` → `real_data/eval_t13_complexity`

The symlink, per the repo convention.

### `__init__.py`, `tests/__init__.py` — track A's files, stubbed

Created as minimal stubs because `mypy` cannot type-check without them (see *Defects found* #3).
`__init__.py` exports nothing, per CONTRACTS §1, so the merge conflict should be a docstring
only. **Track A should keep its own version and discard mine.**

---

## Acceptance criteria

| # | Criterion | Command | Result |
|---|---|---|---|
| 1 | Frozen dataclass, `schema_version = "t13.1"`, validator rejects missing/extra | `pytest tests/test_schema.py` | **PASS** — `66 passed in 0.08s`. `FIELDS` pinned against a hand-written list; every one of the 39 fields is parametrised for the missing-field case. |
| 2 | The thirteen representations resolve | `$PY -c "from benchmarks.eval_t13_complexity import measure; print(measure.resolve_representations())"` | **PASS** — 13/13 `ok`, none flagged. Output below. |
| 3 | Timing rule implemented once, both branches pinned with a fake clock | `pytest -k TestTimingRule` | **PASS** — slow branch: 1 call, `repeats=1`; fast branch: 4 calls, `repeats=3`, median of `(0.2, 0.9, 0.3) = 0.3`. Threshold is `>=`, so 1.0 s takes the slow branch. |
| 4 | Budget is a killed subprocess; slow unit `censored`, parent survives | `pytest -k TestBudgetIsAKilledSubprocess` | **PASS** — a 60 s child under a 0.2 s budget returns `status="censored"`, `error_kind="wallclock_kill"`, `seconds == 0.2`, `length_chars is None`, in well under 30 s; the parent then runs a second unit to completion. Module contains no `signal.alarm(`, no `signal.setitimer(`, no `import signal`. |
| 5 | Engine gate **aborts** on a wrong build hash | `pytest -k TestEngineGate` | **PASS** — monkeypatched `build_info` → `EngineMismatchError`; monkeypatched `engine()` → `EngineMismatchError`; the real environment returns `298fc1188bf1b051`. |
| 6 | Ablation arms restore both toggles, including on an exception | `pytest -k TestAblationArmsRestoreState` | **PASS** — the fake-native test asserts the exact call sequence; a second test drives the **real** `_native`, asserts `pairs_memo()`/`branch_and_bound()` are `False` inside the block, raises, and asserts both are `True` after. |
| 7 | A declining backend is `unsupported`, never dropped | `pytest -k TestDecliningBackendsAreRecorded` | **PASS** — `agm_cam` on `C_13` → `status="unsupported"`, `error_kind="SuiteScopeError"`; the same graph at `n=12` → `ok` with `length_chars = 66 = C(12,2)`. Same for `isalgraph_canonical`. |
| 8 | Sharding partitions the grid exactly for `N ∈ {1, 7, 64}` | `pytest -k TestSharding` | **PASS** — 1,040 units; union over shards equals the key set, no overlap, no loss, every shard occupied; membership independent of enumeration order. |
| 9 | Local smoke on **real** data, green | see *Smoke run* | **PASS** — 246 records, all `ok`, all 13 representations present, all re-validate against the schema. |
| 10 | SLURM per `picasso-sbatch`, `bash -n`, ≥ 2 h/task | see *SLURM* | **PASS** — both scripts `bash -n` clean; preview below. |
| 11 | The two test files pass | `$PY -m pytest .../tests/test_schema.py .../tests/test_measure.py -q` | **PASS** — `124 passed in 1.47s`. |
| 12 | `ruff` and `mypy` clean | below | **PASS** — `All checks passed!` / `Success: no issues found in 6 source files`. |

### Criterion 2, verbatim

```
$ $PY -c "from benchmarks.eval_t13_complexity import measure; print(measure.resolve_representations())"
isalgraph_canonical  : ok (ReprBackend)
isalgraph_exhaustive : ok (ReprBackend)
isalgraph_pruned     : ok (ReprBackend)
isalgraph_greedy     : ok (ReprBackend)
nauty_graph6         : ok (ReprBackend)
sparse6_nauty        : ok (ReprBackend)
min_dfs              : ok (ReprBackend)
agm_cam              : ok (ReprBackend)
adjacency            : ok (ReprBackend)
graph6               : ok (ReprBackend)
sparse6              : ok (ReprBackend)
wl_subtree           : ok (VectorBackend)
size_null            : ok (ReprBackend)
```

`size_null` carries `Capability.BASELINE`, so `available_backends()` returns **12** and omits it.
`REPRESENTATIONS` therefore **names** all thirteen rather than discovering them; a test asserts
`"size_null" not in available_backends()` so that a future refactor to discovery fails loudly
instead of silently dropping the null arm.

### Criteria 11 and 12, verbatim

```
$ $PY -m pytest benchmarks/real_data/eval_t13_complexity/tests/ -q
124 passed in 1.47s

$ $PY -m ruff check --fix benchmarks/real_data/eval_t13_complexity/
All checks passed!

$ MYPYPATH=. $PY -m mypy --explicit-package-bases benchmarks/real_data/eval_t13_complexity/
Success: no issues found in 6 source files
```

---

## Smoke run (criterion 9)

Exactly the command in the brief, on the real local cohort:

```
$ export ISALGRAPH_COHORT_ROOT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data
$ $PY -m benchmarks.eval_t13_complexity.measure --source cohort --dataset iam_letter_low \
      --shard 0 --n-shards 64 --arms default --budget-s 30 --seed 13 --out /tmp/t13_smoke.jsonl

WARNING t13.measure symmetry.py (track A) is absent: the nine symmetry fields will be null
        and no row of this shard may enter the |Aut| regression
INFO    t13.measure grid: 1180 graphs from source=cohort
INFO    t13.measure shard 0/64 complete: {'ok': 246}

real    0m20.546s
```

### Row and status counts

```
rows in file      : 247 (1 header + 246 records)
header build_hash : 298fc1188bf1b051  engine=cpp
symmetry_available: False
status counts     : {'ok': 246}
distinct graphs   : 227
n range           : 2-7
budget_spec       : search_nodes=200000,max_projections=50000,timeout_s=30.0
```

### Per-representation median `seconds`

| representation | rows | median seconds | statuses |
|---|---:|---:|---|
| `isalgraph_canonical` | 17 | 3.932e-05 | ok |
| `isalgraph_exhaustive` | 16 | 3.911e-05 | ok |
| `isalgraph_pruned` | 21 | 3.905e-05 | ok |
| `isalgraph_greedy` | 17 | 4.876e-05 | ok |
| `nauty_graph6` | 23 | 9.998e-05 | ok |
| `sparse6_nauty` | 19 | 1.398e-04 | ok |
| `min_dfs` | 25 | 3.184e-05 | ok |
| `agm_cam` | 20 | 5.256e-05 | ok |
| `adjacency` | 17 | 1.944e-05 | ok |
| `graph6` | 19 | 5.036e-05 | ok |
| `sparse6` | 16 | 4.186e-05 | ok |
| `wl_subtree` | 14 | 2.339e-05 | ok |
| `size_null` | 22 | 2.660e-06 | ok |

**All 13 representations appear.** `size_null` is the cheapest at 2.7 µs and `sparse6_nauty` the
dearest at 140 µs, which is the ordering the design predicts. Every one of the 246 rows
re-validates against `schema.validate_mapping` after being read back from disk.

**These are not evidence for or against the cost law and must not be quoted as such.**
`iam_letter_low` is `n = 2..7`; at that size the canonical search is trivial and the three
IsalGraph arms are indistinguishable (3.90–3.93e-05, a 0.7 % spread). The law lives at
`n = 8..64` on the constructed grid, which is exactly why the controlled experiment exists.

### Resume

Re-running the identical command produced `{}` — zero new records — and left the file at 247
lines. The shard file is append-only and a requeued task picks up where it stopped.

---

## SLURM

### `slurm/launch.sh`

Login-node entry point. Validates `--source`, computes the array size from the shard count,
creates the log and results directories, and submits. Every resource flag is on the `sbatch`
line; the worker carries no `#SBATCH` header, per `experiments/paper_pipeline/launch.sh`.

Three things it does that are not obvious:

1. **Job-ID capture takes the last line first.** Picasso's Lua `sbatch` wrapper writes ANSI
   codes and a multi-line warning to stdout, so `--parsable` does not return just the id. A
   line-by-line `sed 's/[^0-9]//g'` leaves the warning's newlines in place, and the guard that
   then rejects the result fires *after* the job was submitted — leaving an untracked job on the
   cluster, which is worse than no guard.
2. **The arm list is shipped colon-separated.** `--export` splits on **every** comma, so
   `T13_ARMS=default,no_bnb` would deliver `default` and parse `no_bnb` as a junk variable name,
   silently. The worker translates `:` back to `,`.
3. **It refuses a shard count that is not a multiple of the shards-per-task.** A ragged last
   task is precisely the short job SCBI asked this account to stop submitting.

### `slurm/worker.sh`

Per array task: resolve the shard range, gate the engine **once** (turning a 128-way identical
failure into one legible line), stage prior shard files in from `$LOCALSCRATCH`, launch one
`taskset`-pinned shard per core, wait, mirror the whole output tree back, report failures.

- **`unset PYTHONPATH`.** The skill's worker template sets `PYTHONPATH=<repo>/src`; here that is
  actively harmful — a src-first path shadows the editable install and the engine falls back to
  pure Python with **no error**, making every timing in the campaign fiction. `--chdir` puts the
  repo on `sys.path` for `-m benchmarks...`, which is all that is needed. `measure.run_unit` also
  pops `PYTHONPATH` from the child environment, so the guard survives an operator who exports it.
- **Every thread pool pinned to 1.** `time.process_time` sums CPU across all threads of the
  process, so an unpinned BLAS would inflate every reading by its thread count.
- **`$LOCALSCRATCH` with a whole-tree copy-back**, `trap ... EXIT` plus `trap 'exit 143' TERM`.
  Without the TERM trap a wall-clock kill terminates the shell without running the EXIT trap and
  the task's entire output dies on the node. The copy-back mirrors `out/` wholesale and never
  enumerates expected filenames; each file goes to a temp name in the destination and is then
  `mv -f`'d, since a rename within one directory is atomic.
- **Per-task `PYTHONPYCACHEPREFIX`.** Picasso exports a *shared* one, so 128 shards on one node
  would write the same `.pyc` paths. The symptom is an intermittent `ModuleNotFoundError` on a
  module that is plainly present, hitting a small fraction of shards.

### `bash -n`

```
$ bash -n benchmarks/real_data/eval_t13_complexity/slurm/launch.sh && echo OK
launch.sh: bash -n OK
$ bash -n benchmarks/real_data/eval_t13_complexity/slurm/worker.sh && echo OK
worker.sh: bash -n OK
```

### Submission preview

`sbatch --test-only` cannot be run here — there is no SLURM on the workstation, and I am
forbidden to touch the cluster. `launch.sh --test-only` passes the same argument vector to
`sbatch --test-only` for the orchestrator to run on the login node. `--dry-run` prints it:

```
$ bash slurm/launch.sh --source constructed --dry-run

Run ID:          t13_20260826T102251Z_constructed
Source:          constructed
Arms:            default
Shards:          128 (128 per task, one per core)
Array tasks:     1 (indices 0-0)
Node family:     sr (pinned: wall clock is the reported quantity)
Budget:          300 s per (graph, representation, arm)
Wallclock:       1-00:00:00

[DRY-RUN] sbatch --parsable --job-name=t13_constructed --array=0-0 --time=1-00:00:00
  --nodes=1 --ntasks=1 --exclusive --constraint=sr --account=tic_163_uma
  --chdir=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph
  --output=/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/logs/t13_constructed_%A_%a.out
  --error=/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/logs/t13_constructed_%A_%a.err
  --export=ALL,ISALGRAPH_REPO_DIR=...,ISALGRAPH_CONDA_ENV=...,ISALGRAPH_COHORT_ROOT=...,
           T13_RESULTS_DIR=...,T13_RUN_ID=...,T13_SOURCE=constructed,T13_N_SHARDS=128,
           T13_SHARDS_PER_TASK=128,T13_ARMS_COLON=default,T13_BUDGET_S=300,T13_SEED=13
  <repo>/benchmarks/real_data/eval_t13_complexity/slurm/worker.sh
```

### Sizing arithmetic behind the ≥ 2 h grouping

Measured input: **82 ms wall per work unit**, including a fresh interpreter, on this workstation
(10 repetitions of `run_unit` on a 6-node cycle with `isalgraph_pruned`).

The constructed grid is ~700 specs × 13 representations ≈ 9,100 default-arm units, plus roughly
1,800 ablation units (3 ablatable representations × 3 non-default arms × 2 graphs per stratum).
At 82 ms the *uncensored floor* is only ≈ 0.25 core-h — the entire cost is the censored tail,
which is bounded by construction at `units × 300 s` because nothing runs past its budget. The
design note's estimate is **400–700 core-h**.

One `sr` node carries 128 cores, hence 128 shards:

```
  400 core-h / 128 shards = 3.1 h per array task    >= 2 h   OK
  700 core-h / 128 shards = 5.5 h per array task    >= 2 h   OK

  256 shards (two nodes)  = 1.6 h at the low estimate         REFUSED
```

**So the fan-out stops at one node per source.** The whole campaign is two array tasks — one
constructed, one cohort — not two hundred, and every submitted task clears SCBI's floor at both
ends of the estimate. Wallclock is set to `1-00:00:00` for constructed (worst case
`11,000/128 × 300 s = 7.2 h`) and `2-00:00:00` for cohort.

The node family is **pinned to `sr`**. Wall clock is the reported quantity of this ticket, and
`sd` (Intel, 2.1 GHz) against `sr` (AMD, 2.6 GHz) differ enough that an unpinned pool would turn
the cost law into a measurement of the scheduler. `sr` over `bc` (256 cores, 2.25 GHz) because a
single-threaded latency benchmark is served by the higher base clock, not by cores it cannot use.
The worker records `lscpu` per task regardless.

---

## NOTE TO THE ANALYSIS — censoring is not a small number

*(Headed for lifting into the report, at the orchestrator's request.)*

`seconds` is the **observation time**: the point at which observation stopped. `status` says
whether that point is a completion or a censoring point.

- `status="ok"` → an observed completion time.
- `status="censored"` with a **time** mechanism (`wallclock_kill`, `timeout_s`) → observation
  stopped at the budget, so `seconds == budget_s`.
- `status="censored"` with a **cap** mechanism (`max_projections`, `search_nodes`) → observation
  stopped when the cap fired, so `seconds` is the **measured** time to that point.

**The trap.** A cap-censored `min_dfs` row enters the file as a *small* number — 41 ms, say.
Under any naive summary (a median, a mean, an OLS fit on `log t`) that reads as *"min-DFS is
fast"* when what it means is *"min-DFS did not finish"*. It is the opposite of the wall-clock
case, where censoring inflates the summary and is therefore obvious.

Consequences, all mandatory:

1. **The cost-law fit must be censoring-aware.** A cap-censored row asserts "the completion time
   is greater than 41 ms" — true and informative as a right-censored observation, and actively
   misleading as a point estimate.
2. **Completion rate must be reported beside every timing summary**, never folded into it. A
   median of 40 µs at 12 % completion and a median of 40 µs at 100 % completion are different
   results and must not print the same.
3. **Filter on `status`, never on `seconds`.** Any threshold rule on `seconds` alone silently
   mixes the two censoring mechanisms back together.
4. `status="unsupported"` rows carry `seconds` measured up to the refusal, which is the cost of
   the scope check and not of the representation. Exclude them from timing summaries; count them
   in coverage.

---

## Decisions and assumptions

**D1 — the budget is enforced twice, and both paths produce the same record.** The parent's
subprocess kill is the outer, universal guard (`wallclock_kill`). The IsalGraph arms additionally
receive the engine's own `timeout_s`, so they stop cleanly rather than being killed
(`timeout_s`). Both write `status="censored"`, `seconds = budget_s`, `length_chars = None`.
`SIGALRM` is not used anywhere; T-05 finding 5 established that it does not interrupt the C++
engine, so a signal-based timeout silently fails to fire.

**D2 — one fresh subprocess per unit, not a persistent worker.** Isolation, not overhead: a
persistent worker would carry the pair-memoisation cache across units and make a graph's measured
cost depend on which graphs preceded it in the hash-ordered shard. Measured at 82 ms/unit, under
1 % of the campaign estimate.

**D3 — two keys per work unit.** `WorkUnit.key` hashes the **raw integer** `params` and decides
shard membership; `WorkUnit.record_key` is reconstructible from a written row and drives resume.
One string for both would force resume to invert the base-index rendering, and a rendering that
must round-trip is a rendering that cannot be improved. More importantly, a name-based hash means
that reordering `families.LADDER_BASES` silently moves every unit to a different shard.

**D4 — one fully populated `Budget` threaded through every backend**, at each field's frozen
published value: `search_nodes = 200_000` (`n <= 12`) / `100_000` (above), `max_projections =
50_000`, `timeout_s = budget_s`. This reproduces every backend's default exactly while making the
caps explicit enough to serialise into `budget_spec`. Leaving a field `None` is **not**
equivalent: `min_dfs.py:372` reads `cap = MAX_PROJECTIONS if budget is None else
budget.max_projections`, so a budget with the field unset runs it **unbounded** and re-opens the
OOM kill that ended the first Suite-2 run. A test asserts the three constants against the
backends' own module-level values, so a change there breaks here.

**D5 — AGM's node budget is suite-conditional and the constructed grid has no suite**, so the
rule is expressed on `n` at the `SUITE1_MAX_NODES = 12` boundary the backends already use. In
practice `agm_cam` refuses above `n = 12` anyway, so the Suite-2 value is never exercised by that
arm; it is set for correctness, not effect.

**D6 — `isalgraph_exhaustive` is the exhaustive-canonical arm for the constructed grid**;
`isalgraph_canonical` runs where its `SUITE1_ONLY` guard permits (`n <= 12`) and is recorded
`unsupported` above it. Frozen by the orchestrator, on three grounds I verified in the source:
the two arms share one `encode` path and one `variant="canonical"` and differ only in a scope
guard; the guard is a cohort policy, not a property of the algorithm, and a guard encoding T-13's
conclusion cannot be allowed to censor the experiment establishing it; and T-06 already ran
`isalgraph_exhaustive` to `n = 98`, which keeps T-13 joinable to its records.
`canonical_identity_violations()` implements the orchestrator's free consistency gate — at
`n <= 12` both arms must report identical `status` and `length_chars`. It runs per shard (free,
on whichever pairs co-reside) and is exported for the merge-time check, since sharding splits the
two arms by hash. A test drives it on the real Petersen graph and both arms agree.

**D7 — the ablation subsample rule, fixed in code before any result exists.** Ablation arms run
only on `ABLATABLE_REPRESENTATIONS = (isalgraph_canonical, isalgraph_exhaustive,
isalgraph_pruned)` — the two native toggles gate the canonical search's pair memoisation and its
branch-and-bound bound, so ablating a serialisation or the greedy encoder would cost four full
budgets to measure nothing. Within each `(source, family|dataset, n)` stratum, graphs are ranked
by `blake2b` digest and the first **2** are taken. Ranking by digest rather than by position
means the subsample cannot move when the enumeration order changes, and cannot be steered by a
later edit once the interesting cells are known.

**D8 — WL is timed as `fit([graph])` then `features(graph)`**, with `length_chars` = the number
of distinct colours. Building the colour vocabulary is part of computing a WL representation, not
part of setting up to compute one. The `VectorBackend` docstring's warning against per-batch
fitting is about *distances*, which this module does not compute.

**D9 — `fallback_used` can never be `True`.** `ReprBackend.fallback_variant` is declarative:
`isalgraph_ref.py`'s `encode` (`:196-212`) is the single shared path, has no `except`, and never
reads the attribute; the module docstring is explicit that the campaign driver, not the backend,
applies D14. This runner does not apply D14 either, so the field is `None` where no fallback is
declared and `False` where one is. The validator **rejects** a `True`, because it would mean a
substituted encoding was timed as if it were the requested one — with a `seconds` that is
exhaustive-time plus pruned-time and a `length_chars` from a different algorithm, in precisely
the high-`|Aut|` cells the cost law is fitted on.

**A1 — track A absent.** `symmetry.py` and `families.py` do not exist on this branch, as
expected. `measure.py` imports both lazily via `importlib`: a missing `symmetry` nulls the nine
symmetry fields, logs a WARNING, and stamps `symmetry_available: false` in the shard header;
a missing `families` refuses `--source constructed` by name with `TrackAMissingError`. The smoke
run therefore used `--source cohort`, which is what the brief specified. After the merge, a
constructed run will fill the nine fields; `symmetry_fields()` additionally raises if
`resolution_record` returns a key set other than the nine CONTRACTS §2 freezes, since a silently
different key set would null a column the regression is fitted on.

**A2 — `LADDER_BASES` mapping unknown to me.** `rendered_params()` resolves the `("base", int)`
index at runtime from `families.LADDER_BASES`, so the *name* lands in the JSONL whatever the
table turns out to be; when `families` is absent the raw integer is kept rather than crashing.
Track A reports the table; the orchestrator pins it in `PROVENANCE.md`.

**A3 — grid size is never hard-coded.** Sharding is a hash of the work-unit key, so track A's
grid growing from 644 specs (plus `spider_ladder`) costs nothing here. No constant in my files
mentions a grid size.

---

## Defects found in the brief

**1. CONTRACTS §5.2 listed twelve representations, not thirteen.** It omitted
`isalgraph_canonical`, while both my brief and the design note say "thirteen". Reported to the
orchestrator, who corrected the list and confirmed that `available_backends()` returns 12 because
`size_null` carries `Capability.BASELINE` and is returned only when named. `REPRESENTATIONS`
names all thirteen.

**2. `fallback_used` cannot be `True`, and the concern it encodes does not reach this module.**
The orchestrator asked me to record `status="censored"` + `fallback_used=true` when
`isalgraph_exhaustive` falls back. It never falls back: `fallback_variant` is declarative only
(evidence in D9). Accepted by the orchestrator after independent verification.

**3. 🔴 The `t13.1` record as specified made the primary analysis impossible.** CONTRACTS §5.1
carries no field for a family's construction parameters. Every rung of one `symmetry_ladder`
shares `family`, `n_target`, `replicate`, `n` and `m` — holding `(n, m)` exactly constant across
rungs *is the design* — so `k = 0, 1, 2, …` all serialise to byte-identical addresses. Design
note rule 7 makes the ladder the **primary** evidence and requires ordering the rungs by `k`,
which is not recoverable. Re-deriving the order from `log10_aut` would be circular, since
`log10_aut` is one of the two variables in the correlation. Fixed by adding `params: str | null`.
Accepted; the orchestrator is amending CONTRACTS §5.1 at merge.

**4. CONTRACTS §5.3.3's `seconds = budget_s` rule is wrong for cap-based censoring.** A min-DFS
projection cap fires on a *count*, typically in milliseconds; writing 300.0 would inject a
fabricated 300 s into the timing distribution the cost law is fitted on. Split into
`TIME_CENSORING_KINDS` and `CAP_CENSORING_KINDS`, each enforced in both directions. Adopted by
the orchestrator under the single rule *"`seconds` is the observation time; `status` says whether
that point is a completion or a censoring point"*, which is standard right-censoring semantics.
See the note to the analysis above.

**5. `$PY -m mypy benchmarks/real_data/eval_t13_complexity/` cannot pass as written.**
`benchmarks/` and `benchmarks/real_data/` are namespace packages with no `__init__.py`, so mypy
reports *"Source file found twice under different module names"* and checks nothing. Pre-existing
and repo-wide — the same command fails on the untouched `eval_t06_figures/`. Adding
`__init__.py` alone is not sufficient; `MYPYPATH=. --explicit-package-bases` is required. The
orchestrator confirmed independently. **The command in `.claude/CLAUDE.md` should be amended.**

**6. `ISALGRAPH_THREADS` is not read anywhere in `src/isalgraph/`.** CONTRACTS §5.3.1 names it,
and I set it in the child environment and the worker as specified, but `grep -rn ISALGRAPH_THREADS
src/isalgraph/` returns nothing — the only environment variable the engine reads is
`ISALGRAPH_ENGINE`, and `_native` exposes no thread-setting API. Harmless, because the engine's
threading already defaults to 1 (`CLAUDE.md`), but the variable is decorative and the real
protection is the BLAS pinning I added alongside it. Worth striking from the contract so nobody
later believes it is load-bearing.

**7. Base-commit mismatch, benign.** CONTRACTS names base `10eae30`; my brief names `1ace4f9`,
which is what my worktree is at. `1ace4f9` is `docs(T-13): design note and wave contracts`, the
child of `10eae30` — i.e. the commit that added CONTRACTS itself. No action needed; recorded so
the discrepancy is not rediscovered.

---

## What I did not do

- **No cluster contact of any kind.** No `ssh`, no `sbatch`, no `rsync`. `sbatch --test-only` is
  therefore not run; `launch.sh --test-only` exists for the orchestrator to run on the login node,
  and `--dry-run` produced the preview above.
- **`families.py` and `symmetry.py` are not implemented**, per the prohibition. They are imported
  lazily and their absence degrades explicitly rather than silently.
- **`src/isalgraph/` is untouched** — `git diff --stat 1ace4f9..HEAD -- src/` is empty.
- **No new third-party dependency.** `measure.py` and `schema.py` use the standard library plus
  what `isalgraph.competitors` already requires.
- **The full repo suite was not re-run.** `testpaths = ["tests"]`, and everything I added lives
  under `benchmarks/`, so the 2,618/321 reference figure cannot move. I registered no backend.
- **No analysis, no figures, no tables.** Out of scope; the note above is what I owe the agent
  that writes them.
- **The ablation arms were never exercised end-to-end on the cluster scale** — only unit-tested,
  plus the real-`_native` restoration test. They are off by default (`--arms default`).
