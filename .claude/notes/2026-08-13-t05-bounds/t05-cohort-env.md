# Work log — t05-cohort-env

## Identity

| Field | Value |
|---|---|
| Agent | `t05-cohort-env` |
| Wave | `2026-08-13-t05-bounds` |
| Model / effort | Opus 5 / xhigh |
| Branch | `worktree-agent-a289eab6662e6822e` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-a289eab6662e6822e` |
| Base commit | `da9a87d` |
| Head commit | this log's own commit; four code commits listed in §3 |
| Started / finished | 2026-08-14 / 2026-08-14 |
| Status | complete |

## 1. Prompt as received

```
You are agent `t05-cohort-env`, an implementation agent in an **isolated git worktree** on your own
branch. You never see the orchestrator's conversation; everything you need is here.

This is for a *Pattern Recognition* major revision (PR-D-26-03293) due 2026-08-31. **Correctness
beats speed. An honest negative result beats a convenient one.** You are changing code that produced
a closed ticket's published census, so **byte-parity is the acceptance criterion, not a nice-to-have.**

## Standing obligations
1. Work only inside your worktree and your ownership set below. Confirm at the start that
   `git rev-parse --show-toplevel` differs from `/home/mpascual/research/code/IsalGraph`.
2. Commit in logical commits **as you go**, not at the end. Uncommitted work cannot be merged.
3. Maintain a work log at `.claude/notes/2026-08-13-t05-bounds/t05-cohort-env.md`, using the template
   at `.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md` verbatim. Commit it last.
4. Never `git push`, rebase, or merge.
5. **No Picasso.** No `ssh`, `rsync`, `sbatch`, `squeue`, `scp`.
6. You cannot ask the user anything. Message `main` with a specific question, record the assumption
   in your log, keep working. Do not block.
7. **Finding that this brief is wrong is a success.** Report it with evidence.

---

# Task: one GEDLIB environment per dataset, not per pair

## Mission
`GedlibBackend` currently rebuilds the GEDLIB environment for **every pair**: `bounds()` calls
`_fresh_env()` -> `env.restart_env()` (`ged_backends.py:~1006-1014`), then `add_nx_graph` twice, then
runs. Add a **cohort mode** in which a worker adds all N graphs of a dataset to one `GEDEnvGXL` once,
calls `init()` once, and thereafter runs pairs by **graph index**. Wire it through
`ged_exact_runner.py`. Working means: byte-identical output to today's code, and materially faster.

## Why this exists -- measured, on the cluster, today
The production runner was benchmarked on Picasso against real data:

| Dataset | pairs | measured | note |
|---|---:|---:|---|
| IAM Protein (n 31.68, n_max 96) | 161,596 | **18.58 ms/pair** on 1 core | production runner |
| T-27 bake-off, n 29.51 | - | **285 us/pair** | one env per dataset |

A ~33x gap that is **entirely per-pair environment setup**, not solving. The measured cost curve is
`n^1.12` -- near-linear where `BRANCH_FAST` is `O(n^2 D^2 + n^3)` -- because a ~1.5 ms/pair fixed cost
dominates the solver at every size in this cohort. Summed over the real bin table, the four T-05
roles cost **~810+ core-hours** as-is against **~25** with env reuse.

This was invisible in T-03, where a pair cost ~6.5 s of exact A* and env rebuild was noise. At ~100 us
of actual solving it is the whole cost.

**The orchestrator's design note already required this** (`T-05-design.md` section 5: *"Each worker holds one
`GEDEnvGXL` built once per process -- GEDLIB env construction is not free and must not be per-pair"*)
and the implementation never met it. So this is a conformance fix, not a new idea.

## The fact that de-risks the whole change -- read this before designing
**T-27's bake-off already runs in cohort mode**, and the current per-pair runner **reproduces T-27's
values byte-identically**. The orchestrator verified this on all 3,916 LINUX pairs: `BRANCH_FAST` and
`BIPARTITE` both matched T-27 with max abs diff 0.0 and identical sha256.

Therefore **the two modes are already known to agree on real data**. Your job is to preserve that,
not to discover whether it holds. If your cohort-mode output ever differs from the checksums below,
the bug is in your change -- do not rationalise it as "a different but valid GEDLIB path".

Read `benchmarks/real_data/eval_setup/ged_bound_bakeoff.py` (its `_worker_init` / cell evaluation)
for the working cohort-mode pattern. **Do not edit that file.**

## Your ownership (exclusive write access)
Create or modify ONLY:
- `benchmarks/real_data/eval_setup/ged_backends.py`
- `benchmarks/real_data/eval_setup/ged_exact_runner.py`
- `tests/unit/test_ged_backends.py`
- `tests/unit/test_ged_exact_runner.py`
- `.claude/notes/2026-08-13-t05-bounds/t05-cohort-env.md` (your log)

Everything else is read-only. **Do not touch** `ged_bound_bakeoff.py`, `ged_merge_shards.py`,
`ged_pair_index.py`, `ged_gates.py`, `ged_bounds.py`, `approx_ged_*.py`, `export_graphs*.py`,
`cohort_audit.py`, `slurm/`, or anything under `src/isalgraph/`.

## Base state
Base commit: `da9a87d` -- "fix(T-05): record the code commit that ran, not the one .git happens to
hold". Do not rebase, merge or cherry-pick.

## Design constraints -- frozen
- **Additive.** `--compute {lb,ub,both}`, `--lb-method/--lb-options`, `--ub-method/--ub-options`,
  `--record-orientations`, `--role`, `--pair-list`, `--chunk-index/--n-chunks`, checkpointing and the
  shard schema (`pair_index, ged, lb, ub, certified, seconds, meta`) all keep their current meaning.
  **Every existing test must pass with its assertions unchanged.** Say so explicitly in your log if
  you had to change one, and why.
- **The old path stays reachable.** Cohort mode is opt-in via a flag (suggest `--env-mode
  {per-pair,cohort}`, default `per-pair` so T-03's behaviour is untouched). The orchestrator will
  flip the default only after parity is proven.
- **One env per (worker process x dataset).** Build it lazily on first use, keep it for the life of
  the chunk, and make sure a `ProcessPoolExecutor`/`fork` pool does not share one across processes --
  GEDLIB state is not fork-safe after `init()`.
- **Cost model D6** `[1,1,0,1,1,0]`, `CONSTANT`, `init_option="EAGER_WITHOUT_SHUFFLED_COPIES"`.
- **Upper bounds run in BOTH orientations and take the min** (`run_method(i,j)` and `run_method(j,i)`).
  This is a proven-bound requirement, not an optimisation -- do not drop it in cohort mode.
- **`set_method`/`init_method` should be hoisted out of the per-pair loop** where GEDLIB allows it;
  measure whether it changes anything and record it.
- **The read guards stay exactly as they are.** A lower bound of 0.0 is valid and counted. An upper
  bound of 0.0 is rejected **unless** a zero-cost edit path exists (`zero_distance_is_attainable`);
  Suite 1 alone has 306,768 certified off-diagonal pairs with exact GED 0. The accessor probe on
  P4 vs C4 at init must still fire.
- **`PairResult` stays at seven fields.** `ged_gates.py` iterates `__slots__` and you do not own it.

## Environment
```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
export PYTHONPATH=~/opt/build_gedlib/graphkit-learn
cd "$(git rev-parse --show-toplevel)"
```
GEDLIB import order is load-bearing: `libraries_import` must load before `gedlibpy_gxl`, and ruff/isort
will reorder plain imports -- use `importlib.import_module`, as the existing code does. Do **not** put
`<worktree>/src` on `PYTHONPATH` and **do not import `isalgraph`**. A subagent's `cd` does not persist
between Bash calls, so prefix every command with `cd "<abs worktree path>" && ...`.

## Verification
```bash
$PY -m pytest tests/unit/test_ged_backends.py tests/unit/test_ged_exact_runner.py -q
$PY -m pytest tests/unit/ -q          # before your final commit
$PY -m ruff check benchmarks/ tests/
```

## Data
Read-only, `SANDISK=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph`:
- `$SANDISK/data/source/APPROX_GED/exported_suite2/linux.npz` -- 89 graphs, your parity fixture.
- `$SANDISK/data/source/APPROX_GED/exported_suite2/protein.npz` -- 569 graphs, n 31.68, n_max 96.
  **Your speed fixture** -- this is the dataset the 18.58 ms/pair was measured on.
- `$SANDISK/data/source/APPROX_GED/exported_suite2/mutagenicity.npz` -- 4,040 graphs, n_max 98. Use it
  to check that a 4,040-graph env builds and `init()` does not blow up in time or memory; report both.
- `$SANDISK/results/reports/T-27-ged-bound-bakeoff/data/cells/linux__{BRANCH_FAST,BIPARTITE,BP_BEAM_DET}.npz`
  -- key `value`, float64, 3,916 entries, canonical `numpy.triu_indices(89, k=1)` order.
- `$SANDISK/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/linux.npz` -- T-03's exact
  values. **Censored pairs carry `inf`, not NaN** (92 non-finite entries); select on `certified_mask`
  and filter with `np.isfinite`.

## Definition of done
1. **Parity, the criterion that matters.** Cohort mode over all 3,916 LINUX pairs reproduces, with
   **max abs diff 0.0 and identical sha256** of the float64 value array ordered by `pair_index`:

   | role | method + options | sum | sha256[:16] |
   |---|---|---:|---|
   | `lb` | `BRANCH_FAST`, `--threads 1` | 15740 | `e95b44c7edad1369` |
   | `ub` | `BIPARTITE`, `--threads 1` | 42936 | `2528fd19b98accb0` |
   | `ubs` | `BP_BEAM`, `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1` | 23984 | `ba116a0290986360` |

   These are the orchestrator's own pre-change measurements and match T-27's recorded census.
   **Also assert cohort mode == per-pair mode element-wise on the same run.** If any of this fails,
   that is the single most important thing to report -- do not paper over it.
2. **`lb <= exact <= ub`** on all 3,870 certified LINUX pairs, 0 violations, tolerance `1e-9`.
3. **Speed, measured and recorded with numbers**: us/pair for per-pair vs cohort mode on `linux`,
   `protein`, and a >=50,000-pair slice of `mutagenicity`, single worker, `time.process_time()`.
   State the realised speed-up. **If it is under 5x, say so plainly** -- that is a result, and the
   orchestrator has ~810 core-hours riding on the number.
4. **Scale**: report wall time and peak RSS for building and `init()`-ing a 4,040-graph environment.
   If it is prohibitive, say so -- that changes the whole design.
5. Both orientations still computed for upper bounds; `--record-orientations` still emits `ub_fwd`
   and `ub_rev`, and `ub == min(fwd, rev)`.
6. `--compute lb` / `--compute ub` still skip the other end in cohort mode; report us/pair for each.
7. Every existing test passes with assertions unchanged. New tests cover: cohort==per-pair equality,
   the pool not sharing an env across processes, an env rebuilt for a second dataset in the same
   worker, and the accessor probe still firing.
8. All work committed, tree clean, log written and committed.

## Explicitly out of scope
- Any Picasso interaction, SLURM script, or submission.
- Changing the merge, cross-fill, gates, exporter or sampler.
- The calibration ladder, any analysis, figure, correlation or D13 evaluation.
- Changing T-03's defaults or `ged_pair_index.py`'s chunking.
- "Improving" the bound methods or their options strings -- they are selected and frozen.

## Final message
At most 15 lines: STATUS, BRANCH, WORKTREE, HEAD, LOG, TESTS, then **lead with the parity result of
item 1 and the measured speed-up of item 3**, then anything the orchestrator must know, then anything
unfinished.
```

### 1a. Correction to the brief's stated motivation

The brief's premise — that the 33x gap between 18.58 ms/pair and 285 us/pair is *"entirely per-pair
environment setup, not solving"* — **is refuted**, and the orchestrator has withdrawn it. Recorded
here so the motivation on file is the measured one, not the one that turned out to be wrong.

The orchestrator re-measured on Picasso itself, IAM Protein, 2,000 seeded pairs, one process, the
same pairs through both paths, bare GEDLIB calls with no runner machinery:

| | us/pair | share |
|---|---:|---:|
| per-pair total | 9,502 | |
| &nbsp;&nbsp;environment setup (`restart_env` + 2 adds + `init` + `set_method`) | 2,077 | 22 % |
| &nbsp;&nbsp;`BRANCH_FAST` solve | 7,424 | 78 % |
| cohort total | 7,366 | |
| **speed-up** | **1.29x** excl. build, 1.24x incl. a 0.65 s build for 569 graphs | |
| identical output | max abs diff 0.0 | |

**The dominant cost is the solve, not environment setup.** My own local measurements (§6a) show the
same shape at a smaller absolute scale: the saving is a near-constant per-pair offset, ~50-60 us on
LINUX and ~280 us on Protein, and it is the environment rebuild and nothing else.

Two hypotheses for the original 33x gap were tested and both failed. Mine — that the 18.58 ms/pair
figure came from `--compute both` with IPFP — is wrong: the orchestrator read the metadata of the
file the cancelled Picasso job actually wrote and it records `compute=lb`,
`method=BRANCH_FAST`, `options='--threads 1'`, with `ub_matrix` all `+inf`. IPFP was never invoked.
The gap is the **size distribution**: T-27's cost probe was a deliberately narrow band, 160 graphs
with 25 <= n <= 35, while Protein's real pair population runs to n = 96 with its mass in the [30,50)
bins. T-27's own limitation 3 says so — *"the Suite-2 projections in section 5 are lower bounds on
true cost"*.

Cohort mode still earns its place: 1.29x on the expensive datasets is roughly 150 core-hours off an
~810 core-hour campaign, it is parity-exact on two independent machines, and it removes a per-pair
failure surface. It is simply not a 33x change, and no number in this log is presented as though it
were.

## 2. Understanding and plan

**Restatement of the task in my own words:** `GedlibBackend` tears down and rebuilds its GEDLIB
environment for every pair. Add an opt-in mode in which a worker adds a whole dataset to one
`GEDEnvGXL`, initialises it once, and then addresses pairs by graph index, and wire that through the
runner behind a flag whose default leaves T-03's behaviour untouched. The output must be
byte-identical to what the current code produces.

**Approach chosen:** factor the bound evaluation out of `bounds()` into one private `_bracket(env,
i0, i1, zero_ok)` and let both modes call it. Parity then holds *by construction* rather than by
inspection: the read guards, the two upper-bound orientations, the lower-bound symmetry probe and
the inverted-bracket check exist exactly once, and neither mode can drift from the other without the
other changing too. The only thing the two modes do differently is decide where `i0` and `i1` come
from — a freshly rebuilt two-graph environment, or a standing N-graph one.

**Alternatives considered and rejected:**

- *Duplicate the bracket logic in a separate `bounds_by_index`* — rejected because the guards are the
  part most likely to be quietly weakened in a copy, and the brief is explicit that the read guards
  stay exactly as they are. Two copies of a guard is one guard.
- *Make cohort mode the default and keep `per-pair` as the escape hatch* — rejected: the brief freezes
  the default at `per-pair` so the orchestrator flips it only after parity is proven, and T-03's
  closed census depends on the default.
- *Have the runner detect cohort capability by probing for `pair_by_index`* — rejected because a
  backend built in per-pair mode would then be silently switched. The runner keys on the backend's
  declared `env_mode` instead, and raises if a backend declares cohort mode but cannot serve it.
- *Look the graph up by object identity inside `pair()` so the runner needs no change* — rejected as
  fragile; an index is what the pair index already means.
- *Hoist `set_method`/`init_method` in per-pair mode too* — rejected. `restart_env()` invalidates the
  configuration on every pair there, so hoisting would be both wrong and a change to the frozen path.
  The per-pair call sequence is left byte-for-byte as it was.

**Plan as executed:**

1. Confirm worktree isolation; read `ged_backends.py`, `ged_exact_runner.py`, both test files, and
   the working cohort pattern in `ged_bound_bakeoff.py` (read only, never edited).
2. Add `ENV_MODES`, `env_mode`/`hoist_methods` to `GedlibBackend`, `load_cohort`,
   `bounds_by_index`, `pair_by_index`, and refactor `bounds()` onto the shared `_bracket`.
3. Wire `--env-mode` through the runner: `_backend_options`, `_load_cohort`, `_compute_one(...,
   by_index=)`, `_init_worker`, the single-worker path in `run_chunk`, and the shard meta.
4. Verify parity against real GEDLIB on all 3,916 LINUX pairs, in both modes and against T-27.
5. Check containment against T-03's exact values.
6. Measure speed and scale.
7. Write tests; run the full unit suite; write this log.

**Deviations from the plan:** two.

- I expected to have to defer the real-GEDLIB parity check as unrunnable outside Picasso. GEDLIB is
  in fact installed on this workstation at `~/opt/build_gedlib/graphkit-learn`, so every parity,
  containment, speed and scale number below is a real measurement against the real library, not an
  extrapolation from a fake.
- My first speed pass was methodologically wrong and I threw it away. See §6.

## 3. Changes made

**Created** — none. Every change is to a file that already existed.

**Modified**

| Path | Change | Reason |
|---|---|---|
| `benchmarks/real_data/eval_setup/ged_backends.py` | `ENV_MODES`; `env_mode`/`hoist_methods` constructor arguments and properties; `load_cohort`, `bounds_by_index`, `pair_by_index`; `_zero_ok` and `_bracket` extracted; `_fresh_env` now clears the cohort and the configured method; `_run` hoists `set_method`/`init_method` in cohort mode; `specification()` records `env_mode` | one environment per dataset instead of per pair, with both modes sharing one bracket implementation so they cannot drift |
| `benchmarks/real_data/eval_setup/ged_exact_runner.py` | `--env-mode {per-pair,cohort}` (default `per-pair`); `_load_cohort`; `_WORKER_BY_INDEX`; `_compute_one(..., by_index=)`; cohort loaded in `_init_worker` and in the single-worker path of `run_chunk`; `env_mode` in `_backend_options` and in the shard meta | expose the mode without changing the meaning of any existing flag |
| `tests/unit/test_ged_backends.py` | `graph_values` table on the fake env; `TestCohortMode` (16 tests); two cohort accessor-probe tests | cover cohort==per-pair and every guard the change could plausibly bypass |
| `tests/unit/test_ged_exact_runner.py` | `_CohortBackend`; `TestCohortEnvMode` (8 tests); `TestCohortIsBuiltPerProcess` (2 tests) | cover the wiring, and that no environment crosses a `fork()` |

**No existing assertion was changed or removed.** The only edit to existing test code is additive:
`_FakeEnv._value` gained a `graph_values` lookup that is consulted only when a behaviour dict
supplies that key, so every pre-existing behaviour takes the identical path it took before. One
*new* test of mine was split into two — see §5.

**Commits**

| SHA | Message |
|---|---|
| `0338834` | `feat(T-05): one GEDLIB environment per dataset instead of per pair` |
| `0e2c3ca` | `feat(T-05): wire --env-mode through the runner, defaulting to per-pair` |
| `bdb8d73` | `test(T-05): cohort mode equals per-pair mode, pair for pair` |
| `56ad1df` | `test(T-05): the runner addresses a cohort backend by graph index` |
| (final) | `docs(notes): t05-cohort-env work log` |

## 4. Tests

**Tests created or extended**

| Test | File | What it verifies | Why it matters |
|---|---|---|---|
| `test_cohort_equals_per_pair_on_every_pair` | `test_ged_backends.py` | every pair agrees on both ends, and the two backends end with identical `stats` | the acceptance criterion, in a form that runs without GEDLIB |
| `test_the_environment_is_built_once_for_the_whole_cohort` | `test_ged_backends.py` | no `restart_env`, `init` or `add_nx_graph` during the run | the defect being fixed; without this the mode could be a no-op |
| `test_set_method_is_hoisted_out_of_the_pair_loop` | `test_ged_backends.py` | one `set_method` and one `init_method` for a single-role campaign | the hoist the brief asked for, asserted rather than assumed |
| `test_the_hoist_can_be_switched_off_and_changes_no_value` | `test_ged_backends.py` | hoisted and unhoisted give identical values | whether re-configuring per pair matters is measured, not assumed |
| `test_both_upper_bound_orientations_are_still_run` | `test_ged_backends.py` | `run_method(i,j)` and `run_method(j,i)`, and `ub == min(fwd, rev)` | a proven-bound requirement cohort mode must not drop |
| `test_a_second_dataset_replaces_the_first_in_the_same_worker` | `test_ged_backends.py` | the previous cohort's graphs are gone and its indices are refused | a stale graph id is silent corruption, not an error |
| `test_the_probe_discards_the_cohort_rather_than_leaving_stale_ids` | `test_ged_backends.py` | `probe_accessors` empties the cohort and says so | the probe empties the env; pretending otherwise would mis-pair every graph |
| `test_the_probe_passes_in_cohort_mode`, `test_a_wrong_accessor_still_fires_the_probe_in_cohort_mode` | `test_ged_backends.py` | the P4/C4 probe still fires | the one guard a "build the environment differently" change could bypass |
| `test_the_read_guards_are_unchanged_in_cohort_mode`, `test_a_legal_zero_is_still_accepted_in_cohort_mode` | `test_ged_backends.py` | an impossible zero raises, a legal zero does not | 306,768 certified pairs at exact GED 0 must survive; a zero from the wrong accessor must not |
| `test_pair_by_index_returns_a_seven_field_result` | `test_ged_backends.py` | `PairResult` is still seven fields | `ged_gates.py` iterates `__slots__` |
| `test_the_cohort_is_loaded_once_and_pairs_go_by_index` | `test_ged_exact_runner.py` | one load per chunk, and the indices the pair index names | a transposition here fills the matrix with other pairs' distances while staying symmetric and finite |
| `test_cohort_and_per_pair_produce_the_same_shard` | `test_ged_backends.py`'s criterion at runner level | shard arrays equal, array for array | the wiring could reorder or mis-key rows even with a correct backend |
| `test_a_cohort_backend_that_cannot_serve_it_raises_rather_than_falls_back` | `test_ged_exact_runner.py` | a broken cohort backend raises | a silent fallback runs the campaign at the cost it was meant to avoid, undetectably |
| `test_the_parent_never_builds_a_backend_for_a_pool_run` | `test_ged_exact_runner.py` | with `workers > 1` the parent never calls `make_backend` | GEDLIB state is not fork-safe after `init()` |
| `test_each_worker_loads_its_own_cohort` | `test_ged_exact_runner.py` | `_init_worker` loads the cohort in the calling process | one env per (process, dataset) |
| `test_the_reference_stub_is_untouched_by_the_flag` | `test_ged_exact_runner.py` | a backend that declares no `env_mode` is byte-identical with and without the flag | the flag must not perturb anything that did not opt in |

**Coverage of the behaviour that matters:** both environment modes over the same pairs; the hoist on
and off; both upper-bound orientations; a cohort swap within one backend; the probe before and after
a cohort load; both zero guards; the runner's index addressing, its refusal to fall back, its shard
metadata; and the absence of any parent-side backend construction on the pool path.

The cohort double in the runner tests **refuses `pair()` outright.** That is deliberate: a runner
that quietly kept passing graph objects would still produce a correct shard against a real GEDLIB
backend, at exactly the per-pair cost this change exists to remove, and no assertion on the values
would notice.

**Not tested, and why:**

- *A real `ProcessPoolExecutor` running GEDLIB in cohort mode across several processes.* The fork
  guarantee is asserted structurally (the parent never constructs a backend) rather than by forking
  a real GEDLIB environment, because a fork-safety violation manifests as a crash or as silent
  corruption inside the C++ library, which a unit test cannot distinguish from a passing run. The
  structural property is the one that makes the violation impossible in the first place.
- *`--record-orientations` under cohort mode end to end.* `_last_ub_orientations` is set inside the
  shared `_bracket`, so it is mode-independent by construction, and `_compute_one` reads it on the
  same path in both modes. Covered indirectly by `test_both_upper_bound_orientations_are_still_run`.
- *Datasets other than the three named fixtures.* Out of scope.

## 5. Test results

**Command:** `$PY -m pytest tests/unit/ -q`

```
====================== 1189 passed, 47 skipped in 46.12s =======================
```

**Result:** 1189 passed, 0 failed, 47 skipped · **Duration:** 46 s · **Run at:** `56ad1df`

Reference, measured by checking the four owned files out at `da9a87d` and re-running the same
command in the same worktree:

```
====================== 1159 passed, 47 skipped in 46.08s =======================
```

+30 tests, identical skip count, no failures either side.

Per file: `test_ged_backends.py` 96 passed (was 79), `test_ged_exact_runner.py` 61 passed (was 48).

**Failures and their resolution:** three, all mine, all resolved.

1. `test_cohort_equals_per_pair_on_every_pair` failed with `(2.0, 11.5) != (1.0, 10.0)`. **My test
   fixture's fault, not the code's.** The fake env's behaviour table keyed on GEDLIB *ids*, and a
   cohort environment addresses graph 3 as id 3 where a per-pair environment addresses the same
   graph as id 1. An id-keyed fake therefore reports a difference the real library does not have,
   and the comparison would have been vacuous either way. Fixed by adding a `graph_values` table
   keyed on the two graphs. Passes.
2. `test_the_probe_discards_the_cohort_rather_than_leaving_stale_ids` failed because the cohort
   behaviour's lower bound is a function of node count and returns 5.0 on P4 vs C4, so the probe
   correctly refused it. Fixed by using a behaviour whose P4/C4 probe passes. Passes.
3. **Three failures in `tests/unit/test_ged_bound_bakeoff.py`** — a file I do not own — with
   `AttributeError: module 'gklearn.gedlib.gedlibpy_gxl' has no attribute '__file__'`. They passed
   in isolation and failed only in a full-suite run, and they do not fail at `da9a87d`, so they were
   **caused by my change**. Cause: one of my new tests called the `fake_gedlib` fixture *twice*
   inside a single test. On the second call `monkeypatch.delitem` records the *fake* module installed
   by the first call as the value to restore, so monkeypatch's undo re-inserts a fake `gklearn` into
   `sys.modules` after the fixture's own teardown has cleaned up. The leak made the bake-off's
   availability check stop skipping and then fail on a module with no `__file__`. Fixed by splitting
   that test into two, each installing the fake once. Full suite green afterwards, with the same 47
   skips as the base commit — which is the number that proves the bake-off tests are skipping again
   rather than passing for a new reason.

## 6. Verification beyond unit tests

GEDLIB is installed on this workstation at `~/opt/build_gedlib/graphkit-learn`, so everything in this
section is a measurement against the real library. No Picasso interaction of any kind took place.

### Parity — item 1

All 3,916 LINUX pairs, both modes, fresh backend per mode, `pair_from_index` order (asserted equal to
`numpy.triu_indices(89, k=1)` in the script).

| role | cohort vs per-pair max abs diff | sum | expected sum | sha256[:16] | expected | vs T-27 cell |
|---|---:|---:|---:|---|---|---:|
| `lb` (`BRANCH_FAST`, `--threads 1`) | **0.0** | 15740 | 15740 | `e95b44c7edad1369` | `e95b44c7edad1369` | **0.0** |
| `ub` (`BIPARTITE`, `--threads 1`) | **0.0** | 42936 | 42936 | `2528fd19b98accb0` | `2528fd19b98accb0` | **0.0** |
| `ubs` (`BP_BEAM`, det. string) | **0.0** | 23984 | 23984 | `ba116a0290986360` | `ba116a0290986360` | **0.0** |

The per-pair sha256 equals the cohort sha256 for every role, so element-wise equality holds in both
directions and against the recorded census.

### Containment — item 2

`ged_matrix` and `certified_mask` from
`GED_PRECOMPUTED/extended_merged_exact_ged/computed/linux.npz`, reduced to the upper triangle.

| quantity | value |
|---|---:|
| certified pairs | **3,870** |
| non-finite `ged` entries in the triangle | 46 |
| `lb > exact` violations (tol 1e-9) | **0** |
| `ub_BIPARTITE < exact` violations | **0** |
| `ub_BP_BEAM < exact` violations | **0** |

(The brief says 92 non-finite entries; the *upper triangle* holds 46, i.e. 92 counting both
triangles of the symmetric matrix. Certified count matches the brief exactly.)

### Speed — items 3 and 6

**First pass discarded.** It timed per-pair mode on a shorter prefix of the canonical pair order than
cohort mode. A prefix of the upper triangle is not a uniform sample of it — the first k pairs all
involve the first few graphs — so the two modes were being compared on different graph-size
distributions. The numbers below use a single seed-42 uniform sample of the whole triangle, walked
identically by both modes, single worker, `time.process_time()`.

**All figures below are this workstation, one core, `time.process_time`, seed-42 uniform sample of
the whole upper triangle, both modes walking the identical sample.** Absolute numbers do not transfer
to Picasso — the orchestrator measured 9,502 us/pair for `lb` on Protein where I measure 640 — but
the *decomposition* does, and it is the same on both machines.

| dataset | N | pairs | n̄ | n_max | sampled | role | per-pair us | cohort us | saved us | speed-up |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| linux | 89 | 3,916 | 8.71 | 10 | 3,916 | `lb` | 69.7 | 12.1 | 57.6 | **5.77x** |
| linux | | | | | 3,916 | `ub` | 99.0 | 47.3 | 51.7 | **2.09x** |
| linux | | | | | 3,916 | `ubs` | 246.5 | 185.4 | 61.1 | **1.33x** |
| linux | | | | | 2,000 | `both`† | 18,859.3 | 19,041.4 | −182.1 | **0.99x** |
| protein | 569 | 161,596 | 31.68 | 96 | 5,000 | `lb` | 639.7 | 356.7 | 283.1 | **1.79x** |
| protein | | | | | 5,000 | `ub` | 1,078.1 | 791.6 | 286.5 | **1.36x** |
| protein | | | | | 5,000 | `ubs` | 3,502.7 | 3,224.1 | 278.6 | **1.09x** |
| mutagenicity | 4,040 | 8,158,780 | 28.53 | 98 | **50,000** | `lb` | 390.4 | 200.8 | 189.6 | **1.94x** |
| mutagenicity | | | | | **50,000** | `ub` | 658.4 | 475.2 | 183.3 | **1.39x** |
| mutagenicity | | | | | **50,000** | `ubs` | 2,434.3 | 2,251.7 | 182.6 | **1.08x** |

† `both` = two-sided, `BRANCH_FAST` + `IPFP --randomness PSEUDO --initial-solutions 10`. Included
only to show that a mode change buys nothing when the solve dominates by two orders of magnitude.

**The realised speed-up is single-digit, and for two of the three roles it is under 1.5x.** Stated
plainly, as the brief asked. The best case is `lb` on the small LINUX cohort at 5.77x, and that is
the least expensive role on the least expensive dataset — i.e. the case where the saving matters
least in absolute core-hours.

**The saving is a per-pair constant, and it is exactly the environment rebuild.** Timing the rebuild
alone — `restart_env`, two `add_nx_graph`, `set_edit_cost`, `init`, no solve at all, warmed up —
gives:

| dataset | bare environment rebuild | observed saving, across all roles |
|---|---:|---|
| protein | **276.4 us/pair** | 283.1 / 286.5 / 278.6 |
| mutagenicity | **185.1 us/pair** | 189.6 / 183.3 / 182.6 |

Agreement to within 3 %, and the saving is flat across roles whose totals span a factor of six. That
is the decomposition, independently reproduced: cohort mode removes the environment rebuild, nothing
more and nothing less, and whether that is 2x or 1.08x depends entirely on how expensive the solve
is next to it. (The LINUX rebuild figure from the same script is unusable — it ran first and absorbed
the one-off GEDLIB module import over its 2,000 iterations. Its true rebuild cost is the ~57 us the
role rows agree on, consistent with its much smaller graphs.)

**Cohort build cost, amortised.** Building the 4,040-graph Mutagenicity environment took 0.37-0.39 s,
measured three times. Over 50,000 pairs that is 7.6 us/pair against a 183-190 us/pair saving; over
the full 8.16 M-pair triangle it is negligible. The build never threatens the saving.

### Scale — item 4

Building and `init()`-ing the whole 4,040-graph Mutagenicity cohort in one environment:

| quantity | value |
|---|---:|
| graphs | 4,040 |
| `load_cohort` wall time (adds + `set_edit_cost` + `init`) | **0.382 s** |
| peak RSS before | 247.9 MiB |
| peak RSS after | 367.3 MiB |
| delta | **+119.4 MiB** |

**Not prohibitive.** One env per worker process costs ~120 MiB on the largest Suite-2 cohort, so a
64-way node would pay ~7.6 GiB for the environments — worth stating in the SLURM request, but far
from a design problem.

### 6b. Is T-03's recorded upper bound reproducible? — requested by the orchestrator

T-03 ran `--ub-method IPFP --ub-options "--threads 1"`. At that string IPFP inherits GEDLIB's default
`--randomness REAL`. I first met this as a flaky accessor probe (10/40 fresh backends failed the
P4/C4 check in per-pair mode, 5/40 in cohort mode — pre-existing and mode-independent, confirmed by
running both modes 40 times each). The orchestrator asked me to turn the anecdote into a measured
statement about a closed ticket's published intervals.

**Method.** Read-only. Five repetitions, each from a *fresh* backend and therefore a fresh
environment, exactly as a fresh worker process would be. Values are `bounds()[1]`, i.e. the
min over both orientations — the same quantity T-03 recorded. Compared against the `ub_matrix` in
`data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/{linux,aids}.npz`, whose
`graph_ids` I asserted equal, in order, to the graph export used. **Nothing T-03 owns was written or
edited.**

**Reproducibility across five fresh runs**

| dataset | pairs | option string | pairs whose value changed | max spread | 5 runs identical |
|---|---:|---|---:|---:|---|
| LINUX | 3,916 (all) | `--threads 1` (T-03's default) | **2,904 — 74.2 %** | 10 | no |
| LINUX | 3,916 (all) | `--threads 1 --randomness PSEUDO --initial-solutions 10` | **0 — 0.0 %** | 0 | **yes** |
| LINUX | 46 (censored only) | `--threads 1` | **33 — 71.7 %** | 4 | no |
| LINUX | 46 (censored only) | frozen PSEUDO string | **0 — 0.0 %** | 0 | **yes** |
| AIDS | 400 censored, seed 42 | `--threads 1` | **328 — 82.0 %** | 6 | no |
| AIDS | 400 censored, seed 42 | frozen PSEUDO string | **0 — 0.0 %** | 0 | **yes** |
| AIDS | 400 certified, seed 42 | `--threads 1` | **326 — 81.5 %** | 6 | no |
| AIDS | 400 certified, seed 42 | frozen PSEUDO string | **0 — 0.0 %** | 0 | **yes** |

Run-to-run sums, LINUX all pairs at T-03's default: 23668, 23702, 23638, 23622, 23688. At the frozen
string: 21326 five times. AIDS censored sample: 3299, 3307, 3313, 3253, 3293 against 2973 five times.

This corroborates T-27 §4.2, which measured GEDLIB's LS-based upper bounds changing on 91.5-93.6 % of
pairs between runs at library defaults. **T-03's default upper-bound configuration is not
reproducible. The frozen role string is, exactly.**

**How far the consequence actually reaches — the part that limits the damage**

`ExactPlusBoundsBackend.pair()` sets `ub = min(ub_gedlib, A*_best_cost)`. On a **certified** pair the
A* cost *is* the optimum and is therefore always the tighter of the two, so the recorded `ub` equals
the recorded exact value and IPFP's own number never survives into the file. I checked this rather
than assumed it:

| dataset | `ub_matrix == ged_matrix` on every certified pair | certified | censored |
|---|---|---:|---:|
| LINUX | **True** | 3,870 | 46 |
| AIDS | **True** | 234,258 | **61,038** |

So the exposure is exactly the **D11 interval-censored pairs**, and only their **upper** end. Their
lower end is `BRANCH_FAST`, which was 5/5 identical in every run I made, and their exact value is
`inf` by construction. Recomputing the censored pairs today reproduces T-03's recorded upper end on
only:

| dataset | censored pairs compared | differ from T-03's `ub_matrix` | max abs diff |
|---|---:|---:|---:|
| LINUX | 46 | **15 — 32.6 %** | 4 |
| AIDS | 400 of 61,038 | **212 — 53.0 %** | 6 |

**Statement of the defect.** T-03's census is sound wherever it is certified — 234,258 of 295,296
AIDS pairs and 3,870 of 3,916 LINUX pairs — because those upper ends come from A*, not from IPFP.
The **61,038 AIDS and 46 LINUX interval-censored pairs carry an upper end that a rerun reproduces
only about half the time**, because it was produced by a randomised heuristic seeded from a
non-deterministic source. Any interval width, censoring analysis or bound-quality figure computed
from those intervals inherits that irreproducibility. Recomputing them under the frozen
`--randomness PSEUDO` string would fix it and is cheap — 61,038 pairs, not 21.7 M — but it is T-03's
file and outside my ownership, so I have not touched it.

**Caveats.** LINUX has only 46 censored pairs, so its percentages are noisy; AIDS is the load-bearing
measurement. The AIDS graphs came from `data/exported/aids.npz`, whose 769 `graph_ids` I verified
identical and in order to the census's — the two Suite-2 AIDS exports (`aids_iam`, `aids_graphedx`)
have different graph sets and would have silently mis-paired everything.

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| Real data | `scratchpad/parity.py` against real GEDLIB | 3,916 LINUX pairs x 3 roles x 2 modes | pass; max abs diff 0.0, sha256 matches on all three |
| Real data | `scratchpad/ipfp_repro.py`, `scratchpad/ipfp_aids.py` | 5 fresh runs x 2 option strings x LINUX (3,916) and AIDS (800 sampled) | §6b; T-03's default irreproducible, frozen string exact |
| Real data | containment against T-03's exact census | 3,870 certified pairs | pass; 0 violations at 1e-9 |
| Edge cases | second cohort in one backend; probe after load; index out of range | unit tests | pass |
| Failure paths | cohort backend with no `load_cohort`; `load_cohort` raising | `RunnerError` in both | pass |
| Scale / performance | 4,040-graph env; three fixtures both modes | table above and §6a | pass |
| Environment | Debian 12, Python 3.11.15, conda `isalgraph-cpp`, GEDLIB via `graphkit-learn` at `~/opt/build_gedlib`, single core | | |

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off:**

- *One shared `_bracket` rather than two implementations* — costs a slightly larger diff to a frozen
  file; buys parity by construction and one copy of every guard.
- *Hoisting `set_method`/`init_method` only in cohort mode* — costs a branch in `_run`; buys a
  per-pair path that is byte-for-byte what T-03 ran, so the frozen default cannot regress.
- *The runner keys on the backend's declared `env_mode`, and raises rather than falling back* —
  costs a hard failure where a fallback would have "worked"; buys the guarantee that a cohort
  campaign never silently runs at per-pair cost, which is the failure this whole change exists to
  prevent and which no assertion on the values would catch.
- *`load_cohort` is called after `probe_accessors`, and `probe_accessors` discards the cohort* —
  costs one redundant environment rebuild per worker; buys the impossibility of a stale graph id.

**Assumptions I proceeded on:**

- That adding an `env_mode` key to `specification()` and to the shard `meta` is safe. Checked: no
  code outside `ged_backends.py` calls `specification()`, and neither `ged_merge_shards.py` nor
  `ged_gates.py` validates the meta against a fixed key set. If a gate does compare meta key sets,
  that key must be allowed.
- That `--env-mode` belongs in `_backend_options` (which only fires for `--backend gedlib`) rather
  than as a separate `BackendSpec` field. This keeps every non-GEDLIB backend's construction
  bit-identical to before.

**Open questions for the orchestrator:** see §10; the speed result is the one that needs a decision.

## 8. Coordination

**Messages sent:** one to `main`, mid-task, reporting (a) exact parity on all three roles, (b) that
the brief's causal diagnosis of the 33x gap appears to be wrong and the realised speed-up is well
under 5x, (c) the non-determinism of T-03's default IPFP option string, and (d) the scale numbers.
No reply required to continue, so I did not block. A second message on completion with the final
tables.

**Messages received and how they changed the work:** one, from the orchestrator, after my interim
report. It (a) confirmed parity independently on Picasso, (b) withdrew the brief's "entirely per-pair
environment setup" premise and supplied the Picasso decomposition now recorded in §1a, (c) refuted
*my* competing explanation as well — the cancelled job's metadata says `compute=lb`,
`method=BRANCH_FAST`, `ub_matrix` all `+inf`, so IPFP was never invoked and the real cause is the
size distribution, T-27 limitation 3 — and (d) asked me to quantify the IPFP non-determinism against
T-03's published intervals before finishing. That last item is new work and became §6b; it is the
only change to scope. Everything else was already done as briefed.

**Contracts I depend on and confirmed unchanged:** `PairResult` is still exactly seven slots
(asserted by the pre-existing `test_contract_b_still_has_exactly_seven_fields`, which passes
unchanged); the six frozen CONTRACT C shard arrays are unchanged when `--record-orientations` is off
(asserted by the pre-existing orientation tests, which pass unchanged); `ged_pair_index` chunking is
untouched.

## 9. Deliberately not done

- Any Picasso interaction, SLURM script, or submission — forbidden by the brief. All measurements are
  local, on this workstation's single core, and the absolute numbers will differ on Picasso hardware.
  The *ratios* are what transfer.
- Flipping the default to `cohort` — the brief reserves that for the orchestrator, after parity.
  Parity is now proven; the flag is ready.
- Changing T-03's default `--ub-method IPFP --ub-options "--threads 1"`, despite finding it
  non-deterministic (§10). The options strings are frozen and explicitly out of scope.
- Touching `ged_bound_bakeoff.py`, the merge, the gates, the exporter or the sampler.
- Fixing the three `test_ged_bound_bakeoff.py` tests in any way other than removing the leak I
  introduced. They now skip exactly as they did at `da9a87d`.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| **T-03's 61,038 censored AIDS pairs carry an irreproducible upper end** | **high** | §6b. IPFP at T-03's `--threads 1` changed value on 82.0 % of a 400-pair censored AIDS sample across five fresh runs, and a rerun reproduces T-03's recorded `ub_matrix` on only 47 % of them. Certified pairs are unaffected — their `ub` is the A* optimum, verified `ub_matrix == ged_matrix` on all 234,258. Recomputing just the censored pairs under the frozen `--randomness PSEUDO --initial-solutions 10` string would fix it and costs 61,038 pairs, not 21.7 M. **Not done: T-03's file, outside my ownership.** | orchestrator |
| **T-03's default `--ub-options "--threads 1"` makes IPFP non-deterministic at init too** | medium | The same defect surfacing earlier: the P4/C4 accessor probe failed 10/40 fresh backends in per-pair mode and 5/40 in cohort mode, so a `--compute both` campaign at defaults can abort at init on a spurious probe failure. The frozen role strings with `--randomness PSEUDO` were 40/40 clean and 5/5 bit-identical. | orchestrator |
| The realised speed-up is single-digit, not 33x | medium | §1a and §6a. Withdrawn by the orchestrator after his own Picasso measurement (1.29x on Protein); my local figures agree in shape. Recorded so the number on file is the measured one. No decision outstanding — cohort mode stays, opt-in. | — |
| Cohort mode holds N graphs per worker process | low | +119 MiB for 4,040 graphs; a 64-way node pays ~7.6 GiB for environments alone. Worth putting in the `--mem` request. | slurm owner |
| `probe_accessors` after `load_cohort` silently discards the cohort | low | Deliberate and tested, but it means a future caller that probes mid-run gets `no cohort is loaded` rather than wrong numbers. That is the intended failure mode. | — |
| Fake-GEDLIB fixture re-entrancy | low | Calling `fake_gedlib` twice in one test leaks a fake `gklearn` into `sys.modules` past teardown and breaks unrelated tests. The fixture could be hardened; I only avoided the pattern. | next wave |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | Cohort == per-pair and == the three checksums on all 3,916 LINUX pairs | **yes** | §6, parity table: max abs diff 0.0, all three sums and sha256 exact, T-27 diff 0.0 |
| 2 | `lb <= exact <= ub`, 3,870 certified pairs, 0 violations at 1e-9 | **yes** | §6, containment table |
| 3 | us/pair both modes on linux, protein, >=50k mutagenicity slice; state the speed-up | **yes** | §6a, all three fixtures, mutagenicity at exactly 50,000 matched pairs. **Single-digit throughout and under 1.5x for two of the three roles; stated plainly.** |
| 4 | Wall time and peak RSS for a 4,040-graph env | **yes** | §6: 0.382 s, +119.4 MiB. Not prohibitive. |
| 5 | Both orientations; `--record-orientations` still emits `ub_fwd`/`ub_rev`; `ub == min` | **yes** | `test_both_upper_bound_orientations_are_still_run`; the pre-existing orientation tests pass unchanged |
| 6 | `--compute lb` / `--compute ub` still one-sided in cohort mode, with us/pair | **yes** | §6a rows `lb` and `ub`; `test_compute_lb_makes_no_upper_bound_call` and its `ub` twin pass unchanged |
| 7 | Every existing test passes unchanged; new tests cover the four named behaviours | **yes** | §5. 1189/47 vs 1159/47 at base. No existing assertion altered. |
| 8 | All committed, tree clean, log written and committed | **yes** | §3 |
| + | (added mid-task by the orchestrator) quantify IPFP non-determinism against T-03's published intervals | **yes** | §6b: LINUX 3,916 pairs and AIDS 800 sampled pairs, 5 fresh runs, 2 option strings; read-only |

**Overall:** I am confident in the parity result — it is exact on three roles by three independent
comparisons (against per-pair mode, against the orchestrator's checksums, against T-27's cells), it
is against real GEDLIB rather than a fake, and the orchestrator has since reproduced it on Picasso.
I am confident in the containment result and in the shape of the speed decomposition, which two
machines and two people now agree on. I am **not** confident that my absolute timings transfer to
Picasso; they are 15x faster than his for the same role on the same dataset, and only the ratios and
the decomposition should be read across.

What the orchestrator should scrutinise first is **§6b**, not this change. Cohort mode is a
conformance fix that is parity-exact and modestly faster, and the risk it carries is low. The IPFP
finding is the one with consequences for a closed ticket: 61,038 AIDS pairs in T-03's published
census carry an upper end that a rerun reproduces about half the time. The certified majority is
sound and I verified that rather than assumed it, so the blast radius is bounded — but the fix
belongs to whoever owns T-03's files, and it is cheap.

Second, scrutinise the fixture leak in §5. It was mine, it was silent, and it surfaced only in a
full-suite run — the kind of defect that would have been merged if I had trusted the per-file run.
