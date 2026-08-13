# Work log — t05-runner

## Identity

| Field | Value |
|---|---|
| Agent | `wave-t05-runner` |
| Wave | `2026-08-13-t05-bounds` |
| Model / effort | `claude-opus-5` / `xhigh` |
| Branch | `worktree-agent-ab10166d8f9bb07a0` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-ab10166d8f9bb07a0` |
| Base commit | `885d98d8e6b37dfeb98c4df741510fc28d4a8615` |
| Head commit | `e69c0f4` (plus this log) |
| Started / finished | `2026-08-13` / `2026-08-13` |
| Status | complete |

## 1. Prompt as received

The delegation prompt is reproduced in full in §12 so that the operative sections here stay
readable. Two mid-task messages from `main` are recorded verbatim in §8.

## 2. Understanding and plan

**Restatement of the task in my own words:** T-27 established that a GEDLIB method name without its
options string is not a specification, because upper bounds move on 91.5–93.6 % of pairs between runs
at library defaults. The production backend passed one `--threads {n}` string to *both* bound ends,
which cannot express the four roles of CONTRACTS §3 — the two sensitivity roles carry different
strings from each other and from the defaults. My job was to make `ged_backends.py`,
`ged_exact_runner.py` and `ged_merge_shards.py` express a per-end method + options specification,
strictly additively, write the cross-fill that joins the separate role campaigns into one bracket,
and prove the result element-wise against T-27's recorded LINUX census.

**Approach chosen:** additive by construction. Every new constructor parameter and CLI flag defaults
to the value that reproduces T-03's behaviour exactly, and the reproduction gate is run against real
data rather than argued from the code.

**Alternatives considered and rejected:**

- *Store the compute mode as an eighth `PairResult` field* — rejected **after it was implemented and
  measured**, because it broke `test_ged_gates.py::test_the_payload_carries_every_field`, which
  iterates `PairResult.__slots__` and requires `ged_gates._pair_payload` to emit every field.
  `ged_gates.py` is outside my ownership and the test is correct to fire. Replaced by a derived
  property: `+inf` above means the upper end was never evaluated, `-inf` below means the lower one
  was. The sentinels already encode the mode unambiguously, so storing it was redundant.
- *Fabricate `value_fwd`/`value_rev` from the symmetrised `ub`* — rejected because it would read as a
  measurement that the two orientations agreed. Written `NaN` with the reason in metadata, and the
  gap reported to `main`. See §7.
- *Widen `ged_merge_shards.py` with a flat mode for the `ubt` role* — rejected by the orchestrator's
  ruling and independently the right call: the dense path is load-bearing for closed ticket T-03.
- *Run the accessor probe lazily inside `bounds()`* — rejected because it would change the call
  sequence every existing fake-GEDLIB test observes, and would fail the tests that deliberately
  configure a zero-returning fake. Made an explicit `probe_accessors()` method the runner calls once
  per worker at campaign init instead.
- *Blanket `0 < v < inf` assertion on every read* — rejected on the evidence in the brief and
  re-confirmed by measurement: Letter LOW is 15.5 % exact-zero pairs, so this aborts on correct data.

**Plan as executed:**

1. Read CONTRACTS.md and the three target modules; confirm the worktree is isolated.
2. Verify the local GEDLIB build works *before* designing anything that depends on it.
3. `ged_backends.py`: `lb_options`/`ub_options`, `compute` mode, lazy `zero_ok`, accessor probe.
4. `ged_exact_runner.py`: the six new CLI flags, threaded via `BackendSpec.options` into shard meta.
5. `ged_merge_shards.py`: `--ged-from`, `--role`, `--seconds-role`, the G4 zero-fraction check.
6. `approx_ged_crossfill.py`: new module, atomic three-file cross-fill.
7. The reproduction gate against T-27's `linux__*.npz` cells.
8. (Added mid-task by `main`) `approx_ged_subsample_merge.py`: the flat `ubt` join.

**Deviations from the plan:** two. The `PairResult` design was reworked after `test_ged_gates`
caught it (step 3, above). And step 8 was added by the orchestrator after I had finished steps 1–7;
I had not yet reported complete, so I took it.

## 3. Changes made

**Created**

| Path | Purpose |
|---|---|
| `benchmarks/real_data/eval_setup/approx_ged_crossfill.py` | Joins the three role campaigns into one shared bracket; derives `certified_mask` |
| `benchmarks/real_data/eval_setup/approx_ged_subsample_merge.py` | Flat `(dataset_key, pair_i, pair_j)` join for the `ubt` role |
| `tests/unit/test_approx_ged_crossfill.py` | 21 tests |
| `tests/unit/test_approx_ged_subsample_merge.py` | 18 tests |
| `.claude/notes/2026-08-13-t05-bounds/t05-runner.md` | This log |

**Modified**

| Path | Change | Reason |
|---|---|---|
| `ged_backends.py` | `lb_options`/`ub_options` replace the single `_heuristic_options`; `compute` mode; lazy `zero_ok`; `probe_accessors()`; `ROLE_SPECS`; `specification()`; `PairResult.computed` property | CONTRACTS §3, §6, §6.1 |
| `ged_exact_runner.py` | Six CLI flags; `BackendSpec.options`/`probe_accessors`; one-sided `_outcome_from_result`; `--pair-list` reads the pooled schema; role/options in shard meta | CONTRACTS §6, amendment 3 |
| `ged_merge_shards.py` | `--ged-from`/`--role`/`--seconds-role`; G4 zero-fraction check; G4 `computed` and `ged_from` parameters; `_computed_mode`, `_agreed` | CONTRACTS §7 |
| `tests/unit/test_ged_backends.py` | +20 tests, insertions only | New behaviour |
| `tests/unit/test_ged_exact_runner.py` | +14 tests, insertions only | New behaviour |
| `tests/unit/test_ged_merge_shards.py` | +9 tests, insertions only | New behaviour |

**Removed** — nothing.

**Commits**

| SHA | Message |
|---|---|
| `817d310` | `feat(T-05): per-end GEDLIB options, compute mode, lazy zero guard` |
| `f860727` | `feat(T-05): runner CLI expresses the per-role method specification` |
| `1493594` | `feat(T-05): merge writes a role file and gate 4 checks the zero fraction` |
| `d611e59` | `test(T-05): cover per-end options, compute modes, lazy guard, zero fraction` |
| `38340ef` | `feat(T-05): cross-fill joins the role campaigns into one bracket` |
| `20537eb` | `fix(T-05): derive the compute mode instead of storing it on PairResult` |
| `e69c0f4` | `feat(T-05): flat subsample join for the ubt role` |

**A bookkeeping defect in the commits, stated plainly.** `d611e59` and `38340ef` carry each other's
subject lines: `approx_ged_crossfill.py` landed in the commit labelled `test(...)` and
`test_approx_ged_crossfill.py` landed in the one labelled `feat(...)`, because `git add -A` picked up
the module while I was committing the tests. Every file is present and correct and the tree at HEAD
is right; I did not rewrite history to fix it because rebasing is forbidden by my brief. Verify with
`git show --stat d611e59 38340ef`.

`git diff --name-only 885d98d8..HEAD` returns exactly the eleven paths in the tables above.

## 4. Tests

**Tests created or extended** (63 new; 39 in the two new files, 24 appended to the three existing)

| Test | File | What it verifies | Why it matters |
|---|---|---|---|
| `test_the_four_roles_match_the_contract_verbatim` | `test_ged_backends.py` | `ROLE_SPECS` equals CONTRACTS §3 character for character | One wrong character silently changes what the paper reports |
| `test_the_two_ends_receive_their_own_strings` | `test_ged_backends.py` | `BRANCH_FAST` gets `--threads 1`, `BP_BEAM` gets `_DET_START` | The exact defect this ticket exists to fix |
| `test_the_default_is_exactly_what_t03_ran` | `test_ged_backends.py` | Default emits `--threads 1` on both ends | Guards a closed ticket's 2,081 core-hours |
| `test_compute_lb_makes_no_upper_bound_call` | `test_ged_backends.py` | No `get_upper_bound` call, `ub == +inf` | A one-sided campaign must not pay for the other end |
| `test_the_predicate_is_not_called_when_no_read_returns_zero` | `test_ged_backends.py` | `zero_distance_is_attainable` is never invoked | The whole point of §6.1: 21.7 M pairs each avoiding a VF2 call |
| `test_an_illegal_zero_upper_bound_is_still_rejected` | `test_ged_backends.py` | Deferring does not weaken the guard | A performance change must not become a correctness change |
| `test_a_method_read_through_the_wrong_accessor_...` | `test_ged_backends.py` | Probe raises when the upper accessor returns 0.00 | The failure GEDLIB reports as `0.00` and no exception |
| `test_contract_b_still_has_exactly_seven_fields` | `test_ged_backends.py` | `PairResult.__slots__` is unchanged | Pins the coupling `test_ged_gates` discovered |
| `test_only_the_vacuous_sentinel_is_admitted_on_each_side` | `test_ged_backends.py` | `-inf` below and `+inf` above only; `(inf, inf)` still raises | Keeps the original invariant intact while admitting one-sided runs |
| `test_the_defaults_are_exactly_what_t03_ran` | `test_ged_exact_runner.py` | Six CLI defaults | Same guard at the CLI layer |
| `test_the_unevaluated_end_must_be_exactly_the_sentinel` | `test_ged_exact_runner.py` | A plausible number in an unevaluated slot is refused | An unevaluated end must be unmistakable |
| `test_a_pooled_pair_i_pair_j_list_is_read_and_filtered` | `test_ged_exact_runner.py` | Amendment 3's schema, filtered by `dataset_key` | The sampler writes no `pair_index` |
| `test_a_failing_probe_aborts_the_run` | `test_ged_exact_runner.py` | A probe failure is fatal, not swallowed per pair | Otherwise the campaign writes a matrix of zeros |
| `test_an_all_zero_matrix_fails_the_gate` | `test_ged_merge_shards.py` | G4 refuses at zero-fraction 1.0 | The shape of the wrong-accessor failure |
| `test_a_zero_bound_needs_no_certificate` | `test_ged_merge_shards.py` | Zero is legal for a bound, illegal for a distance | `BRANCH_FAST` returns 0 on real pairs of distance 2 and 6 |
| `test_no_backend_field_can_reach_the_mask` | `test_approx_ged_crossfill.py` | Mask is True where every input file said False | CONTRACTS §4.1; `ANCHOR_AWARE_GED`'s false certificate |
| `test_ged_matrix_and_seconds_matrix_are_never_touched` | `test_approx_ged_crossfill.py` | Byte-identical after cross-fill; the three roles stay distinct | Otherwise `UB_SENSITIVITY` becomes a copy of `UB` |
| `test_disagreeing_graph_ids_are_refused` | `test_approx_ged_crossfill.py` | Refusal on identity, not length | A mismatch misattributes every bound and all of it looks plausible |
| `test_a_refused_crossfill_leaves_every_file_untouched` | `test_approx_ged_crossfill.py` | Atomicity | A partial rewrite is indistinguishable from a correct one |
| `test_row_order_is_the_pair_lists` | `test_approx_ged_subsample_merge.py` | Output order is the sampler's | Reproducible from seed 42 alone |
| `test_a_listed_pair_with_no_computed_value_is_refused` (+3 more) | `test_approx_ged_subsample_merge.py` | Join exactness both directions | A partial join is undetectable downstream |

**Coverage of the behaviour that matters:** both compute modes, both bound ends with distinct option
strings, the lazy guard in both branches (fires / does not fire), the probe passing and failing, the
G4 additions in both directions, cross-fill's write path and all six refusals, and the subsample
join's exactness in both directions. The GEDLIB interactions are covered twice over: against the
in-repo fake library for the call sequence and argument values, and against the **real** library on
the real LINUX cohort for the values (§6).

**Not tested, and why:**

- Multi-worker (`--workers > 1`) probe behaviour. `_probe_backend` is called from `_init_worker`, but
  the pool path is not exercised by a test; the in-process path is. The risk is that a probe failure
  inside a pool worker surfaces as a pool exception rather than the `RunnerError` I raise. Low
  severity — it is still fatal either way — but it is untested.
- The COIL-DEL / Mutagenicity scale (~30-node graphs) that motivates the lazy `zero_ok` change. Those
  cohorts are not exported locally, so its speed-up there is **unmeasured and unclaimed**.
- `--seconds-role` has no behavioural effect beyond a metadata string, and is tested only through the
  `ged_from='lb'` metadata assertion.

## 5. Test results

**Command:** `$PY -m pytest tests/unit/ -q`

```
8 failed, 929 passed, 44 skipped in 16.95s

FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_low]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_med]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_high]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[linux]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[aids]
FAILED tests/unit/test_export_graphs.py::test_real_export_all_five_totals
FAILED tests/unit/test_export_graphs.py::test_real_export_is_deterministic
FAILED tests/unit/test_export_graphs.py::test_real_aids_retains_within_split_structure
```

Per owned file: `test_ged_backends.py` 78 passed (58 pre-existing, unchanged), `test_ged_exact_runner.py`
44 passed (30 pre-existing), `test_ged_merge_shards.py` 31 passed (22 pre-existing),
`test_approx_ged_crossfill.py` 21 passed, `test_approx_ged_subsample_merge.py` 18 passed.
`test_ged_gates.py` 38 passed, untouched.

`$PY -m ruff check` clean on all ten owned files. (Repo-wide `ruff check benchmarks/` reports one
pre-existing E501 in `eval_setup/eval_setup.py:579`, which I do not own and did not touch.)

**Failures and their resolution.**

- **The 8 `test_export_graphs.py` failures are pre-existing and not mine.** Cause:
  `FileNotFoundError` on `data/source/IAM_Database/extracted/Letter/LOW` and
  `data/source/GED_PRECOMPUTED/AIDS` — missing source directories. I verified this rather than
  asserting it: `git checkout 885d98d8` (base commit) and re-ran, giving **the same 8 failures**,
  `8 failed, 32 passed`. I then returned to my branch, clean. I touch none of the files on that code
  path. Note that `IAM_Database/extracted` is the path CONTRACTS §1 explicitly says does not exist.
- **One failure I did cause and fixed**:
  `test_ged_gates.py::TestPairResultIsWhatGatesConsume::test_the_payload_carries_every_field`, from
  adding `computed` as an eighth `PairResult` field. Fixed in `20537eb` by making the mode a derived
  property; `test_ged_gates.py` now passes with no change to it or to `ged_gates.py`, neither of
  which I own. Recorded in full because it is exactly the kind of silent coupling worth knowing about.
- Three transient self-inflicted failures during development, all fixed before their commits: the
  repo's format-on-write hook stripped three newly-added imports whose uses did not yet exist
  (`indices_of_pairs`, `ROLE_SPECS`, `ged_backends_module`), and one `Edit` inserted a block into the
  middle of an existing merge test, splitting off its last assertion. The last one is the reason I
  now diff before committing test-file edits.

## 6. Verification beyond unit tests

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| **Reproduction gate (DoD 1)** | Real CLI, 4 roles × 3,916 LINUX pairs, vs T-27's cells | see table below | **PASS** |
| **Containment (DoD 2)** | `lb <= exact <= ub` vs T-03's `linux.npz` | 3,870 certified pairs; `lb > exact` on 0, `ub < exact` on 0; `max(lb−exact) = 0.0`, `min(ub−exact) = 0.0` | **PASS** |
| **Compute-mode cost (DoD 4)** | 3,916 LINUX pairs, `BRANCH_FAST`+`BIPARTITE` | both 129.6 µs/pair · lb 71.5 (1.81×) · ub 101.1 (1.28×) | **PASS, but see below** |
| **Lazy guard (DoD 5)** | 3,000-pair slices, 4 cohorts | linux 1.04× · aids 1.03× · letter_low 1.13× · letter_high 1.05× | **PASS, modest** |
| **End-to-end (DoD 7, 8)** | runner → merge → cross-fill on real LINUX | 10/10 keys, dtypes match T-03's file, ids identical, idempotent | **PASS** |
| Environment | Python 3.11.15, GEDLIB in-place at `~/opt/build_gedlib/graphkit-learn`, `numpy`, `networkx` | P₄ vs C₄ probe: `BRANCH_FAST` lb 1.0, `BIPARTITE` ub 1.0, `BP_BEAM` ub 1.0, `IPFP` ub 1.0 | verified before any design work |

**DoD 1, in full.** Element-wise over all 3,916 pairs, in canonical `numpy.triu_indices(89, k=1)`
order. `sha` is sha256 of the float64 array's `tobytes()`, truncated to 16 hex characters; the
orchestrator's independently-taken checksums are in the right-hand column.

| Role | Cell | max abs diff | n differing | my sum / T-27 sum | my sha / orchestrator's sha | µs/pair |
|---|---|---:|---:|---|---|---:|
| `lb` | `BRANCH_FAST` | **0.0** | 0 | 15740 / 15740 | `e95b44c7edad1369` / `e95b44c7edad1369` | 304 |
| `ub` | `BIPARTITE` | **0.0** | 0 | 42936 / 42936 | `2528fd19b98accb0` / `2528fd19b98accb0` | 327 |
| `ubs` | `BP_BEAM_DET` | **0.0** | 0 | 23984 / 23984 | `ba116a0290986360` / `ba116a0290986360` | 471 |
| `ubt` | `IPFP_MS` | **0.0** | 0 | 21326 / 21326 | `c6305be5fcfb461f` / (not supplied) | 19,640 |

`ubt` was not required by the brief; I ran it because it is the one randomised role and an exact
match there is the strongest available evidence that `--randomness PSEUDO` really is reproducible
across processes. It is.

**DoD 4 — the brief's premise is wrong, and the correction matters for the compute budget.** The
brief says `--compute lb` and `--compute ub` "each halve the work". They do not, and the asymmetry is
structural rather than a defect: under CONTRACTS §3 every upper bound is computed in **both
orientations** and the minimum taken, while `BRANCH_FAST` is computed in one. So a two-sided pair is
1 lower-bound solve + 2 upper-bound solves. Dropping the upper end removes 2 of 3 solves (measured
1.81×) and dropping the lower end removes 1 of 3 (measured 1.28×). **The `ub` and `ubs` campaigns
should be budgeted at roughly 78 % of a two-sided run, not 50 %.**

**DoD 5 — measured, and smaller than the contract anticipated.** 1.03×–1.13× across the four locally
exported cohorts. The predicate fires on 35/3000 (linux), 40/3000 (aids), 460/3000 (letter_low) and
133/3000 (letter_high) pairs under the lazy path — the letter_low figure, 15.3 %, independently
corroborates the 15.5 % zero-pair rate in the brief. The change is still correct and still strictly
cheaper, but the ~30-node COIL-DEL / Mutagenicity case that motivates it is **not measurable here**
because those cohorts are not exported locally, and I make no claim about it.

**End-to-end detail.** Three real role files were produced through the real CLIs and cross-filled.
All ten CONTRACTS §4 keys present with dtypes identical to T-03's `linux.npz`
(`ged/lb/ub` float64, `certified_mask` bool, `seconds_matrix` float32, `node_counts`/`edge_counts`
int32, `graph_ids`/`labels`/`metadata` `<U`); `graph_ids` identical to T-03's; `ged_matrix` and
`seconds_matrix` distinct between `UB` and `UB_SENSITIVITY`, i.e. not overwritten; a second
cross-fill leaves every array bit-identical. Metadata carried
`method=BP_BEAM, options_string='--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1'`
verbatim for the `ubs` file.

**The strongest independent check in the whole exercise.** Cross-fill's derived `certified_mask`
closes on **79 of 3,916** LINUX pairs (2.0 %). All 79 lie inside T-03's independently computed exact
set, and on all 79 the derived value agrees with T-03's exact GED **exactly, 0 disagreements**. That
is `BRANCH_FAST`'s lower bound and `BIPARTITE`'s upper bound, computed by two separate campaigns that
never saw each other, meeting on the number a completely separate A* search proved. It is the
clearest evidence available that the derived certificate of CONTRACTS §4.1 is sound.

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off**

- *The compute mode is derived from the sentinels, not stored.* Costs a little indirection; buys an
  unchanged seven-field Contract B, so `ged_gates.py` — which I do not own — needs no change, and the
  encoding cannot disagree with the declaration because there is only one of them. The runner still
  takes the mode from the *backend* as the campaign authority and cross-checks the result's encoding
  against it, so a backend that returned a one-sided bracket during a two-sided run is caught.
- *`-inf` / `+inf` as the unevaluated sentinels.* Costs a relaxation of a finiteness invariant; buys
  an unevaluated end that is impossible to mistake for a measurement. Narrowed so the relaxation is
  one-directional: `-inf` is admitted only below, `+inf` only above, `PairResult(inf, inf, …)` still
  raises exactly as the pre-existing parametrised test requires, and a result whose *both* ends are
  sentinels is refused outright as "measured nothing".
- *G4's zero-fraction check applies to every source, including `exact`.* Costs a new way for a
  previously-passing merge to fail; buys uniform protection. An exact census that came out 99 % zeros
  would be just as wrong and just as quiet. No existing test or real dataset is anywhere near the
  limit (real LINUX: 0.0105 for `lb`, 0.0000 for `ub`).
- *The certified-zero rule now applies only when `ged_matrix` holds a distance.* A zero *distance*
  claims isomorphism and needs the certificate; a zero *bound* claims nothing. Applying the old rule
  to a `BRANCH_FAST` matrix would reject correct data, since its trivial bound is 0 on real pairs
  whose exact distance is 2 and 6.

**Assumptions I proceeded on**

- That `--pair-list` should accept the sampler's pooled `dataset_key`/`pair_i`/`pair_j` schema
  directly, since it previously required a `pair_index` key the sampler does not write. Messaged to
  `main`. If instead the SLURM worker is meant to convert, my addition is harmless — `pair_index`
  still wins where both are present.
- That `probe_accessors` should default **on** in the runner while the `GedlibBackend` constructor
  itself does not probe. Any other arrangement changes the call sequence every existing fake-GEDLIB
  test observes.

**Open questions for the orchestrator** — both messaged, both with a working default shipped.

1. **CONTRACTS §5 asks for something §6.2 cannot carry.** The subsample file wants `value_fwd` and
   `value_rev`; the frozen shard carries only `ub`, already reduced to `min(fwd, rev)` inside the
   backend. The two orientations do not exist anywhere in the shard. I write `NaN` and record the
   reason. Options are (a) accept `NaN`, (b) unfreeze the shard to carry `ub_fwd`/`ub_rev`, (c) drop
   the two columns. I shipped (a) because it is the only one that neither fabricates data nor touches
   a frozen schema without authority.
2. Whether the `ub`/`ubs` compute budget should be revised from 50 % to ~78 % of a two-sided run
   (§6, DoD 4).

## 8. Coordination

**Messages sent.** One to `main` at the end: DoD item 1's result with all three checksums matching,
the two contract gaps above, the `PairResult`/`ged_gates` design correction, the pre-existing export
failures with the evidence that they pre-exist, and the commit-labelling slip.

**Messages received and how they changed the work.**

1. *Amendments 1 and 3.* Amendment 1 (`graph_ids` are the loader's native ids, not
   `{key}_{split}_{sourceid}`) required no change — I pass ids through verbatim everywhere and my
   cross-fill compares them by identity. Amendment 3 (the subsample is two files) is why
   `--pair-list` now reads the pooled schema.
2. *The `ubt` merge ruling plus three cross-track findings.* Added
   `approx_ged_subsample_merge.py` and its tests to my ownership; I built both. Of the three
   findings: (1) filtering censored pairs on `np.isfinite` rather than `np.isnan` — my gate already
   did this, and `3916 − 3870 = 46` censored matches the reported 92/2; (2) refusing on `graph_ids`
   identity rather than length — my cross-fill already checks both, with a
   `test_disagreeing_graph_ids_are_refused` test at equal length; (3) the zero-guard reading —
   preserved unchanged, comment included, and independently corroborated by my letter_low
   measurement.

**Contracts I depend on and confirmed unchanged:** §3 roles and option strings (transcribed into
`ROLE_SPECS` and asserted character-for-character), §4 ten output keys and dtypes (verified against
T-03's real file), §4.1 derived `certified_mask`, §6 CLI, §6.1 lazy `zero_ok`, §6.2 frozen shard
schema, §7 merge CLI.

## 9. Deliberately not done

- **SLURM scripts and any Picasso interaction** — `wave-t05-slurm` owns them. I made no `ssh`,
  `rsync`, `sbatch`, `squeue`, `scancel` or `scp` call of any kind.
- **The Suite-2 exporter and the sampler** — `wave-t05-export` owns them. My `--pair-list` reads the
  schema they will write, which I could only test against a fabricated file.
- **Aligning anything with `wave-t05-slurm`'s gates** — that independence is deliberate.
- **`ged_gates.py`, `ged_bound_bakeoff.py`, `ged_pair_index.py`, `ged_bounds.py`, `export_graphs.py`
  and everything else outside my ownership** — untouched, including where a change there would have
  been the more obvious fix (§2, the `PairResult` rework).
- **The calibration ladder, and any analysis, figure, correlation or bootstrap** — out of scope.
- **Rewriting history to fix the two swapped commit subjects** — forbidden by my brief; documented in
  §3 instead.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| §5 wants `value_fwd`/`value_rev`; §6.2's shard cannot carry them | **medium** | Shipped as `NaN` with the reason in metadata. If the analysis needs real per-orientation values, the shard schema must be unfrozen — a decision above my authority | orchestrator |
| `ub`/`ubs` campaigns cost ~78 % of a two-sided run, not 50 % | **medium** | Measured, not estimated (§6). A 50 % assumption under-provisions the SCBI wallclock | orchestrator |
| Probe not exercised through the multi-worker pool | low | Fatal either way, but the exception type crossing the pool boundary is untested | next wave |
| Lazy `zero_ok` unmeasured at COIL-DEL / Mutagenicity scale | low | Those cohorts are not exported locally. Worth re-measuring once they are, since that is where the contract expects the gain | next wave |
| 8 pre-existing `test_export_graphs.py` failures | low | Missing source dirs, including the `IAM_Database/extracted` path CONTRACTS §1 says does not exist. Verified present at the base commit | `wave-t05-export` |
| Two commits carry each other's subject lines | low | Content correct at HEAD; `git show --stat d611e59 38340ef` | orchestrator, if it matters for the wave summary |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | `lb`/`ub` reproduce T-27's LINUX census element-wise; `ubs` too | **yes** | §6 table: max abs diff 0.0 and identical sha256 on all four roles, including `ubt` |
| 2 | `lb <= exact <= ub` on all 3,870 certified pairs at 1e-9 | **yes** | 0 violations either side; filtered on `certified_mask & isfinite` |
| 3 | Every existing test passes with assertions unchanged | **yes** | `git diff` on the three test files is insertions only (0 deletions). One existing assertion was displaced by an Edit and restored to its original position before commit; no assertion was altered. `test_ged_gates.py` passes untouched after the §2 rework |
| 4 | `--compute lb`/`ub` each halve the work; µs/pair recorded | **partial** | Measured and recorded, but **the premise is wrong**: 1.81× and 1.28×, not 2× and 2×, for the structural reason in §6. Reported rather than smoothed over |
| 5 | Lazy `zero_ok` behaviour-identical, not invoked when bounds non-zero, speed-up measured | **yes** | `test_the_predicate_is_not_called_when_no_read_returns_zero`; 1.03×–1.13× measured, with the unmeasured case named |
| 6 | The accessor probe fires | **yes** | `test_a_method_read_through_the_wrong_accessor_...` and `test_an_infinite_read_fires_the_probe`; also fires at construction via the static table |
| 7 | Cross-fill: three files, atomic, idempotent, refuses mismatched ids, leaves `ged`/`seconds` alone | **yes** | 21 tests plus the real-data run in §6 |
| 8 | Merged output loads with the exact loader, ten keys, dtypes; G4 additions fire | **yes** | Real-data dtype comparison against T-03's `linux.npz`; `test_an_all_zero_matrix_fails_the_gate` |
| 9 | All work committed, tree clean, log written | **yes** | 7 commits + this log; `git status` clean |

**Overall.** I am most confident about item 1, because it is not an argument: four roles, 3,916 pairs
each, byte-identical sha256 against checksums the orchestrator took independently before I started,
through the real CLI against the real library on the real cohort. I am equally confident about the
derived `certified_mask`, for the reason in §6 — 79 closures, all 79 agreeing exactly with a
completely separate A* computation.

What the orchestrator should scrutinise first, in order: **(1)** the `value_fwd`/`value_rev` decision
in §7, because it is a contract conflict I resolved on my own judgement and it is the one place I
write `NaN` into a published file; **(2)** the compute-budget correction in §6, because a 50 %
assumption baked into a SLURM wallclock will under-provision; **(3)** the `PairResult` sentinel
relaxation in `__post_init__`, because it loosens a Contract B invariant, and although I narrowed it
so the pre-existing test still passes unchanged, it is the change most likely to have a consequence I
have not thought of.

What I am *not* confident about: the lazy `zero_ok` change is justified by the contract's reasoning
rather than by my measurements, which show only 1.03×–1.13× on the cohorts I can reach. It is
strictly cheaper and definitely correct, but if anyone is counting on a large speed-up from it at
Suite-2 scale, that number does not exist yet.

## 12. Prompt as received, verbatim

```
You are agent `wave-t05-runner`, an implementation agent working inside an **isolated git worktree**
on a branch of your own, in parallel with two peers who own different files. You never see the
orchestrator's conversation; everything you need is in this prompt and in the repository.

This work is for a *Pattern Recognition* major revision (PR-D-26-03293) due 2026-08-31, read by
reviewers who checked every number last round. **Correctness beats speed. An honest negative result
beats a convenient one.**

You are editing code that produced a closed ticket's results (T-03's exact-GED census, 2,081
core-hours, numbers already propagated into the plan). **Every change you make must be additive:
existing defaults and existing behaviour must be identical after your change.** A regression here
silently invalidates published numbers.

## Standing obligations
1. Work only inside your worktree. Every file you create or edit must lie inside your declared
   ownership set. Everything else is read-only reference. Confirm at the start that
   `git rev-parse --show-toplevel` differs from `/home/mpascual/research/code/IsalGraph`; if it does
   not, stop and message `main`.
2. Commit your work in logical commits **as you go**, not at the end. Sessions die; uncommitted work
   cannot be merged, because the orchestrator merges your branch, not your working tree.
3. Maintain your work log at `.claude/notes/2026-08-13-t05-bounds/t05-runner.md` from your first
   action to your last, using the template committed at
   `.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md`, and commit it as your final commit.
4. Never run `git push`, never rebase or merge, never touch a peer's branch or worktree.
5. **You have no access to Picasso.** No `ssh`, `rsync`, `sbatch`, `squeue`, `scancel`, `scp`.
6. You cannot ask the user anything. On an ambiguity, message `main` with a specific question, record
   the assumption you are proceeding on in your log, and keep working. Do not block.
7. Never change a frozen contract yourself. Propose it to `main`. **Finding that your brief is wrong
   is a success** -- report it with evidence.
8. Report failure honestly. "This does not work and here is why" beats a plausible-looking
   implementation that was never exercised.
9. Plan before editing and write the plan into your log. Implement in small verified steps. Write
   tests as you go. Run the suite before your final commit and record the real output, failures
   included.

---

# Task: make the production runner express T-27's selected specification, and emit the T-05 schema

## Mission
Extend `ged_backends.py`, `ged_exact_runner.py` and `ged_merge_shards.py` -- **additively, with every
existing default unchanged** -- so that a campaign can name a GEDLIB method *together with its options
string*, compute a single bound end, and merge into the T-05 output schema. Then write
`benchmarks/real_data/eval_setup/approx_ged_crossfill.py`, which joins the separate role campaigns
into the bracket every output file carries. Working means: run on the real 89-graph LINUX cohort, your
`BRANCH_FAST` and `BIPARTITE` values equal T-27's recorded census **element-wise on all 3,916 pairs**.

## Why this exists
T-27 selected both ends of the proven bracket by measurement against 3,836,827 certified exact GED
values, and found that **GEDLIB's upper bounds change on 91.5-93.6 % of pairs between runs at library
defaults** -- one random start under `REAL` randomness. Its conclusion: *a method name without its
options string is no longer a valid specification.* The current backend emits only `--threads {n}`
for **both** ends from one shared string (`ged_backends.py:777`). `BRANCH_FAST` and `BIPARTITE`
happen to share `--threads 1`; the two sensitivity-arm methods do not. **The backend cannot currently
express the specification the paper will print**, and that is the defect you exist to fix.

## Repository orientation
- Repository root: your worktree (`git rev-parse --show-toplevel`).
- **Read first, in this order**:
  1. `.claude/notes/2026-08-13-t05-bounds/CONTRACTS.md` -- SS3 the four roles, SS4 the output schema and
     SS4.1 why `certified_mask` is legitimate, SS6 your CLI, SS6.1 the lazy-`zero_ok` change, SS7 the
     merge CLI. **This is your specification.**
  2. `.claude/notes/review/tasks/T-05-design.md` SS1 (method specification and the read guards), SS3.2
     (schema), SS4 (gates).
  3. `benchmarks/real_data/eval_setup/ged_backends.py` -- `GedlibBackend` (:688), `module()` (:798),
     `env()`/`_fresh_env()` (:820/:834), `_read` (:843), `_run` (:889), `bounds` (:895), `pair` (:964),
     `zero_distance_is_attainable` (:402), `LOWER_BOUND_METHODS` (:148), `UPPER_BOUND_METHODS` (:151),
     `make_backend` (:1232), `BackendSpec` (:1210).
  4. `benchmarks/real_data/eval_setup/ged_exact_runner.py` -- `SHARD_KEYS` (:110),
     `_write_npz_atomic` (:584), `_pairs_fingerprint` (:638), `_target_pairs` (:762),
     `_load_checkpoint` (:803), `run_chunk` (:859), the CLI (:1087).
  5. `benchmarks/real_data/eval_setup/ged_merge_shards.py` -- `gate4` (:246), `merge_shards` (:409),
     the write block (:516-527).
  6. `benchmarks/real_data/eval_setup/ged_bound_bakeoff.py` :168-191 -- the **already-correct** option
     strings `_MULTI_START` and `_DET_START`. Read them; **do not edit that file.**
- Conventions: `CLAUDE.md` is loaded. Additionally: NumPy-style docstrings, full type annotations,
  `logging` never `print`, Python 3.11.

## Your ownership (exclusive write access)
Create or modify ONLY:
- `benchmarks/real_data/eval_setup/ged_backends.py`
- `benchmarks/real_data/eval_setup/ged_exact_runner.py`
- `benchmarks/real_data/eval_setup/ged_merge_shards.py`
- `benchmarks/real_data/eval_setup/approx_ged_crossfill.py` (new)
- `tests/unit/test_ged_backends.py`
- `tests/unit/test_ged_exact_runner.py`
- `tests/unit/test_ged_merge_shards.py`
- `tests/unit/test_approx_ged_crossfill.py` (new)
- `.claude/notes/2026-08-13-t05-bounds/t05-runner.md` (your log)

Everything else is read-only. **Do not touch** `ged_bound_bakeoff.py`, `ged_bakeoff_analysis.py`,
`ged_bounds.py`, `ged_pair_index.py`, `ged_gates.py`, `cohort_audit.py`, `iam_gxl_loader.py`,
`export_graphs.py`, or anything in `src/isalgraph/`. If `ged_pair_index.py` needs a change, message
`main` -- its chunking is T-03's and a closed ticket relies on it.

## Base state
- Base commit: `885d98d8e6b37dfeb98c4df741510fc28d4a8615`.
- Your peers branch from the same commit. Do not rebase, merge or cherry-pick.

## Frozen contracts
From `CONTRACTS.md`; code against them exactly.

- **The four roles**, CONTRACTS SS3 -- method, options string **verbatim**, accessor:
  - `lb`  = `BRANCH_FAST`, `--threads 1`, lower
  - `ub`  = `BIPARTITE`, `--threads 1`, upper
  - `ubs` = `BP_BEAM`, `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1`, upper
  - `ubt` = `IPFP`, `--threads 1 --randomness PSEUDO --initial-solutions 10`, upper
  Cost model D6 `[1, 1, 0, 1, 1, 0]`, `CONSTANT`, always.
- **New CLI flags**, CONTRACTS SS6, all with T-03's current behaviour as the default:
  `--lb-method` (BRANCH_FAST), `--lb-options` ("--threads 1"), `--ub-method` (**IPFP** -- T-03's
  default, deliberately unchanged), `--ub-options` ("--threads 1"),
  `--compute {lb,ub,both}` (both), `--role STR` (metadata only).
  `--compute lb` skips every upper-bound call and leaves `ub = inf` in the shard; `--compute ub`
  skips every lower-bound call and leaves `lb = -inf`. The inverted-bracket guard
  (`ged_backends.py:957`) is skipped when only one end is computed.
- **`GedlibBackend.__init__` gains `lb_options` and `ub_options`**, replacing the single
  `_heuristic_options` used for both.
- **Shard schema unchanged**: `pair_index` int64, `ged` float64, `lb` float64, `ub` float64,
  `certified` bool_, `seconds` float32, `meta` JSON (`SHARD_KEYS`, :110).
- **Merge CLI**, CONTRACTS SS7: `--ged-from {exact,lb,ub}` (default `exact`), `--role`,
  `--seconds-role`. Structural gate G4 gains one check: the **off-diagonal exact-zero fraction is
  recorded in metadata and a fraction >= 0.99 raises `MergeError`** -- that is the shape of the
  silent-all-zeros failure a wrong accessor produces.
- **Output schema**, CONTRACTS SS4 -- exactly the ten keys of
  `GED_PRECOMPUTED/extended_merged_exact_ged/computed/*.npz`, same dtypes, so one loader reads exact
  and approximate files alike. `metadata` JSON keys are listed in SS4.
- **`certified_mask` is derived, never self-reported** (CONTRACTS SS4.1). `GedlibBackend.pair()`
  returns `certified=False` always and **must keep doing so** -- `ANCHOR_AWARE_GED` was retracted for
  issuing a false optimality certificate. The mask is computed by your cross-fill module as
  `|lb_matrix - ub_matrix| <= 1e-9`, the derived statement *"a proven lower bound of k and an
  exhibited edit path of cost k together prove GED = k"*. Diagonal `True`.
- **Cross-fill** writes the same `lb_matrix`/`ub_matrix`/`certified_mask` into all three role files
  and **never touches** `ged_matrix` or `seconds_matrix`. Writes must be atomic.

### The read guards -- get these exactly right
- A read of `0.00` from an **upper-bound** method is the signature of the wrong accessor and must be
  rejected -- **except** where a zero-cost edit path genuinely exists, which under D6 means the graphs
  are isomorphic. Suite 1 alone holds **306,768 certified off-diagonal pairs with exact GED = 0**
  (15.5 % of Letter LOW), so a blanket `0 < v < inf` assertion aborts on correct values.
- A **lower** bound of zero is always mathematically valid (trivial, merely uninformative) and is
  counted, not rejected. The existing code is already right about this; preserve it and its comment.
- **Add**: at campaign init, an accessor probe on P4 vs C4 (true GED 1) asserting the configured
  method returns exactly 1.00 through the accessor it is being read with. This is the check that
  catches a wrong accessor before 21.7 M pairs of zeros are written.

### CONTRACTS SS6.1 -- make `zero_ok` lazy
`bounds()` calls `zero_distance_is_attainable(g1, g2, costs)` **eagerly on every pair** (:919). Under
D6 that reaches `nx.is_isomorphic` whenever `n1 == n2 and m1 == m2` -- most Letter pairs, and a VF2
call on ~30-node graphs for COIL-DEL and Mutagenicity, **21.7 M times**. Make it a zero-argument
callable evaluated **only when a read returns 0.0**. Pure performance change: the value computed and
the guard's behaviour must be identical, and a test must assert the callable is *not* invoked for a
pair whose bounds are non-zero. Measure and record the speed-up on real graphs.

## Environment bootstrap
```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
export PYTHONPATH=~/opt/build_gedlib/graphkit-learn      # in-place GEDLIB build, working today
cd "$(git rev-parse --show-toplevel)"
```
GEDLIB import order is load-bearing and **isort/ruff will break it if written as plain imports** --
use `importlib.import_module`, as the existing code already does. Do **not** put `<worktree>/src` on
`PYTHONPATH`, and **do not import `isalgraph`**. A subagent's `cd` does not persist between Bash
calls, so prefix every command with `cd "<your absolute worktree path>" && ...`.

## Verification commands
```bash
$PY -m pytest tests/unit/test_ged_backends.py tests/unit/test_ged_exact_runner.py \
              tests/unit/test_ged_merge_shards.py tests/unit/test_approx_ged_crossfill.py -q
$PY -m pytest tests/unit/ -q                       # before your final commit
$PY -m ruff check benchmarks/ tests/
```

## Data and shared resources
Read-only, under `SANDISK=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph`:
- `$SANDISK/data/exported/linux.npz` -- the exported 89-graph LINUX cohort, in the schema
  `export_graphs.py::load_exported` reads. **Your real-data fixture.**
- `$SANDISK/results/reports/T-27-ged-bound-bakeoff/data/cells/linux__BRANCH_FAST.npz`,
  `.../linux__BIPARTITE.npz`, `.../linux__BP_BEAM_DET.npz`, `.../linux__IPFP_MS.npz` -- T-27's recorded
  values, key `value` (float64, 3,916 entries, symmetrised for upper bounds), in canonical
  `numpy.triu_indices(89, k=1)` order.
- `$SANDISK/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/linux.npz` -- T-03's exact
  values, for the `lb <= exact <= ub` check.
- Your peers do not touch these paths. **No Picasso.**

## Definition of done
1. **The reproduction gate, run for real and recorded in your log**: a campaign over all 3,916 LINUX
   pairs with role `lb` reproduces `linux__BRANCH_FAST.npz`'s `value` array **element-wise**, and a
   campaign with role `ub` reproduces `linux__BIPARTITE.npz`'s `value` array **element-wise**. Also
   run role `ubs` against `linux__BP_BEAM_DET.npz`. Report max absolute difference for each -- the
   expected answer is 0.0. **If any of these fails, that is the most important thing you can tell the
   orchestrator; do not paper over it.**
2. `lb <= exact <= ub` holds on all 3,870 certified LINUX pairs at tolerance `1e-9`, against T-03's
   `linux.npz`.
3. Every existing test in `test_ged_backends.py`, `test_ged_exact_runner.py` and
   `test_ged_merge_shards.py` passes **with its assertions unchanged** -- additive means additive. Say
   explicitly in the log if you had to change an existing assertion and why.
4. `--compute lb` and `--compute ub` each halve the work: measure and record us/pair for
   `both` / `lb` / `ub` on the LINUX cohort.
5. The lazy-`zero_ok` change is behaviour-identical (a test asserts the callable is not invoked when
   bounds are non-zero) and its speed-up on real graphs is measured and recorded.
6. The accessor probe fires: a test configuring a lower-bound method with the upper accessor fails
   loudly at init rather than returning zeros.
7. `approx_ged_crossfill.py` writes `lb_matrix`, `ub_matrix` and `certified_mask` into all three role
   files atomically, leaves `ged_matrix` and `seconds_matrix` untouched, is idempotent, and refuses
   files whose `graph_ids` disagree.
8. Merged output loads with the exact-GED loader and carries the ten CONTRACTS SS4 keys with the
   stated dtypes; the G4 additions fire on a deliberately all-zero matrix.
9. All work committed on your branch; working tree clean. Work log written and committed.

## Explicitly out of scope
- Writing SLURM scripts, or any Picasso interaction. `wave-t05-slurm` owns that.
- Writing the Suite-2 exporter. `wave-t05-export` owns that.
- The calibration ladder (exact GED above n = 12). A later wave.
- Any analysis, figure, correlation, bootstrap or D13 evaluation.
- Changing T-03's defaults, `ged_pair_index.py`'s chunking, or `ged_bounds.py`.

## Work log -- mandatory
Maintain `.claude/notes/2026-08-13-t05-bounds/t05-runner.md` using the template at
`.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md` **verbatim** (read it first). Write it
continuously. Commit it last with `docs(notes): t05-runner work log`.

## Peers in this wave
- `main` -- the orchestrator. Message it for ambiguities, contract defects, blockers, or anything
  needing a decision outside your ownership. It owns every Picasso interaction.
- `wave-t05-export` -- exports the ten Suite-2 datasets and emits the subsample pair list. Owns
  `export_graphs_suite2.py`, `approx_ged_sampling.py` and their tests. Its output (CONTRACTS SS2) is
  what your runner consumes in production.
- `wave-t05-slurm` -- writes the Picasso launcher/worker pair and the **independent** validation gates.
  Owns `slurm/approx_ged/*`, `approx_ged_gates.py` and its test. It codes its gates against
  CONTRACTS, not against your implementation -- that independence is deliberate, so **do not** offer to
  align it with your code. If your CLI must differ from CONTRACTS SS6, message `main`.

## Final message format
At most 15 lines: STATUS, BRANCH, WORKTREE, HEAD, LOG, TESTS (counts + command), then three bullets
on what you built, anything the orchestrator must know, and anything unfinished. **Lead with the
result of definition-of-done item 1.**
```

### Mid-task messages from `main`, verbatim

**Message 1 — amendments 1 and 3.**

```
Orchestrator. Two CONTRACTS amendments from wave-t05-export's recon, both verified by me. Neither
should change your work, but you consume the affected schema so you are being told rather than left
to discover it.

AMENDMENT 1 — CONTRACTS §2, graph_ids. It is NOT '{key}_{split}_{sourceid}' as originally written.
It is the source loader's native id, byte-identical to what export_graphs.py writes for Suite 1.
Verified: Letter ids are bare stems ('IP1_0000', 'AP1_0001'); only GraphEdX ids look like
'linux_train_0000'. Consequence for you: none, except that this makes your CONTRACTS §4 output
graph_ids match T-03's exact-GED files exactly, which is what you want — do not normalise, rewrite or
prefix ids anywhere in the runner, the merge or the cross-fill. Pass them through verbatim. Your
cross-fill's "refuse files whose graph_ids disagree" check is unaffected and stays.

AMENDMENT 3 — CONTRACTS §5, the subsample is two files. The sampler writes
UB_TIGHT/subsample_pairs.npz ahead of the run (dataset_key, pair_i, pair_j, n_max, bin_index,
metadata); the ubt campaign reads it via --pair-list and writes its result to UB_TIGHT/subsample.npz
with value/value_fwd/value_rev/seconds added. Keeping them separate stops the campaign overwriting
its own input. If your --pair-list handling assumed one file, this is the correction.

Also amended, for completeness and no action from you: §2 no longer asserts class counts for the
labels column (the numbers there were raw dataset counts, not post-filter).

Unchanged and still binding: §3 the four roles and their verbatim options strings, §4 the ten output
keys, §4.1 certified_mask derived and never self-reported, §6 the CLI, §6.1 lazy zero_ok, §7 the
merge CLI.

Reminder on priority: definition-of-done item 1 is the one I will re-run myself. For reference, here
are the checksums I took from T-27's LINUX cells before the wave started, so you can compare directly
rather than eyeballing:
  linux__BRANCH_FAST   n=3916  sum=15740  sha256[:16]=e95b44c7edad1369  head=[4,4,3,4,2]
  linux__BIPARTITE     n=3916  sum=42936  sha256[:16]=2528fd19b98accb0  head=[11,7,7,7,6]
  linux__BP_BEAM_DET   n=3916  sum=23984  sha256[:16]=ba116a0290986360  head=[7,5,5,5,4]
(sha over v.tobytes() of the float64 'value' array.) If your campaign reproduces those three sums and
hashes, item 1 is done and I will confirm it independently.

Carry on.
```

**Message 2 — the `ubt` merge ruling and three cross-track findings.**

```
Orchestrator. One addition to your ownership set, from a contract gap that is mine, not yours.
Non-blocking if you are mid-flight — finish what you are on first.

THE GAP. CONTRACTS §7's merge CLI writes a DENSE (N,N) matrix per dataset. But the `ubt` role's
output (CONTRACTS §5, `UB_TIGHT/subsample.npz`) is a FLAT file pooled across all ten datasets —
`--n-graphs` is meaningless for it and no `--key` names one cohort. §7 simply cannot express it.
wave-t05-slurm found this while wiring its worker and was right to refuse to invent a flag.

THE RULING. A separate entry point, NOT a flat mode added to `ged_merge_shards.py`. The dense merge
path is load-bearing for T-03, a closed ticket, and I am not widening it for a 28,000-row special
case. So you additionally own:

  benchmarks/real_data/eval_setup/approx_ged_subsample_merge.py
  tests/unit/test_approx_ged_subsample_merge.py

CLI: `--shards --pair-list --out --role --method --options`. It joins shard `pair_index` against the
rows of `UB_TIGHT/subsample_pairs.npz` and writes the CONTRACTS §5 flat schema: `dataset_key, pair_i,
pair_j, n_max, bin_index, value, value_fwd, value_rev, seconds, metadata`. `value` is the symmetrised
min over both orientations, as for every upper bound. Requirements:
- Fail loudly if any pair_index in the shards is absent from the pair list, or vice versa — a silent
  partial join here is undetectable downstream.
- Preserve the pair list's row order, so the file is reproducible from seed 42 alone.
- `metadata` carries method, options_string, accessor, cost_model, bin_edges, seed, per-bin realised
  counts, code_commit, computed_utc, schema_version.
- No dense matrix anywhere in this path.

Keep it small and self-contained; it is a join, not a merge. If you have already finished and
committed, say so and I will take this myself as a scoped follow-up rather than have you re-enter.

THREE THINGS FROM THE OTHER TRACKS THAT BEAR ON YOUR CODE — all verified by me, none requiring you to
change course:

1. T-03's `ged_matrix` carries **inf** on censored pairs, not NaN — linux has 92 non-finite entries
   (2 x 46 censored), n_nan = 0. If anything you wrote filters censored pairs with `np.isnan`, it
   will pass 92 infinities straight through, and `inf <= x` is False while raising nothing. Filter on
   `np.isfinite` and select on `certified_mask` first. This applies to your definition-of-done item
   2, the `lb <= exact <= ub` check against T-03's linux.npz.

2. Suite-1 `aids` (769 graphs) is a strict SUBSET of Suite-2 `aids_graphedx` (819) — verified,
   overlap exactly 769, structural because Suite 1 is Suite 2 plus `n_max = 12`. So any comparison
   between the two cohorts must join on `graph_ids`, never positionally. Your cross-fill's "refuse
   files whose graph_ids disagree" check already has the right instinct; just make sure the refusal
   is on identity, not on length.

3. The zero-guard reading has been independently confirmed correct by wave-t05-slurm's gate work,
   with a measured number worth having: T-03's iam_letter_low census holds 215,968 exactly-zero
   off-diagonal entries out of 1,391,220 — 15.5 %. A per-pair `v > 0` assertion fails on correct
   data. What identifies a wrong accessor is that it returns 0.00 for EVERY pair. Preserve the
   existing lower-bound behaviour and its comment, exactly as briefed.

Nothing else changes. §3 roles and options strings, §4 keys, §4.1 derived certified_mask, §6 CLI,
§6.1 lazy zero_ok, §7 dense merge — all unchanged and still binding.
```
