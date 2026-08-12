# task-gedlib-gates — work log

Wave `2026-08-12-exact-ged`. Base `29886f879a9190ad8c869eaf0979c7cf8af364ef`.

## Mission

Write the GED backend layer (Contract B) and the four validation gates, on a machine that does not
have GEDLIB and never will. Every GEDLIB code path is therefore lazy, fake-tested, and declared
untested below with the command that exercises it on Picasso.

## What I built

| File | Lines | What |
|---|---:|---|
| `benchmarks/real_data/eval_setup/ged_backends.py` | 888 | Contract B: `PairResult`, `GedBackend`, `GedlibBackend`, `NetworkxBackend`, `StubBackend`, `BackendSpec`, `make_backend` |
| `benchmarks/real_data/eval_setup/ged_gates.py` | ~1500 | gates 0–3, a GEDLIB `probe` subcommand, Contract A reader, pooled pair evaluation, JSON reports |
| `tests/unit/test_ged_backends.py` | 625 | 57 tests, all GEDLIB interaction against a fake on `sys.meta_path` |
| `tests/unit/test_ged_gates.py` | ~470 | 38 tests, no GEDLIB, no torch, no source tree |

Contract B is implemented **verbatim**. No field, name or default differs.

### Three design decisions worth stating up front

**`PairResult` validates itself in `__post_init__`.** Invariant 4 is enforced in one place rather
than three, so any future backend inherits it. Constructing an uncertified result that carries an
`exact` value raises. This is the `ged_computer.py::compute_ged_pair` defect made unrepresentable.

**GEDLIB method/accessor pairings are validated at construction, not at read time.** The capability
table measured on Picasso is encoded as three frozensets. `GedlibBackend(lb_method="IPFP")` raises
immediately rather than filling a matrix with zeros. `HED` is refused in every role.

**`lb`/`ub` are the tightest certified bracket, so gate 1 uses an independent one.** Contract B says
`certified` means `lb == ub`, which forces the exact solver's certificate to be folded back into the
bounds. Testing the exact value against the backend's own bounds would then be vacuous. Gate 1
therefore brackets every certified value against `ged_bounds`, which makes it a real cross-check of
GEDLIB against a separate implementation of the Blumenthal–Gamper bound rather than a tautology.

## The five invariants and how each is tested

| # | Invariant | Where enforced | Tests |
|---|---|---|---|
| 1 | `0 < value < inf` on every GEDLIB read; `0` only if isomorphic | `GedlibBackend._read`, `zero_distance_is_attainable` | `TestInvariant1ZeroGuard` (5): zero from the wrong accessor raises; `inf` raises; negative raises; zero **accepted** for isomorphic graphs; a lower bound above a certified optimum raises |
| 2 | `HED` never used | `_validate_method`, `FORBIDDEN_METHODS` | `TestInvariant2NoHed` (3+3 param): HED refused in all three roles; unverified method refused; each role refuses a method not measured to serve it |
| 3 | Upper bounds symmetrised by min over both orientations | `GedlibBackend.pair`, `NetworkxBackend.heuristic_bracket` | `TestInvariant3UpperBoundOrientation` (4): both `run_method(0,1)` and `(1,0)` observed, min taken; asymmetry rate recorded; LB symmetry *measured* not assumed; the probe budget is honoured |
| 4 | `exact is None` unless certified | `PairResult.__post_init__` + both backends | `TestPairResultContract` (7) and `TestInvariant4ExactOnlyWhenCertified` (3); plus `TestNetworkxBackend` (4) for the timeout path |
| 5 | `libraries_import` strictly before `gedlibpy_gxl` | `GedlibBackend.module()` via `importlib.import_module` | `TestInvariant5ImportOrder` (3) |

**Invariant 1's zero rule is cost-model dependent, which the contract does not say.** Under
`GRAPHEDX_COSTS` node operations are free, so adding or removing isolated nodes costs nothing and a
zero distance is legal whenever the graphs are isomorphic *after dropping degree-zero nodes*.
`zero_distance_is_attainable` implements exactly that and is tested both ways.

**Invariant 5 is tested by making the wrong order fail.** The fake GEDLIB is installed as a
`MetaPathFinder` on `sys.meta_path`, not seeded into `sys.modules`, because `import_module` returns a
cached module without executing anything and the order would be unobservable. The fake's loader
raises the real `libdoublefann.so.2: cannot open shared object file` if `gedlibpy_gxl` executes
first, and `test_the_wrong_order_is_what_the_fake_punishes` asserts that it does — so the ordering
test cannot pass vacuously.

**Ruff did not reorder the GEDLIB imports.** They are `importlib.import_module` calls inside
`GedlibBackend.module()`, which is a function body, so there is no import statement for a formatter
to move. Verified: `ruff check` clean on all four files, and the two calls remain in source order.
The repo's format-on-write hook *did* strip `argparse`, `asdict`, `Callable` and `Iterator` from
module import blocks while they were momentarily unused mid-edit — worth knowing, but it cannot
touch the GEDLIB path.

## Gate results actually obtained

All runs on this workstation with `--backend networkx`, Python 3.11.15, nx 3.6.1, numpy 1.26.4,
13th Gen Intel Core i7-13700KF, 24 cores, `OMP_NUM_THREADS` unset.

### Gate 1 — PASS

```
--gate 1 --backend networkx --n-pairs 120 --max-nodes 9 --timeout 20 --workers 4
passed True   n_pairs 120   seconds 25.7   datasets: linux, aids
cost_model [1,1,0,1,1,0]   n_evaluated 120   n_certified 120   n_violations 0
independent_lb_slack  min 0.0  p25 0.0  median 1.0  p75 2.0  max 4.0   mean 1.175
independent_ub_slack  min 0.0  p25 4.0  median 6.0  p75 8.0  max 14.0  mean 5.533
backend_stats: certification_rate 1.0, mean_seconds 0.514, n_timed_out 0
               ub_asymmetry_rate 0.65 (78/120), max_ub_gap 10.0
```

Non-vacuous: 120 of 120 pairs certified, and the slack quantiles show the independent bracket is
genuinely open (median LB slack 1, median UB slack 6), so the containment test had something to
test.

### Gate 2 — FAIL, diagnosed, then PASS

First run failed on **30 of 100** replayed LINUX pairs, every one with the replay upper bound
*below* the archived value, never above. Diagnosis, verified directly:

```
idx  1: archived=11.0  forward=11.0  reverse=9.0   min=9.0
idx  8: archived=13.0  forward=13.0  reverse=7.0   min=7.0
idx 13: archived=10.0  forward=10.0  reverse=8.0   min=8.0
idx 20: archived=12.0  forward=12.0  reverse=12.0  min=12.0
idx 25: archived=14.0  forward=14.0  reverse=16.0  min=14.0
over 100 pairs: archived == forward-only 100/100,  archived == symmetrised 70/100
```

The archive predates the `symmetrise` parameter of `bipartite_upper_bound`; its upper bound is the
single forward orientation. The 30 differences are the gain from taking the minimum over both
orientations, which is what invariant 3 requires. **This is not a disagreement between
implementations, and the original gate rule would have reported it as one.**

Gate 2 now reproduces the *forward* orientation exactly as a hard condition, permits the symmetrised
bound to be tighter, fails only if it is looser, and reports the gain rate.

```
--gate 2 --backend networkx --n-pairs 100 --timeout 20 --workers 4
passed True   replayed 100 of 400 archived pairs
n_identity_failures 0        (all 400 regenerated pairs match archived n1,n2,m1,m2)
n_bound_mismatch 0           (lb and forward-orientation ub reproduced exactly)
n_ub_improved_by_symmetrisation 30   ub_symmetrisation_gain_rate 0.30
n_ub_regression 0
n_exact_agree 98   n_exact_regression 0   n_archive_suboptimal 0
backend_stats: certification_rate 0.98, ub_asymmetry_rate 0.68, n_timed_out 2
```

Two pairs did not certify at a 20 s budget; they are excluded from the exact comparison and counted,
not silently dropped.

### Gate 3 — PASS (harness only; not the real benchmark)

```
--gate 3 --backend networkx --n-pairs 60 --max-nodes 10 --timeout 20   (serial, workers ignored)
passed True   n_comparable 59   n_disagreements 0   datasets: aids, linux
benchmark_meaningful False
```

| stratum `max(n1,n2)` | pairs | backend median s | reference median s | speedup | backend certified | reference certified |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 2 | 0.0119 | 0.0117 | 0.98 | 2 | 2 |
| 8 | 3 | 0.0256 | 0.0256 | 1.00 | 3 | 3 |
| 9 | 15 | 0.3064 | 0.3068 | 1.00 | 15 | 15 |
| 10 | 40 | 1.2657 | 1.2981 | 1.03 | 39 | 39 |

Both columns are NetworkX A*, so the speedup is 1.00 by construction and the report says so
(`benchmark_meaningful: false`). **What this table does establish** is the NetworkX reference cost
that the GEDLIB run will be compared against, on known hardware: the cost of one exact pair grows
roughly 4× per node from n = 8, reaching ~1.27 s at n = 10. Extrapolating that curve is what makes
the n = 12 ceiling in the plan plausible and what a GEDLIB speedup would move.

### Gate 0 — FAIL, and the failure is GraphEdX's, not ours

```
--gate 0 --backend networkx --n-pairs 300 --timeout 15 --workers 6
AIDS: 769 of 911 graphs pass the cohort filter      <- reproduces CONTRACTS §2 exactly
passed False   n_pairs 208   seconds 487.5   cost_model [0,0,0,1,1,0]
n_sampled 300   n_certified 208   n_uncertified 92
n_equal 58      n_ours_lower 150      n_ours_higher 0
signed_delta: min -8.0  p25 -2.0  median -1.0  p75 0.0  max 0.0  mean -1.582
```

Signed discrepancy histogram (`ours - published`):

| delta | -8 | -7 | -6 | -5 | -4 | -3 | -2 | -1 | 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairs | 2 | 2 | 6 | 2 | 12 | 21 | 37 | 68 | 58 |

**The distribution is strictly one-sided. Not one pair of 208 has our value above theirs.** That is
the whole reason the gate records a distribution instead of a boolean, and it is decisive: a solver
defect on our side would produce errors in both directions, or at minimum one pair where we exceed a
true optimum. Zero such pairs, and 150 where we are strictly below, is the signature of GraphEdX's
published values being **upper bounds from an approximate solver**, not exact distances.

Our side is provably optimal on every one of the 208: `certified` means `lb == ub`. I additionally
re-ran six disagreeing pairs through NetworkX A* with **no timeout at all**, and every one
reproduced our value exactly:

```
aids_train_0164 / aids_train_0263   ours 3.0   published 4.0   unbounded A* 3.0   BRANCH LB 2.0
aids_train_0035 / aids_train_0037   ours 4.0   published 5.0   unbounded A* 4.0   BRANCH LB 1.0
aids_train_0109 / aids_train_0254   ours 5.0   published 7.0   unbounded A* 5.0   BRANCH LB 2.0
aids_train_0257 / aids_train_0502   ours 6.0   published 9.0   unbounded A* 6.0   BRANCH LB 5.0
aids_train_0184 / aids_train_0336   ours 5.0   published 9.0   unbounded A* 5.0   BRANCH LB 4.0
aids_train_0250 / aids_train_0407   ours 2.0   published 3.0   unbounded A* 2.0   BRANCH LB 1.0
```

The gate is left **failing**. Its stated pass condition is exact agreement, that condition is not
met, and weakening it to accommodate the reference would destroy the only check we have on our own
solver. The verdict to carry forward is the distribution, not the boolean.

**This has a consequence for the manuscript.** The submitted AIDS correlation (rho ~ 0.35) is
computed against this published matrix. If the reference carries a mean error of +1.58 edit
operations on within-split pairs, with a tail to +8, then that correlation was measured against a
noisy target — which is an argument *for* R3.5b's recomputation, and a number the response letter can
use. It is also a claim that needs checking against how GraphEdX generated the AIDS matrix before it
goes anywhere near a reviewer; 92 of 300 pairs did not certify at a 15 s budget, so the measured
distribution covers the tractable 69%, not the whole cohort.

### Test suite

```
tests/unit/test_ged_backends.py   57 passed
tests/unit/test_ged_gates.py      38 passed
tests/unit/test_ged_bounds.py     35 passed   (unchanged)
tests/unit/                      481 passed   (386 before this branch, +95)
ruff check                        clean on all four files
```

## Untested on this machine

Everything below runs GEDLIB and has therefore **never executed**. It is covered by the fake, which
tests the call sequence and the argument values, not the library's behaviour.

Prelude for every command:

```bash
CE=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph
export PYTHONPATH=/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn:${REPO_DIR}
cd ${REPO_DIR}          # repo root, never ${REPO_DIR}/src
```

**1. Run the probe first. It is one command and it checks everything else on this list.**

```bash
$CE/bin/python -m benchmarks.real_data.eval_setup.ged_gates --gate probe --out probe_out
```

It asserts the import order, prints the method and edit-cost lists, runs P4 vs C4 and checks
`exact == 1.0`, demonstrates that `get_lower_bound()` on `IPFP` returns `0.00` without raising,
confirms our construction guard converts that into an exception, probes which options
`ANCHOR_AWARE_GED` accepts, and reports which environment-reset path was taken. Exit 0 means every
mandatory check held.

**2. `restart_env()` — the one API call I am least sure of.** `GedlibBackend` builds `GEDEnvGXL`
once and calls `env.restart_env()` between pairs, because rebuilding per pair violates the
once-per-process rule while never resetting would accumulate graphs without bound over ~3.9 M pairs.
If `restart_env` is absent the code falls back to rebuilding the environment and logs which path it
took; the probe reports it as `reset_mode`. Check with:

```bash
$CE/bin/python -c "
import importlib; importlib.import_module('gklearn.gedlib.libraries_import')
g=importlib.import_module('gklearn.gedlib.gedlibpy_gxl'); e=g.GEDEnvGXL()
print('restart_env', hasattr(e,'restart_env'))"
```

If it is missing, tell me — the fallback is correct but slow and I would restructure to a batch API.

**3. `--time-limit` on `ANCHOR_AWARE_GED` is unverified, so it defaults off.** GEDLIB rejects unknown
options, so passing one we have not confirmed would abort the run. Consequence: **`timeout_s` does
not abort a GEDLIB solve.** The C++ call is not interruptible from Python, so the budget is detected
after the fact and recorded in `timed_out`; a pathological pair can overrun. The probe answers
whether the option exists; if it does, pass `exact_time_limit_s` and the budget becomes real.

**4. Whether `BRANCH_FAST`'s lower bound equals ours on real data.** `ged_bounds`' own docstring
argues that on unlabelled graphs with uniform edge costs BRANCH and BRANCH-FAST coincide, so gate 2
asserts equality. If that assertion fires on Picasso the docstring is wrong, not the library.

```bash
$CE/bin/python -m benchmarks.real_data.eval_setup.ged_gates \
  --gate 2 --backend gedlib --n-pairs 400 --workers 8 --out gate_out
```

**5. GEDLIB's upper-bound asymmetry rate.** Measured 0.65–0.68 for our bipartite bound on real
pairs. IPFP's rate is unmeasured; it is recorded automatically as `ub_asymmetry_rate` in every gate
report.

**6. The full gate set with GEDLIB**, which is job 1 of the execution plan:

```bash
$CE/bin/python -m benchmarks.real_data.eval_setup.ged_gates \
  --gate all --backend gedlib --input-dir <exports> --out gate_out \
  --n-pairs 500 --seed 42 --timeout 300 --workers 4
```

Gate 0 will refuse to run from `--input-dir` alone: the Contract A export does not carry GraphEdX's
published matrix. Run gate 0 locally with `--source-dir`, or hand me the reference and I will add a
loader for it.

## Decisions and why

**Gate 1 brackets against `ged_bounds`, not against the backend's own bounds.** Explained above; the
alternative is a tautology.

**Gate 2 treats an archived value above a new certified optimum as evidence, not failure.** The
archive was produced with a 30 s NetworkX timeout and `nx.graph_edit_distance` returns its
best-so-far cost when that expires, so an archived "exact" can be an uncertified upper bound. A new
value *above* the archive is a hard failure; below it is counted as `n_archive_suboptimal` and
explained in the report. On the 100 pairs replayed this count was 0, so the archived LINUX values
show no evidence of timeout contamination at n ≤ 10.

**Gate 3 runs strictly serially** and ignores `--workers`. Pooled execution contends for cores and
the per-pair seconds are what sizes the production run.

**Backends expose `.stats`.** The upper-bound asymmetry rate was asked for explicitly; certification
rate, timeout count and mean seconds come along at no cost and land in every gate report.

**`StubBackend` returns the two trivial bounds** — size difference below, delete-all-then-insert-all
above — which are valid under both cost models and require no solver. Its `seconds` is a synthetic
constant so its output is byte-reproducible for `task-runner`'s plumbing tests.

**Three optional keyword-only arguments added to `GedlibBackend`**: `threads=1`,
`exact_time_limit_s=None`, `lb_symmetry_probes=32`. Every documented parameter keeps its name,
position and default, so every call in Contract B still works unchanged. This is additive, not a
deviation — but it is a change to a frozen file and I am flagging it rather than burying it.

## Assumptions

1. **`GEDEnvGXL` exposes `restart_env()`.** Falls back to rebuilding if not; logged and probed.
2. **`add_nx_graph(graph, "")` accepts integer node ids**, since gedlibpy `str()`s them itself. Only
   attribute *values* are required to be strings, and a constant `"1"` label is attached to every
   node and edge. Substitution is free under both cost models, so the label cannot affect a distance.
3. **`ANCHOR_AWARE_GED` accepts `--threads`.** Every GEDLIB method takes it; if it does not, the
   probe says so.
4. **The gate-2 archive is the authority on its own numbers.** My brief quoted
   `rho(exact,UB) = 0.522` and certification `1.5%`; the file says **0.4785** and **0.0125**. I used
   the file. Worth checking which the manuscript quotes.
5. **Gate 0's reference is `graphedx_loader`'s decoding** of `*_result.pt`, including its
   round-to-nearest-integer step for values like `4.999999`.
6. **The cohort filter is `2 <= n <= 12` and connected**, inlined from CONTRACTS §2 rather than
   imported from `dataset_filter`, which I do not own. LINUX reproduced exactly: 89 graphs.

## Follow-ups for the orchestrator

0. **Gate 0 fails, and you need to decide what that means.** GraphEdX's published AIDS values sit
   *above* our certified optima on 150 of 208 pairs and below on none. I did not weaken the gate.
   Three things follow: (a) verify how GraphEdX generated that matrix before the response letter
   cites this; (b) the submitted AIDS correlation was computed against it, so its noise is now
   quantified (mean +1.58, max +8); (c) if you accept the reference as approximate, gate 0's pass
   condition should be restated as "no pair where ours exceeds theirs", which **passes** on this
   sample — but that is a scientific decision about what the gate certifies, and it is yours.
1. **Run the probe before anything else.** One command, answers five open questions.
2. **`compute_ged_pair`'s timeout defect is real and I did not touch it** (frozen, another ticket).
   `ged_computer.py::compute_ged_pair` returns `nx.graph_edit_distance(..., timeout=t)`'s best-so-far
   cost as an exact value. Every "exact GED" in the submitted study above the timeout threshold is an
   uncertified upper bound. Gate 2 found no contamination on 100 LINUX pairs at n ≤ 10, which bounds
   the problem but does not clear AIDS or larger graphs.
3. **Gate 0 cannot run on Picasso from Contract A alone.** The export carries no published matrix.
   Either run gate 0 locally, or add the reference values to the export — a decision for you and
   `task-export`, and one I could not make unilaterally.
4. **A batch GEDLIB API is the biggest available speedup and Contract B forbids it.** `pair(g1, g2)`
   forces two `add_nx_graph` calls plus an `init()` per pair. Adding all N graphs once and calling
   `run_method(i, j)` over all pairs would amortise that across `C(N,2)` pairs. For AIDS that is
   295,296 pairs against 769 graph insertions. If the probe shows per-pair setup is material, this is
   worth a contract amendment before the production run.
5. **`ub_asymmetry_rate` measured 0.65–0.68**, not the ~0.33 quoted in my brief. On 120 LINUX/AIDS
   pairs, 78 had a strictly tighter orientation, max gap 10. If the manuscript quotes 33%, it needs
   restating; if 33% came from a different sample, the window should be named the way the
   directedness collision rate has to be.
6. **Stale constants confirmed, not fixed** (outside every agent's ownership): `eval_setup.py:75
   DEFAULT_SOURCE_DIR` and `eval_message_length.py:36 DEFAULT_DATA_ROOT` point at the pre-move path.
7. **`load_contract_a` duplicates `load_exported`.** Replace it with the real loader once
   `task-export` merges; the keys it reads are exactly CONTRACTS §4.
