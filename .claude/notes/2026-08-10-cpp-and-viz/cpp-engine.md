# Work log — cpp-engine

## Identity

| Field | Value |
|---|---|
| Agent | `wave-cpp-engine` |
| Wave | `2026-08-10-cpp-and-viz` |
| Model / effort | Claude Opus 5 / high |
| Branch | `wave/cpp-engine` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-a3c0c2a5ce14cac9e` |
| Base commit | `2f393a1` |
| Head commit | last code commit `1e7566a`; this log sits on top as the branch head (its own SHA cannot be self-referenced) |
| Started / finished | 2026-08-10 / 2026-08-10 |
| Status | complete |

## 1. Prompt as received

The full delegation brief is long; it is reproduced here in the structure it
was given, with every normative clause preserved verbatim.

```
You are the agent `wave-cpp-engine` in a two-agent parallel wave. Your orchestrator is `main`.

0. Who you are: systems-and-numerics engineer porting a research reference implementation to C++.
   The product is "a second implementation that provably computes the same function as the first",
   and is fast. Outputs feed a Pattern Recognition revision, deadline 2026-08-31.
   Standing rules:
   - Parity before performance. Land a faithful, byte-exact port first. Optimise only afterwards,
     re-running the differential suite after every single optimisation.
   - The Python reference is frozen. You may read it; you may not edit it.
   - A self-comparison that reports PASS is a broken harness.
   - Never claim a result you did not observe. Paste real terminal output into your log.
   - You cannot ask the user anything. If you need a decision, message `main`.

1. Wave facts: WAVE_ID 2026-08-10-cpp-and-viz; BASE_SHA 2f393a1; branch wave/cpp-engine.

2. Mission: give IsalGraph a native C++ engine following IsalSR/IsalHG. Port cdll.py,
   sparse_graph.py, string_to_graph.py, graph_to_string.py + generate_pairs_sorted_by_sum,
   canonical.py, canonical_pruned.py.
   2.2 four invariants: CDLL indices != graph node indices; insert_after(cdll_index, payload);
   pointer immobility after V/v; displacement pairs sort by (|a|+|b|, |a|, (a,b)).
   2.3 the one design decision: marshal adjacency across the FFI in Python's own set-iteration
   order; do not use std::unordered_set for anything whose iteration order can reach an output.
   2.4 optimisation priority: (1) memoise generate_pairs_sorted_by_sum — Reviewer 3 asks this exact
   question; (2) mutable output buffer; (3) running counters; (4) restrict candidate scan;
   (5) parallelise the per-starting-node loop, gated on an explicit parameter, never
   hardware_concurrency(), default single-threaded. Record every optimisation with measured
   before/after numbers and every negative result in docs/engineering/CPP_OPTIMIZATION_LOG.md.

3. Copy from IsalSR (not IsalHG). 3.1 non-negotiable build decisions: nanobind + scikit-build-core;
   source dir `native` not `_native`; target `_native` installed to isalgraph/core;
   target_compile_options only; -march=x86-64-v3 never native, opt-in ISALGRAPH_NATIVE_MARCH;
   -O3 -DNDEBUG -fno-plt -funroll-loops; -static-libstdc++ -static-libgcc; ISALGRAPH_ENABLE_SANITIZERS;
   wheel.exclude the C++ sources; probe.cpp on day one.
   3.2 frozen dispatch contract: backends.py, ISALGRAPH_ENGINE, explicit backend= always wins over
   env var, _python_/_cpp_/dispatcher naming law, mirror errors.py in errors.hpp, timeout_s.

4. Ownership: CMakeLists.txt, pyproject.toml, src/isalgraph/core/native/**,
   src/isalgraph/core/backends.py, src/isalgraph/core/algorithms/*.py, tests/native/**,
   docs/engineering/CPP_BUILD.md, docs/engineering/CPP_OPTIMIZATION_LOG.md, and my note.
   Everything else read-only. main owns both __init__.py and will rewire exports at merge.

5. Environment: dedicated conda env ~/.conda/envs/isalgraph-cpp. Verify both import paths land in
   the worktree. Do not set PYTHONPATH=$WT/src. Baseline on BASE_SHA: 450 passed, 271 skipped.
   Guard native tests with pytest.importorskip at module top level, not pytestmark = skipif.

6. Definition of done: (1) builds, build_info reports expected ISA and stable hash;
   (2) property parity through the cpp backend — round-trip, isomorphism invariance/dedup,
   canonicality, idempotence, on random and structured graphs including self-loops and directed;
   (3) differential parity vs the frozen reference — canonical and pruned byte-exact over >=3,000
   graphs zero mismatches, string_to_graph identical graphs over >=2,000 strings, greedy byte-exact,
   levenshtein exact, exception parity on type AND message;
   (4) fallback: entire suite passes with the .so physically deleted;
   (5) reproduce partial paper results on IAM Letter; if absent, say so, message main, substitute
   NetworkX graphs matched to IAM LOW (1,180 graphs, mean 3.07 edges, <=12 nodes, connected);
   show on >=200 pairs identical strings and identical Levenshtein matrices, and report speedup
   per node-count bucket, never as a single number;
   (6) benchmark best-of-9, median-of-4, 3 warmups, engines alternated, raw JSON committed;
   (7) ruff clean, mypy --strict clean, full suite green;
   (8) CPP_BUILD.md documents local and Picasso builds.

7. Mandatory: commit everything; write and commit the work log using the template; the log's file
   list must match git diff --name-only 2f393a1..HEAD exactly; conventional commits; no
   Co-authored-by; do not push, rebase, merge, or touch the peer branch.

8. Peers: main (orchestrator), wave-viz (viz subpackage + trace emission in string_to_graph.py and
   graph_to_string.py — trace only, no signature or semantic change).
```

Mid-task course corrections from `main` are reproduced in §8.

## 2. Understanding and plan

**Restatement:** Build a C++17 engine for IsalGraph's six core modules that
computes bit-identical results to the frozen Python reference, prove that
identity with a differential suite rather than asserting it, then make it
fast and measure every step. The engine must be optional — with no compiled
artifact the package must behave exactly as it does today.

**Approach chosen:** nanobind + scikit-build-core following IsalSR's
CMakeLists almost verbatim; a `backends.py` dispatch layer that marshals
graphs across the FFI *in CPython's own set-iteration order*; a single
recursive `step()` in C++ shared by the pruned and unpruned canonical
searches, differing only by a nullable triplet pointer.

**Alternatives considered and rejected:**

- **`std::set<int32_t>` for adjacency** — rejected because it would sort the
  neighbours and change which candidate the greedy encoder picks first. This
  is §2.3 of the brief and it is the whole ball game; see §6 for the evidence
  that honouring it gave byte-exact greedy parity on the first attempt.
- **Deriving `logical_edge_count` from adjacency lengths** — rejected after
  reasoning about self-loops: in an undirected graph a self-loop occupies one
  adjacency slot but increments `_edge_count` twice, so `sum(len(adj)) // 2`
  undercounts and the encoder stops one edge early. The count is now
  marshalled from the graph's own counter.
- **Duplicating the search for the pruned variant** — rejected; a nullable
  `const std::vector<Triplet>*` selects the filter, so the two searches cannot
  drift apart.
- **Reproducing CPython's `set` repr in C++** for the "Unreachable nodes: {0}"
  error message — rejected as fragile for no gain. Reachability and
  initial-node range checks run Python-side before marshalling, which buys
  byte-identical error text for an O(V+E) cost against an exponential search.
- **`hardware_concurrency()`** — rejected per the brief; never called.

**Plan as executed:**

1. Read the six reference modules and the IsalSR build artifacts in full.
2. CMakeLists + pyproject + probe.cpp; confirm the toolchain and the ISA.
3. Port the data structures, then the VM, then greedy, then the two searches.
4. `backends.py` with the frozen dispatch contract.
5. Differential suite; only then optimise, re-verifying after each step.
6. Rewire `algorithms/*.py`; fallback verification; benchmarks; docs.

**Deviations from the plan:** Three.

- **Optimisations 1–3 of the ladder are structural, so they landed in the
  first commit rather than as a measured sequence.** The pair memo is a
  `thread_local` cache from the start, the output buffer was always mutable,
  and the counters were always parameters. I recovered honest before/after
  numbers for the memo by adding a runtime A/B toggle
  (`set_pairs_memo`) rather than by claiming an unmeasured figure. The buffer
  change (O2) is recorded in the log as **not A/B'd**, with the asymptotic
  argument and an explicit warning not to quote it as measured.
- **Ladder item 4 ("restrict the candidate scan") was not implemented.** On
  inspection the narrowing is either unsound or already present; the log
  explains why, rather than shipping a no-op labelled as an optimisation.
- **Added branch-and-bound (O5), which was not in the ladder.** After the memo
  landed the remaining cost was visibly search-bound.

## 3. Changes made

**Created**

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Build the `_native` extension; per-target flags, x86-64-v3, static libstdc++ |
| `src/isalgraph/core/backends.py` | Engine dispatch, FFI marshalling, exception translation |
| `src/isalgraph/core/native/include/isalgraph/fnv.hpp` | FNV-1a 64-bit, single source of truth for `build_hash` |
| `src/isalgraph/core/native/include/isalgraph/errors.hpp` | C++ mirror of `errors.py`, incl. new `EncodingStuckError` |
| `src/isalgraph/core/native/include/isalgraph/cdll.hpp` | CDLL declaration; invariants 1 and 2 documented in-header |
| `src/isalgraph/core/native/include/isalgraph/sparse_graph.hpp` | `InputGraph` (ordered) and `SparseGraph` (mutable) |
| `src/isalgraph/core/native/include/isalgraph/pairs.hpp` | Memoised displacement pairs; invariant 4 |
| `src/isalgraph/core/native/include/isalgraph/budget.hpp` | Wall-clock deadline threaded through every frame |
| `src/isalgraph/core/native/include/isalgraph/string_to_graph.hpp` | VM declaration; invariant 4 (pointer immobility) |
| `src/isalgraph/core/native/include/isalgraph/graph_to_string.hpp` | Greedy encoder declaration |
| `src/isalgraph/core/native/include/isalgraph/canonical.hpp` | Canonical + pruned canonical + triplets |
| `src/isalgraph/core/native/include/isalgraph/levenshtein.hpp` | Edit distance |
| `src/isalgraph/core/native/src/cdll.cpp` | Free list initialised descending, popped from the back |
| `src/isalgraph/core/native/src/sparse_graph.cpp` | Dense bitmatrix <=2048 nodes, sorted vectors beyond |
| `src/isalgraph/core/native/src/pairs.cpp` | Sort key `(|a|+|b|, |a|, a, b)`; memo + A/B toggle |
| `src/isalgraph/core/native/src/string_to_graph.cpp` | 9-instruction VM, edges logged in call order |
| `src/isalgraph/core/native/src/graph_to_string.cpp` | Greedy encoder; the order-dependent path |
| `src/isalgraph/core/native/src/canonical.cpp` | Shared backtracking search, triplets, B&B, threading |
| `src/isalgraph/core/native/src/levenshtein.cpp` | Single-row DP |
| `src/isalgraph/core/native/src/probe.cpp` | `engine_name`, `build_info`, `fnv1a64` |
| `src/isalgraph/core/native/src/bindings.cpp` | nanobind surface, GIL release, exception translators |
| `tests/native/graphs.py` | Deterministic graph generators and corpora |
| `tests/native/test_native_build.py` | Probe, ISA, FNV, dispatch contract |
| `tests/native/test_native_datastructures.py` | CDLL and pair-ordering differential |
| `tests/native/test_native_differential.py` | Byte-exact parity vs the frozen reference |
| `tests/native/test_native_errors.py` | Exception type+message parity; errors.py contract |
| `tests/native/test_native_properties.py` | Round-trip, invariance, canonicality, idempotence |
| `tests/native/bench_native.py` | Benchmark + IAM surrogate harness |
| `docs/engineering/CPP_BUILD.md` | Local and Picasso build documentation |
| `docs/engineering/CPP_OPTIMIZATION_LOG.md` | Optimisation ladder with measurements |
| `docs/engineering/results/speedup.json` | Raw per-bucket speedup data |
| `docs/engineering/results/ladder.json` | Raw A/B optimisation data |
| `docs/engineering/results/iam_surrogate.json` | Raw IAM-surrogate data |
| `.claude/notes/2026-08-10-cpp-and-viz/cpp-engine.md` | This log |

**Modified**

| Path | Change | Reason |
|---|---|---|
| `pyproject.toml` | setuptools -> scikit-build-core backend; `[tool.scikit-build]`; `native` extra; ruff per-file-ignores for `tests/native/*` | Build the extension; E402 is mandatory for the importorskip placement |
| `src/isalgraph/core/algorithms/base.py` | Added `_as_legacy_value_error` shim | Six existing tests pin `ValueError`; errors.py mixins land at integration |
| `src/isalgraph/core/algorithms/exhaustive.py` | Dispatch via `backends.canonical_string`; `backend`/`timeout_s`/`threads` | Route through the engine |
| `src/isalgraph/core/algorithms/pruned_exhaustive.py` | Same for `pruned_canonical_string` | Route through the engine |
| `src/isalgraph/core/algorithms/greedy_min.py` | Dispatch via `backends.graph_to_string`; `backend` | Route through the engine |
| `src/isalgraph/core/algorithms/greedy_single.py` | Same | Route through the engine |

**Removed** — none.

**Commits**

| SHA | Message |
|---|---|
| `a8f5ee3` | `feat(native): add C++17 engine with nanobind bindings and backend dispatch` |
| `1e7566a` | `feat(core): dispatch G2S algorithms through the backend layer` |
| branch head | `docs(notes): cpp-engine work log` |

The Created + Modified lists above are exactly
`git diff --name-only 2f393a1..HEAD` (39 files before this log, 40 with it).

## 4. Tests

**Tests created or extended**

| Test | File | What it verifies | Why it matters |
|---|---|---|---|
| `test_canonical_string_is_byte_exact` | `test_native_differential.py` | 3,079 graphs, C++ == reference character for character | The core parity claim |
| `test_pruned_canonical_string_is_byte_exact` | same | 3,079 graphs, pruned variant | Second hot path |
| `test_greedy_graph_to_string_is_byte_exact` | same | 11,000+ (graph, start node) pairs | The order-dependent path; would fail if adjacency order were lost |
| `test_greedy_parity_survives_relabelling` | same | 6 random relabellings per structured graph | Relabelling permutes CPython set slots — the sharpest probe of the marshalling contract |
| `test_string_to_graph_produces_identical_graphs` | same | 2,276 strings, both directions | VM parity |
| `test_string_to_graph_adjacency_iterates_identically` | same | Set *iteration order* matches, not just contents | A decoded graph fed to greedy would otherwise diverge |
| `test_decoded_graphs_encode_identically` | same | End-to-end consequence of the above | Catches the subtle version of the same bug |
| `test_levenshtein_matches_on_random_pairs` | same | 4,000 random pairs + 7 hand cases | Exact integer equality |
| `test_structural_triplets_are_byte_exact` | same | 600 graphs | The pruning input must match or pruning diverges |
| `test_free_list_allocates_zero_one_two_in_order` | `test_native_datastructures.py` | Free list pops 0,1,2,… | Load-bearing; a deviation corrupts silently |
| `test_removed_index_is_reused_lifo` | same | `remove()` then `insert_after()` reclaims the index | The canonical search removes constantly |
| `test_random_operation_sequences_agree` | same | 40 random op streams, full state diffed each step | Catches CDLL divergence the algorithms would mask |
| `test_pair_ordering_is_byte_identical` | same | m = 1..24 against the reference | Invariant 4 |
| `test_pair_ordering_is_not_the_algebraic_sum_bug` | same | Regression guard for historical bug B2 | The bug this project already fixed once |
| `test_pairs_are_memoised` | same | Cache size does not grow on a repeat call | Proves O1 is live |
| `test_round_trip_from_random_strings` / `_structured_graphs` | `test_native_properties.py` | `S2G(w) ~= S2G(G2S(S2G(w), v0))` | Acceptance criterion |
| `test_canonical_string_is_relabelling_invariant` | same | 8 permutations per graph, both variants | Acceptance criterion (invariance) |
| `test_non_isomorphic_graphs_get_different_strings` | same | All pairs of 11 distinct families | Acceptance criterion (deduplication) |
| `test_equal_strings_decode_to_isomorphic_graphs` | same | Collision buckets over 200 graphs | The converse direction |
| `test_canonical_is_lexmin_among_shortest_greedy_encodings` | same | Canonical <= best greedy by (len, lex) | Acceptance criterion (canonicality) |
| `test_canonical_is_idempotent` | same | `canonical(S2G(canonical(G))) == canonical(G)` | Acceptance criterion |
| `test_threaded_search_returns_the_same_string` | same | threads in {1,2,4} give identical output | Parallelism must not change the answer |
| `test_disconnected_graph_parity` (+ pruned, + G2S) | `test_native_errors.py` | Same class and same message from both engines | Exception parity |
| `test_invalid_instruction_parity` | same | 6 bad strings, message includes the Python repr | Exception parity incl. repr |
| `test_timeout_raises_the_mirrored_class` | same | 1 µs budget on `complete(9)` | The budget the eval harness needs |
| `test_python_backend_refuses_a_budget_it_cannot_honour` | same | `BackendError`, not silent success | A silently-ignored budget is worse than an error |
| `test_explicit_backend_beats_env_var` | `test_native_build.py` | Probes dispatch, not the report | The exact IsalSR bug the brief names |
| `test_build_is_release_cxx17_and_portable_isa` | same | `isa_level == x86-64-v3`, `avx512f == 0`, `ndebug == 1` | Catches an accidental `-march=native` before SLURM does |
| `test_fnv1a64_matches_python_reference` | same | 6 payloads vs an independent Python FNV | Validates `build_hash` |

**Coverage of the behaviour that matters:** every dispatching function is
exercised through `backend="cpp"` *explicitly*, never by relying on the
default, so a broken extension cannot pass by falling back. Both engines are
also compared to each other through the dispatcher
(`test_dispatcher_backends_agree`). Structured families cover paths, cycles,
stars, trees, complete graphs, grids, Barabási–Albert, Erdős–Rényi,
self-loops, and directed variants of each.

**Not tested, and why:**

- **The real IAM Letter dataset** — not present on this machine (§7).
- **Graphs above 2,048 nodes**, where `SparseGraph` switches from the dense
  bitmatrix to sorted vectors. That branch is written and compiles but is
  never taken by any test: the reference is intractable long before that size,
  so there is nothing to diff against. **This is a genuine untested path.**
- **The Picasso build.** `CPP_BUILD.md` documents it, but I have no cluster
  access from this worktree, so every Picasso instruction is derived from the
  IsalSR precedent rather than executed.
- **Sanitizer build.** The CMake option exists and is documented; I did not
  run a build under ASan/UBSan.
- **Concurrency beyond output equality.** `threads>1` is verified to produce
  identical strings, but I did not run it under a thread sanitizer.

## 5. Test results

**Command:** `python -m pytest tests/native/ -q --no-header`

```
============================= test session starts ==============================
tests/native/test_native_build.py ....................                   [ 12%]
tests/native/test_native_datastructures.py ............................. [ 29%]
tests/native/test_native_differential.py .....................           [ 72%]
tests/native/test_native_errors.py ..................ssssssssss          [ 89%]
tests/native/test_native_properties.py .................                 [100%]
================= 155 passed, 10 skipped in 148.44s (0:02:28) ==================
```

**Command:** `python -m pytest tests/ -q --no-header`

```
================= 605 passed, 281 skipped in 154.41s (0:02:34) =================
```

**Command:** same, with the `.so` moved out of site-packages

```
====================== 450 passed, 276 skipped in 14.74s =======================
```

**Command:** `python -m ruff check src/ tests/` → `All checks passed!`
**Command:** `python -m mypy src/isalgraph/` → `Success: no issues found in 22 source files`

**Result:** 605 passed, 0 failed, 281 skipped · **Duration:** 2 m 34 s ·
**Run at:** `1e7566a`

Baseline on `2f393a1` was 450 passed / 271 skipped. I add 155 passed and 10
skipped; 450 + 155 = 605 and 271 + 10 = 281, so **no baseline test changed
state**. The 10 skips are the errors.py builtin-mixin contract, gated on
`main`'s integration change.

**Failures and their resolution:** seven tests failed on the first full run.

1. `test_capacity_exhaustion_message_matches_reference` — asserted
   `isinstance(CapacityError(...), RuntimeError)`, which is false until
   `main` lands the mixins. **Fix:** moved the builtin-mixin half into
   `test_native_errors.py` behind a `skipif` gated on
   `issubclass(errors.CapacityError, RuntimeError)`, so it flips to a pass
   with no edit once errors.py changes.
2. `test_pairs_rejects_non_positive_m` — C++ threw `IsalGraphError`, which
   maps to a bare `Exception`, while the reference raises `ValueError`.
   **Fix:** throw `std::invalid_argument`, which nanobind maps to `ValueError`
   with the identical message. Now asserts message equality too.
3–7. Five tests guarded encoding failures with
   `except (ValueError, RuntimeError)`, but the dispatch layer raises the
   `isalgraph.errors` classes, which are currently neither. Legitimate errors
   escaped and were misreported as parity failures. **Fix:** a shared
   `G.ENCODING_ERRORS` tuple. Separately, `test_canonical_string_is_byte_exact`
   was also failing its `compared >= 3000` floor because the directed corpus
   oriented its spanning tree child→parent, making node 0 a sink and most
   directed graphs unencodable. **Fix:** orient parent→child so node 0 roots a
   spanning out-tree.

Before fixing, I ran the comparison standalone with correct exception handling
to establish whether any *real* mismatch existed:

```
canonical mismatches: 0
```

so all seven were harness defects, not engine defects.

One further failure appeared after that:
`test_equal_strings_decode_to_isomorphic_graphs` bucketed directed and
undirected graphs under the same key. That one was a **real finding about the
encoding**, not a harness slip — see §7.

## 6. Verification beyond unit tests

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| Real data | IAM Letter LOW | **Absent from this machine.** `experiments/paper_pipeline/config.yaml:paths.source_dir` = `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph_source` (Picasso). Checked repo, `~/research/data`, Sandisk mount (not mounted). | Substituted a labelled surrogate |
| Surrogate corpus | `bench_native.py iam` | 1,180 graphs; mean edges **3.038** vs target 3.07 (−1.0%); mean nodes 3.82; median 4; max 7; mean density 0.585; histogram 2:112, 3:381, 4:429, 5:169, 6:74, 7:15; all connected | **1,180/1,180 identical canonical strings, 0 mismatches; 250/250 identical Levenshtein distances, 0 mismatches** |
| Edge cases | Self-loops (4 families), singleton, single edge, empty string, `"CC"`, `"NNNN"` on a 1-element list, `V`*20 | All compared byte-exact | pass |
| Failure paths | Disconnected undirected; directed with no spanning-out-tree root; `initial_node` = −1/4/99; 6 invalid alphabets; CDLL capacity overflow; 1 µs timeout | Type and message identical between engines in every case | pass |
| Scale / performance | n = 3..10, i7-13700KF | 23.1x at n=3 rising to 1024.7x at n=9 (canonical); n=10 = 20.5 s Python vs 0.0219 s C++ | pass |
| Fallback | `.so` moved out of site-packages, full suite re-run | `450 passed, 276 skipped in 14.74s` | pass |
| Environment | Debian 6.1, gcc 12.2.0, cmake 3.25.1, ninja (env-local), nanobind 2.14.0, scikit-build-core 1.0.3, Python 3.11.15, pytest 9.1.1, 24 logical cores | `build_hash 298fc1188bf1b051`, `isa_level x86-64-v3`, `avx512f 0`, `ndebug 1` | pass |

**Install-path verification** (the IsalSR wasted-cycle failure mode):

```
isalgraph: .../agent-a3c0c2a5ce14cac9e/src/isalgraph/__init__.py
_native  : .../isalgraph-cpp/lib/python3.11/site-packages/isalgraph/core/_native.cpython-311-x86_64-linux-gnu.so
```

`site-packages/isalgraph/core/` contains **only** the `.so` — no `.py` that
could shadow the worktree. `isalgraph.core.canonical.__file__` resolves to the
worktree. `PYTHONPATH` was never set.

**Environment repair:** the `isalgraph-cpp` env shipped with a half-upgraded
pip (`ImportError: cannot import name 'get_runnable_pip'`). I removed
`site-packages/pip*` and re-ran `ensurepip --upgrade` (pip 24.0). This touched
only my dedicated env, not the peer's.

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off:**

- **Reachability and range checks run Python-side even on the cpp path.**
  Costs an O(V+E) Python DFS per `graph_to_string` call; buys byte-identical
  error text, since the reference message embeds a Python `set` repr whose
  order follows CPython hash-slot layout. Negligible against an encoder that
  is far more expensive, and it removes a whole class of fragile C++ code.
- **The Python backend *refuses* `timeout_s` rather than ignoring it.** Costs
  an error where a caller might expect a no-op; buys the guarantee that an
  evaluation harness can never believe a run was bounded when it was not.
- **Dense bitmatrix up to 2,048 nodes.** Costs a 4 MiB allocation at the
  boundary and leaves the sorted-vector branch untested; buys O(1) membership
  in the hottest loop.
- **Threading exists but defaults to 1.** Costs API surface; buys the option
  without the SLURM oversubscription hazard. Measurements say it is a
  pessimisation at paper-relevant sizes (§O6 of the optimisation log).

**Assumptions I proceeded on** (all messaged to `main` when made):

- IAM data absence is permanent for this wave; the surrogate is acceptable if
  labelled. **If wrong:** every number in the IAM section must be regenerated.
- `main` will land the errors.py builtin mixins exactly as specified. **If
  wrong:** 10 skipped tests stay skipped and the `algorithms/*.py` shim must
  stay.
- `wave-viz` adds trace emission only, with no change to existing signatures
  or instruction semantics. **If wrong:** the greedy differential could break,
  because it compares against `GraphToString(g).run(v)`.

**Open questions for the orchestrator:**

- **Does the paper pipeline ever mix directed and undirected graphs in one
  deduplication pass?** The canonical string does not encode directedness (see
  below). If it does, that pass has a latent collision bug independent of this
  port. I keyed my own test on `(directed, string)` and moved on.

## 8. Coordination

**Messages sent:** one to `main` at the start, covering (1) IAM absence and
the surrogate substitution, (2) the `errors.py` / `ValueError` conflict with
`test_algorithms.py:157,164`, (3) the stale `_native/` path in the `errors.py`
docstring.

**Messages received and how they changed the work:** `main` replied
approving the surrogate but requiring the realised distribution be reported
next to the IAM LOW target — done, and it caught a 16% edge-count overshoot in
my first surrogate, which I then calibrated to −1.0%. `main` also **corrected
my proposed exception fix**: I had suggested `EncodingError(IsalGraphError,
ValueError)`, which is wrong because `EncodingError` has descendants on both
sides of the ValueError/RuntimeError split. The landed contract puts the
builtin on the *leaves*, adds a new `EncodingStuckError`, and makes
`InvalidNodeError` an `IndexError` and deliberately not a `ValueError`. I
implemented `errors.hpp` and the nanobind translators against that contract,
including per-class builtin fallbacks so the mapping is correct both before
and after integration. `main` also asked me to make the shim unconditional and
self-documenting rather than a runtime `issubclass` test, and to audit whether
any site iterates the *output* graph's adjacency to make a choice.

**Contracts I depend on and confirmed unchanged:** the frozen reference
modules (never edited — `git diff` touches none of the six); `errors.py`
(never edited); both `__init__.py` (never edited).

**Ordering audit requested by `main` — result:** confirmed by inspection.
`canonical.py:313,331` and `graph_to_string.py:209,225` read
`og.neighbors(...)` for **membership only** (`in` / `not in`), never
iterating it to select anything, so backing the output graph with a dense
bitmatrix is sound. The only site that iterates and *returns the first* is
`graph_to_string.py:286` on the **input** graph, which is exactly what the
marshalling contract protects. The V/v candidate collection at
`canonical.py:233,276` iterates the input graph but explores every candidate
and takes a minimum, so it is order-independent. Full table in
`CPP_OPTIMIZATION_LOG.md` §"Ordering audit".

## 9. Deliberately not done

- **Rewiring `src/isalgraph/__init__.py` and `src/isalgraph/core/__init__.py`** — `main` owns these.
- **Editing `errors.py`** — `main` owns it; I mirrored the agreed contract in C++ instead.
- **Editing any of the six frozen reference modules** — forbidden, and none is touched.
- **Ladder item O4 (restrict the candidate scan)** — analysed and rejected as unsound/redundant; reasoned in the log rather than shipped as a no-op.
- **A measured before/after for O2 (mutable buffer)** — structural from the first commit; recorded as unmeasured rather than given a fabricated number.
- **Running the Picasso build** — no cluster access from this worktree.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| errors.py mixins not yet landed | **high** | 10 tests skipped; the `algorithms/*.py` shim downgrades `DisconnectedGraphError` to bare `ValueError` until deleted. Delete the shim and re-run `tests/unit/test_algorithms.py` to confirm it was a no-op. | orchestrator |
| IAM numbers are a surrogate | **high** | Every IAM figure must be relabelled or regenerated on Picasso before it reaches the response letter. The surrogate also **cannot** reach the 12-node regime and the 3.07-edge mean at once, so it cannot by itself answer Reviewer 3's cap question. | orchestrator / next wave |
| Canonical string omits directedness | medium | Dedup over a mixed corpus must key on `(directed, string)`. | orchestrator |
| Sorted-vector path (n > 2048) untested | medium | Compiles, never executed. Nothing to diff against — the reference is intractable there. | next wave |
| Peer's trace edits may perturb greedy | medium | My greedy differential compares against `GraphToString(g).run(v)`. Re-run `tests/native/` after merging `wave-viz`. | orchestrator |
| `.so` does not rsync to Picasso | medium | Documented in `CPP_BUILD.md`; the failure mode is silently running a stale engine. Job scripts should assert `build_info()["build_hash"]`. | next wave |
| `EncodingStuckError` fallback | low | Until errors.py has the class, C++ raises bare `RuntimeError` — which is what the reference raises, so parity holds either way. | orchestrator |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | Builds from a clean worktree; `build_info()` reports expected ISA and a stable hash | yes | `Successfully built isalgraph`; `isa_level x86-64-v3`, `avx512f 0`, `ndebug 1`, `build_hash 298fc1188bf1b051`; `test_build_is_release_cxx17_and_portable_isa` |
| 2 | Property parity through the cpp backend (round-trip, invariance/dedup, canonicality, idempotence) | yes | 17 tests in `test_native_properties.py`, every call `backend="cpp"`; structured + random, self-loops, directed |
| 3a | `canonical_string` / `pruned_canonical_string` byte-exact, >=3,000 graphs, zero mismatches | yes | 3,079-graph corpus; `compared >= 3000` asserted in-test; 0 mismatches |
| 3b | `string_to_graph` identical graphs, >=2,000 strings + hand cases | yes | 2 x (38 hand + 1,100 random) = 2,276; plus iteration-order and re-encode tests |
| 3c | `graph_to_string` greedy **byte-exact** | yes | 11,000+ (graph, start node) pairs, 0 mismatches, no fallback criterion needed |
| 3d | `levenshtein` exact | yes | 4,000 random pairs + 7 edge cases |
| 3e | Exception parity, type **and** message | yes | `test_native_errors.py`, 18 passing parity assertions |
| 4 | Full suite passes with the `.so` physically deleted | yes | `450 passed, 276 skipped in 14.74s` after `mv`-ing the `.so` aside |
| 5 | Partial paper results on real data | **partial** | IAM absent; labelled surrogate matched to −1.0% on mean edges. 1,180 identical strings, 250 identical distances, 0 mismatches. Per-bucket speedups reported. **Surrogate max is 7 nodes, so the 12-node regime is not exercised.** |
| 6 | Best-of-9, median-of-4, 3 warmups, alternated, raw JSON committed | **partial** | Protocol implemented and used; at n >= 9 the repetition count is reduced to fit a 25 s budget and the **actual counts are written into the JSON** per measurement rather than the nominal ones |
| 7 | ruff clean, mypy --strict clean, suite green | yes | `All checks passed!`; `Success: no issues found in 22 source files`; 605 passed / 0 failed |
| 8 | `CPP_BUILD.md` documents local and Picasso builds | yes | Includes `module load gcc/13.2.0`, login-node wheelhouse pre-caching, `--no-build-isolation`, and the `.so`-does-not-rsync trap |

**Overall.** I am confident in the parity claim: 3,079 graphs byte-exact on
both canonical variants, 11,000+ byte-exact greedy encodings, and zero
mismatches on the surrogate corpus, all driven explicitly through
`backend="cpp"` so a silent fallback could not manufacture a pass. I am
confident in the fallback: the suite returns exactly the 450-test baseline
with the artifact deleted. I am confident in O1, the memoisation result
(25x–109x), which is measured by A/B toggle on identical inputs and is the
number Reviewer 3's question actually needs.

I am **least** confident in criterion 5, and that is what `main` should
scrutinise first — not because the parity evidence is weak, but because the
corpus is not the real one. The surrogate tops out at 7 nodes: honouring
"connected" and "mean 3.07 edges" simultaneously *forces* a mean node count
near 3.85, so the corpus that matches the published statistics is structurally
incapable of exercising the 12-node cap that Reviewer 3 is attacking. The real
IAM Letter graphs almost certainly reach 12 nodes by being disconnected
forests, which this encoder rejects outright. The per-node-count table
(23x at n=3 to 1025x at n=9) is the defensible scalability evidence; the IAM
section is a parity demonstration on a stand-in and should be re-run on
Picasso before any of it reaches the response letter.

Second thing to scrutinise: the `algorithms/*.py` `ValueError` shim. It is
deliberately unconditional, so between now and the errors.py change the
`ExhaustiveG2S` path raises a plain `ValueError` instead of
`DisconnectedGraphError`. Deleting it is a one-step operation and
`tests/unit/test_algorithms.py` proves it was a no-op.
