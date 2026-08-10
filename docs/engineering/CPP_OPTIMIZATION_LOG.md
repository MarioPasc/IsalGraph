# C++ engine optimisation log

Every entry records a change, the measured effect, and whether it was kept.
Negative results are recorded with the same detail as wins: an optimisation
that did not pay is a result, and re-attempting it later costs more than
writing it down now.

**Rule observed throughout:** parity first, performance second. The faithful
port landed and passed the full differential suite before any optimisation was
attempted, and the suite was re-run after every entry below. No entry was kept
that changed a single output byte.

---

## Measurement protocol

| Parameter | Value |
|---|---|
| Warmups | 3, discarded |
| Repetitions | best-of-9 |
| Blocks | median-of-4 |
| Engine ordering | alternated inside each block, same thermal state |
| Harness | `tests/native/bench_native.py` |
| Raw data | `docs/engineering/results/*.json` |

`best-of` rather than mean: scheduler and cache noise can only *add* time, so
the minimum is the least biased estimator of the true cost.

At `n >= 9` a single Python reference execution already takes seconds, so the
full 39-execution protocol would cost about an hour per bucket. Rather than
quietly dropping repetitions, `measure_detailed()` probes one execution,
derives the largest `(best_of, median_of)` fitting a 25 s budget, and writes
the counts actually used into the JSON beside the number they produced. Fewer
repetitions is defensible here only because the effect is a factor of ~10^3,
orders of magnitude above the run-to-run spread.

### Hardware

| Field | Value |
|---|---|
| CPU | 13th Gen Intel Core i7-13700KF, 24 logical cores |
| Compiler | gcc 12.2.0 |
| ISA target | `x86-64-v3` (AVX2 + FMA, no AVX-512) |
| Build hash | `298fc1188bf1b051` |
| Python | 3.11.15 |
| Build | Release, `-O3 -DNDEBUG -fno-plt -funroll-loops`, IPO on |

---

## O0 — Faithful port (baseline)

Direct transliteration of `cdll.py`, `sparse_graph.py`, `string_to_graph.py`,
`graph_to_string.py`, `canonical.py` and `canonical_pruned.py` to C++17.

Three representation choices were made at this stage because they are
*structural* — retrofitting them later would have meant rewriting the search:

1. **Ordered input adjacency across the FFI.** See §"Why greedy parity works".
2. **`i2o` / `o2i` as `vector<int32_t>` with a `-1` sentinel** instead of
   `dict[int, int]`. Input node ids are contiguous, so the dictionary was
   never buying anything but hashing.
3. **Dense byte adjacency matrix for the output graph** up to 2,048 nodes,
   sorted vectors beyond. The output graph is only ever *queried*, never
   iterated, so no ordering contract applies to it (verified by inspection —
   see §"Ordering audit").

**Result:** byte-exact on 3,079 graphs and 4,000 string pairs, first run.
Kept.

---

## O1 — Memoise `generate_pairs_sorted_by_sum` **(kept — largest win)**

The Python reference calls `generate_pairs_sorted_by_sum(og.node_count())` at
*every* recursion frame, rebuilding and re-sorting `(2m+1)^2` pairs. That is
`Theta(m^2 log m)` per frame for a value that depends only on `m`. The C++
engine caches it in a `thread_local unordered_map<int32_t, vector<Pair>>`.

A/B via `_native.set_pairs_memo(bool)`, both settings verified to produce
identical strings on the same inputs.

| n | memo on (s) | memo off (s) | gain |
|---|---|---|---|
| 6 | 0.00073 | 0.0186 | **25.5x** |
| 8 | 0.0182 | 0.762 | **41.9x** |
| 9 | 0.0620 | 3.573 | **57.6x** |
| 10 | 0.219 | 23.83 | **108.6x** |

The gain grows with `n` because the frame count grows exponentially while the
per-frame rebuild cost grows polynomially, so the rebuild dominates ever more
completely. **This single change accounts for most of the speedup**; the
translation to C++ accounts for far less than the memo does.

> **Reviewer 3 asks whether the ordered displacement lists are recomputed at
> each iteration or precomputed.** In the published Python implementation they
> are **recomputed at every recursion frame**. The native engine precomputes
> and memoises them on first use per distinct `m`. The tables above quantify
> what that costs: 25x at 6 nodes rising to 109x at 10 nodes, on identical
> outputs. This is the number to quote in the response letter.

---

## O2 — Mutable output buffer instead of string concatenation **(kept)**

The reference builds `prefix + mov + "V"` at every frame, allocating and
copying an `O(L)` string per frame. The engine keeps one `std::string buf`,
appends, recurses, and `resize()`s back on the way out; only leaves copy.

**Not A/B'd.** Adopted from the first commit because retrofitting it means
rewriting the recursion's return-value plumbing, and holding a second
implementation solely to benchmark it was judged worse value than the entries
that could be toggled. The asymptotic argument is not in doubt — it removes an
`O(L)` allocation from a frame that otherwise does `O(1)` array writes — but
this row carries **no measured number**, and it should not be quoted as if it
did.

---

## O3 — Running counters for `nleft` / `eleft` and `i2o` occupancy **(kept)**

`nleft` and `eleft` are already parameters in the reference, so this is
faithful rather than novel; the change is that `i2o` occupancy is a `-1`
sentinel test on a flat vector rather than a `not in dict` probe. Folded into
O0 and not separately measured.

---

## O4 — Restrict the candidate scan **(not implemented — see O5)**

The intended optimisation was to narrow the displacement-pair scan to pairs
that can actually qualify. On inspection this is not sound in general: the
first *applicable* pair in cost order is what defines the output, and deciding
applicability without walking the CDLL to the tentative position is what the
scan already does. The available narrowing (skip `b` entirely while testing
the V branch, since V ignores `b`) is already implicit — the V branch is
evaluated before `ts` is computed, exactly as in the reference. **No change
made.**

---

## O5 — Branch and bound on the search **(kept — modest)**

Not in the planned ladder; added after O1 because the profile made the
remaining cost visibly search-bound.

Every remaining logical edge costs at least one instruction, and
`eleft >= nleft` always holds (each uninserted node is attached by a distinct
uninserted edge), so `|buf| + eleft` is a valid lower bound on the length of
any completion. A subtree is pruned when that bound *strictly* exceeds the
incumbent's length. Pruning at equality would be wrong: a completion of
exactly the incumbent length can still win lexicographically.

The incumbent is carried across starting nodes, since the final answer is the
global minimum over all of them.

A/B via `_native.set_branch_and_bound(bool)`; identical outputs both ways.

| n | bound on (s) | bound off (s) | gain |
|---|---|---|---|
| 6 | 0.00073 | 0.00077 | 1.05x |
| 8 | 0.0182 | 0.0210 | 1.15x |
| 9 | 0.0620 | 0.0806 | 1.30x |
| 10 | 0.219 | 0.315 | **1.44x** |

Kept: the gain grows with `n`, which is the regime the revision cares about,
and the cost is one comparison per frame. But it is a 1.4x, not a 10x — the
bound is weak because the (length, lex) objective forces exploration of every
subtree that could tie.

---

## O6 — Threading the starting-node loop **(implemented, OFF by default — largely a NEGATIVE result)**

The `for v in range(n)` loop is embarrassingly parallel. Implemented with an
explicit `threads` parameter, each worker holding its own `Search` (its own
graph, CDLL and incumbent), merged at the end.

| n | 1 thread (s) | 2 (s) | 4 (s) | 8 (s) | 4t gain |
|---|---|---|---|---|---|
| 6 | 0.00073 | 0.00108 | 0.00132 | 0.00230 | **0.55x (slower)** |
| 8 | 0.0182 | 0.0165 | 0.0130 | 0.0154 | 1.40x |
| 9 | 0.0620 | 0.0522 | 0.0416 | 0.0424 | 1.49x |
| 10 | 0.219 | 0.184 | 0.163 | 0.171 | 1.35x |

**Negative result at the sizes this paper actually uses.** At 6 nodes four
threads are *1.8x slower* than one: thread creation costs more than the whole
search. Even at 10 nodes the 4-thread gain is 1.35x against a 4x core budget —
about 34% efficiency — because carrying the incumbent across starting nodes
serially is itself a pruning mechanism that per-thread incumbents forfeit, and
because 8 threads are never better than 4.

The IAM surrogate has a mean of 3.8 nodes, so **threading would make the
paper's real workload slower.** Default is 1 and should stay 1.

`hardware_concurrency()` is never consulted: inside a SLURM cgroup it reports
the whole node and silently oversubscribes, a failure that only appears on the
cluster.

---

## O7 — `-march=native` **(rejected, not measured)**

Deliberately not adopted, and the option `ISALGRAPH_NATIVE_MARCH` defaults
OFF. 178 of 333 Picasso nodes lack AVX-512, so a `native` build presents as a
random fraction of SLURM tasks dying with SIGILL — which reads like flaky
hardware, not a build fault. The pinned `x86-64-v3` target also keeps one
build hash valid across the whole fleet, which is what makes
`build_info()["build_hash"]` a usable staleness check.

This search is branch-bound and integer-bound with no vectorisable inner
loop, so the expected gain from a wider ISA is near zero anyway. Not measured,
because the portability argument decides it regardless of the number.

---

## End-to-end speedup, per node-count bucket

`docs/engineering/results/speedup.json`. Erdos-Renyi, p = 0.35, connected.
Correctness is asserted alongside every timing: a fast wrong answer is not a
speedup.

| nodes | canonical Python (s/graph) | canonical C++ (s/graph) | **speedup** | pruned speedup |
|---|---|---|---|---|
| 3 | 0.000085 | 0.0000037 | 23.1x | 26.9x |
| 4 | 0.000426 | 0.0000036 | 118.9x | 91.1x |
| 5 | 0.00310 | 0.0000123 | 253.1x | 211.1x |
| 6 | 0.0178 | 0.0000367 | 484.7x | 412.8x |
| 7 | 0.0866 | 0.000145 | 596.9x | 474.4x |
| 8 | 0.766 | 0.00118 | 649.0x | 651.7x |
| 9 | 3.42 | 0.00334 | 1024.7x | 829.9x |
| 10 | 20.5 | 0.0219 | 937.0x | 868.4x |

**Never quote a single aggregate figure.** The FFI marshalling cost is fixed
per call while the search cost is exponential in `n`, so the ratio is 23x at
3 nodes and ~1,000x at 9. An average over a corpus is a statement about the
corpus's node distribution, not about the engine.

The ratio flattens between n = 9 and n = 10 (1025x, then 937x). Both engines
are then deep in the exponential regime and the constant factor has saturated;
the dip is within the noise of the reduced repetition count at n = 10 (one
graph, one block).

---

## IAM Letter LOW — **surrogate**, not the real dataset

The IAM Letter dataset is **not present on this machine**.
`experiments/paper_pipeline/config.yaml:paths.source_dir` points at
`/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph_source`, a
Picasso path. Every figure in this section is a **surrogate** and must be
labelled as one wherever it is reproduced, including figure captions.

Target versus realised, so a reader can judge the match:

| Statistic | IAM LOW target | Surrogate realised |
|---|---|---|
| Graphs | 1,180 | 1,180 |
| Mean logical edges | 3.07 | **3.038** (−1.0%) |
| Max nodes | <= 12 | 7 (cap 12 never reached) |
| Connected | yes | yes, all 1,180 |
| Mean nodes | — | 3.82 |
| Median nodes | — | 4 |
| Mean density | — | 0.585 |
| Node histogram | — | 2:112, 3:381, 4:429, 5:169, 6:74, 7:15 |

Connectedness is required, and a connected graph on `n` nodes has at least
`n − 1` edges, so a 3.07-edge mean pins the mean node count near 3.85. That is
why the surrogate has no graphs near the 12-node cap: **the surrogate cannot
exercise the large-`n` regime and the 3.07 target simultaneously.** The
real IAM Letter graphs reach 12 nodes because many are disconnected forests,
which this encoder cannot process at all. Anyone using these numbers to argue
about the 12-node cap should use the per-bucket table above instead.

### Engine agreement on the surrogate

| Check | Compared | Mismatches |
|---|---|---|
| Canonical strings, Python vs C++ | **1,180** | **0** |
| Levenshtein distances, Python vs C++ | **250 pairs** | **0** |

### Per-bucket speedup on the surrogate

| nodes | graphs | Python (s/graph) | C++ (s/graph) | speedup |
|---|---|---|---|---|
| 2 | 8 | 0.0000098 | 0.0000012 | 8.3x |
| 3 | 8 | 0.000082 | 0.0000034 | 24.4x |
| 4 | 8 | 0.000420 | 0.0000059 | 71.5x |
| 5 | 8 | 0.00307 | 0.0000202 | 152.0x |
| 6 | 8 | 0.0170 | 0.0000700 | 242.9x |
| 7 | 8 | 0.0808 | 0.000273 | 296.2x |

Weighted by the surrogate's own node histogram the mean speedup is about 60x —
which is exactly why the per-bucket table is the honest presentation. The
corpus is dominated by 3- and 4-node graphs where the FFI overhead is a large
fraction of the total.

---

## Why greedy parity works (the one decision that made it possible)

`GraphToString._find_new_neighbor` returns **the first** uninserted neighbour
obtained by iterating a Python `set[int]`. CPython iterates small-int sets in
slot order, `i & (table_size - 1)`, not ascending value order: for `{2, 9}`
with table size 8, `9` lands in slot 1 and `2` in slot 2, so Python yields
`9, 2`. A `std::set<int32_t>` yields `2, 9` and produces a different — equally
valid, but different — greedy string.

`backends._marshal` therefore hands each adjacency across the FFI as
`list(graph.neighbors(u))`, in CPython's own order, and `InputGraph` stores it
verbatim. "First neighbour" in C++ is then by construction "first neighbour"
in Python. Cost: zero. Result: byte-exact greedy parity on 5,000+ (graph,
start node) pairs, first attempt.

The same reasoning applies in reverse to `string_to_graph`: the engine returns
its edges in `add_edge` call order and the Python wrapper replays them in that
order, so the reconstructed adjacency sets match not only in contents but in
*iteration order*, because CPython resolves set slot collisions by insertion
order. `test_string_to_graph_adjacency_iterates_identically` pins this, and
`test_decoded_graphs_encode_identically` pins its consequence.

### Ordering audit

Requested by `main`: confirm no site iterates the *output* graph's adjacency
to make a choice. Verified by inspection of the frozen reference:

| Site | Reads | Verdict |
|---|---|---|
| `canonical.py:313`, `:331` | `og.neighbors(...)` | membership test only (`in` / `not in`) — safe |
| `canonical.py:233`, `:276` | `ig.neighbors(...)` | iterated, but every candidate is explored and the minimum taken — order-independent |
| `canonical_pruned.py:236`, `:284` | `ig.neighbors(...)` | same, plus a `max()` filter — both order-independent |
| `graph_to_string.py:209`, `:225` | `og.neighbors(...)` | membership test only — safe |
| `graph_to_string.py:286` | `ig.neighbors(...)` | **returns the first** — order-dependent, and the reason for the marshalling contract |
| `canonical.py:134`, `graph_to_string.py:327` | `ig.neighbors(...)` | reachability DFS, result is a count — order-independent |
| `canonical_pruned.py:77` | `ig.neighbors(...)` | BFS distance counts — order-independent |

**No site iterates the output graph's adjacency to select anything.** Backing
it with a dense bitmatrix is therefore sound.

---

## Defects found and fixed during the port

1. **Self-loop edge counting.** Deriving `logical_edge_count` from
   `sum(len(adj)) // 2` undercounts a self-loop in an undirected graph, which
   occupies one adjacency slot but increments `SparseGraph._edge_count`
   *twice*. The encoder would have stopped one edge early. Fixed by marshalling
   the graph's own counter instead of deriving it. Caught before it reached a
   test, by reasoning about the self-loop family rather than by a failure.

2. **`pytest.importorskip` does not catch a deleted `.so`.** pytest catches
   only `ModuleNotFoundError` by default, but under a scikit-build-core
   editable install a deleted `.so` still resolves through the import redirect
   and fails with a plain `ImportError: cannot open shared object file`. The
   extension-absent run died at collection instead of skipping. Fixed with
   `exc_type=ImportError`. The library-level fallback in `backends.py` was
   already correct — it catches `ImportError` — so only the harness was wrong.

3. **The canonical string does not encode directedness.** The 3-node directed
   path and the 3-node undirected path canonicalise to the same string. The
   string is a complete invariant *within* a directedness class only, so any
   deduplication over a mixed corpus must key on `(directed, string)`. This is
   a property of the encoding, not a defect in either implementation, but it
   is a live trap for the evaluation pipeline.
