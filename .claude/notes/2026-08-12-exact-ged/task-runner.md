# task-runner — T-03 compute driver

Wave `2026-08-12-exact-ged`. Base `29886f879a9190ad8c869eaf0979c7cf8af364ef` on `main`.
Environment: `~/.conda/envs/isalgraph-cpp/bin/python`, 3.11.15, numpy 1.26.4, networkx 3.6.1.

---

## Mission

The upper-triangle pair index, the frozen stage-1 sampler, the chunked resumable GED
runner and the shard merge — CONTRACTS §6, §7, §8. The code will spend on the order of
a thousand CPU-hours computing 3,897,911 exact graph edit distances whose values go into
a Pattern Recognition table, so the index arithmetic and the resume logic got the effort.

---

## What I built

| File | Lines | What it is |
|---|---:|---|
| `benchmarks/real_data/eval_setup/ged_pair_index.py` | 439 | Forward/inverse linear upper-triangle index, three independent inverses, even chunk splitting |
| `benchmarks/real_data/eval_setup/ged_sampling.py` | 573 | The pre-registered stage-1 design, verbatim; `pair_list.npz` + `sampling_report.json` |
| `benchmarks/real_data/eval_setup/ged_exact_runner.py` | 1270 | CONTRACT C chunk driver: process pool, in-place checkpoint, resume, `--seed-from`, SIGTERM |
| `benchmarks/real_data/eval_setup/ged_merge_shards.py` | 612 | CONTRACT D merge + gate 4 |
| `tests/unit/test_ged_pair_index.py` | 94 tests | |
| `tests/unit/test_ged_sampling.py` | 24 tests | |
| `tests/unit/test_ged_exact_runner.py` | 30 tests | |
| `tests/unit/test_ged_merge_shards.py` | 22 tests | |

All four modules run as `python -m benchmarks.real_data.eval_setup.<mod>` from the repo
root and import cleanly as bare modules from inside `eval_setup/`. The package branch is
preferred over `sys.path.insert` so `ged_pair_index` is never loaded twice under two
module identities — a hazard the incumbent `validate_ged_bounds.py` pattern has but does
not currently trip, because it imports no sibling that defines an exception class.

Nothing imports `isalgraph`, `torch`, `gklearn`, `ged_backends` or `export_graphs`.
`ged_bounds` is imported **lazily, inside `_edit_costs`**, so the cost model comes from
`UNIT_COSTS`/`GRAPHEDX_COSTS` rather than restated numbers (CONTRACTS §1) without making
the runner unimportable if that module ever moves.

---

## The index inversion and how it is proven correct

The forward map is frozen:

```
k(i, j) = i*N - i*(i+1)//2 + (j - i - 1),   0 <= i < j < N,   0 <= k < C(N,2)
```

Write `R(i) = i*N - i*(i+1)//2`. The identity `R(i) = sum_{a<i} (N - a - 1)` holds, which
is what makes rows contiguous; `test_row_start_equals_the_cumulative_row_widths` checks it
directly against the running total rather than against the algebra.

Inverting means finding the unique `i` with `R(i) <= k < R(i+1)`. `R(i) <= k` rearranges to

```
i^2 - i(2N - 1) + 2k >= 0     with smaller root  i* = ((2N-1) - sqrt((2N-1)^2 - 8k)) / 2
```

**Why this is where a silent failure lives.** An off-by-one in the inverse transposes GED
values between pairs. Every merged matrix stays symmetric, finite, positive and entirely
plausible; gate 4 passes; the correlations are simply wrong. Nothing downstream is looking
for it. So the module carries three inverses and never trusts one:

1. **`pair_from_index` — scalar, no floating point at all.** The seed uses `math.isqrt`,
   which is exact, so it is off by at most one; a correction loop then closes on
   `R(i) <= k < R(i+1)` and only returns from inside that branch. The loop is bounded at 64
   iterations and *raises* on non-convergence rather than returning a best guess.
2. **`pairs_from_indices` — vectorised, `float64` seed + vectorised correction loop.** This
   is the contract-mandated shape. For `N = 2059` the discriminant peaks near `1.7e7`, far
   inside float64's exact-integer range, so a correctly-rounded `sqrt` is within one of the
   truth. It then runs `_verify_inverse`, which re-derives `k` from `(i, j)` elementwise and
   raises on any mismatch. **That verification is unconditional, not a debug flag** — it is
   a handful of integer ops per pair against seconds of solver time, and the failure it
   guards is silent.
3. **`pairs_from_indices_searchsorted` — vectorised, no floating point.** Materialises the
   row-start table and bisects. Independent in method, not just in code, which is what makes
   agreement between (2) and (3) evidence rather than a tautology.

Coverage actually run:

- **Exhaustive**, every `k` for every `N` in `2..200` — **1,333,300 indices** = C(201,3),
  asserted as that exact count so the test cannot silently shrink. Run through all three
  inverses (three separate tests), and all three cross-checked against each other
  exhaustively for `N` in `2..120`.
- **10^5 random `k` per production cohort size**, `N ∈ {769, 1180, 1253, 2059}`, through the
  vectorised and searchsorted paths, with 500 scalar spot-checks each.
- **Row boundaries**, the classic off-by-one site: for each production `N`, both `R(i)` and
  `R(i+1) - 1` for every `i`, plus `k = 0 -> (0,1)` and `k = C(N,2)-1 -> (N-2, N-1)`.
- **Out-of-range indices raise** rather than clamp, in all three inverses. A clamp here would
  be exactly the silent corruption the whole exercise is about.
- `np.triu_indices(N, 1)` order is asserted equal to the linear index order, because both
  `ged_sampling` and `ged_merge_shards` rely on that correspondence.

**Chunk splitting.** `base, rem = divmod(total, n_chunks)`;
`start(t) = t*base + min(t, rem)`. Sizes differ by at most one and the remainder lands on
the **low-numbered** chunks. `test_remainder_goes_to_the_low_numbered_chunks_not_a_ragged_tail`
pins the exact sequence `[15,15,14,14,14,14,14]` for `split_range(100, 7, ·)` and asserts the
size sequence is non-increasing, i.e. no chunk is shorter than a later one. That is the
SCBI two-hour-floor requirement expressed as a test. `n_chunks > C(N,2)` yields empty
chunks rather than an error; `n_chunks == 1` owns everything; `n_chunks` not dividing the
total is the normal case and is covered by 49 parametrised partition tests.

---

## Sampling design as implemented

CONTRACTS §8 verbatim. Core = **simple** random sample of `K` graphs (all `C(K,2)` pairs);
halo = `q` partners per non-core graph drawn uniformly without replacement from the other
`N-1`; top-up = every **population**-non-empty stratum raised to `min(f, |stratum|)`.
Strata = unordered pair of node-count bins `{2-5, 6-9, 10-12}` (6 cells) × unordered pair of
density quintiles (15 cells) = 90. Seed 42. `K/q/f` are CLI flags defaulting to 180/10/30.

RNG consumption order is fixed — core draw, then non-core graphs in ascending index order,
then strata in ascending id order — so the sample is reproducible from the design note alone.

### Amendment 2026-08-12: `np.searchsorted(..., side="right")`

CONTRACTS §8 fixes the quantiles and the use of `searchsorted` but **not the side**. The
first implementation took numpy's default, `side="left"`, which puts a density equal to an
edge in the *lower* bin. Consequence: when `q80` equals the maximum density the top quintile
is unreachable. That is not hypothetical — after `min_nodes=2` the AIDS cohort contains `n=2`
graphs, and a connected two-node graph has exactly one edge, so its density is exactly `1.0`.

Corrected to `side="right"` on the orchestrator's instruction, before any production pair was
computed. This is a disambiguation of an under-specified detail, not a change to the
pre-registered design: `K/q/f` remain 180/10/30, the seed remains 42, and the core remains a
simple random sample. The orchestrator is recording it in the design changelog.

`test_the_top_quintile_stays_reachable_when_q80_equals_the_maximum_density` pins it on a
fixture where `q80 == max == 1.0` by construction. The guard bites rather than restating the
implementation — verified directly:

```
quantile edges: [0.45333333 0.57333333 0.8  1.0]   max density: 1.0
side=left   bins present=[0, 1, 2, 3]   count in quintile 4 =  0     <- top quintile lost
side=right  bins present=[0, 1, 2, 4]   count in quintile 4 = 40
```

**The mirror case, for the record.** `side="right"` empties the *bottom* quintile if `q20`
equals the minimum density. That needs at least a fifth of the corpus sharing one exact
density value at the floor, which the dry run does not exhibit — the graph-level quintile
populations came out `[154, 154, 151, 151, 159]` against a perfect fifth of 153.8. Worth
re-checking once the real AIDS densities are in hand; it is a one-line assertion.

### Dry run, 769 synthetic graphs (seed 20260812 for the population, 42 for the sample)

Population fabricated to span the frozen bins: `n ∈ [2,12]` uniform, `m` uniform between
connected (`n-1`) and complete (`n(n-1)/2`). **Not real AIDS** — real densities will differ
and so will these counts.

```
K=180 q=10 f=30 seed=42
graphs=769  population pairs=295296  sampled pairs=22106  (7.49%)
core pairs=16110  halo new=5860  top-up=136
graphs covered=769 complete=True
strata: 90 total, 90 non-empty, 0 empty; all meet floor=True
density quintile edges: ['0.4381', '0.6127', '0.7857', '1.0000']
graphs per density quintile: [154, 154, 151, 151, 159]
```

22,106 pairs, just under the design note's 22,500–24,500 envelope. Core 16,110 is exact.
**All 90 strata are now non-empty** and the graph-level quintiles are balanced against a
perfect fifth of 153.8 — both are consequences of the `side="right"` correction. Under
`side="left"` this same population gave 60 non-empty strata, 30 empty, and only 18 top-up
pairs; the corrected binning finds 30 more strata to fill and the top-up rises to 136.

```
id   size cells     dens cells      pop  sampled  floor
0    2-5/2-5        0,0              15       15     15     <- smaller than f, taken in full
1    2-5/2-5        0,1             204       38     30
2    2-5/2-5        0,2             360       82     30
3    2-5/2-5        0,3             186       33     30
4    2-5/2-5        0,4             798      118     30
5    2-5/2-5        1,1             561       33     30
6    2-5/2-5        1,2            2040      194     30
7    2-5/2-5        1,3            1054       82     30
8    2-5/2-5        1,4            4522      327     30
9    2-5/2-5        2,2            1770      193     30
10   2-5/2-5        2,3            1860      169     30
11   2-5/2-5        2,4            7980      657     30
12   2-5/2-5        3,3             465       32     30
13   2-5/2-5        3,4            4123      266     30
14   2-5/2-5        4,4            8778      524     30
15   2-5/6-9        0,0             426       98     30
16   2-5/6-9        0,1            2792      269     30
17   2-5/6-9        0,2            4596      544     30
18   2-5/6-9        0,3            2645      246     30
19   2-5/6-9        0,4            9557      769     30
20   2-5/6-9        1,1            2142      140     30
21   2-5/6-9        1,2            5684      396     30
22   2-5/6-9        1,3            4469      287     30
23   2-5/6-9        1,4            9025      514     30
24   2-5/6-9        2,2            3360      251     30
25   2-5/6-9        2,3            6176      482     30
26   2-5/6-9        2,4            8588      508     30
27   2-5/6-9        3,3            2294      144     30
28   2-5/6-9        3,4           10431      634     30
29   2-5/6-9        4,4            2527      112     30
30   2-5/10-12      0,0             462       63     30
31   2-5/10-12      0,1            2960      250     30
32   2-5/10-12      0,2            4830      404     30
33   2-5/10-12      0,3            2663      226     30
34   2-5/10-12      0,4           10283      621     30
35   2-5/10-12      1,1            1938      165     30
36   2-5/10-12      1,2            4610      443     30
37   2-5/10-12      1,3            3331      328     30
38   2-5/10-12      1,4            7819      649     30
39   2-5/10-12      2,2            2100      120     30
40   2-5/10-12      2,3            3845      449     30
41   2-5/10-12      2,4            5075      248     30
42   2-5/10-12      3,3            1426      148     30
43   2-5/10-12      3,4            6335      615     30
44   2-5/10-12      4,4             931       50     30
45   6-9/6-9        0,0            2485      274     30
46   6-9/6-9        0,1            4473      329     30
47   6-9/6-9        0,2            3976      292     30
48   6-9/6-9        0,3            5254      447     30
49   6-9/6-9        0,4            1349       64     30
50   6-9/6-9        1,1            1953       94     30
51   6-9/6-9        1,2            3528      198     30
52   6-9/6-9        1,3            4662      251     30
53   6-9/6-9        1,4            1197       51     30
54   6-9/6-9        2,2            1540       79     30
55   6-9/6-9        2,3            4144      239     30
56   6-9/6-9        2,4            1064       41     30
57   6-9/6-9        3,3            2701      154     30
58   6-9/6-9        3,4            1406       58     30
59   6-9/6-9        4,4             171       30     30     <- exactly at the floor, topped up
60   6-9/10-12      0,0            5467      433     30
61   6-9/10-12      0,1            8898      718     30
62   6-9/10-12      0,2            6797      396     30
63   6-9/10-12      0,3            8964      804     30
64   6-9/10-12      0,4            1960       90     30
65   6-9/10-12      1,1            3591      246     30
66   6-9/10-12      1,2            5397      321     30
67   6-9/10-12      1,3            7116      577     30
68   6-9/10-12      1,4            1524       78     30
69   6-9/10-12      2,2            1960       82     30
70   6-9/10-12      2,3            5166      333     30
71   6-9/10-12      2,4            1057       51     30
72   6-9/10-12      3,3            3404      336     30
73   6-9/10-12      3,4            1392       82     30
74   6-9/10-12      4,4             133       30     30     <- exactly at the floor, topped up
75   10-12/10-12    0,0            2926      169     30
76   10-12/10-12    0,1            4389      313     30
77   10-12/10-12    0,2            2695      129     30
78   10-12/10-12    0,3            3542      330     30
79   10-12/10-12    0,4             539       30     30
80   10-12/10-12    1,1            1596      166     30
81   10-12/10-12    1,2            1995      103     30
82   10-12/10-12    1,3            2622      348     30
83   10-12/10-12    1,4             399       30     30
84   10-12/10-12    2,2             595       30     30
85   10-12/10-12    2,3            1610       93     30
86   10-12/10-12    2,4             245       30     30
87   10-12/10-12    3,3            1035      170     30
88   10-12/10-12    3,4             322       30     30
89   10-12/10-12    4,4              21       21     21     <- smaller than f, taken in full

EMPTY strata (population, never topped up): []
```

Two strata fall below the floor and are taken in full — stratum 0 (15 pairs) and stratum 89
(21 pairs), the two extremes of the cross product. Eight more sit exactly at `f = 30` after
the top-up. Nothing is over-drawn: `sampled <= population` everywhere, asserted in
`test_every_non_empty_stratum_reaches_its_floor`.

**Still measure the real AIDS densities before fixing `K/q/f`.** This table is synthetic, and
the top-up budget scales with the number of non-empty strata. Here the corrected binning took
that count from 60 to 90 and the top-up from 18 to 136 pairs — a small absolute change against
a 22,106-pair total, but the same mechanism on a differently-shaped real density distribution
could move it further.

---

## Resume semantics and the kill/resume proof

**Checkpoint.** One file, overwritten in place — fscratch limits file *count*, not space.
Written to `<path>.tmp<pid>` in the same directory, `fsync`'d, then `os.replace`d, which is
atomic on POSIX. A `SIGKILL` mid-write leaves the *previous* checkpoint intact rather than a
truncated one. `test_the_checkpoint_is_a_single_file_overwritten_in_place` runs eight flushes
and asserts the directory afterwards contains exactly `["c.ckpt.npz"]` — one file, no `.tmp`.

**The checkpoint stores explicit pair indices, not a completion count.** Workers finish out
of order under a bounded submission window, so a count would be meaningless and a
prefix-based scheme would be wrong. Explicit indices make out-of-order completion safe.

**Identity checking, which is the part that matters.** The checkpoint carries a SHA-256 of
its chunk's *target pair set*. On resume:

- a fingerprint mismatch raises — resuming chunk 7's checkpoint into chunk 8 would produce a
  fully populated, symmetric, plausible shard containing another chunk's answers;
- any checkpointed index the chunk does not own raises, even if the fingerprint is absent.

Both are tested (`test_resume_skips_completed_pairs_and_reproduces_the_shard`,
`test_a_checkpoint_holding_foreign_pairs_is_rejected`).

**SIGTERM.** SLURM sends `SIGTERM` and waits 30 s. The parent traps `SIGTERM`/`SIGINT`, stops
submitting, cancels what has not started, collects what has already finished, flushes the
checkpoint and exits **143**. It writes **no shard** — an incomplete chunk must not produce a
file the merge would accept. Worker processes `SIG_IGN` the signal so the parent controls
shutdown rather than each child dying independently and losing in-flight pairs.

### The proof, as a tracked test

`tests/unit/test_ged_exact_runner.py::test_sigterm_flushes_the_checkpoint_and_the_resumed_shard_is_byte_identical`
— a real subprocess, a real `SIGTERM`, re-runnable by the orchestrator. 250 graphs, 31,125
pairs, `--workers 2 --checkpoint-every 200`. Observed:

```
resume identity: pair_index  dtype=int64    n=31125
resume identity: ged         dtype=float64  n=31125
resume identity: lb          dtype=float64  n=31125
resume identity: ub          dtype=float64  n=31125
resume identity: certified   dtype=bool     n=31125
resume identity: seconds     dtype=float32  n=31125
kill/resume over 31125 pairs: killed at 3180 checkpointed rows, exit 143,
no shard written; resumed run reused 3180 and computed 27945
```

3180 + 27945 = 31125, asserted in the test. The kill threshold is 3,000 and the observed
flush sat at 3,180 because several batches land between two polls of the `since_ckpt`
counter — the checkpoint holds *at least* as much as the trigger, never less. An earlier run
of this same test used a 200-row threshold, killed at 320 rows and was equally identical; the
threshold was raised so the interruption bites deeper.

**All six arrays byte-identical, including `seconds`.** That is only a meaningful claim
because `reference_stub_backend` reports a *deterministic* per-pair time derived from a CRC32
of the two graphs' invariants rather than a measured one, and the runner records
`PairResult.seconds` verbatim (falling back to its own measurement only when the backend
returns a non-finite or negative value). Against a real solver `seconds` is wall time and
would legitimately differ run to run; the other five arrays are the ones that must match.

The runner also asserts `len(shard) == pairs.size` before writing and exits 1 otherwise, so a
shard can never be short by construction.

### `--seed-from`

Loads an earlier result file, **skips computing** any pair already present, and **carries its
values into the output shard**. See Assumptions — this is a decision, not a reading.

---

## Decisions and why

**`--batch-size` defaults to 1.** Process-pool round-trip is tens of microseconds against
seconds of GEDLIB time, so batching buys nothing measurable, while a batch of size `b`
multiplies the worst-case tail by `b`: at `--timeout-per-pair 300`, a batch of 16 could block
a worker for 80 minutes inside a 2.5-hour task. The flag exists if measurement contradicts me.

**Bounded submission window, not `Executor.map`.** A stage-2 task owns ~10^5 pairs;
`ProcessPoolExecutor.map` submits every chunk up front. The driver keeps `max(4*workers, 16)`
futures outstanding and uses `wait(..., FIRST_COMPLETED, timeout=1.0)`, so memory is bounded
and the `SIGTERM` flag is noticed within a second even when nothing completes.

**One backend per worker process, built in the initializer.** GEDLIB env construction is not
free (CONTRACT B). Workers reload the CONTRACT A file themselves rather than having graphs
pickled to them, which is cheaper and keeps `_WorkerSpec` to plain scalars.

**A raising backend produces a flagged row, not a hole.** D11 forbids dropping pairs, and an
absent pair would leave a gap no downstream assertion looks for. The row is written with a
degenerate `[0, inf]` bracket and `failed=True`; the runner exits **1**; gate 4 rejects the
`inf` upper bound at merge. The failure is loud twice.

**`exact` is never promoted without certification.** `_outcome_from_result` sets `ged` only
when `certified` is true and `exact` is not `None` and lies inside its own bracket. This is
the defect the design note found in `ged_computer.py::compute_ged_pair`, where
`nx.graph_edit_distance(timeout=t)`'s best-so-far cost was recorded as exact; there is a test
named for it.

**Merge duplicates: coverage is "at least once", agreement is exact.** CONTRACTS §6 says both
"present exactly once" and "no conflicting values on any k present in more than one shard".
Those two cannot both be literal. I implemented: every `k` present in *at least* one shard,
and every `k` present in more than one must agree exactly on `ged/lb/ub/certified`. That is
the reading under which the stated verification mechanism — stage-1 reuse checked at merge —
actually functions. `inf == inf` compares true, so agreeing censored pairs do not trip it;
there is a test for that specifically.

**Gate 4 and off-diagonal zeros.** See Assumptions and the message to `main`.

---

## Assumptions

1. **Gate 4's zero clause.** CONTRACTS §7 ("`0 < v < inf` or censored") contradicts CONTRACT B
   §5 invariant 1 ("0 is legal when the graphs are isomorphic") on any corpus with isomorphic
   duplicates. Implemented: an off-diagonal 0 passes **only** when certified with
   `lb == ub == 0`, is counted as `n_zero_offdiag_certified`, and is rejected under the added
   `--strict-nonzero`. An *uncertified* zero always fails — that trap stays shut.
   **Confirmed by the orchestrator**, who added that `--strict-nonzero` must stay opt-in (it
   would fail on true duplicates) and that `n_zero_offdiag_certified` is a *reported* quantity,
   not a diagnostic: it is the `GED > 0` rung of the per-dataset pair-accounting ladder
   `raw → connected → GED-available → GED > 0 → Lev > 0 → analysed`. It is written to the
   merged metadata both top-level and inside `gate4`, and a test now asserts its presence.
2. **The merge needs a CONTRACT A input.** `node_counts`, `edge_counts`, `graph_ids` and
   `labels` exist in no shard. Added an optional `--input`, with convention-based fallback and
   an explicit error. No frozen flag changed. Reported.
3. **`--backend-factory module:callable`**, restricted to `--backend stub`. Without it the CLI
   cannot be exercised end to end on a machine with no `ged_backends`. `--backend stub` itself
   still resolves `ged_backends.StubBackend` and raises if absent — no silent fallback.
4. **`--seed-from` carries values through** rather than omitting them. If stage 2 omitted the
   overlap, the merge's cross-shard agreement check would have nothing to compare and the
   stated verification of stage-1 reuse would be vacuous. Flag it if you meant otherwise.
5. **Density quintile `side="right"`.** §8 does not specify the side. Initially implemented as
   numpy's default `side="left"`; corrected to `"right"` on the orchestrator's instruction
   after the dry run showed the top quintile collapsing. See the amendment above. Resolved,
   no longer an open assumption.
6. **Halo partners exclude self**, drawn without replacement from the other `N-1` graphs. §8
   says "uniformly from all 769"; a self-pair is not a pair.
7. **`--timeout-per-pair` is enforced by the backend**, not the runner. Killing a running C++
   call from Python is not safe; CONTRACT B gives the backend a `timeout_s` for this.
8. **The CONTRACT A reader is mine, temporarily.** `load_contract_a` reads the §4 format
   directly because `export_graphs` is on a peer's branch. Swapping it for `load_exported` is
   a one-line change; see Follow-ups.

---

## What I could not verify

- **Anything involving GEDLIB or a real solver.** No cluster access, `ged_backends` not on
  this branch. Every number here comes from a deterministic stub. The runner has never
  computed a real graph edit distance.
- **Real CONTRACT A files.** All fixtures are fabricated from the §4 key table. If
  `export_graphs` emits, say, `edges` as `int64` or `edge_offsets` without the leading zero,
  `load_contract_a` will reject it — loudly, but it will reject it. First integration step
  should be loading one real exported `.npz`.
- **Real AIDS strata.** The dry-run table is synthetic. The empty-stratum count and therefore
  the top-up budget will differ.
- **Cohort counts.** I never ran the §2 filter; `task-export` owns that assertion.
- **Cluster-scale behaviour.** Largest run here is 44,850 pairs on 2 workers. Not tested:
  64 workers, a 2.5-hour task, `$LOCALSCRATCH`, or SLURM's actual `SIGTERM`-then-`SIGKILL`
  timing. The 30-second window is ample for the flush measured here (milliseconds), but that
  is an inference.
- **`mypy --strict`** is enforced on `src/` only and I did not add `benchmarks/` to it.

---

## Tests

```
$ PY=~/.conda/envs/isalgraph-cpp/bin/python
$ $PY -m ruff check benchmarks/real_data/eval_setup/ged_{pair_index,sampling,exact_runner,merge_shards}.py \
      tests/unit/test_ged_{pair_index,sampling,exact_runner,merge_shards}.py
All checks passed!

$ $PY -m pytest tests/unit/test_ged_pair_index.py tests/unit/test_ged_sampling.py \
      tests/unit/test_ged_exact_runner.py tests/unit/test_ged_merge_shards.py -q
============================= 170 passed =======================================

$ $PY -m pytest tests/unit/ -q
============================= 556 passed in 13.96s =============================

$ $PY -m pytest tests/unit/ -q --ignore=<the four new files>      # the before state
============================= 386 passed in 2.51s ==============================
```

386 + 170 = 556. No pre-existing test changed behaviour or runtime. (Two of the 170 were
added by the `side="right"` amendment: the `q80 == max` guard and a binning-stability check.)

Slowest test is the kill/resume subprocess proof at ~12 s; the exhaustive 1,333,300-index
scalar sweep runs in 0.66 s.

### End-to-end smoke, actually run

40-graph CONTRACT A cohort (780 pairs), three chunks, `--workers 2`, deterministic stub,
then merge with gate 4.

```
$ $PY -m benchmarks.real_data.eval_setup.ged_exact_runner \
    --input $S/smoke40.npz --out $S/shards/smoke40_c0000.npz \
    --backend stub --backend-factory benchmarks.real_data.eval_setup.ged_exact_runner:reference_stub_backend \
    --cost-model unit --chunk-index 0 --n-chunks 3 --workers 2 \
    --timeout-per-pair 300 --checkpoint-every 100 --checkpoint $S/shards/smoke40_c0000.ckpt.npz

INFO __main__: smoke40: 40 graphs, chunk 0/3 -> positions [0, 260), 260 pairs
INFO __main__: chunk: 260 pairs owned, 0 already known, 260 to compute
INFO __main__: wrote .../smoke40_c0000.npz: 260 pairs (0 resumed, 0 seeded, 260 computed, 0 failed)

INFO __main__: smoke40: 40 graphs, chunk 1/3 -> positions [260, 520), 260 pairs
INFO __main__: wrote .../smoke40_c0001.npz: 260 pairs (0 resumed, 0 seeded, 260 computed, 0 failed)

INFO __main__: smoke40: 40 graphs, chunk 2/3 -> positions [520, 780), 260 pairs
INFO __main__: wrote .../smoke40_c0002.npz: 260 pairs (0 resumed, 0 seeded, 260 computed, 0 failed)
```

260 + 260 + 260 = 780 = C(40,2). Even split, no ragged tail.

```
$ $PY -m benchmarks.real_data.eval_setup.ged_merge_shards \
    --shards $S/shards --key smoke40 --n-graphs 40 --out $S/smoke40_merged.npz

INFO __main__: using .../smoke40.npz as the CONTRACT A cohort file
INFO __main__: merging 3 shards for smoke40 (40 graphs, 780 pairs)
INFO __main__: gate 4 passed: 531 certified, 249 censored, 0 certified zeros, max asymmetry 0
INFO __main__: wrote .../smoke40_merged.npz
```

531 + 249 = 780. The three `*.ckpt.npz` files sitting in the same directory were correctly
excluded — the merge found 3 shards, not 6.

The smoke fixtures come from tracked code (`tests.unit.test_ged_exact_runner.write_contract_a`
and `_make_graphs`), so the whole sequence is reproducible; only the output `.npz` files were
transient.

---

## Follow-ups for the orchestrator

1. **Patch CONTRACTS §6** to give `ged_merge_shards` a CONTRACT A input, so `task-slurm`
   generates the right command line next wave. My `--input` is optional with a fallback, but
   the contract should say so.
2. **Resolve the gate-4 zero ambiguity** in CONTRACTS §7 against CONTRACT B §5.1, and tell me
   whether `--strict-nonzero` should be the default.
3. **Swap `load_contract_a` for `export_graphs.load_exported`** once `task-export` merges —
   one call site in `ged_exact_runner.py`, one in `ged_merge_shards._load_cohort`, one in
   `ged_sampling._load_counts`. Three readers of one format is two too many; I kept them
   separate only because the peer module does not exist on this branch.
4. **Measure the density quintiles on real AIDS before recomputing `K/q/f`.** The non-empty
   stratum count drives the top-up budget. Also re-check the mirror of the amended binning:
   `side="right"` empties the bottom quintile if `q20` equals the minimum density. It does not
   here (quintile populations `[154, 154, 151, 151, 159]`), but it is one assertion to add
   once the real densities exist.
5. **Benchmark `--batch-size 1` against the real solver** on gate 3's timings. If GEDLIB
   round-trip turns out to be a non-trivial share of a fast pair, raise it — but weigh the
   tail-latency cost stated above.
6. **`ged_bounds` is imported by dotted package path** (`{__package__}.ged_bounds`). On
   Picasso, `PYTHONPATH` must be the repo root, never `${REPO_DIR}/src` — already CONTRACTS §9,
   restated because the cost model silently depends on it.
7. **Stale constants, untouched as instructed**: `eval_setup.py:75 DEFAULT_SOURCE_DIR` and
   `eval_message_length.py:36 DEFAULT_DATA_ROOT` still point at the moved tree.
