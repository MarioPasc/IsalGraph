# CONTRACTS — wave `2026-08-26-t13-complexity`

Frozen by the orchestrator before any agent started. **Code against this file, never against a
peer's branch.** If a contract is wrong, message the orchestrator; do not renegotiate with a peer.

Base commit: `10eae30492982492cfc45db845b71c91a08e7883`.
Design note: `.claude/notes/review/tasks/T-13-design.md`.

---

## 0. Global rules

- **Nothing under `src/isalgraph/` is modified.** Not one line. T-13 is measurement only, and the
  frozen Python reference is what the C++ differential compares against.
- Everything new lives under `benchmarks/real_data/eval_t13_complexity/`.
- Python 3.10+ syntax, full type hints, NumPy-style docstrings, `ruff` and `mypy --strict` clean.
- `logging`, never `print()`, outside a `__main__` CLI.
- No `ssh`, no `sbatch`, no `rsync`, no network. The orchestrator owns every cluster action.
- Nothing written to the repo's `scratchpad/`.
- Commit incrementally on your own branch. Uncommitted work cannot be merged.

## 1. Module ownership

| File | Owner | May be imported by |
|---|---|---|
| `families.py` | **A** | C |
| `symmetry.py` | **A** | C |
| `instrumented.py` | **B** | — |
| `counters.py` (CLI) | **B** | — |
| `schema.py` | **C** | — |
| `measure.py` (CLI) | **C** | — |
| `slurm/{launch,worker}.sh` | **C** | — |
| `__init__.py` | **A** (creates it, exports nothing) | all |
| `tests/test_families.py`, `tests/test_symmetry.py` | **A** | |
| `tests/test_instrumented.py` | **B** | |
| `tests/test_schema.py`, `tests/test_measure.py` | **C** | |
| `benchmarks/eval_t13_complexity` symlink → `real_data/eval_t13_complexity` | **C** | |

No other file in the repository is written by any agent.

---

## 2. `symmetry.py` — owner A

```python
def log10_aut(g: nx.Graph) -> float
```
`log10|Aut(G)|` from `pynauty.autgrp`, computed as `log10(mantissa) + exponent`.
**Never** `mantissa * 10**exponent` — that overflows above ~1e308. Returns `0.0` for a rigid graph.

```python
def orbits(g: nx.Graph) -> dict[Hashable, int]
```
Node → orbit id, dense ids from 0.

```python
def wl_partition(g: nx.Graph, *, rounds: int | None = None) -> dict[Hashable, int]
```
Exact 1-WL colour refinement run **to stability** when `rounds is None`. Self-contained: no
`grakel`, no `numpy` (grakel is unusable on Picasso under numpy 2). Node → colour id.

```python
def triplet_partition(g: nx.Graph) -> dict[Hashable, tuple[int, int, int]]
```
The incumbent pruning key: `(|N_1(v)|, |N_2(v)|, |N_3(v)|)`, BFS shell sizes truncated at radius 3.
**Must agree with `isalgraph.compute_structural_triplets` on the same graph** — a test asserts this
on ≥ 500 graphs.

```python
def refines(fine: Mapping[H, object], coarse: Mapping[H, object]) -> bool
```
`True` iff every `fine` class lies inside a single `coarse` class. **Exact containment. A class-count
comparison is not a refinement test and may not be used as one** — that error is what T-13 exists to
correct.

```python
def resolution_record(g: nx.Graph) -> dict[str, object]
```
Returns exactly these keys, no more, no fewer:
`log10_aut, n_orbits, max_orbit_size, n_wl_classes, n_triplet_classes, wl_refines_triplet,
triplet_refines_wl, wl_equals_orbits, triplet_equals_orbits`.

**Invariant that must hold on every graph** (Proposition 1 of the design note): neither the WL nor
the triplet partition may be *finer* than the orbit partition, i.e.
`refines(orbits(g), wl_partition(g))` and `refines(orbits(g), triplet_partition(g))` are both `True`.
A property test asserts this; a failure is a stop-and-ask, not a fix.

---

## 3. `families.py` — owner A

```python
FAMILIES: tuple[str, ...] = (
    "path", "cycle", "star", "complete", "complete_bipartite", "hypercube",
    "prism", "caterpillar", "rigid_er", "symmetry_ladder",
)

@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    n: int                       # realised order, not the requested one
    replicate: int               # 0 for deterministic families
    params: tuple[tuple[str, int], ...]   # hashable; e.g. (("swaps", 3),)
    log10_aut_expected: float | None      # None where no closed form exists

def build(spec: FamilySpec, *, seed: int) -> nx.Graph
def enumerate_grid(*, sizes: Sequence[int], replicates: int, seed: int) -> tuple[FamilySpec, ...]
```

Closed forms that **must** be asserted against `symmetry.log10_aut` inside `build`, raising
`FamilyVerificationError` on mismatch beyond `1e-6`:

| family | `n` | `m` | `\|Aut\|` |
|---|---|---|---|
| `path` | n | n−1 | 2 (n ≥ 2) |
| `cycle` | n | n | 2n (n ≥ 3) |
| `star` | n | n−1 | (n−1)! |
| `complete` | n | n(n−1)/2 | n! |
| `complete_bipartite` `K_{a,a}` | 2a | a² | 2·(a!)² |
| `hypercube` `Q_d` | 2^d | d·2^{d−1} | 2^d · d! |
| `prism` `C_a × K_2` | 2a | 3a | 4a (a ≥ 4; a = 3 is `K_{3,3}`-adjacent, handle or exclude) |
| `caterpillar` | n | n−1 | 2^k, k = number of degree-1 pairs sharing a spine node |
| `rigid_er` | n | random | **`None`** — verified as `log10_aut == 0.0` instead, resampling up to 50 times, then raising |
| `symmetry_ladder` | fixed | fixed | **`None`** — see below |

**`symmetry_ladder` is the primary design.** For a given `(n, m)` it emits a sequence of graphs
holding `n` and `m` **exactly** constant while `|Aut|` decreases monotonically: start from a
maximally symmetric base on those parameters (cycle, prism, or `K_{a,a}` as `m` allows) and apply
`k = 0, 1, 2, …` **degree-preserving double edge swaps** (`nx.double_edge_swap`) with a fixed seed,
keeping the graph connected. `params` carries `(("swaps", k), ("base", …))`. It must assert
`n` and `m` unchanged at every rung and record the realised `log10_aut`, which is measured, not
predicted. **A rung that does not lower `log10_aut` is kept and recorded, not discarded** — the
ladder is the experiment, and dropping non-monotone rungs would make the result outcome-dependent.

`enumerate_grid` uses `sizes = (8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64)` and
`replicates = 5` for `rigid_er` and `symmetry_ladder`, `1` for the deterministic families,
snapping each family to its nearest realisable order (`hypercube` to powers of two, and so on) and
de-duplicating specs.

---

## 4. `instrumented.py` / `counters.py` — owner B

An instrumented **mirror** of the frozen reference. `src/isalgraph/core/` is not touched; this is
the same device `viz/encoder_trace.py` used for T-09.

```python
@dataclass(frozen=True, slots=True)
class OperationCounts:
    frames: int              # payload instructions emitted; == m for a greedy encode
    pair_trials: int         # (a, b) pairs examined, summed over frames
    scan_depth_total: int    # same as pair_trials; kept separate for the per-frame max
    scan_depth_max: int
    pointer_steps: int       # unit CDLL moves executed while trialling and committing
    neighbour_checks: int    # adjacency / uninserted-neighbour tests
    backtrack_nodes: int     # recursion frames entered (canonical arms only; 0 for greedy)
    search_leaves: int       # complete strings produced by the search (canonical arms only)
    string_length: int

def greedy_counts(g: SparseGraph, start: int) -> tuple[str, OperationCounts]
def canonical_counts(g: SparseGraph) -> tuple[str, OperationCounts]
def pruned_counts(g: SparseGraph) -> tuple[str, OperationCounts]
```

**Parity is the deliverable, not the counts.** Each function's returned string must be
byte-identical to, respectively,
`isalgraph.core.graph_to_string.GraphToString().encode(g, start)` (or whatever the frozen entry
point is — read it, do not guess),
`isalgraph.core.canonical.canonical_string(g)` and
`isalgraph.core.canonical_pruned.pruned_canonical_string(g)`,
**imported from `isalgraph.core.*` so the pure-Python reference is compared, never the engine.**

Acceptance: ≥ 50,000 (graph, start) parity pairs, **0 mismatches**, over connected graphs with
`2 ≤ n ≤ 12` drawn deterministically. Report the exact count in the work log.

Counters must satisfy, and a test must assert, on every graph:
`frames == m` for `greedy_counts`; `pair_trials >= frames`; `pointer_steps >= 0`;
`backtrack_nodes >= search_leaves >= 1` for the canonical arms.

`counters.py` CLI: `python -m benchmarks.eval_t13_complexity.counters --spec-file <jsonl> --out <jsonl>`
emitting one row per (graph, encoder) with `schema_version = "t13c.1"` and fields:
`schema_version, source, family, n_target, replicate, dataset, graph_index, n, m, encoder,
frames, pair_trials, scan_depth_max, pointer_steps, neighbour_checks, backtrack_nodes,
search_leaves, string_length, parity_ok`.
`encoder ∈ {"greedy", "canonical", "pruned"}`. `parity_ok` must be `true` in every emitted row.

---

## 5. `schema.py` / `measure.py` — owner C

### 5.1 The record — `schema_version = "t13.1"`

One **JSON Lines** row per `(graph, representation, arm)`. Field set, frozen:

```
schema_version, run_id, host, engine, build_hash, isalgraph_version, timestamp_utc
source            "constructed" | "cohort"
family            str | null      n_target int | null     replicate int | null
dataset           str | null      graph_index int | null  graph_id str | null
n, m, density, max_degree, connected
log10_aut, n_orbits, max_orbit_size, n_wl_classes, n_triplet_classes,
  wl_refines_triplet, triplet_refines_wl, wl_equals_orbits, triplet_equals_orbits
representation    str             arm  "default" | "no_pairs_memo" | "no_bnb" | "no_pairs_memo_no_bnb"
status            "ok" | "censored" | "error" | "unsupported"
error_kind        str | null
seconds           float           repeats int             budget_s float
length_chars      int | null
```

The nine symmetry fields come from `symmetry.resolution_record` verbatim.

### 5.2 Representations, frozen

Search-based: `isalgraph_exhaustive`, `isalgraph_pruned`, `isalgraph_greedy`, `nauty_graph6`,
`sparse6_nauty`, `min_dfs`, `agm_cam`.
Search-free controls: `adjacency`, `graph6`, `sparse6`, `wl_subtree`, `size_null`.

Resolve them through the existing registry in `isalgraph.competitors` — do **not** re-implement any
backend. A backend that declines a graph (e.g. `agm_cam`'s Suite-1 scope guard) is recorded
`status = "unsupported"`, never dropped.

### 5.3 Timing rule, frozen — implement here, nowhere else

1. `time.process_time`, single thread, `ISALGRAPH_THREADS=1`.
2. One warm-up run. If it took `>= 1.0 s`: `repeats = 1`, `seconds = warmup`.
   Otherwise `repeats = 3` and `seconds = median` of three further runs.
3. **Budget `budget_s = 300.0`, enforced by a killed subprocess.** `SIGALRM` does **not** interrupt
   the C++ engine (T-05 finding 5) and may not be used. On expiry:
   `status = "censored"`, `seconds = budget_s`, `length_chars = null`.
4. Every shard asserts `isalgraph.engine() == "cpp"` at start and writes `build_info()` into its
   header line. **A run whose `build_hash != "298fc1188bf1b051"` must abort**, not warn.
5. The engine ablation arms toggle `isalgraph.core._native.set_pairs_memo` /
   `set_branch_and_bound` and **must restore both to `True` afterwards**, in a `finally`.
   Ablation arms run only on a stratified subsample, selected by a rule fixed in code before any
   result exists.

### 5.4 CLI

```
python -m benchmarks.eval_t13_complexity.measure \
    --source constructed|cohort  --shard K --n-shards N \
    [--dataset D]  --arms default[,no_pairs_memo,...] \
    --budget-s 300 --seed 13 --out <path>.jsonl
```
Sharding is by a deterministic hash of the work-unit key, so shard membership never depends on
ordering. Output is append-safe and one file per shard:
`records_<source>_<shard>of<n_shards>.jsonl`.

### 5.5 SLURM

Invoke the `picasso-sbatch` skill and write `slurm/launch.sh` + `slurm/worker.sh` following it.
`bash -n` both, and paste a `--dry-run`/`--test-only`-shaped preview into the work log.
**Do not submit, ssh, or rsync.** The orchestrator does that. Constraints that matter:
`cpu_partition`, whole-node `--exclusive`, one shard per core via `taskset`, **≥ 2 h per task**
(SCBI's floor, asked of this account in writing), `account=tic_163_uma`, logs under
`~/execs/isalgraph/logs`. Picasso repo: `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph`;
env: `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph`;
cohort root: `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph`.
**Never set `PYTHONPATH=<repo>/src`** — it shadows the editable install and silently drops to pure
Python, which would make every timing fiction.

---

## 6. Environment, verbatim

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pytest benchmarks/real_data/eval_t13_complexity/tests/ -q
$PY -m ruff check --fix benchmarks/real_data/eval_t13_complexity/
$PY -m mypy benchmarks/real_data/eval_t13_complexity/
export ISALGRAPH_COHORT_ROOT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data
```

`isalgraph.engine()` must read `cpp` and `build_info()["build_hash"]` must be `298fc1188bf1b051`.
