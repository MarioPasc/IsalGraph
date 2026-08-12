# Wave 2026-08-12-exact-ged — frozen contracts

**Authoritative for all agents in this wave.** These interfaces are frozen by the orchestrator. An
agent that finds a contract wrong must `SendMessage` the orchestrator (`main`) and continue on a
recorded assumption — it must **never** change a contract unilaterally, because another agent is
coding against it right now.

Design rationale: `.claude/notes/review/tasks/T-03-design.md`. Read it first — it is short and it
explains *why* each of these exists.

---

## 0. Prohibitions — all agents

1. **No SSH, no `ssh picasso`, no cluster access of any kind.** The orchestrator owns the cluster.
   A wave agent that submits a job has broken the run.
2. **No `import gklearn` at module import time.** GEDLIB exists **only on Picasso**; it is not
   installed on this machine and never will be. Every GEDLIB import is lazy, inside a function body,
   via `importlib.import_module`.
3. **No `import isalgraph`.** Exact GED does not touch the encoder. The Picasso conda env does not
   even have `isalgraph` installed, and adding a dependency on it would drag the C++ engine build
   into this ticket for no reason.
4. **No `import torch` outside the export path.** `torch` is needed only to read GraphEdX's `.pt`
   files, which happens once, locally. Picasso has no torch and must never need it.
5. **Nothing in `scratchpad/`.** Everything is tracked in the repository. Untracked scratch files are
   the documented failure mode that lost thirteen measurement scripts from this project.
6. **Never change the cohort, the cost model, or the stage-1 sampling design.** They are locked by
   `decisions.md` and by the design note. If your code cannot reproduce a locked count, that is a
   finding to report, not a parameter to adjust.

---

## 1. Cost model — the only two permitted

Already implemented in `benchmarks/real_data/eval_setup/ged_bounds.py`; **reuse those objects, do not
redefine the numbers**, so the two implementations cannot drift.

```python
from ged_bounds import EditCosts, UNIT_COSTS, GRAPHEDX_COSTS
UNIT_COSTS      # EditCosts()                                  -> [1, 1, 0, 1, 1, 0]  PRODUCTION (D6)
GRAPHEDX_COSTS  # EditCosts(node_ins=0.0, node_del=0.0)         -> [0, 0, 0, 1, 1, 0]  GATE 0 ONLY
EditCosts.as_gedlib_constant()   # -> [n_ins, n_del, n_rel, e_ins, e_del, e_rel]
```

`GRAPHEDX_COSTS` is used by **gate 0 and nothing else**. Running gate 0 under `UNIT_COSTS` produces a
guaranteed mismatch that reads exactly like a solver bug.

---

## 2. Cohort — reproduce exactly or fail loudly

Filter: `dataset_filter.filter_graphs(min_nodes=2, require_connected=True, n_max=12)`, splits merged.

| key | dataset | graphs | pairs |
|---|---|---:|---:|
| `iam_letter_low` | IAM Letter LOW | 1,180 | 695,610 |
| `iam_letter_med` | IAM Letter MED | 1,253 | 784,378 |
| `iam_letter_high` | IAM Letter HIGH | 2,059 | 2,118,711 |
| `linux` | LINUX | 89 | 3,916 |
| `aids` | AIDS (GraphEdX) | 769 | 295,296 |
| | **Total** | **5,350** | **3,897,911** |

Every pair count is `C(kept, 2)`. **Assert these. Exit non-zero on mismatch. Do not adjust the
filter.**

---

## 3. Data locations (local machine)

```
SOURCE = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source
  ${SOURCE}/GED_PRECOMPUTED/{AIDS,LINUX}/{train,val,test}_{graphs,result}.pt   # torch pickles
  ${SOURCE}/IAM_Database/extracted/Letter/{LOW,MED,HIGH}/{train,validation,test}.cxl + *.gxl
EXPORT_DIR = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/exported
```

⚠ `eval_setup.py:75 DEFAULT_SOURCE_DIR` and `eval_message_length.py:36 DEFAULT_DATA_ROOT` point at
`…/research/isalgraph/data/…`, **which no longer exists** — the tree was moved under
`ISAL/completed/`. `validate_ged_bounds.py::DEFAULT_SOURCE` already has the correct path. Do not fix
the stale constants in this wave; they are outside every agent's ownership. Report them and move on.

---

## 4. CONTRACT A — the exported dataset file

**Owner: `task-export`. Consumers: `task-gedlib-gates`, `task-runner`.**

One `.npz` per dataset, written by `export_graphs.py`, plus one `manifest.json` for all five.
Six files total. The IAM Letter GXL tree is 6,767 files and is **never transferred to the cluster**;
that is the entire point of this step.

`np.savez_compressed(f"{EXPORT_DIR}/{key}.npz", **{...})` with **exactly** these keys:

| key | dtype | shape | meaning |
|---|---|---|---|
| `graph_ids` | `<U…` | `(N,)` | stable id, in kept order |
| `n_nodes` | `int32` | `(N,)` | node count per graph |
| `n_edges` | `int32` | `(N,)` | edge count per graph |
| `edge_offsets` | `int64` | `(N+1,)` | CSR offsets into `edges`; `edge_offsets[0] == 0`, `edge_offsets[-1] == n_edges.sum()` |
| `edges` | `int32` | `(2, M_total)` | concatenated undirected edge lists, endpoints **local to each graph**, `0 <= u < v < n_nodes[g]` |
| `splits` | `<U…` | `(N,)` | `train`/`val`/`test` for GraphEdX, `train`/`validation`/`test` for Letter |
| `labels` | `<U…` | `(N,)` | class label where the source has one, `""` otherwise |
| `metadata` | `<U…` | `()` (0-d) | JSON string, schema below |

`metadata` JSON, exactly these fields:

```json
{"dataset": "aids", "source": "graphedx", "n_raw": 911, "n_kept": 769,
 "n_dropped_size": 0, "n_dropped_disconnected": 0, "n_dropped_trivial": 0,
 "n_pairs": 295296, "filter": {"min_nodes": 2, "require_connected": true, "n_max": 12},
 "splits_merged": true, "exported_utc": "2026-08-12T...Z", "code_commit": "<sha>",
 "schema_version": 1}
```

**Loader, frozen signature**, in the same module:

```python
@dataclass(frozen=True, slots=True)
class ExportedDataset:
    key: str
    graphs: list[nx.Graph]        # relabelled 0..n-1, undirected, no self-loops, no attributes
    graph_ids: list[str]
    splits: list[str]
    labels: list[str]
    n_nodes: np.ndarray           # int32 (N,)
    n_edges: np.ndarray           # int32 (N,)
    metadata: dict[str, object]

def load_exported(path: str | os.PathLike[str]) -> ExportedDataset: ...
def save_exported(dataset: ExportedDataset, path: str | os.PathLike[str]) -> None: ...
```

`load_exported` **must not import torch**. Round-trip identity (`save` then `load` reproduces the
graphs edge-for-edge) is a required test.

**Consumers**: until `task-export` is merged, `task-gedlib-gates` and `task-runner` fabricate a
conforming `.npz` in their own test fixtures from the key table above. Do **not** import
`export_graphs` — you do not own it and it does not exist on your branch.

`manifest.json`: `{"<key>": {"sha256": "...", "bytes": N, "n_kept": N, "n_pairs": N,
"content_sha256": "..."}, ...}`.

> **Patch, 2026-08-12** — `content_sha256` added after `task-export` measured that
> `np.savez_compressed` stamps every zip member with local time, so two byte-identical exports of the
> same data produce **different** file `sha256` values (`aids.npz` even changed size, 13,307 → 13,305
> bytes, because the timestamp bytes compress differently). The contracted `sha256` therefore
> certifies **transfer integrity only** and can never certify build reproducibility;
> `content_sha256` digests the array contents alone and was identical across runs for all five
> datasets. `--verify-only` checks it only when the manifest carries it.

---

## 5. CONTRACT B — the GED backend

**Owner: `task-gedlib-gates`. Consumer: `task-runner` (via `Protocol`, never by import).**

```python
# benchmarks/real_data/eval_setup/ged_backends.py

@dataclass(frozen=True, slots=True)
class PairResult:
    lb: float             # certified lower bound
    ub: float             # certified upper bound, symmetrised over both orientations
    exact: float | None   # set IFF certified; None otherwise. NEVER a best-so-far value
    certified: bool       # lb == ub within 1e-9  ->  exact is optimal
    seconds: float        # wall time for this pair, this backend
    timed_out: bool
    method: str           # e.g. "ANCHOR_AWARE_GED+BRANCH_FAST+IPFP" or "networkx_astar"

class GedBackend(Protocol):
    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult: ...
    @property
    def name(self) -> str: ...
```

> ### 🔴 Patch, 2026-08-12 — `ANCHOR_AWARE_GED` is RETIRED. Read this before §5's invariant 4.
>
> Measured on Picasso: **non-deterministic on 14/15 real AIDS pairs** (same pair, six fresh
> environments, e.g. `[10, 6, 6, 6, 6, 4]` where brute force says **2**), **wrong on 4/18** small
> pairs against exhaustive enumeration — always over, never under — and it reports `LB == UB` on
> those wrong values. No option (`--threads 1`, `--map-root-to-root`, `--search-method DFS`) restores
> it. `networkx` A* was correct 18/18 on the same oracle.
>
> **So `LB == UB` from that method is a false certificate, and invariant 4 below cannot rest on it.**
> New assignment, PI-authorised (design note, amendment 2): **exact = `networkx` A* run to
> completion**; **GEDLIB is bounds-only**, `BRANCH_FAST` for LB and `IPFP` for UB. Certification is
> now decided by *whether A\* completed*, not by any solver's self-report — a pair whose A* hit its
> timeout is interval-censored `[lb, ub]` under D11, never promoted to exact. `ANCHOR_AWARE_GED` and
> `HED` are hard-guarded and no core-hour is spent on either.

Concrete backends, all in `ged_backends.py`:

- `GedlibBackend(costs=UNIT_COSTS, *, timeout_s=300.0, exact_method="ANCHOR_AWARE_GED", lb_method="BRANCH_FAST", ub_method="IPFP")`
  — **superseded**: `exact_method` is retired, the backend returns `exact=None, certified=False`
- `NetworkxBackend(costs=UNIT_COSTS, *, timeout_s=300.0)` — uses `ged_bounds` for `lb`/`ub` and
  `nx.graph_edit_distance` for `exact`
- `StubBackend(...)` — deterministic, no solver; exists so `task-runner` can test without GEDLIB

### Five invariants the backend must enforce, each with a test

1. **Assert `0 < value < inf` on every GEDLIB read**, except that a value of exactly `0` is legal
   only when the two graphs are isomorphic. An upper-bound method returns `get_lower_bound() = 0.00`
   and `HED` returns `inf`; **neither raises**, and a whole matrix fills silently with zeros. Raise
   `GedBackendError` instead.
2. **`HED` is never used.** It returns `LB = 0 / UB = inf` under default options and is undiagnosed.
3. **Upper bounds are direction-dependent.** `BIPARTITE`, `IPFP`, `REFINE`, `BP_BEAM` build an edit
   path from a *directed* assignment. Compute both orientations and take the `min`. Do not assume
   symmetry — measure it, and record the asymmetry rate.
4. **`exact` is `None` unless certified.** `ANCHOR_AWARE_GED` gives both bounds and `lb == ub` is the
   optimality certificate. Never promote an uncertified value: that is precisely the defect in
   `ged_computer.py::compute_ged_pair`, which returns `nx.graph_edit_distance`'s best-so-far cost as
   if it were exact.
5. **Import order.** `importlib.import_module("gklearn.gedlib.libraries_import")` **must** run before
   `gklearn.gedlib.gedlibpy_gxl`, or `libdoublefann.so.2` fails to load. Use `importlib` — ruff and
   isort reorder plain `from … import` lines alphabetically and silently break this.

`GEDEnvGXL.add_nx_graph` requires **string-valued** node and edge attributes; attach a constant dummy
label. Build the env **once per process**, not per pair.

Names that changed in the GEDLIB refactor (most online tutorials are stale):
`librariesImport` → `libraries_import`; `gedlibpy` → **`gedlibpy_gxl`**; `GEDEnv` → **`GEDEnvGXL`**.

---

## 6. CONTRACT C — the runner CLI and the shard format

**Owner: `task-runner`. Consumer: `task-slurm` (next wave) and the orchestrator.**

```
python -m benchmarks.real_data.eval_setup.ged_exact_runner \
  --input <path>/<key>.npz            # CONTRACT A file
  --out   <path>/shards/<key>_c0007.npz
  --backend {gedlib,networkx,stub}    # required
  --cost-model {unit,graphedx}        # default unit
  --chunk-index 7 --n-chunks 24       # contiguous upper-triangle index range
  [--pair-list <path>.npz]            # stage 1: explicit linear indices, key `pair_index` int64
  [--seed-from <path>.npz]            # stage 2: reuse already-computed pairs
  --workers 64 --timeout-per-pair 300 --checkpoint-every 2000
  --checkpoint <path>/<key>_c0007.ckpt.npz
```

`--chunk-index`/`--n-chunks` split `C(N,2)` **evenly**, remainder spread over the low-numbered chunks
— never a fixed block with a ragged tail; the short remainder task is exactly what SCBI's two-hour
floor forbids. With `--pair-list`, the *pair list* is split instead, by the same rule.

**Shard `.npz`** — sparse, indexed by linear upper-triangle index:

| key | dtype | meaning |
|---|---|---|
| `pair_index` | `int64` | linear upper-triangle index `k`, ascending |
| `ged` | `float64` | certified exact value, or `inf` when censored |
| `lb` | `float64` | always finite and > 0 unless isomorphic |
| `ub` | `float64` | always finite |
| `certified` | `bool_` | `True` iff `ged` is a certified optimum |
| `seconds` | `float32` | per-pair wall time — **required** for the D12 censoring analysis |
| `meta` | `<U…` 0-d | JSON: dataset, backend, cost model, chunk index/count, timeout, hostname, cpu model, start/end UTC |

**Linear upper-triangle index**, frozen:
```
k(i, j) = i*N - i*(i+1)//2 + (j - i - 1),      0 <= i < j < N,  0 <= k < C(N,2)
```
The inverse must be **integer-exact**. A `float` `sqrt` is acceptable only with an explicit
correction step that re-derives `i` and adjusts by ±1 until `k(i,j) == k`. Required property test:
round-trip over **every** `k` for `N` in `{2,…,200}`, and over 10⁵ random `k` for
`N ∈ {769, 1180, 1253, 2059}`.

**Merge CLI**, same owner:
```
python -m benchmarks.real_data.eval_setup.ged_merge_shards \
  --shards <dir> --key aids --n-graphs 769 --out <dir>/aids.npz \
  [--input <key>.npz] [--delete-shards] [--strict-nonzero]
```

> **Patch, 2026-08-12 — `--input` was missing and the contract was unbuildable as written.**
> Contract D requires `node_counts`, `edge_counts`, `graph_ids` and `labels`, and **no shard carries
> them** — they exist only in the Contract A file. `task-slurm` must pass `--input`; the fallback
> resolution order is `<shards>/<key>.npz` then `<shards>/../<key>.npz`, and a missing input is a
> named error rather than a silent omission.
>
> **Patch — off-diagonal zeros.** §7's `0 < v < inf` contradicted §5 invariant 1's allowance for
> isomorphic pairs, and both IAM Letter and AIDS contain isomorphic duplicates. Resolution: an
> off-diagonal zero passes **only** when certified with `lb == ub == 0`; an **uncertified** zero
> always fails, which keeps the "matrix silently fills with zeros" trap closed. The count is
> exported as `n_zero_offdiag_certified` — it is a *reported* quantity, the `GED > 0` rung of the
> pair-accounting ladder the statistical protocol requires. `--strict-nonzero` makes any
> off-diagonal zero fail and is **opt-in**, never the default.
Merge asserts: every `k ∈ [0, C(N,2))` present exactly once; no conflicting values on any `k` present
in more than one shard (this is how stage-1 reuse is verified); then **gate 4** below.

---

## 7. CONTRACT D — the final per-dataset `.npz`

**Owner: `task-runner` (written by the merge step).**

Must be consumable **unchanged** by `eval_correlation.py`, `method_comparator.py`,
`dataset_filter.py` and `validator.py`. Those read `ged_matrix`, `node_counts`, `edge_counts`,
`graph_ids`, `labels`, `metadata` — so those six keys keep **exactly** their existing names, dtypes
and semantics (see `ged_computer.py::save_ged_matrix`). Additions are ignored by downstream code:

| key | dtype | meaning |
|---|---|---|
| `ged_matrix` | `float64 (N,N)` | symmetric, diag 0, **`inf` where censored** — the existing convention |
| `node_counts`, `edge_counts` | `int32 (N,)` | as before |
| `graph_ids`, `labels` | `<U…` `(N,)` | as before |
| `metadata` | `<U…` 0-d | JSON, superset of the existing schema |
| `lb_matrix`, `ub_matrix` | `float64 (N,N)` | **new** — the D11 censoring interval, always finite |
| `certified_mask` | `bool_ (N,N)` | **new** — `True` where `ged_matrix` is a certified optimum |
| `seconds_matrix` | `float32 (N,N)` | **new** — per-pair wall time, for D12 |

**Gate 4, asserted at merge on every matrix**, exit non-zero on any failure:
`ged_matrix` symmetric to machine precision; diagonal exactly 0; every off-diagonal entry either
`0 < v < inf` **or** `inf` with `certified_mask == False` and a finite `lb <= ub`; `lb_matrix` and
`ub_matrix` symmetric and finite; `lb <= ged <= ub` wherever `ged` is finite.

**Censored pairs are interval-censored, never dropped** (D11). The censoring rate is reported **per
stratum, never pooled** (D12).

---

## 8. Stage-1 sampling design — frozen, `task-runner` implements verbatim

Population: the 769 AIDS graphs after the Suite-1 filter. **Seed 42.** Full rationale in the design
note §4; the short form is that the core must be a *simple random sample* so the D2 graph-level
cluster bootstrap is exact on a complete induced submatrix, while the halo and top-up carry the
"spans all 769 graphs and every stratum" requirement that the core alone cannot.

```
core     : simple random sample of K = 180 graphs (seed 42); ALL C(180,2) = 16,110 pairs
halo     : for each of the 589 non-core graphs, q = 10 partners drawn uniformly from all 769
top-up   : every NON-EMPTY pair-stratum holding < f = 30 sampled pairs is filled to
           min(f, |stratum|) by uniform draw without replacement from that stratum
```

**Pair strata**: size cell = unordered pair of node-count bins over `{2–5, 6–9, 10–12}` (6 cells);
density cell = unordered pair of AIDS-internal density quintiles (15 cells); stratum = the cross
product. Quintile edges from `np.quantile(density, [.2,.4,.6,.8])` with
**`np.searchsorted(..., side="right")`**, so ties fall consistently.

> **Patch, 2026-08-12** — `side` was under-specified. With the default `"left"` a density equal to
> the top edge falls in the *lower* bin, so the top quintile is unreachable whenever `q80` equals the
> maximum — which fires on the real cohort, since AIDS after `min_nodes = 2` contains n = 2 graphs at
> density exactly 1.0. Corrected before any pair was computed. Measured on the 769-graph dry run:
> non-empty strata **60/90 → 90/90**, quintile populations `[154, 154, 151, 151, 159]` against a
> perfect fifth of 153.8. "Non-empty" is judged on the **population**, not on the sample. Empty strata are
reported as empty, never topped up.

`K`, `q`, `f` are recalculated **once** by the orchestrator from the measured per-pair rate, holding
the ~100 core-hour budget fixed and holding the ratios `K : q : f`. Expose them as CLI flags with
these values as defaults. Emit a `pair_list.npz` (`pair_index` int64) plus a
`sampling_report.json` recording: `K`, `q`, `f`, seed, per-stratum population and sampled counts,
the number of distinct graphs covered (**must be 769**), and the total pair count.

---

## 9. Environment and commands — verbatim

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python      # python 3.11.15, nx 3.6.1, scipy 1.17.1, numpy 1.26.4
$PY -m pytest tests/unit/ -q                   # your unit tests
$PY -m pytest tests/unit/test_ged_bounds.py -q # 35 tests, must stay green
$PY -m ruff check --fix <your files>
$PY -m mypy <your files>                       # strict is enforced on src/ only; aim for clean anyway
```

`benchmarks/` is **not** a package on `sys.path` by default. Existing modules in `eval_setup/` use
`sys.path.insert` + bare imports (see `validate_ged_bounds.py`). New modules must be importable
**both** as `python -m benchmarks.real_data.eval_setup.<mod>` from the repo root **and** as a bare
module from within `eval_setup/`. Follow the pattern already in `validate_ged_bounds.py`.

Picasso runs with `PYTHONPATH="${REPO_DIR}:${GEDLIB_CHECKOUT}"` — **repo root, never `${REPO_DIR}/src`**.

## 10. Reference material worth reading

`benchmarks/real_data/eval_setup/ged_bounds.py` (429 l) and `validate_ged_bounds.py` (247 l) are the
existing, tested cross-check implementation — 35 passing unit tests, gate 2 passed with 0 violations
on 400 LINUX pairs. Reuse them; do not reimplement bounds.
`.claude/notes/review/plan/gedlib.md` is the GEDLIB API reference including both silent traps.
`.claude/notes/review/plan/gate2-linux-400-seed42.json` is the archived per-pair sample gate 2 replays.
