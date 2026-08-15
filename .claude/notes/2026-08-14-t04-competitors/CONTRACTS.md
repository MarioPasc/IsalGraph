# T-04 CONTRACTS — frozen wave-0 interfaces

**Written and committed by the orchestrator before any agent started.**
Wave id `2026-08-14-t04-competitors`. Branch `ticket/T-04-competitors`.

**Code against this file, never against a peer's progress.** If you need a change to
anything here, **message the orchestrator** — do not negotiate a contract directly with a
peer and do not work around it silently.

Authoritative above this file: [`T-04-design.md`](../review/tasks/T-04-design.md), as
amended 2026-08-15. Read its changelog first — **three numbers your brief quotes were
wrong** and are corrected there and in §0 below.

---

## 0. Read this before you write a line — wave 0 refuted three premises

Full evidence: [`WAVE0-FINDINGS.md`](WAVE0-FINDINGS.md). All three are PI-signed.

| # | What your brief may still imply | What is actually true |
|---|---|---|
| 1 | `grakel(n_iter=3) ≡ ours(h=2)`; grakel has an off-by-one | **No off-by-one.** `grakel(n_iter=k) ≡ h = k`. `grakel(n_iter=2) = 5.830952`, `grakel(n_iter=3) = 7.211103`. The off-by-one was in `scratch/backends.py::wl_features`, which compresses colours **per graph, per round** — do not port it. Frozen `h = 2` means **`n_iter = 2`** |
| 2 | adjacency on the running example is `'101001000100111'` | That is **row-major**. The frozen order is **column-wise** and the correct strings are `G = '101101000100011'`, `H = '101001000100011'`. graph6 `'ElCW'` unpacks to exactly the first of those |
| 3 | README §4.1's ρ table is the measurement to reproduce | It is a **composite of three draws** and differs from `real_suite1.out` by up to 0.074. **Do not reproduce ρ at all** — that is `reproduce.py`, and it is the orchestrator's |

**Environment, re-measured 2026-08-15**: `pynauty==2.8.8.1` **is now installed** in
`isalgraph-cpp`. `grakel` is `GraKeL-0.1.10` with a stale `__version__ == '0.1.8'` string —
the folder and the design note describe **one** installation; do not re-open it.
`networkx` 3.6.1 · `numpy` 1.26.4 · `rapidfuzz` 3.14.5 · `scipy` 1.17.1 · Python 3.11.15 ·
`isalgraph.engine() == "cpp"`, `build_hash 298fc1188bf1b051`.

---

## 1. What exists already, and is yours to use but not to edit

Everything below is **committed and read-only to you**. Report defects; do not fix them.

```
src/isalgraph/competitors/
  __init__.py        public API
  base.py            the ABCs and value objects          <- §2
  registry.py        the two lazy registries             <- §3
  bits.py            the ONLY producer of a BitCount     <- §5
  fixtures.py        stdlib-only graph fixtures          <- §6
  datasets.py        real-cohort loading, GRAPHS ONLY    <- §7
  ged_reference.py   certified exact GED. DO NOT IMPORT  <- §7
  smoke.py           the frozen CLI                      <- §8
  grid.py  f5.py  reproduce.py                           orchestrator's
  metrics/{levenshtein,hamming,kernel,size_null}.py
  backends/{isalgraph_ref,size_null}.py
```

**You create only the files your brief lists.** Both suites' cohorts are on disk, including
Suite 2 — the raw IAM GXL is *not* on this workstation, but the exported CSR `.npz` files
were recovered from Picasso on 2026-08-15 and `datasets.load("mutagenicity")` works.

---

## 2. `base.py` — the ABCs and value objects

### `Encoding` — why `encode()` does not return `str`

```python
@dataclass(frozen=True, slots=True)
class Encoding:
    backend: str                      # provenance, carried into every record
    symbols: tuple[str, ...]          # THE COMPARISON UNIT. one entry == one edit.
    alphabet_size: int                # |Sigma| for the entropy bound; may depend on n
    n_nodes: int
    n_edges: int
    text: str                         # FIGURES AND DEBUGGING ONLY. never measured.
    wire: bytes | None = None         # the realised serialisation, exactly as emitted
    payload_bits: int | None = None
    frame: PositionalFrame | None = None
```

- **`symbols` is the unit of edit.** One DFS tuple for min-DFS, one ASCII byte for graph6,
  one triangle bit for adjacency. Every edit distance runs on this and nothing else.
- **`text` is never measured.** A test asserts nothing under `metrics/` or `bits.py`
  references it, except `levenshtein_char`, which exists to *report* the character-level
  answer as a supplementary number.
- **`wire` is measured, never computed.** graph6's `1 + ⌈n(n−1)/12⌉` is right only for
  `n ≤ 62` and Suite 2 reaches `n = 98`. Populate `wire` with the bytes `networkx` emitted;
  `bits.py` counts them. Closed forms belong in your tests as oracles inside their range.

### `PositionalFrame` — **changed in wave 0, read this**

```python
@dataclass(frozen=True, slots=True)
class PositionalFrame:
    n_nodes: int
    pairs: tuple[tuple[int, int], ...]   # (i, j), i < j, strict upper triangle COLUMN-WISE
    bits: tuple[str, ...]                # '0'/'1' per entry of pairs, same length, same order
```

`bits` is **new and mandatory**. The first draft zipped `pairs` against `symbols`, which is
wrong for half the pool: graph6's symbols are six-bit ASCII *bytes* while its frame is the
unpacked *bit* triangle underneath. `padded_hamming` now reads **only** `frame.pairs` and
`frame.bits`. For `adjacency` the two coincide; for `graph6`, `nauty_graph6` and `agm_cam`
they do not, so populate `bits` with the triangle, not with the symbols.

Declare `Capability.POSITIONAL_FRAME` **iff** you populate `frame`. sparse6, min-DFS,
IsalGraph and WL have none — `padded_hamming` is then *undefined* there, which is a
**reported F1 result**, not an error to work around.

### `Budget` — a budget that runs out raises

```python
@dataclass(frozen=True, slots=True)
class Budget:
    search_nodes: int | None = None      # AGM
    max_projections: int | None = None   # min-DFS  -- a MEMORY cap
    timeout_s: float | None = None       # IsalGraph
```

`None` means unbounded; a backend reads only the field it declares. **Never return an
incumbent, a heuristic or a degraded value** — that puts a non-canonical code in a column
headed canonical, which is precisely the error `graph6` is in the pool to expose.

### `Capability`

`POSITIONAL_FRAME` · `CANONICAL` · `COMPLETE_INVARIANT` · `REVERSIBLE` ·
`HANDLES_DISCONNECTED` · `SUITE1_ONLY` · `BASELINE`.

Declared, never inferred. `SUITE1_ONLY` **raises** above its scope rather than producing a
76 %-complete column; the reference arm's guard is `SUITE1_MAX_NODES = 12`
(`backends/isalgraph_ref.py`), and Suite 1's true maximum is `n = 12` on AIDS, so that
bound admits every Suite-1 graph and rejects everything above.

### The three protocols

```python
class ReprBackend(ABC):
    name: str
    capabilities: frozenset[Capability]
    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding: ...
    def decode(self, encoding: Encoding) -> nx.Graph:   # or raise NotReversible
    def bits(self, encoding: Encoding) -> BitCount:     # delegates to bits.py. DO NOT OVERRIDE.
    @classmethod
    def is_available(cls) -> bool: ...

class VectorBackend(ABC):            # WL only. NOT a ReprBackend.
    def fit(self, graphs: Sequence[nx.Graph]) -> None:   # per DATASET, never per batch
    def features(self, graph: nx.Graph) -> Mapping[str, int]: ...
    # bits() is intentionally ABSENT -> the Claim A cell is empty, with the reason printed

class DistanceMetric(ABC, Generic[ComparableT]):
    name: str
    consumes: Literal["symbols", "text", "frame", "features", "order"]
    is_pseudometric: bool            # DECLARED, per F2. never inferred.
    def is_defined(self, a, b) -> bool: ...
    def distance(self, a, b) -> float: ...
```

`DistanceMetric` is generic so a concrete metric can state `DistanceMetric[Encoding]`
without violating Liskov against a base declared over the union. You do not write metrics.

---

## 3. `registry.py` — registration and lookup

```python
register_backend(name: str, factory: Callable[..., ReprBackend | VectorBackend]) -> None
register_metric(name: str, factory: Callable[..., DistanceMetric[Any]]) -> None

get_backend(name, **kwargs)         -> ReprBackend | VectorBackend
get_repr_backend(name, **kwargs)    -> ReprBackend        # raises if it is the VectorBackend
get_vector_backend(name, **kwargs)  -> VectorBackend
get_metric(name, **kwargs)          -> DistanceMetric[Any]

available_backends(*, include_baseline: bool = False) -> tuple[str, ...]
registered_backends() -> tuple[str, ...]
unavailable_backends() -> dict[str, str]
```

**Register at module import time**, at the bottom of your module:

```python
register_backend("graph6", Graph6Backend)
```

`_LAZY_MODULES` in `registry.py` already maps every one of the eleven names to its module.
**Your module path is fixed by that table** — `nauty_graph6` *and* `sparse6_nauty` both
resolve to `backends/nauty.py`, so agent B registers both from that one file.

**Factories take keyword arguments.** That is how `WLSubtree` is instantiated at an `h`
other than the frozen 2 for the identity check, without a mutable attribute anything else
could reach — see §4.

**A missing dependency raises `BackendUnavailableError`, never a silent degrade.** Put your
third-party import inside `is_available()` and inside the method that needs it, never at
module top level. `import isalgraph.competitors` must succeed with `networkx`, `pynauty`,
`grakel` and `rapidfuzz` all absent, and a test enforces it.

---

## 4. The one cross-edge, and the one constructor

**A → B.** Agent B imports agent A's sparse6 serialiser to register `sparse6_nauty`:

```python
# in backends/sparse6.py, agent A -- module-level function, not a method
def serialise(graph: nx.Graph) -> Encoding: ...
```

Exact signature, exact name, module `isalgraph.competitors.backends.sparse6`. One-way, no
cycle. **B codes against this signature, not against A's progress.** If A needs to change
it, A messages the orchestrator and the orchestrator relays.

**C's WL constructor.** `reproduce.py` calls `get_vector_backend("wl_subtree", h=k)`, so:

```python
class WLSubtree(VectorBackend):
    def __init__(self, h: int = 2, *, normalize: bool = False) -> None: ...
```

`h = 2` is the frozen default and **must not be tuned on ρ**. The keyword exists so the
identity check can instantiate `h ∈ {1, 2, 3}` without mutating a shared object.

---

## 5. `bits.py` — the only producer of a `BitCount`

**Do not override `ReprBackend.bits()`.** Its table already has your row:

| Backend | `entropy_bits` | `realised_bits` | `payload_bits` |
|---|---|---|---|
| `adjacency`, `agm_cam` | `n(n−1)/2` | `8·⌈n(n−1)/16⌉` | = entropy |
| `graph6`, `nauty_graph6` | `6·len(wire)` | `8·len(wire)` | `n(n−1)/2` |
| `sparse6`, `sparse6_nauty` | `6·len(wire) − 6` *(`':'` excluded)* | `8·len(wire)` *(`':'` included)* | — |
| `min_dfs` | `m · 2⌈log₂ n⌉` | `8·len(text)` — **`inflated=True`** | — |
| `isalgraph_*` | `L · log₂ 9` | `8·L` | — |
| `wl_subtree`, `size_null` | **raises `BitCountUndefined`** | raises | raises |

`bits.count()` reads `encoding.wire`, `n_nodes`, `n_edges`, `alphabet_size` and `text`.
**Populate `wire` for the four formats that have one**, or it raises with a message telling
you so. Nothing in `bits.py` reads `.text` except the `min_dfs` row, whose value is flagged
`inflated=True` precisely so it cannot be quoted unlabelled.

**Never `len(text) * 8` in your own code.** A dedicated test asserts
`adjacency.bits(e).realised_bits < len(e.text)` on every fixture.

---

## 6. `fixtures.py` — and the relabeller you must use

Fixtures are `(n_nodes, edges)` tuples so the module needs no `networkx`:
`RUNNING_EXAMPLE` · `RUNNING_EXAMPLE_MINUS_EDGE` · `K33` · `PRISM` ·
`C4_PLUS_K3_DISJOINT` · `PATH_2` · `EMPTY_3`, plus `ALL_FIXTURES` and
`CONNECTED_FIXTURES`. `to_networkx(fixture)` builds the graph.

**Use `fixtures.shuffled_copy(graph, rng)` for every F3 test.** Do not write your own and do
not use `nx.relabel_nodes(copy=True)`, which **preserves insertion order** and makes
order-dependent formats look invariant (finding 13). `shuffled_copy` rebuilds the graph with
a fresh insertion order: three shuffles, in the order mapping → insertion order → edges.

> **That draw sequence is load-bearing.** `reproduce.py` replays the scout's `Random(42)`
> stream through 5 × 50 × 20 of these calls to land on the same ρ sample. **Changing the
> number of `rng` draws desynchronises the reproduction gate.** Do not modify it.

**Write a test that your relabeller can make `graph6` fail.** An F3 harness that cannot fail
is worthless, and this one is shared by all three of you.

---

## 7. Data — and the F5-blindness rule that is structural

```python
from isalgraph.competitors import datasets
cohort = datasets.load("iam_letter_low")     # Cohort(name, suite, graphs, graph_ids)
idx    = cohort.sample(200, seed=42)         # sorted indices, fresh Random(seed)
datasets.SUITE1   # iam_letter_low, iam_letter_med, iam_letter_high, linux, aids
datasets.SUITE2   # grec, aids_iam, coil_del, mutagenicity, protein
```

Root is `$ISALGRAPH_COHORT_ROOT`, defaulting to
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data`.

> 🔴 **Never import `isalgraph.competitors.ged_reference`, and never import anything that
> does.** Decision 24's whole defence is that T-04a's exclusion rule is **F5-blind by
> construction**: it cannot see the correlation it would otherwise be selecting on. That is
> enforced by the import graph — `datasets.py` loads graphs and nothing else, `grid.py`
> imports only `datasets`, and **a test asserts `grid.py`'s import closure reaches no GED
> loader**. An import in your backend module lands inside that closure and breaks it.
>
> **You have no reason to compute ρ at all.** If you find yourself doing so, you have left
> the ticket.

---

## 8. `smoke.py` — the frozen CLI

```
python -m isalgraph.competitors.smoke \
    --backends <comma-separated> --dataset <name> --n-graphs 200 \
    --seed 42 --out <path.json>
```

Per backend: `available`, `import_error`, `capabilities`, `n_encoded`, `n_failed`,
`failures[{graph_id, n_nodes, exception, elapsed_s}]`, `ms_per_graph{p50,p90,max}`,
`f3_invariant_of_50`, `bits{entropy_p50, realised_p50}`; plus a `header` carrying
`platform`, `python`, package versions, `isalgraph_engine` and `isalgraph_build_hash`.

It dispatches through the registry, so **registering your backend is all you need** — do
not edit `smoke.py`. Run it locally on real data and paste the JSON into your work log.

**Every failure is recorded, never dropped.** The failure *rate* is a reported number: AGM's
24 % on GREC and min-DFS's 24/400 on Mutagenicity are results the paper prints. A stated
ceiling is a result; a silent one is a defect.

**You do not run anything on Picasso.** No ssh, no rsync, no `sbatch`. The orchestrator runs
one loginexa session for all eleven backends and sends you your slice by `SendMessage`; you
close your Picasso criterion with what it returns.

---

## 9. Frozen conventions — one row per backend

| Backend | One symbol is | Frame? | Primary distance |
|---|---|:---:|---|
| `adjacency` | one triangle bit, **strict upper triangle COLUMN-WISE** | ✔ | Levenshtein / padded Hamming |
| `graph6`, `nauty_graph6` | one ASCII byte of `to_graph6_bytes(header=False)` | ✔ (unpacked bits) | as above |
| `sparse6`, `sparse6_nauty` | one ASCII byte, `':'` **excluded** from `symbols` | ✘ | Levenshtein |
| `agm_cam` | one code bit, **minimum**, strict lower triangle row-wise ≡ upper column-wise | ✔ | padded Hamming |
| `min_dfs` | **one DFS tuple** | ✘ | Levenshtein, tuple-level |
| `isalgraph_*` | one instruction over Σ, `\|Σ\| = 9` | ✘ | Levenshtein |
| `wl_subtree` | n/a — feature vector | ✘ | kernel distance |
| `size_null` | n/a — `n` | ✘ | `\|n₁ − n₂\|` |

**The reading order is one convention and it is asserted in code.** `adjacency.symbols`,
graph6's unpacked payload and `agm_cam` on the identity permutation must be the **same bit
sequence**. Verified in wave 0: `'ElCW'` → `l=101101`, `C=000100`, `W=011000`, first 15 bits
`101101000100011`; `scratch/agm_cam.py::_code_from_perm` walks `for k in 1..n-1: for j in
0..k-1`, the same order. **AGM takes the minimum; FFSM takes the maximum** — state it once
in `agm.py`'s module docstring.

**Frozen budgets — these are the values behind published failure rates. Do not change them.**
AGM `200_000` (Suite 1) / `100_000` (Suite 2); min-DFS `max_projections = 50_000`;
IsalGraph `timeout_s = 2.0` (reproduction gate only — T-06 sets production).

**Exception names, frozen** (`isalgraph.errors`): `CompetitorError` ·
`BackendUnavailableError` · `BackendNotFoundError` · `BudgetExceeded` → `AGMBudgetExceeded`,
`MinDfsBudgetExceeded` · `BitCountUndefined` · `DistanceUndefined` · `NotReversible` ·
`SuiteScopeError`.

---

## 10. House rules

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pytest <your test files> -q
$PY -m ruff check --fix src/ tests/          # must end clean
$PY -m mypy --strict src/isalgraph/          # must end clean
```

`src/isalgraph/` currently passes both across 62 files. **Never
`export PYTHONPATH=$REPO/src`** — a src-first path shadows the installed package and
silently falls back to pure Python, so a benchmark measures nothing.

Use `TYPE_CHECKING` for `networkx` types, matching the rest of the package:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import networkx as nx
```

**Commit incrementally on your own branch, not at the end.** Sessions die; uncommitted work
cannot be merged. Write your work log at
`.claude/notes/2026-08-14-t04-competitors/track-<X>-<name>.md` with the sections your brief
lists, and put the real `git diff --stat` in it.

**An agent reporting that the brief is wrong is a success.** Wave 0 found three such things
before you started. Bring evidence, do not tune a test to pass.
