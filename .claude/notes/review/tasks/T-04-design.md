# T-04 — Competitor backends: frozen design

**Written 2026-08-14, before any competitor code lands in `src/`.** Owner: T-04.
Serves **AE.4a** (requirement modal, the real owner), **AE.3**, **R1.1**, **R1.2a/b**, **R3.6a**.
Gates **T-04a**, **T-06**, **T-17**.

This file exists so that the module contract, the symbol and bit conventions, the budgets, the
failure policy and the reproduction gate are on record **before** any number is produced by the
shipped code. Nothing here may be changed after the first agent starts without a dated changelog
entry at the bottom.

**Inputs**: [competitors/README](../plan/competitors/README.md) **first**, then
[competitors](../plan/competitors.md), the seven per-competitor files, and
[preregistration](../plan/preregistration.md) §4–§5.
**Signed decisions that bind this ticket**: 2 (IsalHG `iso_backends` style), 18 (distances selected
by measurement), 23 (`N_max = 197` frozen), 24 (BH over `N_actual`, exclusion rule **F5-blind**).

---

## 0. State measured on 2026-08-14, and where it differs from the plan

Every row was read live on this workstation, not assumed. **Four differ from what the plan
predicted, and two of them change the design.**

| Item | Plan says | Measured 2026-08-14 | Consequence |
|---|---|---|---|
| **`pynauty`** | `competitors/README` §"Environment": `pynauty 2.8.8.1` | **absent from `isalgraph-cpp`**; the only copy on this machine is in the **`isalhg`** env | ⚠ **The scout's evidence was produced in a different interpreter than the shipped code will run in.** `scratch/backends.py:133` shells out to `isalgraph-cpp` for IsalGraph strings, which is the tell. `pynauty==2.8.8.1` must be installed into `isalgraph-cpp` **and** built on Picasso |
| **`grakel`** | `README` §"Environment": **0.1.8** | **0.1.10** in `isalgraph-cpp` | the `n_iter` off-by-one (finding 12) was verified against 0.1.8. **Re-verify `grakel(n_iter=3) ≡ ours(h=2) = 5.830952` under 0.1.10** before any WL number is quoted |
| **`wl_kernel_computer.py`** | E10: "WL kernel computed, never reported" | exists at `benchmarks/real_data/eval_setup/wl_kernel_computer.py`, **`n_iter = 5` by default**, `normalize=False`, consumed by `eval_setup.py::wl_n_iter` | ⚠ **grakel `n_iter = 5` is our `h = 4`.** The folder selects **`h = 2`**, and measures `h = 3` strictly worse than `h = 2` on all five datasets. The repo's existing default is **two refinement rounds past the selected one**. Finding 12 made concrete: reconcile, do not re-quote |
| `src/isalgraph/competitors/` | — | **does not exist** | greenfield; no merge risk against existing source |
| Suite-1 exported graphs | `data/exported/<ds>.npz` | **present**, 5 files + `manifest.json` | the reproduction gate has its input |
| T-03 certified exact GED | `extended_merged_exact_ged/computed/<ds>.npz` | **present** under `…/results/exact_ged` | ρ reproduction is possible without recomputing GED |
| `rapidfuzz` | not mentioned | **3.14.5 installed**; `Levenshtein.distance(("0-1","1-2","2-0"), ("0-1","2-0")) == 1` (symbol level), `== 4` at character level; `Hamming.distance` **raises `ValueError`** on unequal lengths | **decides §2.2**: symbol-level edit distance needs no new dependency and no interning, and Hamming's raise is exactly the F1-undefined signal |
| `python-Levenshtein` 0.27.4, `networkx` 3.6.1, `numpy` 1.26.4, engine | — | present; **`isalgraph.engine() == "cpp"`**, `build_hash 298fc1188bf1b051` | reference arm runs on the engine |
| `pyproject.toml` extras | — | no `competitors` extra | must be added |
| Base commit | — | **`152c80d18293d6e699bd36cf301a88a7596c6464`**, tree clean | |

### The consequence that is not a row

The competitors folder's numbers are the **evidence base for five plan-level findings that reach a
printed number** (board header, T-04 warning). They were produced by scratch code, in an env with a
different `grakel` and a `pynauty` the shipped env does not have. **Until the shipped module
reproduces them, the plan is resting on unreproduced measurements.** That is why §8's reproduction
gate is criterion 1 and not a nice-to-have.

---

## 1. What T-04 is, and what it is not

**T-04 builds the machinery. It does not run the science.**

| In scope | Owner |
|---|---|
| `src/isalgraph/competitors/` — two ABCs, lazy registry, 11 backends, 6 metrics, bit accounting | **T-04** |
| The validation oracles: AGM brute force, min-DFS exhaustive DFS enumeration, the `kaviniitm` gate, the WL cross-check | **T-04** |
| The **reproduction gate** — the shipped module reproduces `competitors/`'s headline numbers | **T-04** |
| A runnable but **not run** (representation × distance) grid, F5-blind | **T-04** ships it, **T-04a** runs it |
| The `size_null` baseline as a first-class backend + metric | **T-04** |
| A language-matched timing mode (Python reference vs Python competitors) for Fig. 2 | **T-04** ships it, **T-06** runs it |

| Out of scope | Owner |
|---|---|
| Selecting each primary distance under `competitors.md` §3.4 | **T-04a** |
| Production distance matrices, Claim A/B tables, the confirmatory family | **T-06** |
| The AE.3 comparison table | **T-17** |
| Restating `competitors.md` §4's pre-committed outcome 3 in terms of `m/n` (finding 4) | **T-20**, prose |
| Adding the missing reduction-rule case for a Suite-1-only representation (finding 6) | **T-02's owner** |

**If an agent finds itself computing ρ for a paper table, it has left the ticket.** ρ appears in
T-04 exactly once, inside the reproduction gate, and its job there is to equal a number that is
already written down.

---

## 2. Architecture

### 2.1 Placement and the import contract

```
experiments/ benchmarks/     -> anything
isalgraph.competitors        -> optional: networkx, pynauty, grakel, rapidfuzz  [NEW TIER]
isalgraph.viz                -> optional: matplotlib, networkx, igraph
isalgraph.adapters           -> optional: networkx, igraph, pyg
isalgraph.core               -> stdlib only (+ the optional C++ engine)
```

`competitors` sits at the same tier as `viz` and `adapters` and inherits their contract, which a
test enforces:

- **`import isalgraph.competitors` must succeed with `networkx`, `pynauty`, `grakel` and
  `rapidfuzz` all uninstalled.** Every third-party import lives inside a function body or behind
  the lazy registry.
- **A missing dependency raises `BackendUnavailableError`. It never degrades silently.** Same rule
  as `isalgraph.core.backends`, same reason.
- **`isalgraph/__init__.py` does not import `competitors`.** The top-level import chain stays
  stdlib-only.

```
src/isalgraph/competitors/
  __init__.py            public API only: the two ABCs, get_backend/get_metric,
                         available_backends/available_metrics, the value objects
  base.py                ReprBackend, VectorBackend, DistanceMetric, Encoding,
                         BitCount, PositionalFrame, Capability
  registry.py            two lazy registries (backends, metrics), _LAZY_MODULES
  bits.py                the ONLY place a bit count is produced
  fixtures.py            the running example, K3,3, the prism, C4+K3 — stdlib only
  smoke.py               `python -m isalgraph.competitors.smoke`      (frozen CLI, §7)
  grid.py                `python -m isalgraph.competitors.grid`       (F1-F4, F6 — NEVER F5)
  f5.py                  `python -m isalgraph.competitors.f5`         (F5 only; NOT an input to selection, §4.5)
  reproduce.py           `python -m isalgraph.competitors.reproduce`  (the reproduction gate, §8.1)
  metrics/
    levenshtein.py       symbol-level (primary) and character-level (supplementary)
    hamming.py           plain and padded
    kernel.py            RKHS distance for VectorBackend
    size_null.py         |n1 - n2|
  backends/
    adjacency.py  graph6.py  sparse6.py          [agent A]
    nauty.py  agm.py                             [agent B]
    min_dfs.py  wl.py                            [agent C]
    isalgraph_ref.py  size_null.py               [orchestrator]
```

### 2.2 The value object — why `encode()` does not return `str`

Four of the seven "watch for" traps in `competitors/README` §6 are the same mistake: **a string was
measured with the wrong unit.** Counting `'1010…'` at 8 bits per character inflates the adjacency
matrix 8×. Running Levenshtein on `'0-1 1-2 2-0'` charges 4 edits for one deleted DFS tuple, a 2×
difference. Both are invisible; both produce a plausible number.

A `str` return type cannot prevent either. So `encode()` returns a value object whose fields make
the unit explicit and make the wrong unit unreachable.

```python
@dataclass(frozen=True, slots=True)
class Encoding:
    backend: str                      # provenance, carried into every record
    symbols: tuple[str, ...]          # THE comparison unit. one entry == one edit.
    alphabet_size: int                # |Σ| for the entropy bound; may depend on n
    n_nodes: int
    n_edges: int
    wire: bytes | None                # the realised serialisation, exactly as the format emits it
    payload_bits: int | None          # format-defined payload where it differs from the wire
    frame: PositionalFrame | None     # present iff the code is a positional bit vector
    text: str                         # FIGURES AND DEBUGGING ONLY. never measured.
```

- **`symbols` is the unit of edit.** For min-DFS one symbol is one DFS tuple; for graph6 one symbol
  is one ASCII byte; for adjacency one symbol is one bit. Every edit distance runs on `symbols` and
  on nothing else.
- **`text` is never measured.** A test asserts that no module under `metrics/` or `bits.py`
  references `.text`.
- **`wire` is measured, never computed.** See §4.2.
- **`frame` gates padded Hamming.** A backend without a positional frame (sparse6, min-DFS,
  IsalGraph, WL) makes padded Hamming *undefined*, which is a reported F1 result, not an error.

Measured 2026-08-14: `rapidfuzz.distance.Levenshtein.distance` accepts `tuple[str, ...]` and
returns the **symbol-level** answer (1 where the character-level answer is 4). No interning, no new
dependency. `rapidfuzz.distance.Hamming.distance` raises `ValueError` on unequal lengths, which the
metric catches and reports as **undefined** rather than propagating.

### 2.3 The two ABCs

```python
class ReprBackend(ABC):
    name: str
    capabilities: frozenset[Capability]   # POSITIONAL_FRAME, CANONICAL, COMPLETE_INVARIANT,
                                          # REVERSIBLE, HANDLES_DISCONNECTED, SUITE1_ONLY, BASELINE
    def encode(self, G, *, budget: Budget | None = ...) -> Encoding: ...
    def decode(self, e: Encoding): ...        # or raise NotReversible
    def bits(self, e: Encoding) -> BitCount:  # delegates to bits.py; not overridden lightly
    def is_available(self) -> bool: ...

class VectorBackend(ABC):                     # WL only. NOT a ReprBackend.
    name: str
    def fit(self, graphs: Sequence) -> None:  # per DATASET, never per batch
    def features(self, G) -> Mapping[str, int]: ...
    # bits() intentionally absent -> Claim A cell is empty with the reason printed

class DistanceMetric(ABC):
    name: str
    consumes: Literal["symbols", "frame", "features", "order"]
    is_pseudometric: bool                     # DECLARED, per F2. never inferred.
    def is_defined(self, a, b) -> bool: ...   # F1
    def distance(self, a, b) -> float: ...    # precondition: is_defined
```

`VectorBackend` is a separate protocol rather than a `ReprBackend` with a raising `bits()`, per
[wl-subtree-kernel](../plan/competitors/wl-subtree-kernel.md) §7: *"Give it its own protocol or make
`bit_length` raise; returning a fabricated number is the failure mode to avoid."* A separate type
makes the fabrication unreachable rather than merely forbidden.

**`GEDBackend` from `competitors.md` §1 is not built here.** GEDLIB already has a home in
`benchmarks/real_data/eval_setup/ged_backends.py` (T-27, 59 KB, tested). Duplicating it into
`competitors/` would fork the cost model. Out of scope, stated so nobody re-derives it.

### 2.4 Registry

IsalHG's `iso_backends/registry.py` idiom, verbatim in shape: `_REGISTRY`, `_LAZY_MODULES`,
`register_*`, `get_*`, `available_*` (which lazily imports everything first so the listing reflects
the installable set), `_reset_for_testing`. Two registries, one for backends and one for metrics,
so T-04a's grid is `product(available_backends(), available_metrics())` and a new cell cannot be
forgotten.

---

## 3. The roster — frozen

Eleven registered backends. The board row names five; the folder documents seven; the
preregistration fixes **6 Claim-A serialisations and 7 Claim-B comparators**. The roster below is
what satisfies all three plus the reference arm and the baseline.

| Name | Protocol | Family | Claim A | Claim B | Owner |
|---|---|---|:---:|:---:|---|
| `adjacency` | Repr | `n²` | ✔ | ✔ | **A** |
| `graph6` | Repr | `n²` | ✔ | ✔ | **A** |
| `sparse6` | Repr | `m log n` | ✔ | ✔ | **A** |
| `nauty_graph6` | Repr | `n²`, canonical | ✔ | ✔ | **B** |
| `sparse6_nauty` | Repr | `m log n`, canonical | supplementary | supplementary | **B** |
| `agm_cam` | Repr | `n²`, canonical, **Suite 1 only** | ✔ (≡ adjacency) | ✔ | **B** |
| `min_dfs` | Repr | `m` | ✔ | ✔ | **C** |
| `wl_subtree` | **Vector** | — | ✘ *(no bit count)* | ✔ | **C** |
| `isalgraph_pruned` | Repr | `m` | reference arm | reference arm | orch |
| `isalgraph_canonical` | Repr | `m`, **Suite 1 only** | reference arm | reference arm | orch |
| `size_null` | Repr | trivial | ✘ | **baseline only** | orch |

`sparse6_nauty` is **supplementary, not a family member.** `preregistration.md` §4.1 fixes the
Claim-A set at 6 and the Claim-B set at 7, and decision 23 freezes `N_max = 197`. Adding an eighth
comparator would change `N_max`. It is registered because [sparse6](../plan/competitors/sparse6.md)
§3 says the canonicalised variant "costs nothing and removes a reviewer objection", and it is
reported in the supplementary grid.

### 3.1 `size_null` — a backend, against the folder's advice, and why

[competitors/README](../plan/competitors/README.md) §6 says *"Port as an analysis, not a backend:
`real_size_null.py`. Finding 1 is not a competitor property."*

**Overruled, PI-approved 2026-08-14.** `|n₁ − n₂|` *is* a distance on the trivial representation
`encode(G) = n`. Registering it makes the null column fall out of every ρ table and out of T-04a's
grid **by construction**, so a printed ρ without its null becomes unreachable rather than merely
discouraged. Finding 1's own text is *"every printed ρ needs the null beside it"*, and a column
produced by a separate script is a column a later ticket can forget — which is exactly how finding 1
stayed unowned until the scout tripped over it.

Two guards, both mandatory:

1. `size_null` carries `Capability.BASELINE`. **`grid.py` refuses to select a `BASELINE` backend as
   any representation's primary distance**, and `get_backend` returns it only when named explicitly.
2. **`size_null` is outside the frozen confirmatory family.** It is not a Claim-A serialisation and
   not one of the 7 Claim-B comparators. It changes neither `N_max = 182` nor `N_actual`. It is a
   descriptive baseline row. **An agent that adds it to a family has broken decision 23 — stop.**

It is a pseudometric (identity of indiscernibles fails: two non-isomorphic graphs on `n` nodes get
distance 0), declared as such, exactly like WL.

---

## 4. Frozen conventions

Everything in this section is fixed **before** the run that produces the numbers it governs.

### 4.1 Symbol convention — one row per backend

| Backend | One symbol is | Primary | Also exposed |
|---|---|---|---|
| `adjacency` | one triangle bit, **strict upper triangle read COLUMN-WISE** | Levenshtein / padded Hamming | — |
| `graph6`, `nauty_graph6` | one ASCII byte of `to_graph6_bytes(header=False)` | as above | payload bits separately |
| `sparse6`, `sparse6_nauty` | one ASCII byte, `':'` **excluded** from `symbols` | Levenshtein | — |
| `agm_cam` | one code bit, **minimum**, strict lower triangle row-wise ≡ upper column-wise | padded Hamming | — |
| `min_dfs` | **one DFS tuple** | Levenshtein, tuple-level | character rendering, supplementary grid only |
| `isalgraph_*` | one instruction over `Σ`, `\|Σ\| = 9` | Levenshtein | — |
| `wl_subtree` | n/a — feature vector | kernel distance | — |
| `size_null` | n/a — `n` | `\|n₁ − n₂\|` | — |

**The reading order is one convention, asserted in code.** `adjacency`, `graph6` and `agm_cam` must
produce the *same bit sequence* for the same labelling, and a test asserts
`adjacency.symbols == unpack(graph6.wire payload)` bit for bit on a fixture set. That assertion is
what keeps `competitors/README` §2's four-member-family argument true in code rather than in prose.

**AGM takes the minimum; FFSM takes the maximum.** They are mirror images. The convention is stated
once in `agm.py`'s module docstring and in the paper, or the numbers are unreproducible
([agm](../plan/competitors/agm.md) §1).

### 4.2 Bit accounting — measure, never compute

`bits.py` is the only module that produces a `BitCount`. Both conventions
([competitors](../plan/competitors.md) §5) are always emitted.

| Backend | `entropy_bits` | `realised_bits` | `payload_bits` |
|---|---|---|---|
| `adjacency`, `agm_cam` | `n(n−1)/2` | `8·⌈n(n−1)/16⌉` | = entropy |
| `graph6`, `nauty_graph6` | `6·len(wire)` | `8·len(wire)` | `n(n−1)/2` |
| `sparse6`, `sparse6_nauty` | `6·len(wire) − 6` *(`':'` excluded)* | `8·len(wire)` *(`':'` included)* | — |
| `min_dfs` | `m · 2⌈log₂ n⌉` | `8·len(character rendering)` — **inflated, labelled so** | — |
| `isalgraph_*` | `L · log₂ 9` | `8·L` | — |
| `wl_subtree`, `size_null` | **raises `BitCountUndefined`** | raises | raises |

> **`len(wire)`, not a closed form.** graph6's `1 + ⌈n(n−1)/12⌉` is correct only for `n ≤ 62`;
> above that `N(n)` is 4 bytes and Suite 2 reaches **`n = 98`**, so the branch is live
> ([graph6](../plan/competitors/graph6.md) §7). The closed forms exist **only as test oracles inside
> their valid range**. Production code measures the bytes `networkx` actually emitted.

> **Never `len(text) * 8`.** `'101001…'` is a debugging view. Counting it as 8 bits per character
> inflates the adjacency matrix 8× and hands us a baseline we beat for free
> ([adjacency-matrix](../plan/competitors/adjacency-matrix.md) §7). A dedicated test asserts
> `adjacency.bits(e).realised_bits < len(e.text)` on every fixture.

### 4.3 Budgets and the failure policy

**A budget that runs out raises. It never returns an incumbent, a heuristic, or a degraded value.**
Returning AGM's greedy incumbent would put a non-canonical code in a column headed canonical, which
is precisely the error `graph6` is in the pool to expose.

| Backend | Budget | Value, frozen | Raises |
|---|---|---|---|
| `agm_cam` | search nodes | **200,000** (Suite 1) / **100,000** (Suite 2) | `AGMBudgetExceeded` |
| `min_dfs` | **projections (memory)** | **50,000** | `MinDfsBudgetExceeded` |
| `isalgraph_*` | wall clock | **2.0 s** *(for the reproduction gate only; T-06 sets production)* | `CanonicalizationTimeoutError` |
| everything else | — | none | — |

The `min_dfs` budget is on **memory, not time**: the construction holds every embedding realising
the current minimal prefix, and the first Suite-2 run was **OOM-killed (exit 137)** on Mutagenicity —
not slow, *killed* ([gspan-mdfsc](../plan/competitors/gspan-mdfsc.md) §7). A wall-clock cap does not
prevent this.

New exception hierarchy in `isalgraph/errors.py`, under the existing `IsalGraphError`:

```
CompetitorError(IsalGraphError)
├── BackendUnavailableError(CompetitorError, ImportError)   # dependency absent
├── BudgetExceeded(CompetitorError, RuntimeError)
│   ├── AGMBudgetExceeded
│   └── MinDfsBudgetExceeded
├── BitCountUndefined(CompetitorError, TypeError)
├── DistanceUndefined(CompetitorError, ValueError)          # F1 failure, a RESULT
└── NotReversible(CompetitorError, TypeError)
```

**Every failure is recorded, never dropped**: `(dataset, graph_id, backend, exception_class,
elapsed_s)`. The failure *rate* is a reported number — AGM's 24 % on GREC and min-DFS's 24/400 on
Mutagenicity are results the paper prints. A stated ceiling is a result; a silent one is a defect.

### 4.4 F3 — the isomorphism-invariance protocol

**50 real graphs × 20 relabellings per dataset, seed 42**, matching `competitors/README` §3.

> ⚠ **A relabelling built with `nx.relabel_nodes(copy=True)` alone preserves insertion order** and
> makes order-dependent formats look invariant (finding 13). Every relabelling **rebuilds the copy
> with a fresh insertion order**. A test asserts the relabeller actually changes insertion order —
> i.e. that `graph6` fails F3 on it — because an F3 harness that cannot fail is worthless.

Reported per dataset as `k / 50`. Expected, and part of the reproduction gate:
`graph6`/`sparse6`/`adjacency` land in **0–6 / 50**; `nauty_graph6`, `agm_cam`, `min_dfs`,
`wl_subtree`, `isalgraph_*` land at **50 / 50**.

### 4.5 F5-blindness is structural, not procedural

Decision 24 rests on T-04a's exclusion rule being **F5-blind by construction**: ties break on cost,
never on correlation with GED. Prose cannot enforce that. Therefore:

> **`grid.py` computes F1, F2, F3, F4 and F6. It has no code path that reads a GED value, and it
> cannot compute F5.** F5 lives in a separate command, `python -m isalgraph.competitors.f5`, whose
> output is reported and is **not an input to selection**. A test asserts that `grid.py`'s import
> closure does not reach any GED loader.

This is the cheapest way to make decision 24 defensible to a reviewer: the selection tool could not
have seen the outcome, because it cannot load it.

---

## 5. The (representation × distance) grid — shipped, not run

Six metrics: `levenshtein`, `levenshtein_char`, `hamming`, `padded_hamming`, `kernel`, `size_null`.
Eleven backends. Every cell is attempted; a cell that fails is a **result**
([competitors](../plan/competitors.md) §3.2).

| Metric | `consumes` | Defined when | Pseudometric |
|---|---|---|---|
| `levenshtein` | `symbols` | always | no |
| `levenshtein_char` | `text` **(the one sanctioned reader)** | always | no |
| `hamming` | `symbols` | equal length only | no |
| `padded_hamming` | `frame` | both backends declare `POSITIONAL_FRAME` | no |
| `kernel` | `features` | `VectorBackend` only | **yes** — declared |
| `size_null` | `order` | always | **yes** — declared |

**`padded_hamming` reads `frame`, not the string.** The frame is the triangle as index pairs *under
the backend's own labelling*, so a canonical backend pads its canonical frame and a non-canonical
one pads its incident frame, automatically. This is `adjacency-matrix.md` §7's rule made structural.
`scratch/backends.py::padded_hamming` takes two **graphs** and builds both triangles from the
incident node order, so it cannot express that rule — **do not port it**.

**sparse6 has no positional frame**, so `padded_hamming` is `undefined` there and is printed as such
([sparse6](../plan/competitors/sparse6.md) §7). That cell is one of the reasons the grid exists.

T-04 delivers `grid.py` and proves it runs end to end on a **20-graph dry run**. **T-04a** runs it on
the 200-graph stratified sample under its own protocol and applies §3.4's selection rule.

---

## 6. Decomposition

Three waves. **Contracts are written and committed by the orchestrator before any agent starts.**

### Wave 0 — orchestrator, alone

Owns everything shared, because an agent owning it blocks the other two and a mistake in it is
systemic.

`base.py` · `registry.py` · `bits.py` · `fixtures.py` · `metrics/**` · `backends/isalgraph_ref.py` ·
`backends/size_null.py` · `errors.py` additions · `pyproject.toml` extra ·
`.claude/notes/2026-08-14-t04-competitors/CONTRACTS.md` · the four entry points `smoke.py`,
`grid.py`, `f5.py` and `reproduce.py` (signature and JSON schema frozen in wave 0; they dispatch
through the registry, so an agent adding a backend extends them without editing them).

Also: install `pynauty==2.8.8.1` into `isalgraph-cpp`; build it on Picasso under
`module load gcc/12.2.0`; re-verify the `grakel` 0.1.10 off-by-one.

### Wave 1 — three agents, one wave, isolated worktrees

Worktrees are **safe here**: no track touches `src/isalgraph/core/native/`, and the only timings any
agent reports are *its own competitor's*, in pure Python, which the C++ extension does not affect.
The reference arm — the one thing that needs the engine — is the orchestrator's, and its timings are
taken **in place, alone**, in wave 2.

| Agent | Owns (create) | Competitors | The thing one owner prevents from drifting |
|---|---|---|---|
| **A** `competitor-serial` | `backends/{adjacency,graph6,sparse6}.py`, `tests/unit/test_competitors_serial.py` | adjacency · graph6 · sparse6 | the **strict upper triangle, column-wise** reading order, shared by all three and by AGM |
| **B** `competitor-canonical` | `backends/{nauty,agm}.py`, `tests/unit/test_competitors_canonical.py` | nauty→graph6 · sparse6-nauty · AGM CAM | the **`pynauty` dependency and the `canon_label` inversion assertion**, needed identically by both |
| **C** `competitor-mining` | `backends/{min_dfs,wl}.py`, `tests/unit/test_min_dfs.py`, `tests/unit/test_wl_subtree.py` | gSpan min-DFS · WL subtree | the **exhaustive oracles**, which are the deliverable, not the backends |

**One cross-edge, frozen in CONTRACTS.md**: `nauty.py` imports `sparse6.serialise(G) -> Encoding`
from A's module to register `sparse6_nauty`. Direction is one-way, no cycle. **B codes against the
frozen signature, not against A's progress.**

### Wave 2 — orchestrator

Merge one branch at a time from a clean checkout, fast suite after each. Then the cross-backend
gates that need every backend present (§8 criteria 3–6), the Picasso loginexa session, the reference
arm's in-place timings, the T-04a handoff, and `review-close`.

---

## 7. Picasso

**PI decision 2026-08-14: the orchestrator owns the cluster; no subagent ssh's, rsyncs or submits.**

The `review-ticket` non-negotiable and SCBI's **2-hour job floor** — a floor this account has already
been written to about, after a 12,600-task campaign — together forbid a few-minute `sbatch`. The
smoke therefore runs on **loginexa**: an interactive login node, 30-minute wallclock, **no queue and
no scheduler**, so the floor does not apply to it at all. That is what the `test-picasso-loginexa`
skill exists for. `.claude/loginexa.yaml` does not exist yet; the orchestrator writes it on first
invocation.

Frozen CLI, so all three agents and the cluster run the same thing:

```
python -m isalgraph.competitors.smoke \
    --backends <comma-separated> --dataset <name> --n-graphs 200 \
    --seed 42 --out <path.json>
```

JSON schema, per backend: `available`, `import_error`, `n_encoded`, `n_failed`,
`failures[{graph_id, exception, elapsed_s}]`, `ms_per_graph{p50,p90,max}`, `f3_invariant_of_50`,
`bits{entropy_p50, realised_p50}`, plus a run header carrying `platform`, `python`, package
versions, `isalgraph.engine()` and `build_info().build_hash`.

**Flow**: agents run it **locally on real data** and paste the JSON into their work log → the
orchestrator runs it **once on loginexa for all eleven backends** (~10 min) → each agent receives
its own slice by `SendMessage` and closes its Picasso criterion. Agents never hold cluster
credentials, and the timings are attributable because one process produced them.

**What the Picasso run is actually gating**: that `pynauty` **builds from source** under gcc 12.2.0
where the production run will happen. It is rehearsed
([nauty](../plan/competitors/nauty.md) §1) but not done, and a failure takes `nauty_graph6`,
`sparse6_nauty` and AGM's orbit pruning down together — which changes `k` in
`N_actual = 182 − 15k − 8d`. The `.so` does not rsync, for the same reason the C++ engine does not.

---

## 8. Acceptance criteria

Numbered, checkable, each naming the command or artifact that proves it. The orchestrator re-runs
every one of these itself; an agent's log is not evidence.

1. **The shipped module reproduces the evidence base.** `python -m isalgraph.competitors.reproduce
   --out repro.json` reproduces, from `src/`, on the same seeds and samples:
   - the **running-example** strings — `graph6 'ElCW'`, `nauty→graph6 'E@ro'`, `sparse6 ':EaWIzR'`,
     `adjacency '101001000100111'`, `AGM '000001110011110'`, min-DFS
     `(0,1)(1,2)(2,0)(2,3)(3,4)(4,5)(5,2)`, and their `H = G − (0,3)` counterparts — **exactly**;
   - the **K₃,₃ / prism** witness: `wl_subtree` distance **0.0000**, every other backend non-zero,
     with `nauty 'Es\o'` vs `'E{Sw'` and `AGM '000111111011100'` vs `'001101110111100'`;
   - **`grakel(n_iter=3) ≡ ours(h=2) = 5.830952`** under grakel **0.1.10**;
   - the **F3 real-cohort** column of `competitors/README` §3, per dataset, **exactly**;
   - the **Claim A** median entropy bits and the **% IsalGraph strictly shorter** tables of §4.3,
     all ten datasets, **exactly**;
   - the **ρ** tables of §4.1 and §4.2 on the seed-42 200-graph samples, to `1e-12`.

   > These are deterministic given the same seed and sample. **A mismatch is not a tolerance
   > question — it is a behaviour change, and it stops the ticket** (§9). ρ moving 0.07 between two
   > *independent* draws (finding 14) is not licence for a tolerance on the *same* draw.

2. **Validation oracles pass, and they are tests, not scripts.**
   `$PY -m pytest tests/unit/test_min_dfs.py tests/unit/test_agm_cam.py -q`
   - AGM vs brute force over all `n!` permutations: **327 graphs, 0 mismatches**; reversibility on
     all 327.
   - min-DFS **V1** vs exhaustive DFS enumeration: agrees on all **30** connected isomorphism
     classes with `n ≤ 5`. **V3**: distinct codes **1 / 2 / 6 / 21 / 112** at `n = 2…6` (OEIS
     A001349), **no collisions**. **V2**: **4,440** relabellings over `6 ≤ n ≤ 10`, 0 mismatches.
   - The `kaviniitm/DFSCode` gate is ported as an **acceptance test any third-party canonical
     backend must pass**, with **K2 (invariance) first** — K2 needs no oracle and is where that
     implementation died (46/90 non-invariant).

3. **The family identity holds in code.** A test asserts `adjacency`, `graph6` payload and `agm_cam`
   agree bit for bit on the reading order over a fixture set, so `competitors/README` §2's
   four-member-family claim is executable.

4. **No fabricated bit count and no 8× inflation.** `wl_subtree.bits()` and `size_null.bits()` raise
   `BitCountUndefined`; `adjacency.bits(e).realised_bits < len(e.text)` on every fixture; no module
   under `metrics/` or `bits.py` references `.text` except `levenshtein_char`.

5. **The dependency contract holds.** `import isalgraph.competitors` succeeds with `networkx`,
   `pynauty`, `grakel` and `rapidfuzz` all uninstalled; each absent dependency raises
   `BackendUnavailableError` on request, never a silent degrade. Enforced by a test that patches
   `sys.modules`.

6. **F5-blindness is structural.** A test asserts `grid.py`'s import closure reaches no GED loader.

7. **The grid runs.** `python -m isalgraph.competitors.grid --sample dryrun-20 --out grid.json`
   emits every (11 × 6) cell with F1/F2/F3/F4/F6 and an explicit `undefined` where applicable —
   including `padded_hamming × sparse6`.

8. **Picasso.** One loginexa session runs `smoke.py` for all eleven backends against one Suite-1 and
   one Suite-2 dataset; `pynauty` imports from a **from-source build** under gcc 12.2.0; every
   backend either encodes or raises a typed budget error; the JSON is archived under
   `.claude/notes/2026-08-14-t04-competitors/`.

9. **Suite scope is enforced, not documented.** `agm_cam` and `isalgraph_canonical` carry
   `Capability.SUITE1_ONLY`; requesting either on a Suite-2 dataset raises rather than silently
   producing a 76 %-complete column.

10. **House rules.** Full suite at or above the reference state (**726 passed / 271 skipped** with
    the engine, **561 / 276** without); `$PY -m ruff check src/ tests/` clean;
    `$PY -m mypy --strict src/isalgraph/` clean.

11. **Docs.** `src/isalgraph/competitors/README.md` in the idiom of `src/isalgraph/viz/README.md`,
    plus a row in `experiments/README.md`'s registry naming which competitor feeds which artifact.

---

## 9. Stop-and-ask conditions

Halt and escalate to the PI rather than proceeding. Bring a **diagnosed** problem with costed
options, and say what has already been ruled out.

1. **Any number in criterion 1 fails to reproduce.** The competitors folder is the evidence base for
   five plan-level findings that reach a printed number. A mismatch means either the shipped code
   differs from the scout's or the scout's environment mattered — and §0 shows the environments
   *were* different. Both readings change what the plan may assert.
2. **`pynauty` fails to build on Picasso.** Takes `nauty_graph6`, `sparse6_nauty` and AGM's orbit
   pruning down together, and changes `k` in `N_actual = 182 − 15k − 8d`. Do not re-open bliss /
   Traces (cut, decision S-g, and the counter-case expired) without the PI.
3. **`grakel` 0.1.10 does not reproduce `ours(h=2) = 5.830952`.** The E10 reconciliation becomes a
   version-pinning job, and every existing WL number is suspect.
4. **A backend needs a third-party dependency not already in the environment.** Vendor nothing —
   three gSpan/DFS-code repositories were tested and all three rejected.
5. **Any pressure to change the preregistered comparator sets, `N_max`, or the reduction rule.**
   Decision 23 is signed. Finding 6's missing case belongs to **T-02's owner**, not to T-04.
6. **A budget change that alters a published failure rate** — AGM's 200k/100k and min-DFS's 50,000
   are the values behind `24 % GREC`, `82 % AIDS-IAM` and `24/400 Mutagenicity`. Changing one
   silently rewrites a number the plan already prints.
7. **A second failed iteration round with any agent.**

---

## 10. Rejected alternatives

| Option | Rejected because |
|---|---|
| `encode() -> str`, distances on strings | four of the seven documented traps are "a string was measured with the wrong unit". A `str` return cannot prevent the 8× bit inflation or the 2× character/tuple Levenshtein gap. §2.2 |
| One `Backend` ABC with a raising `bits()` for WL | `wl-subtree-kernel` §7 explicitly warns that fabricating a WL bit count is the failure mode. A separate `VectorBackend` makes it unreachable rather than forbidden |
| Port `scratch/backends.py` as-is | its `padded_hamming(G, H)` builds both triangles from the **incident** labelling, so it cannot express `adjacency-matrix` §7's canonical-frame rule; and its subprocess bridge to the conda env is explicitly on the do-not-port list |
| Seven agents, one per competitor file | 2.3× the tokens, three waves instead of one, and seven-way contention on the reading order that adjacency, graph6 and AGM must all agree on |
| Agents ssh to Picasso themselves | violates the `review-ticket` non-negotiable; three credential holders can blow the `fscratch` inode quota or cancel jobs; overlapping runs make timings unattributable |
| A few-minute `sbatch` smoke | SCBI's 2-hour floor, in writing, to this account. loginexa is the sanctioned route and needs no scheduler |
| `size_null` as an analysis script (the folder's advice) | a column a later ticket can forget. §3.1 |
| Building a `GEDBackend` here | GEDLIB already lives in `eval_setup/ged_backends.py`; a second one forks the cost model |
| Vendoring `LasseRegin/gSpan` (decision 8) | **superseded by measurement**: it does not run on numpy ≥ 1.24 and its `G2DFS` is not the minimum code. Three repositories tested, three rejected. Vendor nothing; cite Yan & Han |

---

## 11. Effort

Board estimate **2–5 days** (was 3–8), P0. Wave 0 ≈ 1 day (orchestrator), wave 1 ≈ 1 day
(3 agents in parallel), wave 2 ≈ 0.5–1 day. The scout's finding 10 already cut the gSpan estimate
from 2–3 days to ~1 by doing the hard part; what remains is integration plus tests.

---

## Changelog

- **2026-08-14** — written, before any competitor code lands in `src/`. Three PI decisions taken at
  design time: the orchestrator owns Picasso and runs one loginexa session (§7); three agents in one
  wave grouped by shared trap rather than seven one-per-competitor (§6); `size_null` is a registered
  backend, against `competitors/README` §6's advice, and is hard-excluded from the confirmatory
  family (§3.1).
