# `isalgraph.competitors`

Eleven graph representations behind one pair of protocols, so the *Pattern
Recognition* revision can put IsalGraph beside the formats every reviewer
already has in mind. Serves **AE.4a** (the requirement-modal owner), AE.3,
R1.1, R1.2a/b and R3.6a.

Ticket **T-04**. Design note: `.claude/notes/review/tasks/T-04-design.md`.
Frozen interfaces: `.claude/notes/2026-08-14-t04-competitors/CONTRACTS.md`.

---

## Hello world

```python
from isalgraph.competitors import get_backend, get_metric

g6 = get_backend("graph6")
enc = g6.encode(my_graph)

enc.symbols          # ('E', 'l', 'C', 'W')  <- THE COMPARISON UNIT
enc.wire             # b'ElCW'               <- what was actually emitted
g6.bits(enc)         # BitCount(entropy_bits=24.0, realised_bits=32, payload_bits=15)

lev = get_metric("levenshtein")
lev.distance(enc, g6.encode(other_graph))
```

---

## The pool is not eleven arbitrary methods

**Family I — the `n²` serialisations.** All four emit *the same bit sequence*:
the strict upper triangle of the adjacency matrix, read **column-wise**. They
differ on exactly two orthogonal choices, which is what isolates canonicity as
a variable at fixed format.

| | raw bits | 6-bit ASCII packing |
|---|---|---|
| **incident labelling** | `adjacency` | `graph6` |
| **nauty canonical labelling** | — | `nauty_graph6` |
| **lex-min labelling** | `agm_cam` | — |

A test asserts that `adjacency.symbols`, graph6's unpacked payload and
`agm_cam` on the identity permutation agree bit for bit, so that argument is
**executable rather than prose**.

**Family II — the mining-literature canonical forms**, per Jiang, Coenen &
Zito (*Knowledge Engineering Review* 28(1):75–105, 2013): CAM (`agm_cam`) and
M-DFSC (`min_dfs`). R1.2 named one of each.

**Outliers.** `sparse6` is the only non-canonical format whose length scales
with `m`. `wl_subtree` is not a serialisation at all. `size_null` is not a
method.

---

## Public API

```
__init__.py     the two ABCs, get_backend/get_metric, available_*, value objects
base.py         ReprBackend · VectorBackend · DistanceMetric · Encoding ·
                BitCount · PositionalFrame · Capability · Budget
registry.py     two lazy registries; get_repr_backend / get_vector_backend narrow
bits.py         the ONLY module that produces a BitCount
fixtures.py     stdlib-only graphs + the relabeller every F3 test must use
datasets.py     real-cohort loading -- GRAPHS ONLY, never GED
ged_reference.py  certified exact GED -- imported by f5.py and reproduce.py alone
metrics/        levenshtein · levenshtein_char · hamming · padded_hamming ·
                kernel · size_null
backends/       adjacency graph6 sparse6 · nauty agm · min_dfs wl ·
                isalgraph_ref size_null
```

Four entry points, all dispatching through the registry, so adding a backend
extends them without editing them:

| Command | Does |
|---|---|
| `python -m isalgraph.competitors.smoke` | encode a real cohort, time it, record every failure |
| `python -m isalgraph.competitors.grid` | F1–F4 and F6. **Never F5** |
| `python -m isalgraph.competitors.f5` | F5 alone — reported, not an input to selection |
| `python -m isalgraph.competitors.reproduce` | the reproduction gate, `--mode artefacts\|table` |

---

## Why `encode()` does not return `str`

Four of the seven documented traps in `competitors/README.md` §6 are the same
mistake: **a string was measured with the wrong unit.** Counting `'1010…'` at
eight bits per character inflates the adjacency matrix **8×**. Running
Levenshtein on `'0-1 1-2 2-0'` charges four edits for one deleted DFS tuple, a
**2×** difference. Both are invisible and both produce a plausible number.

A `str` return type cannot prevent either, so `encode()` returns an `Encoding`
whose fields make the unit explicit:

- **`symbols` is the unit of edit.** One DFS tuple for min-DFS, one ASCII byte
  for graph6, one triangle bit for adjacency. Every edit distance runs on this
  and on nothing else.
- **`text` is never measured.** A test asserts nothing under `metrics/` or
  `bits.py` reads it, except `levenshtein_char`, which exists to *report* the
  character-level answer as a supplementary number.
- **`wire` is measured, never computed.** graph6's `1 + ⌈n(n−1)/12⌉` is right
  only for `n ≤ 62`, and Suite 2 reaches `n = 98`.
- **`frame` carries its own bits.** The edit unit and the positional unit
  differ for half the pool — graph6's symbols are six-bit *bytes* while its
  frame is the *bit* triangle underneath — so `padded_hamming` reads
  `frame.pairs` and `frame.bits` and never `symbols`.

`VectorBackend` is a separate protocol with **no `bits()` at all**. A
feature-vector "bit cost" would measure the choice of container rather than the
encoding, so the Claim A cell is empty with the reason printed, and fabricating
one is *unreachable* rather than merely forbidden.

---

## Contracts that matter

- **Import.** `import isalgraph.competitors` succeeds with `networkx`,
  `pynauty`, `grakel` and `rapidfuzz` all uninstalled; every third-party import
  lives in a function body or behind the lazy registry, and
  `isalgraph/__init__.py` does not import this package.
- **A missing dependency raises `BackendUnavailableError` on request.** Never a
  silent degrade, and never a bare `ImportError` from inside a method after
  `get_backend` has already said the backend was fine — `is_available()` is
  what decides, and *available means usable*.
- **A budget that runs out raises**, and one that **cannot be enforced** also
  raises. `timeout_s` is `cpp`-only; on the Python engine the reference arm
  refuses rather than running unbounded and letting the bit counts be quoted as
  if budgeted. Pass `Budget(timeout_s=None)` to opt out deliberately.
- **Every failure is recorded.** The failure *rate* is a reported number: AGM's
  24 % on GREC and min-DFS's 24/400 on Mutagenicity are results the paper
  prints. A stated ceiling is a result; a silent one is a defect.
- **F3 uses `fixtures.shuffled_copy`, never `nx.relabel_nodes(copy=True)`**,
  which preserves insertion order and makes order-dependent formats look
  invariant. A test asserts the relabeller *can* make `graph6` fail: a harness
  that cannot fail is worthless. `_f3` also reports a **skip count**, so a
  `0/50` can never be confused with a harness that never called the backend.

### F5-blindness is structural, not procedural

Decision 24 rests on T-04a's exclusion rule being F5-blind *by construction*:
ties break on cost, never on correlation with GED. Prose cannot enforce that,
so the import graph does.

```
grid.py  -> datasets.py   (graphs only)        F1 F2 F3 F4 F6
f5.py    -> ged_reference.py                   F5, reported, never an input
```

**A test asserts `grid.py`'s import closure reaches no GED loader**, and the
complement — that `f5.py`'s does. Adding a GED import anywhere in that closure
breaks decision 24.

### Suite scope

`agm_cam` and `isalgraph_canonical` carry `Capability.SUITE1_ONLY` and **raise
per graph** above it, so their ceilings stay measurable (criterion 5). A
separate guard, `base.table_scope_error`, refuses them a **printed row** on a
Suite-2 dataset (criterion 9): a column built from whichever graphs happened to
finish is conditioned on tractability and biased in the direction that flatters
the method. Measuring a ceiling and printing a column are different acts.

### `size_null` is registered, and hard-excluded from the family

`|n₁ − n₂|` *is* a distance on the representation "the node count", and it
scores **ρ = 0.71–0.93** against certified exact GED. Registering it makes the
null column fall out of every table **by construction**, so a printed ρ without
its null is unreachable rather than merely discouraged. Two guards:
`Capability.BASELINE` keeps it out of `available_backends()` and out of any
primary-distance selection, and it is **outside the frozen confirmatory
family** — it changes neither `N_max` nor `N_actual` (decision 23).

---

## Bit accounting

Both conventions from `competitors.md` §5, always together.

| Backend | `entropy_bits` | `realised_bits` |
|---|---|---|
| `adjacency`, `agm_cam` | `n(n−1)/2` | `8·⌈n(n−1)/16⌉` = `8·⌈T/8⌉` |
| `graph6`, `nauty_graph6` | `6·len(wire)` | `8·len(wire)` |
| `sparse6`, `sparse6_nauty` | `6·len(wire) − 6` *(`':'` excluded)* | `8·len(wire)` *(included)* |
| `min_dfs` | `m · 2⌈log₂ n⌉` | `8·len(text)` — **`inflated=True`** |
| `isalgraph_*` | `L · log₂ 9` | `8·L` |
| `wl_subtree`, `size_null` | **raises `BitCountUndefined`** | raises |

> `8·⌈n(n−1)/16⌉` is `8·⌈T/8⌉` — `T` bits packed into **bytes** — because
> `n(n−1) = 2T`. Reading the 16 as a word size halves every adjacency and AGM
> count. Both track A and track B found that independently.

---

## Testing

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pytest tests/unit/test_competitors_core.py -q          # the shared contracts
$PY -m pytest tests/unit/test_competitors_serial.py -q        # adjacency graph6 sparse6
$PY -m pytest tests/unit/test_competitors_canonical.py -q     # nauty sparse6_nauty agm
$PY -m pytest tests/unit/test_agm_cam.py -q                   # AGM brute force, slow
$PY -m pytest tests/unit/test_min_dfs.py -q                   # V1/V2/V3 + the kaviniitm gate
$PY -m pytest tests/unit/test_wl_subtree.py -q
```

383 tests. The oracles are the value, not the backends: AGM against the
lexicographic minimum over **all `n!` permutations** on 327 graphs; min-DFS
against exhaustive DFS enumeration, and its distinct codes at `n = 2…6` are
**1 / 2 / 6 / 21 / 112** — OEIS A001349, so no collisions.

---

## Reference

- Yan & Han, *gSpan*, **ICDM 2002**, 721–724, doi:10.1109/ICDM.2002.1184038.
- Shervashidze et al., *Weisfeiler-Lehman graph kernels*, **JMLR** 12:2539–2561, 2011.
- McKay & Piperno, *Practical graph isomorphism, II*, **J. Symb. Comput.** 60:94–112, 2014.
- Inokuchi, Washio & Motoda (AGM), **PKDD 2000**, 13–23.
- Jiang, Coenen & Zito, **Knowledge Engineering Review** 28(1):75–105, 2013.
