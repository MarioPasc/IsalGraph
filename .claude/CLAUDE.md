# CLAUDE.md -- IsalGraph

## Project Identity

**IsalGraph**: representation of graph structure as instruction strings.
Authors: Ezequiel Lopez-Rubio (supervisor), Mario Pascual Gonzalez (PhD student).
University of Malaga. Extends IsalChem (molecular graphs) to graphs with
unlabeled, indistinguishable nodes and no degree restrictions.

**Status: major revision at Pattern Recognition** (PR-D-26-03293), decision
received 2026-08-10, **revision due 2026-08-31**.
**The plan is `.claude/notes/review/plan/` — start at its `README.md`, or at
`tickets.md`, which names per ticket exactly which files to read.** Reviewer
notes, the decision letter and the verified-claims audit are the *inputs*, in
`.claude/notes/review/source/`. Read the relevant ones before changing anything
the manuscript reports.

> Mathematical foundation, architecture and adapter design:
> `src/isalgraph/core/README.md`.
> Which code produces which paper artifact: `experiments/README.md`.

### The Isal family

IsalGraph is the **first** of three sibling projects and therefore the least
mature. When a pattern is unclear here, check how the siblings solved it -- they
are later and better.

| Project | Code | Article | Venue |
|---|---|---|---|
| **IsalGraph** (this) | `/home/mpascual/research/code/IsalGraph` | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199` | Pattern Recognition -- major revision |
| **IsalSR** (DAGs, symbolic regression) | `/home/mpascual/research/code/IsalSR` | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a` | TPAMI -- major revision |
| **IsalHG** (hypergraphs) | `/home/mpascual/research/code/IsalHG` | `/media/mpascual/Sandisk2TB/research/ISAL/isalhg/article/preprint/6a31285eea907326ad1128f2` | preprint, not yet submitted |

Where to look in the siblings:

- **C++ engine**: `IsalSR/CMakeLists.txt` and `IsalSR/src/isalsr/core/native/`
  are the reference. `IsalSR/docs/engineering/CPP_BUILD.md` covers Picasso.
  `IsalHG/docs/engineering/CPP_OPTIMIZATION_LOG.md` is the optimisation ladder,
  including its negative results.
- **Visualization**: `IsalHG/src/isalhg/viz/` is the architecture `isalgraph.viz`
  is modelled on.
- **Revision workflow**: `IsalSR/.claude/notes/review/` is the structure
  `.claude/notes/review/` mirrors.

---

## Scientific Mindset

- Approach every task as a world-class deep learning scientist: think step by
  step, reason, and justify decisions with literature references and rigour.
- Do NOT please the user. If something won't work, has theoretical flaws, or is
  scientifically incorrect -- say so. We are doing serious research.
- Be proactive. If a task sparks a connection to another concept, report it.
- Prioritize correctness over speed. Every algorithm must be justified.
- **Verify before asserting.** Reviewers found real defects in this manuscript
  that the test suite could not catch. Check a claim against the code or the
  `.tex` source before repeating it.

---

## Environment

- **Conda env**: `isalgraph` -- pure Python, no compiled extension.
- **Conda env**: `isalgraph-cpp` -- the same packages **plus the built C++
  engine**. Use this one for anything that runs the encoder.
- Python 3.11.15. Toolchain: gcc 12.2.0, cmake 3.25.1, ninja, nanobind 2.14,
  scikit-build-core 1.0.3.

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pip install -e ".[dev,native]"     # builds the extension
```

| Command | Purpose |
|---|---|
| `$PY -m pytest tests/ -q` | Full suite (~2.5 min with the engine) |
| `$PY -m pytest tests/unit/ -q` | Unit tests |
| `$PY -m pytest tests/native/ -q` | C++ parity and differential suite |
| `$PY -m pytest tests/viz/ -q` | Visualization |
| `$PY -m pytest tests/property/ -q` | Hypothesis property tests |
| `$PY -m ruff check --fix src/ tests/` | Lint |
| `$PY -m mypy src/isalgraph/` | Type check (strict) |
| `$PY -m isalgraph.viz` | Regenerate `docs/figures/` |

**Reference state (measured 2026-08-25, T-09 close):** **2,583 passed / 321 skipped**
with the engine, in 9 min 18 s. A change that lowers either number needs an explanation.
T-09 added 33: `tests/viz/test_encoder_trace.py` (12) and `tests/viz/test_worked_example.py`
(21). `testpaths = ["tests"]`, so unrelated work under `benchmarks/` cannot move this figure.

> ⚠ **This figure was stale by ~3.5× until 2026-08-24.** It read *"726 passed /
> 271 skipped (integration of wave 2026-08-10)"* while the suite had grown across
> T-04, T-04a, T-05, T-27 and T-06 to 2,544. **A stale floor here is worse than
> no floor**: the instruction above tells the next agent that a drop needs
> explaining, so an agent trusting 726 would have accepted a suite that had
> silently lost ~1,800 tests. **Re-measure and update this line at every ticket
> close**, not when someone notices.
>
> The `.so`-removed figure (previously *"561 passed / 276 skipped"*) is
> **unverified since 2026-08-10** and deliberately not restated — it has not been
> re-measured, and inventing a number here would recreate the defect this note
> exists to record.

---

## Always use the C++ engine

`isalgraph.core._native` is a nanobind extension built from
`src/isalgraph/core/native/`. It is **byte-exact** against the Python reference
(3,079 graphs for each canonical variant, 11,000+ greedy encodings, zero
mismatches) and 23x-1025x faster depending on node count.

```python
import isalgraph
isalgraph.engine()          # 'cpp' when the extension is importable, else 'python'
isalgraph.build_info()      # compiler, ISA level, build_hash -- detects a stale .so
```

Rules:

- **Import the dispatching entry points**, i.e. `from isalgraph import
  canonical_string` or `from isalgraph.core.backends import ...`. These run on
  whichever engine is active.
- `from isalgraph.core.canonical import canonical_string` gets the **pure-Python
  reference** and bypasses the engine. Do that only in differential tests or
  when you specifically want the reference.
- Force an engine with `backend="cpp" | "python"` or the `ISALGRAPH_ENGINE`
  environment variable. The keyword always wins over the variable.
- `backend="cpp"` with no extension **raises `BackendError`**; it never
  degrades silently. Absent an explicit request, the default falls back to
  Python and the whole suite still passes.
- **Never `export PYTHONPATH=$REPO/src`** when the engine matters: a src-first
  path shadows the installed package and silently falls back to pure Python, so
  a benchmark measures nothing. Assert `isalgraph.engine() == "cpp"` in any
  script whose timings you intend to report.
- The `.so` installs into site-packages, so **it does not rsync**. Build it on
  the cluster as part of environment setup.
- Build flags are `-march=x86-64-v3`, never `-march=native`: Picasso is
  heterogeneous and `native` produces SIGILL on a fraction of nodes, which reads
  like flaky hardware rather than a build fault.

Threading exists but **defaults to 1, deliberately**. Measured on this
workload, 4 threads are 1.8x *slower* at n=6 and only 1.35x faster at n=10;
the paper's graphs average under 4 nodes.

### Coverage, and two things that are not native

Native: `string_to_graph` (S2G), `graph_to_string` (greedy G2S from one start),
`canonical_string`, `pruned_canonical_string`, `levenshtein`,
`compute_structural_triplets`, `Cdll`. `graph_distance` and
`pruned_graph_distance` are compositions of native halves.

1. **`GreedyMinG2S` is a hybrid** -- native per starting node, Python loop over
   `range(n)`, so n FFI crossings per encode. Measured, it retains 93-102% of
   the single-call speedup, so the marshalling is cheap relative to one greedy
   encode. Leaving the loop in Python is a deliberate non-optimisation.
2. **The class API is Python-only.** Only the free functions in
   `isalgraph.core.backends` dispatch. `StringToGraph`, `GraphToString` and the
   `run_with_trace()` methods `isalgraph.viz` calls always execute the frozen
   reference, whatever `engine()` reports. Intentional -- traces draw 6-8 node
   examples, and routing the reference through the engine would make the
   differential circular -- but tracing a large graph gets no speedup.

**Branch and bound (O5) is an addition, not a translation**: it has no
counterpart in the Python reference. Parity over 3,079 graphs says it is
output-preserving, but if a canonical string is ever suspected wrong, re-run
with `_native.set_branch_and_bound(False)` first; that isolates it in one step.
`set_pairs_memo(False)` does the same for the memoisation.

Full measurements and negative results: `docs/engineering/CPP_OPTIMIZATION_LOG.md`.

---

## GED computation on Picasso (GEDLIB)

The Pattern Recognition revision recomputes every GED itself under **one cost model**
(node ins/del = 1, edge ins/del = 1, substitutions free -- see
`.claude/notes/review/plan/statistics.md` D6). Exact GED comes from `networkx` below ~12 nodes;
above that we report a bracket from **GEDLIB**.

### Why GEDLIB and not our own implementation

GEDLIB is by Blumenthal and Gamper -- the authors of the BRANCH/BRANCH-FAST lower bound we cite
(*IEEE TKDE* 30(3):503-516, 2018). Using the reference implementation is what makes the bounds
defensible to a reviewer. Repo status, checked 2026-08-11:

| Repo | Last push | Verdict |
|---|---|---|
| `Ryurin/gedlibpy` | **2019-10-03** | dead -- **do not use** |
| `dbblumenthal/gedlib` | 2023-06-22 | canonical C++ library |
| `jajupmochi/graphkit-learn` | **2025-06-07** | **maintained**; carries the Cython wrapper *and* its own gedlib fork |

**`pip install graphkit-learn` is not enough** -- the PyPI wheel ships Python glue with no compiled
`.so` and no `.pyx`. The Cython sources exist only in the **git** repo.

### Install (login node, ~20 min)

```bash
CE=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph
module load gcc/12.2.0 cmake/3.31.4
$CE/bin/python -m pip install cython numpy scipy networkx

cd /mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib
git clone --depth 1 https://github.com/jajupmochi/graphkit-learn.git
cd graphkit-learn/gklearn/gedlib
$CE/bin/python setup.py build_ext --inplace     # fetches its own gedlib fork and builds it
```

`setup.py` downloads `jajupmochi/gedlib` into `include/gedlib-master/` and builds NOMAD, fann,
libsvm, lsape and Eigen from the bundled `ext/` tree. **No network access is needed beyond the two
clones**, and no separate Boost module is required.

**Do not also clone `dbblumenthal/gedlib` separately.** `setup.py` fetches its own copy; a manual
second build is redundant and costs ~92,000 files against the quota (see below).

### Verify the install (verified working 2026-08-11)

The build is **in-place**, so `PYTHONPATH` must point at the checkout, not at site-packages:

```bash
export PYTHONPATH=/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn
$CE/bin/python - <<'EOF'
from gklearn.gedlib import libraries_import          # note: snake_case, loads the .so files
from gklearn.gedlib import gedlibpy_gxl as g         # GXL bindings -- our data is GXL
print(g.list_of_method_options)
env = g.GEDEnvGXL()                                  # class is GEDEnvGXL, not GEDEnv
EOF
```

**Names that changed in the refactor** -- most stale tutorials online use the old ones:

| Old (broken) | Current |
|---|---|
| `librariesImport` | `libraries_import` |
| `gedlibpy` | **`gedlibpy_gxl`** (GXL input) / `gedlibpy_attr` (attribute input) |
| `GEDEnv` | **`GEDEnvGXL`** |

**Verified method list** (21): `BRANCH`, **`BRANCH_FAST`**, `BRANCH_TIGHT`, `BRANCH_UNIFORM`,
`BRANCH_COMPACT`, `PARTITION`, `HYBRID`, `RING`, **`ANCHOR_AWARE_GED`**, `WALKS`, **`IPFP`**,
`BIPARTITE`, `SUBGRAPH`, `NODE`, `RING_ML`, `BIPARTITE_ML`, **`REFINE`**, `BP_BEAM`,
`SIMULATED_ANNEALING`, `HED`, `STAR`.

**Verified edit-cost list** (11): `CONSTANT` (our unit model -- see `statistics.md` D6), plus the
published IAM per-dataset models `LETTER`, `LETTER2`, `GREC_1`, `GREC_2`, `CHEM_1`, `CHEM_2`,
`PROTEIN`, `FINGERPRINT`, `CMU`, `NON_SYMBOLIC`. We use **`CONSTANT`**; the per-dataset models are
available as a sensitivity analysis but reintroduce exactly the heterogeneity R3.5b objects to.

Roles for this project: **`BRANCH_FAST`** = lower bound, **`IPFP`** / **`REFINE`** = tight upper
bounds, `BIPARTITE` = the loose Riesen-Bunke reference point, `BRANCH_TIGHT` = anytime lower bound,
**`ANCHOR_AWARE_GED`** = exact -- benchmark it against `networkx` A*, it may push the exact-GED
ceiling past n = 12.

### Working invocation, and two traps that fail silently

```python
import importlib
importlib.import_module("gklearn.gedlib.libraries_import")   # MUST come first
g = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")

env = g.GEDEnvGXL()
i0 = env.add_nx_graph(g0, "")     # node/edge attrs must be STRINGS
i1 = env.add_nx_graph(g1, "")
env.set_edit_cost("CONSTANT", edit_cost_constant=[1, 1, 0, 1, 1, 0])
#                  [node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]
env.init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")
env.set_method("BRANCH_FAST", ""); env.init_method()
env.run_method(i0, i1)
lb = env.get_lower_bound(i0, i1)
```

**Trap 1 -- import order.** `libraries_import` `dlopen()`s libdoublefann/libsvm/libnomad and must
load *before* `gedlibpy_gxl`, or you get `ImportError: libdoublefann.so.2: cannot open shared object
file`. **isort/ruff will reorder plain `from ... import` lines alphabetically and break this** --
use `importlib.import_module`, which formatters cannot reorder.

**Trap 2 -- wrong accessor returns garbage, not an error.** Methods differ in what they can produce:

| Capability | Methods | Read |
|---|---|---|
| **Exact** | `ANCHOR_AWARE_GED` | both; `LB == UB` certifies optimality |
| **Lower bound** | `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT`, `STAR` | `get_lower_bound()` |
| **Upper bound** | `BIPARTITE`, `IPFP`, `REFINE`, `BP_BEAM` | `get_upper_bound()` |

Calling `get_lower_bound()` on an upper-bound method returns **0.00**; `HED` returns
`get_upper_bound() = inf`. Neither raises.
~~**Assert `0 < value < inf` on every read** -- otherwise a whole GED matrix silently fills with
zeros.~~

> ## ⚠ CORRECTED 2026-08-15 (T-05) -- `0 < value < inf` per pair is WRONG and will abort a correct
> run. The failure it guards against is real; the guard is at the wrong level.
>
> **GED is legitimately 0 for isomorphic graphs.** Measured on Suite 2: **28.05 %** of IAM Letter LOW
> pairs are certified with `LB == UB`, and **1.01 %** of the whole 21,710,892-pair cohort has
> `UB == 0`. Suite 1 alone holds 306,768 certified off-diagonal pairs with exact GED = 0. A blanket
> per-pair `value > 0` assertion fires on all of them.
>
> **The correct guards, which T-05 ran on all 21.7 M pairs with zero false aborts:**
>
> - **Per read** -- reject non-finite; reject `< 0`. Reject `== 0` **only when the pair cannot
>   attainably have distance 0** (`ged_backends.py:402 zero_distance_is_attainable`).
> - **Per campaign, at init** -- an accessor probe on P4 vs C4 (true GED 1) asserting the method
>   returns 1.00 *through the accessor it is being read with*. This is the check that actually
>   catches a wrong accessor, and it is cheap.
> - **Per merged matrix** -- record the off-diagonal exact-zero fraction and abort if it is
>   **>= 0.99**. That is the shape of the silent-zero failure; a per-pair rule is not.

Verified on Picasso 2026-08-11 with P4 vs C4 (true GED = 1): `ANCHOR_AWARE_GED` 1.00/1.00,
`BRANCH_FAST` LB 1.00 (0.20 ms), `IPFP` UB 1.00 (0.33 ms), `BIPARTITE` UB 1.00, `STAR` LB 1.00.
`HED` returned LB 0.00 / UB inf under default options -- **unresolved, do not use yet**.

A failure of the form `libdoublefann.so: cannot open shared object file` means the wheel is
installed but the C++ side was never built -- rerun `build_ext`.

**Cross-check, do not skip**: `scratchpad/ged_bounds.py` in the revision notes implements BP and
BRANCH-FAST directly. GEDLIB and it must agree on the same pairs; disagreement is a bug in one of
them and we need to know which.

### fscratch quota: it is a FILE COUNT limit, not a space limit

```
fscratch  0.47TB / 1.40TB space   <- fine
          227.2k / 250.0k files   <- under the soft limit, hard limit 400.0k
```

> **Re-measured 2026-08-25.** This block read ~~`399.7k / 250.0k files <- EXCEEDED`~~ and is no
> longer true: the GEDLIB build trees have been pruned and there are now ~23k files of headroom
> under the soft limit and ~173k under the hard one. **The mechanism below is unchanged and still
> the thing to watch** — the limit is a file *count*, one GEDLIB build is 50-90k files, and two
> would still exceed it. Re-run `quota -s` before any build; do not trust either number here.

A GEDLIB build creates **50,000-90,000 small files** (headers, objects). Two builds will hit the
hard limit and the failure surfaces as a confusing `shutil.Error: [Errno 122] Disk quota exceeded`
mid-`copytree`, not as a compile error. Check with:

```bash
quota -s                    # the fancy banner shows both space and file quotas
find <dir> -type f | wc -l  # per-directory file count
```

Delete build trees once the `.so` is produced, and prefer `--depth 1` clones.

## Visualization: `isalgraph.viz`

All figures go through this package. **Do not hand-roll matplotlib in a figure
script.** Full API in `src/isalgraph/viz/README.md`.

```
src/isalgraph/core/trace.py     StepSnapshot / AlgorithmTrace -- stdlib only
src/isalgraph/viz/
  base.py                       GraphVizBackend ABC + Position
  style.py                      palettes, IEEE sizes, rcParams, save_figure
  layout.py                     cdll_ring_positions, compact_graph_layout
  registry.py                   backend plugin registry with is_available()
  instruction_view.py           the instruction strip
  cdll_view.py                  the CDLL ring with pointer arrows
  graph_view.py                 backend dispatcher
  composite.py                  multi-panel step figures
  search_tree.py                canonical search-space schematic (Reviewer 3)
  backends/matplotlib_backend.py   DEFAULT -- no third-party drawing library
  backends/{networkx,igraph}_backend.py   optional
```

Contracts that matter:

- **Every third-party import lives inside a function body.** `import
  isalgraph.viz` must succeed with matplotlib uninstalled; a test enforces it.
- **A backend never creates a figure.** It paints on a caller-supplied `Axes`
  and returns the layout it used, which is how positions stay pinned across the
  columns of a step figure.
- Traces carry **graph node ids, already resolved from CDLL indices**. A view
  must never call `cdll.get_value`.
- G2S traces are produced by **replaying the emitted string**, not by
  instrumenting the encoder. A g2s step figure shows an interpreter executing a
  finished string -- it does not show tentative pointer walks, rejected
  displacement pairs, or the priority cascade. For the decision structure use
  `search_tree.canonical_search_tree_figure`.
- `benchmarks/plotting_styles.py` re-exports from `isalgraph.viz.style` so the
  published palette cannot drift. Its values are byte-identical to what is in
  the submitted PDF, and a test asserts it.

---

## Architecture Overview

### Instruction set (alphabet Sigma = {N,n,P,p,V,v,C,c,W})

| Instr | Semantics |
|-------|-----------|
| `N/P` | Move primary pointer next/prev in CDLL |
| `n/p` | Move secondary pointer next/prev in CDLL |
| `V`   | New node + edge from primary's graph node, insert into CDLL after primary |
| `v`   | New node + edge from secondary's graph node, insert into CDLL after secondary |
| `C`   | Edge from primary's graph node to secondary's graph node |
| `c`   | Edge from secondary's graph node to primary's graph node |
| `W`   | No-op |

### Dependency layering

```
experiments/ benchmarks/  -> anything (torch, matplotlib, ...)
isalgraph.viz             -> optional: matplotlib, networkx, igraph
isalgraph.adapters        -> optional: networkx, igraph, pyg
isalgraph.core            -> stdlib only (+ the optional C++ engine)
isalgraph.core.native     -> C++17, nanobind, no Python beyond the bindings
```

### Repository layout

```
src/isalgraph/core/         Reference implementation + native engine + dispatch
src/isalgraph/viz/          Figure toolkit
src/isalgraph/adapters/     NetworkX, igraph, PyG bridges
experiments/                ORCHESTRATION: what runs, where, with what resources
  paper_pipeline/             Real-data paper pipeline (steps 1-4). CANONICAL.
  synthetic_suite/            Synthetic validation. NOT in the paper.
benchmarks/                 ROUTINES: the Python that does the science
  real_data/ synthetic_data/  invoked via the benchmarks/<name> symlinks
docs/original_code_and_files/  Advisor's original code, READ-ONLY
.claude/notes/review/plan/    THE PLAN -- one file per edge; start at README.md
.claude/notes/review/source/  INPUTS -- decision letter, reviewer notes, audits
```

`benchmarks/<name>` are **symlinks** into `real_data/` or `synthetic_data/`;
every worker invokes `python -m benchmarks.<name>.<name>` through them. Do not
delete them or convert them to real directories.

---

## Critical Invariants

Violating these causes silent corruption -- wrong graphs, no error raised.

1. **CDLL indices are not graph node indices.** Pointers are CDLL indices; the
   graph node is `cdll.get_value(ptr)`. They coincide only while no CDLL node is
   removed, which is not guaranteed.
2. **`insert_after(cdll_index, graph_node_payload)`** -- first arg CDLL index,
   second arg payload.
3. **`SparseGraph.add_edge(source, target)`** -- both are graph node indices.
4. **Pointer immobility on V/v.** The pointer does not advance after insertion.
5. **Displacement pairs sort by `(|a|+|b|, |a|, (a,b))`** -- all three key
   components. Sorting by `a+b` is historical bug B2.
6. **The canonical string does not encode directedness.** The witness is exact
   and needs no enumeration: a single undirected edge and a single directed arc
   both canonicalise to `"V"`. So `S2G` needs the `directed` flag as separate
   metadata, and any deduplication over a mixed corpus must key on
   `(directed, string)`. The complete-invariant theorem holds **within** a
   fixed directedness class, not across.

   Do not quote a collision *rate* without stating its enumeration window --
   the ratio moves with the window and is not a property of the encoding.
   Over labeled edge sets with `n <= 4` and `<= 4` edges: 63 of 441 encode to
   the same string under both semantics. Over distinct canonical strings in
   that same window: 6 of the 7 undirected classes are also produced by some
   directed graph. Those are two different measurements of one phenomenon.
7. **Greedy G2S depends on set iteration order.** `_find_new_neighbor` returns
   the *first* neighbour from a Python `set`, whose order for small ints is slot
   order, not ascending. The C++ engine achieves byte-parity only because
   adjacency crosses the FFI in Python's own iteration order. Any change to how
   adjacency is marshalled breaks greedy parity.

---

## Code Organization Rules

- `isalgraph.core`: stdlib only. The C++ engine is optional and must degrade.
- Full type annotations on all signatures; Google-style docstrings.
- `__slots__` on performance-critical structures (CDLL, SparseGraph).
- No `print()` for diagnostics -- use `logging` or raise.
- All of `src/isalgraph/` must pass `ruff check` and `mypy --strict`.
- Conventional commits. No `Co-authored-by` trailers.
- **The Python reference in `core/{canonical,canonical_pruned,cdll,sparse_graph,
  string_to_graph,graph_to_string}.py` is frozen.** It is what the differential
  suite compares the C++ engine against. Changing it means re-proving parity.

---

## Mathematical Foundation (brief)

**Round-trip**: for any valid string w, `S2G(w)` is isomorphic to
`S2G(G2S(S2G(w), v0))`.

**Canonical string**: `w*_G = lexmin{ w in argmin_{v in V} |G2S(G, v)| }`, a
complete graph invariant within a directedness class: `w*_G = w*_H` iff `G ~ H`.

**Graph distance**: `Levenshtein(w*_G, w*_H)` approximates graph edit distance.
Correlation is density-dependent -- strong on sparse IAM (rho ~ 0.93), weak to
moderate on LINUX (~0.43) and AIDS (~0.35).

Full details: `src/isalgraph/core/README.md`.

---

## Key References

- Lopez-Rubio (2025). arXiv:2512.10429v2. `docs/references/2512_10429v2.pdf`
- Design notes: `docs/references/Idea.pdf`
- You et al. (2018). GraphRNN. ICML.
- Fey & Lenssen (2019). PyTorch Geometric. ICLR Workshop.

## Detailed Specifications

- @src/isalgraph/core/README.md -- full mathematics and architecture
- @experiments/README.md -- experiment-to-paper-artifact registry
- @src/isalgraph/viz/README.md -- visualization API
- @docs/DEVELOPMENT.md -- development workflow
- Save large outputs in `/media/mpascual/Sandisk2TB/research/isalgraph`
