# GEDLIB — the GED engine, its API and its silent traps

**Owner**: T-05 · **Serves**: every GED number in the revision
**Status**: LOCKED (decision 11). Verified working on Picasso 2026-08-11.

**One library, one cost model, one provenance chain.** This is what makes the numbers defensible to
R3.5b. Read this before writing any GED code; read [exact_ged](exact_ged.md) and
[approx_ged](approx_ged.md) for *what* gets computed with it.

---

## 1. Why GEDLIB and not our own implementation

GEDLIB is by Blumenthal and Gamper — **the authors of the BRANCH / BRANCH-FAST lower bound we cite**
(*IEEE TKDE* 30(3):503–516, 2018). Using the reference implementation is the difference between a
bound a reviewer accepts and one they audit. Our own plain BP measured **+78 % overestimate**, the
loosest member of its family.

| Repo | Last push | Verdict |
|---|---|---|
| `Ryurin/gedlibpy` | 2019-10-03 | dead — **do not use** |
| `dbblumenthal/gedlib` | 2023-06-22 | canonical C++ library |
| `jajupmochi/graphkit-learn` | 2025-06-07 | **maintained**; carries the Cython wrapper *and* its own gedlib fork |

**`pip install graphkit-learn` is not enough** — the PyPI wheel ships Python glue with no compiled
`.so` and no `.pyx`. The Cython sources exist only in the **git** repo.

---

## 2. Install on Picasso (login node, ~20 min)

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
libsvm, lsape and Eigen from the bundled `ext/` tree. No network beyond the two clones; no separate
Boost module.

**Do not also clone `dbblumenthal/gedlib` separately** — `setup.py` fetches its own copy, and a
second build costs ~92,000 files against the quota.

> ### fscratch quota is a FILE COUNT limit, not a space limit
> A GEDLIB build creates **50,000–90,000 small files**. Two builds hit the 400k hard limit, and the
> failure surfaces as `shutil.Error: [Errno 122] Disk quota exceeded` mid-`copytree`, not as a
> compile error. `quota -s` shows both quotas. Delete build trees once the `.so` exists.
> **T-23 owns clearing this and it blocks T-03.**

The build is **in-place**, so `PYTHONPATH` must point at the checkout, not at site-packages:

```bash
export PYTHONPATH=/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn
```

`libdoublefann.so: cannot open shared object file` means the wheel is installed but the C++ side was
never built — rerun `build_ext`.

---

## 3. Names that changed in the refactor

Most tutorials online use the old ones and fail.

| Old (broken) | Current |
|---|---|
| `librariesImport` | `libraries_import` |
| `gedlibpy` | **`gedlibpy_gxl`** (GXL input) / `gedlibpy_attr` (attribute input) |
| `GEDEnv` | **`GEDEnvGXL`** |

---

## 4. Working invocation

```python
import importlib
importlib.import_module("gklearn.gedlib.libraries_import")   # MUST come first
g = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")

env = g.GEDEnvGXL()
i0 = env.add_nx_graph(g0, "")     # node/edge attrs must be STRINGS
i1 = env.add_nx_graph(g1, "")
env.set_edit_cost("CONSTANT", edit_cost_constant=[1, 1, 0, 1, 1, 0])
#                  order: [node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]
env.init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")
env.set_method("BRANCH_FAST", ""); env.init_method()
env.run_method(i0, i1)
lb = env.get_lower_bound(i0, i1)   # valid for BRANCH*/STAR/ANCHOR_AWARE only
```

`add_nx_graph` requires string-valued node and edge attributes — attach a constant dummy label
before adding. Topology-only is what we want anyway.

### Trap 1 — import order

`libraries_import` `dlopen()`s libdoublefann/libsvm/libnomad and must load **before** `gedlibpy_gxl`.
**isort and ruff reorder plain `from … import` lines alphabetically and break this.** Use
`importlib.import_module`, which formatters cannot reorder.

### Trap 2 — the wrong accessor returns garbage, not an error

| Capability | Methods | Read |
|---|---|---|
| **Exact** | `ANCHOR_AWARE_GED` | both; `LB == UB` certifies optimality |
| **Lower bound** | `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT`, `STAR` | `get_lower_bound()` |
| **Upper bound** | `BIPARTITE`, `IPFP`, `REFINE`, `BP_BEAM` | `get_upper_bound()` |

Calling `get_lower_bound()` on an upper-bound method returns **0.00**; `HED` returns
`get_upper_bound() = inf`. **Neither raises.** A whole GED matrix fills silently with zeros.

> **Assert `0 < value < inf` on every read.** This is not optional defensive coding — it is the only
> thing standing between a misconfigured method and a published correlation computed against zeros.

---

## 5. Capability matrix — measured on Picasso 2026-08-11

Smoke test `scratchpad/gedlib_api.py`, P₄ (path) vs C₄ (cycle), unit costs, true GED = 1:

| Method | `get_lower_bound()` | `get_upper_bound()` | runtime | capability |
|---|---:|---:|---:|---|
| `ANCHOR_AWARE_GED` | **1.00** | **1.00** | 0.72 ms | **exact** |
| **`BRANCH_FAST`** | **1.00** | 1.00 | 0.20 ms | **LB** (+ incidental UB) |
| `BRANCH` | 1.00 | 1.00 | 0.19 ms | LB, tighter, costlier |
| `BRANCH_TIGHT` | 1.00 | 1.00 | 0.55 ms | LB, anytime |
| `STAR` | 1.00 | 1.00 | 0.09 ms | LB (Zeng et al. 2009) |
| `BIPARTITE` | **0.00** | **1.00** | 0.20 ms | **UB only** |
| **`IPFP`** | **0.00** | **1.00** | 0.33 ms | **UB only** |
| `REFINE` | 0.00 | 1.00 | 0.35 ms | UB only |
| `BP_BEAM` | 0.00 | 1.00 | 0.89 ms | UB only |
| `HED` | 0.00 | **inf** | 0.20 ms | **investigate before use** |

**21 methods available**: BRANCH, BRANCH_FAST, BRANCH_TIGHT, BRANCH_UNIFORM, BRANCH_COMPACT,
PARTITION, HYBRID, RING, ANCHOR_AWARE_GED, WALKS, IPFP, BIPARTITE, SUBGRAPH, NODE, RING_ML,
BIPARTITE_ML, REFINE, BP_BEAM, SIMULATED_ANNEALING, HED, STAR.

`HED` was earmarked as a *Pattern Recognition*-venue lower bound (Fischer et al. 2015) for EiC.b.
It returns `LB = 0, UB = inf` under default options — **unresolved, not usable until diagnosed**,
most likely it needs explicit method options. **Cite it in related work regardless**
([compliance](compliance.md)); report numbers from it only if the accessor issue resolves.

---

## 6. Cost model — one, and it is D6

**`CONSTANT` edit costs set to `[1, 1, 0, 1, 1, 0]`**: node insert = node delete = 1,
edge insert = edge delete = 1, **substitutions free**. Full justification in
[statistics](statistics.md) D6; the short form is that zero node cost makes GED a *pseudo*metric
while Corollary 2.13 asserts the IsalGraph distance **is** a metric, and validating a metric against
a pseudometric reference is incoherent.

**11 edit-cost models are available** — `CONSTANT` plus the published IAM per-dataset models
(`LETTER`, `LETTER2`, `GREC_1/2`, `CHEM_1/2`, `PROTEIN`, `FINGERPRINT`, `CMU`, `NON_SYMBOLIC`).
Those are available as a **sensitivity analysis** but must never be primary: per-dataset costs
reintroduce exactly the heterogeneity R3.5b objects to. They are cut candidate #2 in
[schedule](schedule.md) — cheap to run, expensive in pages.

> **One exception, and it must be written down.** The GraphEdX agreement gate runs under GraphEdX's
> own model — `[0, 0, 0, 1, 1, 0]`, zero node cost — **not** the production model. Running it under
> `[1,1,0,1,1,0]` produces a guaranteed mismatch that looks exactly like a solver bug. See
> [exact_ged](exact_ged.md) §4.

---

## 7. Cross-check that no longer exists

`.claude/CLAUDE.md` names `scratchpad/ged_bounds.py` as an independent BP + BRANCH-FAST
implementation and says "cross-check, do not skip". **It does not exist and never did** —
`find / -name 'ged_bounds.py'` returns nothing. This makes validation gate 2 unexecutable and leaves
the ρ(exact, LB) = 0.966 vs ρ(exact, UB) = 0.840 evidence unreproducible. Decision **S-e**;
options in [decisions](decisions.md).
