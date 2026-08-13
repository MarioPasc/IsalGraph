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
bound a reviewer accepts and one they audit. Our own plain BP measures **+135 % overestimate** on
LINUX (§7), the loosest member of its family — which is the case for `IPFP` as the reported upper
bound, and for keeping our BP as a cross-check rather than a source.

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
> A GEDLIB build creates **50,000–90,000 small files** — 12–22 % of the 400k hard limit. Two builds
> hit it, and the failure surfaces as `shutil.Error: [Errno 122] Disk quota exceeded`
> mid-`copytree`, not as a compile error. `quota -s` shows both quotas.
>
> **This build tree is the only part of the GED work that pressures the quota.** T-03 and T-05's
> combined output is **30 files** ([exact_ged](exact_ged.md) §5.1), so the answer is not to delete
> another project's data — it is to **prune the build tree once the shared objects exist**:
>
> ```bash
> cd $BUILD/graphkit-learn
> find . -type f | wc -l                       # before pruning
> find . -name '*.so*' -o -name '*.py'         # what must survive
> rm -rf gklearn/gedlib/include/gedlib-master  # headers, build-time only
> rm -rf gklearn/gedlib/ext                    # NOMAD/fann/libsvm/lsape/Eigen sources
> find . -type f | wc -l                       # after
> ```
>
> The runtime needs the `gklearn/` package and the `.so` files it `dlopen()`s, nothing else.
> **Verify with the counts above before and after, and re-run the §4 smoke test after pruning** —
> if it still imports and returns 1.00 on P₄ vs C₄, the prune was safe.
> **T-23 is this prune, not a quota cleanup, and it does not block T-03's output.**

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

> ## ⚠ CORRECTED 2026-08-13 — GraphEdX uses UNIT node costs. There is no exception.
>
> This section previously said the GraphEdX agreement gate must run under `[0, 0, 0, 1, 1, 0]`
> because GraphEdX charges zero for node operations. **That is wrong**, and T-03 measured it by
> recomputing AIDS pairs under both models and comparing to the published file:
>
> | pair | Δn | published | zero-node | unit-node |
> |---|---:|---:|---:|---:|
> | 241, 475 | 1 | 8.0 | 7.0 | **8.0** |
> | 207, 377 | 3 | 8.0 | 5.0 | **8.0** |
> | 135, 339 | 1 | 2.0 | 1.0 | **2.0** |
> | 211, 67 | 4 | 9.0 | 5.0 | **9.0** |
>
> **Unit-node 4/4, zero-node 0/4**, with the published value exceeding the zero-node value by
> exactly `|n₁ − n₂|` every time. GraphEdX's AIDS matrix uses **the same model as D6**, so the gate
> runs under `[1, 1, 0, 1, 1, 0]` like everything else.
>
> **What the old text cost.** T-03 configured gate 0 from this paragraph, measured "150 below, 58
> equal, 0 above", and concluded GraphEdX's matrix was an approximate upper bound. It was the
> arithmetic of the wrong cost model — each value low by exactly `Δn`. **That conclusion is
> retracted.** Measured like-for-like over the full 131,148-pair AIDS overlap: **0 pairs where ours
> exceeds theirs**, agreement on all but 2. See `.claude/notes/review/tasks/T-03-design.md`
> amendment 4.
>
> D6's own justification is untouched: it argues that *zero node cost in general* makes GED a
> pseudometric, not that GraphEdX shipped one.

---

## 7. The independent cross-check — written 2026-08-12

`.claude/CLAUDE.md` names `scratchpad/ged_bounds.py` as an independent BP + BRANCH-FAST
implementation and says "cross-check, do not skip". **That file never existed.** It has now been
written, and it lives in the repository rather than a scratchpad:

| Artifact | Path |
|---|---|
| BRANCH lower bound + Riesen–Bunke upper bound + exact A*, one cost model | `benchmarks/real_data/eval_setup/ged_bounds.py` |
| Gate runner, seeded sample, per-pair JSON for replay | `benchmarks/real_data/eval_setup/validate_ged_bounds.py` |
| Invariant tests | `tests/unit/test_ged_bounds.py` (35 passing) |

It reproduces this file's §5 smoke test exactly — P₄ vs C₄ → LB 1.00 / exact 1.00 / UB 1.00 — and
passed gate 2 with **0 bracket violations on 400 LINUX pairs**.

**Two things to carry into any GEDLIB work here:**

1. **Every upper-bound method is direction-dependent.** `BIPARTITE`, `IPFP`, `REFINE` and `BP_BEAM`
   construct an edit path from a *directed* assignment; swapping the two graphs can change the
   answer. Measured on our own implementation: differences of 12 vs 14 and 5 vs 7, tighter in one
   orientation on 33 % of pairs. **Fill both triangles and take the `min`, or assert symmetry and
   fail loudly** — otherwise the GED matrix is not symmetric and is not a distance matrix. The lower
   bound is unaffected.
2. **The published bound-quality figures do not reproduce.** ρ(exact, LB) measures **0.859**, not
   0.966; ρ(exact, UB) **0.522**, not 0.840; certification **1.5 %**, not 9.8–11.3 %. Re-derive per
   dataset. Full result: [exact_ged](exact_ged.md) §4.
