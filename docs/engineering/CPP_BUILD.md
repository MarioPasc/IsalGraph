# Building the IsalGraph C++ engine

The native engine is an optional accelerator. **Nothing requires it**: with no
`.so` present the package falls back to the pure-Python reference and the full
test suite passes (verified: 450 passed, 276 skipped with the extension
physically deleted).

---

## What gets built

| Item | Value |
|---|---|
| Extension target | `_native` |
| Import path | `isalgraph.core._native` |
| Sources | `src/isalgraph/core/native/` |
| Install destination | `<site-packages>/isalgraph/core/` |
| Build backend | scikit-build-core + nanobind |
| Standard | C++17, `CMAKE_CXX_EXTENSIONS OFF` |
| Default ISA | `x86-64-v3` (AVX2 + FMA) |

The source directory is spelled `native`, **not** `_native`. A sibling
directory named exactly `_native` would be a candidate implicit namespace
package next to the extension module of the same name. CPython's `FileFinder`
tries extension loaders before the namespace fallback so the `.so` would still
win, but the margin is one import-machinery detail and the names are free to
differ.

---

## Local build

```bash
conda activate isalgraph          # or your env of choice
export CMAKE_BUILD_PARALLEL_LEVEL=8
pip install -e ".[dev,native]"
```

Then **verify you are testing the code you think you are**:

```bash
python -c "import isalgraph, isalgraph.core._native as n; print(isalgraph.__file__); print(n.__file__)"
```

`isalgraph.__file__` must point at your checkout. `n.__file__` points at
site-packages and **that is correct**: under a scikit-build-core editable
install the Python sources are redirected to the checkout while the compiled
`.so` is installed normally. What must *not* appear in site-packages is any
`.py` from the package — if it does, you have a stale non-editable install
shadowing your work.

```bash
ls <site-packages>/isalgraph/core/     # expect ONLY _native.*.so
```

Do **not** set `PYTHONPATH=$PWD/src`. A `src`-first path shadows the installed
package, the extension silently fails to import, and every benchmark then
measures the Python fallback while appearing to succeed.

### Checking the build

```python
from isalgraph.core import backends
backends.build_info()
# {'engine': 'cpp', 'compiler': 'gcc 12.2.0', 'cplusplus': '201703',
#  'isa_level': 'x86-64-v3', 'avx2': '1', 'fma': '1', 'avx512f': '0',
#  'ndebug': '1', 'build_hash': '298fc1188bf1b051'}
```

`build_hash` is an FNV-1a of the flag string. It changes whenever the flags do
and is the cheapest way to detect a stale or wrong-ISA `.so` without running
the algorithm.

---

## Build options

| Option | Default | Purpose |
|---|---|---|
| `ISALGRAPH_NATIVE_MARCH` | `OFF` | Swap `-march=x86-64-v3` for `-march=native`. Local profiling only. |
| `ISALGRAPH_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan. Mutually exclusive with Release/LTO. |

```bash
# Sanitizer build (development only)
pip install -e ".[dev,native]" \
  --config-settings=cmake.define.ISALGRAPH_ENABLE_SANITIZERS=ON \
  --config-settings=cmake.define.CMAKE_BUILD_TYPE=Debug
```

**Never submit an `ISALGRAPH_NATIVE_MARCH=ON` build to SLURM.** See the
Picasso section.

---

## Selecting the engine at run time

| Mechanism | Precedence |
|---|---|
| `backend="cpp"` / `backend="python"` keyword | highest — always wins |
| `ISALGRAPH_ENGINE=cpp` / `=python` | middle |
| `backends.DEFAULT_BACKEND` | lowest — `cpp` if the `.so` imported |

An explicit `backend="cpp"` with no extension **raises `BackendError`**; it
never silently degrades. With no explicit request and no extension, the
default is `python` and everything still works.

The keyword-beats-environment ordering is load-bearing. IsalSR shipped a bug
where the environment variable was read before the keyword argument was
honoured, so `ISALSR_ENGINE=python` reported `"python"` from `engine()` while
still dispatching every call to C++. `test_explicit_backend_beats_env_var`
probes the dispatch, not the report.

---

## Picasso (UMA HPC)

### Toolchain

The login default is **gcc 7.5.0, which has no C++17**. Load a modern compiler
*at build time*:

```bash
module load gcc/13.2.0
```

The extension links `-static-libstdc++ -static-libgcc`, so **no `module load`
is needed at run time**. That is deliberate: it removes a module line from
every SLURM task and removes an entire class of "works on the login node,
dies in the job" failures.

### Compute nodes have no outbound internet

Pre-cache the build dependencies on the **login** node, then build on the
compute node with isolation disabled:

```bash
# --- login node (has internet) ---
module load gcc/13.2.0
conda activate isalgraph
pip download -d ~/wheelhouse "scikit-build-core>=0.9" "nanobind>=2.0" "ninja>=1.10" "cmake>=3.18"
pip install --no-index --find-links ~/wheelhouse \
    "scikit-build-core>=0.9" "nanobind>=2.0" ninja cmake

# --- compute node (inside the job) ---
module load gcc/13.2.0
conda activate isalgraph
cd $FSCRATCH/repos/IsalGraph
pip install -e ".[dev,native]" --no-build-isolation --no-index --find-links ~/wheelhouse
```

`--no-build-isolation` is mandatory on the compute node: with isolation pip
creates a fresh venv and tries to fetch the build requirements from PyPI,
which times out rather than failing fast.

### `-march`: why `native` is banned

178 of Picasso's 333 nodes lack AVX-512. A `-march=native` build compiled on
an AVX-512 login or compute node runs correctly on some nodes and dies with
**SIGILL on others**. Because SLURM allocates nodes arbitrarily, the symptom
is a random fraction of array tasks crashing — which reads like flaky hardware
rather than a build fault, and costs days to diagnose.

`x86-64-v3` is the lowest common denominator across all four node classes and
gives one `build_hash` for the whole campaign. Confirm on any node with:

```python
from isalgraph.core import backends
assert backends.build_info()["isa_level"] == "x86-64-v3"
assert backends.build_info()["avx512f"] == "0"
```

### The `.so` does not rsync

**This is the most common way to run a stale engine on the cluster.** The
extension installs into `site-packages`, not into the repository tree, so the
usual

```bash
rsync -av --delete ./ picasso:$FSCRATCH/repos/IsalGraph/
```

syncs the C++ *sources* and leaves whatever `.so` was last built on Picasso in
place. After any change under `src/isalgraph/core/native/`, **re-run the
`pip install -e` step on Picasso**. Verify by comparing `build_hash` and, if
the flags did not change, the `.so` mtime.

A cheap guard inside a job script:

```bash
python -c "
from isalgraph.core import backends
info = backends.build_info()
assert info['engine'] == 'cpp', 'C++ engine not active'
assert info['isa_level'] == 'x86-64-v3', info
print('engine ok:', info['build_hash'])
"
```

### Threads

`canonical_string(..., threads=N)` parallelises the starting-node loop.
**Leave it at the default of 1.** `hardware_concurrency()` is never consulted
because inside a SLURM cgroup it reports the whole physical node and silently
oversubscribes. Measurements in `CPP_OPTIMIZATION_LOG.md` §O6 show threading
is *slower* below ~7 nodes and reaches only ~34% parallel efficiency at 10, so
for the paper's workload it is a pessimisation.

---

## Running the tests

```bash
pytest tests/native/ -v          # differential + property suite (~2.5 min)
pytest tests/ -q                 # everything
ruff check src/ tests/
mypy src/isalgraph/
```

Native tests skip cleanly when the extension is absent. The skip guard is

```python
pytest.importorskip("isalgraph.core._native", reason="...", exc_type=ImportError)
```

at **module top level**, before the other imports. Both details matter:

* `@pytest.mark.parametrize` decorators are evaluated at collection time, so a
  `pytestmark = skipif` arrives too late to prevent an `ImportError`.
* `exc_type=ImportError` is required because pytest catches only
  `ModuleNotFoundError` by default, while a *deleted* `.so` under an editable
  install still resolves through the import redirect and raises a plain
  `ImportError: cannot open shared object file`.

### Benchmarks

```bash
python tests/native/bench_native.py speedup --out docs/engineering/results/speedup.json
python tests/native/bench_native.py ladder  --out docs/engineering/results/ladder.json
python tests/native/bench_native.py iam     --out docs/engineering/results/iam_surrogate.json
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ImportError: cannot open shared object file` | `.so` deleted or never built. `pip install -e ".[dev,native]"`. |
| Extension imports but is old | Stale `.so` in site-packages. Compare `build_hash`; rebuild. |
| SIGILL on some SLURM tasks only | `-march=native` build on a heterogeneous fleet. Rebuild with the default. |
| Benchmarks show ~1x speedup | `PYTHONPATH` shadowing, or `ISALGRAPH_ENGINE=python` leaked into the environment. Check `backends.build_info()["engine"]`. |
| GCC LTO/ODR error in `nb_types.h` | A global `add_compile_options()` leaked flags into `nanobind-static`. All flags must go through `target_compile_options(_native PRIVATE ...)`. |
| `pip install` fails on a compute node | Missing `--no-build-isolation`, or the wheelhouse was not pre-populated on the login node. |
