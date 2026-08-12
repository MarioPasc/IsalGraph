# task-export — work log

Wave `2026-08-12-exact-ged`. Base `29886f879a9190ad8c869eaf0979c7cf8af364ef` on `main`.

---

## Mission

Write `export_graphs.py`: serialize the five Suite-1 datasets into one file each, reproducing the
locked cohort counts exactly, so the cluster never reads the source trees.

Two constraints drive the design. Picasso's `fscratch` enforces a **file-count** quota, and the IAM
Letter GXL tree is 6,767 files. The GraphEdX graphs live in `torch` pickles, and `torch` is not
installed on Picasso and must never need to be. One `.npz` per dataset removes both: six files
totalling **53 KB** travel instead of ~6,800, and the read path imports only `numpy`, `networkx` and
the standard library.

`data.md` §6 lists `export_graphs.py` among the scripts that survived the loss of thirteen
measurement scripts. It did not exist. This is a new implementation, not a restoration, and it is
committed to the repository rather than left as scratch — untracked scratch is the documented failure
mode that lost the other thirteen.

---

## What I built

### `benchmarks/real_data/eval_setup/export_graphs.py` (new, ~760 lines)

Implements CONTRACT A verbatim: the eight-key `.npz` layout, the `metadata` JSON schema, and the
frozen `ExportedDataset` / `save_exported` / `load_exported` signatures.

| Component | Purpose |
|---|---|
| `DATASETS: dict[str, DatasetSpec]` | The locked cohort as data — key, loader family, expected graphs, expected pairs |
| `build_exported(spec, source_dir, commit)` | Load via the existing loader, filter, assert the cohort, package |
| `save_exported` / `load_exported` | CONTRACT A write / read, both running the full validator |
| `_graphs_to_csr` | Flattens to the CSR layout, edges sorted ascending within each graph |
| `_validate_arrays` | Every CONTRACT A invariant, applied on **both** write and read |
| `assert_cohort` | Raises `CohortMismatchError` with observed beside expected |
| `export_all` / `verify_exports` | The two CLI modes |
| `sha256_file` / `content_sha256` / `write_manifest` | `manifest.json` |

CLI, runnable as `python -m benchmarks.real_data.eval_setup.export_graphs`:

```
--source DIR   (default: the CONTRACTS §3 path)   --out DIR   (default: …/data/exported)
--datasets all|comma,separated                    --verify-only        --log-level
```

`--verify-only` re-reads the existing exports, recomputes counts and checksums, re-runs the CONTRACT
A validator, writes nothing, and exits non-zero on any failure. It reports **every** failure rather
than stopping at the first, because a partial transfer usually damages more than one file.

Reuse, as instructed — no loading or filtering was reimplemented. `dataset_filter.filter_graphs`,
`iam_letter_loader.load_iam_letter` and `graphedx_loader.load_graphedx_dataset` are called as they
stand. The `sys.path.insert` + bare-import pattern follows `validate_ged_bounds.py:34`. The two
source loaders are imported **inside** the functions that use them, so `load_exported` cannot pull
`torch` in even transitively if a loader later moves its import to module level.

### `tests/unit/test_export_graphs.py` (new, 40 tests)

Covers everything the brief listed, plus the corruption cases. Real-data tests are marked
`@pytest.mark.integration` and skip cleanly when the Sandisk tree is absent.

### `.claude/notes/2026-08-12-exact-ged/task-export.md`

This file.

---

## Measured counts

Observed on the real data at
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source`.
**Every assertion passed.** No filter was adjusted.

| key | raw | kept | expected | pairs | expected | drop: size / disconnected / trivial |
|---|---:|---:|---:|---:|---:|---|
| `iam_letter_low` | 2,250 | **1,180** | 1,180 | **695,610** | 695,610 | 0 / 1,069 / 1 |
| `iam_letter_med` | 2,250 | **1,253** | 1,253 | **784,378** | 784,378 | 0 / 993 / 4 |
| `iam_letter_high` | 2,250 | **2,059** | 2,059 | **2,118,711** | 2,118,711 | 0 / 183 / 8 |
| `linux` | 89 | **89** | 89 | **3,916** | 3,916 | 0 / 0 / 0 |
| `aids` | 911 | **769** | 769 | **295,296** | 295,296 | 51 / 91 / 0 |
| **Total** | | **5,350** | 5,350 | **3,897,911** | 3,897,911 | |

AIDS raw is **911**, confirming the brief. The 819 figure elsewhere is the connected count.

Every row satisfies `raw − size − disconnected − trivial = kept` and `pairs = C(kept, 2)`.

Written artifacts, `…/data/exported/`:

| file | bytes | graphs | edges |
|---|---:|---:|---:|
| `iam_letter_low.npz` | 9,369 | 1,180 | 3,618 |
| `iam_letter_med.npz` | 10,374 | 1,253 | 3,969 |
| `iam_letter_high.npz` | 17,940 | 2,059 | 9,381 |
| `linux.npz` | 3,299 | 89 | 743 |
| `aids.npz` | 13,305 | 769 | 8,226 |
| `manifest.json` | ~1.3 K | | |

**53 KB and six files, against 6,767 source files.** Full export: **2.0 s** wall clock.

---

## Decisions and why

**1. The split label is stored per graph.** Gate 0 needs within-split AIDS pairs and cannot recover
them once the splits are merged. IAM Letter's loader returns `splits` directly. GraphEdX's does not
— it appends in `SPLITS` order and records only sizes in an insertion-ordered dict. I reconstruct the
label from those sizes and then **cross-check it against the split encoded in each `graph_id`**
(`aids_train_0042`). A future reordering inside the loader therefore fails loudly here instead of
silently mislabelling gate 0's pairs. Verified on the real data: `{train, val, test}` with every id
prefix agreeing, asserted in `test_real_aids_retains_within_split_structure`.

**2. Defensive relabelling to `0..n-1` in sorted order.** Measured: all 5,350 kept graphs are
*already* labelled `0..n-1`, with no self-loops and none directed, so this is the identity today. It
is kept because it makes the CONTRACT A guarantee a property of the exporter rather than something
inherited from three loaders — see the `graphedx_loader` note under Follow-ups.

**3. Edges sorted ascending within each graph.** CONTRACT A does not require an ordering, but without
one the serialization depends on NetworkX's iteration order and the export stops being a function of
the data. `load_exported` does **not** require sorted input, so a peer's fabricated fixture is still
accepted.

**4. The validator runs on read as well as write.** A truncated or hand-edited `.npz` fails loudly
instead of yielding wrong graphs. This is the same class of defect as the GEDLIB accessor trap in
CONTRACTS §5 — silent garbage rather than an exception. `np.load(..., allow_pickle=False)` is set,
which also guarantees no pickled tensor can ride along.

**5. Duplicate-edge detection.** A duplicated column would silently collapse when the graph is
rebuilt, making `n_edges` disagree with reality. Detected with a vectorised `lexsort` over
`(owner, u, v)`.

**6. `content_sha256` added to `manifest.json` — the one extension, additive only.** Measured, not
assumed: `np.savez_compressed` stamps every zip member with local time, so two byte-identical exports
produce different file digests. I ran the exporter twice on the real data; **all five file `sha256`
differ between runs**, and `aids.npz` even changed size (13,307 → 13,305 bytes) because the timestamp
bytes compress differently. The contracted `sha256` therefore certifies transfer integrity only,
never build reproducibility. `content_sha256` digests the array contents alone and was **identical
across both runs for all five datasets**. `--verify-only` checks it only when present, so a manifest
written without the field still verifies. Reported to `main`; happy to drop it on request.

**7. Cohort failure is a hard stop.** `assert_cohort` raises before anything is written, and
`export_all` re-checks the five-dataset totals. `main()` returns 1 and logs
"The filter is locked. This is a finding to report, not a parameter to adjust."

---

## Assumptions

1. **GraphEdX split order.** `load_graphedx_dataset` records `split_sizes` in insertion order, which
   is `SPLITS = ("train", "val", "test")`. I rely on dict insertion order and then verify against the
   `graph_id` prefix, so the assumption is checked at runtime rather than trusted.
2. **GraphEdX has no class labels**, so `labels` is `""` throughout. IAM Letter carries its CXL
   `class` attribute (`A`, `E`, …). CONTRACT A permits `""` where the source has none.
3. **CLI defaults.** `--source` and `--out` default to the CONTRACTS §3 paths rather than being
   required, matching `validate_ged_bounds.py::DEFAULT_SOURCE`. Both are still explicit flags.
4. **`_normalise` sorts node ids**, so it requires them to be mutually comparable. True for all five
   datasets (all `int`). A non-comparable id would raise `TypeError` rather than be silently
   reordered, which is the behaviour I want.

---

## What I could not verify

1. **That the exports load correctly on Picasso.** I have no cluster access by standing prohibition.
   The read path uses only `numpy`, `networkx`, `json` and `hashlib`, and the no-torch test below
   proves the import graph is clean, but the round trip through `rsync` onto `fscratch` is unverified.
2. **That `content_sha256` is stable across machines.** It is stable across runs on this machine. It
   digests `int32`/`int64` arrays via `tobytes()`, which is little-endian here and on Picasso, but I
   did not test a big-endian host. Not a practical risk for x86-64.
3. **The 819 vs 911 AIDS discrepancy at its source.** I confirmed the `.pt` loader yields 911 raw and
   that 91 are disconnected, which is consistent with the brief's account of `graphs.json` being
   pre-filtered. I did not open `graphs.json` itself; it is not on my path.
4. **mypy is not clean** — 14 errors remain, all structural, none of them sloppiness:
   - 11 × `Missing type arguments for generic type "ndarray"`. CONTRACT A specifies
     `n_nodes: np.ndarray` verbatim. Parametrising to `npt.NDArray[Any]` would silence these but
     changes annotation text two peers are reading. I chose contract fidelity; trivially reversible.
   - 3 × `Cannot find implementation or library stub for module named "dataset_filter"` etc. mypy
     cannot follow `sys.path.insert`, and that pattern is mandated by CONTRACTS §9.

   For calibration, the baseline `ged_bounds.py` has **7** mypy errors of its own, so strict-clean is
   not the standard in `benchmarks/`. All four errors that were genuinely mine (`int()` on an
   `object` out of the `dict[str, object]` metadata) are fixed via a narrowing `_as_int` helper.

---

## Tests

```
$ ~/.conda/envs/isalgraph-cpp/bin/python -m pytest tests/unit/test_export_graphs.py -q
40 passed in 2.21s

$ ~/.conda/envs/isalgraph-cpp/bin/python -m pytest tests/unit/ -q
426 passed in 4.79s

$ ~/.conda/envs/isalgraph-cpp/bin/python -m pytest tests/unit/ -q --ignore=tests/unit/test_export_graphs.py
386 passed in 2.49s        # baseline: 386 + 40 = 426, no regression

$ ~/.conda/envs/isalgraph-cpp/bin/python -m ruff check \
    benchmarks/real_data/eval_setup/export_graphs.py tests/unit/test_export_graphs.py
All checks passed!
```

The integration tests **ran** rather than skipped — the Sandisk tree is present on this machine, and
0 tests were skipped. All five datasets are covered end to end in `tests/unit/`, which is affordable
because the whole export takes 2.0 s.

Coverage against the brief's checklist:

| Required | Test |
|---|---|
| round-trip identity, edge-for-edge, ids/splits/labels intact | `test_round_trip_reproduces_every_graph_edge_for_edge` |
| CSR `edge_offsets` invariants | `test_csr_offset_invariants` |
| `u < v` and `0 <= u,v < n` | `test_every_edge_is_ordered_and_in_range` |
| empty-edge and single-edge graphs | `test_single_edge_and_empty_edge_graphs`, `test_empty_dataset_round_trips` |
| `metadata` JSON round-trip, full §4 schema | `test_metadata_round_trips_with_every_contract_field` |
| count mismatch exits non-zero | `test_assert_cohort_raises_on_mismatch`, `test_main_exits_non_zero_on_cohort_mismatch` |
| `load_exported` with `torch` absent | `test_load_exported_works_without_torch` |

The no-torch test is a **subprocess** with a `MetaPathFinder` that raises on any `torch` import,
which proves the module *import graph* is clean, not merely that one function avoids it. It also
asserts `"torch" not in sys.modules` after the load completes.

Beyond the checklist: corrupt-file rejection (bad `edge_offsets[0]`, reversed `u > v`, `n_edges`
disagreeing with the offsets, out-of-range endpoint, duplicate edge), determinism of
`content_sha256`, tamper detection in `verify_exports`, idempotent re-save, and — on real data —
per-dataset cohort reproduction, the five-dataset total, and AIDS within-split structure.

Real-data acceptance run, twice, both clean:

```
$ python -m benchmarks.real_data.eval_setup.export_graphs --out …/data/exported
INFO __main__: Totals: 5350 graphs / 3897911 pairs -- match
$ python -m benchmarks.real_data.eval_setup.export_graphs --verify-only --out …/data/exported
iam_letter_low: OK … iam_letter_med: OK … iam_letter_high: OK … linux: OK … aids: OK
INFO __main__: Totals: 5350 graphs / 3897911 pairs -- match
EXIT=0
```

---

## Follow-ups for the orchestrator

1. **`content_sha256`** — accept or reject the additive manifest field (decision 6 above). Nothing
   else deviates from CONTRACT A.

2. **Stale path constants, confirmed as CONTRACTS §3 predicted, deliberately not fixed** (outside my
   ownership): `eval_setup.py:75 DEFAULT_SOURCE_DIR` and `eval_message_length.py:36
   DEFAULT_DATA_ROOT` both point at `…/research/isalgraph/data/…`, which no longer exists.

3. **Possible latent defect in `graphedx_loader.py::_strip_node_attributes`** — read-only to me,
   **not touched**. It does `clean.add_nodes_from(range(g.number_of_nodes()))` and then
   `clean.add_edge(u, v)` using the **original** labels. If a GraphEdX graph were labelled other than
   `0..n-1`, `add_edge` would silently create additional nodes and inflate the node count, which
   would then feed a wrong `n` into the size filter. I verified empirically that all 911 AIDS and 89
   LINUX graphs are already labelled `0..n-1`, so it is **inert today**, and my exporter relabels
   defensively regardless. Worth a ticket, not a hotfix.

4. **The manifest's `sha256` cannot certify reproducibility** (decision 6). If the response letter
   claims the export is reproducible, cite `content_sha256`, not `sha256`.

5. **Transfer list for Picasso**: the six files in
   `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/exported`, 53 KB total.
   `--verify-only` re-run cluster-side after the `rsync` will confirm both the counts and the bytes.
