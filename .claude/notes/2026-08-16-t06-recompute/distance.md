# distance — work log

**Branch** `t06/distance` · **Base** `8afa59e` (merged `ticket/T-06` at `ccb0b48`) · **Head** see
`git log -1`

---

## What I built

`benchmarks/real_data/eval_distance/` is a new package that turns a CONTRACTS §3 encodings `.npz`
into CONTRACTS §4 distance matrices, sharded by contiguous **row bands**. The encodings file is the
only input — `node_counts` and `edge_counts` ride inside it, so this track never opens a cohort file
and the ownership partition against `t06-encoding` holds by construction. Three inner loops are
dispatched by metric: `rapidfuzz.process.cdist` for the edit and Hamming distances, a NumPy
broadcast for anything reading only the node count, and a per-pair `is_defined`/`distance` loop for
everything else. Two registered metrics are **refused** rather than approximated, because §3 does
not carry what they read. The symbol-vs-character rule of §3.1 is implemented as a resolution order
plus a hard cross-check against the file's `length` column, which is the guard that turns a wrong
separator from a plausible number into an error.

| File | Purpose | Lines |
|---|---|---|
| `benchmarks/real_data/eval_distance/__init__.py` | package docstring; why row bands, not pair-index slices | 29 |
| `benchmarks/real_data/eval_distance/schema.py` | §3 reader, §4/shard writers, §5 metadata with `isalgraph_build_hash` + `src_commit`, exception hierarchy | 498 |
| `benchmarks/real_data/eval_distance/bands.py` | `split_bands`, `band_for`, `verify_tiling` — no gap, no overlap | 124 |
| `benchmarks/real_data/eval_distance/gates.py` | the four structural properties + the campaign-level degeneracy guard | 187 |
| `benchmarks/real_data/eval_distance/distance_runner.py` | the CLI, symbol reconstruction, three compute paths, single-chunk dense emission | 669 |
| `benchmarks/real_data/eval_distance/distance_merge.py` | shard discovery, completeness refusal, tiling check, dense emission | 248 |
| `benchmarks/real_data/eval_distance/size_null.py` | `\|n_i - n_j\|`, same §4 schema as any distance file | 131 |
| `benchmarks/real_data/eval_distance/masks.py` | equal-`n` mask (computed, never stored), strict upper triangle, paired extraction | 89 |
| `benchmarks/real_data/eval_distance/throughput.py` | reproducible sizing harness; times the chosen loop against the two rejected ones | 190 |
| `benchmarks/eval_distance` | symlink to `real_data/eval_distance`, per the repo convention | — |
| `tests/unit/test_t06_distance.py` | 35 tests: the nine criteria, §3.1, the bands, the masks | 704 |

CLI surface:

```bash
python -m benchmarks.eval_distance.distance_runner \
    --encodings ENC.npz --metric levenshtein --out DIR \
    [--chunk-index K --n-chunks N] [--jobs 1] [--symbol-sep SEP] \
    [--suite suite2] [--on-length-mismatch raise|warn]
python -m benchmarks.eval_distance.distance_merge \
    --shard-dir DIR --basename {dataset}__{repr}__{metric} --out DIR [--expect-chunks N]
python -m benchmarks.eval_distance.size_null --encodings ENC.npz --out DIR
python -m benchmarks.eval_distance.throughput --encodings ENC.npz [--jobs 1]
```

---

## Acceptance criteria

`PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python`, run from the worktree root.

| # | Criterion | Command | Result (verbatim) |
|---|---|---|---|
| 1 | Schema conformance, dense keys | `$PY -m pytest tests/unit/test_t06_distance.py -q -k test_dense_file_carries_exactly_the_contract_keys` | `1 passed, 34 deselected in 0.16s` |
| 1 | §5 metadata on a **real** emitted file, incl. `isalgraph_build_hash` and `src_commit` | `… -k test_metadata_conformance_on_a_real_emitted_file` | `1 passed, 34 deselected in 0.21s` |
| 2 | Structural gate on real matrices (symmetric, zero diagonal, finite where defined, `>= 0`), 3 representations | `… -k test_structural_gate_holds_on_real_matrices` | `3 passed, 32 deselected in 0.21s` |
| 3 | Differential vs `levenshtein_computer.compute_levenshtein_matrix`, **element-wise integer equality**, 200 real graphs, 2 representations | `… -k test_levenshtein_matches_the_existing_oracle_exactly` | `2 passed, 33 deselected in 0.23s` |
| 4 | `--n-chunks 1` vs `--n-chunks 7` + merge, array-equal on 150 real graphs | `… -k test_seven_shards_merge_to_the_unsharded_matrix` | `1 passed, 34 deselected in 0.23s` |
| 5 | Merge **raises** on a missing shard and writes nothing | `… -k test_merge_raises_when_a_shard_is_missing` | `1 passed, 34 deselected in 0.16s` |
| 6 | `size_null` equals `abs(n_i - n_j)`, symmetric, zero diagonal, on real data | `… -k test_size_null_is_the_absolute_node_count_difference` | `1 passed, 34 deselected in 0.20s` |
| 7 | Undefined pairs are `nan` + `defined_mask False`, never `0.0` (padded-length Hamming on real graph6) | `… -k test_hamming_on_unequal_lengths_is_nan_and_unmasked` | `1 passed, 34 deselected in 0.21s` |
| 8 | End to end **through the CLI** on a 200-graph real slice: 3 shards → merge → size null | `… -k test_end_to_end_through_the_cli_on_real_data` | `1 passed, 34 deselected in 0.62s` |
| §3.1 | min-DFS symbol level ≠ character level, and matches a reference over the split lists | `… -k test_min_dfs_symbol_level_differs_from_character_level` | `1 passed, 34 deselected in 0.20s` |
| 9 | Own tests | `$PY -m pytest tests/unit/test_t06_distance.py -q` | `35 passed in 0.92s` |
| 9 | Whole unit suite | `$PY -m pytest tests/unit/ -q` | `1758 passed, 50 skipped, 1 warning in 361.30s (0:06:01)` |
| 9 | Lint, my files | `$PY -m ruff check benchmarks/real_data/eval_distance/ tests/unit/test_t06_distance.py` | `All checks passed!` |
| 9 | Type check | `$PY -m mypy src/isalgraph/` | `Success: no issues found in 69 source files` |

`$PY -m ruff check benchmarks/ tests/` reports **28 errors, all pre-existing and none in a file this
track owns** — `eval_visualizations/*` (E501, B007, SIM108, I001), `synthetic_data/*` (SIM113, F841),
`eval_setup/eval_setup.py` (E501). I did not touch them; they belong to earlier tickets.

**Criterion 8, as run.** `t06-encoding` had produced no encodings when this was written, so the test
**synthesises a conforming §3 file itself** from real cohort graphs: it rebuilds 200 `iam_letter_low`
graphs from the exported CSR, encodes them through `isalgraph.competitors`, and writes every §3 key
including `error_kind` and `metadata.symbol_sep`. Building that fixture is the only place the test
file touches a cohort; the driver under test never does. `isalgraph.engine() == "cpp"` and
`build_hash == 298fc1188bf1b051` are asserted through the emitted metadata.

---

## Throughput — sizing for the orchestrator, explicitly NOT a published timing

Three agents shared this workstation throughout, so every number below is contaminated by concurrent
load and is an order-of-magnitude input to the shard-count decision, nothing more. Reproduce with
`python -m benchmarks.eval_distance.throughput`.

200 real graphs, whole `200 x 200` ordered cell block, `--jobs 1`:

| Dataset / representation | symbol length min/med/max | **cdist (chosen)** | Python loop through `DistanceMetric` (rejected) | `isalgraph.core.backends.levenshtein` (rejected) |
|---|---|---|---|---|
| `iam_letter_low` / `isalgraph_pruned` | 1 / 4 / 10 | **55,524,092 cells/s** | 1,403,490 | 1,120,837 |
| `mutagenicity` / `graph6` | 3 / 60 / 599 | **4,136,967 cells/s** | 949,853 | 334,031 |
| `iam_letter_low` / `min_dfs` | 1 / 3 / 6 | **93,183,835 cells/s** | 2,913,060 | 1,571,650 |

**The number that decided it: 4.1 M vs 0.95 M cells/s on the longest strings in the pool** — a 4.4×
gap at the worst case and 40-60× at the best, on a code path that is otherwise identical. The
`DistanceMetric` loop is retained as the *generic* path for any future metric outside the cdist
table, so nothing was thrown away.

Worker scaling was also measured and is length-dependent: on `mutagenicity`/`graph6`,
`workers=-1` gave ~19.6 M cells/s against 6.1 M at `workers=1`, but on `iam_letter_low`'s
median-4 strings `workers=-1` was **slower** than `workers=1` (16.6 M vs 32.5 M) — thread spawn
dominates. Hence `--jobs` defaults to **1** and raising it is the orchestrator's call per dataset.

**What this implies for the full cohort.** Suite 2 is 21,710,892 unordered pairs, i.e. 43,421,784
ordered cells. At the *worst* measured rate that is **~11 s per (representation, metric)** for the
whole suite; the largest single matrix, `mutagenicity` at G = 4040, is 16.3 M cells ≈ 4 s. A
compressed `(G, G)` float64 + bool write was probed at G = 1000 → 1.2 MiB / 0.22 s and G = 2000 →
4.6 MiB / 0.81 s, so G = 4040 extrapolates to ~19 MiB / ~3 s. **The distance stage is neither CPU-
nor I/O-bound.** Shard it for resumability and wall-clock overlap with the encoding campaign, not
because it is expensive: 4-8 chunks per dataset is ample and 64 would be pure per-task overhead.

---

## Decisions I made, and why

1. **`rapidfuzz.process.cdist` over a per-pair loop.** Numbers above. `rapidfuzz` 3.14.5 accepts
   sequences of **multi-character** strings natively — verified directly, symbol-level 1 where the
   character-level answer is 4 — so the codepoint-remapping fallback the orchestrator sanctioned was
   **not needed and not written**. Nothing is remapped, so there is no injectivity assumption to
   defend.
2. **A band spans all `G` columns, not its strict-upper part.** Twice the cells, but work per band
   is then proportional to band height (so equal-height bands are equal-cost) and the merge is a
   concatenation that cannot mis-assemble a triangle. At 4-93 M cells/s the factor of two is
   seconds. Rejected: a strict-upper band, which needs a scatter on merge and makes a partial merge
   look like a matrix of zeros.
3. **The runner always writes a shard; `--n-chunks 1` additionally writes the dense file** through
   the same `merge_bands` the merge tool uses. One assembly path, so criterion 4 compares like with
   like. Rejected: a separate dense code path for the unsharded case, which is exactly the shape of
   a divergence nobody notices.
4. **Refuse `padded_hamming` and `kernel` instead of approximating them** — see defect 2.
5. **Refuse `levenshtein_char` when `symbol_sep != ""`.** The stored string is then the
   `\x1f`-joined rendering, not the `Encoding.text` the backend emitted, so a character-level
   distance over it would measure the separator. Allowed when `symbol_sep == ""`, where the two
   coincide.
6. **A row with `status == "error"` or `length < 0` is undefined off the diagonal**, `nan` with
   `defined_mask False`. Comparing a missing encoding against `''` returns the other string's
   length, which is a number rather than an error. The **diagonal stays `0.0` with the mask `True`**
   even for such a row: a graph is at distance 0 from itself whether or not it encoded, and §4
   requires a zero diagonal. An empty string is *not* by itself treated as a fault — a one-node
   graph legitimately encodes to zero symbols.
7. **The degeneracy guard is per matrix, never per pair.** `>= 0.99` of defined off-diagonal cells
   exactly 0 aborts; a per-pair `value > 0` rule would abort a correct run, exactly as the corrected
   `CLAUDE.md` note records for T-05's GED matrices.
8. **`alphabet_size` is reconstructed from `entropy_bits / length` rather than fabricated.** No
   metric reads it; it is set to 0 where not recoverable rather than to a plausible constant.
9. **`--jobs` defaults to 1.** Shared workstation, and the measurement above shows more threads are
   a pessimisation on short strings.

---

## Assumptions I recorded rather than blocking on

1. **`t06-encoding` had produced no encodings**, so every real-data test synthesises a conforming §3
   file from real cohort graphs and says so in its docstring. Recorded in the log rather than
   waiting; nothing about the driver depends on who wrote the file.
2. **`suite` is read from the input metadata, falling back to `--suite`, then to `"unknown"`.** §3
   does not state that an encodings file must carry it, but §5 requires it in my output.
3. **`{dataset}__{representation}` is recovered from the metadata first and the filename second.**
   If they ever disagree, metadata wins and the filename is only a fallback.
4. **`--on-length-mismatch warn` exists but defaults to `raise`** — the escape hatch for defect 1
   until the producer side is corrected. A file produced under it carries
   `symbol_length_matches_npz_length: false` in its own metadata, so the caveat travels with the
   data rather than living in a log.

All four, plus both defects and the sizing table, were sent to `main`.

---

## What I could NOT do, and why

1. **`padded_hamming` and `kernel` produce no matrix.** §3 carries a joined string, node counts and
   edge counts; it carries neither `Encoding.frame` nor a fitted WL feature multiset, and neither is
   derivable from a string without the cohort. If Claim B needs those columns, the work does not fit
   this track as specified. Flagged to `main`; no assumption made.
2. **No production campaign, per prohibition 4.** Nothing above 200 graphs was ever computed. The
   full-cohort numbers in this log are arithmetic on measured rates, not measurements.
3. **No SLURM script was written.** The measured cost makes a cluster submission unnecessary for
   this stage — the whole of Suite 2 is minutes of local CPU — so writing one would have been
   speculative. Say the word if the orchestrator wants one anyway.
4. **The generic per-pair path is currently unreachable through the registry**: every metric
   registered today is served by the cdist path, the vector path, or a refusal. It is covered by a
   test that registers a toy metric via `monkeypatch.setitem`, so it is exercised rather than merely
   present, but it has never run on a real metric.
5. **I could not verify my output against `t06-stats`' loader**, which did not exist. §4 conformance
   is asserted against the written file, not against a consumer.

---

## Contract defects found

**Defect 1 — §3.1's `symbol_sep == ""` branch is false for sparse6, and would abort its campaign.**
§3.1 says `encoding` is `Encoding.text` when `symbol_sep == ""`, and that `length == len(encoding)`
there. Measured on 50 real `iam_letter_low` graphs:

| backend | `len(text) - len(symbols)` |
|---|---|
| `sparse6` | **1 on every row** |
| `sparse6_nauty` | **1 on every row** |
| `graph6`, `nauty_graph6`, `adjacency`, `agm_cam`, `isalgraph_pruned` | 0 |
| `min_dfs` | 2-5 (expected; it uses the separator branch) |

`Encoding.text` carries sparse6's `':'` format marker; `Encoding.symbols` does not. So a producer
following §3.1 literally emits `len(encoding) == length + 1` on every sparse6 row, and my length
cross-check — the guard that catches a wrong separator — fires on row 0.

**Proposed fix, producer side:** store `"".join(Encoding.symbols)` when `symbol_sep == ""`, not
`Encoding.text`. That is what "`length` is always the symbol count" already implies, it makes both
branches uniform, and it changes **no distance**: a constant prefix present in both operands shifts
neither Levenshtein nor an equal-length test. Until then, `--on-length-mismatch warn`.

**Defect 2 — §3 cannot feed two registered metrics.** `padded_hamming` consumes
`Encoding.frame` and `kernel` consumes a fitted feature multiset. §3 carries neither. Both are
refused with `MetricUnsupportedError` rather than silently approximated. This is a scope question
for the orchestrator, not something a worker should decide.

**Defect 3 (minor) — §3 contradicts itself on the empty encoding.** The key table says `encoding` is
`''` when `status != "ok"`; the invariant list says `status == "censored"` ⟹ `encoding != ''`. Both
cannot hold. I key invalidity on `status == "error"` **or** `length < 0` and never on `encoding ==
''`, because an empty string is a legitimate encoding for a one-node graph.

**Confirmation, not a defect — §3.1 is right and the magnitude is larger than stated.** On 200 real
`iam_letter_low` graphs under `min_dfs`, **15,706 of 19,900 pairs (78.9 %)** differ between
symbol-level and character-level Levenshtein, with a mean ratio of **3.86×** and a maximum of
**4.00×**. `test_min_dfs_symbol_level_differs_from_character_level` is the regression guard.
