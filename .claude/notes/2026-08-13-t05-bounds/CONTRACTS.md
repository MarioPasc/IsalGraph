# CONTRACTS — wave `2026-08-13-t05-bounds`

**Frozen by the orchestrator before any agent started. Agents code against this file, never against
each other.** If you believe a contract is wrong, **message the orchestrator** — do not negotiate
with a peer and do not "fix" it locally. A contract defect found early is a success; two agents
silently diverging is not.

Base commit: `34e3ade822ce82424b6fb4d12045b678d56ad798`.
Design note: `.claude/notes/review/tasks/T-05-design.md` — read §1, §2, §3 before writing code.

---

## 1. Dataset keys — the ten Suite-2 datasets

Canonical, lowercase, used as `.npz` basenames and as dict keys everywhere.

| Key | Source root | Subpath | graphs | pairs |
|---|---|---|---:|---:|
| `iam_letter_low` | `$IAM_ROOT` | `Letter/LOW` | 1,180 | 695,610 |
| `iam_letter_med` | `$IAM_ROOT` | `Letter/MED` | 1,253 | 784,378 |
| `iam_letter_high` | `$IAM_ROOT` | `Letter/HIGH` | 2,059 | 2,118,711 |
| `linux` | `$GRAPHEDX_ROOT` | — | 89 | 3,916 |
| `aids_graphedx` | `$GRAPHEDX_ROOT` | — | **819** | 334,971 |
| `grec` | `$IAM_ROOT` | `GREC/data` | 650 | 210,925 |
| `aids_iam` | `$IAM_ROOT` | `AIDS/data` | 1,811 | 1,638,955 |
| `coil_del` | `$IAM_ROOT` | `COIL-DEL/data` | **3,900** | 7,603,050 |
| `mutagenicity` | `$IAM_ROOT` | `Mutagenicity/data` | 4,040 | 8,158,780 |
| `protein` | `$IAM_ROOT` | `Protein/data` | 569 | 161,596 |
| | | **Total** | **16,370** | **21,710,892** |

```
IAM_ROOT      = $SANDISK/data/source/APPROX_GED/datasets/IAM_Database/extracted
GRAPHEDX_ROOT = $SANDISK/data/source/GED_PRECOMPUTED
SANDISK       = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph
```

**Two roots, not one.** The plan and T-27's §9 repro line both say
`$SANDISK/data/source/IAM_Database/extracted`; **that path does not exist**. LINUX and AIDS-GraphEdX
are not IAM datasets and live under `GED_PRECOMPUTED`.

**Filter**: `min_nodes = 2`, `require_connected = True`, **no `n_max`**, splits merged.
IAM datasets are enumerated by their **`.cxl` split index, not by the directory** (decision 27) —
this binds `coil_del`, where 7,200 `.gxl` files ship and the index defines 3,900.

**`aids_graphedx` (819) is a different cohort from Suite 1's `aids` (769)** and must never overwrite
or be confused with it.

---

## 2. Suite-2 graph export — `exported_suite2/{key}.npz`

Schema **identical** to `export_graphs.py:310 save_exported`. Do not invent a new one.

| Key | dtype | shape |
|---|---|---|
| `graph_ids` | `<U` | (N,) — **the source loader's native id**, byte-identical to what `export_graphs.py` writes for Suite 1 (amendment 1) |
| `n_nodes` | int32 | (N,) |
| `n_edges` | int32 | (N,) |
| `edge_offsets` | int64 | (N+1,) — CSR row pointers |
| `edges` | int32 | (2, M) — CSR, sorted by `(u, v)`, node ids **local to each graph, 0-based** |
| `splits` | `<U` | (N,) |
| `labels` | `<U` | (N,) — **the loader's class label where present**, `''` otherwise. No class count is asserted (amendment 2) |
| `metadata` | `<U` | () — JSON |

`metadata` JSON keys: `dataset, source, n_raw, n_kept, n_dropped_min_nodes, n_dropped_disconnected,
n_pairs, filter{min_nodes, require_connected, n_max}, splits_merged, enumeration, exported_utc,
code_commit, schema_version`. `n_max` is `null` for Suite 2; `enumeration` is `"split_index"`.

`load_exported` (`export_graphs.py:324`) must read these files unchanged (`allow_pickle=False`).

**Graph order within a file is the export order and is load-bearing** — every pair index downstream
is `numpy.triu_indices(N, k=1)` over it. It must be deterministic: splits in the order
`train, valid, test`, and within a split the order the `.cxl` index lists.

Output: `$SANDISK/data/source/APPROX_GED/exported_suite2/{key}.npz` + `manifest.json`.

### 2.1 Amendments, 2026-08-13 — both found by `wave-t05-export` and both verified by the orchestrator

**Amendment 1 — `graph_ids` is the loader's native id, not `{key}_{split}_{sourceid}`.** Measured in
`extended_merged_exact_ged/computed/*.npz`: Letter ids are bare filename stems (`IP1_0000`,
`AP1_0001`), and only the GraphEdX ids (`linux_train_0000`, `aids_train_0001`) match the pattern
originally written here, because `graphedx_loader` happens to build them that way. Applying the
original wording literally would have broken the element-wise reproduction of the three Letter
`graph_ids` arrays, which is one of this track's acceptance criteria. **The original wording was
wrong; the exporter matches Suite 1's behaviour exactly.**

**Amendment 2 — no class count is asserted.** The counts previously written here (Letter 15, GREC 22,
Mutagenicity 2, Protein 6, AIDS 2, COIL-DEL 100) are the **raw** dataset class counts — orchestrator
re-verified against the `.cxl` indices, all five correct as raw figures. They are not the counts that
survive `require_connected`. The realised per-dataset class count is a **measured output** recorded in
each file's `metadata` and in `manifest.json`, never an assertion.

> ⚠ **Carry this to T-18 and T-06.** `wave-t05-export` measured that **Letter LOW retains 9 of its 15
> classes** and **GREC 17 of its 22** after the connectivity filter, and that **LINUX and
> AIDS-GraphEdX carry no class label at all** (`graphedx_loader` has no label field; T-01 already
> measured LINUX as carrying no node or edge attribute either). Any manuscript sentence of the form
> "Letter, 15 classes" or "GREC, 22 classes" is **false of the filtered cohort**. This is the labels
> counterpart of the size-biased connectivity discard already recorded in `decisions.md` §7.

**Finding 3 — a frozen artifact cannot load GraphEdX from today's tree.** `export_graphs.py:430` and
`cohort_audit.py:254` both resolve GraphEdX as `<source>/GED_PRECOMPUTED/<NAME>`. The real path is
`<source>/GED_PRECOMPUTED/datasets/<NAME>`, and `<source>/GED_PRECOMPUTED/LINUX` does not exist —
orchestrator verified. Because IAM now lives under `APPROX_GED/datasets/IAM_Database/extracted` and
GraphEdX under `GED_PRECOMPUTED/datasets`, **no single `--source` value makes either module resolve
both**, which is why CONTRACTS §1 specifies two roots. Neither file is patched in this wave — both are
frozen T-01/T-03 artifacts and `cohort_audit.py` is decision 22's reproduction script. **Recorded for
the T-05 close to propagate**: T-01's tracked reproduction cannot re-derive the LINUX and
AIDS-GraphEdX rows on the current tree without a path fix.

---

## 3. Method roles — the frozen specification

Cost model **D6** `[1, 1, 0, 1, 1, 0]`, `CONSTANT`, every run, every role.

| Role id | Method | Options string, **verbatim** | Accessor | Scope |
|---|---|---|---|---|
| `lb` | `BRANCH_FAST` | `--threads 1` | lower | all 21,710,892 pairs |
| `ub` | `BIPARTITE` | `--threads 1` | upper | all 21,710,892 pairs |
| `ubs` | `BP_BEAM` | `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1` | upper | all 21,710,892 pairs |
| `ubt` | `IPFP` | `--threads 1 --randomness PSEUDO --initial-solutions 10` | upper | the §5 subsample only |

**The options string is part of the method name.** GEDLIB's upper bounds change on 91.5–93.6 % of
pairs between runs at library defaults (T-27 §4.2). A run whose `metadata.options_string` does not
record the string verbatim is invalid and will be rejected at the gate.

Every upper bound is computed in **both orientations and the minimum taken**. `BRANCH_FAST` is
symmetric and is computed in one orientation, with symmetry **asserted** on the first 32 pairs.

---

## 4. Output contract — `{ROLE_DIR}/{key}.npz`

```
$SANDISK/data/source/APPROX_GED/LB/{key}.npz               role lb
$SANDISK/data/source/APPROX_GED/UB/{key}.npz               role ub
$SANDISK/data/source/APPROX_GED/UB_SENSITIVITY/{key}.npz   role ubs
```

**Exactly the ten keys of `GED_PRECOMPUTED/extended_merged_exact_ged/computed/*.npz`**, same dtypes,
so one loader reads exact and approximate files alike.

| Key | dtype | shape | Contents |
|---|---|---|---|
| `ged_matrix` | float64 | (N,N) | **this role's own value** — `lb` in `LB/`, `ub` in `UB/` and `UB_SENSITIVITY/` |
| `lb_matrix` | float64 | (N,N) | role `lb`'s values — identical array in all three files |
| `ub_matrix` | float64 | (N,N) | role `ub`'s values — identical array in all three files |
| `certified_mask` | bool | (N,N) | `lb_matrix == ub_matrix` at `1e-9`; diagonal `True` |
| `seconds_matrix` | float32 | (N,N) | wall time for **this file's own method**, both orientations summed |
| `node_counts` | int32 | (N,) | |
| `edge_counts` | int32 | (N,) | |
| `graph_ids` | `<U` | (N,) | |
| `labels` | `<U` | (N,) | class label or `''` |
| `metadata` | `<U` | () | JSON, keys below |

`metadata` JSON: `dataset, role, method, options_string, accessor, cost_model, n_graphs, n_pairs,
n_zero_offdiag, n_certified, certification_rate, seconds_total, mean_seconds_per_pair, filter,
splits_merged, gedlib_source, code_commit, computed_utc, schema_version, slurm_job_id`.

### 4.1 `certified_mask` is a derived proof, and this is why it is legitimate

`GedlibBackend.pair()` returns `certified=False` **always**, deliberately: no GEDLIB method may
self-certify after `ANCHOR_AWARE_GED` was measured issuing a false optimality certificate
(`T-03-design.md` amendment 2). The mask here is **not** a self-report. It is the derived statement
*"a proven lower bound of `k` and an exhibited edit path of cost `k` together prove GED = `k`"* —
two independent proofs meeting, computed by a separate cross-fill step over two separate campaigns.
Do not source it from any backend field.

### 4.2 Cross-fill

After the three role campaigns merge, one step opens `LB/{key}.npz`, `UB/{key}.npz`,
`UB_SENSITIVITY/{key}.npz`, writes the same `lb_matrix`/`ub_matrix`/`certified_mask` into all three,
and rewrites them atomically. `ged_matrix` and `seconds_matrix` are **never** touched by cross-fill.

---

## 5. `IPFP_MS` subsample — `UB_TIGHT/subsample.npz`

**Frozen sampling design.** Stratum = bin of `max(n₁, n₂)` with edges

```
[2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99)      # 14 bins, right-open
```

pooled across all ten datasets. Draw **uniformly within bin, without replacement, seed 42**,
`min(2000, bin_population)` per non-empty bin. Bin membership uses `np.searchsorted(edges, n, side="right") - 1`.

Flat file, not a matrix:

| Key | dtype | shape |
|---|---|---|
| `dataset_key` | `<U` | (P,) |
| `pair_i`, `pair_j` | int32 | (P,) — indices into that dataset's graph order |
| `n_max` | int32 | (P,) — `max(n₁, n₂)` |
| `bin_index` | int8 | (P,) |
| `value` | float64 | (P,) — `min(fwd, rev)` |
| `value_fwd`, `value_rev` | float64 | (P,) |
| `seconds` | float32 | (P,) |
| `metadata` | `<U` | () — JSON, plus `bin_edges`, `seed`, `n_per_bin` |

**Two files, not one** (amendment 3, 2026-08-13): the sampler emits the pair list ahead of the run to
`UB_TIGHT/subsample_pairs.npz` (`dataset_key, pair_i, pair_j, n_max, bin_index, metadata`), and the
runner writes the result to `UB_TIGHT/subsample.npz` with the value and timing columns added. Keeping
them separate stops the campaign overwriting its own input.

The pair list is emitted **before** the run by the sampler and is reproducible from seed 42 alone.
`BRANCH_FAST`, `BIPARTITE` and `BP_BEAM_DET` values for these pairs are **read off the dense
matrices**, never recomputed.

**This is a size-stratified sample, not a random sample of Suite-2 pairs.** Every figure derived
from it is reported per bin and never pooled into a cohort-level mean.

---

## 6. Runner CLI — additive, T-03's behaviour preserved

`benchmarks/real_data/eval_setup/ged_exact_runner.py` gains, and nothing existing changes meaning:

```
--lb-method STR          default BRANCH_FAST
--lb-options STR         default "--threads 1"
--ub-method STR          default IPFP          (T-03's default, unchanged)
--ub-options STR         default "--threads 1"
--compute {lb,ub,both}   default both          (T-03's behaviour)
--pair-list PATH         already exists — used by the ubt role
--role STR               recorded in shard meta; no behavioural effect
```

`GedlibBackend.__init__` gains `lb_options` and `ub_options`, replacing the single
`_heuristic_options` used for both (`ged_backends.py:777`). **`BRANCH_FAST` and `BIPARTITE` happen
to share `--threads 1`; `BP_BEAM_DET` and `IPFP_MS` do not.** A single shared string cannot express
the frozen specification and is the defect this change exists to fix.

`--compute lb` skips the upper-bound calls entirely and leaves `ub = inf` in the shard;
`--compute ub` skips the lower-bound calls and leaves `lb = -inf`. The inverted-bracket guard
(`ged_backends.py:957`) is skipped when only one end is computed.

### 6.1 `zero_ok` must become lazy

`bounds()` currently calls `zero_distance_is_attainable(g1, g2, costs)` **eagerly on every pair**
(`ged_backends.py:919`). Under D6 that reaches `nx.is_isomorphic` whenever `n₁ == n₂ and m₁ == m₂`,
which on Letter is most pairs and on COIL-DEL / Mutagenicity is a VF2 call on ~30-node graphs,
21.7 M times. Make it a zero-argument callable evaluated **only when a read returns 0.0**, and add a
test that it is not called on a pair whose bounds are non-zero. This is a pure performance change:
the value it computes and the guard's behaviour must be identical.

### 6.2 Shard schema — unchanged

`pair_index` int64, `ged` float64, `lb` float64, `ub` float64, `certified` bool_, `seconds` float32,
`meta` JSON on the final shard (`ged_exact_runner.py:110 SHARD_KEYS`). Shards are deleted after the
merge passes its structural gate.

---

## 7. Merge CLI

`ged_merge_shards.py` gains:

```
--ged-from {exact,lb,ub}   which array becomes ged_matrix.  default exact  (T-03 unchanged)
--role STR                 written into metadata
--seconds-role STR         label for seconds_matrix provenance
```

Structural gate **G4**, raised as `MergeError`, unchanged plus two additions:
symmetric to machine precision · diagonal zero · every entry finite and `>= 0` ·
**off-diagonal exact-zero fraction recorded and `< 0.99`** · `certified_mask` diagonal `True`.

---

## 8. Picasso environment — verbatim

```
ACCOUNT            tic_163_uma
CONSTRAINT         sr                    # 128 c / 450 GB, AMD EPYC 7H12, homogeneous
CONDA_ENV_PREFIX   /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph
PY                 $CONDA_ENV_PREFIX/bin/python          # absolute path; conda is NOT in PATH
GEDLIB_DIR         /mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn
REPO_DIR           /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph
DATA_DIR           /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/suite2
OUT_DIR            /mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged
PYTHONPATH         $REPO_DIR:$GEDLIB_DIR
```

`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` and a per-task `PYTHONPYCACHEPREFIX`.
No `module load`. Verified working 2026-08-13: `BRANCH_FAST` 1.00, `BIPARTITE` 1.00,
`BP_BEAM --initialization-method BIPARTITE --initial-solutions 1` 1.00 on P₄ vs C₄.

`SCBI two-hour floor`: `FLOOR_SECONDS=7200`, `TARGET_SECONDS=10800`. A job projected under the floor
is **not submitted short** — cores are reduced, or the role is merged into an adjacent job.

---

## 9. Prohibitions — every agent

- **No SSH, no `rsync`, no `sbatch`, no `squeue`, no `scancel`, no contact with Picasso of any kind.**
  You may *write* SLURM scripts and run `bash -n` on them. The orchestrator submits.
- **Do not edit** anything under `.claude/notes/review/plan/`, `tickets.md`, or another track's files.
- **Do not touch** `benchmarks/real_data/eval_setup/canonical_computer.py`, `ged_bound_bakeoff.py`,
  `ged_bakeoff_analysis.py`, `cohort_audit.py`, `iam_gxl_loader.py`, `dataset_filter.py`,
  `export_graphs.py`, or anything in `src/isalgraph/core/` — read them, do not modify them.
  `cohort_audit.py` and `iam_gxl_loader.py` are T-01's certified artifacts and are frozen.
- **Nothing in `scratchpad/`.** That is what lost thirteen measurement scripts from this project.
- **`isalgraph.core` is not imported** by any file in this wave. This ticket does not touch the
  encoder and must not acquire a dependency on the C++ engine.
- **Commit incrementally**, on your own branch, with conventional-commit messages and no
  `Co-authored-by` trailer. Uncommitted work is work that cannot be merged.

## 10. Environment

```
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pytest tests/unit/ -q
$PY -m ruff check --fix src/ tests/ benchmarks/
$PY -m mypy src/isalgraph/
export PYTHONPATH=~/opt/build_gedlib/graphkit-learn        # local in-place GEDLIB build
```
