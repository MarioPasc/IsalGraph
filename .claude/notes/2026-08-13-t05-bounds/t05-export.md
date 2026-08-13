# Work log — t05-export

## Identity

| Field | Value |
|---|---|
| Agent | `wave-t05-export` |
| Wave | `2026-08-13-t05-bounds` |
| Model / effort | `claude-opus-5` / `xhigh` |
| Branch | `worktree-agent-aa518b70750701b10` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-aa518b70750701b10` |
| Base commit | `885d98d8e6b37dfeb98c4df741510fc28d4a8615` |
| Head commit | `<pending>` |
| Started / finished | `2026-08-13T00:00:00Z` / `<pending>` |
| Status | in progress |

## 1. Prompt as received

```
You are agent `wave-t05-export`, an implementation agent working inside an **isolated git worktree**
on a branch of your own, in parallel with two peers who own different files. You never see the
orchestrator's conversation; everything you need is in this prompt and in the repository.

This work is for a *Pattern Recognition* major revision (PR-D-26-03293) due 2026-08-31, read by
reviewers who checked every number last round. **Correctness beats speed. An honest negative result
beats a convenient one.**

## Standing obligations
1. Work only inside your worktree. Every file you create or edit must lie inside your declared
   ownership set. Everything else in the repository is read-only reference. Confirm at the start that
   `git rev-parse --show-toplevel` differs from `/home/mpascual/research/code/IsalGraph`; if it does
   not, stop and message `main`.
2. Commit your work in logical commits **as you go**, not at the end. Sessions die; uncommitted work
   cannot be merged, because the orchestrator merges your branch, not your working tree.
3. Maintain your work log at `.claude/notes/2026-08-13-t05-bounds/t05-export.md` from your first
   action to your last, using the template committed at
   `.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md`, and commit it as your final commit.
4. Never run `git push`, never rebase or merge, never touch a peer's branch or worktree.
5. **You have no access to Picasso.** No `ssh`, `rsync`, `sbatch`, `squeue`, `scancel`, `scp`. The
   orchestrator owns every cluster interaction.
6. You cannot ask the user anything. On an ambiguity, message `main` with a specific question, record
   the assumption you are proceeding on in your log, and keep working. Do not block.
7. Never change a frozen contract yourself. Propose it to `main`. **Finding that your brief is wrong
   is a success** — report it with evidence.
8. Report failure honestly. "This does not work and here is why" beats a plausible-looking
   implementation that was never exercised.
9. Plan before editing and write the plan into your log. Implement in small verified steps. Write
   tests as you go. Run the suite before your final commit and record the real output, failures
   included.

---

# Task: Suite-2 graph export + the IPFP subsample pair list

## Mission
Write `benchmarks/real_data/eval_setup/export_graphs_suite2.py`, which serialises the **ten Suite-2
datasets** to one `.npz` each in the schema `export_graphs.py::save_exported` already defines, and
**asserts T-01's certified graph and pair counts, exiting non-zero on any mismatch**. Then write
`benchmarks/real_data/eval_setup/approx_ged_sampling.py`, which emits the frozen seed-42
size-stratified subsample pair list. Working means: you have actually run the exporter over the real
331 MB IAM tree, all ten counts reproduce exactly, and the sampler is reproducible from seed 42
alone.

## Why this exists
Ticket T-05 computes a proven GED bracket over all **21,710,892** Suite-2 pairs on the Picasso HPC
cluster. Your ten files are the *only* input that reaches the cluster — the IAM GXL tree is 35,604
small files and Picasso's fscratch quota is a **file-count** limit (224.3k of a 250k soft cap), so
the tree is never transferred. If your graph ordering, filter or enumeration differs by one graph
from T-01's certified cohort, every downstream number is computed on a cohort the paper does not
describe, and nothing will error. The counts are the check.

## Repository orientation
- Repository root: your worktree (`git rev-parse --show-toplevel`).
- **Read first, in this order**:
  1. `.claude/notes/2026-08-13-t05-bounds/CONTRACTS.md` — §1 dataset keys and roots, §2 your output
     schema, §5 the subsample design. **This is your specification.**
  2. `.claude/notes/review/tasks/T-05-design.md` §2, §3.3 — the cohort table and why it is locked.
  3. `benchmarks/real_data/eval_setup/export_graphs.py` — the Suite-1 exporter. **Your exemplar.**
     Read `DatasetSpec` (:70), `DATASETS` (:96), `save_exported` (:310), `load_exported` (:324), and
     the metadata construction (~:530). Copy its structure; do not import its Suite-1 registry as if
     it were yours.
  4. `benchmarks/real_data/eval_setup/cohort_audit.py` :66-92 — `SUITE1_N_MAX`, `NO_N_MAX`,
     `SUITE2_KEYS`. This is T-01's certified enumeration and it already produces the exact counts you
     must reproduce.
  5. `benchmarks/real_data/eval_setup/iam_gxl_loader.py` :97 `IAM_DATASETS`, :165 `dataset_dir`,
     :254 `load_iam_gxl` — the Suite-2 IAM loader.
  6. `benchmarks/real_data/eval_setup/dataset_filter.py::filter_graphs`.
- Conventions: `CLAUDE.md` is loaded for you. Additionally: NumPy-style docstrings, full type
  annotations, `logging` never `print` in library code, a module-level custom exception, Python 3.11.

## Your ownership (exclusive write access)
Create or modify ONLY:
- `benchmarks/real_data/eval_setup/export_graphs_suite2.py`
- `benchmarks/real_data/eval_setup/approx_ged_sampling.py`
- `tests/unit/test_export_graphs_suite2.py`
- `tests/unit/test_approx_ged_sampling.py`
- `.claude/notes/2026-08-13-t05-bounds/t05-export.md` (your log)

Plus this **data output directory**, outside the repository, yours alone in this wave:
- `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED/exported_suite2/`

Everything else is read-only. In particular **`cohort_audit.py`, `iam_gxl_loader.py`,
`dataset_filter.py` and `export_graphs.py` are frozen T-01/T-03 artifacts — import them, never edit
them.** If T-01's loader has a defect, report it to `main`; do not patch it.

## Base state
- Base commit: `885d98d8e6b37dfeb98c4df741510fc28d4a8615` — "chore(T-05): stage the wave note
  template and archive the agent definitions".
- Your peers branch from the same commit. Do not rebase, merge or cherry-pick.

## Frozen contracts
From `CONTRACTS.md`; code against them exactly.

- **The ten keys, sources and counts** — CONTRACTS §1. Reproduce every count or exit non-zero.
- **Two source roots, not one:**
  - `IAM_ROOT = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED/datasets/IAM_Database/extracted`
  - `GRAPHEDX_ROOT = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/GED_PRECOMPUTED`
  The plan and T-27's reproduction line both name `…/data/source/IAM_Database/extracted`; **that path
  does not exist on this machine.** LINUX and AIDS-GraphEdX are not IAM datasets.
- **Filter**: `min_nodes = 2`, `require_connected = True`, **no `n_max`**, splits merged.
- **Decision 27 — enumerate by split index, not by directory.** This binds `coil_del`: 7,200 `.gxl`
  files ship in `COIL-DEL/data/` and `train.cxl`/`valid.cxl`/`test.cxl` name **3,900** of them
  (2,400/500/1,000). The other 3,300 carry no class label. Exporting 7,200 reproduces the *retracted*
  19,670-graph / 40,024,242-pair cohort and is the single most likely way to get this wrong.
- **`aids_graphedx` has 819 graphs, not 769.** Suite 1's `aids` applies `n_max = 12`; Suite 2 does
  not. They are different cohorts and must never share a filename.
- **Output schema** — CONTRACTS §2, byte-identical key set to `save_exported`. `load_exported` must
  read your files unchanged with `allow_pickle=False`.
- **Graph order is load-bearing.** Every downstream pair index is `numpy.triu_indices(N, k=1)` over
  it. Order: splits `train, valid, test`; within a split, the order the `.cxl` index lists. It must
  be deterministic across runs and machines — verify by exporting twice and comparing `graph_ids`.
- **`labels` is populated for Suite 2**, unlike the Suite-1 files: the class label per graph where
  the dataset has one (Letter 15 classes, GREC 22, Mutagenicity 2, Protein 6, AIDS 2, COIL-DEL 100),
  `''` where it does not (LINUX carries no attribute at all — T-01 measured this).
- **Subsample design** — CONTRACTS §5, verbatim: 14 right-open bins on `max(n₁,n₂)` with edges
  `[2,4,6,8,10,12,15,20,25,30,40,50,60,80,99)`, bin index by
  `np.searchsorted(edges, n, side="right") - 1`, uniform draw within bin without replacement,
  **seed 42**, `min(2000, bin_population)` per non-empty bin, pooled across all ten datasets.

## Environment bootstrap
Your worktree is a fresh checkout. The conda environment already exists; nothing needs installing.
```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
export PYTHONPATH=~/opt/build_gedlib/graphkit-learn
cd "$(git rev-parse --show-toplevel)"
```
Do **not** put `<worktree>/src` on `PYTHONPATH` — it shadows the installed package. You do not need
`isalgraph`; **do not import it.** A subagent's `cd` does not persist between Bash calls, so prefix
every command with `cd "<your absolute worktree path>" && …`.

## Verification commands
```bash
$PY -m pytest tests/unit/test_export_graphs_suite2.py tests/unit/test_approx_ged_sampling.py -q
$PY -m pytest tests/unit/ -q                       # before your final commit
$PY -m ruff check benchmarks/ tests/
```

## Data and shared resources
- **IAM tree** (read-only, 331 MB, 35,604 files): `$IAM_ROOT` above, subdirs
  `Letter/{LOW,MED,HIGH}`, `GREC/data`, `AIDS/data`, `COIL-DEL/data`, `Mutagenicity/data`,
  `Protein/data`.
- **GraphEdX** (read-only): `$GRAPHEDX_ROOT/datasets/{AIDS,Letter,LINUX}`.
- **Reference for the Suite-1 subset**: `$GRAPHEDX_ROOT/extended_merged_exact_ged/computed/*.npz`
  carries `graph_ids`, `node_counts`, `edge_counts` for the Suite-1 cohorts. For
  `iam_letter_{low,med,high}` and `linux` the Suite-2 cohort is **identical** to Suite 1, so your
  export **must reproduce those four `graph_ids` arrays element-wise.** That is a free, exact,
  end-to-end check of your loader and ordering against a census already on record — use it.
- Your peers do not touch any of these paths. **No Picasso.**

## Definition of done
1. `export_graphs_suite2.py --verify-only` reproduces **all ten** CONTRACTS §1 rows exactly (graphs
   and pairs) and exits 0; a deliberately corrupted expectation makes it exit non-zero.
2. Ten `.npz` files written to `exported_suite2/` plus a `manifest.json`, all readable by
   `export_graphs.py::load_exported`, all keys and dtypes per CONTRACTS §2.
3. `graph_ids` for `iam_letter_low`, `iam_letter_med`, `iam_letter_high` and `linux` match the
   corresponding arrays in `extended_merged_exact_ged/computed/*.npz` **element-wise**.
4. `coil_del` has exactly 3,900 graphs and they are exactly class-balanced, 100 classes × 39 — assert
   both, since the balance is what distinguishes the index enumeration from the directory one.
5. `approx_ged_sampling.py` emits ≤ 28,000 pairs, reproducible byte-for-byte from seed 42 across two
   independent runs, with every bin's realised count recorded; pairs are valid `(i, j)` with `i < j`
   into the exported graph order of their named dataset.
6. Tests cover: each of the ten counts, the COIL-DEL index-vs-directory distinction, order
   determinism, label population, the `n_max`-free filter, the four-dataset `graph_ids` reproduction,
   subsample reproducibility, bin-edge boundary values (n exactly 4, 12, 30, 98), and a bin whose
   population is under 2,000.
7. Real-data verification recorded in the log with **numbers**: per dataset the raw count, kept
   count, pair count, wall time, file size, and the realised per-bin subsample counts.
8. All work committed on your branch; working tree clean. Work log written and committed.

## Explicitly out of scope
- Computing any GED, bound or distance. You export graphs and emit a pair list; that is all.
- Editing `cohort_audit.py`, `iam_gxl_loader.py`, `dataset_filter.py` or `export_graphs.py`.
- Anything touching `src/isalgraph/`, the C++ engine, or `isalgraph.core`.
- Transferring anything to Picasso.
- Adding COIL-RAG, Fingerprint or Web — they are deliberately not in the cohort.

## Work log — mandatory
Maintain `.claude/notes/2026-08-13-t05-bounds/t05-export.md` using the template at
`.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md` **verbatim** (read it first; it is committed
in your worktree). Write it continuously. Commit it last with `docs(notes): t05-export work log`.

## Peers in this wave
- `main` — the orchestrator. Message it for ambiguities, contract defects, blockers, or anything
  needing a decision outside your ownership. It owns every Picasso interaction.
- `wave-t05-runner` — makes the production GED runner express T-27's method+options specification and
  emit the T-05 output schema. Owns `ged_backends.py`, `ged_exact_runner.py`, `ged_merge_shards.py`,
  `approx_ged_crossfill.py` and their tests.
- `wave-t05-slurm` — writes the Picasso launcher/worker pair and the independent validation gates.
  Owns `slurm/approx_ged/*`, `approx_ged_gates.py` and its test.

Both peers consume your output schema and your dataset keys. If you must deviate from CONTRACTS §1,
§2 or §5, message `main` **immediately** — the orchestrator relays contract changes, not you.

## Final message format
At most 15 lines: STATUS, BRANCH, WORKTREE, HEAD, LOG, TESTS (counts + command), then three bullets
on what you built, anything the orchestrator must know, and anything unfinished.
```

## 2. Understanding and plan

**Restatement of the task in my own words:** Build the input side of T-05. Ten Suite-2 datasets must
be serialised to ten `.npz` files in `export_graphs.py`'s existing CONTRACT A schema, with T-01's
certified graph and pair counts asserted rather than trusted, and with the graph order frozen because
every downstream pair index is `triu_indices(N, 1)` over it. Then emit the frozen seed-42
size-stratified pair list that selects which pairs the expensive `IPFP_MS` sensitivity arm runs on.

**Approach chosen:** Reuse, never reimplement. Both new modules import the frozen T-01/T-03 artifacts
(`iam_gxl_loader.load_iam_gxl`, `dataset_filter.filter_graphs`, `export_graphs.save_exported`,
`export_graphs._normalise`, `cohort_audit.NO_N_MAX` and `cohort_audit.SUITE2_KEYS`) so that
"Suite 2 reproduces T-01's cohort" is a genuine check on shared code rather than an agreement between
two independent implementations. The registry keys are asserted equal to `cohort_audit.SUITE2_KEYS`
at import time, which makes divergence from T-01's certified enumeration impossible to introduce
silently.

**Alternatives considered and rejected:**
- Copy `export_graphs.py` and edit the constants — rejected: it forks the schema. Any later fix to
  `save_exported` or `_validate_arrays` would then apply to Suite 1 only, and the two suites would
  drift apart without any test noticing.
- Reimplement the CSR packing and validation locally — rejected for the same reason, and because
  `_validate_arrays` is the only thing standing between a truncated file and a silently wrong graph.
- Two-pass streaming sampler that never materialises the 21.7 M-pair pool — rejected: the pool costs
  ~250 MB with int32/int16 casts, which is affordable once, and the rank-to-pair arithmetic a
  streaming version needs is exactly the kind of index bug this ticket cannot afford.
- Exposing `--seed` and `--max-per-bin` on the sampler CLI — rejected: CONTRACTS §5 freezes both. They
  stay function parameters so tests can vary them, and the CLI cannot.

**Plan as executed:**
1. Confirm worktree isolation; read CONTRACTS, T-05-design §1.1/§2/§3.3, and the four frozen modules.
2. Reconnaissance on the real data *before* writing code: load all ten datasets, filter, compare
   counts and the four reference `graph_ids` arrays. Report contract defects to `main` immediately.
3. Write `export_graphs_suite2.py`; run `--verify-only`; run the real export.
4. Write `approx_ged_sampling.py`; run it on the real export; record realised per-bin counts.
5. Write both test modules; run them, then the whole unit suite; ruff.
6. Finish the log, commit.

**Deviations from the plan:** Recorded in §7 as they arose. Two CONTRACTS §2 statements turned out to
be measurably wrong (see §7) and were reported to `main` before any code was written.

## 3. Changes made

<!-- filled in as work proceeds -->

## 4. Tests

<!-- filled in as work proceeds -->

## 5. Test results

<!-- filled in as work proceeds -->

## 6. Verification beyond unit tests

### Reconnaissance run, real data, before any code was written

Ten datasets loaded through `iam_gxl_loader.load_iam_gxl(enumeration="cxl")` and
`graphedx_loader.load_graphedx_dataset`, filtered with
`filter_graphs(n_max=10**9, require_connected=True, min_nodes=2)`.

| key | raw | kept | expected | pairs | expected | drop trivial | drop disconnected | n max | n min | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iam_letter_low` | 2,250 | 1,180 | 1,180 | 695,610 | 695,610 | 1 | 1,069 | 7 | 2 | 0.1 s |
| `iam_letter_med` | 2,250 | 1,253 | 1,253 | 784,378 | 784,378 | 4 | 993 | 8 | 2 | 0.1 s |
| `iam_letter_high` | 2,250 | 2,059 | 2,059 | 2,118,711 | 2,118,711 | 8 | 183 | 9 | 2 | 0.1 s |
| `linux` | 89 | 89 | 89 | 3,916 | 3,916 | 0 | 0 | 10 | 4 | 0.5 s |
| `aids_graphedx` | 911 | 819 | 819 | 334,971 | 334,971 | 0 | 92 | 20 | 2 | 0.1 s |
| `grec` | 1,100 | 650 | 650 | 210,925 | 210,925 | 0 | 450 | 24 | 4 | 0.1 s |
| `aids_iam` | 2,000 | 1,811 | 1,811 | 1,638,955 | 1,638,955 | 0 | 189 | 85 | 2 | 0.2 s |
| `coil_del` | 3,900 | 3,900 | 3,900 | 7,603,050 | 7,603,050 | 0 | 0 | 77 | 3 | 0.6 s |
| `mutagenicity` | 4,337 | 4,040 | 4,040 | 8,158,780 | 8,158,780 | 0 | 297 | 98 | 4 | 0.5 s |
| `protein` | 600 | 569 | 569 | 161,596 | 161,596 | 0 | 31 | 96 | 2 | 0.2 s |
| **Total** | | **16,370** | **16,370** | **21,710,892** | **21,710,892** | | | **98** | **2** | |

All ten rows reproduce. The four reference `graph_ids` arrays in
`extended_merged_exact_ged/computed/` match element-wise:
`iam_letter_low` (1,180), `iam_letter_med` (1,253), `iam_letter_high` (2,059), `linux` (89).

`coil_del` under `enumeration="cxl"`: 3,900 graphs, 100 classes, every class exactly 39, split order
`train, valid, test` (2,400 / 500 / 1,000). Under `enumeration="directory"`: 7,200 graphs kept,
25,916,400 pairs, 3,300 of them carrying no class label. The two enumerations are separated by 3,300
graphs and 18,313,350 pairs.

<!-- further verification filled in as work proceeds -->

## 7. Decisions, assumptions, open questions

### Contract defect 1 — CONTRACTS §2 `graph_ids` format is wrong, and DoD 3 proves it

CONTRACTS §2 specifies `graph_ids` as `{key}_{split}_{sourceid}`. Measured in the reference census at
`extended_merged_exact_ged/computed/*.npz`, the Suite-1 ids are the **source loader's own id**:

| file | first id | shape |
|---|---|---|
| `iam_letter_low.npz` | `IP1_0000` | (1180,) |
| `iam_letter_med.npz` | `AP1_0011` | (1253,) |
| `iam_letter_high.npz` | `AP1_0001` | (2059,) |
| `linux.npz` | `linux_train_0000` | (89,) |

The Letter ids carry neither key nor split; they are the `.gxl` filename stem
(`iam_letter_loader.py:156`). Only the GraphEdX ids match the §2 pattern, and only incidentally,
because `graphedx_loader` happens to build them as `f"{name_lower}_{split}_{idx:04d}"`. Applying §2
literally would make DoD criterion 3 unsatisfiable for three of the four datasets it names.

**Assumption proceeded on:** emit the loader-native id, byte-identical to `export_graphs.py`'s
Suite-1 behaviour. Reported to `main`. What breaks if wrong: a peer keying on the §2 pattern would
not find its graphs — but no peer can, since the pattern does not exist in the Suite-1 files either.

### Contract defect 2 — CONTRACTS §2's label class counts are wrong for two datasets

§2 states "Letter 15 classes, GREC 22, Mutagenicity 2, Protein 6, AIDS 2, COIL-DEL 100". Measured
after the filter:

| key | classes kept | §2 says |
|---|---:|---|
| `iam_letter_low` | **9** | 15 |
| `iam_letter_med` | 15 | 15 |
| `iam_letter_high` | 15 | 15 |
| `grec` | **17** | 22 |
| `aids_iam` | 2 | 2 (AIDS) |
| `coil_del` | 100 | 100 |
| `mutagenicity` | 2 | 2 |
| `protein` | 6 | 6 |
| `linux` | 0 (all `''`) | — |
| `aids_graphedx` | **0 (all `''`)** | 2 (AIDS) |

The discrepancies are consequences of the connectivity filter, not of the loader: GREC drops 450 of
1,100 graphs and loses five classes entirely; Letter LOW drops 1,069 of 2,250 and retains 9 of 15.
`aids_graphedx` has **no** label at all — `graphedx_loader.GraphEdXDatasetResult` has no label field
and `export_graphs.py:446` hardcodes `[""] * len(graphs)` — so §2's "AIDS 2" is true of `aids_iam`
only. Reported to `main`. The measured counts are asserted in the registry, so a future label
regression fails loudly.

### Contract defect 3 — both frozen GraphEdX loaders point at a path that does not exist

`export_graphs.py:430` and `cohort_audit.py:254` both resolve GraphEdX as
`<source>/GED_PRECOMPUTED/<NAME>`. On this machine the tree is `GED_PRECOMPUTED/datasets/<NAME>`;
`GED_PRECOMPUTED/LINUX` does not exist. Neither frozen module can therefore load `linux` or
`aids_graphedx` from the real tree under any single `--source`, which is exactly why CONTRACTS §1
mandates two roots. **Not patched** — they are frozen T-01/T-03 artifacts and the brief forbids it.
Reported to `main`. This module calls `load_graphedx_dataset` directly with
`$GRAPHEDX_ROOT/datasets` and replicates `export_graphs._load_graphedx`'s split reconstruction and
its id-versus-split cross-check verbatim.

### Open question raised with `main` (non-blocking)

CONTRACTS §5 names `UB_TIGHT/subsample.npz` for the **result** file carrying `value`/`seconds`. The
sampler emits only the pair list, ahead of the run, so writing to that name would guarantee the
runner overwrites it. Proceeding with `UB_TIGHT/subsample_pairs.npz` plus one runner-consumable
`UB_TIGHT/pair_lists/{key}.npz` per dataset carrying the `pair_index` key that
`ged_exact_runner.py:794` requires.

## 8. Coordination

**Messages sent:** to `main`, before writing any code — the three contract defects above and the
`subsample.npz` naming question, each with the measurement that establishes it.

**Messages received and how they changed the work:** <pending>

**Contracts I depend on and confirmed unchanged:** CONTRACTS §1 (ten keys, two roots, counts, filter,
decision 27), §2 (output schema, minus the two defects above), §5 (subsample design).

## 9. Deliberately not done

<!-- filled in as work proceeds -->

## 10. Risks and follow-ups

<!-- filled in as work proceeds -->

## 11. Self-assessment against the definition of done

<!-- filled in as work proceeds -->
