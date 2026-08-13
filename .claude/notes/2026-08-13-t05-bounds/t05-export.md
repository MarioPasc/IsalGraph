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
| Head commit | `3a7c257` + this log |
| Started / finished | `2026-08-13T15:45Z` / `2026-08-13T16:40Z` |
| Status | complete |

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

**Created**

| Path | Purpose |
|---|---|
| `benchmarks/real_data/eval_setup/export_graphs_suite2.py` | The Suite-2 registry and driver: loads, filters, asserts the locked cohort, writes ten `.npz` + `manifest.json` |
| `benchmarks/real_data/eval_setup/approx_ged_sampling.py` | Draws and emits the frozen seed-42 size-stratified `IPFP_MS` pair list |
| `tests/unit/test_export_graphs_suite2.py` | 62 tests over the exporter |
| `tests/unit/test_approx_ged_sampling.py` | 46 tests over the sampler |
| `.claude/notes/2026-08-13-t05-bounds/t05-export.md` | This log |

**Modified** — none. Every frozen artifact is byte-identical to the base commit, verified with
`git show <base>:<f> | diff -` for `export_graphs.py`, `cohort_audit.py`, `iam_gxl_loader.py`,
`dataset_filter.py`, `graphedx_loader.py` and `tests/unit/test_export_graphs.py`.

**Removed** — none.

**Commits**

| SHA | Message |
|---|---|
| `3df997c` | `feat(T-05): Suite-2 graph exporter with the locked cohort asserted` |
| `625a84a` | `feat(T-05): frozen seed-42 size-stratified subsample pair list` |
| `3a7c257` | `test(T-05): cover the Suite-2 export and the frozen subsample` |
| `172c309` | `docs(notes): t05-export work log` |
| `33b9209` | `feat(T-05): probe pair list and the per-bin cost table` |
| *(final)* | `docs(notes): t05-export work log` (second assignment) |

**Data written** (outside the repository, my declared output directory):
`.../APPROX_GED/exported_suite2/` — 10 `.npz` + `manifest.json`, 552,756 B; plus, from the second
assignment, `probe_pairs.npz` (26,760 B), `probe_pair_lists/{key}.npz` x 10 (30,381 B) and
`bin_table.json` (7,137 B).
`.../APPROX_GED/UB_TIGHT/` — `subsample_pairs.npz` (190,147 B) + `pair_lists/{key}.npz` x 10.

## 4. Tests

**Tests created**

| Test | File | What it verifies | Why it matters |
|---|---|---|---|
| `test_registry_is_t01_certified_enumeration_in_order` | export | registry key tuple `==` `cohort_audit.SUITE2_KEYS` | pair indices are positional; a reordered registry maps every downstream result onto the wrong dataset |
| `test_locked_graph_and_pair_counts_reproduce[10 keys]` | export | each CONTRACTS §1 row | the primary check; a cohort off by one graph is silently wrong everywhere |
| `test_graph_ids_reproduce_the_suite1_census[4 keys]` | export | element-wise vs `extended_merged_exact_ged/computed` | catches a **reordering**, which every count in the cohort table would pass |
| `test_coil_del_split_index_is_balanced_and_the_directory_is_not` | export | 3,900 / 100x39 / all labelled vs 7,200 / 3,300 unlabelled | the retraction guard; nothing else separates the two enumerations |
| `test_label_classes_are_populated_as_measured` | export | the ten post-filter class counts | GREC 17 and Letter LOW 9 are printed-number risks |
| `test_no_n_max_keeps_graphs_suite1_would_drop` | export | `aids_graphedx` 819, not 769 | the two AIDS cohorts must never be confused |
| `test_export_is_deterministic_across_two_runs` | export | `content_sha256` equal across two exports | order is load-bearing and must not depend on run |
| `test_written_files_load_and_carry_the_contract_schema[10]` | export | `load_exported` + exact metadata key set | peers read these files; a schema drift breaks them, not me |
| `test_main_exits_non_zero_on_a_corrupted_expectation` | export | corrupted `expected_kept` -> exit 1 | a wrong count must stop the pipeline, not be reconciled |
| `test_bin_of_at_and_around_every_boundary[16]` | sampling | n = 2,3,4,5,6,11,**12**,14,15,29,**30**,39,40,79,**80**,**98** | a right-open edge read the wrong way moves thousands of pairs between strata |
| `test_draw_takes_the_whole_bin_when_its_population_is_below_the_cap` | sampling | `min(2000, population)` on a small bin | the under-2,000 branch never fires on real data, so only a fixture reaches it |
| `test_draw_is_reproducible_from_the_seed_alone` / `test_a_different_seed_gives_a_different_draw` | sampling | same seed same draw, different seed different draw | the second guards against a draw that only *looks* reproducible because it ignores the RNG |
| `test_pair_index_inverts_to_the_emitted_i_and_j` | sampling | `pair_from_index(k, n) == (i, j)` | `pair_index` is what the runner consumes; it must name the same pair |
| `test_check_dataset_keys_rejects_a_truncated_column` | sampling | the exact `np.full(..., dtype=np.str_)` `<U1` corruption | this is a measured defect, reproduced as a regression test |
| `test_real_draw_hits_the_ceiling_and_reproduces` | sampling | 28,000 pairs, all 14 bins > 2,000, digest stable | end-to-end on the real cohort |

**Coverage of the behaviour that matters.** Both loader families, both enumerations, every locked
count, both totals, the metadata schema, the CSR round trip, every bin boundary, both draw branches
(capped and whole-bin), and both reproducibility directions.

**Not tested, and why.**
- No test drives `export_graphs_suite2` end-to-end with a **fabricated** IAM tree. Every real-data
  test skips on a machine without the Sandisk tree, so on such a machine the exporter is covered only
  by its registry and assertion tests. Building a synthetic GXL/CXL fixture tree was judged lower
  value than the element-wise census check, which is a stronger statement on the machine that
  matters. This is a real gap on CI.
- `_load_graphedx`'s split-reconstruction failure paths are not induced; they would need a mocked
  loader returning inconsistent `split_sizes`.
- Nothing verifies the files against Picasso, by design — I have no cluster access.

## 5. Test results

**Command:** `~/.conda/envs/isalgraph-cpp/bin/python -m pytest tests/unit/ -q -p no:randomly`

*After the second assignment* (probe + bin table, commit `33b9209`):

```
================= 8 failed, 1002 passed, 44 skipped in 36.42s ==================
```

My own two modules: **138 passed** (was 108; +30 for the probe and the bin table), 0 failed, 20.3 s.
`ruff check` clean on all four files. The eight failures are the same eight pre-existing ones,
unchanged in identity and count.

*Before the second assignment*, for comparison:

```
================== 8 failed, 972 passed, 44 skipped in 36.01s ==================
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_low]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_med]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_high]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[linux]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[aids]
FAILED tests/unit/test_export_graphs.py::test_real_export_all_five_totals
FAILED tests/unit/test_export_graphs.py::test_real_export_is_deterministic
FAILED tests/unit/test_real_aids_retains_within_split_structure
E   FileNotFoundError: GraphEdX dataset not found:
    .../isalgraph/data/source/GED_PRECOMPUTED/AIDS
```

**My own two modules:**

```
~/.conda/envs/isalgraph-cpp/bin/python -m pytest \
    tests/unit/test_approx_ged_sampling.py tests/unit/test_export_graphs_suite2.py -q -p no:randomly
============================= 108 passed in 19.40s =============================
```

**Result:** 108 passed, 0 failed for the code I own. **Duration:** 19.4 s. **Run at:** `3a7c257`.

**Failures and their resolution.** The 8 failures are **pre-existing and not mine**, and I have not
fixed them because both offending files are frozen and outside my ownership. Evidence:

1. `git diff --name-status 885d98d..HEAD` lists three **added** files and zero modified.
2. All six relevant frozen files are byte-identical to the base commit.
3. The failure is a missing directory, on paths my code never constructs:
   `GED_PRECOMPUTED/AIDS`, `GED_PRECOMPUTED/LINUX` and `data/source/IAM_Database/extracted` are all
   absent; the real tree is `GED_PRECOMPUTED/datasets/<NAME>` and
   `APPROX_GED/datasets/IAM_Database/extracted`.

This is defect 3 made concrete: **T-01's tracked reproduction of the Suite-1 cohort is currently red
on this machine.** Both roots moved, so the Letter tests fail on the IAM path and the LINUX/AIDS
tests on the GraphEdX path. Reported to `main` with the same evidence. My exporter reproduces all
four overlapping Suite-1 cohorts element-wise, so the cohort itself is intact; only the Suite-1
exporter's path resolution is broken.

**Two defects the tests caught during development, both fixed:**

1. *Silent `<U1` truncation, found by running on real data.*
   `np.full(size, key, dtype=np.str_)` produces a `<U1` array and cuts every dataset key to its first
   character. The pooled file was written with `dataset_key` entries `'m'`, `'c'`, `'i'`, `'a'`,
   `'g'`, `'p'`, `'l'` — collapsing the **three Letter datasets into `'i'`** and **both AIDS cohorts
   into `'a'`**. Nothing raised. The only visible symptom was that the ten per-dataset pair lists
   were not written, because `dataset_key == "iam_letter_low"` matched nothing. A unit test on
   fabricated data would not have caught this: it needs keys that share a first character, which only
   the real registry has. Fixed by deriving the dtype width from the registry, and guarded by
   `_check_dataset_keys`, which rejects any value that is not a Suite-2 key.
1b. *Two more test assertions corrected during the second assignment, both by the test failing.*
   I asserted an equal-per-bin probe draw is flat. It is flat only when every bin can absorb its
   share — true of the real cohort, false of a 97-graph fixture whose bin 0 holds a single pair.
   Rather than weaken the assertion I split it: `test_probe_spreads_equally_across_bins_not_
   proportionally` on a fixture that spans every bin, and `test_probe_redistributes_from_a_bin_too_
   small_to_take_its_share` asserting that a starved bin is drained entirely and the surplus lands
   elsewhere. The second is the branch the real cohort never exercises. I also asserted a population
   spread of `> 50x` where the fixture gives 28x; corrected to `> 20x` after measuring rather than
   adjusting the fixture to fit the claim.

2. *An over-strong test assertion, caught by the test failing.*
   I asserted that `build_pairs` rejects a 1-node graph. It does not, and should not: the stratum is
   `max(n1, n2)`, so `max(1, 5) = 5` is a valid bin and a lone small graph is invisible at pair level.
   The guard fires only when **both** graphs are below the design. Corrected to assert the true
   semantics, plus a second test pinning the boundary of what the guard covers so it is not later
   mistaken for a filter. `filter_graphs(min_nodes=2)` excludes such graphs upstream regardless.

**Lint:** `ruff check` passes on all four files I own. `ruff check benchmarks/ tests/` reports 28
errors, all pre-existing in files I do not own (`roundtrip_fixed_point.py`,
`starting_node_sensitivity.py`, `eval_setup.py` and others); I introduced none and fixed none.

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

### Table A — the written export, per dataset

Produced by `python -m benchmarks.real_data.eval_setup.export_graphs_suite2`, read back from
`manifest.json`. Build time is per dataset, measured separately from a warm page cache.

| key | raw | kept | pairs | n max | edges | bytes | build s | classes kept | classes raw | classes lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `iam_letter_low` | 2,250 | 1,180 | 695,610 | 7 | 3,618 | 9,482 | 0.09 | **9** | 15 | A, E, F, H, K, T |
| `iam_letter_med` | 2,250 | 1,253 | 784,378 | 8 | 3,969 | 10,487 | 0.09 | 15 | 15 | — |
| `iam_letter_high` | 2,250 | 2,059 | 2,118,711 | 9 | 9,381 | 18,051 | 0.10 | 15 | 15 | — |
| `linux` | 89 | 89 | 3,916 | 10 | 743 | 3,358 | 0.50 | 0 | 0 | — |
| `aids_graphedx` | 911 | 819 | 334,971 | 20 | 9,194 | 15,012 | 0.10 | 0 | 0 | — |
| `grec` | 1,100 | 650 | 210,925 | 24 | 8,077 | 10,072 | 0.13 | **17** | 22 | 5, 6, 9, 15, 21 |
| `aids_iam` | 2,000 | 1,811 | 1,638,955 | 85 | 26,263 | 35,544 | 0.28 | 2 | 2 | — |
| `coil_del` | 3,900 | 3,900 | 7,603,050 | 77 | 211,524 | 252,832 | 0.80 | 100 | 100 | — |
| `mutagenicity` | 4,337 | 4,040 | 8,158,780 | **98** | 119,385 | 138,123 | 0.81 | 2 | 2 | — |
| `protein` | 600 | 569 | 161,596 | 96 | 34,957 | 50,914 | 0.18 | 6 | 6 | — |
| **Total** | | **16,370** | **21,710,892** | **98** | **427,111** | **543,875** | **3.08** | | | |

Plus `manifest.json` at 8,881 B. **11 files, 552,756 B**, standing in for 35,604 files and 331 MB.
That ratio is the entire reason this module exists: Picasso's `fscratch` limit is on file **count**.

### Table B — class survival, per the orchestrator's ruling of 2026-08-13

Two datasets lose whole classes to the connectivity filter. The lost class list is recorded in each
`.npz` metadata and in `manifest.json` as `label_classes_lost`.

| key | raw classes | retained | lost | which |
|---|---:|---:|---:|---|
| `iam_letter_low` | 15 | **9** | 6 | A, E, F, H, K, T |
| `grec` | 22 | **17** | 5 | 5, 6, 9, 15, 21 |
| all eight others | — | unchanged | 0 | — |

**This is a printed-number risk and is not to be softened.** Any manuscript sentence of the form
"IAM Letter, 15 classes" or "GREC, 22 classes" is **false of the filtered cohort** that every Suite-2
number is computed on. Letter LOW loses 1,069 of 2,250 graphs to disconnection — the LOW distortion
level draws letters as strokes that frequently do not touch — and six letters vanish entirely. GREC
loses 450 of 1,100 and five symbol classes. Carried to T-18/T-06 by the orchestrator at close.

### Table C — pair populations, 14 bins x 10 datasets

Requested by the orchestrator to size the Picasso jobs from a measured distribution rather than an
`n̄`-based projection. Columns are pair counts; rows sum to the pooled population.

| bin | range | let_lo | let_md | let_hi | linux | aids_gx | grec | aids_iam | coil_del | mutag | protein | **pooled** | drawn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | [2, 4) | 73,920 | 73,920 | 72,390 | 0 | 1 | 0 | 6 | 741 | 0 | 1 | **220,979** | 2,000 |
| 1 | [4, 6) | 495,858 | 562,836 | 1,218,031 | 3 | 20 | 378 | 184 | 9,129 | 36 | 5 | **2,286,480** | 2,000 |
| 2 | [6, 8) | 125,832 | 146,370 | 809,804 | 102 | 414 | 4,375 | 3,726 | 98,941 | 630 | 15 | **1,190,209** | 2,000 |
| 3 | [8, 10) | 0 | 1,252 | 18,486 | 1,665 | 9,718 | 23,927 | 80,339 | 217,217 | 4,087 | 57 | **356,748** | 2,000 |
| 4 | [10, 12) | 0 | 0 | 0 | 2,146 | 133,227 | 28,950 | 597,273 | 276,225 | 17,825 | 153 | **1,055,799** | 2,000 |
| 5 | [12, 15) | 0 | 0 | 0 | 0 | 154,226 | 71,148 | 447,225 | 474,525 | 82,992 | 1,147 | **1,231,263** | 2,000 |
| 6 | [15, 20) | 0 | 0 | 0 | 0 | 22,794 | 63,112 | 48,592 | 784,707 | 498,880 | 3,978 | **1,422,063** | 2,000 |
| 7 | [20, 25) | 0 | 0 | 0 | 0 | 14,571 | 19,035 | 160,085 | 1,148,346 | 885,951 | 11,849 | **2,239,837** | 2,000 |
| 8 | [25, 30) | 0 | 0 | 0 | 0 | 0 | 0 | 56,185 | 1,449,760 | 1,451,124 | 19,923 | **2,976,992** | 2,000 |
| 9 | [30, 40) | 0 | 0 | 0 | 0 | 0 | 0 | 101,970 | 1,832,787 | 2,997,656 | 43,473 | **4,975,886** | 2,000 |
| 10 | [40, 50) | 0 | 0 | 0 | 0 | 0 | 0 | 59,381 | 665,937 | 1,157,847 | 55,380 | **1,938,545** | 2,000 |
| 11 | [50, 60) | 0 | 0 | 0 | 0 | 0 | 0 | 62,335 | 366,563 | 385,618 | 18,309 | **832,825** | 2,000 |
| 12 | [60, 80) | 0 | 0 | 0 | 0 | 0 | 0 | 18,035 | 278,172 | 495,369 | 5,605 | **797,181** | 2,000 |
| 13 | [80, 99) | 0 | 0 | 0 | 0 | 0 | 0 | 3,619 | 0 | 180,765 | 1,701 | **186,085** | 2,000 |
| **all** | | 695,610 | 784,378 | 2,118,711 | 3,916 | 334,971 | 210,925 | 1,638,955 | 7,603,050 | 8,158,780 | 161,596 | **21,710,892** | **28,000** |

Three things a job-sizing reader should take from this table.

1. **The spread is 22x.** Bin 9 `[30, 40)` holds 4,975,886 pairs against bin 0 `[2, 4)`'s 220,979.
   Since per-pair cost scales roughly as `max(n1, n2)^3`, an `n̄`-based projection is wrong by a large
   factor in both directions across this cohort; the launcher should sum cost over bins.
2. **The two top bins, stated explicitly as requested.** `[60, 80)` has a realised population of
   **797,181** pairs and `[80, 99)` **186,085**. At `min(2000, population)` the sampling fractions are
   **0.251 %** and **1.075 %** respectively — the thickest sampling of any bin, because bin 13 is the
   smallest pool. It is nevertheless the most expensive 2,000 pairs in the design.
3. **The top bin rests on three datasets, and effectively on one.** `[80, 99)` draws only from
   `mutagenicity` (180,765 pairs, 97.1 %), `aids_iam` (3,619) and `protein` (1,701). `coil_del`
   contributes nothing above n = 77. So any per-bin conclusion at the top of the size range is very
   nearly a statement about Mutagenicity alone, and should be reported that way.

### Table D — the drawn subsample, per dataset

`subsample_pairs.npz`, 190,147 B, `content_sha256 = 9deed0b553a69215e9adeaa6cee0579a7d48a22afe98c6a4786fe93a60c2ff9d`.

| key | drawn | `pair_lists/{key}.npz` |
|---|---:|---:|
| `iam_letter_low` | 1,307 | 5,449 B |
| `iam_letter_med` | 1,420 | 5,741 B |
| `iam_letter_high` | 3,218 | 10,349 B |
| `linux` | 9 | 2,191 B |
| `aids_graphedx` | 586 | 3,710 B |
| `grec` | 414 | 3,272 B |
| `aids_iam` | 2,896 | 9,516 B |
| `coil_del` | 8,777 | 24,764 B |
| `mutagenicity` | 9,161 | 25,386 B |
| `protein` | 212 | 2,759 B |
| **Total** | **28,000** | |

All 14 bins have a population above 2,000, so every bin is capped and the draw lands exactly on the
28,000 ceiling. **The `min(2000, population)` branch therefore never fires on the real cohort** and is
covered only by fixture tests — recorded so nobody assumes it was exercised in production.

### Table E — the probe pair list (second assignment, 2026-08-13)

`$EXPORT_ROOT/probe_pairs.npz`, 26,760 B, `content_sha256 = 665db5f5a51a18d9…`, plus ten
`probe_pair_lists/{key}.npz` totalling 30,381 B.

**3,000 pairs · all 10 datasets · all 14 bins · n spans 2 to 98 · disjoint from the §5 subsample.**

Per-bin draw: `[215, 215, 215, 215, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214]` = 3,000.

| dataset | probe pairs | | dataset | probe pairs |
|---|---:|---|---|---:|
| `iam_letter_low` | 102 | | `grec` | 212 |
| `iam_letter_med` | 126 | | `aids_iam` | 560 |
| `iam_letter_high` | 125 | | `coil_del` | 531 |
| `linux` | **80** | | `mutagenicity` | 543 |
| `aids_graphedx` | 206 | | `protein` | 515 |

**The allocation rule, and why this one.** Stated here and in the file metadata, per instruction.

1. **Equal per bin.** 3,000 over 14 bins by water filling: `3000 = 14 × 214 + 4`, so bins 0-3 get
   215 and the rest 214. Remainders go to the lowest index, so the allocation is a pure function of
   `(total, capacities)` with no RNG in it.
2. **Equal per dataset within a bin**, over the datasets present in that bin, by the same routine.
   A dataset holding fewer pairs than its share contributes all it has and the shortfall is
   redistributed to the others.

**Where the two axes conflict, the bin axis wins** — as instructed, and for the reason given: the
probe fits a cost curve in `n`, it does not estimate a cohort mean. With cost scaling as
`max(n₁,n₂)³`, a proportional draw would put most of its 3,000 pairs in the small-`n` corner, where
the pool is largest and the curve is flattest, and would measure a rate biased **low** — which is
exactly the failure mode the orchestrator identified in the contiguous-`grec` fallback, and it would
under-size every job in the wave.

Rule 2 is what carries dataset representation. `linux` has only **3** candidate pairs in bin 1 and
none above bin 4, yet still lands 80 pairs, because within each bin it is given an equal share of
that bin's allocation rather than a share proportional to its size. All ten datasets appear.

**Disjointness: achieved, not merely preferred.** The §5 subsample is excluded from the candidate
pool before the probe draws, and `_overlap` asserts the result is 0 shared pairs — checked in the
CLI and in `test_probe_excludes_the_subsample_pairs`. A companion test confirms the disjointness is
produced by the exclusion rather than by luck: the same draw without `exclude` yields a different
digest. This matters because a shared pair would let a probe timing and an `IPFP_MS` measurement
sit on the same work, coupling the calibration to the thing it calibrates. Candidate populations are
the §5 populations minus 2,000 per bin, visible in the log as e.g. bin 13: 186,085 → 184,085.

### Table F — `bin_table.json` (second assignment, 2026-08-13)

`$EXPORT_ROOT/bin_table.json`, 7,137 B. Schema exactly as specified: `bin_edges` (15 values,
identical to CONTRACTS §5), `totals` (14), `datasets` (key → 14 counts). `totals` sums to
**21,710,892**, and each dataset row sums to that dataset's `C(N, 2)` — both asserted by test.

Per-bin dominance, emitted in the file so it reaches the analysis rather than stopping at this log:

| bin | range | total | dominant dataset | share | flagged |
|---|---|---:|---|---:|---|
| 0 | [2, 4) | 220,979 | `iam_letter_med` | 0.335 | |
| 1 | [4, 6) | 2,286,480 | `iam_letter_high` | 0.533 | |
| 2 | [6, 8) | 1,190,209 | `iam_letter_high` | 0.680 | |
| 3 | [8, 10) | 356,748 | `coil_del` | 0.609 | |
| 4 | [10, 12) | 1,055,799 | `aids_iam` | 0.566 | |
| 5 | [12, 15) | 1,231,263 | `coil_del` | 0.385 | |
| 6 | [15, 20) | 1,422,063 | `coil_del` | 0.552 | |
| 7 | [20, 25) | 2,239,837 | `coil_del` | 0.513 | |
| 8 | [25, 30) | 2,976,992 | `mutagenicity` | 0.487 | |
| 9 | [30, 40) | 4,975,886 | `mutagenicity` | 0.602 | |
| 10 | [40, 50) | 1,938,545 | `mutagenicity` | 0.597 | |
| 11 | [50, 60) | 832,825 | `mutagenicity` | 0.463 | |
| 12 | [60, 80) | **797,181** | `mutagenicity` | 0.621 | |
| 13 | [80, 99) | **186,085** | `mutagenicity` | **0.971** | **yes** |

Bin 13 crosses the 0.90 threshold and carries a `caveat` string in the file:

> bin 13 [80, 99) is 97.1 % mutagenicity; any number quoted for this bin is very nearly a statement
> about mutagenicity alone, not a property of graphs of this size

Bin 12 `[60, 80)` is the other bin the orchestrator asked about: `mutagenicity` at **62.1 %**, with
`coil_del` at 34.9 %. Dominant but genuinely shared, so it is reported with its share and **not**
flagged — the flag has to mean something, and applying it at 62 % would make it noise.

Two further observations from the completed table, neither asked for:

- **No bin is single-dataset at the bottom of the range either.** Bin 0's largest contributor is
  `iam_letter_med` at 33.5 %, and bins 0-2 are a three-way Letter split. So the small-`n` end is a
  Letter statement in aggregate even though no single dataset dominates it — worth the same
  care as the top when a figure plots cost against `n` across the whole range.
- **The distribution is bimodal in provenance, not just in size.** Bins 0-2 are ~90 % Letter and
  bins 8-13 are ~50-97 % Mutagenicity plus COIL-DEL. A curve fitted across all 14 bins is therefore
  fitting a dataset transition as well as a size transition, and the two are confounded by
  construction of the cohort. This is not fixable by sampling — it is a property of which datasets
  exist at which sizes — but it should be said out loud wherever the size-scaling figure appears.

### Summary table

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| Real data, export | `export_graphs_suite2` over the 331 MB / 35,604-file IAM tree + GraphEdX | Table A; 10 `.npz` + manifest, 552,756 B, 3.08 s build | pass — all ten counts and both totals |
| Real data, verify | `--verify-only` | exit 0, `verify()` returns `[]` | pass |
| Negative control | `expected_kept` for `coil_del` set to 7,200, then `--verify-only` | exit **1** | pass — a corrupted expectation stops the pipeline |
| External cross-check | `graph_ids` vs `extended_merged_exact_ged/computed` | 1,180 / 1,253 / 2,059 / 89 ids, element-wise equal | pass |
| Order determinism | two full exports to separate directories | all ten `content_sha256` equal | pass |
| Real data, sampler | `approx_ged_sampling --verify-reproducible` | 28,000 pairs, digest equal across two runs, 0.46 s | pass |
| Failure paths | 99-node graph, 1-node pair, truncated `dataset_key`, off-by-one cohort, reordered ids, unbalanced COIL-DEL | each raises the intended exception | pass |
| Scale / performance | full pool of 21,710,892 pairs materialised | 0.46 s, **peak RSS 811 MB** | pass — fits comfortably; no streaming needed |
| Environment | Debian 12, `Linux-6.1.0-52-amd64` | Python 3.11.15, numpy 1.26.4, networkx 3.6.1, torch 2.13.0+cpu | — |

`isalgraph` is **not imported** by either module, as CONTRACTS §9 requires; neither touches the
encoder or the C++ engine.

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

### Contract defect 3 — the Suite-1 exporter's path resolution, which is not a cohort problem

**Two statements, deliberately kept separate.**

*Statement 1 — the path resolution is broken.* `export_graphs.py:430` and `cohort_audit.py:254` both
resolve GraphEdX as `<source>/GED_PRECOMPUTED/<NAME>`, and `export_graphs.py:411` resolves IAM as
`<source>/IAM_Database/extracted/Letter`. On this machine neither exists: the trees are
`GED_PRECOMPUTED/datasets/<NAME>` and `APPROX_GED/datasets/IAM_Database/extracted`. They now live
under **different parents**, so **no single `--source` resolves both**, and the consequence is
measurable: `tests/unit/test_export_graphs.py` has 8 real-data tests failing at the base commit,
confirmed independently by the orchestrator on a clean checkout of `885d98d` (8 failed / 32 passed,
same eight ids, same `FileNotFoundError`). T-01's *tracked reproduction script* for the Suite-1
cohort is therefore currently red.

*Statement 2 — the Suite-1 cohort itself is not in doubt.* Independently of the above, this module
reproduces all four overlapping Suite-1 cohorts **element-wise**: `iam_letter_low` (1,180),
`iam_letter_med` (1,253), `iam_letter_high` (2,059) and `linux` (89) match the
`extended_merged_exact_ged/computed` census `graph_ids` exactly, in order. So the cohort is intact and
independently re-derived; what is broken is only how the Suite-1 exporter *finds its inputs*.

**Not patched**, and the orchestrator confirmed this was correct twice over: beyond ownership, the
repair is not a corrected constant but the two-root `--iam-root` / `--graphedx-root` split that
CONTRACTS §1 specifies and this module already implements. That is a design change to a closed
ticket's deliverable, out of T-05's scope, and the PI's call. Routed by the orchestrator and recorded
in the T-05 design note changelog.

### The original framing of defect 3, retained for the record

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

Second message to `main`: the 8 pre-existing suite failures, with the three-part evidence
(`git diff --name-status`, byte-identical frozen files, missing directories), flagged before merge so
they would not read as a regression.

**Messages received and how they changed the work:**

1. *Rulings 1-4.* All three defects adopted; CONTRACTS amended at §2.1 and §5. Ruling 1 adopted my
   `graph_ids` assumption verbatim. Ruling 2 added two requirements beyond what I proposed: record
   the **sorted class list**, not only the count, in both the file metadata and `manifest.json`
   (implemented as `label_classes` / `label_classes_lost`), and put the class-survival table in this
   log explicitly (Table B). Ruling 4 approved `subsample_pairs.npz` as CONTRACTS §5 amendment 3.
   Also a new requirement: the 14-bin x 10-dataset pair population table (Table C).
2. *Confirmation of the 8 failures*, re-run independently on a clean checkout of the base commit.
   Two follow-ups, both actioned: keep the path-resolution defect and the cohort integrity as
   **separate statements** (§7 defect 3, rewritten), and state the realised populations of bins
   `[60, 80)` and `[80, 99)` explicitly (Table C, note 2: 797,181 and 186,085).
3. *Track accepted after independent re-verification, plus two further deliverables* — the probe
   pair list and `bin_table.json`. The probe had no owner in this wave and `wave-t05-slurm`'s
   fallback was a contiguous first chunk of `grec`, which over a size-ordered export oversamples
   small `n` and biases the measured rate low. Both delivered; see Tables E and F. The instruction
   to favour spanning the `n` range over dataset balance where they conflict is implemented as
   documented, and the rule plus its rationale are written into the file metadata as required, not
   only into this log.

**Contracts I depend on and confirmed unchanged:** CONTRACTS §1 (ten keys, two roots, counts, filter,
decision 27), §2 as amended at §2.1, §5 as amended by amendment 3. §9's prohibition on importing
`isalgraph` is respected — neither module imports it.

**What my peers consume from me, unchanged from the contract:** the ten dataset keys and their order;
the CONTRACT A array schema, readable by `export_graphs.load_exported`; and
`UB_TIGHT/pair_lists/{key}.npz` carrying the `pair_index` key that `ged_exact_runner.py:794`
requires, so `wave-t05-runner` consumes it with no translation step.

## 9. Deliberately not done

- **No GED, bound or distance is computed.** Out of scope; that is the runner's work.
- **`cohort_audit.py`, `iam_gxl_loader.py`, `dataset_filter.py`, `export_graphs.py` untouched** —
  verified byte-identical to base. The 8 failing Suite-1 tests are left failing on purpose.
- **`UB_TIGHT/subsample.npz` is not written.** That is the runner's output file; writing the pair
  list there would guarantee it is overwritten. CONTRACTS §5 amendment 3.
- **No fabricated GXL/CXL fixture tree**, so on a machine without the Sandisk source every real-data
  test skips. Named as a real CI gap in §4 rather than left to be discovered.
- **Nothing transferred to Picasso**, and no `ssh`/`rsync`/`sbatch` was run. The orchestrator owns
  every cluster interaction.
- **COIL-RAG, Fingerprint and Web are absent**, and a test asserts their absence.
- **`--seed` and `--max-per-bin` are not CLI flags.** They are frozen; exposing them would let the
  draw be re-rolled after seeing the result, which is the one thing the pre-registration forbids.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| The Suite-1 exporter cannot resolve both roots | **high** | 8 tests red at base; T-01's tracked reproduction of the Suite-1 cohort does not run on today's tree. The cohort is independently fine (element-wise check), but the *script* is not. Needs the two-root split, a design change to a closed ticket | orchestrator / PI |
| "Letter, 15 classes" and "GREC, 22 classes" are false of the filtered cohort | **high** | Retained are 9 and 17. Any manuscript sentence quoting the raw counts describes a cohort no number is computed on | T-18 / T-06 |
| Bin `[80, 99)` is 97.1 % Mutagenicity | medium | Any per-bin conclusion at the top of the size range is nearly a statement about one dataset. **Now carried in `bin_table.json` as a `caveat` string and a `single_dataset: true` flag**, so it reaches the analysis rather than stopping here | T-05 analysis |
| Size and provenance are confounded across the bins | medium | Bins 0-2 are ~90 % Letter, bins 8-13 are ~50-97 % Mutagenicity + COIL-DEL. A cost or tightness curve fitted across all 14 bins fits a **dataset transition** as well as a size transition. Not fixable by sampling — it is a property of which datasets exist at which sizes — but it must be stated wherever the size-scaling figure appears | T-05 analysis |
| The probe measures cost on a deliberately non-representative sample | low | Equal-per-bin over-weights large `n` roughly 22-fold relative to the cohort. Correct for a curve fit, wrong for any pooled mean. Metadata carries `not_a_cohort_estimate: true`; the launcher must integrate the fitted curve against `bin_table.json`, never multiply the probe's mean rate by 21.7 M | `wave-t05-slurm` |
| The `min(2000, population)` branch never fires in production | low | All 14 bins exceed 2,000, so the branch is covered by fixtures only. Harmless today; would matter if the cohort shrank | next wave |
| No synthetic source-tree fixture | low | Every real-data test skips without the Sandisk tree, so CI covers the registry and assertions only | next wave |
| `content_sha256`, not file bytes | low | `savez_compressed` stamps zip members with local time, so files are never byte-identical across runs. Reproducibility is asserted over array content. Documented in both modules | — |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | `--verify-only` reproduces all ten rows, exits 0; corrupted expectation exits non-zero | **yes** | `verify()` returns `[]`; exit 0 measured. Negative control with `coil_del.expected_kept = 7200` exits **1**. `test_main_verify_only_exits_zero`, `test_main_exits_non_zero_on_a_corrupted_expectation` |
| 2 | Ten `.npz` + `manifest.json`, readable by `load_exported`, keys and dtypes per §2 | **yes** | Table A; `test_written_files_load_and_carry_the_contract_schema` over all ten, asserting the exact metadata key set and `filter.n_max is None` |
| 3 | `graph_ids` match the census element-wise for the four datasets | **yes** | `test_graph_ids_reproduce_the_suite1_census[4]`; 1,180 / 1,253 / 2,059 / 89 ids equal in order |
| 4 | `coil_del` is 3,900 and exactly 100 x 39, both asserted | **yes** | `assert_coil_del_balance` runs inside `build_exported`; `test_coil_del_split_index_is_balanced_and_the_directory_is_not` shows both enumerations side by side |
| 5 | Sampler emits <= 28,000 pairs, reproducible from seed 42, per-bin counts recorded, valid `i < j` | **yes** | Exactly 28,000; digest equal across two independent runs; `n_per_bin` and `bin_population` in metadata; `test_real_draw_indexes_into_the_exported_graph_order`. **Caveat:** reproducible over array content, not file bytes — see §10 |
| 6 | Tests cover all ten listed behaviours | **yes** | Ten counts; COIL-DEL index-vs-directory; order determinism; label population; `n_max`-free filter; four-dataset census; subsample reproducibility; boundaries at n = 4, 12, 30, 98; a bin under 2,000 (fixture) |
| 7 | Real-data verification in the log with numbers | **yes** | Tables A-D: raw, kept, pairs, n max, edges, build seconds, file bytes, per-bin realised counts, peak RSS |
| 8 | All work committed, tree clean, log committed | **yes** | Six commits; `git status` clean at the final commit |

**Second assignment (2026-08-13), against its own brief:**

| # | Criterion | Met | Evidence |
|---|---|---|---|
| A1 | `probe_pairs.npz`: 3,000 pairs, seed 42, same file conventions as `subsample_pairs.npz` | **yes** | Table E; identical key set and dtypes, asserted by `test_write_probe_emits_the_same_conventions_as_the_subsample` |
| A2 | Per-dataset `pair_lists` so the runner consumes it unchanged | **yes** | Ten `probe_pair_lists/{key}.npz`, ascending `pair_index` int64, 30,381 B |
| A3 | Stratified on **both** axes; every dataset and every bin represented | **yes** | 10/10 datasets, 14/14 bins, `n` spanning 2-98; `test_real_probe_is_3000_pairs_over_all_ten_datasets_and_all_fourteen_bins` |
| A4 | Favour spanning `n` where the axes conflict; say which rule and why | **yes** | Equal per bin, then equal per dataset within a bin; recorded in `allocation_rule` / `allocation_rationale` in the file metadata and in Table E |
| A5 | State whether disjoint from the §5 subsample | **yes** | **Disjoint by construction**, `_overlap == 0` asserted in the CLI and in two tests |
| B1 | `bin_table.json` with `bin_edges`, `totals`, `datasets` | **yes** | Table F; `bin_edges` byte-identical to CONTRACTS §5, `totals` sums to 21,710,892 |
| B2 | Per-bin dominant-dataset share in the metadata | **yes** | `dominance` block for all 14 bins with `dominant_dataset`, `dominant_share`, `n_datasets_present`, `single_dataset`; `caveat` string on bin 13 |
| B3 | Same for `[60, 80)` if one dataset dominates | **yes** | Reported at 62.1 % `mutagenicity` and deliberately **not** flagged — the 0.90 threshold has to mean something |

**Overall.** I am confident in the cohort. All ten counts reproduced on the first attempt before any
code was written, and the four-dataset element-wise census check is an external, exact confirmation
of both the loader and the export order against a record that predates this ticket — that is the
strongest evidence available here and it passes.

What the orchestrator should scrutinise first is the `<U1` truncation, not because it is unfixed but
because of what it says about the class of defect in play. It was invisible to reasoning, invisible
to a fabricated fixture, and produced a well-formed file that named datasets `'i'` and `'a'`. It
surfaced only because I ran on the real registry, where three keys share a first letter. **The same
category of silent, well-formed corruption is exactly what the ten asserted counts exist to catch**,
and it is worth assuming one more instance is still hiding somewhere in this wave.

Two things I am less confident about, both recorded rather than smoothed over. First, on a machine
without the Sandisk tree this work is barely tested — every real-data test skips, and that is most of
the value. Second, the `min(2000, population)` branch is never exercised in production, so its only
evidence is a fixture. Neither affects the numbers being shipped today.

**On the second assignment**, the thing to scrutinise is not the probe's mechanics but its
*interpretation*. The draw is deliberately non-representative: equal-per-bin over-weights large `n`
by roughly 22-fold against the cohort, which is correct for fitting a cost curve and wrong for every
pooled statistic. If the launcher takes the probe's mean per-pair rate and multiplies it by
21,710,892 it will over-size the jobs by a large factor — the mirror image of the bias the
contiguous-`grec` fallback would have produced. The metadata says `not_a_cohort_estimate: true` and
`bin_table.json` exists precisely so the rate is integrated per bin instead, but that is a
convention a consumer has to honour, and no file format can enforce it. Worth one line in
`wave-t05-slurm`'s review.
