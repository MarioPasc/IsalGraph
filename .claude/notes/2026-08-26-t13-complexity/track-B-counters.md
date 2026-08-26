# track-B-counters — wave `2026-08-26-t13-complexity`

Branch `worktree-agent-af43748d6b39c8a02`, base `1ace4f9`.

## Mission

Build an instrumented mirror of the frozen IsalGraph encoder that counts the four operations
Reviewer 3's comment R3.4b asks to see costed — pair scanning, pointer walking, neighbour checks
and canonical backtracking — and prove it emits byte-identical output to the pure-Python
reference. The counts are what turn the T-13 §2.1 derivation from an assertion into a validated
one. Timings are not mine; the orchestrator runs those on the cluster.

`src/isalgraph/` is untouched: `git diff --name-only 1ace4f9..HEAD -- src/ | wc -l` → `0`.

---

## What I built

### `benchmarks/real_data/eval_t13_complexity/instrumented.py`

A line-by-line transcription of the three frozen encoders with counter increments interleaved —
the same device `viz/encoder_trace.py` used for T-09. It shares every data structure and helper
with the reference (`SparseGraph`, `CircularDoublyLinkedList`, `generate_pairs_sorted_by_sum`,
`_primary_moves`, `_secondary_moves`, `_undo_edge`, `_undo_node`, `_is_reachable`,
`compute_structural_triplets`) rather than re-deriving them, so the only difference between mirror
and reference is the counting.

Public surface:

| symbol | role |
|---|---|
| `OperationCounts` | frozen-slots dataclass, exactly the nine fields of CONTRACTS §4 |
| `FrameRecord` | per-frame detail (`pair_scope`, `pairs_generated`, `pair_trials`, `pointer_steps`, `neighbour_checks`, `opcode`, `a`, `b`, `disp_emitted`, `n_cands`, `branch_factor`, `depth`, `start_node`) — the material the derivation checks need; not part of the frozen contract |
| `greedy_counts(g, start)` | mirrors `GraphToString(g).run(start)[0]` |
| `canonical_counts(g)` | mirrors `canonical.canonical_string(g)` |
| `pruned_counts(g)` | mirrors `canonical_pruned.pruned_canonical_string(g)` |
| `greedy_min_counts(g)` | mirrors `GreedyMinG2S` — the registered `isalgraph_greedy` unit of work |
| `greedy_detail` / `canonical_detail` / `pruned_detail` | as above, plus the `FrameRecord` list |
| `pair_generation_work(m)` | `(pairs, sort_comparisons, P·log₂P)` for one `generate_pairs_sorted_by_sum(m)` call |
| `InstrumentationError` | raised where the reference raises `RuntimeError` |

The exhaustive and pruned searches share one driver, `_step_counted`, selected by
`ctx.triplets is None` — the two frozen functions differ only in the triplet filter applied to the
candidate list, and duplicating 150 lines to mirror that would have been a second place for the
mirror to drift.

**Counting conventions**, fixed in the module docstring so they cannot drift:

- `frames` — payload instructions emitted. Greedy: one per `while` iteration. Canonical arms:
  summed over the whole search tree and every reachable start, and equal to
  `backtrack_nodes − search_leaves`.
- `pair_trials` / `scan_depth_total` — iterations of `for (a, b) in pairs`, accepted pair
  included. `scan_depth_max` is the per-frame maximum of the same quantity.
- `pointer_steps` — executions of the loop body inside `_move_pointer` / `_walk`. A trial of
  `(a, b)` costs `|a|` always and `|b|` **only when control reaches the secondary walk**: a `V`
  acceptance returns before the secondary pointer moves, and the count reflects that.
- `neighbour_checks` — greedy's `_find_new_neighbor` short-circuits, so one check per neighbour
  *examined*; the canonical arms materialise the candidate list, so one check per neighbour of the
  node. The `C`/`c` guards contribute one check per membership test actually evaluated, so the
  count short-circuits exactly as Python's `and` does.
- `backtrack_nodes` — `_step` invocations entered, roots and terminals included. `0` for greedy.
- `search_leaves` — invocations reaching the terminal `nleft <= 0 and eleft <= 0` branch. `0` for
  greedy, which produces its string without a search.

### `benchmarks/real_data/eval_t13_complexity/counters.py`

The `t13c.1` CLI. Emits one JSONL row per `(graph, encoder)` with exactly the twenty fields
CONTRACTS §4 names, `parity_ok` computed per row against the pure-Python reference, and a non-zero
exit status if any row fails.

`--spec-file` rows carry the graph **explicitly** (`n`, `edges`) plus optional provenance
(`source`, `family`, `n_target`, `replicate`, `dataset`, `graph_index`, `directed`), copied verbatim
into the output. That keeps this module independent of track A's `families.py`, which the brief
forbids me to import. **Approved by the orchestrator**, who generates the spec file at campaign time
from `families.enumerate_grid` / `families.build` and passes provenance through the optional fields.

`--self-test K` replaces the spec file with a deterministic pool, so the CLI is runnable standalone.
`random_connected_graphs(...)` and `to_sparse(...)` are exported for the tests and the offline
sweep.

**`encoder` takes four values, not three** — a CONTRACTS §4 amendment made by the orchestrator on my
report of the hazard:

| value | object priced | frame accounting |
|---|---|---|
| `greedy_single` | one greedy encode from node `0` | `frames == m` |
| `greedy_min` | the whole `GreedyMinG2S` unit — one encode per start node, lexmin shortest kept; what the registered `isalgraph_greedy` arm times | `frames == n · m` |
| `canonical` | `canonical_string` | summed over the search tree |
| `pruned` | `pruned_canonical_string` | summed over the search tree |

The two greedy rows price different objects, so the distinction travels **in the data**. A single
`greedy` value plus an invocation-time flag was the earlier design and was a trap: a consumer
asserting `frames == m` on the wrong row would have got a wrong answer with no error.
`--greedy-mode {min,single,both}` selects which greedy rows are emitted; default `greedy_min`,
because counts and timings must price the same object.

### `benchmarks/real_data/eval_t13_complexity/tests/test_instrumented.py`

21 tests: parity for all four entry points, the CONTRACTS §4 structural invariants, the §2.1
derivation checks, and two CLI round-trips.

### Stubs I had to create (expect a trivial merge conflict)

`benchmarks/real_data/eval_t13_complexity/__init__.py` and `tests/__init__.py` are **track A's**.
They did not exist on my branch, so I created one-line docstring stubs that export nothing, as the
brief instructs. Take track A's version on merge.

---

## Acceptance criteria

| # | criterion | verdict | command and output |
|---|---|---|---|
| 1 | `OperationCounts`, `greedy_counts`, `canonical_counts`, `pruned_counts` with the CONTRACTS §4 signatures and field names | **PASS** | field set asserted in `test_cli_self_test_emits_the_frozen_schema`; all nine names present, no extras |
| 2 | strings byte-identical to the **pure-Python** reference, imported from `isalgraph.core.*` | **PASS** | `counters.reference_string` imports `GraphToString`, `canonical_string`, `pruned_canonical_string` from `isalgraph.core.{graph_to_string,canonical,canonical_pruned}`; the top-level dispatching package is never imported |
| 3 | ≥ 50,000 (graph, start) parity pairs, 0 mismatches, connected, 2 ≤ n ≤ 12, fixed seed | **PASS** | see *Parity evidence* |
| 4 | structural invariants on every graph in the sweep | **PASS** | see *Parity evidence* |
| 5 | derivation checks (a)–(d) | **PASS with one correction to my own prior** | see *Derivation checks* |
| 6 | CLI emits the `t13c.1` schema with `parity_ok` true in every row | **PASS** | `$PY -m benchmarks.eval_t13_complexity.counters --help` renders (verified through a transient symlink; the symlink itself is track C's file and is **not** in my diff). `--self-test 3 --greedy-mode both` wrote 48 rows across all four encoder values, `0 parity failures`; `greedy_single` satisfies `frames == m` and `greedy_min` satisfies `frames == n·m` on every row |
| 7 | `pytest .../tests/test_instrumented.py -q` | **PASS** | `21 passed in 7.32s` |
| 8 | `ruff check` and `mypy` clean | **PASS** | `All checks passed!` / `Success: no issues found in 5 source files` |

### Criterion 8, the one wrinkle

`$PY -m mypy benchmarks/real_data/eval_t13_complexity/` fails with
`Source file found twice under different module names`. **This is not my defect** — the identical
failure occurs on the already-landed `eval_t06_figures`, because an `__init__.py` inside a
namespace-package tree makes mypy root the package at that directory. The working invocation is:

```
$PY -m mypy --explicit-package-bases benchmarks/real_data/eval_t13_complexity/
Success: no issues found in 5 source files
```

### Full-suite floor

Not re-measured, and it cannot have moved: `pyproject.toml:122` sets `testpaths = ["tests"]`, and
my whole diff is under `benchmarks/`. The 2,618 / 321 figure stands untouched.

---

## Parity evidence

Generator: `counters.random_connected_graphs`, a single `random.Random` stream. `G(n, p)` with
`p ~ U[p_min, p_max]`, rejected unless connected, identity node labels — so the pool exercises the
`set`-slot-order dependence of `_find_new_neighbor` rather than hiding it behind sorted labels.

### Greedy arm — seed 13, sizes 2…12, 660 graphs per size, all densities `p ~ U[0.15, 0.95]`

| quantity | value |
|---|---|
| graphs | 7,260 |
| **(graph, start) parity pairs** | **50,820** |
| **mismatches vs `GraphToString(g).run(v)[0]`** | **0** |
| frames | 1,109,460 |
| `frames == m` | 50,820 / 50,820 |
| wall clock | 119.5 s |

### Canonical and pruned arms — same generator

A `canonical_counts` call loops over every reachable start internally, so one call covers `n`
`(graph, start)` pairs. Density is capped above `n = 7` (`m ≤ n + 3`) because the exhaustive search
is super-exponential in the branching factor — a `K₁₂` canonicalisation does not terminate in any
useful time. **The cap is on the cohort, never on the encoder**, and the greedy arm above covers
`2 ≤ n ≤ 12` at every density.

CANONICAL_TABLE_PLACEHOLDER

---

## Derivation checks

DERIVATION_PLACEHOLDER

---

## Decisions and assumptions

1. **Base commit.** My brief pins `1ace4f9`; `CONTRACTS.md` names `10eae30`. I worked from
   `1ace4f9`, which is what my worktree had at `git rev-parse --short HEAD`. Recorded, not
   escalated — the frozen encoder is identical either way.
2. **`search_leaves = 0` for greedy**, matching CONTRACTS §4's "canonical arms only" and its
   explicit "`0` for greedy" on `backtrack_nodes`. Greedy produces one string without a search;
   calling that "one leaf" would make the greedy and canonical columns mean different things.
   Criterion 4's `backtrack_nodes >= search_leaves >= 1` is asserted for the canonical arms only,
   as the brief specifies.
3. **Two greedy encoder values, `greedy_single` and `greedy_min`.** I first shipped one `greedy`
   value plus a `--greedy-mode` flag, flagged the hazard to the orchestrator, and was told to fix
   it in the data instead. Done: the schema now carries the mode, `frames == m` is assertable
   exactly on `greedy_single` and `frames == n · m` exactly on `greedy_min`, and no consumer has to
   remember an invocation. The default remains `greedy_min` because that is the object the
   registered `isalgraph_greedy` arm times.
4. **Triplet work is not counted.** `compute_structural_triplets` is a fixed `O(n(n+m))`
   preprocessing step and the `max`/filter at each branch is `O(|cands|)`; there is no
   `OperationCounts` field for either, and folding them into `neighbour_checks` would make the
   pruned and unpruned neighbour columns incomparable. Noted in the module docstring.
5. **Private imports from the frozen modules** (`_undo_edge`, `_undo_node`, `_is_reachable`,
   `_primary_moves`, `_secondary_moves`). Read-only use, and it maximises parity fidelity: a
   transcribed copy would be a second place for the mirror to drift.
6. **`_step_counted` serves both canonical arms.** Justified above.
7. **Spec-file format.** See `counters.py` above — the one unilateral interface decision.
8. **`pair_generation_work` measures sort comparisons for real**, by re-sorting the identical key
   sequence through a `__lt__`-counting wrapper. Timsort's comparison count is deterministic given
   the input sequence, so this is a measurement of the frozen sort, not a model of it. It is
   `lru_cache`d and is never called on the hot path.

---

## Lemma (one-sided displacement at insertion frames) — for the article notes

*This heading is written to be lifted verbatim. It is a small original result and it belongs to
T-13.*

**Lemma.** In a G2S encode under the frozen scan order, every frame whose accepted instruction is
`V` has `b = 0`, and every frame whose accepted instruction is `v` has `a = 0`.

*Proof.* Displacement pairs are scanned in the order `(|a| + |b|, |a|, (a, b))`, and the `V` guard —
"the primary tentative node has an uninserted neighbour" — depends on `a` alone. Suppose `V` is
accepted at `(a, b)` with `|b| > 0`. The pair `(a, 0)` has cost `|a| < |a| + |b|` and is therefore
examined strictly earlier, with the same `a` and hence the same `V` guard, so `V` would have fired
there: contradiction. The `v` case is symmetric. `v` is reached only after the `V` guard has failed
at the current pair; the pair `(0, b)` has cost `|b| < |a| + |b|` and is examined strictly earlier,
its `V` guard also fails (otherwise the frame would be a `V` frame), and its `v` guard is the same,
so `v` would have fired there. ∎

**What the lemma does and does not say.** It constrains only frames whose accepted instruction is
`V` or `v`. `C` and `c` frames may still carry two-sided displacement, so the general form

    |w| = m + Σ_f (|a_f| + |b_f|)

**stands unchanged** — the lemma does not correct it. What the lemma buys is a sharper split of that
sum along the frame decomposition of §2.1: the `n − 1` insertion frames cost **one-sided**
displacement, the `m − n + 1` chord frames **two-sided**.

**Measured.** Zero exceptions in 215,270 frames across four arms:

| arm | pool | `V` frames | with `b ≠ 0` | `v` frames | with `a ≠ 0` |
|---|---|---|---|---|---|
| greedy | seed 99, sizes 2…12, 120 graphs/size, 9,240 encodes | 43,942 | **0** | 24,698 | **0** |
| canonical | seed 98, sizes 4…7, 60 graphs/size | 119,666 | **0** | 26,964 | **0** |

And the split is not merely formal — the two frame classes differ by 6.1× in realised cost on the
greedy pool:

| frame class | frames | movement characters emitted | mean per frame |
|---|---|---|---|
| insertion (`V`/`v`), one-sided | 68,640 | 16,761 | **0.244** |
| chord (`C`/`c`), two-sided | 128,879 | 193,272 | **1.500** |

So `|w| = m + Σ_f (|a_f| + |b_f|)` is dominated by the chord frames, and a graph's string length is
driven by its cyclomatic number `m − n + 1` far more than by its node count. That is worth a
sentence next to the `O(mn)` bound.

---

## Defects found in the brief

**One suspected, investigated, and refuted.** Criterion 5(d) of my brief and §2.1 of the design note
state `|w| = m + Σ_f (|a_f| + |b_f|)`. The frozen `V` branch emits only `_emit_primary_moves(a)` and
the `v` branch only `_emit_secondary_moves(b)`, so I expected the formula to over-count on every
`V`/`v` frame, and instrumented both readings (`FrameRecord.disp_emitted` versus `|a| + |b|`) to
prove it. **They agree on all 50,820 greedy encodes**, for the reason proved in the lemma above.
The derivation is right and I was wrong. The lemma is still worth stating in the manuscript,
because without it the formula reads as an error to a careful reviewer — which is exactly what
happened here.

**A second, smaller correction, to my own test rather than to the brief.** I first asserted that
the canonical arms perform *more* neighbour checks per pair trial than greedy, on the strength of
§2.1's "`_find_new_neighbor` (first match, `O(deg)`); canonical materialises all candidates". They
do not, in aggregate: on the same graph, exhaustive canonical ran 2.56 checks per trial against
greedy's 3.12. The design note's claim is about the **per-call** cost and is correct; the aggregate
ratio is confounded, because most canonical frames sit deep in the search tree where `nleft == 0`
and the `V`/`v` scan is skipped entirely. The test now isolates the claim to a single frame
(`test_neighbour_checks_show_the_short_circuit_gap`): on a star encoded from its hub, the opening
`V` frame charges greedy **1** check and canonical **Δ = 6**. Anyone quoting an aggregate
neighbour-check ratio in the manuscript should be aware of the confound.

---

## What I did not do

- No timings of any kind. No `ssh`, `sbatch`, `rsync` or network.
- Nothing written to the repository's `scratchpad/`; the offline sweep script and its JSON report
  live in the session scratchpad only.
- No new third-party dependency (`networkx` is not imported — `random_connected_graphs` builds
  edge lists with the stdlib, which also removes a nondeterminism source).
- No plan file, ticket board or `CONTRACTS.md` edited.
- Did not import `families.py`, `symmetry.py`, `schema.py` or `measure.py`.
- Did not create the `benchmarks/eval_t13_complexity` symlink (track C's). I verified the contract
  CLI path through a transient symlink and removed it; `git status --porcelain` is clean.
- Did not run the full 2,618-test suite — `testpaths = ["tests"]` and my diff is entirely under
  `benchmarks/`, so it cannot have moved.
