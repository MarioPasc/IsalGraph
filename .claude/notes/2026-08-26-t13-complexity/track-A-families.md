# track-A-families — wave `2026-08-26-t13-complexity`

Branch `worktree-agent-aacf1cc5eeaa3b6bd`, base `1ace4f9`.
Environment: `~/.conda/envs/isalgraph-cpp/bin/python` 3.11.15, `isalgraph.engine() == "cpp"`,
`build_info()["build_hash"] == "298fc1188bf1b051"`, networkx 3.6.1, pynauty 2.8.8.1.

> The CONTRACTS header names base `10eae30`; my worktree HEAD is `1ace4f9`, which is what the
> brief specifies and what `git rev-parse` reports. Assumed `1ace4f9` is `10eae30` plus the
> wave notes; nothing I built depends on the difference.

---

## Mission

Build the constructed graph families with closed-form `|Aut|`, and the symmetry /
partition-resolution toolkit the rest of T-13 rests on.

The families exist because the cost law cannot be established observationally. On the IAM cohort
`n`, `m`, density and `|Aut|` co-vary, so marginal ρ(log|Aut|, log t) is only +0.189 against +0.326
for log n, and only a within-fixed-`(n, m)` contrast recovers the effect (+0.655). The controlled
experiment is therefore the primary evidence, and these are its graphs.

The toolkit exists because `corrections.md` §5 item 4 and `decisions.md` §17 are wrong: the triplet
pruning key is **not** provably coarser than 1-WL, and the "2.4–2.6× fewer classes" figure is not a
cohort figure. `symmetry.py` is the instrument that establishes the replacement claim —
**Proposition 1**: any node invariant induces a partition coarser than the orbit partition of
`Aut(G)`, so there is no headroom for a finer pruning invariant at all.

---

## What I built

### `benchmarks/real_data/eval_t13_complexity/__init__.py`
Package docstring; exports nothing on purpose, so a missing peer module fails at the call site
rather than at package import.

### `benchmarks/real_data/eval_t13_complexity/symmetry.py`
Everything in CONTRACTS §2, plus three additions.

- `log10_aut(g) -> float` — `log10(mantissa) + exponent` from a single `pynauty.autgrp` call.
- `orbits(g) -> dict[Hashable, int]` — dense ids from 0, in first-appearance order over
  `g.nodes()`, so arbitrary node labels survive.
- `wl_partition(g, *, rounds=None)` — 1-WL to stability by default. Uniform initial colouring;
  each round's signature is `(old_colour, sorted(neighbour_colours))`, recompressed by sorting the
  signature set, which keeps the ids isomorphism-invariant. Stability is detected by class count,
  which is sound because refinement is monotone (a round that adds no class cannot have changed
  the partition). No `grakel`, no `numpy` — the arm has to run on Picasso under numpy 2.
- `triplet_partition(g) -> dict[Hashable, tuple[int, int, int]]` — the incumbent key
  `(|N_1|, |N_2|, |N_3|)`, a `networkx`-side mirror of
  `isalgraph.core.canonical_pruned._bfs_distance_counts` including its early stop and its
  treatment of unreachable vertices.
- `refines(fine, coarse) -> bool` — exact class containment. Raises `ValueError` on mismatched key
  sets, because a refinement relation between partitions of different sets is undefined and
  silently intersecting the domains would let a truncated partition pass.
- `resolution_record(g) -> dict[str, object]` — exactly the nine frozen keys, one `autgrp` call.
- **Added** `RESOLUTION_FIELDS` — the nine names as a tuple, so `schema.py` can assert against it
  instead of restating them.
- **Added** `witness_prism_k33()` — the 3-prism spliced to `K_{3,3}` (criterion 6).
- **Added** `witness_incomparable()` — see defect 3 below; criterion 5 needs it.

The module docstring carries the statement and proof of Proposition 1.

`_autgrp` deliberately does not reuse `competitors.backends.nauty.automorphism_group_size` /
`automorphism_orbits`: those run nauty twice for what T-13 always needs together, and the former
forms the forbidden product — measured, it **raises `OverflowError`** on `K_200` rather than
returning `inf`.

### `benchmarks/real_data/eval_t13_complexity/families.py`
`FAMILIES`, `FamilySpec`, `FamilyVerificationError`, `build`, `enumerate_grid` per CONTRACTS §3,
plus `SIZES`, `LADDER_BASES`, `LADDER_SWAPS`, `SPIDER_CELLS`, `AUT_TOLERANCE`, `spider_legs`,
`spider_rungs` and `ladder_spans`.

**`spider_ladder` is an eleventh family, added mid-wave on the orchestrator's instruction** to give
the matched design a tree-density arm — see "The sparse matched design" below.

`build` verifies, on every call: realised order equals `spec.n`; connectivity; the closed form
within `AUT_TOLERANCE = 1e-6`; rigidity for `rigid_er`; and, for a ladder rung, that `m` **and the
whole degree sequence** match the rung-0 base. Any failure raises `FamilyVerificationError`.

Determinism: every random draw is seeded from a BLAKE2b digest of
`(seed, family, n, replicate, params)`, so `build` is a pure function of its arguments and a shard
can rebuild any graph from its spec alone. Every graph is returned on `range(n)`.

### `tests/test_symmetry.py` (47 tests), `tests/test_families.py` (43 tests)
Includes the four load-bearing ones: the `K_200` overflow rule, the 600-graph triplet parity, the
2,000-graph Proposition 1 property test, and the count-vs-containment separation.

---

## Acceptance criteria

| # | Criterion | Command | Result |
|---|---|---|---|
| 1 | every CONTRACTS §2 signature present | `$PY -c "from benchmarks.real_data.eval_t13_complexity import symmetry as s; print(sorted(n for n in dir(s) if not n.startswith('_')))"` | `['H', 'Hashable', 'Mapping', 'ModuleType', 'RESOLUTION_FIELDS', 'TRIPLET_RADIUS', 'TypeVar', 'annotations', 'deque', 'log10_aut', 'math', 'nx', 'orbits', 'refines', 'resolution_record', 'triplet_partition', 'witness_incomparable', 'witness_prism_k33', 'wl_partition']` — all six contract functions present. **PASS** (path deviation: the `benchmarks.eval_t13_complexity` symlink is track C's to create and does not exist on my branch) |
| 2 | `log10_aut` is `log10(m) + e`, finite on `K_200` | `pytest -k "k200 or product_form"` | 2 passed. `log10_aut(K_200) = 374.8969…`, matches `log10(200!)` to 1e-6. The product form **raises `OverflowError`** — so does the public `automorphism_group_size`. **PASS** |
| 3 | triplet parity, ≥ 500 graphs, 0 disagreements | standalone sweep + `pytest -k triplet` | **2,320 graphs (1,637 connected, 683 disconnected), 0 disagreements** against `canonical_pruned.compute_structural_triplets`; the in-suite test fixes 600. Also 50-graph parity against `_native`. **PASS** |
| 4 | Proposition 1 gate | standalone sweep + `pytest -k proposition_1` | **5,000 random connected graphs, 3 ≤ n ≤ 14, 0 violations**, in 0.5 s; plus all **664 constructed graphs, 0 violations**. In-suite test fixes 2,000. Nothing to stop-and-ask about. **PASS** |
| 5 | `refines` is exact containment, and counts are not interchangeable | `pytest -k class_counts_are_not` | 1 passed. See defect 3 — the specified witness does not separate them; `witness_incomparable()` does. **PASS, on a different graph** |
| 6 | the prism/`K_{3,3}` witness reproduces | `pytest -k witness_prism_k33` | 1 passed. Connected, 3-regular, n=12, m=18, `n_wl_classes == 1`, `n_triplet_classes == 4`, `wl_refines_triplet is False`, `triplet_refines_wl is True`. **PASS** |
| 7 | closed forms verified inside `build` | `pytest -k "closed_form or every_grid_spec or caterpillar or prism_at_a4"` | all passed; all 664 grid cells build and self-verify in ~1.3 s. **PASS**, with the prism formula corrected (defect 2) |
| 8 | ladder holds `n` and `m` exactly | `pytest -k "holds_n_and_m or degree_sequence or non_monotone"` | 4 passed. `len({(g.number_of_nodes(), g.number_of_edges()) for g in ladder}) == 1`, every rung connected, and the **degree sequence** is constant too. All seven swap counts survive to the grid at every ladder — non-monotone rungs kept. **PASS** |
| 9 | `enumerate_grid` deduplicated and deterministic | `$PY -c "..."` with `sizes=(8,10,12,14,16,20,24,28,32,40,48,64)`, `replicates=5`, `seed=13` | `len = 664`; `path=12 cycle=12 star=12 complete=12 complete_bipartite=12 hypercube=4 prism=12 caterpillar=12 rigid_er=60 symmetry_ladder=496 spider_ladder=20`; deterministic `True`, deduplicated `True`. **PASS** |
| 10 | tests pass | `$PY -m pytest benchmarks/real_data/eval_t13_complexity/tests/ -q` | `90 passed in 5.34s` **PASS** |
| 11 | ruff and mypy clean | `$PY -m ruff check --fix benchmarks/real_data/eval_t13_complexity/` | `All checks passed!` **PASS** |
| 11 | | `$PY -m mypy --explicit-package-bases benchmarks/real_data/eval_t13_complexity/` | `Success: no issues found in 6 source files` **PASS, flag required** — see defect 4 |

---

## Measurements

**Grid size: 664 specs.** `symmetry_ladder` is 496 of them and `spider_ladder` 20, so the matched
designs are 78 %. That is proportionate: §3 rule 7 makes the within-`(n, m)` contrast the primary
analysis and everything else supporting evidence.

Per family: `path=12 cycle=12 star=12 complete=12 complete_bipartite=12 hypercube=4 prism=12
caterpillar=12 rigid_er=60 symmetry_ladder=496 spider_ladder=20`.

**Triplet parity: 2,320 graphs, 0 disagreements.** 1,637 connected and 683 disconnected `G(n, p)`
draws with `2 ≤ n ≤ 16`, plus every graph of a 2-replicate grid. Disconnected draws are in on
purpose: they have vertices the BFS never reaches, and the two implementations must agree on *not*
counting them, not merely agree where everything is reachable.

**Proposition 1: 5,000 random connected graphs + all 664 constructed graphs, 0 violations.**

### Ladder spans — realised `log10|Aut|` at every rung

Minimum over the five replicates at each rung (the swaps are a random search for asymmetry, so the
lowest `|Aut|` reached is what the ladder attains).

| n | base | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | span | >=3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | `complete_bipartite` | 3.061 | 1.204 | 0.602 | 0.602 | 0.602 | 0.602 | 0.602 | **2.459** | **no** |
| 10 | `complete_bipartite` | 4.459 | 2.158 | 0.602 | 0.602 | 0.000 | 0.000 | 0.000 | **4.459** | yes |
| 12 | `complete_bipartite` | 6.016 | 3.362 | 1.380 | 0.000 | 0.000 | 0.000 | 0.000 | **6.016** | yes |
| 14 | `complete_bipartite` | 7.706 | 4.760 | 2.459 | 0.000 | 0.000 | 0.000 | 0.000 | **7.706** | yes |
| 16 | `complete_bipartite` | 9.512 | 6.317 | 3.760 | 0.602 | 0.000 | 0.000 | 0.000 | **9.512** | yes |
| 20 | `complete_bipartite` | 13.421 | 9.813 | 6.861 | 2.158 | 0.000 | 0.000 | 0.000 | **13.421** | yes |
| 24 | `complete_bipartite` | 17.662 | 13.722 | 10.415 | 4.937 | 0.301 | 0.000 | 0.000 | **17.662** | yes |
| 28 | `complete_bipartite` | 22.182 | 17.963 | 14.324 | 7.764 | 1.204 | 0.000 | 0.000 | **22.182** | yes |
| 32 | `complete_bipartite` | 26.942 | 22.483 | 18.565 | 11.369 | 3.158 | 0.602 | 0.000 | **26.942** | yes |
| 40 | `complete_bipartite` | 37.073 | 32.215 | 27.845 | 20.492 | 9.211 | 1.380 | 0.000 | **37.073** | yes |
| 48 | `complete_bipartite` | 47.886 | 42.704 | 37.976 | 29.076 | 15.541 | 3.459 | 0.000 | **47.886** | yes |
| 64 | `complete_bipartite` | 71.141 | 65.449 | 60.172 | 50.187 | 33.795 | 14.462 | 0.903 | **70.238** | yes |
| 8 | `hypercube` | 1.681 | 0.602 | 0.602 | 0.602 | 0.602 | 0.602 | 0.602 | **1.079** | **no** |
| 16 | `hypercube` | 2.584 | 0.301 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **2.584** | **no** |
| 32 | `hypercube` | 3.584 | 0.602 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **3.584** | yes |
| 64 | `hypercube` | 4.664 | 0.903 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **4.664** | yes |

**13 of 16 ladders clear three orders of magnitude.** The three that do not are `(8, K_{4,4})`,
`(8, Q_3)` and `(16, Q_4)`, and the cause is arithmetic rather than a defect: their rung-0 groups
are 1152, 48 and 384, so three orders were never available at those orders. They are kept — the
low-`|Aut|` end is still a valid fixed-`(n, m)` contrast — but a slope must not be read off them
alone.

### `spider_ladder` — the sparse matched design

Added mid-wave on the orchestrator's instruction, because the two `symmetry_ladder` bases with
factorial groups are dense (`K_{32,32}` at n=64 has m=1024, density 0.508) or medium (`Q_6` at n=64
has m=192, density 0.095), and a cost law demonstrated only there would not transfer to the sparse
IAM cohort (mean density 0.094–0.607).

A spider is a hub with `k >= 3` disjoint paths of lengths `L_1..L_k` summing to `n-1`. It is the
sparsest connected graph there is and still reaches a factorial group, which no regular base does at
these orders. Rung `j` lengthens the first `j` legs to `L+j, …, L+1` and shortens the next `j` to
`L-1, …, L-j`; the displacement is antisymmetric, so:

- `n = 1 + sum L_i` is fixed **by construction**, not by a check;
- `m = n - 1`, because every spider is a tree;
- the degree sequence is `(k, 2^(n-1-k), 1^k)` at every rung — the hub always has degree `k`, there
  are always `k` leaves, and `sum_i (L_i - 1) = (n-1) - k` does not depend on the partition.

So the same three confounds are held as by `double_edge_swap`, but **proven by construction rather
than by the operation**, and at tree density.

`|Aut| = prod_d (m_d)!` where `m_d` counts the legs of length `d`: for `k >= 3` the hub is the unique
vertex of degree `>= 3` and is fixed by every automorphism, and each leg is rigid once the hub is, so
an automorphism is exactly a length-preserving permutation of legs. Rung `j` leaves `k - 2j` legs
equal, so `|Aut| = (k - 2j)!`. **This is a closed form, so `build` verifies this ladder the same way
it verifies `complete`** — unlike `symmetry_ladder`, whose `|Aut|` can only be measured. Confirmed
against nauty on all 25 rungs of seven `(k, leg)` cells, zero mismatches.

`k = 2` is excluded: the hub would have degree 2, the spider is a path, and `|Aut| = 2` however
unequal the legs are. `spider_legs` raises on it.

| n | m | density | k | j=0 | j=1 | j=2 | j=3 | j=4 | j=5 | span | >=3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 9 | 0.2000 | 3 | 0.778 | 0.000 | — | — | — | — | **0.778** | **no** (gate cell) |
| 31 | 30 | 0.0645 | 10 | 6.560 | 4.606 | 2.857 | — | — | — | **3.702** | yes |
| 33 | 32 | 0.0606 | 8 | 4.606 | 2.857 | 1.380 | 0.301 | — | — | **4.304** | yes |
| 61 | 60 | 0.0328 | 10 | 6.560 | 4.606 | 2.857 | 1.380 | 0.301 | 0.000 | **6.560** | yes |
| 65 | 64 | 0.0308 | 8 | 4.606 | 2.857 | 1.380 | 0.301 | 0.000 | — | **4.606** | yes |

Four of five clear three orders. `(k=3, leg=3)` at n=10 spans only `log10(3!) = 0.778` and is not
meant to: it is the `n <= 12` consistency gate, where exhaustive search and exact GED are both
tractable and the arms can be cross-checked against each other. The rung count is
`min(k // 2, leg - 1) + 1`; `(8, 4)` and `(10, 3)` stop short of rigid because a further rung would
drive a leg to length 0.

**Spider orders are `k * leg + 1` and therefore do not coincide with `SIZES`** (they are 10, 31, 33,
61, 65). `SPIDER_CELLS` is a fixed design, not a size sweep: snapping to a requested order would have
to move `k`, which changes the span (`log10(k!)`), or `leg`, which changes the density. Neither is a
free parameter. `enumerate_grid` therefore emits these five cells whatever `sizes` it is given, and
says so in its docstring.

**Net effect: three ladders at matched `n` and three separated densities** — tree (0.031–0.20), `Q_d`
(0.095–0.161), `K_{a,a}` (0.508 at n=64). That turns "is the effect really density?" from an
objection into a measured answer.

Two facts the analysis should know. **The ladders saturate fast**: `K_{a,a}` reaches rigid by
k = 4–16 and `Q_d` by k = 2, so the informative rungs are k ∈ {0, 1, 2, 4} and k ∈ {8, 16, 32} are
mostly `|Aut| = 1` duplicates. They are kept because a flat tail at `|Aut| = 1` is itself the
prediction under test (time should stop falling once symmetry is exhausted). **`(8, K_{4,4})`
floors at `log10|Aut| = 0.602`, not 0** — 32 swaps never found a rigid connected 4-regular graph on
8 vertices. Reported as measured; I did not search for whether one exists.

---

## Decisions and assumptions

**Ladder bases: `K_{a,a}` and `Q_d` only; prism and circulant rejected.** The orchestrator's
correction requires a `d`-regular base with `d ≥ 3` and a span of three orders. For any *sparse*
regular base `|Aut|` is polynomial in `n` — the prism's is exactly `4a`, the circulant `C_n(1, k)`'s
is at least `2n` — so `log10|Aut| ≤ 2.2` at `n ≤ 64` and the ladder cannot fall three orders however
many swaps it takes. Only `K_{a,a}` (`2(a!)²`) and `Q_d` (`2^d d!`) have factorial groups. `Q_d` is
kept despite two short rungs because it is the sparse counterweight: `Q_6` at n = 64 has m = 192
against `K_{32,32}`'s 1024, so if the exhaustive arm censors on the dense ladder there is still a
tractable matched design at the top sizes.

**`spider_ladder` cells.** The orchestrator's four suggested cells `(8,4)`, `(8,8)`, `(10,3)`,
`(10,6)`, plus `(3,3)` at n=10 as the `n <= 12` consistency gate — chosen at n=10 rather than an
arbitrary small order so the gate is directly comparable with the other families' n=10 cells.

**Rung set derived, not fixed.** `spider_rungs(k, leg) = range(min(k // 2, leg - 1) + 1)`: `2j <= k`
because rung `j` needs `2j` legs to displace, and `j <= leg - 1` because a leg may not reach length
0. That is why `(8,4)` has four rungs and stops at `|Aut| = 2` while `(8,8)` has five and reaches 1.

**Swap procedure.** One `nx.double_edge_swap` at a time onto a copy, accepted only if the result is
still connected, up to 200 attempts per rung. `nx.connected_double_edge_swap` would also work but
gives less control over the per-rung count; the contract names `double_edge_swap` explicitly.

**Rung 0 forced to `replicate = 0`.** It has no random component, so all five replicates are the
same graph and de-duplication collapses them. Without this the grid would carry four duplicate
rows per ladder.

**`params` values are ints**, as the declared type requires, so the ladder base is
`("base", index_into_LADDER_BASES)` rather than a string. `LADDER_BASES` is public for decoding.

**`FamilySpec` has no `n_target`.** The realised order is the identity of the graph; carrying the
request would defeat de-duplication (12 requested sizes collapse to 4 hypercubes). Track C should
set the schema's `n_target` from `spec.n`. Flagged to `main`.

**`rigid_er` uses `p = 0.5`,** the maximum-entropy draw at fixed `n`, with a 50-draw resampling
budget. Every grid cell found a connected rigid draw; the budget was never close to exhausted.

**Caterpillar construction.** Spine of `n − 2k` vertices with two leaves on each of the first
`k = n // 4`. That keeps `0 < k < spine`, which makes the leaf-count sequence non-palindromic and
kills the mirror factor, so `|Aut| = 2^k` exactly. Verified against nauty at all twelve sizes.
The contract's "k = number of degree-1 pairs sharing a spine node" is only correct under exactly
this kind of restriction — a general caterpillar picks up a factor 2 when its leaf sequence is a
palindrome.

**`star` excludes n = 2.** `K_{1,1} = K_2` has `|Aut| = 2`, not `1! = 1`. The grid starts at n = 8
so nothing is lost.

**Absolute intra-package imports** (`from benchmarks.real_data.eval_t13_complexity import
symmetry`), matching `eval_t06_figures`. Note the consequence: importing this package through the
`benchmarks/eval_t13_complexity` symlink creates a *second* module object for `families` while
`symmetry` stays canonical, so `isinstance` checks on `FamilySpec` would fail across the two paths.
Track C should import through one path consistently. The repo already has this property; I matched
the convention rather than diverging alone.

---

## Defects found in the brief

**1. `isalgraph.compute_structural_triplets` does not exist** (CONTRACTS §2). Not at top level, not
in `isalgraph.core.backends`. The Python reference is
`isalgraph.core.canonical_pruned.compute_structural_triplets` and takes a `SparseGraph`. The native
one is `isalgraph.core._native.compute_structural_triplets` and takes **five marshalled arguments**
(`node_count, max_nodes, directed, logical_edge_count, adjacency`) — use
`isalgraph.core.backends._marshal`. Both are now covered by tests. Track B will hit this.

**2. The prism exception is backwards** (CONTRACTS §3). The contract says `a >= 4; a = 3 is
K_{3,3}-adjacent, handle or exclude`. Measured with pynauty:

| a | n | measured `log10\|Aut\|` | `log10(4a)` | |
|---|---|---|---|---|
| 3 | 6 | 1.079181 | 1.079181 | formula holds |
| **4** | **8** | **1.681241** (= 48) | 1.204120 (= 16) | **mismatch** |
| 5..10 | | matches `4a` | | |

The 4-prism **is** the cube `Q_3`, hence `|Aut| = 2³·3! = 48`. Following the contract literally —
excluding a = 3, keeping a = 4 under `4a` — would have aborted the campaign at n = 8 with a
`FamilyVerificationError`. Both are kept; a = 4 gets the hypercube formula. Pinned by
`test_prism_at_a4_is_the_cube`.

**3. Criterion 5's witness does not separate the two tests.** On the 3-prism/`K_{3,3}` graph
`n_wl = 1` and `n_triplet = 4`, so the count rule `|P| >= |Q| ⇒ P refines Q` reports
`wl_refines_triplet = False` and `triplet_refines_wl = True` — which is the truth. The count rule
is right there, by accident.

The separation needs a genuinely *incomparable* pair at counts that suggest refinement. Found by a
seeded sweep and frozen as `symmetry.witness_incomparable()`: connected, n = 9, m = 15, edges
`(0,2)(0,3)(0,8)(1,2)(1,5)(1,8)(2,5)(2,7)(3,6)(3,7)(4,5)(4,7)(4,8)(6,7)(6,8)`. Both partitions have
**4 classes** and **neither refines the other**, so every count rule reports refinement in both
directions and is wrong in both. That is the substitution `corrections.md` §5 made.

It also strengthens §1.3: on the cohort the two were never incomparable (0/250), on the prism
witness they are ordered the *opposite* way from the plan's claim, and here they are genuinely
unordered — so no orientation of the claim survives.

**4. The mypy acceptance command cannot pass, for anyone.**
`$PY -m mypy benchmarks/real_data/eval_t13_complexity/` fails with *"Source file found twice under
different module names"*, because `benchmarks/` and `benchmarks/real_data/` are namespace packages
with no `__init__.py`. This is **pre-existing and repo-wide**: the identical command on the
untouched `benchmarks/real_data/eval_t06_figures/` fails the same way. `--explicit-package-bases`
fixes it and my package is clean under `strict = true`. Tracks B and C need the flag.

**5. `symmetry_ladder`'s `replicates = 5` is under-specified.** The contract gives a replicate count
but not a rung count, and a ladder is a *sequence*. I read it as sizes × bases × rungs × replicates
with rung 0 collapsed, which is what produces 496 ladder specs. If the orchestrator wanted 5 whole
ladders and fewer rungs, the grid shrinks; `LADDER_SWAPS` is a one-line change.

---

## What I did not do

- **Nothing under `src/isalgraph/`.** Not one line. `git diff --stat 1ace4f9` touches only
  `benchmarks/real_data/eval_t13_complexity/` and this log.
- **Did not create the `benchmarks/eval_t13_complexity` symlink** — track C owns it. My acceptance
  commands therefore use `benchmarks.real_data.eval_t13_complexity`.
- **Did not touch** `instrumented.py`, `counters.py`, `schema.py`, `measure.py`, `slurm/`, any plan
  file, the ticket board, or `CONTRACTS.md`. The defects above are reported, not patched.
- **Did not run the repository suite** (`tests/`). `testpaths = ["tests"]` and I added nothing
  there, so the 2,618 / 321 reference figure is untouched by construction.
- **No cluster access, no network, no new dependency.** `networkx` and `pynauty` only.
- **Did not search for whether a rigid connected 4-regular graph on 8 vertices exists.** The
  `(8, K_{4,4})` ladder floors at `log10|Aut| = 0.602` and I report that as measured rather than
  claiming it is the true minimum.
- **Did not cut rungs `k = 8, 16, 32`** from `symmetry_ladder`, per the orchestrator: they anchor
  the flat end, and the decision to drop them must not be made after seeing timings.
- **Did not tune the grid for cost.** 664 specs × 13 representations ≈ 8,600 measurements is the
  design as specified; trimming is the orchestrator's call, and `LADDER_SWAPS` / `LADDER_BASES` are
  where to do it.
