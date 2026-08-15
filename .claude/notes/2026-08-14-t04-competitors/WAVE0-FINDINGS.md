# T-04 wave 0 — three refuted premises, measured before any agent started

**Date**: 2026-08-15. **Orchestrator**, branch `ticket/T-04-competitors`, base `7bf266c`.
Probes preserved beside this file; every number below was produced by running them in
`isalgraph-cpp`, not quoted from the plan.

Design note §9 conditions **1** (a criterion-1 number does not reproduce) and **3** (the grakel
identity) both fire. Work is halted at the wave-0 / wave-1 boundary pending PI sign-off.

**None of the three flips a stated conclusion.** min-DFS still beats IsalGraph on all five Suite-1
datasets; the size null still dominates; IsalGraph still clears the null on Letter LOW and MED and
falls below it on the other three. All three change **printed numbers**, which is why they stop the
ticket rather than being absorbed.

---

## Environment, re-measured (design note §0 had two rows wrong)

| Item | Design note §0 | Measured 2026-08-15 | Consequence |
|---|---|---|---|
| `grakel` | **0.1.10**, "the folder recorded 0.1.8" — treated as an environment change | `GraKeL-0.1.10.dist-info` **and** `grakel.__version__ == '0.1.8'` (stale string in `grakel/__init__.py:89`) | **the scout and this ticket are on the same installation.** The "different grakel" hazard in §0 dissolves; the WL defect below is not a version problem |
| `pynauty` | absent from `isalgraph-cpp` | absent — **now installed**, `pynauty==2.8.8.1`, cp311 wheel | ✔ closed |
| data root | `data/exported/<ds>.npz` | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/exported/` | paths in the plan are relative to the scout's `ROOT`, not to the repo |
| exact GED | `.../extended_merged_exact_ged/computed/` | present, 5 files, `graph_ids` align | ✔ |
| `rapidfuzz` 3.14.5, `networkx` 3.6.1, `numpy` 1.26.4, engine `cpp` (`298fc1188bf1b051`) | — | confirmed | ✔ |

---

## W0-1 — finding 12 is wrong: grakel has **no** off-by-one

**Claim under test** (README §5 finding 12; design §8 criterion 1 bullet 3; agent C criterion 2):
`grakel(n_iter=3) ≡ ours(h=2) = 5.830952`.

**Measured on the running example** `G = C₄(0,1,2,3) + K₃(3,4,5)`, `H = G − (0,3)`:

| | `n_iter`/`h` = 1 | 2 | 3 |
|---|---:|---:|---:|
| grakel | 2.000000 | **5.830952** | 7.211103 |
| a shared-vocabulary WL | 2.000000 | **5.830952** | 7.211103 |

`grakel(n_iter=3) = 7.211103`, not 5.830952. **`grakel(n_iter=k) ≡ h = k`.**

**Why, from the source**, not from the output: `grakel/kernels/weisfeiler_lehman.py:109` sets
`self._n_iter = self.n_iter + 1` and the refinement loop is `for i in range(1, self._n_iter)`, i.e.
`k` refinement rounds plus the base histogram at index 0. Independently confirmed by arithmetic:
at `n_iter=1`, `K(G,G) = 62 = 36` (base, six identical labels) `+ 26` (degree histogram `5² + 1²`).

**Where the off-by-one actually lives.** `scratch/backends.py::wl_features` compresses colours to
small integers **per graph, per round** (lines 109–110) and builds round `it+1`'s signature from
those compressed labels. The compression table is built from that one graph's own signature set, so
**features from rounds ≥ 2 are not comparable across graphs.** Its own docstring claims the opposite.
That implementation — not grakel — produced README §4.1's WL row.

**What a correct WL costs**, ρ against certified exact GED, same draw as `real_wl.py`:

| Dataset | README §4.1 (scout WL) | shared-vocabulary WL | grakel `n_iter=2` | Δ |
|---|---:|---:|---:|---:|
| Letter LOW | 0.895 | **0.7792** | 0.7792 | **−0.116** |
| Letter MED | 0.869 | **0.7746** | 0.7746 | **−0.094** |
| Letter HIGH | 0.580 | 0.5674 | 0.5674 | −0.013 |
| LINUX | 0.573 | 0.5665 | 0.5665 | −0.007 |
| AIDS | 0.459 | **0.4714** | 0.4714 | +0.013 |

Two independent implementations agree to four decimals on every dataset and at `h = 2` and `h = 3`.
The scout's column reproduces its own README row exactly, which is how the defect stayed invisible.

**Downstream, unasked-for but load-bearing**: `benchmarks/real_data/eval_setup/wl_kernel_computer.py`
defaults to `n_iter = 5`. Design §0 reads that as `h = 4`. It is **`h = 5`** — *three* refinement
rounds past the frozen `h = 2`, not two. Finding 12's edit for T-06 changes accordingly.

**Unaffected**: the K₃,₃ / prism witness. 1-WL cannot separate two 3-regular graphs on six vertices
under any convention, so `d_WL = 0` stands.

---

## W0-2 — the adjacency reading order: frozen column-wise, measured row-major

The design freezes (§4.1) *"strict upper triangle read **COLUMN-WISE**"*, and criterion 3 requires
`adjacency.symbols == unpack(graph6 payload)` bit for bit.

- **graph6 is column-wise.** `'ElCW'` → `'l'=45=101101`, `'C'=4=000100`, `'W'=24=011000`, first 15
  bits `101101000100011` — the column-wise triangle of `G`.
- **AGM is column-wise.** `scratch/agm_cam.py::_code_from_perm` walks `for k in 1..n-1: for j in 0..k-1`.
- **The scout's adjacency is row-major.** `scratch/backends.py:70`, docstring included:
  *"Strict upper triangle, row-major, on the incident node order."*

So the two literals the design and agent A's brief quote as the expected adjacency output —
`G = '101001000100111'`, `H = '100001000100111'` — are the **row-major** strings. The column-wise
answers are `G = '101101000100011'`, `H = '101001000100011'`.

**Cost of switching to the frozen convention**, ρ against exact GED (same draw, both orders):

| Dataset | row-major (as measured) | column-wise (as frozen) | Δ |
|---|---:|---:|---:|
| Letter LOW | 0.8655 | 0.8817 | +0.016 |
| Letter MED | 0.8691 | 0.8919 | +0.023 |
| Letter HIGH | 0.8342 | 0.8345 | +0.000 |
| LINUX | 0.7540 | 0.7365 | −0.018 |
| AIDS | 0.8193 | 0.8155 | −0.004 |

No ordering flips: adjacency stays below the null on the Letter sets and above it on LINUX.
**Criterion 1 and criterion 3 cannot both be satisfied as written** — one asserts a row-major
literal, the other asserts agreement with a column-wise format.

---

## W0-3 — README §4.1 is a composite of three draws, and it is not the folder's own raw output

`real_suite1.out` is the scout's own log. Its Letter LOW ρ block reads:

| method | `real_suite1.out` | README §4.1 | Δ |
|---|---:|---:|---:|
| adjacency | 0.8591 | 0.873 | −0.014 |
| graph6 | 0.6466 | 0.691 | **−0.044** |
| sparse6 | 0.7171 | 0.748 | −0.031 |
| nauty→graph6 | 0.6333 | 0.677 | **−0.044** |
| AGM CAM | 0.9105 | 0.911 | −0.000 |
| min-DFS | 0.9760 | 0.972 | +0.004 |
| IsalGraph pruned | 0.9279 | 0.925 | +0.003 |

On AIDS the gap reaches **0.074** (IsalGraph 0.3288 raw vs 0.255 printed) and **0.062** (min-DFS).

**Provenance, resolved.** `real_size_null.json` holds §4.1's printed values to five decimals
(Letter LOW adjacency 0.8727755, graph6 0.6913950, sparse6 0.7475727, nauty 0.6771511,
min-DFS 0.9719640, IsalGraph 0.9252960, null 0.8990792). So:

- most rows of §4.1 come from **`real_size_null.py`**,
- the **AGM** row comes from **`real_suite1.py`** (`real_size_null.py` has no AGM column),
- the **WL** row comes from **`real_wl.py`**.

Three scripts, three different `Random(42)` streams, three different 200-graph draws, presented as
one table. This is finding 14 — *"ρ moved by up to 0.07 between two independent draws"* — appearing
**inside** the table rather than beside it. LINUX is the control: `N = 89 < 200`, so every script
draws the same set and all three agree to four decimals.

**The shipped code is not the problem.** Replicating `real_suite1.py`'s rng stream exactly — the
50-graph F3 draw, then 5 encoders × 50 graphs × 20 `shuffled_copy(rng)` calls, then the ρ draw —
reproduces `real_suite1.out` on Letter LOW **on every row**: ρ to four decimals, F3 = `graph6 4/50 ·
sparse6 4/50 · nauty 50/50 · adjacency 4/50 · min-DFS 50/50`, and the Claim A medians
`adjacency 6.0 · graph6 12.0 · sparse6 24.0 · min-DFS 12.0 · IsalGraph 12.7`.
Reproduction is achievable; what it reproduces is the open question.

---

## Verified unchanged — the running example and the witness

Every other criterion-1 literal reproduces exactly, with `pynauty` freshly installed:

| | `G` | `H` |
|---|---|---|
| graph6 | `'ElCW'` ✔ | `'EhCW'` ✔ |
| sparse6 | `':EaWIzR'` ✔ | `':EaYms'` ✔ |
| nauty→graph6 | `'E@ro'` ✔ | `'E@po'` ✔ |
| AGM CAM | `'000001110011110'` ✔ | `'000001011111000'` ✔ |
| min-DFS | `(0,1)(1,2)(2,0)(2,3)(3,4)(4,5)(5,2)`, 7 tuples ✔ | 6 tuples |
| `\|Aut(G)\|` | 4 ✔ | 2 |

K₃,₃ vs prism: nauty `'Es\o'` / `'E{Sw'` ✔ · AGM `'000111111011100'` / `'001101110111100'` ✔ ·
min-DFS `0-1 1-2 2-3 3-0 3-4 4-1 4-5 5-0 5-2` / `0-1 1-2 2-0 2-3 3-4 4-0 4-5 5-1 5-3` ✔.
The `canon_label` inversion assertion (`nx.is_isomorphic(G, relabelled)`) passes on every encode.

---

## Status

Wave 1 is blocked: agent A's criterion 1 quotes a row-major literal, agent C's criterion 2 is
unsatisfiable as written, and the reproduction target of criterion 1 is undecided. Escalated.
