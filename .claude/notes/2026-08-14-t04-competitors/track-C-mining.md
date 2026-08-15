# Track C — `min_dfs` and `wl_subtree`

**Agent** `competitor-mining`. **Branch** `track-C-mining`, cut from
`9d2291b87b37c9b4fbee9ce994f779dfd69db01c` (`ticket/T-04-competitors`).
**Date** 2026-08-15. Every command below was run through `./wtpy` (CONTRACTS §0.5 shim), so
`isalgraph.engine() == "python"` and **every timing here is pure-Python, single-threaded**.

---

## 1. Files created

```
 src/isalgraph/competitors/backends/min_dfs.py | 419 +++++++++++++
 src/isalgraph/competitors/backends/wl.py      | 369 +++++++++++
 tests/unit/test_min_dfs.py                    | 863 ++++++++++++++++++++++++++
 tests/unit/test_wl_subtree.py                 | 425 +++++++++++++
 4 files changed, 2076 insertions(+)
```

`git diff --stat 9d2291b8 HEAD`, verbatim. Nothing outside the ownership list was touched.
`git status --porcelain` is empty except `sitecustomize.py` and `wtpy`, which are **not committed**
(the worktree's `.git/worktrees/.../info/exclude` is not writable from inside the worktree and
`git config --worktree` refuses without `worktreeConfig`, so they are simply never `git add`ed).

Commits, incremental:

| | |
|---|---|
| `eb5740d` | `feat(T-04): min-DFS and WL subtree backends, track C` |
| `4191700` | `test(T-04): V1/V2/V3 oracles, the kaviniitm gate, and the WL identity` |

---

## 2. Acceptance criteria

| # | Command | Expected | Actual | |
|---|---|---|---|---|
| 1 | `pytest -k "running_example or two_hundred or k33_and_prism"` | code `(0,1)(1,2)(2,0)(2,3)(3,4)(4,5)(5,2)`, 7 tuples = m; 200 relabellings → 1 code; WL `h=3` → 10 / 13 features | all exact | **PASS** |
| 2 | `pytest -k grakel_n_iter` | `grakel(n_iter=2) ≡ ours(h=2) = 5.830952` under 0.1.10 | `5.830951894845301` both, Δ = 0 | **PASS** |
| 3 | `pytest -k v1_exhaustive` | 30 classes at `n ≤ 5`, 1/2/6/21 | 30 checked, 0 mismatches | **PASS** |
| 4 | `pytest -k v3_complete` | 1 / 2 / 6 / 21 / 112, no collisions | exactly that | **PASS** |
| 5 | `pytest -k v2_isomorphism` | 4,440 relabellings, 0 mismatches; reversibility | `(4440, 0)`; reversible on all 142 classes | **PASS** |
| 6 | `pytest -k kaviniitm` | K1 `1/6`, `7/21`, `56/112`; K2 `46/90` | **not reproducible — binary absent.** All 7 counterexamples re-verified independently | **PARTIAL** — §4 |
| 7 | `pytest -k "k33 or witness"` | `d_WL = 0.0000` at `h = 1,2,3,5`; min-DFS separates | exactly `0.0`; both codes match the published strings | **PASS** |
| 8 | `measure.py --what zeros` | 0 false zeros on Letter, ≈1 LINUX, ≈6 AIDS | 0 / 0 / 0, **1** LINUX, **11** AIDS | **PARTIAL** — §4 |
| 9 | `measure.py --what claim_a` | 24/400 Mutagenicity, 0 elsewhere | **24/400**, 0 on the other nine | **PASS** |
| 10 | `measure.py --what f3` | 50/50 both backends, all five Suite-1 | 50/50 × 10 | **PASS** |
| 11 | `measure.py --what claim_a` | README §4.3 min-DFS column, ten datasets | 9 exact, Protein 620.0 vs 615.0 | **PARTIAL** — §4 |
| 12 | `smoke --backends min_dfs,wl_subtree --dataset iam_letter_low` | green | green, JSON in §2.5 | **PASS** |
| 13 | Picasso loginexa | orchestrator's slice | **not run — the orchestrator owns Picasso** | **OPEN** |
| 14 | `pytest` / `ruff` / `mypy --strict` | clean | 75 passed, 1 skipped; `All checks passed!`; `no issues found in 64 source files` | **PASS** |

### 2.1 Criterion 1 — the running example

```
running example: 0-1 1-2 2-0 2-3 3-4 4-5 5-2 | tuples = 7  m = 7
200 relabellings -> 1 distinct code(s)
H = G-(0,3):     0-1 1-2 2-0 2-3 3-4 4-5              6 tuples
k33:             0-1 1-2 2-3 3-0 3-4 4-1 4-5 5-0 5-2
prism:           0-1 1-2 2-0 2-3 3-4 4-0 4-5 5-1 5-3
h=3: nonzero features G=10 H=13
```

### 2.2 Criterion 2 — the grakel identity, re-verified under 0.1.10

`GraKeL-0.1.10.dist-info`, `grakel.__version__ == '0.1.8'` (the stale string wave 0 found).

| `n_iter` / `h` | grakel | ours | `K(G,G)` | `K(H,H)` | `K(G,H)` | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.000000 | 2.000000 | 62 | 54 | 56 | 0 |
| **2** | **5.830952** | **5.830952** | 80 | 66 | 56 | **0** |
| 3 | 7.211103 | 7.211103 | 90 | 74 | 56 | 0 |
| 5 | 9.380832 | 9.380832 | 110 | 90 | 56 | 0 |

`grakel(n_iter=k) ≡ ours(h=k)`, confirmed. `K(G,G) = 62` at `h = 1` is `36` (base, six identical
labels) `+ 26` (degree histogram `5² + 1²`) — the arithmetic, not just the output.
**Frozen `h = 2` means `n_iter = 2`.** Agreement is entrywise on the Gram matrix and holds over
every pair of a 19-graph mixed fixture set at `h ∈ {1,2,3,5}`.

### 2.3 Criteria 9 and 11 — Claim A and the budget, real cohort

Suite 1 = all retained graphs; Suite 2 = the seed-42 400-graph sample, matching `real_suite2.py`.

| Dataset | `n̄` | `m̄` | median entropy bits | README §4.3 | failures | p50 ms | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 4.07 | 3.07 | **12.0** | 12.0 ✔ | 0 | 0.041 | 0.045 |
| Letter MED | 4.11 | 3.17 | **12.0** | 12.0 ✔ | 0 | 0.031 | 0.038 |
| Letter HIGH | 4.58 | 4.56 | **24.0** | 24.0 ✔ | 0 | 0.075 | 0.095 |
| LINUX | 8.71 | 8.35 | **64.0** | 64.0 ✔ | 0 | 0.247 | 0.364 |
| AIDS | 10.56 | 10.70 | **88.0** | 88.0 ✔ | 0 | 0.556 | 0.691 |
| GREC | 11.54 | 12.59 | **96.0** | 96.0 ✔ | 0 | 0.591 | 0.946 |
| AIDS-IAM | 14.01 | 14.48 | **88.0** | 88.0 ✔ | 0 | 0.624 | 1.813 |
| COIL-DEL | 21.30 | 53.48 | **450.0** | 450.0 ✔ | 0 | 51.4 | 79.3 |
| **Mutagenicity** | 27.91 | 28.87 | **250.0** | 250.0 ✔ | **24 / 400** | 5.04 | 850.9 |
| Protein | 31.83 | 61.83 | **620.0** | 615.0 ✘ | 0 | 49.3 | 75.6 |

**Criterion 9 reproduces exactly.** 24 of 400 Mutagenicity graphs raise `MinDfsBudgetExceeded` at
`max_projections = 50_000`, and **zero** graphs fail anywhere else in the ten-dataset cohort. The
failing graphs run `36 ≤ n ≤ 97`; the first five are `n = 89, 97, 76, 58, 85`, each burning
7–29 s before the cap fires. Mutagenicity's mean of 851 ms/graph against a **p50 of 5.04 ms** and a
max of 58.4 s is the folder's "dominated by the 24 graphs that run to the cap", quantified: the
median graph is 170× cheaper than the mean.

The budget is **not** re-derivable from a time limit. The cheapest failure took 7.2 s and the most
expensive successful encode took 58.4 s, so no wall-clock threshold separates the two populations.

**`wl_subtree.bits()` does not exist** (criterion 11's second half): `VectorBackend` has no `bits`
method, and `bits.count()` raises `BitCountUndefined` on the name. Both asserted.

### 2.4 Criterion 10 — F3, 50 graphs × 20 relabellings, seed 42

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---|---|---|---|---|
| `min_dfs` | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| `wl_subtree` | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |

Via `fixtures.shuffled_copy`, which rebuilds each copy with a fresh insertion order.

### 2.5 Criterion 12 — local smoke, green

```
min_dfs            ok= 200 failed=  0 p50=   0.034ms F3=50/50
wl_subtree         ok= 200 failed=  0 p50=   0.009ms F3=50/50
```

```json
{
  "backends": {
    "min_dfs": {
      "available": true, "backend": "min_dfs",
      "bits": {"entropy_p50": 12.0, "realised_p50": 88.0},
      "capabilities": ["canonical", "complete_invariant", "reversible"],
      "f3_invariant_of_50": "50/50", "failures": [],
      "ms_per_graph": {"max": 0.2918520003731828, "p50": 0.03352450039528776,
                       "p90": 0.0726649996067863},
      "n_encoded": 200, "n_failed": 0, "n_failures_recorded": 0
    },
    "wl_subtree": {
      "available": true, "backend": "wl_subtree",
      "bits": {"entropy_p50": null, "realised_p50": null},
      "capabilities": ["canonical", "handles_disconnected"],
      "f3_invariant_of_50": "50/50", "failures": [],
      "fit_s": 0.002047428999503609,
      "ms_per_graph": {"max": 0.014237000868888572, "p50": 0.008902501576812938,
                       "p90": 0.012092998076695949},
      "n_encoded": 200, "n_failed": 0, "n_failures_recorded": 0
    }
  },
  "header": {
    "dataset": "iam_letter_low", "isalgraph_build_hash": "", "isalgraph_engine": "python",
    "n_graphs_requested": 200,
    "packages": {"grakel": "0.1.8", "networkx": "3.6.1", "numpy": "1.26.4",
                 "pynauty": "2.8.8.1", "rapidfuzz": "3.14.5"},
    "platform": "Linux-6.1.0-52-amd64-x86_64-with-glibc2.36", "processor": "",
    "python": "3.11.15",
    "registered_backends": ["isalgraph_canonical", "isalgraph_pruned", "min_dfs",
                            "size_null", "wl_subtree"],
    "seed": 42
  },
  "n_graphs_drawn": 200, "suite": "suite1"
}
```

`isalgraph_engine: "python"` and `build_hash: ""` are the CONTRACTS §0.5 shim, expected and correct
for this track: neither backend calls the engine and neither timing may be compared against it.

### 2.6 Criterion 14 — house rules

```
$ ./wtpy -m pytest tests/unit/test_min_dfs.py tests/unit/test_wl_subtree.py -q
75 passed, 1 skipped, 1 warning in 39.68s
$ ./wtpy -m ruff check src/ tests/
All checks passed!
$ ./wtpy -m mypy --strict src/isalgraph/
Success: no issues found in 64 source files
```

The one skip is the `kaviniitm` binary differential — see §4.

---

## 3. The E10 measurement — `h = 2` versus `h = 5`

`benchmarks/real_data/eval_setup/wl_kernel_computer.py` defaults to `n_iter = 5`, consumed by
`eval_setup.py::wl_n_iter`. Under the corrected convention that is **`h = 5`** — three refinement
rounds past the `h = 2` this ticket freezes, and past the `h = 3` already measured strictly worse on
all five datasets. **I did not edit that file.**

60-graph seed-42 sample per dataset, 1,770 pairs, both `h` fitted on the same 60 graphs:

| Dataset | `dim(h=2)` | `dim(h=5)` | growth | median `d` `h=2` | `h=5` | ratio | `frac(d=0)` `h=2` | `h=5` | Spearman `ρ(d₂, d₅)` | pair orderings flipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 17 | 83 | **4.9×** | 3.7417 | 7.4833 | 2.00× | 0.144068 | **0.144068** | **0.9094** | 175,435 / 1,565,565 = **11.2 %** |
| LINUX | 44 | 785 | **17.8×** | 6.2450 | 11.0904 | 1.78× | 0.0 | **0.0** | **0.8588** | 259,158 / 1,565,565 = **16.6 %** |
| AIDS | 42 | 1,014 | **24.1×** | 6.6332 | 11.7473 | 1.77× | 0.0 | **0.0** | **0.8993** | 212,338 / 1,565,565 = **13.6 %** |

Three findings, for the orchestrator to route:

1. **They are not interchangeable.** `ρ(d₂, d₅)` is 0.86–0.91, and between one in nine and one in six
   of all pair-of-pair orderings flips. A number computed at `n_iter = 5` cannot be quoted beside a
   number computed at `h = 2`.
2. **The extra rounds buy zero discrimination on this cohort.** `frac(d = 0)` is *identical* at
   `h = 2` and `h = 5` on all three datasets — to the digit. Every pair WL collapses at `h = 2` it
   still collapses at `h = 5`. The three extra refinements cost a 24× larger feature space on AIDS
   and separate not one additional pair. This is the strongest argument for `h = 2` that does not
   route through ρ, and it is therefore admissible under the "do not tune `h` on ρ" rule.
3. **The witness is invariant to the choice.** `d(K₃,₃, prism) = 0.0000` at `h = 2` and at `h = 5`
   (and at 1, 3). The completeness claim does not depend on which convention T-06 settles on.

The `h = 5` distances are uniformly ≥ the `h = 2` distances, which is structural: the feature map at
`h = 5` contains the `h = 2` map as a prefix, so the squared distance is a sum of non-negative
increments. Asserted in `test_e10_h2_versus_h5_changes_the_distances`.

---

## 4. Numbers that did not reproduce

### 4.1 Criterion 6 — the `kaviniitm` verdict is not reproducible here. **No fix applied.**

`scratch/test_kavin.py` shells out to `./dfscode_kavin/dfscode`. **That binary is not in this
repository and I did not clone or build it** ("vendor nothing", no network). So the counts
`K1 = 1/6, 7/21, 56/112` and `K2 = 46/90` cannot be re-derived by me. They are preserved verbatim in
`scratch/test_kavin.out`.

What I did instead, which I believe carries the reviewer-facing argument better than a re-run would:

* **All seven K1 counterexamples the archived run printed are re-verified against the oracle**, and
  each is asserted to be (a) a **valid** DFS code of its graph, present in the exhaustive
  enumeration, and (b) **strictly larger** than the minimum under the general Yan & Han order. That
  is the finding — the tool returns a real DFS code that is not the minimal one — and it now holds
  as a test rather than as a recorded print.
* The smallest counterexample is shipped as the named fixture: `K₄` minus edge `(2,3)`, tool
  `<0,1><1,2><2,0><2,3><3,1>`, minimum `<0,1><1,2><2,0><2,3><3,0>`.
* The gate is shipped **reusable and parameterised** — `gate_k2_isomorphism_invariance(candidate)`
  and `gate_k1_agreement(candidate)`, K2 first — so any third-party candidate proposed later runs
  through it in two lines. `test_kaviniitm_differential_against_the_binary` re-runs the full
  differential and asserts the archived counts when `KAVIN_DFSCODE_BIN` is set; it is the one skip.
* **The gate is shown capable of failing.** `greedy_no_branch_code` reproduces the structural defect
  (greedy extension, no tie branching — what `LasseRegin/gSpan` does internally) and K2 rejects it.
  A gate that cannot fail is worthless, and the reusable lesson from `kaviniitm` is precisely that it
  passed every check anyone bothered to run.

### 4.2 Criterion 8 — Letter and LINUX reproduce; AIDS does not

Measured with the **corrected** shared-vocabulary WL at `h = 2`, over **every** pair of the seed-42
200-graph sample. `frac(GED = 0)` is computed as `frac(isomorphic)` via `nx.is_isomorphic`: under
the D6 unit cost model `GED = 0 ⟺ isomorphic`, so the two are identical and **no GED reference is
imported** (CONTRACTS §7).

| Dataset | pairs (mine) | pairs (folder, certified) | `frac(d_WL = 0)` mine | folder | isomorphic among the zeros | **false zeros** | folder |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 19,900 | 19,900 | 0.143769 | 0.1438 | 2,861 | **0** | 0 ✔ |
| Letter MED | 19,900 | 19,900 | 0.139497 | 0.1530 | 2,776 | **0** | 0 ✔ |
| Letter HIGH | 19,900 | 19,900 | 0.043920 | 0.0453 | 874 | **0** | 0 ✔ |
| LINUX | 3,916 | 3,870 | 0.000255 | 0.00026 | 0 | **1** | ≈1 ✔ |
| AIDS | 19,900 | 15,686 | 0.000553 | 0.00038 | 0 | **11** | ≈6 ✘ |

**The qualitative claim survives in full and is the one the paper makes**: on all three Letter sets
WL's zero-set is *exactly* the isomorphic set — the pseudometric behaves as a metric there — while
on LINUX and AIDS it does not. Report both halves, as the folder says.

The AIDS count is **11, not ≈6**, and both halves of the difference are visible in the table:

1. **Different denominator.** The folder's 15,686 is the certified-exact subset; mine is all 19,900
   pairs. Rescaling gives `11 × 15686/19900 ≈ 8.7` — still above 6, so the denominator explains part
   of the gap and not all of it.
2. **Different implementation.** The folder's WL row came from `scratch/backends.py::wl_features`,
   which wave 0 showed compresses colours per graph per round. Letter MED (0.1530 → 0.1395) and
   Letter HIGH (0.0453 → 0.0439) also move, in the direction a per-graph compression predicts: it
   manufactures spurious cross-graph matches, so it reports *more* zeros. Letter LOW does not move
   (0.1438 both) because at `n̄ = 4.07` there is nothing to refine.

**I did not fix anything.** The correct number for AIDS on the certified subset needs the GED
reference, which is the orchestrator's `reproduce.py`, not mine. My recommendation is that
`wl-subtree-kernel.md` §2.2's AIDS row be restated from that run, not from mine and not from the
folder's.

### 4.3 Criterion 11 — Protein is 620.0, not 615.0

Nine of ten datasets reproduce README §4.3's min-DFS column exactly. Protein gives **620.0** against
a printed 615.0.

**Diagnosis: a different 400-graph draw, not a different convention.** My `n̄ = 31.83, m̄ = 61.83`
against the printed `31.88 / 61.81`. `real_suite2.py` loads Suite 2 from the **raw IAM GXL** via
`iam_gxl_loader.load_iam_gxl`; `datasets.py` loads the recovered `.npz`. The raw GXL is not on this
workstation (wave 0 recovered the `.npz` from Picasso), so the scout's retained ordering — and
therefore `Random(42).sample(range(N), 400)` — cannot be replicated from here. AIDS-IAM shows the
same signature with no effect on its median: `n̄ = 14.01` against a printed `13.63`, median 88.0
either way; its **mean** is 133.4 against a printed 128.2.

The convention itself is exact and is asserted independently:
`entropy_bits = m · 2⌈log₂ n⌉` on every fixture, `realised_bits = 8·len(text)`, `inflated=True`.

**No fix applied.** If the orchestrator wants the printed figures bit-for-bit, `reproduce.py` must
either replay the GXL loader or restate §4.3's Suite-2 rows from the `.npz` cohort — which is the
same choice W0-3 already forced for §4.1.

### 4.4 A correction to the design note, minor

Design §8 criterion 2 and the evidence file both write V2 as "4,440 relabellings over
**6 ≤ n ≤ 10**". `scratch/validate_min_dfs.py` loops `for n in (5, 6, 7, 8, 9, 10)`. The count 4,440
is a function of that `Random(42)` draw *stream*, so it is only reachable with `n = 5` included. The
test uses the script's range and says so in its docstring; the range in prose should be corrected to
`5 ≤ n ≤ 10`.

---

## 5. Contract defects found — reported, not fixed

### D1 — `smoke.py::_f3` catches only `CompetitorError`, so a plain exception aborts the run

`smoke.py:92`. `min_dfs` raises **`ValueError`** on a disconnected or edgeless graph — that is the
documented AE.3 behaviour and the design note asks for exactly it. `ValueError` is not a
`CompetitorError`, so it propagates out of `_f3`, out of `run_backend`, and kills the whole smoke
process instead of being recorded as a per-graph failure. The encode loop at `smoke.py:152` catches
bare `Exception` and is fine; only the F3 harness has the gap.

It never fires on the cohort — both suites are `require_connected = True`, and all 3,600 Suite-2 and
5,350 Suite-1 encodes above confirm it. It would fire the first time anyone points `smoke.py` at a
cohort that is not connectivity-filtered. One-line fix in the orchestrator's file:
`except (CompetitorError, ValueError):`.

Evidence:

```
>>> min_dfs_code(fixtures.to_networkx(fixtures.C4_PLUS_K3_DISJOINT))
ValueError: DFS code undefined: graph is disconnected
```

### D2 — `smoke.py::_signature` fits a `VectorBackend` on **one graph at a time**

`smoke.py:101`, `backend.fit([graph])` inside the F3 loop. That is a per-batch fit of size 1, which
is exactly the pattern `wl-subtree-kernel.md` §7 and CONTRACTS §2 forbid.

It is harmless **only because** this implementation's colours are digests of canonical signatures, so
`features()` does not depend on the fitted corpus — asserted by
`test_features_do_not_depend_on_what_was_fitted` and
`test_distance_matrix_is_independent_of_batching`. Under any WL that compresses colours per corpus —
the scout's, or a thin grakel wrapper — `_f3` would compare vocabularies built from different single
graphs and would report **50/50 invariant for the wrong reason**, i.e. an F3 harness that cannot
fail. It is latent, not live, and it is worth a comment in `smoke.py` naming the assumption it
relies on.

### D3 — `grakel.Graph` cannot be built from an edge list on an edgeless graph

Not a defect in an orchestrator module, but it bites anyone writing the grakel bridge, so it is
recorded. `Graph([], node_labels=...)` reads the empty list as an empty adjacency matrix and raises
`IndexError: tuple index out of range` at `grakel/graph.py:950`. `fixtures.EMPTY_3` triggers it.
The **edge-dictionary** form `{v: {} for v in G}` with `graph_format="all"` carries the vertex set
explicitly and handles both edgeless graphs and isolated vertices; `wl.py::_to_grakel` uses it.

### D4 — `bits.py`'s min-DFS width formula is right, and worth pinning

`bits.py:100` computes the width as `2 * max(n - 1, 1).bit_length()` where the design note writes
`2⌈log₂ n⌉`. These agree for every `n ≥ 2` (`(n-1).bit_length() == ⌈log₂ n⌉`) and differ at `n = 1`,
which cannot occur because an edgeless graph raises first. Not a defect — noted because the two
forms look different and a future reader may "fix" one of them.
`test_entropy_bits_are_the_fixed_width_upper_bound` asserts against `math.ceil(math.log2(n))`
directly, so a divergence would fail rather than drift.

---

## 6. Decisions the design note did not cover

**C1 — the connected-isomorphism-class enumerator is `networkx`'s graph atlas, not mask
enumeration.** `scratch/validate_min_dfs.py` deduplicates `2^15` labelled graphs with chained
`nx.is_isomorphic` calls, which is ~1–2 M isomorphism tests at `n = 6`. I use
`nx.graph_atlas_g()` (Read & Wilson, *An Atlas of Graphs*, Oxford 1998) filtered by order and
connectivity. It gives 1 / 2 / 6 / 21 / 112 / 853 at `n = 2…7`, it uses **no code under test**, and
V3 drops from minutes to under a second — which is what makes it affordable to also run V3 with the
budget installed.

**C2 — a WL colour is a `blake2b` digest of the canonical signature, not a per-corpus integer.**
This is the substantive design choice in `wl.py`. grakel compresses colours to integers over the
fitted corpus; the scout compressed them per graph per round, which is the bug wave 0 found. A
content-addressed colour makes `features()` a function of local structure alone, so the per-batch-fit
corruption is **unreachable** rather than merely discouraged — the same reasoning that made
`VectorBackend` a separate protocol rather than a `ReprBackend` with a raising `bits()`. Cost: a
64-bit digest, collision probability below `1e-10` over the few thousand colours a Suite-2 dataset
produces. `fit()` is retained because the protocol requires it and because the vocabulary is worth
reporting; it is inert with respect to every distance, and two tests assert that.

**C3 — `normalize=True` raises instead of being implemented.** The constructor must accept the
keyword (CONTRACTS §4), and `normalize=False` is frozen. Implementing the normalised variant would
put a second, unfrozen measurement one keyword away from the frozen one. Raising with the reason —
it divides by `√(K(x,x)K(y,y))` and removes the graph-size signal GED depends on — makes the frozen
choice structural.

**C4 — the 5-tuple `(i, j, l_i, l_ij, l_j)` is kept internally, not collapsed to `(i, j)`.** Our
corpus degenerates it, but keeping the full tuple makes the module a literal transcription of Yan &
Han and lets the brute-force oracle compare the object the paper defines. Only `code_symbols()`
projects to `(i, j)`.

**C5 — `budget=None` uses the frozen default; `Budget(max_projections=None)` is unbounded.** The
contract says "`None` means unbounded" of the *field* and "`None` uses the backend's frozen default"
of the *argument*, which are different `None`s. The exhaustive oracles pass
`max_projections=None` explicitly to the free function, so V1/V3 are unaffected by the cap — and
`test_v1_holds_with_the_frozen_budget_in_place` then re-runs V1 *with* the cap, because design §9
condition 6 requires the validation suite to pass after the budget is installed, not before.

**C6 — criterion 8's `frac(GED = 0)` is computed as `frac(isomorphic)`.** CONTRACTS §7 forbids
importing `ged_reference` from anywhere in my track. Under D6 unit costs `GED = 0 ⟺ G ≅ H`, so
`nx.is_isomorphic` gives the identical quantity with no forbidden import and no ρ. The only residual
difference from the folder's table is the *denominator* — all pairs against the certified subset —
which is stated in §4.2.

**C7 — no timing of min-DFS against IsalGraph, anywhere.** Every timing in this log is labelled
pure-Python single-threaded, and `isalgraph.engine()` reads `"python"` in this worktree, so the
comparison R1.1 objects to is not merely avoided but unavailable.

---

## 7. Open questions

1. **Criterion 13 (Picasso)** is open by design. I need the `min_dfs` / `wl_subtree` slice of the
   loginexa JSON. The two things worth checking there: that `min_dfs`'s Mutagenicity failure count
   is still 24/400 on a different CPU (it is a memory cap, so it should be exactly reproducible —
   if it is not, the construction is non-deterministic somewhere and that is a bug), and that
   `grakel` on Picasso reproduces `5.830952`.
2. **Which WL numbers does the paper print?** §4.2 shows the AIDS false-zero count moving from ≈6 to
   11 under the corrected implementation and a wider denominator. `wl-subtree-kernel.md` §2.2's
   table needs restating from `reproduce.py`, not from either existing source. **This is a printed
   number and it is currently wrong in the folder.**
3. **E10 routing.** `wl_kernel_computer.py` stays at `n_iter = 5` = `h = 5`. §3 gives the numbers
   T-06 needs to decide whether to change the default or to document that its consumers report a
   different `h` from the competitor table. I did not edit it.
4. **Protein's 620 vs 615** (§4.3) is a Suite-2 draw-provenance question of exactly the kind W0-3
   raised for §4.1. If `reproduce.py --mode table` is going to restate Suite 2 from the `.npz`
   cohort, this row resolves itself; if §4.3 is meant to stand as printed, someone needs the raw
   GXL, which is not on this workstation.
5. **`m·2⌈log₂ n⌉` is an upper bound and the module docstring says so**, with the R3.6a consistency
   argument. That argument only holds if `B_GED` really does keep `2M⌈log₂ N⌉` — I did not verify
   `statistics.md` §2 against the shipped GED-constructor code. Worth one check by whoever owns it.
