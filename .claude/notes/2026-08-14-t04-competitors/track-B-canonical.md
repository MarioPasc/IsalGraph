# Track B — the canonical-labelling backends

**Agent** `competitor-canonical`. **Wave** `2026-08-14-t04-competitors`.
**Branch** `track-B-canonical`, cut from **BASE_SHA `9d2291b87b37c9b4fbee9ce994f779dfd69db01c`**.
**Interpreter** `./wtpy` (CONTRACTS.md §0.5 shim), `isalgraph.engine() == "python"`.
**Every timing below is pure-Python, single-threaded.**
`pynauty 2.8.8.1` · `networkx 3.6.1` · `numpy 1.26.4` · Python 3.11.15 ·
`Linux-6.1.0-52-amd64-x86_64-with-glibc2.36`.

**Headline.** Every criterion-1 and criterion-2 target reproduces byte for byte. The AGM
brute-force oracle is **327 graphs, 0 mismatches**. F3 is **50/50** on all five Suite-1
datasets for both backends. The ceiling reproduces on six of the seven rows.

**Three things in the source material are wrong**, all with evidence in §3 and §4:

1. `nauty.md` §1/§7's `canon_label` inversion guard is wrong on **both** halves — the
   inversion fails F3 loudly (30/30), and `nx.is_isomorphic` cannot catch it *ever*.
2. `agm.md` §2.3's worked example prints a nauty payload that is neither the column-wise nor
   the row-major reading of the graph, and the true payload **equals** AGM's code on that
   graph — so the example as printed argues the opposite of its own conclusion. The
   conclusion survives; §3 supplies evidence that actually carries it.
3. `bits.py::_packed_bits` halved `realised_bits` for `adjacency` **and `agm_cam`**. Found
   here, independently fixed by the orchestrator mid-wave.

---

## 1. Files created

```
 src/isalgraph/competitors/backends/agm.py   | 379 ++++++++++++++
 src/isalgraph/competitors/backends/nauty.py | 433 ++++++++++++++++
 tests/unit/test_agm_cam.py                  | 389 ++++++++++++++
 tests/unit/test_competitors_canonical.py    | 778 ++++++++++++++++++++++++++++
 4 files changed, 1979 insertions(+)
```

`git diff --stat 9d2291b8 HEAD`, verbatim. Nothing outside the ownership list was touched:
no edit to `base.py`, `registry.py`, `bits.py`, `fixtures.py`, `metrics/`, `smoke.py`, any
plan file, or any peer's module. `sitecustomize.py` and `wtpy` are in the worktree-local
`info/exclude` and are not committed; `git status --porcelain` is empty.

**Public API added**, beyond the three registered backends:

| Symbol | Module | Why it is public |
|---|---|---|
| `canonical_relabel(G, *, verify=True) -> nx.Graph` | `nauty` | `sparse6_nauty` uses it; so will T-17 |
| `automorphism_group_size(G) -> float` | `nauty` | **T-13** needs `\|Aut(G)\|` for the complexity section's worst case |
| `automorphism_orbits(G) -> tuple[int, ...]` | `nauty` | AGM orbit pruning, if it is ever wanted — see §5.3 |
| `certificate(G) -> bytes` | `nauty` | F3 assertions only; never a compactness row |
| `graph6_payload_bits`, `graph6_prefix_bytes`, `upper_triangle_pairs` | `nauty` | the `n > 62` branch is live at Suite 2's `n = 98` |
| `agm_canonical_code(G, *, node_budget) -> (code, expanded)` | `agm` | the algorithm, without the suite policy — this is what makes the ceiling measurable |
| `identity_code(G) -> str` | `agm` | criterion 7's reading-order identity, on a pinned labelling |
| `code_to_graph(code, n) -> nx.Graph` | `agm` | reversibility |

---

## 2. Acceptance criteria

| # | Command | Expected | Actual | |
|---|---|---|---|:--:|
| **1** | `pytest -k "running_example or automorphism"` | `nauty_graph6` `G='E@ro'` `H='E@po'`; `agm_cam` `G='000001110011110'` `H='000001011111000'`; `n=6, m=7`; `\|Aut(G)\|=4` | all exact; `autgrp` → `4.0e0` | **PASS** |
| **2** | `pytest -k "k33 or prism"` | `nauty` `'Es\o'` vs `'E{Sw'`; `agm` `'000111111011100'` vs `'001101110111100'`; both separate | all exact; both separate; `pynauty.certificate` also separates | **PASS** |
| **3** | `pytest tests/unit/test_agm_cam.py -m slow -k "not ceiling"` | 327 graphs, 0 mismatches vs the lex-min over all `n!`; reversibility on all 327 | **327 / 0**; reversible on all 327; class counts **2·4·11·34·156** = OEIS A000088; 11 passed in **19.8 s** | **PASS** |
| **4** | `pytest -m slow -k f3_on_the_real_cohort` | `nauty_graph6` and `agm_cam` **50/50** on every Suite-1 dataset, 50 graphs × 20 relabellings, seed 42 | 10/10 parametrisations at 50/50; the negative control (`graph6`, same relabeller, same cohort) fails, so the harness can fail | **PASS** |
| **5** | `pytest tests/unit/test_agm_cam.py -m slow -k ceiling` | see the table below | 7 passed in **208.5 s**; six rows exact, AIDS-IAM 80.25 % vs 82 % — §3 | **PASS**, one row diagnosed |
| **6** | `pytest -k "inverted or isomorphism_guard"` | a deliberately inverted `canon_label` is caught | **caught by F3, not by `nx.is_isomorphic`** — the brief is wrong, §4.1 | **PASS**, criterion restated |
| **7** | `pytest -k "identity_code or reading_order or family"` | AGM at the identity == `'101101000100011'` and == graph6's unpacked payload | exact on both literals and on all seven fixtures; the cross-backend form vs A's `adjacency.symbols` **skips** (module absent), closed by the orchestrator at merge | **PASS** (cross-backend deferred) |
| **8** | `python -m isalgraph.competitors.smoke --backends nauty_graph6,sparse6_nauty,agm_cam --dataset iam_letter_low --n-graphs 200 --seed 42 --out smoke_B.json` | green | green; `sparse6_nauty` UNAVAILABLE by design in an isolated worktree. JSON below | **PASS** |
| **9** | Picasso loginexa | `pynauty` imports from a from-source gcc 12.2.0 build | **not run by me — orchestrator's** | **OPEN** |
| **10** | `pytest <both files> -q`; `ruff check src/ tests/`; `mypy --strict src/isalgraph/` | all pass, both clean | **122 selected: 116 passed, 5 skipped, 1 xfailed**; ruff `All checks passed!`; mypy `Success: no issues found in 64 source files` | **PASS** |

### Criterion 5 in full — the ceiling, at the frozen budgets

`./wtpy scratchpad/ceiling_report.py`, full artefact at `agm_ceiling_B.json`.

| Dataset | budget | N | `n̄` | `n_max` | exact | **agm.md §2.2b** | median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 200k | 1180 | 4.07 | 7 | **100.00 %** | 100 % | 0.0 |
| Letter MED | 200k | 1253 | 4.11 | 8 | **100.00 %** | 100 % | 0.0 |
| Letter HIGH | 200k | 2059 | 4.58 | 9 | **100.00 %** | 100 % | 0.0 |
| LINUX | 200k | 89 | 8.71 | 10 | **100.00 %** | 100 % | 4.5 |
| **AIDS** | 200k | **769** | 10.56 | 12 | **99.61 %** (3 fail) | **99.6 % (3 fail)** | 21.9 |
| **GREC** | 100k | 400 | 11.54 | 24 | **76.00 %** (96 fail) | **76 % (96 fail)** | 35.2 |
| **AIDS-IAM** | 100k | 400 | 14.01 | 73 | **80.25 %** (79 fail) | 82 % (73 fail) | 22.2 |

**The 3 AIDS failures, recorded and printed** — they are why AGM has no ρ column on AIDS:

| index | `graph_id` | `n` | `m` |
|---:|---|---:|---:|
| 43 | `aids_train_0049` | 12 | 11 |
| 319 | `aids_train_0378` | 12 | 12 |
| 394 | `aids_train_0465` | 12 | 11 |

All three are `n = 12`, `m ∈ {11, 12}` — **sparse**, which is agm.md §2.2b's own diagnosis:
near-empty prefixes tie constantly and prefix pruning never bites. GREC reproduces to the
graph: 96 failures, `n̄ = 11.54`, `n_max = 24`, every figure in §2.2b.

### Criterion 8 — `smoke_B.json`, verbatim

```json
{
  "backends": {
    "agm_cam": {
      "available": true, "backend": "agm_cam",
      "bits": { "entropy_p50": 6.0, "realised_p50": 8.0 },
      "capabilities": ["canonical","complete_invariant","handles_disconnected",
                       "positional_frame","reversible","suite1_only"],
      "f3_invariant_of_50": "50/50",
      "failures": [], "n_encoded": 200, "n_failed": 0, "n_failures_recorded": 0,
      "ms_per_graph": { "p50": 0.0286505000985926, "p90": 0.125868999020895,
                        "max": 0.37935900036245584 }
    },
    "nauty_graph6": {
      "available": true, "backend": "nauty_graph6",
      "bits": { "entropy_p50": 12.0, "realised_p50": 16.0 },
      "capabilities": ["canonical","complete_invariant","handles_disconnected",
                       "positional_frame","reversible"],
      "f3_invariant_of_50": "50/50",
      "failures": [], "n_encoded": 200, "n_failed": 0, "n_failures_recorded": 0,
      "ms_per_graph": { "p50": 0.08673900083522312, "p90": 0.12112299737054855,
                        "max": 0.5844389997946564 }
    },
    "sparse6_nauty": {
      "available": false, "backend": "sparse6_nauty",
      "import_error": "BackendUnavailableError: competitor backend 'sparse6_nauty' is registered but its third-party library is not installed"
    }
  },
  "header": {
    "dataset": "iam_letter_low", "seed": 42, "n_graphs_requested": 200,
    "isalgraph_engine": "python", "isalgraph_build_hash": "",
    "packages": { "grakel": "0.1.8", "networkx": "3.6.1", "numpy": "1.26.4",
                  "pynauty": "2.8.8.1", "rapidfuzz": "3.14.5" },
    "platform": "Linux-6.1.0-52-amd64-x86_64-with-glibc2.36",
    "python": "3.11.15",
    "registered_backends": ["agm_cam","isalgraph_canonical","isalgraph_pruned",
                            "nauty_graph6","size_null","sparse6_nauty"]
  },
  "n_graphs_drawn": 200, "suite": "suite1"
}
```

**A second, unrequested smoke on a Suite-2 dataset**, because it is the only way to see both
of AGM's refusal paths fire and to exercise nauty's `n > 62` prefix branch:

```
--dataset aids_iam --n-graphs 200 --seed 42
nauty_graph6       ok= 200 failed=  0 p50=   0.225ms F3=50/50
agm_cam            ok= 169 failed= 31 p50=  10.432ms F3=41/50
        failures: SuiteScopeError x30 (n = 12..56), AGMBudgetExceeded x1 (n = 12)
```

Every refusal is typed and recorded. `agm_cam`'s F3 reads 41/50 because nine of the fifty
were refused and `smoke.py` skips a graph it could not encode — correct, and worth a footnote
wherever that column is printed.

---

## 3. Numbers that did not reproduce

### 3.1 AIDS-IAM: **80.25 % against a stated 82 %** — a sample difference, not an algorithm one

| | `agm.md` §2.2b | measured here |
|---|---:|---:|
| exact | 82 % (73 fail) | **80.25 %** (79 fail) |
| `n̄` of the 400 drawn | **13.63** | **14.01** |
| `n_max` of the 400 drawn | **85** | **73** |

**Diagnosis: the two runs drew different 400-graph samples from the 1811-graph cohort.**
`n̄` and `n_max` both disagree, and they are properties of the draw, not of AGM. The
control is GREC, where the *same* code, the *same* budget and `cohort.sample(400, seed=42)`
reproduce §2.2b to the graph — 76.00 %, 96 failures, `n̄ = 11.54`, `n_max = 24`. An algorithm
fault would not spare GREC.

**No fix applied.** The scout's draw is not recoverable from `agm.md`; recovering it is
`reproduce.py`'s provenance replay, which is the orchestrator's. Two candidate causes, both
cheap to test there: the scout sampled from a differently ordered export of `aids_iam` (the
Suite-2 `.npz` were re-recovered from Picasso on 2026-08-15), or it drew with a different
call. The measured row is internally consistent and reproducible from the committed test.

Everything else in criterion 5 reproduces exactly.

---

## 4. Contract defects found — reported, not fixed

### 4.1 `nauty.md` §1 and §7: the inversion guard is wrong on both halves — **PLAN DEFECT**

> *"Getting this backwards produces a different but still deterministic labelling — it will
> pass an invariance test and be wrong. Assert `nx.is_isomorphic(G, relabelled)` on every
> encode."* — `nauty.md` §1, repeated at §7 and in the T-04 brief.

**Half one is false: the inverted labelling fails F3, loudly.** Over 20 genuine relabellings
the inverted code took **15 / 19 / 5 / 13** distinct values on the running example, `G−(0,3)`,
`K₃,₃` and the prism, and **30 of 30** random `n = 8` graphs were non-invariant. The correct
labelling gave exactly one code in every case.

The reason is one line of algebra. nauty guarantees `C(G) = G^{π_G}` with `π_G = lab_G^{-1}`.
For `G' = G^τ` this forces `lab_{G'} = π_G^{-1} τ`, so the wrong-direction image is
`G^{τ π_G^{-1} τ}`, which depends on `τ`. There is no reason for it to be invariant and it
is not.

**Half two is false a priori: `nx.is_isomorphic` cannot catch the inversion, ever.** Any
bijective relabelling of `G` is isomorphic to `G` **by construction**, so the assertion holds
for every permutation, correct or inverted. Measured: `nx.is_isomorphic(G, inverted)` was
`True` on 100 % of the cases above. A guard that cannot fail is not a guard.

**Cost of the prescribed guard**, since it is not free: `nx.is_isomorphic` is **6.7 ms at
`n = 96`** against **0.33 ms** for the whole relabelling — a 20× tax on a step whose published
cost (`nauty.md` §2) is 0.042–0.351 ms. Suite 2 reaches `n = 98`.

**What was implemented instead**, both guards, each honest about its job:

- Unconditionally, `O(n + m)`: the map is a bijection onto `range(n)` and no two edges
  collided. Together those *prove* isomorphism, because a bijective relabelling is an
  isomorphism — so this is strictly stronger than VF2 and free.
- `verify=True` (the default) additionally runs `nx.is_isomorphic`, as instructed, because
  what it *does* catch is the realistic fault: a wrong `networkx`-label → `pynauty`-index map,
  since `pynauty` requires vertices `0..n−1` while a `networkx` graph may carry any label.
  A test injects a non-permutation `canon_label` and asserts the refusal.
- `verify=False` exists for the language-matched Fig. 2 timing mode.

**Recommended amendment**: `nauty.md` §1 and §7 should say *"the inversion is caught by F3,
which is why F3 must be run before any nauty row is quoted; the isomorphism assertion guards
the index map, not the inversion."*

### 4.2 `agm.md` §2.3: the worked example is wrong, and argues against its own conclusion

§2.3 prints:

```
nauty canonical labelling -> graph6 payload   'E@ro'  ->  bits 001110010011100
AGM lex-min labelling                                 ->  bits 000001110011110
```

Measured: `'E@ro'` → `E`=6 nodes, payload `@`=`000001`, `r`=`110011`, `o`=`110000`, first 15
bits **`000001110011110`**. The row-major reading of the same canonical graph is
`000110011101010`. **`001110010011100` is neither.** It is a third string with no derivation.

**The consequence is worse than a typo.** The true nauty payload on the running example
**equals AGM's code**, so the example as printed is evidence *against* §2.3's thesis that
nauty cannot supply the AGM labelling.

**The thesis is nonetheless correct**, and here is evidence that carries it — agreement
between nauty's graph6 payload and AGM's code on random graphs:

| `n` | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| agreement | 38/60 | 32/60 | 16/60 | 12/60 | 1/60 | **0/60** |

They coincide on small, sparse graphs by coincidence and diverge monotonically. §2.3 should
replace the six-node example with this table, or with any `n ≥ 9` pair. The plan's premise
("derive AGM from nauty's labelling") remains wrong and no orbit pruning was wired in (§5.3).

### 4.3 `bits.py::_packed_bits` halved `realised_bits` — **found here, fixed by the orchestrator**

`_packed_bits(n_bits, word=16)` returns `8·⌈n_bits/16⌉` and is called with
`n_bits = T = n(n−1)/2`, giving `8·⌈T/16⌉`. `T-04-design.md` §4.2 specifies
`8·⌈n(n−1)/16⌉ = 8·⌈T/8⌉`. The shipped value is **half** the correct one and, from `n = 6`
up, **below the payload it is supposed to store**:

| `n` | payload `T` | design `8⌈n(n−1)/16⌉` | shipped | below payload? |
|---:|---:|---:|---:|:--:|
| 4 | 6 | 8 | 8 | no |
| 6 | 15 | **16** | **8** | **yes** |
| 12 | 66 | **72** | **40** | **yes** |
| 98 | 4753 | **4760** | **2384** | **yes** |

Visible in production output: the Suite-2 smoke reports `agm_cam` `entropy_p50 = 55`,
`realised_p50 = 32` — 32 bits claimed to store 55. It hits **`adjacency` and `agm_cam`**,
i.e. the `n²` family's realised-bits row in Claim A. `entropy_bits` was never affected, and
the row is unaffected at Letter's `n̄ ≈ 4`.

The orchestrator relayed mid-wave that this is fixed on the integration side.
`tests/unit/test_competitors_canonical.py::test_agm_realised_bits_match_the_frozen_formula`
asserts the **correct** value, is `xfail`-marked in this worktree, and **flips to XPASS at
merge** — which is the signal that the fix landed.

### 4.4 `T-04-design.md` §8 criterion 4 states an unsatisfiable assertion

*"`adjacency.bits(e).realised_bits < len(e.text)`"*. For the `n²` family `len(text)` is the
payload in bits, so this demands that a lossless packing store `T` bits in fewer than `T`
bits. It is satisfiable **only** under the §4.3 bug. The intended assertion is
`realised_bits < 8·len(text)` — no eightfold inflation. Orchestrator confirms it is being
amended; recorded here because the two defects are the same defect seen from two sides, and
the design's own criterion is what made the bug look correct.

### 4.5 Two minor observations, no action requested

- `registry.get_backend` reports an unavailable `sparse6_nauty` as *"its third-party library
  is not installed"*. In an isolated worktree the real cause is a **sibling module**, not a
  third-party library. Harmless here; a reader chasing a missing package would waste time.
- `smoke.py::_f3` skips a graph the backend refused, so `f3_invariant_of_50` silently becomes
  `k / 50` over a **smaller** denominator when a `SUITE1_ONLY` backend meets Suite-2 graphs
  (measured: `41/50` for `agm_cam` on `aids_iam`). The number is not wrong, but it means
  something different from the same number on Suite 1.

---

## 5. Decisions the design note did not cover

### 5.1 The `nauty_graph6` wire excludes the trailing newline

`nx.to_graph6_bytes(H, header=False)` emits `b'E@ro\n'`. `wire` is stored `.strip()`ed, so
`len(wire) = 4` and `entropy_bits = 24`. Keeping the newline would give 30 and contradict
`nauty.md` §4's `6·(1 + ⌈n(n−1)/12⌉)`, and criterion 1's expected string is `'E@ro'`, not
`'E@ro\n'`. **Agent A's `graph6.py` must make the same choice** or the two columns of the
identical-by-construction pair will differ; a test asserts
`len(nauty_wire) == len(graph6_wire)` on every fixture.

### 5.2 `agm_canonical_code` is a module-level function, and the suite policy is not in it

The `SUITE1_ONLY` refusal lives in `AGMBackend.encode`; the search itself has no suite guard.
Otherwise the ceiling would be unmeasurable — criterion 5's GREC and AIDS-IAM rows are graphs
the *backend* refuses by policy, and the paper cannot state a ceiling it has forbidden itself
to observe. The test calls the function and says so in its docstring.

### 5.3 Orbit pruning is deliberately **not** wired in

`automorphism_orbits` is exposed but unused. Wiring it into the search would change how many
nodes are expanded, and the frozen 200k/100k budgets are calibrated against the unpruned
search — so it would silently move the published 99.6 % / 76 % / 82 % failure rates, which is
stop-and-ask condition 6. It is a constant-factor optimisation that `agm.md` §2.3 itself says
"will not reach `n = 32`". If T-06 ever wants it, the budgets must be re-derived first.

### 5.4 The port is faithful, not improved

`scratch/agm_cam.py`'s greedy incumbent, candidate ordering, prefix-pruning test and budget
check are reproduced operation for operation, including `list(G.nodes())` as the vertex order.
The one change is that `AGMBudgetExceeded` **carries no incumbent** (the scratch version
attached `.best`). Any "optimisation" would move the search-node count and therefore the
failure rate. Confirmation that the port is exact: the running example expands **47** search
nodes, the number `agm.md` §2 prints.

### 5.5 `identity_code` pins the labelling by rebuilding

Following the orchestrator's mid-wave relay:
`nx.convert_node_labels_to_integers(ordering="sorted")` renames values and leaves insertion
order alone, which is what `to_graph6_bytes` actually reads. `identity_code` rebuilds with
`add_nodes_from(sorted(...))`, matching track A's `normalised()`. `agm_canonical_code` is
**not** normalised — see §5.4; the code is a minimum over all permutations and does not need
it, but the incumbent and the node count do depend on it.

### 5.6 `Sparse6NautyBackend.is_available()` includes the sibling module

A backend that registers and then raises `ModuleNotFoundError` on first use is the silent
degrade the registry exists to prevent, so the sparse6 module is part of availability. In
this worktree that makes `sparse6_nauty` cleanly UNAVAILABLE; in the merged tree it is always
present. The cross-edge is resolved via `importlib.import_module` + `cast` rather than a
static import, because `mypy --strict` cannot see a peer's not-yet-merged module and would
fail a correct file. `test_sparse6_serialise_matches_the_frozen_signature` guards the cast.

### 5.7 One test exempts `path_2` from the inversion check, and the exemption is a theorem

`|Aut(K₂)| = 2 = 2!`, so every labelling of it produces `'A_'` and no labelling scheme is
distinguishable on it. The test states the condition `|Aut(G)| = n!` rather than hard-coding
the fixture name, and asserts that at least four fixtures were actually checked.

---

## 6. Open questions

1. **Criterion 9 (Picasso) is open.** I ran nothing on the cluster. What it gates:
   `pynauty` building from source under gcc 12.2.0. A failure takes `nauty_graph6`,
   `sparse6_nauty` and AGM's (unused) orbit pruning together — but **not** `agm_cam` itself,
   which needs only `networkx`. `nauty.md` §8's correction is right about that.
2. **Criterion 7's cross-backend form** (`agm.identity_code` == A's `adjacency.symbols`)
   skips here and is written and committed, ready to run at merge. Same for the three
   `sparse6_nauty` tests.
3. **Is the `SUITE1_ONLY` refusal meant to be per graph or per dataset?** As implemented —
   and identically to `backends/isalgraph_ref.py` — it is **per graph**, so `agm_cam` on
   `aids_iam` produced an 84.5 %-complete column with 31 typed, recorded refusals. Design §8
   criterion 9 says the point is to avoid *"silently producing a 76 %-complete column"*;
   nothing here is silent, but a backend sees one graph and cannot know the dataset. If a
   dataset-level refusal is wanted it belongs in `smoke.py`/`grid.py`, which are yours.
4. **Which draw produced `agm.md` §2.2b's AIDS-IAM row?** §3.1. Needed before that 82 % is
   printed, or the row should be restated as the 80.25 % measured here from a named draw.
5. **`agm.md` §2.3 and `nauty.md` §1/§7 need amending** (§4.1, §4.2). Both are plan files and
   outside my ownership. Both are quoted by T-08 and T-17.
