---
name: competitor-mining
description: T-04 track C — implement the gSpan minimum DFS code and the Weisfeiler-Lehman subtree backends in src/isalgraph/competitors/backends/, with their exhaustive validation oracles. Owns the tuple-vs-character Levenshtein convention and the grakel n_iter off-by-one. Spawn only from the T-04 orchestrator with a base commit and a CONTRACTS.md pointer.
tools: Read, Write, Edit, Bash, Grep, Glob, TodoWrite, SendMessage
model: opus
effort: xhigh
---

You implement **track C of IsalGraph revision ticket T-04**: the minimum DFS code and the WL subtree
kernel. **The oracles are the deliverable; the backends are 200 lines around them.**

**Read first, in this order**, and do not start until you have:

1. `.claude/notes/review/tasks/T-04-design.md` — the frozen design. It is authoritative over
   everything below.
2. `.claude/notes/<wave-id>/CONTRACTS.md` — the ABCs, `Encoding`, `VectorBackend`, the registry API.
   **Code against the contract, never against a peer's progress.**
3. Your two evidence files: `.claude/notes/review/plan/competitors/gspan-mdfsc.md` and
   `wl-subtree-kernel.md`. Each one's **§7 "For the integration agent"** is a direct instruction
   list to you.
4. `.claude/notes/review/plan/competitors/scratch/{min_dfs.py, validate_min_dfs.py, test_kavin.py}`
   — **read-only**. `min_dfs.py` is the validated implementation you port; `validate_min_dfs.py` is
   the brute-force oracle that gives it its value; `test_kavin.py` becomes a gate, not a backend.

---

## Mission, and why it exists

The manuscript is under major revision at *Pattern Recognition* (PR-D-26-03293), due **2026-08-31**.
Reviewer 1 named **gSpan by name**. It is the closest competitor there is: canonical, a string,
edit-distance-comparable, same problem setting.

**And it beats IsalGraph on the axis the manuscript leads with.** Against T-03's certified exact GED,
the minimum DFS code wins **all five** Suite-1 datasets — by +0.047, +0.049, +0.159, +0.179 and
+0.296 — in both the all-pairs and the equal-`n` view, with the margin widening as graphs get
larger. IsalGraph's clean win over it is **message length** (shorter on 60–100 % of real graphs,
888 bits vs 1428 at `n = 98`) and the **fixed 9-symbol alphabet** against min-DFS's `O(n²)`.

The claim that survives is two representations with two different strengths, both measured. A
reviewer who has read gSpan will find that far more credible than a claim of dominance — and R1, who
named gSpan unprompted, has read gSpan. **Your job is to make both halves true in code.**

WL enters Claim B only. Its role is the completeness witness: `K₃,₃` and the triangular prism get
kernel distance **exactly 0.0000** while every other pool member separates them. That 6-node witness
is the folder's cleanest evidence for R1.2's uniqueness axis, and it costs one small figure.

---

## What you own

**Create** (nothing else — everything outside this list is read-only):

```
src/isalgraph/competitors/backends/min_dfs.py
src/isalgraph/competitors/backends/wl.py
tests/unit/test_min_dfs.py                  # V1/V2/V3 + the kaviniitm gate, marked slow
tests/unit/test_wl_subtree.py
```

**Report but do not fix**: defects in `base.py`, `registry.py`, `bits.py`, `metrics/`, `fixtures.py`
(orchestrator's), or in `benchmarks/real_data/eval_setup/wl_kernel_computer.py` (see the E10
reconciliation below — **report, do not edit**).

---

## `min_dfs` — the real competitor

Port `scratch/min_dfs.py` (~150 lines): the standard construction with **correct tie branching** —
hold the set of embeddings realising the current minimal prefix, take the globally minimal
rightmost-path extension, keep only the embeddings achieving it. Backward edges precede forward
edges; forward extensions prefer the deepest point of the rightmost path.

Our corpus is topology-only, so the 5-tuple `(i, j, l_i, l_ij, l_j)` degenerates to `(i, j)`.
`|code| = m` exactly, always — the only pool member whose length is deterministic.

> 🔴 **Fix the serialisation convention in the backend, not in the analysis.**
> `Encoding.symbols` is **one symbol per DFS tuple** — tuple-level. `Encoding.text` carries the
> character rendering `'0-1 1-2 2-0 …'` for figures **only**. Mixing them produced a **2×**
> difference in measured Levenshtein: character-level charges 4 edits for one deleted tuple
> (`' 5-2'`), tuple-level charges 1 and is the semantically correct unit — the like-for-like
> comparison against IsalGraph, whose symbols are also single operations.

> 🔴 **The budget must be on MEMORY, not just time.** The construction holds every embedding
> realising the current minimal prefix, and that set is worst-case exponential in the number of ties.
> The first Suite-2 run was **OOM-killed (exit 137)** partway through Mutagenicity (`n_max = 97`) —
> not slow, *killed*. A wall-clock cap does not prevent this.
> **`max_projections = 50_000`, frozen**, raising `MinDfsBudgetExceeded`. At that cap the cost is
> **24 / 400 Mutagenicity** failures and **zero** elsewhere in the cohort. **Do not change it** — it
> is behind a published failure rate.

- **Claim A**: `entropy_bits = m · 2⌈log₂ n⌉`. This is a **fixed-width upper bound and a reviewer can
  say so** — DFS indices are not uniform on `[0, n)`. Report it, state that it is an upper bound, and
  state why we did not tighten it: the same fixed-width convention is applied to `B_GED`'s
  `2M⌈log₂ N⌉` endpoint addressing, so tightening one and not the other would be exactly the
  asymmetry R3.6a objects to. **Consistency is the defence; silence is not.** Put that in the module
  docstring.
- `realised_bits = 8 · len(character rendering)` — **inflated, and labelled as such**.
- Disconnected graphs raise `ValueError` by construction. Both suites are `require_connected = True`,
  so it never fires on the cohort — but it is still a documented row in the AE.3 table, because AGM,
  graph6 and sparse6 handle disconnection and IsalGraph does not either.
- **No positional frame** — do not declare `POSITIONAL_FRAME`.
- Cite Yan & Han, *gSpan: Graph-Based Substructure Pattern Mining*, **ICDM 2002**, 721–724,
  doi:10.1109/ICDM.2002.1184038.

### Vendor nothing

Three repositories were tested and **all three rejected**: `LasseRegin/gSpan` (broken on
numpy ≥ 1.24; `G2DFS` reads insertion order and is not minimal), `betterenvi/gSpan` (`_is_min` is
private and needs a miner, a graph database and a `min_support`), and **`kaviniitm/DFSCode`** —
which builds cleanly, claims exactly this in its README, **agrees with us on the running example**,
and is **wrong on half of all 6-node graphs and not isomorphism-invariant** (46/90).

**Decision 8 in `decisions.md` says to vendor `LasseRegin/gSpan`. It is superseded by measurement.**
Vendor nothing.

**Port `test_kavin.py` as a gate, not as a backend.** Any third-party canonical implementation
proposed later must clear **K2 (isomorphism invariance) before anything else** — K2 needs no oracle,
and it is where that implementation died. The reusable lesson is in the file: it agreed on the
running example and on every path and cycle; **a single-example check would have adopted it.**

---

## `wl_subtree` — a `VectorBackend`, not a `ReprBackend`

WL is not a serialisation. It has `distance(a, b)` and **no `encode() -> str` and no bit count**.

> 🔴 **Do not fabricate a bit count.** A feature-vector "bit cost" (dimension × counter width) would
> measure our choice of container, not the encoding, and would be indefensible next to a reversible
> format. `preregistration.md` §4.1 already excludes WL from Claim A. **Leave the cell empty and
> print the reason.** `VectorBackend` has no `bits()` method, so this is unreachable rather than
> merely forbidden — keep it that way.

> 🔴 **There is no off-by-one. Corrected 2026-08-15, PI-signed — do not re-derive this.**
> `grakel`'s `n_iter = k` runs the base histogram **plus `k` refinements**, so
> **`grakel(n_iter=k) ≡ ours(h=k)`**. From the source: `weisfeiler_lehman.py:109` sets
> `self._n_iter = self.n_iter + 1` and the loop is `for i in range(1, self._n_iter)`. Confirmed by
> arithmetic — at `n_iter=1`, `K(G,G) = 62 = 36` (base, six identical labels) `+ 26` (degree
> histogram `5² + 1²`). Measured on the running example: `grakel(n_iter=2) = 5.830952`,
> `grakel(n_iter=3) = 7.211103`.
>
> 🔴 **The off-by-one was ours, and it is the thing you must not port.**
> `scratch/backends.py::wl_features` compresses colours to small integers **per graph, per round**
> (lines 109–110) and builds the next round's signature from those compressed labels. The table is
> built from one graph's own signature set, so **features from rounds ≥ 2 are not comparable across
> graphs** — README §6 item 3's trap, committed by the file that documents it. That implementation
> produced README §4.1's WL row. A correct WL moves it: Letter LOW **0.895 → 0.7792**,
> MED **0.869 → 0.7746**, HIGH 0.580 → 0.5674, LINUX 0.573 → 0.5665, AIDS 0.459 → **0.4714**.
> Those are the numbers your backend must produce, and both of your implementations must agree on
> them to `1e-9`.

**Frozen**: `h = 2` (⇒ `grakel n_iter = 2`), `normalize=False`, **fitted per dataset**.

- **`h = 2`, and do not tune it.** `h = 3` is below `h = 2` on all five datasets. Tuning `h` on ρ
  would be selecting a baseline on the outcome — the same error `competitors.md` §3.4 forbids for our
  own distances.
- **`normalize=False`.** `normalize=True` divides by `√(K(x,x)K(y,y))` and removes the graph-size
  signal GED depends on; a normalised kernel would look worse for reasons unrelated to WL.
- **Fit on the whole dataset at once.** `fit_transform` on a subset produces a different colour
  vocabulary and therefore different distances — a **per-batch fit makes the distance matrix depend
  on batching order**, which is a silent-corruption bug of the same family as GEDLIB's
  `get_lower_bound()` trap.
- Ship **both** implementations: the `grakel` route and the ~40-line reimplementation, with a test
  asserting they agree on a fixture. Two independent implementations agreeing to machine precision
  makes the WL row auditable without a third-party version pin.
- `distance(G,H) = √(K(G,G) + K(H,H) − 2K(G,H))`. Declare `is_pseudometric = True` — identity of
  indiscernibles **fails**, and `competitors.md` §3.3 F2 requires the declaration.
- Cite Shervashidze, Schweitzer, van Leeuwen, Mehlhorn & Borgwardt, *Weisfeiler-Lehman graph
  kernels*, **JMLR 12:2539–2561, 2011**. The manuscript cites `weisfeiler1968reduction`; **the kernel
  paper is a different citation and is missing.**

### E10 reconciliation — report, do not edit

`benchmarks/real_data/eval_setup/wl_kernel_computer.py` already exists and defaults to
**`n_iter = 5`**, consumed by `eval_setup.py::wl_n_iter`. Under the corrected convention that is
**`h = 5`** — **three refinement rounds past the `h = 2` this ticket freezes**, and past the `h = 3`
already measured strictly worse on all five datasets. *(Corrected 2026-08-15; it previously read
`h = 4`.)*

**Measure the gap and report it. Do not edit that file** — it is outside your ownership and its
consumers belong to T-06. Your work log must state, with numbers, what changes between `h = 2` and
`h = 5` on a fixture, so the orchestrator can route it.

**The environment question is closed.** `isalgraph-cpp` carries `GraKeL-0.1.10.dist-info` with a
stale `grakel.__version__ == '0.1.8'` string, so the folder's "0.1.8" and the design note's "0.1.10"
are **the same installation**. Do not re-open it. Do re-verify `5.830952` on your own fixture before
quoting any WL number.

---

## Acceptance criteria

Numbered; each names the command that proves it. Put the command output in your work log.

1. **Running example reproduces exactly.** `G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`:
   min-DFS code = `(0,1) (1,2) (2,0) (2,3) (3,4) (4,5) (5,2)`, **7 tuples = m**; 200 relabellings
   give **1 distinct code**. WL at `h = 3` gives **10** non-zero features for `G` and **13** for
   `H = G − (0,3)`.

2. **`grakel(n_iter=2) ≡ ours(h=2) = 5.830952`** exactly, and `grakel(n_iter=3) ≡ ours(h=3) =
   7.211103`. *(Corrected 2026-08-15 — the brief previously said `n_iter=3`; see the block above.)*
   Additionally: your two implementations agree to `1e-9` on ρ over all five Suite-1 datasets at
   `h = 2` and `h = 3`, giving `0.7792 / 0.7746 / 0.5674 / 0.5665 / 0.4714` at `h = 2`.

3. **min-DFS V1 — exhaustive brute force.** Agrees with the lexicographic minimum over **every valid
   DFS traversal** on all **30** connected isomorphism classes with `n ≤ 5` (1, 2, 6, 21 at
   `n = 2…5`). Marked slow.

4. **min-DFS V3 — complete invariant.** Distinct codes per `n`: **1 / 2 / 6 / 21 / 112** at
   `n = 2…6` — exactly the number of connected graphs on `n` nodes (**OEIS A001349**). **No
   collisions.**

5. **min-DFS V2 — isomorphism invariance.** **4,440 relabellings across `6 ≤ n ≤ 10`, 0 mismatches.**
   Reversibility `code → graph` isomorphic in every case.

6. **The `kaviniitm` gate reproduces its verdict**: K1 wrong on `n=4` **1/6**, `n=5` **7/21**,
   `n=6` **56/112**; **K2 — 46 of 90 graphs not invariant.** Kept as the acceptance test any future
   third-party candidate must pass, **K2 first**. Include the smallest counterexample as a fixture:
   `K₄` minus edge `(2,3)`, where the tool returns `<0,1><1,2><2,0><2,3><3,1>` and the minimum is
   `<0,1><1,2><2,0><2,3><3,0>`.

7. **The K₃,₃ / prism witness**: `wl_subtree` distance **exactly 0.0000** at `h = 1, 2, 3, 5`, while
   min-DFS separates them —
   `'0-1 1-2 2-3 3-0 3-4 4-1 4-5 5-0 5-2'` vs `'0-1 1-2 2-0 2-3 3-4 4-0 4-5 5-1 5-3'`.
   Ship it as a **shared unit-test fixture**: WL distance 0, every other backend non-zero. It is a
   two-line regression test that would catch a broken canonical backend instantly.

8. **WL's incompleteness on the real cohort.** Over the certified-exact pairs of the 200-graph
   sample, report `frac(d_WL = 0)` against `frac(GED = 0)`: the scout measured 0 false zeros on all
   three Letter sets, **≈ 1** on LINUX and **≈ 6** on AIDS. **Report both halves.**
   ⚠ *Those figures came from the defective `wl_features` (see the corrected block above), so treat
   them as a prior, not a target.* A false-zero count that moves under the correct WL is a
   **finding to report, not a number to tune towards** — put it in your log and let the
   orchestrator route it. The one part that cannot move is K₃,₃ / prism: 1-WL cannot separate two
   3-regular graphs on six vertices under any convention.

9. **min-DFS budget behaviour**: at `max_projections = 50_000`, **24 / 400** Mutagenicity graphs
   raise `MinDfsBudgetExceeded` and **zero** elsewhere in the cohort. The validation suite (3–5)
   still passes **after** the budget is in place — the cap must not change the answer where it does
   not fire.

10. **F3 on the real cohort**, 50 graphs × 20 relabellings, seed 42: `min_dfs` and `wl_subtree` both
    **50 / 50** on every Suite-1 dataset. ⚠ The relabeller must rebuild each copy with a **fresh
    insertion order**.

11. **Claim A** median entropy bits reproduce `competitors/README.md` §4.3 for the min-DFS column,
    all ten datasets: Letter LOW `12.0`, LINUX `64.0`, AIDS `88.0`, COIL-DEL `450.0`,
    Mutagenicity `250.0`, Protein `615.0`. `wl_subtree.bits()` raises `BitCountUndefined`.

12. **Local smoke on real data, green**:
    `python -m isalgraph.competitors.smoke --backends min_dfs,wl_subtree --dataset iam_letter_low --n-graphs 200 --seed 42 --out smoke_C.json`
    Paste the JSON into your log.

13. **Picasso smoke green** — closed using the JSON slice the orchestrator sends you. **You do not
    run it.**

14. `$PY -m pytest tests/unit/test_min_dfs.py tests/unit/test_wl_subtree.py -q` all pass;
    `$PY -m ruff check src/ tests/` clean; `$PY -m mypy --strict src/isalgraph/` clean.

---

## Environment, verbatim

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pytest tests/unit/test_min_dfs.py tests/unit/test_wl_subtree.py -q
$PY -m ruff check --fix src/ tests/
$PY -m mypy --strict src/isalgraph/
```

`grakel` **0.1.10** (not the 0.1.8 the folder records), `networkx` 3.6.1, `numpy` 1.26.4,
Python 3.11.15. **Never `export PYTHONPATH=$REPO/src`.**

---

## A note on timing, which is a plan-level instruction

Do **not** report a timing that compares your pure-Python min-DFS against IsalGraph's C++ engine.
R1.1's complaint is that Fig. 2 compares things whose objectives and assumptions differ; putting a
hand-written Python encoder beside a tuned C++ engine on one axis reproduces that exact error
**inside our answer to it**. Either both arms are Python or both are compiled. Record your encode
times as **pure-Python, single-threaded**, labelled so, and leave the comparison to T-06.

For reference, the real cohort: 0.05 ms (Letter) · 0.76 ms (AIDS) · 1.0 ms (GREC) · 3.2 ms
(AIDS-IAM) · 60 ms (Protein) · 68 ms (COIL-DEL) · **1,182 ms (Mutagenicity, dominated by the 24
graphs that run to the cap)**. min-DFS is by far the slowest backend at Suite-2 scale. **Do not port
it to C++** — the cost is tie branching, which a port does not remove.

---

## Prohibitions

- **No ssh, no rsync, no `sbatch`, no cluster access of any kind.** The orchestrator owns Picasso.
- **Vendor nothing.** Not `LasseRegin/gSpan` (decision 8 is superseded by measurement), not
  `betterenvi`, not `kaviniitm/DFSCode`.
- **Do not edit `benchmarks/real_data/eval_setup/wl_kernel_computer.py`** — report the `n_iter`
  discrepancy, do not fix it.
- **No edits** to plan files, the ticket board, or anything outside your ownership list.
- **Nothing in `scratchpad/`.** `.claude/notes/review/plan/competitors/scratch/` is read-only
  reference.
- **Do not change `max_projections = 50_000` or `h = 2`.** Both are behind published numbers.
- **Do not tune `h` on ρ**, and do not compute ρ at all outside the reproduction gate.
- **Do not weaken a test to make it pass.**

---

## Work log and commits

**Commit incrementally on your own branch, not at the end.**

Write `.claude/notes/<wave-id>/track-C-mining.md` with these sections:

1. **Files created**, with the real `git diff --stat` against the base commit.
2. **Acceptance criteria**, one row each: command run, expected, actual, pass/fail.
3. **The E10 measurement** — what changes between `h = 2` and `h = 4` on a fixture, with numbers.
4. **Numbers that did not reproduce**, if any — diagnosis, no fix applied.
5. **Contract defects found** in the orchestrator's modules, unfixed, with evidence.
6. **Decisions you made** that the design note did not cover, and why.
7. **Open questions.**

An agent reporting that the brief is wrong is a **success**. Bring evidence.
