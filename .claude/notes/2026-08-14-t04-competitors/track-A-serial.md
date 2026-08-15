# Track A — `adjacency` · `graph6` · `sparse6`

**Wave** `2026-08-14-t04-competitors` · **base** `9d2291b87b37c9b4fbee9ce994f779dfd69db01c`
**Branch** `worktree-agent-a6e805f2cd1aa441f` · engine `python` (worktree shim, per CONTRACTS §0.5)
**All timings below are pure-Python, single-threaded.**

Three of my brief's numbers do not reproduce, and one contract module has an arithmetic
defect. All four are in §3 and §4 with evidence, none of them fixed, none of them tuned away.

---

## 1. Files created

```
 src/isalgraph/competitors/backends/adjacency.py | 232 ++++++++
 src/isalgraph/competitors/backends/graph6.py    | 204 +++++++
 src/isalgraph/competitors/backends/sparse6.py   | 150 +++++
 tests/unit/test_competitors_serial.py           | 752 ++++++++++++++++++++++++
 4 files changed, 1338 insertions(+)
```

`git diff --stat 9d2291b87b37c9b4fbee9ce994f779dfd69db01c HEAD`, verbatim. Nothing outside the
ownership list was touched. `git status --porcelain` is empty except `sitecustomize.py` and
`wtpy`, both added to `.git/info/exclude`.

Commits, incremental:

| SHA | Subject |
|---|---|
| `ba8ee6a` | `feat(T-04a): adjacency, graph6 and sparse6 backends` |
| `a2ed230` | `test(T-04a): 148 tests for adjacency, graph6 and sparse6` |

---

## 2. Acceptance criteria

| # | Command | Expected | Actual | Verdict |
|---|---|---|---|---|
| 1 | `pytest -k running_example` | `adjacency` `'101101000100011'`/`'101001000100011'`; `graph6` `'ElCW'`/`'EhCW'`; `sparse6` `':EaWIzR'` (7 B) / `':EaYms'` (6 B) | identical | **PASS** (with the PI-corrected column-wise literals; see §3.1) |
| 2 | `pytest -k f3_on_the_real_cohort` | all three fail F3 on every Suite-1 dataset | LOW **6/50** · MED **5/50** · HIGH **9/50** · LINUX **0/50** · AIDS **0/50**, identical for all three backends | **PASS on the claim**, `9/50` is outside the brief's `0–6` band — §3.2 |
| 3 | `pytest -k claim_a_medians` | README §4.3, ten datasets, three columns | `adjacency` **10/10 exact**; `graph6` **10/10 exact**; `sparse6` **10/10 exactly 6 bits low** | **PASS for two columns, FAIL for `sparse6`** — §3.3, a deliberate design change the brief predates |
| 4 | `pytest -k inflated or beat_one_bit` | no 8× inflation; nothing reads `Encoding.text` | no inflation on any fixture; AST scan finds zero `.text` reads in all three modules | **PASS**; the brief's literal form is arithmetically impossible — §3.4 |
| 5 | `pytest -k round_trip` | `decode(encode(G))` edge-set identical at `n ∈ {2,15,16,32,62,63,64,98}` | exact for all three at all eight sizes, all 7 fixtures, and 300 random graphs each | **PASS** |
| 6 | `pytest -k family_identity` | `adjacency.symbols` == graph6's unpacked payload | equal on all 7 fixtures, 300 random graphs, and all 8 boundary sizes including the 4-byte-`N(n)` branch | **PASS** |
| 7 | `python -m isalgraph.competitors.smoke --backends adjacency,graph6,sparse6 --dataset iam_letter_low --n-graphs 200 --seed 42 --out smoke_A.json` | green | 200/200 encoded, 0 failures, all three | **PASS** — JSON in §5 |
| 8 | Picasso loginexa | orchestrator's | **not run by me** | **OPEN** |
| 9a | `./wtpy -m pytest tests/unit/test_competitors_serial.py -q` | all pass | **143 passed, 5 xfailed in 3.65 s** | **PASS** |
| 9b | `./wtpy -m ruff check src/ tests/` | clean | `All checks passed!` | **PASS** |
| 9c | `./wtpy -m mypy --strict src/isalgraph/` | clean | `Success: no issues found in 65 source files` (62 → 65) | **PASS** |
| — | `./wtpy -m pytest tests/ -q` | at or above reference | **1703 passed, 324 skipped, 5 xfailed in 59.6 s** | **PASS** |

The 5 `xfailed` are one strict-xfail test, parametrised over five `n`, that carries the `bits.py`
defect of §4.1 as executable evidence. It is not a skip: `strict=True` means the suite turns red
the moment `bits.py` is corrected, which is the point.

---

## 3. Numbers that did not reproduce

### 3.1 The adjacency running-example literals — already known, now confirmed independently

`competitors/adjacency-matrix.md` §2 still prints `'101001000100111'` / `'100001000100111'`.
Those are **row-major**. The frozen order is column-wise and the correct strings are
`'101101000100011'` / `'101001000100011'`, exactly as the PI correction of 2026-08-15 states.

Independent confirmation, not taken on trust: `graph6('ElCW')` unpacks to
`l = 101101`, `C = 000100`, `W = 011000`, whose first 15 bits are `101101000100011`. My
`adjacency` and my `graph6` agree bit for bit on 7 fixtures, 300 random graphs and 8 boundary
sizes, so the reading order is now pinned by a test rather than by a memo. **`adjacency-matrix.md`
§2's literals should be corrected when the plan files are next touched** — not by me, they are
read-only to this track.

### 3.2 F3 Letter HIGH is `9/50`, outside the brief's `0–6/50` band — and the band is the wrong shape

Measured, 50 graphs × 20 `fixtures.shuffled_copy` relabellings, fresh `Random(42)` per backend,
on the seed-42 200-graph draw:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| `adjacency` | 6/50 | 5/50 | **9/50** | 0/50 | 0/50 |
| `graph6` | 6/50 | 5/50 | **9/50** | 0/50 | 0/50 |
| `sparse6` | 6/50 | 5/50 | **9/50** | 0/50 | 0/50 |

Two things differ from the brief and both are explainable.

**(a) All three backends give the same count.** That is a consequence of my normalisation
(§5, decision 1): all three serialise the *same* normalised labelling, so a relabelling invisible
to one is invisible to all. The scout's counts differed across backends (`4/2/6` for graph6)
because its `adjacency_bits` read the raw insertion order while its `graph6` and `sparse6` went
through `convert_node_labels_to_integers` — i.e. its three backends were encoding **three
different labellings of the same graph**. Making them agree is what criterion 6 asks for; the
side effect is that the three F3 columns collapse into one.

**(b) `9/50` is not a sampling artefact, and the evidence file's explanation of the Letter
successes is wrong.** `competitors/graph6.md` §2 says: *"they are tiny graphs with large
automorphism groups, where 20 draws can miss every distinguishable labelling."* I enumerated
**all `n!` relabellings** of every graph in the 50-graph Letter draws:

| dataset | 20-draw count | exhaustive count | what the invariant graphs are |
|---|---:|---:|---|
| Letter LOW | 6/50 | **6/50** | **all 6 are complete graphs** |
| Letter MED | 5/50 | **5/50** | **all 5 are complete graphs** |
| Letter HIGH | 9/50 | **9/50** | **all 9 are complete graphs** |
| LINUX | 0/50 | 0/17 with `n ≤ 8` | — (distinct-code counts 60 … 40 320) |
| AIDS | 0/50 | 0/3 with `n ≤ 8` | — (distinct-code counts 5 040, 10 080) |

The sampled count **equals** the exhaustive count everywhere. The 20-draw harness misses
nothing. Every success is a complete graph, for which `Aut(K_n) = S_n` and the orbit of the code
is a single point — a structural fact, not a draw that got lucky.

That is a *better* result for the paper than the folder's: the F3 successes are not evidence of
"partial invariance" nor of an under-powered protocol; they are the trivial graphs, and they can
be named as such in one clause. It also means the count is a property of how many complete
graphs the 50-graph draw happens to contain, which is exactly why the brief's per-dataset targets
are stream-dependent.

**The claim that does hold exactly, and that I assert in the test**: 0/50 on LINUX and AIDS, the
representative datasets, for all three backends; and on Letter, the invariant graphs are
`{K_n}` and nothing else.

### 3.3 Claim A: `sparse6` is exactly 6 bits low on all ten datasets, and it is the `':'`

Median entropy-bound bits, Suite 1 = every retained graph, Suite 2 = 400-graph sample seed 42,
matching README §4.3's own caption:

| Dataset | `adjacency` | README | `graph6` | README | `sparse6` **shipped** | `sparse6` at `6·len(wire)` | README |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | **6.0** | 6.0 | **12.0** | 12.0 | 18.0 | **24.0** | 24.0 |
| Letter MED | **6.0** | 6.0 | **12.0** | 12.0 | 18.0 | **24.0** | 24.0 |
| Letter HIGH | **10.0** | 10.0 | **18.0** | 18.0 | 30.0 | **36.0** | 36.0 |
| LINUX | **36.0** | 36.0 | **42.0** | 42.0 | 54.0 | **60.0** | 60.0 |
| AIDS | **55.0** | 55.0 | **66.0** | 66.0 | 66.0 | **72.0** | 72.0 |
| GREC | **55.0** | 55.0 | **66.0** | 66.0 | 72.0 | **78.0** | 78.0 |
| AIDS-IAM | **55.0** | 55.0 | **66.0** | 66.0 | 66.0 | **72.0** | 72.0 |
| COIL-DEL | **153.0** | 153.0 | **162.0** | 162.0 | 276.0 | **282.0** | 282.0 |
| Mutagenicity | **300.0** | 300.0 | **306.0** | 306.0 | 162.0 | **168.0** | 168.0 |
| Protein | **465.0** | 465.0 | **474.0** | 474.0 | 384.0 | **390.0** | 390.0 |

`adjacency` and `graph6` reproduce **exactly, 10/10**. `sparse6` is **exactly 6.0 bits low,
10/10** — one six-bit ASCII character, the `':'`.

**Diagnosis, and it is not a defect in anyone's code.** README §4.3 was produced by
`scratch/real_suite2.py::bits`, which is `len(code) * 6.0` with `code` the full stripped string
**including the `':'`**. T-04-design §4.2 and CONTRACTS §5 then *deliberately* froze the opposite
convention — `6·len(wire) − 6`, the `':'` excluded from the entropy bound and included in the
realised bytes — and `bits.py` implements that correctly. So the shipped module is right, README
§4.3's `sparse6` column is the pre-decision value, and **my brief's criterion 3 quotes the
pre-decision value.**

Consequence for downstream: `sparse6.md` §4's own text is internally inconsistent — it says
"Entropy bound `6 · len(sparse6)`" in one paragraph and "exclude it from the entropy bound" in
the next. Whoever rewrites §4.1/§4.2 under gate 1b needs to subtract 6 from every `sparse6`
Claim-A cell, and the qualitative conclusions are untouched (a uniform −6 shift cannot reorder
anything, and sparse6 still beats IsalGraph on Protein and COIL-DEL and loses on Mutagenicity).

My test asserts **both** conventions, so the delta is provably the prefix byte and cannot quietly
become something else.

### 3.4 Criterion 4's literal assertion is arithmetically impossible

The brief asks for `adjacency.bits(e).realised_bits < len(e.text)` **on every fixture**. With
`T = n(n−1)/2`:

* under the design's specified packing, `realised_bits = 8·⌈n(n−1)/16⌉ = 8·⌈T/8⌉ ≥ T = len(text)`,
  so the strict inequality is **never** true — packing to whole bytes can only round up;
* under what `bits.py` actually computes (§4.1), `realised_bits = 8·⌈T/16⌉`, and the inequality
  needs `T ≥ 9`, i.e. `n ≥ 5`. It therefore fails on `PATH_2` (`T = 1`) and `EMPTY_3` (`T = 3`),
  two of the seven fixtures.

The criterion's actual content is *"no 8× inflation"*, and that is what I assert, on every
fixture: `realised_bits < 8·len(text)`. I additionally assert the brief's literal form on the
five fixtures with `n ≥ 5`, where it is arithmetic rather than luck.

---

## 4. Contract defects found, unfixed

### 4.1 🔴 `bits.py` under-counts the adjacency realised cost by a factor of two

`src/isalgraph/competitors/bits.py:39-46`:

```python
def _packed_bits(n_bits: int, word: int = 16) -> int:
    return 8 * math.ceil(n_bits / word)
```

called at line 70 as `_packed_bits(triangle)` with `triangle = n * (n - 1) // 2`.

* Specified (T-04-design §4.2, CONTRACTS §5, `adjacency-matrix.md` §4): `8·⌈n(n−1)/16⌉`.
  Since `n(n−1) = 2T`, that is `8·⌈T/8⌉` — the docstring's own words, *"the triangle packed
  8 bits to a byte"*.
* Implemented: `8·⌈T/16⌉ = 8·⌈n(n−1)/32⌉`. **Half.**

Witness: the running example, `n = 6`, `T = 15`. The specified value is
`8·⌈30/16⌉ = 16` bits = 2 bytes, which is the smallest number of whole bytes that holds 15 bits.
`bits.count()` returns **8 bits = 1 byte**, which cannot hold a 15-bit triangle. Live output
from the smoke run: `adjacency` `realised_p50 = 8.0` on Letter LOW, where `T = 6` and one byte is
in fact correct; at Mutagenicity, `T = 300` and it returns `8·⌈300/16⌉ = 152` bits where 38 bytes
= 304 bits is right.

The docstring's `word: int = 16` reads as "two-byte word", which is where the confusion started;
`adjacency-matrix.md` §4 writes the formula over `n(n−1)` and the code applies it to `n(n−1)/2`.

**Not fixed — `bits.py` is the orchestrator's.** Carried as
`test_adjacency_realised_bits_match_the_frozen_closed_form`, `xfail(strict=True)`, so the suite
goes red the moment it is corrected. **`entropy_bits` is unaffected**, and `entropy_bits` is what
Claim A's headline table reports, so §3.3's numbers stand either way. `realised_bits` for
`adjacency` and `agm_cam` is wrong wherever it is printed.

### 4.2 `nx.convert_node_labels_to_integers(G, ordering="sorted")` does not pin the labelling

Prescribed by `graph6.md` §7 and used by `scratch/backends.py`. It relabels node *values* but
leaves the graph's **insertion order** untouched, and `networkx`'s writers then re-derive their
own labels from the insertion order:

```python
# networkx/readwrite/graph6.py::to_graph6_bytes
H = nx.convert_node_labels_to_integers(G)        # DEFAULT ordering = insertion order
nodes = sorted(H.nodes())
```

Measured over 300 graphs whose insertion order was scrambled with labels held fixed: the §7
recipe and a genuine sorted rebuild disagree on **260 of 300**. Minimal witness — `G` with nodes
inserted `[5, 1, 3]` and edges `(5,1), (1,3)`; the sorted triangle is `110`, the §7 recipe emits
`b'Bg'` whose payload is `101`.

Worse, the two writers disagree with each other: `to_graph6_bytes` labels by insertion order,
`to_sparse6_bytes` calls `convert_node_labels_to_integers(G, ordering="sorted")` itself. Followed
literally, §7 gives a `graph6` and a `sparse6` backend that serialise **different labellings of
the same graph**, which is what the scout's code did.

Also worth recording: `to_graph6_bytes(G, nodes=...)` documents *"Nodes are labeled 0...n-1 in
the order provided"*, but the body is `G = G.subgraph(nodes)` — a view that keeps the original
node order — so **the order provided is ignored and only the node set is used**. That is a
`networkx` 3.6.1 documentation defect; do not rely on the parameter.

Not a defect in a wave-0 module, but it is a defect in the folder's §7 instructions, and it is
shared by agent B (`nauty.py` relabels then serialises through my `sparse6.serialise`, so it
inherits the fix automatically).

### 4.3 Minor: dead `if TYPE_CHECKING: pass` block in `base.py:33-34`

Cosmetic, harmless, `ruff` does not flag it. Noted only so it is not mistaken for a stub someone
still owes.

---

## 5. Decisions I made that the design note did not cover

**1. Normalisation is a rebuild, not `convert_node_labels_to_integers`.**
`adjacency.normalised(G)` returns a fresh `nx.Graph` with nodes `0..n−1` **added in ascending
order**, labelled by `sorted(G.nodes())`, self-loops dropped. Forced by §4.2: this is the only
construction under which `networkx`'s two writers and my triangle agree. Consequences, all
tested: the three backends are invariant to insertion order and sensitive only to label values
(that is determinism, and the docstrings say so); all three encode the same labelling, which is
what makes criterion 6 mechanical; the F3 columns collapse into one (§3.2a).

**2. `normalised` and `upper_triangle_columnwise` live in `adjacency.py` and are imported by the
other two.** The design assigns me "the reading order the whole `n²` family shares"; one
definition with one owner is the only way that survives contact with a fourth member. `graph6.py`
and `sparse6.py` import from `adjacency.py`; no cycle, and agent B's `agm_cam` can import the
same two functions rather than re-deriving the triangle.

**3. `graph6`'s frame is built by unpacking its own wire**, not by re-reading the adjacency.
It costs `O(n²)` — which graph6 already is — and it makes the family identity true *by
construction at encode time* rather than only in a test.

**4. `adjacency.wire` is `None`.** The adjacency matrix has no format-defined serialisation to
measure, `bits.py` derives its realised cost from `n` alone, and `adjacency-matrix.md` §4 says in
terms *"do not invent a packing"*. Populating `wire` with an invented byte string is exactly the
trap.

**5. `decode()` raises `ValueError` on a foreign `Encoding`.** Not in the exception hierarchy;
`NotReversible` would be wrong (these three *are* reversible) and there is no frozen name for
"you handed me another backend's wire". Tested.

**6. Test-file layout: `pytest.importorskip("networkx")` goes *after* the imports.** Required by
`ruff` E402 — the per-file ignore covers `tests/integration/`, not `tests/unit/` — and it is
also the better test: every module imported above it must be importable with `networkx` absent,
which is the subpackage's dependency contract.

**7. `slow` markers on the two real-cohort tests.** They run by default (3 s total) and skip
cleanly with `DatasetNotFoundError` when `$ISALGRAPH_COHORT_ROOT` is unset, so the suite stays
portable to a machine without the cohorts.

---

## 6. Criterion 7 — local smoke JSON

`./wtpy -m isalgraph.competitors.smoke --backends adjacency,graph6,sparse6 --dataset iam_letter_low --n-graphs 200 --seed 42 --out smoke_A.json`

```
adjacency          ok= 200 failed=  0 p50=   0.014ms F3=6/50
graph6             ok= 200 failed=  0 p50=   0.034ms F3=6/50
sparse6            ok= 200 failed=  0 p50=   0.029ms F3=6/50
```

```json
{
  "backends": {
    "adjacency": {
      "available": true, "backend": "adjacency",
      "bits": {"entropy_p50": 6.0, "realised_p50": 8.0},
      "capabilities": ["handles_disconnected", "positional_frame", "reversible"],
      "f3_invariant_of_50": "6/50", "failures": [],
      "ms_per_graph": {"max": 0.3317219998280052, "p50": 0.014183500752551481,
                       "p90": 0.018599999748403206},
      "n_encoded": 200, "n_failed": 0, "n_failures_recorded": 0
    },
    "graph6": {
      "available": true, "backend": "graph6",
      "bits": {"entropy_p50": 12.0, "realised_p50": 16.0},
      "capabilities": ["handles_disconnected", "positional_frame", "reversible"],
      "f3_invariant_of_50": "6/50", "failures": [],
      "ms_per_graph": {"max": 0.31298899921239354, "p50": 0.034327998946537264,
                       "p90": 0.04480600182432681},
      "n_encoded": 200, "n_failed": 0, "n_failures_recorded": 0
    },
    "sparse6": {
      "available": true, "backend": "sparse6",
      "bits": {"entropy_p50": 18.0, "realised_p50": 32.0},
      "capabilities": ["handles_disconnected", "reversible"],
      "f3_invariant_of_50": "6/50", "failures": [],
      "ms_per_graph": {"max": 0.07090099825290963, "p50": 0.029085500500514172,
                       "p90": 0.035385000956011936},
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
    "registered_backends": ["adjacency", "graph6", "isalgraph_canonical",
                            "isalgraph_pruned", "size_null", "sparse6"],
    "seed": 42
  },
  "n_graphs_drawn": 200, "suite": "suite1"
}
```

`isalgraph_engine` reads `python` because the worktree shim removes the
`ScikitBuildRedirectingFinder`, per CONTRACTS §0.5. None of these three backends touches the
engine, so the timings are unaffected — they are **pure-Python, single-threaded** either way.
`sparse6.entropy_p50 = 18.0` is the `':'`-excluded convention of §3.3.

Also run, to exercise the four-byte `N(n)` branch on real data (`mutagenicity` reaches `n = 97`):

```
adjacency          ok= 200 failed=  0 p50=   0.065ms F3=0/50
graph6             ok= 200 failed=  0 p50=   0.240ms F3=0/50
sparse6            ok= 200 failed=  0 p50=   0.091ms F3=0/50
```

`sparse6` costs **0.85×** `graph6` at `n̄ = 4.1` (0.029 vs 0.034 ms) and **0.38×** at `n̄ = 27.9`
(0.091 vs 0.240 ms): it never materialises the `n²` triangle, so its advantage widens with `n`,
consistent with `sparse6.md` §1's "cheaper than graph6 above `n = 9`". `adjacency` is the
cheapest of the three at Letter sizes (0.014 ms) and second at Mutagenicity (0.065 ms). All three
encode 200/200 with zero failures on both suites.

---

## 7. Open questions

1. **Who corrects `bits.py:46`?** §4.1 is the orchestrator's file. Until it is fixed, any printed
   `realised_bits` for `adjacency` or `agm_cam` is half its true value. `entropy_bits` is fine, so
   Claim A's headline table is safe; the realised-bytes column is not.
2. **Does the `sparse6` `':'` decision (§3.3) propagate to `sparse6_nauty`?** `bits.py` applies
   the same row to both, so yes — but README §4.3 has no `sparse6_nauty` column, and gate 1b will
   need to state the convention once for both.
3. **Should `adjacency-matrix.md` §2 and `sparse6.md` §4 be corrected in place?** Both now
   contain literals the shipped code contradicts (row-major triangle; `6·len(sparse6)`). They are
   read-only to this track. Flagged for `review-close`.
4. **`agm_cam` should import `adjacency.normalised` and `adjacency.upper_triangle_columnwise`.**
   Its canonical permutation replaces the *sorted* one, but the triangle walk must be the same
   code, or the family identity holds in three files and is asserted in one. Agent B's call, and
   the functions are exported for it.
5. **Criterion 2's per-dataset targets should be restated as the qualitative claim.** §3.2 shows
   the count is `#{complete graphs in the draw}`, which is a property of the sample, not of the
   representation. The representation-level fact is `0/50` on LINUX and AIDS.
