# T-06-exhaustive — work log

Re-execution of the T-06 IsalGraph arm with the **exhaustive** canonical form
(`canonical_string`) in place of the length-suboptimal `pruned_canonical_string`.

Started 2026-08-25. Base commit `855e4adc63dd20850b406b136785117d97a1145a` (`main`).
Output tree: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06_exhaustive/`

**The original `data/source/T06/` and `results/reports/T-06-full-recompute/` are
read-only for this ticket.** Nothing here writes into either.

---

## Status board

> ## ⚠ DIRECTION CHANGE 2026-08-25 — no local compute
>
> PI decision, relayed mid-run: **everything compute-heavy runs on Picasso.**
> The local encode was killed at 11 of 15 cells, its partial output left on disk
> as a fallback, and the distance and F2 stages were never run locally. What
> this ticket now delivers locally is **code and SLURM scripts only**; the
> orchestrator owns every submission.

| step | state | note |
|---|---|---|
| 0. read plan + contracts | DONE | |
| 1. `isalgraph_exhaustive` backend + tests | DONE | `a691b57` |
| 2. commit code | DONE | `4cbc6cd` stats knobs, `034b3b6` drivers |
| 3. local encode | **STOPPED** | killed at 11/15 by the direction change; output preserved |
| 3b. `isalgraph_greedy` ablation arm | DONE | `b942247`, 35 tests green |
| 3c. Picasso encode scripts | DONE | `b88e4b9`; submitted by the orchestrator as array 2102929 |
| 4. Picasso distances + F2, chained | DONE | `598ceb1`, **not submitted** |
| 5. run the campaign | orchestrator | |
| 6. SUMMARY.md / manifest.json | blocked | needs the cluster tree |

### What the local run completed before it was stopped

Eleven cells, all `isalgraph_exhaustive`, preserved under
`data/source/T06_exhaustive/encodings/`:

- **Suite 1, all 5 cells** — 5,350 graphs, **0 censored, 0 error**.
- **Suite 2, 6 of 10** — `linux`, `grec`, `protein`, `aids_graphedx`,
  `iam_letter_low`, `iam_letter_med`.

`suite2/protein` is the one cell where the canonical search actually bites:
**1,873 s** and **320 graphs censored**, every one retained with a substitute
string under the cascade. Every other completed cell finished in 0-1 s with zero
censoring. That spread is the whole argument for the per-band reporting in
`summarise.py` — a pooled completion rate over a cohort that is 65 % at
`n <= 12` would have hidden it.

**Not run locally, by instruction:** the greedy arm's 15 cells, every distance
cell, F2, and the ablation's Part A.

Base commit at launch: `6befc1a` (another session moved HEAD from `855e4ad`
mid-work; re-checked before committing, per the standing rule).

---

## Architecture findings that shaped the design

Read before changing anything.

### F1. D14's fallback is a DRIVER policy, not a backend one

`t06_encode_worker.py` docstring, verbatim: *"The worker never decides D14. It
reports what happened -- `ok`, or `error` with the exception class name -- and
the driver applies the fallback policy. Keeping the policy in one place is what
stops a censored graph being silently dropped in one code path and retained in
another."*

The mechanism is a **second pass**: `t06_encode._apply_d14` collects every record
whose `error_family(error_kind) == "wall_clock"` and re-runs those indices with
`mode="fallback"`; `_stamp_fallback` then rewrites the record to
`status="censored", fallback_used=True`.

**Consequence for this ticket.** Putting the fallback inside
`_IsalGraphBackend.encode()` would be wrong twice over:

1. The worker would never learn a fallback happened, so the row would land as
   `status="ok", fallback_used=False`. A censored graph laundered into a
   completed one is precisely the bias D14 exists to expose.
2. The budget is enforced by the parent **killing the process**
   (`t06_encode._consume`). A backend-internal `except CanonicalizationTimeoutError`
   never runs for a killed graph, so the graphs most in need of a fallback would
   get none.

So the fallback is wired through the existing driver machinery, and the backend
only *names* its fallback variant. The backend change is: encode with
`canonical_string`, drop `SUITE1_ONLY`, declare `fallback_variant = "pruned"`.

### F2. The fallback target differs from the pruned arm's

D14's fallback for `isalgraph_pruned` is the **greedy-min** string
(`GreedyMinG2S`). For `isalgraph_exhaustive` the task specifies
`pruned_canonical_string`, which is strictly better: it is still a canonical
form, so the row stays inside the completeness theorem, whereas a greedy-min row
does not.

**Cascade, to honour "do not drop a graph":**

    canonical_string(30 s)  ->  pruned_canonical_string(30 s)  ->  greedy_min_string

`pruned` has its own ceiling (T-06 measured 24/400 on Mutagenicity, 4/400 on
Protein at a 2 s budget), so a two-tier cascade would still drop rows. Greedy-min
is O(n) greedy encodes and always terminates, so the third tier closes it.

The parent's per-line deadline in fallback mode is `budget_s + LINE_GRACE_S`
= 45 s, and greedy-min at n~100 is milliseconds, so the cascade fits inside it.

### F3. `\x1f` separator does not apply to this arm

`CONTRACTS.md` §3.1: `symbol_sep` is `"\x1f"` only for `min_dfs`, `size_null`,
`wl_subtree`. IsalGraph symbols are single characters from `{N,n,P,p,V,v,C,c,W}`,
so `symbol_sep = ""` and `length == len(encoding)`. The new arm inherits this by
**not** being added to `SYMBOL_SEP`, and `_join_symbols` asserts it per encode.

### F4. Fifteen cells, not fifteen datasets

10 Suite-2 (`linux grec protein aids_graphedx iam_letter_low iam_letter_med
aids_iam iam_letter_high coil_del mutagenicity`) + 5 Suite-1 (`linux aids
iam_letter_low iam_letter_med iam_letter_high`).

### F5. Distances reuse — symlink, never copy

The stats need every comparator's distance matrix, not just the new arm's.
`data/source/T06/distances/` is 518 MB / 190 files. It is **symlinked** into
`T06_exhaustive/distances/`, not copied and never moved: a symlink is read-only
consumption, costs no disk, and cannot mutate the pre-registered record.
`run_distances.sh`'s `[ -s "$target" ]` guard follows symlinks, so the reused
cells are skipped rather than recomputed.

---

## Decisions

| id | decision | rationale |
|---|---|---|
| X1 | fallback wired through driver `_apply_d14`, not inside `encode()` | F1 — a backend-internal fallback reports `ok/False` and never fires for a killed graph |
| X2 | 3-tier cascade exhaustive -> pruned -> greedy-min | F2 — pruned alone still drops rows, and "do not drop a graph" is the hard constraint |
| X3 | budget 30 s, recorded in every cell's `metadata.encode_budget_s` | a censoring rate is a property of its budget |
| X4 | every competitor stays in the data; reduced view is a flag | dropping from the campaign changes a pre-registered family's cardinality (`N_actual = 79`) |
| X5 | distances symlinked from T06 | F5 |
| X6 | `isalgraph_greedy` is NOT in `ISALGRAPH_ARMS` | that tuple governs the wall clock and the D14 fallback; greedy needs neither — no search to interrupt, and it *is* the terminal fallback tier |
| X7 | `takes_timeout = False` on the greedy arm | otherwise the default budget is refused as unenforceable and the arm is unusable in a campaign that budgets the others |
| X8 | Picasso worker sets `PYTHONPATH=$REPO`, **not** `$REPO/src` | the `picasso-sbatch` template says src-first; for this project that shadows the installed package and silently drops to pure Python. CLAUDE.md wins over the skill template |
| X9 | Picasso array sized in tasks, not units | SCBI's 2 h floor; unit cost is uneven enough that one task per unit would be two dozen seconds-long jobs |

---

## The canonicalisation ablation, measured

`invariance_ablation.py`, 720 relabelling draws over 120 graphs, seed 42.

| non-invariance | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|
| `isalgraph_greedy` | 50.7 % | 68.8 % | 81.2 % | **91.0 %** | 84.7 % |
| `isalgraph_exhaustive` | 0 % | 0 % | 0 % | 0 % | 0 % |
| `isalgraph_pruned` | 0 % | 0 % | 0 % | 0 % | 0 % |

0 of 144 draws at every `n` for both canonical arms.

| paired length | mean | greedy shorter / equal / longer |
|---|---|---|
| `isalgraph_greedy` | 11.46 | — |
| vs `isalgraph_exhaustive` | 11.10 | **0 / 85 / 35** |
| vs `isalgraph_pruned` | 11.43 | 17 / 86 / 17 |

**Two things this says that the brief's framing did not.** Against the
*exhaustive* arm greedy is **never shorter** — 0 of 120 — and longer on 35; the
"mean 15.20 vs 15.14 wash" holds only against *pruned*. And non-invariance
**rises with n**, so it is not a small-graph artefact.

So the sentence for R1.2 is: dropping the canonical search buys nothing in bits
against the true `w*_G` and costs invariance on half to nine tenths of
relabellings.

---

## Test suite

| when | passed | skipped | note |
|---|---|---|---|
| reference (T-09 close) | 2,583 | 321 | the floor in `.claude/CLAUDE.md` |
| this ticket | **2,610** | 321 | +27, all from `tests/unit/test_t06_exhaustive.py` |

Measured in 9 min 26 s. The one failure the first full run reported
(`test_admissibility_e2::test_quick_run_classifies_every_representation`) is
fixed and re-verified 15/15; that run had started before the fix landed.

**Why E2 failed, and why the fix is a test fix and not a grid regeneration.**
Part C classifies a representation from its F3 record in T-04a's **frozen**
admissibility grid (`/media/.../T-04a/grid_200.json`). A backend registered
after that grid was frozen has no record, so `e2_completeness` reports
`class = None` with reason *"no admissible distance and no F3 record; not
classified"* — which is the correct answer. The test asserted every complete
invariant is class III, a premise that is now false. Regenerating the grid to
absorb the new arm would move a pre-registered artifact, so the assertion is
scoped to what the grid covers and the uncovered arms are asserted to be
unclassified *for the stated reason* rather than skipped.

---

## Timeline

- `2026-08-25` — read plan, contracts, drivers; wrote this log.
- `2026-08-25` — backend + 27 tests; ruff clean, `mypy --strict` clean.
- `2026-08-25` — committed `a691b57` (backend) and `4cbc6cd` (stats knobs).
- `2026-08-25` — launched the encoding campaign. Suite 1 finished in seconds:
  5,350 graphs, **0 censored, 0 error** at the 30 s budget.

---

## Independent verification of a written cell

`suite1/linux`, all 89 graphs, re-derived from the cohort without going through
the campaign code path:

| check | mismatches |
|---|---|
| `encoding` equals `canonical_string(to_sparse_graph(G))` | **0** / 89 |
| `length` equals `len(string)` (§3.1, empty separator) | **0** / 89 |
| `entropy_bits` equals `L·log2(9)` | **0** / 89 |
| `S2G(encoding)` isomorphic to the cohort graph | **0** / 89 |

Provenance in the file: budget 30.0 s, engine `cpp`, build hash
`298fc1188bf1b051`, seed 42, `src_commit = 4cbc6cd`.

This is the gate for building statistics on the new arm. It passed before any
distance was computed.

---

# Landing the campaign — 2026-08-26

Picasso results copied back, figures and tables regenerated, headline numbers
measured. Written by the session that picked up `T-06-EXHAUSTIVE-HANDOFF.md`.

## The answer to the question the campaign was launched for

**No. The exhaustive arm does not beat nauty-sparse6 at `n = 40`.**

Median entropy bits, via `tables._bits_at` over 185 encoding cells — the same
code path that writes the paper's tables:

| arm | `n = 20` | `n = 40` |
|---|---|---|
| `isalgraph_exhaustive` (hybrid) | **114.1** | **342.4** |
| `isalgraph_pruned` | 136.3 | 348.7 |
| `sparse6_nauty` | 144.0 | 336.0 |
| `isalgraph_greedy` | 130.0 | 355.0 |

It needed to reach below 336.0 and reached 342.4, closing **6.3 of the
12.7-bit gap**. The pruned and nauty-sparse6 columns reproduce the handoff's
136 / 349 and 144 / 336 exactly, which is what certifies the staging.

At `n = 20` it wins outright and by a wide margin: 114.1 against 144.0, a
20.8 % margin, against the pruned arm's 5.4 %.

## Why it stops working, and it is not the encoding

The D14 cascade means a graph that exhausts the 30 s budget carries the
*pruned* string. That substitution rate rises monotonically with `n`
(all 21,720 graphs, 15 cells):

| `n` band | ≤12 | 13–20 | 21–30 | 31–40 | 41–60 | 61+ |
|---|---|---|---|---|---|---|
| graphs | 14,172 | 2,156 | 2,690 | 1,602 | 860 | 240 |
| fallback | 0.0 % | 4.5 % | 41.2 % | 60.4 % | 98.4 % | 100 % |

Overall 15.00 % (3,259 graphs). **At the `n = 40` anchor stratum itself — 93
graphs — the rate is 96.8 %: three graphs there carry a genuine exhaustive
string.** So `342.4` is very nearly the pruned number.

Split by completion, the decay disappears. Paired against pruned on identical
graphs:

| band | fallback | pooled gain | completed only | gain there |
|---|---|---|---|---|
| 13–20 | 4.5 % | 13.2 % | 2,058 | 12.5 % |
| 21–30 | 41.2 % | 15.9 % | 1,583 | 16.7 % |
| 31–40 | 60.4 % | **8.5 %** | 634 | **17.0 %** |
| 41–60 | 98.4 % | **−0.3 %** | 14 | 12.6 % |

The pooled column collapses to −0.3 %; the completed column holds flat at
**12.5–17.0 % in every band** and is at its *highest* in 31–40. The exhaustive
form is not weaker on large graphs — the clock runs out.

> **`T-06-POSITIONING.md` §5's projection is not refuted, its premise is.** It
> read "the `n = 40` figure moves from 349 toward ≈ 310 — below nauty-sparse6's
> 336", extrapolating an 11 % reduction measured at `n ≤ 26`. Measured at
> `n = 40`, that reduction is available on 3.2 % of the stratum. The projection
> assumed the search would finish; at 30 s it does not.

## Two minimality invariants, verified at 6× the campaign's parity scale

On all **18,461** completed graphs, where the string really is `w*_G`:

| check | violations |
|---|---|
| greedy-min strictly shorter than `w*_G` | **0** |
| pruned strictly shorter than `w*_G` | **0** |
| `w*_G` strictly shorter than pruned | 7,316 of 18,461 (39.6 %) |

Both are implied by `w*_G` being the minimum-length encoding and neither had
been checked at this scale; the C++ parity suite covers 3,079 graphs.

## The ablation at full scale, and a phrasing that does not survive it

Over all 21,720 graphs, greedy against **pruned**: shorter on 5,821, equal on
12,393, longer on 3,506. Against the **hybrid**: shorter on 1,999, equal on
11,627, longer on 8,094.

> ⚠ The section above reads "against the *exhaustive* arm greedy is **never
> shorter** — 0 of 120". That was measured on 120 graphs at `n = 5-9` and it
> **does not generalise to the delivered column**: greedy is shorter than
> `isalgraph_exhaustive` on 1,999 of 21,720 graphs. Every one of those is a
> graph where the hybrid had fallen back, so the claim is exact when scoped to
> *completed searches* — which the 0-violation table above proves — and false
> when stated over the arm. **Scope it in the letter.**

`psi` proper was never measured for the greedy arm: the ablation's Part A never
ran, and what the table above reports is the non-invariance *rate*, a different
statistic (`psi` is a separation ratio; sparse6's reaches 1.15, so a rate cannot
be printed in that column). `PSI_MEASURED["isalgraph_greedy"]` is therefore
`--`. One call fills it if wanted:
`run_e1(grid, backends=["isalgraph_greedy"], parts="B")`, which reads T-04a's
frozen grid for its distance block only and does not regenerate it.

## The handoff was wrong that no code edit was needed

`T-06-EXHAUSTIVE-HANDOFF.md` §3 said adding the two arms "needs **no code
edit**" and that editing a figure module would be "a bug in the module, not a
task". **The arms would have silently vanished from every figure and table.**

`design.REPRESENTATIONS` is a hand-written literal of 11 entries, and
`design.present()` drops unregistered keys *deliberately* — "a figure must never
invent a style for a backend the registry does not know". Every consumer filters
through it: `fig_ic` via `present()`, `eval_size_profile.figures` via
`design.ORDER` and `design.BY_KEY`, `tables` by iterating `REPRESENTATIONS`. No
error is raised anywhere.

The fix is to register them in `design.py`, which is the single design source and
not a figure module — `fig_ic.py`, `tables.py` and `figures.py` were not edited
to make an arm appear. Changes:

- `design.py`: two `Representation` entries. `isalgraph_exhaustive` is
  `CANONICAL_CODE`, `is_ours`, canonical/complete/reversible/metric-admissible,
  `max_n = 98`. `isalgraph_greedy` is `RAW_SERIALISATION` — the taxonomy's axis
  is canonicalisation and greedy fails F3 worse than `adjacency` or `graph6` —
  `is_ours` so it stays out of the *comparator* head-to-head table, not
  canonical, not complete, `max_n = 98`.
- `tables.py`: `PSI_MEASURED`, `COMPLETION_FLOOR` (both 1.0 — complete by
  construction under D14, which is a different quantity from the censoring rate),
  `EXECUTABLE` (both — every prefix is still a construction program),
  `PAYLOAD_PER_BYTE` (3.17 for both; same `|Σ| = 9`). One rendering fix so an
  unmeasurable `--` is not underlined as "worst", which is the caption's own rule.

**Colour and name were not inherited from anywhere and are worth a review.**
`isalgraph_exhaustive` is `"IsalGraph (hyb.)"` at `#EE3377` with a diamond
marker. The first choice, `#882255`, was unreadable against the pruned arm's
`#AA3377` exactly where the two curves separate. The greedy arm's family puts it
under a legend heading reading "serialisations", which is not what an instruction
string is; the alternative is a new `Family` member, which extends a
pre-registered taxonomy and was not taken unilaterally.

## Staging, and one gap the campaign tree had

Picasso's `T06_exhaustive/encodings/` holds **only the 30 new cells**. The
competitor cells are symlinks created by the local run (F5's rule applied to
encodings), and that staging had **omitted `isalgraph_pruned` and
`isalgraph_canonical`** — while `distances/` symlinked them. Left alone, `fig4`
would have had no reference arm at all: `design.REFERENCE_KEY` is
`isalgraph_pruned`.

Twenty symlinks were added to match what `distances/` already did, giving 185
cells over 13 representations. Nothing under `data/source/T06/` was written.

`distances/` was rsynced **selectively** — the 60 new-arm matrices only. Picasso
holds all 250 as real copies (699 MB); the other 190 are already local as
symlinks into `T06/distances/`, and pulling them would replace a symlink with a
copy, which is exactly what F5 exists to prevent, for 518 MB.

## Provenance — §7 of the handoff is narrower than stated

The handoff says "every artifact this campaign writes carries an empty commit
field". Measured on the written cells:

| field | value |
|---|---|
| `code_commit` | `d6a9f4b1033d3d8c6757f5b2ae95d1b77f532bd2` — **correct** |
| `src_commit` | `unknown` — the hard-coded-path defect |
| `isalgraph_engine` | `cpp` |
| `isalgraph_build_hash` | `298fc1188bf1b051` |
| `encode_budget_s` | 30.0 |

So the artifacts **are** attributable to a commit; only `src_commit` is lost.
The engine trap (X8 / handoff §6.1) was avoided — no cell reports `python`.

## What fig1–fig3 cannot show, and why that is not a defect

`fig1_rho_vs_size`, `fig2_rho_by_representation` and `fig3_absolute_scale` are
built from the frozen `results/reports/T-06-full-recompute/data/size_profile.json`,
which contains **7 representations** and neither new arm. They regenerate without
the new arms and that is correct: the profile is pre-registered and §4 forbids
touching it.

Putting the new arms in those three figures requires a fresh `size_profile.py`
run — `--encodings`, `--ged-root`, `--approx-root`, ~45 min on 24 cores,
bootstrap-dominated — written to a **new** output path. That is a Picasso job and
it is outside the handoff's three steps. **The acceptance criterion "every figure
legend shows both arms" is therefore satisfiable for `fig4` only**, as written.

## Verification

| check | result |
|---|---|
| `pytest tests/ -q` | **2,618 passed / 321 skipped** — exactly the reference state |
| `ruff check src/` | clean |
| `mypy --strict src/isalgraph/` | clean, 80 files |
| four tables under `pdflatex` + booktabs/rotating/amsmath | 0 errors |
| files modified under `data/source/T06/` | **0** |
| files modified under the report's `data/` | **0** |
| T-04a `grid_200.json` | untouched |

The suite was run on a rebuilt `isalgraph-cpp` env — the workstation had none,
and the Sandisk was unmounted. Both were restored before any measurement; the
engine reports `cpp` at build hash `298fc1188bf1b051`, matching Picasso.

## Timeline

- `2026-08-25` — encode array 2102929 COMPLETED 30/30, distances 2106062 COMPLETED.
- `2026-08-25` — rsynced encodings (30 cells) and the 60 new distance matrices.
- `2026-08-25` — registered both arms; regenerated `fig4` and all four tables.
- `2026-08-26` — F2 shards 1–4 COMPLETED clean (`fail=0`), peak 4 h 49 m of a
  12 h limit. Cost ran ≈ 4.8-5.1 s per graph, near-linear rather than quadratic
  in pairs.
- `2026-08-26` — shard 0 COMPLETED at 6 h 50 m, 15/15 partials. **Merge 2106064
  FAILED.** Cause and fix below.
- `2026-08-26` — re-submitted the chain as 2106209 → 2106210 → 2106211.

---

## 🔴 F2 run 1 produced an empty Claim A family, and the merge is where it surfaced

**All fifteen shards reported `ok=3 skip=0 fail=0` and every one of them wrote a
partial with `a1_cells: []`.** 6 h 50 m of shard compute produced a complete
Claim B half and no Claim A half at all.

| partial | `a1_cells` | `rho_rows` | `mrm` |
|---|---|---|---|
| the 5 suite1 cells | 0 | 12 | 1 |
| the 10 suite2 cells | **0** | 24 | 2 |

The merge then died in the A2 post-hoc:

```
MultiplicityError: need >= 2 named methods, got 1
=== N_actual: 0 ===   === rho rows: 0 ===   === discrepancy: 999 ===
```

### The cause

`T06_exhaustive/encodings/` **on Picasso** held only the 30 cells the encode
array wrote — `isalgraph_exhaustive` and `isalgraph_greedy`. No competitor
encodings, no `isalgraph_pruned`. F5's symlink staging was applied to
`distances/` and never to `encodings/`.

Claim B reads `distances/`, which was fully staged as 250 real files, so it
computed correctly for all 12 representations. **Claim A reads `encodings/`, and
found one arm to compare against itself.** A single-method A1 family yields one
named method, and `wilcoxon_holm_posthoc` refuses it.

> **Note the asymmetry that let this through.** `f2_worker.sh` has an explicit
> guard for the distance side — *"no `${T06_REFERENCE_ARM}` levenshtein matrix
> under `${DIST}`… every shard would skip its B rows and the family would come
> back empty with a zero exit status"* — and **no equivalent guard on
> `${ENC}`**. That is precisely the failure it describes, on the other axis, and
> the campaign paid 6 h 50 m to discover it at the merge.

**The guard is now in `f2_worker.sh`**, beside the distance one it mirrors. Two
things about it were not obvious and both were wrong in the first draft:

1. **The test is not "how many representations".** A1 emits one cell per
   *comparator*, never per IsalGraph arm — measured on `suite2/linux`, the 12
   cells are 6 competitors (`adjacency`, `agm_cam`, `graph6`, `min_dfs`,
   `nauty_graph6`, `sparse6`) × 2 arms (`primary`, `complete_case`).
   `sparse6_nauty`, `wl_subtree`, `size_null` and every `isalgraph_*` arm are
   absent. Run 1's tree held **two** representations and still emitted nothing,
   so a `< 2` threshold would have passed the exact case it was written for. The
   condition is *"is there anything here that is not one of ours"*.
2. **`grep -c` counts a blank line.** `printf '%s\n' "$EMPTY" | grep -cv '^isalgraph_'`
   returns **1**, so an entirely empty encodings tree — the worst case — sailed
   through. `awk 'NF && $0 !~ /^isalgraph_/' | wc -l` is used instead. `find`
   rather than `ls`, too: a cell may legitimately be a symlink, and an `ls` that
   renders `name -> target` feeds the target path to `sed`.

Verified on four trees: full (13 reps / 9 comparators → pass), run 1's
(2 / 0 → FATAL), empty (0 / 0 → FATAL), single comparator (1 / 1 → pass).

### The fix, verified before resubmitting

155 encoding cells (8.7 MB) staged to Picasso by `rsync -aL` from the Sandisk
tree, resolving the symlinks, giving 185 cells there. Re-ran one cheap cell on
the login node:

```
suite2/linux   ->  24 rho rows, 12 a1 cells, 2 mrm fits    (~15 s)
```

12 A1 cells where there were 0. The old partials are **preserved**, not deleted,
at `families/f2_partials_claimB_only_20260826/` — they are 6 h 50 m of correct
Claim B work and the record of what the defective run produced.

**The re-run costs the ρ work again.** `f2_worker.sh` skips on `[ -s "$partial" ]`
and `t06_f2.py` has no claim selector, so there is no way to compute A1 alone and
patch it in — and hand-merging JSON into a pre-registered confirmatory family
would not be defensible anyway.

### This did not block the three handoff steps

None of the four regeneration commands reads `families/`. `fig4`, the four tables
and every headline number in this log come from `encodings/` plus the frozen
report data, and were all produced before the merge failed. What run 1 lost is
the confirmatory-family record for the exhaustive arm — `N_actual`,
`closed_form_expression`, the BH rejections — not any figure.

---

## The family, and two more defects the re-run exposed

The shard re-run (2106210) was clean — 120 A1 cells, 12 per suite2 partial,
300 ρ rows, 25 MRM fits — but the **merge took three attempts**, and the two
failures in between are worth more than the result.

### Defect 2 — `completion_rates.json` describes whatever `ENC` held when it was written

Merge 2106211 reported `N_actual = 86`, `closed_form = 86`, **`discrepancy = 0`**.
Internally consistent and wrong. `completion_rates.json` had been generated during
run 1, when `ENC` held 30 cells, so it carried rows for `isalgraph_exhaustive` and
`isalgraph_greedy` **alone**. `c` is the count of cells excluded for completion,
computed at merge time from that file — with no competitor row present, nothing
could be excluded, and `c` came out 0 against T-06's 7.

The worker's `if [ ! -s "${COMPLETION}" ]` reuses the file unconditionally. It now
also regenerates when the file covers fewer representations than `ENC` does.

### Defect 3 — my own over-staging, and the FamilyError that caught it

Regenerating the table over the full 13-representation tree gave
`FamilyError: c names an undeclared representation 'isalgraph_pruned'`.

`isalgraph_pruned` and `isalgraph_canonical` were symlinked into the **Sandisk**
tree because `design.REFERENCE_KEY` is `isalgraph_pruned` and `fig4` has no
reference arm without them. `rsync -aL` then carried them to Picasso, where F2
saw them. They belong to the figures, not to this family: the declared set is the
**seven comparators**, and no IsalGraph arm is a member — the reference arm is
what they are compared *against*.

Moved to `_encodings_figure_only/` on the cluster. Picasso's F2 tree is 165 cells
over 11 representations; the Sandisk tree stays at 185 over 13 for the figures.
**The two trees legitimately differ, and acceptance criterion 1 should be read
that way.**

### The result

```
F2: N_actual=79  closed_form=79  discrepancy=+0   (101 - 5*3 - 7)
    BH over N_actual : 76 rejected at q=0.05
    BH over N_max=182: 73 rejected at q=0.05
    120 a1 cells · 300 rho rows · 25 mrm fits
```

| | T-06 (pruned) | T-06-exhaustive |
|---|---|---|
| `N_actual` | 79 | **79** |
| closed form | `101 - 5*3 - 7` | **`101 - 5*3 - 7`** |
| `c` cells | `agm_cam` ×6, `min_dfs` ×1 | **the same seven** |
| BH rejections | 75 of 79 | **76 of 79** |
| declared reps | the 7 comparators | the same 7 |

### 🔴 The campaign cannot move `c`, and the handoff says it exists to

`T-06-EXHAUSTIVE-HANDOFF.md` §4 reads *"`c` is the completion term the campaign
exists to move"*. **It is structurally unmovable by this campaign.**

`c` counts *comparator* cells failing completion, and all seven are `agm_cam`
(6) and `min_dfs` (1) — competitor encodings **reused verbatim** from
`T06/encodings/`, which this campaign does not recompute. The one arm whose
completion the campaign *does* change is the reference arm, and the reference arm
is **exempt from `c`** by pre-registration (`preregistration.md` §5.1 consequence
2, D14 governs the reference arm) — the merge log says so four times, for
`aids_iam` 0.9630, `coil_del` 0.5021, `mutagenicity` 0.7802 and `protein` 0.3638.

So `N_actual = 79` was the only value it could take, and the handoff's warning
not to *assert* 79 was right for the wrong reason: 79 is not an assumption to be
avoided, it is a consequence. What did move is the evidence inside the family —
**76 rejections against 75**.
