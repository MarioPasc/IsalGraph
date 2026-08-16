# encoding — work log

**Branch** `t06/encoding` · **Base** `8afa59e` (rebased onto `ccb0b48` mid-wave) · **Head** `90ed9c2`

## What I built

Five modules under `benchmarks/real_data/eval_encoding/` plus one test file. The campaign turns the
frozen cohort into per-graph encodings and bit counts under every representation, enforcing the
300 s per-graph budget with a **killed subprocess** and applying **D14** — a graph whose canonical
encoding times out re-enters with its greedy-min string and a flag, it is never dropped. Claim A and
the per-`(representation, dataset)` completion rates are computed from that output.

The driver is a *streaming* killer rather than one `subprocess.run` per graph. The worker loads the
cohort once and prints one JSON line per graph; the parent reads with a per-line deadline
(`budget + 15 s`, `budget + 90 s` for the first line) and kills the child when a line is late, then
restarts from the next index. This keeps the wall clock hard while paying interpreter start-up once
per chunk. At 16,370 graphs × 11 representations, one process per graph would have spent tens of
hours in `import numpy` alone.

| File | Purpose | Lines |
|---|---|---|
| `t06_cohort.py` | Cohort loader, Suite-1 filter, `--verify` | 404 |
| `t06_encode_worker.py` | Encode one graph, one JSON line, killable | 465 |
| `t06_encode.py` | Campaign driver: kill, restart, D14, `.npz` | 647 |
| `t06_claim_a.py` | Paired bit comparison + Clopper–Pearson CI | 420 |
| `t06_completion.py` | Completion rates split by failure family | 223 |
| `tests/unit/test_t06_encoding.py` | 75 acceptance tests | 610 |

## Acceptance criteria

| # | Criterion | Command | Result (verbatim) |
|---|---|---|---|
| 1 | Cohort counts | `$PY -m benchmarks.real_data.eval_encoding.t06_cohort --verify` | `Suite 2 = 16370 graphs over 10 keys` / `Suite 1 = 5350 graphs over 5 keys` with `iam_letter_low 1180`, `iam_letter_med 1253`, `iam_letter_high 2059`, `linux 89`, `aids 769`; `OK` |
| 1 | …asserted by a test | `pytest -q -k "suite2_totals or suite1_per_dataset"` | passed (inside the 75) |
| 2 | Round trip, ≥200 real graphs, ≥3 datasets | `pytest tests/unit/test_t06_encoding.py -q -k roundtrip` | 7 params passed — `graph6, sparse6, nauty_graph6, adjacency, agm_cam, min_dfs, isalgraph_pruned` over `iam_letter_low`/`linux`/`grec`, 210 graphs each (`agm_cam` exempt from the ≥200 floor because it refuses above n = 12) |
| 3 | Schema conformance on a real emitted file | `pytest -q -k "schema_keys or dtypes or provenance"` | passed; exact key set `{graph_ids, node_counts, edge_counts, encoding, length, error_kind, entropy_bits, realised_bits, status, fallback_used, seconds, metadata}`, `isalgraph_build_hash` and `src_commit` both non-empty and not `unknown` |
| 4 | D14 invariant, fired not vacuous | `pytest -q -k censor` | passed. `protein` limit 8 at a 1 ms budget: **7 censored, 1 ok**. `censored ⟹ fallback_used ∧ encoding != ''` holds; every substituted string is over `NnPpVvCcW` |
| 5 | No signal-based timeout | `pytest -q -k signal` | 5 params + 1 passed. Checked on docstring-and-comment-stripped source (AST), so the prose explaining the prohibition does not self-trigger; also asserts `subprocess.Popen` and `proc.kill()` are present |
| 6 | Both bit conventions | `t06_claim_a --encodings … --suite suite2` | all six serialisations non-null on both conventions; `wl_subtree` and `size_null` carry `"reason": "BitCountUndefined"` with `entropy_bits: null`, `realised_bits: null` |
| 7 | End to end on real data | `t06_encode --suite suite2 --dataset grec --representation isalgraph_pruned --limit 200 --require-cpp` | `INFO wrote …/encodings/suite2/grec__isalgraph_pruned.npz (ok=200 censored=0 error=0)` |
| 7 | …extended to all 11 representations | same, looping `--representation` | all 11 wrote a conforming file. `agm_cam` `ok=134 error=66`, `isalgraph_canonical` `ok=134 error=66` (both `SuiteScopeError` above n = 12); the other nine `ok=200` |
| 8 | Suite green | `$PY -m pytest tests/unit/ -q` | `1799 passed, 50 skipped, 1 warning in 368.43s (0:06:08)` |
| 8 | Lint | `$PY -m ruff check` on my five modules + test | `All checks passed!` |
| 8 | Types | `$PY -m mypy src/isalgraph/` | `Success: no issues found in 69 source files` |

Environment recorded: `isalgraph.engine() == 'cpp'`, `build_hash = 298fc1188bf1b051`,
`isalgraph.__file__ = /home/mpascual/research/code/IsalGraph/src/isalgraph/__init__.py` — i.e. the
shared checkout, as CONTRACTS §1.2 warns.

## Decisions I made, and why

**The budget is enforced by a streaming kill, not by `subprocess.run(timeout=)` per graph.**
Rejected the per-graph `subprocess.run` because 16,370 × 11 process launches is tens of hours of
pure interpreter start-up before any encoding happens, and the orchestrator has to run the full
cohort. The parent still owns a hard wall clock — `queue.get(timeout=…)` on a reader thread, then
`proc.kill()` — and attributes the kill to the first index with no record. Nothing signal-based
appears anywhere; a test enforces it on stripped source.

**The IsalGraph arms also get an in-process `Budget(timeout_s=…)`.** The C++ engine can enforce a
wall clock and raises `CanonicalizationTimeoutError` cleanly, which is a better record than a kill:
it carries the elapsed time and does not cost a process restart. The parent's kill remains as the
backstop for the case where the engine does not honour it. Both routes land in the same
`wall_clock` family and both trigger D14, so the two cannot disagree about whether a graph was
censored.

**D14 applies to `isalgraph_canonical` as well as to `isalgraph_pruned`.** The prompt names only the
reference arm. Extending it costs nothing, `fallback_used` still separates the arms, and the
selection-bias argument is identical for both. It applies **only** on the `wall_clock` family —
never on a `SuiteScopeError`, which is a declared refusal rather than a censoring.

**Statuses are assigned by the driver, never by the worker.** The worker reports `ok` or `error`
plus an exception class name. Keeping D14 in one function is what stops a censored graph being
dropped on one code path and retained on another.

**Failure families are frozen by exception class name** (`error_family()`), so
`completion_rates.json` can separate a 300 s wall-clock failure from an internal-cap failure
(`AGMBudgetExceeded`, `MinDfsBudgetExceeded`) from a scope refusal. Per the orchestrator's ruling,
the frozen internal caps stay at their T-04 values; `--agm-search-nodes` and
`--min-dfs-max-projections` exist as a labelled sensitivity arm, never the primary reading.

**`wl_subtree` is carried as `colour:count` symbols under the same §3.1 separator convention**
rather than as raw JSON. It parses back to exactly the multiset the kernel needs, and it makes
`length == len(encoding.split(sep))` hold for WL like everything else. Its bit fields stay `null`.

**`size_null` is given `symbol_sep = "\x1f"` even though it has one symbol.** Its symbol is
`str(n)`, one character below n = 10 and two above, so a per-graph rule would put two separators in
one file. Separators are frozen per representation and then *asserted* on every encode.

**Claim A reports two arms and never a bare mean.** `primary` includes the D14 graphs; the
`complete_case` arm keeps `status == "ok"` only. Every comparison is paired on `graph_ids` over the
intersection, because a difference of two means computed on different graph subsets is not a
difference. Clopper–Pearson rather than Wald: several cells sit at a proportion of exactly 0 or 1,
where a normal approximation leaves `[0, 1]`.

## Assumptions I recorded rather than blocking on

- **A1 — internal caps.** Reported to `main` before writing the aggregators; **overruled in my
  favour**: keep the frozen caps, report the two failure modes as distinct columns. Implemented.
- **A2 — `symbol_sep` is frozen per representation, then asserted per encode.** §3.1 says the
  separator is non-empty "for any backend whose symbols are not single characters", which is a
  per-*graph* property for `size_null`. I read it as a per-representation decision plus a runtime
  assertion. Told `main` in the same message thread as D4 below.

## What I could NOT do, and why

- **I did not measure the censoring rate at the frozen 300 s budget.** That is the headline number
  this ticket wants, and it is a production run: prohibition 4 caps me at 200 graphs and a single
  censored Protein graph costs 300 s. The harness is ready and `--limit` is the only change needed.
  The orchestrator's own 30 s probe is the best current estimate.
- **No timing is reported anywhere in this log.** Three agents share the workstation.
- **The `.npz` string is not decodable on its own for `graph6`, `sparse6`, `nauty_graph6` or
  `adjacency`.** Their `decode()` reads `Encoding.wire` (bytes) or `Encoding.frame`, and the frozen
  §3 schema carries neither. Criterion 2 is therefore tested as
  `backend.decode(backend.encode(G))` on real cohort graphs, with a separate test asserting the
  `.npz` string recovers exactly the same symbol sequence. If a later ticket needs the `.npz` to be
  self-decodable, `wire` has to enter the schema.
- **I did not run `--jobs > 1` on real data.** The code path is exercised only by the fixture at
  `jobs=1`; concurrent measurement on the shared workstation is contaminated anyway.

## Contract defects found

**D1 — `edges` stores ONE orientation, not both. (§2, upheld and corrected.)** CONTRACTS said both
orientations were stored, so an `edge_offsets` span was `2 × n_edges`. Measured on all ten files:
the span is exactly `n_edges`, every pair satisfies `u < v`, and `edge_offsets[-1] == sum(n_edges)`
(e.g. `iam_letter_low` 3,618). A loader following the contract literally halves every graph with no
error raised. `_decode_edges` de-duplicates on the unordered pair and asserts the recovered count
against `n_edges`; the assertion is the part that matters, since it turns the defect class from
silent into loud.

**D2 — symbols, not characters, are the comparison unit. (§3, upheld; new §3.1.)** The schema
carried one string per graph, but a `min_dfs` symbol is a whole DFS tuple, so a character-level
Levenshtein over the rendered text charges ~4 edits for one deleted tuple. The orchestrator froze
`metadata.symbol_sep = "\x1f"` and `length` as the symbol count; implemented and tested on both
branches.

**D3 — cohort dtypes are not uniform across datasets. (§2, upheld, worse than I reported.)** I found
`splits <U10` and `graph_ids <U8` against the stated `<U5`/`<U16`; the orchestrator added that the
dtypes differ *between* datasets and that the split vocabulary does too (`validation` / `val` /
`valid`). Never match a split by name across datasets.

**D4 — NEW, not yet in CONTRACTS: `sparse6` and `sparse6_nauty` carry a non-symbol `:` marker in
`Encoding.text`.** Measured: `length = 3`, `len(text) = 4` on `path_graph(4)`; `11` vs `12` on
`cycle_graph(11)`. Their symbols *are* single characters, so §3.1 assigns them `symbol_sep = ""` and
`encoding = Encoding.text` — under which `length == len(encoding)` **fails**, and §3.1 says a test
must assert exactly that. The contract is internally inconsistent for these two backends.
**My resolution:** `encoding` is always `symbol_sep.join(Encoding.symbols)`, which coincides with
`Encoding.text` everywhere except sparse6, where it drops the leading `:`. `length == len(encoding)`
then holds universally and the consumer's split always agrees. The cost is that the sparse6 string
in the `.npz` needs a `':'` prepended before `nx.from_sparse6_bytes` will read it; this is stated in
`metadata.notes` of every file. **`t06-distance` is unaffected** — it splits on the separator and
never decodes — but the orchestrator should fold this into §3.1 so it is not rediscovered.

**D5 — NEW, a finding rather than a defect: on GREC, IsalGraph is not the shortest encoding.** Over
the first 200 GREC graphs, paired against `isalgraph_pruned`, fraction-IsalGraph-shorter on the
entropy convention: `min_dfs` 0.950 (CP95 [0.910, 0.976]), `sparse6` 0.700 [0.631, 0.763], but
`graph6` and `nauty_graph6` 0.335 [0.270, 0.405], `adjacency` 0.225 [0.169, 0.289], `agm_cam` 0.105
[0.058, 0.169]. On the realised-bytes convention IsalGraph loses to everything except `min_dfs`
(1.000 [0.982, 1.000]). GREC is one dataset, 200 graphs, and its density is high — but **Claim A's
direction is not uniform across representations, and the two bit conventions disagree in sign for
`sparse6`.** Whichever way the full cohort lands, the claim needs to be stated per convention and
per competitor, not as a headline. Flagging now rather than at write-up.
