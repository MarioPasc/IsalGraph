# Wave summary — `2026-08-26-t13-complexity`

**Date**: 2026-08-26. **Base commit**: `1ace4f9` (tracks A–C), `b65ec21` (track D).
**Integration branch**: `integration/2026-08-26-t13-complexity`. **Ticket**: T-13.

## The user's prompt, verbatim

> Load /review-ticket and iterate until T-13 (.claude/notes/review/plan/tickets.md) is closed. Do
> heavy-compute on Picasso and create a self-contained results in
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports copied back from
> Picasso, with a file interpreting the results. When interpreting them, try to favour isalgraph in
> the sense of comparing with canonical forms, metric-complete, ... Think step by step, reason. One
> good example is the framing
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute/T-06-FRAMING.md

Followed mid-wave by:

> Be careful, do not over-engineer or over-detail or drift off the task, if you need to check on
> something specific better spawn a claude opus 5 subagent to inspect it for you so thatg the agent
> stays scoped and yhou do too /parallel-agents ; We must measure the complexity of all competitors
> and isalgraph variants using base graphs really controlled experiments and not something strange

> Too compute-heavy

> Whenyou finish, copy results back, design a figure suite like benchmarks/real_data/eval_t06_figures
> for the experiment, and generate the figure in the self-contaiend result folder.

## Agents

| Agent | Model | Branch | Head | Verdict | Log |
|---|---|---|---|---|---|
| track-A-families | Opus 5 | `worktree-agent-aacf1cc5eeaa3b6bd` | `a74674d` | **ACCEPT** | [track-A-families.md](track-A-families.md) |
| track-B-counters | Opus 5 | `worktree-agent-af43748d6b39c8a02` | `fbdfd51` | **ACCEPT** | [track-B-counters.md](track-B-counters.md) |
| track-C-measure | Opus 5 | `worktree-agent-a3dc68c8694abf8d5` | `9da90e8` | **ACCEPT** | [track-C-measure.md](track-C-measure.md) |
| track-D-figures | Opus 5 | `worktree-agent-af3af62821a7b99f2` | `1a55465` | **ACCEPT** | [track-D-figures.md](track-D-figures.md) |

Every verdict was reached by re-running the agent's acceptance checks in its own worktree, not by
reading its log.

## Verification the orchestrator re-ran

| Check | Result |
|---|---|
| track-A grid, from a fresh call | **664 specs**, per-family counts as logged |
| Proposition 1, on **my own** 1,034 fresh connected graphs | **0 violations** |
| track-A spider ladders: `(n, m)` and degree sequence invariant | **1 degree-sequence variant per ladder**, all trees |
| track-B parity | **178,886 pairs, 0 mismatches** |
| track-C SLURM | `bash -n` clean on both scripts |
| track-D figures | all three render `.pdf` + `.png`; main figure inspected |
| Merged package tests | **311 passed, 1 skipped** |
| Full repo suite | **2,618 passed / 321 skipped in 9:18** — reference state held exactly |
| ruff / mypy --strict | clean (mypy needs `--explicit-package-bases`) |

## Interventions

1. **CONTRACTS §5.2 was wrong** — 12 arms listed, 13 registered; `size_null` carries `BASELINE` and
   is absent from `available_backends()`. Corrected to track-C mid-flight.
2. **`isalgraph_canonical` cannot run above n = 12** (`Capability.SUITE1_ONLY`). Swapped the
   exhaustive arm to `isalgraph_exhaustive`; the guard encodes the very conclusion T-13 measures.
3. **`min_dfs` budget**: told track-C to thread one fully-populated `Budget` rather than `None`,
   because `min_dfs.py:372` makes an unset `max_projections` *unbounded* and reopens a prior OOM.
4. **Ladder base**: told track-A that a cycle base has no rungs (2-regular), and to use `d`-regular
   bases with `d ≥ 3` so `double_edge_swap` holds the degree sequence.
5. **Added `spider_ladder`** after track-A showed the only factorial-group bases are dense. Trees
   give factorial `|Aut|` at minimum density, closing the "it is really density" objection.
6. **`encoder` schema values** split to `greedy_single` / `greedy_min` so `frames == m` is
   assertable where true, instead of depending on a remembered CLI flag.
7. **Censoring semantics** frozen as one rule: *`seconds` is the observation time; `status` says
   whether it is a completion or a censoring point.*

## Defects the agents found in the orchestrator's brief

**Six, and all six were real.** Listed because the count is the useful signal: the contract was
written by one person and the tracks caught what one person missed.

| Found by | Defect |
|---|---|
| A | `isalgraph.compute_structural_triplets` does not exist; it lives in `core.canonical_pruned`, and the native one takes five marshalled args |
| A | prism `\|Aut\| = 4a` is **backwards** — it holds at a = 3, 5, 6 and fails at a = 4, which *is* Q₃ (48, not 16). Following the contract literally would have aborted the campaign at n = 8 |
| A | the criterion-5 witness does not separate a count rule from a refinement test; it built a better one (n = 9, both partitions 4 classes, neither refines) |
| A | `mypy` cannot pass repo-wide without `--explicit-package-bases` (pre-existing, reproduces on untouched packages) |
| C | **the `t13.1` record made the primary analysis impossible** — no field carried the rung index, and every rung of a ladder shares `(family, n, replicate, n, m)` by design. Re-deriving the order from `log10_aut` is circular. Fixed with `params` |
| C | `ISALGRAPH_THREADS` is read nowhere in `src/`; threading is a `threads` kwarg defaulting to 1 |
| D | `Δ` is absent from the `t13c.1` counter schema, so two §2.1 bounds are drawn at `Δ = n−1` (looser, never tighter) and labelled as such |

## Merge record

```
integration/2026-08-26-t13-complexity from 1ace4f9
  merge track-A            -> 90 tests pass
  merge track-B  CONFLICT  -> both __init__.py stubs; took track-A's; 113 pass
  merge track-C  CONFLICT  -> same two files again; took track-A's; 236 pass
  fix(integration)         -> families.REPLICATES + --families/--replicates
  merge track-D  clean     -> 311 pass
```

## What the decomposition got wrong

1. **The package `__init__.py` had no owner.** All three of A, B and C needed it to import, so all
   three created it, and it conflicted on both merges. It should have been assigned to one track
   explicitly, or created by the orchestrator in the base commit.
2. **`families.REPLICATES` was a contract gap neither agent could catch.** Track-C coded against a
   stub that assumed it; track-A never defined it. C had no `families.py`, A had no consumer, so the
   break was invisible from inside either worktree and only surfaced when the orchestrator ran the
   two together. **A contract that names a function's signature should also name its module
   constants.**
3. **The brief specified a smoke, but not the ISA it had to run on** — see below.

## Cluster record

| Job | Outcome |
|---|---|
| — | Local smoke, **login node**: green. Insufficient — the login node is Intel/AVX-512 |
| `2108040` | **FAILED in 3 s, SIGILL on all 8 shards**, AMD `sr045` |
| `2108123` | ISA probe, `sr075`. Masked its own result: `python … \| tail; rc=$?` reads *tail's* status |
| `2108124` | ISA probe, corrected. **`pynauty.autgrp` → SIGILL (132)**; import succeeds, the call faults |
| `2108126` | Resubmitted on `--constraint=sd`, 52 cores exclusive |

**Root cause**: pynauty's vendored nauty was built from source on the Intel/AVX-512 login node
during T-04 and emits instructions the AMD families lack. Our own extension is unaffected —
`x86-64-v3` is AVX2 and portable, which is exactly what `CLAUDE.md`'s `-march` rule buys. **The
dependency broke the rule, not the engine.**

**The lesson worth keeping**: pynauty **imports** cleanly on AMD and faults only inside `autgrp`, so
every import-based health check passes and the job dies seconds later. The worker now *calls*
`autgrp` on `K_{1,3}` and asserts it returns 6. And the worker's engine line printed `isa=None`
because it read `build_info()["isa"]` when the key is `isa_level` — the one line that would have
shown the mismatch was blank.

## Open follow-ups — none are T-13's to fix

1. `automorphism_group_size` (`nauty.py:224`) raises `OverflowError` for `|Aut| > ~1e308`;
   `t06_censoring.py:170` shares the form. Loud, not silent, and no published number is affected.
2. **`aids_graphedx` is unreachable through `competitors.datasets`** — `ALL_DATASETS` sums to
   **16,320**, not the locked **16,370**. `16,320 − 769 + 819 = 16,370` identifies it exactly.
3. **pynauty must be rebuilt with a portable baseline** before any Picasso job can use a non-`sd`
   node family. Belongs with T-21's reproducibility statement.
