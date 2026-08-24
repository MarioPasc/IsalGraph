# Wave summary — `2026-08-16-t06-recompute`

**Date**: 2026-08-16 · **Ticket**: T-06 (full recompute) · **Wave**: 1 of 2

**User's original prompt, verbatim:**

> A different agent, called T-04a is implementing the ticket T-04a from
> .claude/notes/review/plan/tickets.md ; You are tasked with implementing of Ticket T06. However,
> since your ticket depends on T-04a, you should wait until T-04a is finished and then start with
> your task. To do this, you can message T-04a directly and tell it to ping you when its done (tell
> it to declare a task to do this in order to not forget). You can set up a wakeup ping to wake up in
> 4 hours just in case and check on T-04a. To complete T-06, load the /review-ticket skill, as well
> as /parallel-agents and /picasso-sbatch Iterate for as long as you need, be careful to not
> overwrite any previous work, commit under my name frequently. Use branches for your agents, at the
> end, merge into main and push under my name. Think step by step, reason. Iterate for as long as you
> need.

**Base commit**: `8afa59e` (`ticket/T-06`, cut from `main` at `f7ad283`)
**Integration**: `integration/2026-08-16-t06-recompute` → merged to `ticket/T-06`
**Orchestrator worktree**: `/home/mpascual/research/code/IsalGraph-T06` (the shared checkout is T-04a's)

---

## Agents

| Name | Model | Branch | Worktree | Head | Verdict | Log |
|---|---|---|---|---|---|---|
| `t06-encoding` | opus-5 | `t06/encoding` | `.claude/worktrees/encoding` | `49dbc85` | **ACCEPT** | [encoding.md](encoding.md) |
| `t06-distance` | opus-5 | `t06/distance` | `.claude/worktrees/distance` | `532d008` | **ACCEPT** | [distance.md](distance.md) |
| `t06-stats` | opus-5 | `t06/stats` | `.claude/worktrees/stats` | `3b5e4c0` | **FIXUP** | [stats.md](stats.md) |

`FIXUP` on stats is a 2-line out-of-ownership isort sweep, reverted in `fix(integration)`. Nothing
else was repaired.

## Verification the orchestrator re-ran personally

| Check | Result |
|---|---|
| Full suite | **2,290 passed / 321 skipped** (T-04 reference: 2,106 / 321 — **+184, no regression**) |
| Track tests | 175 = 75 encoding + 35 distance + 65 stats |
| `mypy --strict src/isalgraph/` | **Success, 69 source files** |
| `ruff check benchmarks/ tests/ src/` | **28 errors — exactly the pre-existing baseline**, measured on `ticket/T-06` with no agent work |
| Cohort verify | Suite 2 = **16,370** over 10 keys; Suite 1 = **5,350** (1,180/1,253/2,059/89/769) |
| Merge conflicts | **zero** — the ownership partition held |
| Worktree status | all three `git status --porcelain` empty |

## Interventions

1. **Branch collision, before the wave.** T-04a `git switch`ed the shared checkout mid-session and
   this session's design commit landed on their branch. Diagnosed from the reflog; moved T-06 into a
   dedicated worktree, cherry-picked the commit, and told T-04a the shared checkout is theirs.
2. **Three mid-flight contract corrections** relayed to all agents (§1 below).
3. **`fix(integration)`** reverting the isort sweep.
4. **Two frozen-file amendments** (pre-registration `s`→`c`, and the `+k·d` / FCR / §5.3 fixes), both
   owned by the orchestrator; no agent edited a plan file.

---

## 1. What the wave found — nine defects, none of them in the agents' own work

**Every one was in something already frozen, and seven were mine.**

| # | Defect | Found by | Verified how |
|---|---|---|---|
| 1 | CONTRACTS said the cohort CSR stores **both** edge orientations; it stores **one** (`u < v`), `edge_offsets[-1] == sum(n_edges)`. **A conforming loader would have read half the edges of every graph, silently** | encoding | orchestrator re-measured on 3 datasets, ratio 1.000 |
| 2 | **Symbols, not characters, are the comparison unit.** `min_dfs` charges ~4 edits per deleted tuple under char-Levenshtein | encoding | distance quantified it: **15,706 / 19,900 (78.9 %)** pairs differ, mean ratio **3.86×** |
| 3 | Cohort dtypes **not uniform across datasets**; split vocabulary differs (`validation`/`val`/`valid`) | encoding | orchestrator measured |
| 4 | **`sparse6`'s `':'` is not a symbol**, so `length == len(encoding)` failed | **encoding AND distance, independently** | orchestrator measured; matches T-04's already-frozen "the `':'` is framing, not payload" |
| 5 | §3 self-contradicted D14 on whether a censored row is empty | distance | reading the contract against itself |
| 6 | Pre-registration's `s` term charged **−20 per representation** where AGM completes at 100 % on 4 of 10 datasets — **anti-conservative** | **T-04a** | orchestrator verified against `competitors/README` finding 5 |
| 7 | The "cancellation argument" for keeping B1a on all pairs was **false** — correlation is not additive, and `ρ(X,f(\|Δn\|)) = ρ(X,\|Δn\|)` exactly for monotone `f` | **T-04a** | Kendall τ between views **−0.111 to 0.467**, where an additive offset gives 1.00 |
| 8 | The closed form **double-charged `k·d` cells** — anti-conservative | stats | orchestrator re-derived; proved `k·d` is the *complete* overlap |
| 9 | **"BH-adjusted CI" named no computable object**; **F0's majority branch had no coefficient** | stats | resolved to FCR intervals (B&Y 2005) and an explicit 81-cell §5.3 |

**Three of these (6, 8, and the `s`-term's original form) are the same failure mode**: reducing
`N_actual` below what the data forces, which lowers the BH burden on every surviving test. That
pattern is worth naming for the next ticket.

## 2. Scientific results produced

- **Both canonical forms are complete invariants** — 995/995 distinct strings, 0 collisions, 0
  invariance failures over every connected graph on 2–7 nodes (per-`n` = OEIS A001349), 20
  relabellings each. But they agree on only **13.8 %** and pruned is longer on **56 %**.
- **`isalgraph_pruned` frozen as the reference arm (F-1)**, on measurement under the C++ engine:
  `pruned` **150/150** complete at a 30 s budget across all ten datasets; `canonical` killed on
  **20/45** of COIL-DEL / Mutagenicity / Protein. Corrects `competitors/README` finding 7b, which is
  a statement about a **2 s** budget, not about the encoder.
- **Claim A's direction is not uniform and the two bit conventions disagree in sign.** On 200 GREC
  graphs `sparse6` flips **0.700 → 0.010** between entropy and realised bits. A bare "IsalGraph
  encodes in fewer bits" is not supportable; it must be per convention and per competitor.
- **Graph-level bootstrap widens the CI 4.02×** on real LINUX (0.11580 vs 0.02879) — R3.5c's
  magnitude, measured.
- **The distance stage is not the bottleneck**: `rapidfuzz.process.cdist` at 4.1 M cells/s worst
  case ⇒ Suite 2's 43.4 M ordered cells ≈ **11 s per (representation, metric)**. Shard 4–8 ways for
  resumability, not for speed.
- **The raw IAM GXL tree is present** (33,187 `.gxl`), refuting T-04 §7.

## 3. Open, carried to wave 2

| Item | Owner | Note |
|---|---|---|
| **`padded_hamming` needs `Encoding.frame`; WL `kernel` needs fitted features.** Neither is in §3 and neither is derivable from a joined string. **Both refused rather than approximated** | orchestrator | **WL is one of Claim B's 7 comparators, so this is not optional.** `wl_subtree` is a `VectorBackend` and never had an encoding — it needs its **own driver** reading the cohort. `padded_hamming` is only needed if T-04a selects it as a primary distance; decide when `k` lands |
| The 300 s D14 censoring rate | orchestrator | production run, `--limit` is the only change |
| `c` determination | orchestrator | needs the production completion rates |
| `k`, primary distance per representation | T-04a | pending |
| Production campaigns (encode → distance → family) | orchestrator | after T-04a merges |
| `scratchpad/verify_canonical.json` → `tests/` | orchestrator | `data.md` §6 forbids a decision resting on a scratchpad artifact |
| **`eval_encoding.py:252,401,422` uses `signal.alarm` around a canonical probe** | unowned | the exact T-05 hang, pre-existing in the repo. Not T-06's to fix, but it is a landmine |

## 4. What the decomposition got wrong

**Two ownership defects, both mine, both cheap:**

1. **The encodings schema initially lacked `node_counts`/`edge_counts`**, which would have forced the
   distance track to open a cohort file and put a loader in two tracks' ownership. Caught in
   preflight and fixed before spawning by carrying both through the `.npz`.
2. **`ruff check --fix benchmarks/` reaches outside a track's ownership.** The prompts scoped file
   *ownership* but gave a lint command with repo-wide scope. Next wave: scope the lint command to the
   track's own paths.

**What the decomposition got right**: three tracks, disjoint sets, **zero merge conflicts**, and each
track shipped despite its upstream not existing — because the contract was frozen first and both
downstream tracks synthesised conforming fixtures rather than blocking. The stats track's end-to-end
ran on a synthesised distance matrix and **said so prominently**: *"no number in my log is a
scientific result."* That is the correct behaviour and it is why nothing has to be re-run.
