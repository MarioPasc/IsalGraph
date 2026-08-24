# T-06 — handoff to `[T06-subagent-01]`, 2026-08-23

Written by the outgoing orchestrator because its context filled. **You are taking over a ticket
that is ~85 % done.** Everything expensive has been computed. What remains is F2, the closing
documentation, and one figure regeneration.

**Ask the orchestrator (`[T-06-new-agent]`) whenever you are unsure.** Every trap listed in §5 cost
this ticket real hours; none of them announce themselves. A question is cheaper than a re-run.

---

## 1. Where you work — three rules that have each bitten

```bash
cd /home/mpascual/research/code/IsalGraph-T06      # branch ticket/T-06
```

1. 🔴 **Never touch `/home/mpascual/research/code/IsalGraph`.** It is a separate shared checkout on
   `main`. Your shell's default cwd *is* that directory and resets there after every command, so use
   `git -C <worktree>` or an explicit `cd` on **every** call. Read-only inspection is fine; never
   write, never `git switch`, never commit there.
2. 🔴 **Re-check the branch immediately before every commit.** A peer session moved the branch under
   us once and a commit landed in the wrong place:
   ```bash
   B=$(git -C $W rev-parse --abbrev-ref HEAD); [ "$B" = "ticket/T-06" ] || exit 1
   ```
3. 🔴 **`git log` and `ps` are proxied through `rtk` and return stale or empty results.** Trust only
   `git rev-parse`, `git show-ref`, `git show -s`, and `pgrep -a`.

### The `pgrep` trap, which fired twice in one session in both directions

`pgrep -a -f f1_verify` **matches the shell wrapper running the pgrep**, so it reports a hit when
nothing is running. Two sessions read that as "still executing". Use the bracket form and confirm
against the artifact:

```bash
pgrep -a -f 'f1_ver[i]fy'      # bracket cannot match the literal pattern text
```

**Confirm a run against its output file, never against a process list.** A run that is alive but has
written nothing for hours is not usefully different from a dead one.

---

## 2. Environment

```bash
PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python
```

- `isalgraph.engine()` must be `'cpp'`; `build_info()['build_hash']` must be `298fc1188bf1b051`.
- 🔴 **`import isalgraph` resolves to the SHARED checkout's `src/`**, not your worktree. The
  scikit-build-core editable finder is path-pinned and outranks `PYTHONPATH`.
  - **Never `export PYTHONPATH=<worktree>/src`** — it silently drops you to pure Python.
  - **Never edit `src/isalgraph/`.**
  - **Never assert a numeric value a competitor backend produced.** Assert API shape and invariants.
- **Verified inert, and re-verify before every production run** (§1.4b.1 of the design note):
  ```bash
  git -C /home/mpascual/research/code/IsalGraph-T06 diff --stat c1d36b1..HEAD -- src/isalgraph/
  # empty => the shared checkout's src/ is equivalent to ours
  ```
- `benchmarks/<name>` are **symlinks** into `real_data/`. **`eval_stats` is the one exception** — it
  has no symlink and imports by the full `benchmarks.real_data.eval_stats.*` path. `eval_distance`
  and `eval_encoding` go through theirs. Getting this wrong is an immediate `ModuleNotFoundError`.

| Command | Purpose |
|---|---|
| `$PY -m pytest tests/ -q` | full suite (~2.5 min) |
| `$PY -m ruff check --fix src/ tests/ benchmarks/` | lint — **28 pre-existing errors is the correct baseline, do not "fix" them** |
| `$PY -m mypy src/isalgraph/` | type check |

---

## 3. Data root

```
T06=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06
DATA=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data
```

| Path | Contents |
|---|---|
| `$T06/encodings/{suite1,suite2}/` | 155 cells, `manifest.json` |
| `$T06/distances/{suite1,suite2}/` | 190 matrices (120 suite2 + 70 suite1) |
| `$T06/families/family_F0.json`, `family_F1.json` | the two gates, **run** |
| `$T06/gates/gate_T06_reproduction.json`, `gate_T06_structural.json` | both **PASS** |
| `$T06/completion_rates.json` | the `c` input |
| `$T06/ladder.json`, `ladder_suite1.json` | A9 |
| `$T06/size_profile.json` (1.1 MB) | 2,355 strata |
| `$T06/figures/size_profile/` | fig1–3, pdf+png |
| `$DATA/eval/ged_matrices/` | Suite-1 **exact** GED |
| `$DATA/source/APPROX_GED/LB/` | Suite-2 `lb_matrix` **and** `ub_matrix` (both live in the LB file) |
| `$DATA/exported/`, `$DATA/exported_suite2/` | cohort CSR, for WL |

---

## 4. What is DONE — do not redo any of it

Branch `ticket/T-06`, head `300ee35`, **54 commits ahead of `main` (`c1d36b1`)**, tree clean.

| Item | Result |
|---|---|
| **F-1** | Closed on `SUITE1_ONLY` scope policy. Kill counts NOT MEASURED (4 diagnosed attempts). `43 s/graph`, `≈520×`, `≥6.8 core-hours` **retracted as unprovenanced** — do not requote |
| **Cohort gate** | 16,370 / 21,710,892 / 5,350 / 3,897,911 — **all four exact** |
| **D14 censoring** | 101 graphs, all Mutagenicity (2.50 %), zero elsewhere |
| **Competitor encodings** | 135 cells, 20 min, 0 failures, 0 contract violations |
| **Reproduction gate (crit. 5)** | **PASS**, 113 values, max \|Δ\| = 0.0000000000 |
| **Distance matrices** | 190 cells, 2 min 47 s, 0 failures |
| **A4 structural gate** | **PASS**, 190 matrices, 0 violations, 190/190 `graph_ids` joins exact |
| **A9 ladder** | 10 Suite-2 + 5 Suite-1 rows |
| **`c`** | **7** non-computable triples (6 × `agm_cam`, 1 × `min_dfs`) |
| **F0** | fires **4 of 5** → majority branch FIRES |
| **F1** | `d = 7 of 10` (reported, but **not applied** under the F0 branch) |
| **`N_actual`** | **79**, enumeration == closed form, discrepancy 0 |
| **Size profile** | 2,355 strata, 3 figures rendered |

### Results that must be reported plainly and never softened

- **The within-`n` correlation collapses** (§17): `isalgraph_pruned` ρ = 1.0000 at `n = 3` →
  **0.2608 at `n = 12`** → 0.135 mean over `n` 13–30. Above `n ≈ 40` no representation is
  distinguishable from ρ = 0. **The pooled ρ ≈ 0.93 is mostly the size channel.**
- **IsalGraph is the best representation on none of 15 records**, in either view (§16). 8 of those
  deficits are resolvable at the graph-level bootstrap; 7 are ties at `n = 200`.
- **The competitors that beat it are NOT excludable** (§15.3): `wl_subtree`, `sparse6_nauty` and
  `nauty_graph6` complete on 100 % of both cohorts and pass every metric axiom.
- **Zero encoding collisions on ~24.8 M pairs** (§16 / ladder) — 3,424,764 Suite-1 pairs against
  *exact* GED, plus 21,339,658 Suite-2 at LB > 0. **This is the ticket's clearest positive result.**
- **The large-`n` extension is now descriptive**, per the PI decision in §18.7.

---

## 5. Traps that have already cost this ticket real time

1. **A per-graph subprocess measures start-up, not the encoder.** One graph per process inflated a
   4.95 ms encode to 579 ms (×117). Amortise warm-up across a band.
2. **A completion rate may be measuring a scope guard.** `agm_cam` and `isalgraph_canonical` are
   refused above `n = 12` *before any encode*. **Always split `error_kind` into scope vs budget;
   never sum a `SuiteScopeError` with a budget failure** — completion rates set `c`.
3. **`readline()` blocks**, so a deadline checked only after a line arrives never fires. Poll a queue
   fed by a reader thread.
4. **`/usr/bin/time` does not exist here**, and `set -uo pipefail` without `-e` lets a script exit 0
   having done nothing. Every campaign script needs a failure counter, a file-count assertion, and a
   `DONE_MARKER` on **every** exit path.
5. **Long local runs must be detached** (`setsid nohup … &`) — a foreground tool call is killed at
   10 minutes. Arm a `Monitor` with `persistent: true` that fires on failure, completion **and
   process death without a `DONE_MARKER`**. Monitors die with the session; a previous handoff lost a
   watcher exactly this way.
6. **`f5_200.json` has two size-null keys.** Row-level `size_null` is the wrong subtrahend;
   per-cell `size_null_on_my_pairs` is right. They agree to 1e-9 on 13 of 15 records — silent
   everywhere except Mutagenicity, the one dataset where IsalGraph wins Claim A.
7. **The `equal_n` size null is UNDEFINED**, not missing: `|n_i − n_j| ≡ 0` there, so Spearman has no
   denominator. Raw ρ *is* the structural signal in that view.
8. **`aids` is 769 in Suite 1 against `aids_graphedx`'s 819 in Suite 2** (F-12). **Join on
   `graph_ids`, never positionally.** Suite-1 `aids` is a verified strict subset.
9. **GED is legitimately 0 for isomorphic graphs.** Never assert `value > 0` per pair. The
   silent-zero guard belongs at the matrix level (off-diagonal exact-zero fraction ≥ 0.99).
10. **A formatter (PostToolUse hook) reflows files after you Write them.** A later
    `str.replace`-style patch will silently no-op because the target text moved. **Verify every patch
    landed** (`grep -c` for the new text) rather than trusting a script that printed "ok".

### The recurring failure mode — now six instances, always the same direction

Reductions that **shrink `N_actual`** and therefore weaken BH on every surviving test. Five were in
prose (`tickets.md`, the design note's own A7, the retired `s` rule, a cancellation argument, and one
more), one latent in `t06_completion` (§15.4). **The code was never wrong** —
`family.py:_closed_form` has always carried `182 − 15k − 8d + k·d − c`.

> **Standing rule: `preregistration.md` §5 is the sole authority for the closed form, enumeration
> outranks any closed form, and any change that shrinks `N_actual` is a defect until re-derived.**
> Cite `family.py:_closed_form` rather than restating the formula.

---

## 6. What REMAINS — your work, in order

### 6.1 F2 — the primary family, over `N_actual = 79`

**This is the main remaining task.** No driver exists yet; `family.py` has `run_f2` but no CLI.

`run_f2(p_values, inputs, q=0.05, omnibus_scores=...)` needs a p-value per admissible cell. Build
`ReductionInputs` exactly like this — **the branch is frozen by the PI (§18.7)**:

```python
ReductionInputs(
    excluded_representations=frozenset({"adjacency", "graph6", "sparse6"}),   # k = 3
    uninformative_datasets=frozenset(),          # d NOT applied under the F0 branch
    noncomputable=<the 7 triples from completion_rates.json>,   # c = 7
    f0_demotes_approximate=True,                 # F0 fired 4/5
)
```

Cell rows and where their p-values come from:

| row | what | machinery |
|---|---|---|
| `A1` | Claim A, bits per (dataset, representation) | Wilcoxon signed-rank on `entropy_bits` / `realised_bits` from the encodings. **Both bit conventions for every method**; `wl_subtree` and `size_null` are `BitCountUndefined` with the reason printed, **never fabricated** |
| `B1e` | Suite-1 ρ difference, IsalGraph vs comparator | `association.bootstrap_associations` with a `DifferenceSpec` |
| `B1a` | Suite-2 ditto | same — **descriptive under this branch** |
| `B3e` / `B3a` | MRM standardised β₁ | `association.mrm` |
| `A2` / `B2` | omnibus | `multiplicity.friedman_omnibus` + `wilcoxon_holm_posthoc` |

Emit `family_F2.json` carrying `k`, `d`, `c`, `N_actual`, `N_max`, **and both threshold columns** —
the BH-over-`N_max` = 182 sensitivity column is a re-threshold of stored p-values and costs nothing.

Model the driver on `benchmarks/real_data/eval_stats/t06_gates.py`, which already solves matrix
loading, the `graph_ids` join, the ID-subset for Suite-1 `aids`, and the shared-resample bootstrap.

**Acceptance:** every printed ρ carries its graph-level bootstrap CI **and** its per-representation
size null; the bootstrap resamples **graphs, never pairs** (a test must assert
`correlation_metrics.bootstrap_correlation` is not in any T-06 module's import closure).

### 6.2 The paired IsalGraph-vs-competitor bootstrap (§16.4)

Currently only IsalGraph-vs-**null** is paired (`paired_null_ci.json`). The
IsalGraph-vs-**competitor** comparison rests on overlapping marginal CIs, which is the weaker test.
Add a paired bootstrap of `ρ(IsalGraph) − ρ(competitor)` on identical pair sets and identical
resamples. Until then the honest wording is *"best on none of 15, with 8 resolvable deficits and
7 ties"* — **not** "beaten on 15 of 15".

### 6.3 Regenerate the figures when F2 lands

```bash
$PY -m benchmarks.real_data.eval_size_profile.figures \
    --profile $T06/size_profile.json --out-dir $T06/figures/size_profile
```

Reusable runner: `experiments/paper_pipeline/run_size_profile.sh`.

### 6.4 Final hygiene, then `review-close`

- `$PY -m pytest tests/ -q` green; ruff at the **28** baseline; mypy clean; `scratchpad/` empty;
  `git status --porcelain` empty.
- Then invoke the **`review-close`** skill. It owns the board strike in `tickets.md`, plan-file
  propagation, the article notes and the response-letter fragment.
- **Ask the PI before merging to `main`.**

---

## 7. Corrections to propagate at close (`review-close` §3)

You may **not** edit `.claude/notes/review/plan/` yourself — report to the orchestrator. These are
already diagnosed and waiting:

1. **`tickets.md:158`** carries the stale `N_actual = 182 − 15k − 8d`, missing both `+k·d` and `−c`.
   `preregistration.md` §5 is correct and is the authority.
2. **`data.md` §7's two-root path defect is already discharged** by T-05, not by T-06 — strike it
   rather than re-implement.
3. **`t06_completion` counts a censored graph as not completed** (§15.4), contradicting D14. Inert
   today; would over-charge `c` for any comparator that censors.
4. **E10's WL numbers predate the `h = 2` fix** (`b7ce447`); the board anticipates §4.1's Letter LOW
   row moving 0.895 → 0.7792.
5. **F0's `GED_approx` was undefined** in `preregistration` §2. Resolved by the PI as the
   conservative reading (§18.7); the pre-registration should record it.

---

## 8. Escalate to the PI, do not decide alone

Any cohort count not reproducing (16,370 / 21,710,892 / 5,350 / 3,897,911); D4's β₁ collapsing;
compute above ~5,000 core-hours; reopening F-1; a second failed iteration on the same task; anything
that would shrink `N_actual` below 79.

**Today is 2026-08-23. The revision is due 2026-08-31 — 8 days.** A previous briefing asserted
2026-08-17 and asked for `2026-08-23` stamps to be "corrected"; that was wrong and the stamps are
correct provenance. **Do not backdate anything.**

---

## 9. Required reading, in order

1. `.claude/notes/review/tasks/T-06-design.md` — **§18.7 first** (the frozen branch), then §5
   (frozen rules), §6 (acceptance criteria), §7 (stop-and-ask), §15–§18 (today's results).
2. `.claude/notes/2026-08-16-t06-recompute/CONTRACTS.md` — §1 prohibitions, §3.1 (symbols not
   characters), §3.2 (a censored row is not an empty row), §4.1 (size null is per
   `(representation, dataset)`), §5 (metadata), §7 (definition of done).
3. `.claude/notes/review/plan/preregistration.md` §5–§5.3 — the reduction rule and its precedence.
