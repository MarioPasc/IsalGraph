# Wave summary — 2026-08-29-t28-metrics

**Date** 2026-08-29 · **Base commit** `89b4c9fc8fee6c0c48c0c9256b7aab8f05c1dbb8`
**Integration branch** `feature/t28-correlation-driver` · **Orchestrator** main session

## The user's request, verbatim

> CONTEXT: We had finished the experimental sectional and have the first draft for the article.
> However, mi PI has raised concerns regarding the bad results of the GED correlation: "    Creo
> que hay que hacer algo con la comparativa de la Subsección 5.4 (fidelity, la del GED). Si la
> dejamos como está, corremos el riesgo de que rechacen el artículo en esta ronda. Los resultados
> son bastante malos. Se me ha ocurrido que, además del GED, pruebes con otras dos métricas de
> similitud de grafos: Weisfeiler-Lehman (WL) Kernel (ya lo hemos usado antes), y "Spectral
> Distance (λ-distance), Euclidean distance between the sorted eigenvalue vectors of two graph
> Laplacians" (es un estándar en teoría de grafos). Me quedaría contento con tal de que IsalGraph
> gane a sus competidores en al menos una de las dos métricas, aunque si gana en ambas, tanto
> mejor. Sólo añadiríamos a la subsección 5.4 las métricas en las que IsalGraph gane. Si no gana
> en ninguna de ellas, en último extremo se pueden usar las métricas que aparecen listadas en la
> página 7949 del artículo de IsalChem:
>
> https://doi.org/10.1021/acs.jcim.5c00354
>
>     Habría que sacar un repertorio de subgrafos con los que comparar, porque estas métricas
> están pensadas para buscar sub-moléculas en las moléculas.
> " Read and understand its concerns and gather context from the article's results, which are in
> /home/mpascual/misc/temp_isalgraph_results_folder/reports they cant be found in the sandisk2tb
> since im going to plug it off, since I have to go and leave with it. The article is in
> /home/mpascual/misc/temp_isalgraph_results_folder/review1 for now. Read all of it, as well as
> the code adn understand the current schema under which we operate, as well as the bad results.
> YOu are going to be working on this workstation, but I must go to a different one, therefore,
> commit+push under my name every X time and leave results in Picasso so that I can see them too.
> Keep a worklog as a md file in the repo in docs and create a ticket for this issue. TASK: You
> are going to create a re-submission with the metric distances that my PI has proposed, have the
> results be in Picasso and try to make the execution finish as early as possible (we only have 2
> days) by re-using intermediate results. Levae the results on Picasso and copy them back to the
> computer too to analyze them if needed. This is a really important task, since it will carry the
> load-bearing results of the second experiment (metric correlation) of the paper. You must load
> the skills: /parallel-agents and /picasso-sbatch ; ACCEPTANCE CRITERIA: We have maintained the
> distances we decided in the T-04a (levenshtein for all competitors except for WL, which is
> kernel distance) but computed the correlation with the distances that my PI has proposed
> "Weisfeiler-Lehman (WL) Kernel (ya lo hemos usado antes), y "Spectral Distance (λ-distance),
> Euclidean distance between the sorted eigenvalue vectors of two graph Laplacians" ; You have
> though step by step, and iterated independently. You must ask me anything you doubt right now
> before I leave. The shutdown of this computer is programmed in 24h You can do this!, good luck

Mid-turn addition: *"Also, dont copy the WHOLE sandisk2tb to disk, only the isalgraph folder at
most, cancel the copy"* — the copy was already scoped to `research/ISAL/completed/isalgraph/`
only (2.8 GB, 44,882 files) and had completed; verified by file count and byte count both sides.

## Agents

| Agent | Model | Branch | Head | Verdict | Log |
|---|---|---|---|---|---|
| `t28c-references` | sonnet (`CLAUDE_CODE_SUBAGENT_MODEL`) | `worktree-agent-ab9169ed5f6d5d3b5` | `0d156db` | **ACCEPT** | `t28c-references.md` |
| `t28b-isalchem` | sonnet | `worktree-agent-a47dd4c3e16b042b3` | `0e86fc1` | **ACCEPT** | `t28b-isalchem.md` |

Track A (reference plumbing, the family guard, SLURM, Picasso submission) was the
orchestrator's own and ran on `feature/t28-correlation-driver`.

## Verification, independent of the agents' logs

- **t28c** — re-ran its 35 tests myself in its worktree (35 passed); diffstat matched the
  claimed file list (6 files, 1,515 insertions); tree clean. Then verified the **75 built
  matrices directly**: G3 structural gate 75/75 clean; **G4 the 15 `wl` matrices byte-identical
  to the cached `wl_subtree__kernel` matrices**; G5 off-diagonal exact-zero fraction max 0.155.
  *One discrepancy:* the agent reported a max zero-fraction of 0.028, I measured 0.155 over the
  full off-diagonal. Both are far below the 0.99 gate and my measurement is what the production
  loader enforces, so it does not change any decision — recorded rather than smoothed over.
- **t28b** — no code to test; its argument (graphlet counts scale linearly in `n`, so a
  graphlet-fingerprint Tanimoto obeys `Tani ≤ n_min/n_max`) was checked by reading and is
  sound. Its conclusion, *do not implement*, is accepted.

## Interventions

None mid-flight; neither agent messaged and neither needed correction. The orchestrator fixed
its own two test defects (a wrong `CorrelationGroup` kwarg, and a 10×10 zero matrix whose
off-diagonal zero fraction was 0.978 — below the 0.99 gate, so the *test* was wrong, not the
code) and one campaign blocker (`OUT_ROOT` pointing at `T06_exhaustive`).

## Merge record

Merged into `feature/t28-correlation-driver`, `--no-ff`, t28c first (it defines the
`references` package), then t28b (docs only). **No conflicts** — the ownership partition held.
Fast tests after merge 1: 78 passed. Full unit suite after merge 2: **2,019 passed / 275
skipped** (1,984 before, +35 from t28c). ruff clean on everything touched; `mypy --strict`
clean on 84 source files. One pre-existing ruff error remains in `tests/unit/test_canonical_atlas.py`,
untouched by this wave.

## Result

**The WL kernel reference carries the result; the spectral family does not.** Under it the arm
beats `agm_cam` 15/15, `sparse6_nauty` 15/15 and `nauty_graph6` 12/15, loses to `min_dfs` by a
mean Δρ of −0.024, and clears its own size null on 12/15 cells against 8/25 under GED. The
pre-declared primary spectral variant fails outright (0/15, 0/15) because zero-padding the
normalised Laplacian spectrum reintroduces the size channel — `tr(L_sym) = n`.

Full campaign on Picasso: array `2132238` → merge `2132239`.

## What the decomposition got wrong

**Track C's task was slightly too large for a single agent under a deadline.** It took ~19
minutes of wall clock and 138 k tokens — comfortably the longest leg — because it bundled four
spectral variants, a CLI, a gate, a builder over 15 cells, and 35 tests. Splitting the
*implementation* from the *15-cell build run* would have let the build start earlier and
overlap with the plumbing work on track A.

**The orchestrator should have smoke-tested on Picasso earlier.** The `OUT_ROOT` blocker was
found only after the SLURM scripts were written, the repo staged and `--test-only` run. A
five-minute login-node run of one small shard, done immediately after the plumbing landed,
would have surfaced it before any of that. It cost nothing this time only because the smoke
test happened *before* the live submission rather than after.

## Open follow-ups

1. The campaign is queued, not finished. `bash slurm/t28_metrics/fetch_results.sh` reports
   status and pulls results; the merge asserts `N_actual == 79` and aborts if the guard failed.
2. **§5.4 rewrite is not drafted.** The result changes the subsection's spine: the concession
   *"where the trivial baseline beats the representation, which competitor wins is
   second-order"* is repaired under a structural reference and should not survive unedited.
3. The PI asked for winners-only reporting; the user chose *compute all, decide after seeing
   results*, so **both** manuscript variants are owed once the campaign lands.
4. Picasso's `repos/IsalGraph` is stale (`d6a9f4b`) and carries 14 modified tracked files. It
   was deliberately not touched. The T-28 checkout is `repos/IsalGraph-t28`, and the manifest's
   `src_commit` will therefore describe the older tree — provenance only, no computed number.
