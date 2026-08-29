# T-28 — independent verification and interpretation

**Written** 2026-08-29, after merging `feature/t28-correlation-driver` into `main`.
**Scope.** Everything here is a re-derivation from the raw probe records
(`docs/worklogs/T-28-artifacts/probe_point_estimates_built_refs.json`, 1,260 records),
not a restatement of `T-28-metric-distances.md`. Where the two agree I say so; where the
framing differs I say what the raw records support instead.

**Verdict in one line.** The WL kernel result is real, it is the strongest thing in the
revision's fidelity section, and it should go in — but three of the numbers as currently
phrased overstate it, and the like-for-like comparison is both *cleaner* and *more
favourable* than the one the worklog quotes.

---

## 1. The reproduction is exact

Recomputing the head-to-head grid from the raw records reproduces every published cell of
the worklog's table with no discrepancy. Under the `wl` reference, `all_pairs`, arm =
`isalgraph_pruned`, over the 15 (suite, dataset) cells:

| competitor | worklog | re-derived |
|---|---|---|
| `agm_cam` | 15/15 (+0.177) | 15/15 (+0.177) |
| `min_dfs` | 3/15 (−0.024) | 3/15 (−0.024) |
| `nauty_graph6` | 12/15 (+0.149) | 12/15 (+0.149) |
| `sparse6_nauty` | 15/15 (+0.206) | 15/15 (+0.206) |
| clears size null | 12/15 | 12/15 |

Two facts the worklog did not report, both favourable:

- Under `wl`, the pruned arm also beats **`isalgraph_canonical` on 4 of the 5** cells where
  both are defined (+0.014). The PI's choice of IsalGraph Pruned as the reported arm is
  supported by the data, not just by convention.
- `spectral_esd` clears the size null on **0 of 15** cells. The worklog said it "does not
  rescue the spectral family"; the raw records say it never clears at all.

---

### 1.1 The reference matrices verified here, not taken from the log

Re-run against the Sandisk copy of all 75 matrices:

| gate | claim | re-verified |
|---|---|---|
| **G4** | the `wl` reference **is** the cached `wl_subtree__kernel` matrix | **15/15 byte-identical**, `graph_ids` included |
| **G3** | symmetric, zero diagonal, finite, non-negative | **75/75 clean** |
| **G5** | off-diagonal exact-zero fraction below the 0.99 silent-zero threshold | max **0.155** (`wl`), min 0.099 (`spectral_esd`) |

G4 has a consequence worth stating in the paper rather than leaving for a reviewer to find:
**the WL reference is literally one competitor's own representation distance.** That is what
makes `ρ ≡ 1.0` for the `wl_subtree` arm the identity rather than a near-miss, and it is why
excluding that arm from the win counts is exact and not a judgement call. It also means the
reference must be described as *an alternative structural similarity measure*, never as
"ground truth". The remaining four competitors — `agm_cam`, `min_dfs`, `nauty_graph6`,
`sparse6_nauty` — are canonical forms and serialisations with no WL-like construction, so
none of them gains from the swap, and neither does IsalGraph.

---

## 2. Three integrity findings

### 2.1 🔴 Four of the fifteen cells are byte-identical duplicates

Suite 1 and Suite 2 share four datasets — `linux`, `iam_letter_low`, `iam_letter_med`,
`iam_letter_high`. For those four, **240 records are identical in ρ, `n_pairs` *and*
`n_graphs` across the two suites**; zero differ.

This is correct behaviour, not a bug. The suite distinction exists to separate *which GED
reference is available* (Suite 1 exact, Suite 2 a bracket). A **structural** reference —
`wl`, the spectral family — is the same matrix in both suites, so the suite axis carries no
information for it and the same measurement is counted twice.

Consequence: under a T-28 reference the cohort is **11 distinct datasets, not 15 cells**.
Deduplicated (Suite 1 preferred where both exist), `all_pairs`, `wl`:

| competitor | as 15 cells | as 11 datasets |
|---|---|---|
| `agm_cam` | 15/15 (+0.177) | **11/11** (+0.187) |
| `min_dfs` | 3/15 (−0.024) | 2/11 (−0.027) |
| `nauty_graph6` | 12/15 (+0.149) | **8/11** (+0.105) |
| `sparse6_nauty` | 15/15 (+0.206) | **11/11** (+0.170) |
| clears size null | 12/15 (80 %) | **8/11** (73 %) |

The direction is unchanged and the two clean sweeps survive. But `12/15` written in a paper
claims twelve independent datasets and there are eight. **Report the deduplicated
denominator.** The same correction applies to the paired-bootstrap tally that the Picasso
campaign will produce — `verdicts_t28.py` counts cells, so it will inherit the duplication.

### 2.2 🔴 The `8 of 25` → `12 of 15` comparison is not like-for-like

The manuscript's existing figure is *"Over the twenty-five records the canonical string
clears its own size null on eight"* (`05_results.tex:351`). Those 25 records are
5 (Suite 1 × exact) + 10 (Suite 2 × lower bound) + 10 (Suite 2 × upper bound).

**The 10 lower-bound records are hopeless by construction.** Measured over the 11 datasets,
the lower bound's own size null has median ρ = **0.972** and exceeds 0.90 on 9 of 10. Nothing
can clear a null that high. Counting them in the GED denominator and not in the WL one
inflates the improvement.

How size-dominated each reference actually is — `ρ(|n_i − n_j|, d_ref)` over 11 datasets:

| reference | min | median | max | > 0.90 | arm clears null |
|---|---:|---:|---:|---:|---:|
| GED exact | 0.713 | 0.914 | 0.920 | 3/5 | 1/5 |
| GED lower bound | 0.879 | **0.972** | 0.997 | 9/10 | **0/10** |
| GED upper bound | 0.340 | 0.702 | 0.754 | 0/10 | 7/10 |
| **WL kernel** | **0.159** | **0.570** | 0.973 | 1/11 | **8/11** |
| spectral (norm L) | 0.818 | 0.923 | 0.986 | 8/11 | 0/11 |
| spectral (comb L) | 0.560 | 0.895 | 0.988 | 5/11 | 2/11 |
| spectral (adjacency) | 0.924 | 0.971 | 0.996 | 11/11 | 0/11 |
| spectral ESD | **−0.061** | **0.303** | 0.551 | 0/11 | 0/11 |

Read against the *best available* GED reference per dataset (exact on Suite 1, upper bound
on Suite 2), the honest comparison is **7/11 under GED against 8/11 under WL** — a gain of
one dataset, not "8 of 25 against 12 of 15".

### 2.3 ✅ …and the like-for-like comparison that *is* clean is much stronger

Restrict to the five Suite-1 datasets, the only ones carrying **exact** GED, where no
bracket argument is available and which the manuscript itself calls the headline risk. Same
five datasets, same representation distances, only the reference changed:

| dataset | excess vs exact GED | excess vs WL kernel |
|---|---:|---:|
| `aids` | −0.5445 | **+0.1027** |
| `iam_letter_high` | −0.2536 | **+0.1676** |
| `iam_letter_low` | +0.0139 | **+0.1432** |
| `iam_letter_med` | −0.0313 | **+0.1950** |
| `linux` | −0.2392 | **+0.3270** |
| **clears its own size null** | **1 / 5** | **5 / 5** |

**This is the result.** Against exact graph edit distance the canonical string clears its own
size null on one of five datasets; against the WL kernel it clears on five of five, and the
sign of the excess flips on four of them. On LINUX the flip is already significant with
paired bootstrap intervals: **−0.2247 [−0.3492, −0.0922] → +0.3189 [+0.1699, +0.4454]**.

On Suite 2 the picture is a wash — the upper bound gives 7/10 and WL gives 7/10, and WL is
*worse* on `coil_del`, `mutagenicity` and `protein`. Say so.

### 2.4 ⚠ The `all_pairs` / `equal_n` split is load-bearing and the manuscript mixes them

The manuscript's headline size-null table is `all_pairs` (source comment,
`05_results.tex:358`); its competitor head-to-head and `fig:rho-vs-size` are `equal_n`.
Neither is declared primary. The two views do not agree under WL:

| competitor | WL / `all_pairs` | WL / `equal_n` |
|---|---|---|
| `agm_cam` | 11/11 (+0.187) | 4/11 (+0.016) |
| `min_dfs` | 2/11 (−0.027) | 2/11 (−0.049) |
| `nauty_graph6` | 8/11 (+0.105) | 8/11 (+0.084) |
| `sparse6_nauty` | 11/11 (+0.170) | 8/11 (+0.070) |

The sweep over `agm_cam` is an `all_pairs` effect; once size is removed by construction it
becomes a tie. **A reviewer who recomputes within-`n` will find this.** State which view each
number comes from, and do not quote the `all_pairs` sweep beside a `fig:rho-vs-size` that is
`equal_n`.

Note that under `equal_n` the improvement over the *submitted* claim is still real: the
manuscript currently concedes `agm_cam` as a loss (`05_results.tex:399`), and under WL it is
a tie.

---

## 3. Does the protocol answer what the PI asked?

| PI's request | status |
|---|---|
| WL kernel as a second similarity metric | **done** — IsalGraph wins |
| Spectral λ-distance (sorted Laplacian eigenvalues, Euclidean) | **done** — IsalGraph loses, all four variants |
| "content if IsalGraph beats its competitors on at least one of the two" | **satisfied by WL** |
| IsalGraph Pruned as the reported arm | **done**, and supported (§1) |
| min-DFS as the key competitor | **tie, not a win** — see §4 |
| Supplementary material | endorsed by the PI; needed (§4) |
| IsalChem p. 7949 metrics as a last resort | analysed, reasoned *do not implement*; **moot**, WL carried it |
| Subgraph repertoire for the IsalChem metrics | **moot** for the same reason |
| Caption "lower/higher is better"; shade the min-DFS gap in the coding-overhead inset | **not T-28** — separate figure-polish task, still open |

The design is sound where it matters most: the **representation** distances never move. Every
`d_R` is read from T-06's cache with its `code_commit` and `build_hash`, so this is a swap of
the yardstick and not a new experiment — which is exactly the acceptance criterion, and it is
enforced structurally rather than by discipline. The pre-registered confirmatory family is
protected: membership now keys on the `(representation, reference)` pair, so an added
reference cannot enter it, and the merge **asserts** `N_actual == 79` rather than reporting it.

---

## 4. What I recommend against, and why

**Do not report only the metrics where IsalGraph wins.** The PI's note says
*"Sólo añadiríamos a la subsección 5.4 las métricas en las que IsalGraph gane."* Five
alternative references were evaluated and one is being reported. If a reviewer asks whether
other metrics were tried — and in a revision whose reviewers are already auditing the
evaluation, one will — "we tried five and reported one" is a far worse position than
reporting five up front.

The cost of doing it properly is close to zero, because the PI has already agreed to a
supplement:

- **§5.4 main text:** GED (retained) + WL kernel, with the 1/5 → 5/5 result and the
  mechanism. One sentence recording that a spectral λ-distance family was also evaluated and
  fails, pointing at the supplement.
- **Supplement:** the full 8-reference grid, win or lose, plus the padding mechanism.

This keeps the win in the main text and removes the metric-shopping objection. The design
note already froze `spectral` as the pre-declared primary *before* results and committed to
reporting all four variants — that pre-registration is worth a great deal in the response
letter, but only if the reporting actually happens.

**Do not claim IsalGraph beats min-DFS.** It does not. Point estimates put it 0.027 below on
9 of 11 datasets; the two cells with bootstrap intervals say TIE (p = 0.70, p = 0.37). "Tied
with the strongest competitor, ahead of the other three" is both true and enough — and it is
the reviewer's own nominated competitor, which makes a tie a defensible outcome.

**The strongest honest framing is not "we win".** The manuscript already argues that GED on
these cohorts is size-dominated (`05_results.tex:384–390`) and already refutes H3. T-28
supplies the constructive half the paper is missing: *here is a reference that is not size-
dominated, and under it the encoding does track structure.* That reframes a refuted
hypothesis as a scoped one, which is a much better position than a win claimed on a metric
nobody asked for.

---

## 5. Open, and not mine to close

1. **The paired bootstrap over the full cohort.** Campaign `2133405` (see §6). Until it
   lands, every win/loss count above is a point estimate and no verdict is a significance
   verdict. The `min_dfs` tie rests on 2 of 11 datasets.
2. **`fig:rho-vs-size` width defect, still open.** `05_results.tex:463–470` records that the
   figure is rendered 7.03 in into a 4.72 in text block, so its 5.5–6.5 pt labels reach the
   page at 3.7–4.4 pt — the comment says *"AND IT IS NOT FIXED"*. **The five new T-28 figures
   are 7.03 in too and inherit it.** A `--width` flag exists but is not sufficient on its own:
   `save_figure` writes `bbox_inches="tight"`, so at 4.72 in the seven-column legend overflows
   and the box expands back to 7.03 in with nothing in the output to say so. A genuine narrow
   render needs a narrower legend and a shorter title. This is the most reviewer-visible
   defect in the section and it affects the main fidelity figure.
3. **Caption polish** the PI asked for (`lower is better` / `higher is better`; shading the
   min-DFS gap in the coding-overhead inset) — separate task, untouched.

---

## 6. Provenance

| item | where |
|---|---|
| merge | `e9f458d` on `main`, bringing `feature/t28-correlation-driver` (18 commits) and `origin/main`'s `89b4c9f` |
| post-merge fix | `ede4af5` — the single-reference figure filtered through `design.ORDER` where its GED sibling used `FIGURE_ORDER`; it could have drawn `isalgraph_exhaustive` where the GED figure drops it |
| campaign of record | Picasso array **`2133405`**, `--qos=short`, 2 h wall, 15 tasks × 1 cell → `T28_metrics/families/f2_partials/` |
| superseded chain | `2132238` → `2132239`, `medium_uma`, **held** (`scontrol hold`), not cancelled — reversible, keeps queue age, and prevents two live arrays racing on the same partial |
| figures + profile | Picasso `T28_metrics/figures/`, copied to Sandisk `data/source/T28/figures/` (5 PDF, 5 PNG, `size_profile_all_references.json`, 8,232 rows) |
| reference matrices | Picasso `T28_metrics/references/` (75 NPZ, 1.1 GB), copied to Sandisk `data/source/T28/references/` |

**Why the campaign moved to the `short` QOS.** `2132238` sat on `Priority/` for six hours
with no SLURM start estimate. Measured on this cluster:

| QOS | priority | MaxWall |
|---|---:|---|
| `short` | **10000** | 02:00:00 |
| `medium_uma` | 1000 | 3-00:00:00 |
| `long_uma` | 500 | 7-00:00:00 |

The site weights QOS at 100000, so the contribution is 100000 under `short` against 10000
under `medium_uma`. `2132238` totalled 29,485 with 22 pending jobs ahead of it; the resubmission
started **within a minute**, all 15 tasks at once. The two LINUX cells returned in 1 m 35 s and
1 m 09 s. The whole cost is the 2 h wall, which is right for the light cells and may not hold
for `mutagenicity` (8.16 M pairs) and `coil_del` (7.60 M) — the shard loop is idempotent, so
releasing the held `medium_uma` array picks up exactly what times out and skips the rest.

`launcher.sh` gained `F2_QOS` and `SKIP_MERGE` for this. Both default to the previous
behaviour.
