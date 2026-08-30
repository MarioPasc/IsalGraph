# T-28 addendum — the WL kernel reference, and a defect in this report

Written 2026-08-29. Companion to `REPORT.md`, which it **corrects in one place**
(§4) and **extends** in the rest. Driver: PI request of 2026-08-29 to try
alternative similarity metrics because §5.4's GED comparison risks rejection.

**A loss inside a confidence interval is a TIE**, as in `REPORT.md`. Point
estimates are labelled as such and are not verdicts.

---

## 0. 🔴 CORRECTION, 2026-08-30 — the campaign overturns the `min_dfs` claim

The paired bootstrap has landed on **14 of 15 cells** (`suite2/mutagenicity`
timed out at 10 h and was not resubmitted; the deadline is closer than the job).
It contradicts what the point estimates and the two-cell pilot said.

**Withdrawn.** The session-1 worklog concluded *"under the WL kernel reference
the arm is beaten by nothing"* — three significant wins and one tie — on the
strength of two cells. **That is wrong.** Over 14 cells:

| WL reference, `all_pairs` | verdict |
|---|---|
| vs `sparse6_nauty` | **14 W / 0 T / 0 L** |
| vs `nauty_graph6` | **12 W / 0 T / 2 L** |
| vs `agm_cam` | **8 W / 4 T / 2 L** |
| vs `min_dfs` | 3 W / 2 T / **9 L** |

**Why the pilot was wrong, and it is a familiar mechanism.** The two cells that
returned TIE (p = 0.70, p = 0.37) were both LINUX — the *smallest* cohort in the
campaign at 3,916 pairs, so the widest intervals and the least power in the whole
grid. A tie there was an artifact of low power, not a finding. `REPORT.md` warns
about exactly this: *"Do not read the unresolved fraction as a tie --- many
underpowered comparisons all leaning one way is evidence, not absence of it."*
The point estimates had been leaning one way the whole time: `min_dfs` ahead on
9 of 11 datasets, mean Δρ −0.027.

**IsalGraph does not beat, and does not tie, the minimum DFS code under any
reference tested.** It is the competitor a reviewer named as the most important
one. Say so plainly in §5.4 rather than let a reviewer recompute it.

**What survives, and it is still the strongest thing in the section:**

1. **Both nauty serialisations lose decisively, in both views** — `sparse6_nauty`
   14-0 and `nauty_graph6` 12-0-2 under `all_pairs`, 10-3-1 and 11-2-1 under
   `equal_n`. That is robust to the view, which the `agm_cam` result is not.
2. **The size-null result, which is the real finding** (§3.2, now with intervals
   on every cell): the arm's excess over its own `|n_i − n_j|` null is
   significantly positive on **12 of 14** cells against WL and on **6 of 14**
   against the best available GED reference. On the five Suite-1 datasets
   carrying *exact* GED it is **1 of 5 against GED and 5 of 5 against WL**.
3. `agm_cam` improves from 1 W / 0 T / 4 L under exact GED to 8 W / 4 T / 2 L
   under WL — **but only under `all_pairs`.** Under `equal_n` it is 2 W / 5 T /
   7 L, a net loss. This one is view-dependent and must be quoted with its view.

Numbers: `data/t28_bootstrap_verdicts.json`, per cell, both views, all eight
references. §3.3's point-estimate tables are left standing as what they are;
where they disagree with this section, **this section is the measurement**.

---

## 1. What changed, and what did not

**Only the reference.** Every representation distance `d_R` behind every number
here is the same cached T-04a matrix `REPORT.md` used — read from
`data/source/T06/distances/`, never recomputed. What moves is `d_ref`: the WL
subtree kernel distance and four spectral λ-distance variants are added beside
`exact` / `lb` / `ub`.

That is the ticket's acceptance criterion and it is enforced structurally: the
two sides of the correlation are stored in separate trees and joined on
`graph_ids` at analysis time, so a reference swap cannot touch the
representation side.

---

## 2. The answer, in four lines

1. **The WL kernel reference carries the result; the spectral family does not.**
   Of the five references added, one produces a win and four do not, and all
   five are reported.
2. **The clean result is on Suite 1.** On the five datasets carrying *exact*
   GED — where `REPORT.md` says *"No framing repairs this and none should be
   attempted"* — the canonical string clears its own size null on **1 of 5**
   against exact GED and **5 of 5** against the WL kernel. Same datasets, same
   representation distances, only the reference changed.
3. **This does not repair the GED finding and must not be presented as
   repairing it.** `REPORT.md` §Claim B stands exactly as written. What the WL
   measurement adds is a second reference under which the same encoding behaves
   differently, which makes the *reference* the binding constraint rather than
   the encoding. That is a scoping of H3, not a rescue.
4. **🔴 This report's `wl_subtree` rows are wrong** — a defect in
   `size_profile.py`, not in the data. See §4. It reaches a manuscript claim.

---

## 3. The WL reference, measured

### 3.1 How size-dominated each reference is

`ρ(|n_i − n_j|, d_ref)` over the 11 distinct datasets. This is the quantity
`REPORT.md` calls the size null, computed for the reference itself.

| reference | min | median | max | > 0.90 | arm clears null |
|---|---:|---:|---:|---:|---:|
| GED exact | 0.713 | 0.914 | 0.920 | 3/5 | 1/5 |
| GED lower bound | 0.879 | **0.972** | 0.997 | 9/10 | **0/10** |
| GED upper bound | 0.340 | 0.702 | 0.754 | 0/10 | 7/10 |
| **WL kernel** | **0.159** | **0.570** | 0.973 | 1/11 | **8/11** |
| spectral (norm L, padded) | 0.818 | 0.923 | 0.986 | 8/11 | 0/11 |
| spectral (comb L, padded) | 0.560 | 0.895 | 0.988 | 5/11 | 2/11 |
| spectral (adjacency) | 0.924 | 0.971 | 0.996 | 11/11 | 0/11 |
| spectral ESD | **−0.061** | **0.303** | 0.551 | 0/11 | 0/11 |

Two things to read off it. The **lower bound is unclearable by construction** —
median size null 0.972 — so the `8 of 25` in `REPORT.md` has ten records in its
denominator that nothing could have cleared. And **`spectral_esd` is the best
size control of the eight and still fails**, which is the cleanest evidence that
the WL result is not an artifact of picking a low-null reference: the lowest-null
reference of all is the one the encoding tracks worst.

### 3.2 Suite 1 — the like-for-like comparison

Excess = arm ρ − its own size null. Five datasets, exact GED, no bracket.

| dataset | vs exact GED | vs WL kernel |
|---|---:|---:|
| `aids` | −0.5445 | **+0.1027** |
| `iam_letter_high` | −0.2536 | **+0.1676** |
| `iam_letter_low` | +0.0139 | **+0.1432** |
| `iam_letter_med` | −0.0313 | **+0.1950** |
| `linux` | −0.2392 | **+0.3270** |
| **clears its own size null** | **1 / 5** | **5 / 5** |

**Now with bootstrap intervals on every cell** (14 of 15; `all_pairs`; "+" means
the excess interval is strictly above zero):

| cell | GED ref | GED excess [95 % CI] | WL excess [95 % CI] |
|---|---|---|---|
| 1/`aids` | exact | −0.4597 [−0.4983, −0.4210] − | **+0.1121 [+0.0593, +0.1668] +** |
| 1/`iam_letter_high` | exact | −0.2536 [−0.2691, −0.2387] − | **+0.1676 [+0.1434, +0.1911] +** |
| 1/`iam_letter_low` | exact | +0.0139 [+0.0057, +0.0235] + | +0.1432 [+0.1209, +0.1664] + |
| 1/`iam_letter_med` | exact | −0.0313 [−0.0438, −0.0190] − | **+0.1950 [+0.1715, +0.2192] +** |
| 1/`linux` | exact | −0.2247 [−0.3492, −0.0922] − | **+0.3189 [+0.1699, +0.4454] +** |
| 2/`aids_graphedx` | ub | +0.0330 [+0.0058, +0.0593] + | +0.0688 [+0.0288, +0.1090] + |
| 2/`aids_iam` | ub | +0.0303 [+0.0213, +0.0388] + | +0.0201 [+0.0085, +0.0313] + |
| 2/`coil_del` | ub | +0.2386 [+0.2259, +0.2507] + | −0.0384 [−0.0424, −0.0346] **−** |
| 2/`grec` | ub | +0.1268 [+0.1057, +0.1488] + | +0.0276 [+0.0084, +0.0479] + |
| 2/`iam_letter_high` | ub | −0.1092 [−0.1281, −0.0907] − | **+0.1676 [+0.1434, +0.1911] +** |
| 2/`iam_letter_low` | ub | −0.0383 [−0.0494, −0.0278] − | **+0.1432 [+0.1209, +0.1664] +** |
| 2/`iam_letter_med` | ub | −0.0482 [−0.0640, −0.0327] − | **+0.1950 [+0.1715, +0.2192] +** |
| 2/`linux` | ub | +0.0214 [−0.0882, +0.1154] 0 | +0.3270 [+0.2003, +0.4474] + |
| 2/`protein` | ub | +0.4094 [+0.3654, +0.4546] + | −0.0975 [−0.1222, −0.0757] **−** |
| **significantly above** | | **6 / 14** | **12 / 14** |

**Seven cells flip** from significantly below (or undetermined) under GED to
significantly above under WL. **Two go the other way** — `coil_del` and
`protein`, where the upper bound is the friendlier reference. On Suite 1, where
exact GED exists and no bracket argument applies, it is **1 of 5 against GED and
5 of 5 against WL**. That is the cleanest statement in this report.

### 3.3 Head-to-head, `all_pairs`, point estimates

Cells where the arm's ρ exceeds the competitor's, over **11 distinct datasets**
(see §5.1 on why not 15), mean Δρ in brackets.

| reference | `agm_cam` | `min_dfs` | `nauty_graph6` | `sparse6_nauty` |
|---|---|---|---|---|
| GED exact | 1/5 (−0.225) | 0/5 (−0.154) | 3/5 (+0.053) | 4/5 (+0.136) |
| GED lower bound | 0/10 (−0.168) | 1/10 (−0.082) | 4/10 (+0.009) | 9/10 (+0.101) |
| GED upper bound | 6/10 (+0.106) | 1/10 (−0.042) | 2/10 (−0.034) | 2/10 (−0.024) |
| **WL kernel** | **11/11 (+0.187)** | 2/11 (−0.027) | **8/11 (+0.105)** | **11/11 (+0.170)** |
| spectral (norm L) | 0/11 (−0.279) | 1/11 (−0.120) | 4/11 (−0.021) | 8/11 (+0.098) |
| spectral (comb L) | 3/11 (−0.032) | 3/11 (−0.070) | 9/11 (+0.156) | 11/11 (+0.188) |
| spectral (adjacency) | 0/11 (−0.317) | 1/11 (−0.134) | 3/11 (−0.127) | 5/11 (−0.001) |
| spectral ESD | 3/11 (−0.243) | 3/11 (−0.056) | 9/11 (+0.090) | 11/11 (+0.153) |

`wl_subtree` is excluded under `wl`: its distance **is** the reference, verified
byte-identical on 15 of 15 cells, so ρ ≡ 1 is the identity and not a result.

### 3.4 Paired bootstrap over the cohort — **this supersedes §3.3**

14 of 15 cells. Per-cell differences in `data/t28_bootstrap_verdicts.json`.

| reference | `agm_cam` | `min_dfs` | `nauty_graph6` | `sparse6_nauty` | clears null |
|---|---|---|---|---|---:|
| GED exact | 1W 0T 4L | 0W 0T 5L | 3W 1T 1L | 3W 2T 0L | 1/5 |
| GED lower bound | 0W 0T 9L | 1W 0T 8L | 4W 0T 5L | 7W 2T 0L | 0/9 |
| GED upper bound | 1W 5T 3L | 1W 2T 6L | 2W 2T 5L | 2W 2T 5L | 5/9 |
| **WL kernel** | **8W 4T 2L** | 3W 2T **9L** | **12W 0T 2L** | **14W 0T 0L** | **12/14** |
| spectral (norm L) | 0W 0T 14L | 1W 0T 13L | 7W 0T 7L | 9W 5T 0L | 0/14 |
| spectral (comb L) | 4W 4T 6L | 3W 2T 9L | 12W 1T 1L | 14W 0T 0L | 2/14 |
| spectral (adjacency) | 0W 0T 14L | 1W 0T 13L | 3W 2T 9L | 6W 2T 6L | 0/14 |
| spectral ESD | 4W 3T 7L | 1W 2T 11L | 9W 5T 0L | 12W 2T 0L | 0/14 |

Same table under `equal_n`, where the size channel is removed by construction:

| reference | `agm_cam` | `min_dfs` | `nauty_graph6` | `sparse6_nauty` |
|---|---|---|---|---|
| GED exact | 0W 0T 5L | 0W 1T 4L | 2W 1T 2L | 1W 3T 1L |
| **WL kernel** | 2W 5T **7L** | 2W 3T **9L** | **11W 2T 1L** | **10W 3T 1L** |
| spectral (comb L) | 2W 8T 4L | 2W 3T 9L | 14W 0T 0L | 11W 3T 0L |

**Read the two views together.** The nauty result holds in both and is the
robust win. The `agm_cam` result does not: 8W 4T 2L pooled becomes 2W 5T 7L once
size is controlled, so most of that gain is size agreement rather than structure.
`min_dfs` is 9 losses either way.

**IsalGraph does not beat and does not tie `min_dfs` under any reference
tested.** §0 records why the two-cell pilot said otherwise.

---

## 4. 🔴 Correction to this report: the `wl_subtree` rows are wrong

`REPORT.md` §"Claim B per competitor, inside equal-`n` strata" reports, for
`wl_subtree` at `n <= 20` against `exact`: **16 higher, 4 lower, median Δρ
+0.0546, sign test p = 0.012**. Recomputed against the cached distance matrices
the same row is **1 higher, 18 lower, median Δρ −0.1116, p = 7.6e-05** — the
opposite direction, at higher significance.

**The cause is a defect in `size_profile.py::_wl_counts`, not in the data.** The
stored WL encoding is `h<level>:<colour>:<count>`; the multiplicity is written
into the symbol. `_wl_counts` took `Counter(seq)` over whole symbols, on the
stated premise that the encoding was *"the `symbol_sep`-joined multiset of WL
colours"*. It is not. A symbol occurs exactly once per sequence, so `Counter`
returned 1 for every one and the result was a **presence indicator, not a count
vector**.

Measured on `suite1/aids`:

| | vocabulary | largest cell | reproduces cached matrix |
|---|---:|---:|---|
| `_wl_counts` as published | 208 `colour:count` tokens | **1.0** | no, max abs difference **18.34** |
| counts parsed out of the symbol | **69** colours | 12.0 | **yes, max abs difference 0.0** |

The cached matrix's own metadata declares `vocabulary_size: 69`, `h: 2`,
`normalize: false`. **The cached matrices were right all along; the profiler was
recomputing them wrongly.** Fixed at `6b89b4f`; the fix reproduces all 15 cached
cells exactly.

**Scope, bounded by measurement rather than by argument.** `wl_subtree` is the
only arm whose primary distance is `kernel`, so it is the only arm that can be
affected. Joining T-06's `size_profile.json` against the T-28 cached profile on
`(suite, dataset, representation, reference, n)`:

| representation | rows differing | common rows | max abs Δρ |
|---|---:|---:|---:|
| `agm_cam` | 0 | 137 | 0.000000 |
| `isalgraph_canonical` | 0 | 23 | 0.000000 |
| `isalgraph_pruned` | 0 | 419 | 0.000000 |
| `min_dfs` | 0 | 401 | 0.000000 |
| `nauty_graph6` | 0 | 419 | 0.000000 |
| `sparse6_nauty` | 0 | 419 | 0.000000 |
| **`wl_subtree`** | **401** | 419 | **1.104798** |

`n_pairs` is identical on every one of the 2,237 common rows — same graphs, same
strata, different distances. **No IsalGraph claim in `REPORT.md` moves.**

### 4.1 It reaches the manuscript

`05_results.tex:395–401` states:

> At $n \leq 20$ against exact graph edit distance the instruction string
> correlates significantly better than \texttt{nauty}-\texttt{graph6},
> \texttt{nauty}-\texttt{sparse6} and the Weisfeiler--Lehman kernel
> ($p = 0.041$, $0.041$, $0.012$) and worse than the minimum DFS code and AGM CAM.

The two `nauty` p-values are computed from `levenshtein` arms and are
**unaffected** — both reproduce exactly. **The `p = 0.012` claim against the
Weisfeiler–Lehman kernel inverts**: significantly worse, `p = 7.6e-05`. It must
be corrected in the revision, not carried forward.

Corrected row set, recomputed on deduplicated `(dataset, n)` strata:

| competitor | band | ref | strata | higher | lower | median Δρ | sign p |
|---|---|---|---:|---:|---:|---:|---:|
| `wl_subtree` | n ≤ 20 | exact | 23 | 1 | 18 | −0.1116 | 7.6e-05 |
| `wl_subtree` | n ≤ 20 | lb | 88 | 10 | 71 | −0.2862 | 1.8e-12 |
| `wl_subtree` | n ≤ 20 | ub | 88 | 49 | 33 | +0.0223 | 0.097 |
| `wl_subtree` | n > 20 | lb | 110 | 0 | 110 | −0.6822 | 1.5e-33 |
| `wl_subtree` | n > 20 | ub | 110 | 29 | 81 | −0.1345 | 7.3e-07 |

Every other row of that table reproduces `REPORT.md` exactly, which is what
localises the defect to this arm.

---

## 5. Two things that would overstate the result

### 5.1 Four of the fifteen cells are duplicates

Suite 1 and Suite 2 share `linux`, `iam_letter_low`, `iam_letter_med` and
`iam_letter_high`. **240 records are identical in ρ, `n_pairs` and `n_graphs`
across the two suites**; none differ. Correct behaviour — the suite axis
separates which *GED* reference is available, and a structural reference is the
same matrix in both — but it means a T-28 count over "15 cells" counts four
datasets twice. The cohort is **11 distinct datasets**. `12/15` is `8/11`.

### 5.2 The `all_pairs` sweep over `agm_cam` is a size effect

| competitor | WL / `all_pairs` | WL / `equal_n` |
|---|---|---|
| `agm_cam` | 11/11 (+0.187) | 4/11 (+0.016) |
| `min_dfs` | 2/11 (−0.027) | 2/11 (−0.049) |
| `nauty_graph6` | 8/11 (+0.105) | 8/11 (+0.084) |
| `sparse6_nauty` | 11/11 (+0.170) | 8/11 (+0.070) |

Once size is removed by construction the `agm_cam` sweep becomes a tie. The
manuscript's headline size-null table is `all_pairs` and `fig:rho-vs-size` is
`equal_n`; do not quote the sweep beside that figure. The improvement is still
real under `equal_n` — the manuscript currently concedes `agm_cam` as a loss.

---

## 6. Figures

| file | what |
|---|---|
| `figures/fig_rho_vs_size_wl.{pdf,png}` | the WL reference alone: one panel, one axis, every arm. No bracket and no regime split, because the WL kernel distance is exact at every size. `wl_subtree` draws flat at ρ ≡ 1 and is annotated as the identity. |
| `figures/fig_rho_vs_size_wl_vs_ged.{pdf,png}` | **(a)** WL kernel, **(b)** exact GED (n ≤ 12), **(c)** the GED bracket (n > 12). Shared y axis, ticks on (a) only, one legend, one BH correction over all 908 points. |

Both are 7.03 in wide, matching `fig1_rho_vs_size.pdf` exactly — **and
inheriting its open placement defect**: rendered 7.03 in into Pattern
Recognition's 4.72 in text block, declared 5.5–6.5 pt labels reach the page at
3.7–4.4 pt. `05_results.tex:463–470` records this as unfixed. A `--width` flag
exists and is **not sufficient alone**: `save_figure` writes with
`bbox_inches="tight"`, so at 4.72 in the seven-column legend overflows and the
box expands back to 7.03 in with nothing in the output to say so.

**The x scale is not shared across the three panels and the figure says so.**
Exact GED spans 10 strata and the other two span 62; a common scale would render
panel (b) about 0.4 in wide. The y axis *is* genuinely shared. Panels (b) and (c)
stay separate for the reason `figure_one` splits them — a bracket sharing an axis
with an exact value invites the reader to read one as the other.

**Panels (b) and (c) draw `wl_subtree` from the corrected cached matrices**, so
that series does **not** match `fig1_rho_vs_size.pdf`, which carries the §4
defect. Every other series matches it exactly.

---

## 7. Data added to this report

| path | contents |
|---|---|
| `data/t28_probe_point_estimates.json` | 1,260 point-estimate records: 15 cells × every representation × all 8 references × 2 views |
| `data/t28_size_profile_all_references.json` | 8,232 stratum rows behind the figures, all 8 references |
| `data/t28_headtohead.json` | the §3.3 and §5.2 tables as data |
| `data/t28_signtest_equal_n.json` | the §4.1 corrected sign-test rows |

The reference matrices themselves (75 NPZ, 1.1 GB) stay under
`data/source/T28/references/`, matching this report's convention of not copying
distance matrices in.

---

## 8. Status

- [x] WL and four spectral variants measured on all 15 cells, point estimates
- [x] Reference matrices independently re-verified: G3 75/75 clean, G4 15/15
      byte-identical, G5 max off-diagonal zero fraction 0.155
- [x] `_wl_counts` defect found, fixed, and verified against 15/15 cached cells
- [x] Both figures rendered
- [x] **Paired bootstrap, 14 of 15 cells** — see §0 and §3.4. It **overturned**
      the two-cell `min_dfs` tie. `suite2/mutagenicity` timed out at 10 h and was
      **not** resubmitted: the deadline is closer than the job. Every claim here
      is therefore over 14 cells / 10 distinct datasets, and says so.
- [ ] §5.4 rewrite, and the `p = 0.012` correction of §4.1

---

## 9. The campaign, and what it cost to measure

`2132238` (`medium_uma`, 10 h) sat on `Priority/` for six hours with no SLURM
start estimate, having produced nothing. Measured on this cluster:

| QOS | priority | MaxWall |
|---|---:|---|
| `short` | **10000** | 02:00:00 |
| `medium_uma` | 1000 | 3-00:00:00 |
| `long_uma` | 500 | 7-00:00:00 |

The site weights QOS at 100000, so the contribution is 100000 under `short`
against 10000 under `medium_uma`. `2132238` totalled 29,485 with 22 pending jobs
ahead of it. Resubmitted as **`2133405`** under `short` with a 2 h wall it
**started within a minute**, all 15 tasks at once.

**6 of 15 completed; 9 hit the 2 h wall.** The shard loop is idempotent — it
skips any cell whose partial exists — so nothing is recomputed.

**The rerun sizing in this section was wrong for the heavy tail and is left
standing as the record.** It predicted 3-5 h per heavy cell from the group
timings. Measured: `coil_del` took **9 h 40 m** and `mutagenicity` exceeded 10 h.
The group lines capture only about a sixth of a shard's wall time -- the
bootstrap is not in them -- so extrapolating from them underestimates by roughly
that factor. A pairs-linear estimate (27-29 h) was too high and this one too low;
the honest reading was that neither method was calibrated.

| landed | elapsed |
|---|---|
| `suite1/linux`, `suite2/linux` | 1 m 09 s, 1 m 35 s |
| `suite2/protein` | 38 m |
| `suite2/grec` | 46 m |
| `suite1/aids` | 60 m |
| `suite2/aids_graphedx` | 1 h 13 m |

**Sizing the rerun from the timed-out shards rather than from pair counts.** A
cell is about four comparison groups, and `equal_n` is far cheaper than
`all_pairs` — on `coil_del`, 349 s against 2806 s for the same group. Within its
2 h `coil_del` reached 3 of 4 groups and `mutagenicity` 2 of 4, so the heavy
cells need roughly **3–5 h each**. A pairs-linear extrapolation predicts 27–29 h
for the same two cells and is wrong, because cost is not linear in pairs across
the group structure. The held 10 h array is therefore adequate and was released
rather than resubmitted with a longer wall, which would only have backfilled
worse.

`2132239` (merge) is left **held** on purpose: its `afterok` cannot be satisfied
if any shard times out again, and a held job is clearer than
`DependencyNeverSatisfied`. Submit it by hand once the fifteenth partial lands —
the merge aborts on an incomplete set, and asserts `N_actual == 79`.
