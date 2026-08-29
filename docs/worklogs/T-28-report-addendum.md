# T-28 addendum — the WL kernel reference, and a defect in this report

Written 2026-08-29. Companion to `REPORT.md`, which it **corrects in one place**
(§4) and **extends** in the rest. Driver: PI request of 2026-08-29 to try
alternative similarity metrics because §5.4's GED comparison risks rejection.

**A loss inside a confidence interval is a TIE**, as in `REPORT.md`. Point
estimates are labelled as such and are not verdicts.

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

On LINUX this is already significant with paired graph-level bootstrap
intervals: excess **−0.2247 [−0.3492, −0.0922]** against exact GED,
**+0.3189 [+0.1699, +0.4454]** against WL.

On Suite 2 it is a wash: the upper bound clears 7/10 and WL clears 7/10, and WL
is *worse* on `coil_del`, `mutagenicity` and `protein`. State that alongside.

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

### 3.4 Paired bootstrap, where it exists

Two of eleven datasets so far (`linux`, both suites). The rest is Picasso array
`2133405`.

| reference | comparator | arm − comparator [95 % CI] | verdict |
|---|---|---|---|
| GED exact | `min_dfs` | −0.1691 [−0.2785, −0.0784] | **LOSS** (p = 0.002) |
| **WL kernel** | `min_dfs` | **−0.0235 [−0.1470, +0.0923]** | **TIE** (p = 0.70) |
| **WL kernel** | `min_dfs` (suite 2) | **−0.0413 [−0.1326, +0.0508]** | **TIE** (p = 0.37) |
| WL kernel | `agm_cam` | +0.1831 [+0.0459, +0.3073] | WIN |
| WL kernel | `nauty_graph6` | +0.2237 [+0.0905, +0.3563] | WIN |
| WL kernel | `sparse6_nauty` | +0.2466 [+0.1378, +0.3524] | WIN |

**IsalGraph does not beat `min_dfs` under any reference.** It ties it under WL
where it loses under exact GED. That is the honest claim and it is enough.

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
- [ ] **Paired bootstrap over all 11 datasets** — see §9. **6 of 15 shards
      landed.** Until the rest do, §3.3 is point estimates and no count there is
      a significance verdict.
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
