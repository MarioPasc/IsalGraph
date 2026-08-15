# T-05 — response-letter fragment

**Serves AE.1** (size-scaling evidence) and **R3.7a** (the `n` ceiling *with its cause*); supplies
**R3.5b**'s per-dataset reference distance. Owner of the final prose: **T-14**. Not letter-ready
prose — it is the evidence, with every claim sourced.

---

## For AE.1 — "small-graph conclusions were licensed to n = 98"

We now report a **proven bracket** `LB ≤ GED ≤ UB` on **all 21,710,892 Suite-2 pairs**, computed
under a single cost model, rather than an unvalidated extrapolation. Every value comes from a method
with a published proof: `BRANCH_FAST` (Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018) for the
lower bound and `BIPARTITE` (Riesen & Bunke, *IVC* 27(7):950–959, 2009) for the upper.

**We measured how the bracket behaves as graphs grow, and we report it on both scales because they
disagree.** The absolute gap `UB − LB` **rises with `n` in all ten datasets** (every CI excluding
zero), while the relative width `(UB − LB)/UB` falls in six — because its denominator grows with `n`.
We lead with the absolute gap. Stating only the relative width would have implied the bracket
tightens at scale, which is not what the data show.

**We also separate whose limitation this is.** A widening bracket admits two readings — the encoding
degrades, or our *reference bound* degrades — so we computed a second, independent upper bound
(`BP_BEAM_DET`) over the same 21.7 M pairs. The frozen bound accounts for **63–88 %** of the widening
on the small-graph datasets but only **35–51 %** on the four spanning the disputed size range. **Most
of the large-`n` looseness is a property of the bound family, not of a choice we made**, and it would
survive replacing `BIPARTITE`. We report both arms; `BIPARTITE` remains primary.

## For R3.7a — the ceiling, with its cause

We extended exact GED from `n = 12` to a **measured ceiling of `n = 17`** with a calibration ladder
(rungs 13–18, 250 pairs each, `networkx.graph_edit_distance` under the same cost model, 1,200 s
per-pair budget, seed 42). The ladder truncates at the first rung certifying below a **pre-declared
25 %** threshold: rung 17 certified 28.4 %, rung 18 20.8 %.

**The cause is combinatorial, not implementational.** Certification falls 81.2 % → 20.8 % across five
nodes. Non-completing pairs are **interval-censored `[LB, UB]`**, never dropped and never promoted to
exact — dropping them would remove precisely the hardest cases and bias the ceiling upward.

## Provenance — one row per claim

| Claim | Source |
|---|---|
| 21,710,892 pairs, 16,370 graphs, ten datasets | `export_graphs_suite2 --verify-only`, exit 0 |
| absolute slopes, CIs, 10/10 positive | `results/reports/T-05-bounded-ged/data/s71_within_dataset_slopes.json` |
| gate share 63–88 % vs 35–51 % | same file, `gate_attribution`; also `data/summary.json` |
| ceiling `n = 17`; per-rung certification | `$SANDISK/data/source/APPROX_GED/ladder/manifest.json` + `rung_*.npz` |
| certification rate per dataset | `data/s72_certification.json` |
| 0 bracket violations over 21.7 M pairs | `$SANDISK/…/APPROX_GED/gates/gate_G3.json` |
| reproduces T-27 element-wise, 10,807,845 pairs, 3 arms | `gates/gate_G2.json` + orchestrator's independent re-check |
| cost model, options strings, per-file sha256 | `$SANDISK/…/APPROX_GED/manifest.json`, `PROVENANCE.md` |

## Do not write into the letter

- Any statement that the bracket **narrows** at scale.
- Any per-pair timing from `seconds_matrix` (in-worker solver time, not comparable across datasets;
  see the article notes' NOT CLAIMABLE section).
- Any claim connecting the bracket to IsalGraph's own accuracy — **§7.5, the deliverable that would
  do that, is deferred to T-06**. Nothing in T-05 measures IsalGraph.
- A certification rate without its dataset (they span a factor of 949).
- The **33.2 %** orientation figure (`decisions.md` §6): it belongs to graphs ~3× larger than those it
  was measured on, and the two upper bounds move in opposite directions in `n`.

## Obligation this ticket creates

The **D11 censored-interval upper ends are heuristic and run-dependent** — T-03 left `IPFP` on
`--randomness REAL`, so 74–82 % of those values change between runs. Exposure is bounded and verified
as **exactly the 61,084 interval upper ends**; the lower ends are unaffected. The PI accepted this
without repair, which makes **stating it obligatory** — otherwise a reader re-running our
reproduction script obtains different numbers, which is the reproducibility failure **R3.5a** raised.
