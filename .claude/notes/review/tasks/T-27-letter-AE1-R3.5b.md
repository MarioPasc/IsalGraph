# T-27 — response-letter fragment

**Serves**: AE.1 (size scaling), R3.5b (one cost model, defensible references), R3.7a
**Owner**: T-14 assembles · **Status**: draft, numbers final
**Do not paste without §Disclosure.** The selection is favourable at the lower end and *constrained*
at the upper end, and a fragment that reports only the first is the version a reviewer catches.

---

## Draft

> Both ends of the proven bracket are now chosen by measurement rather than by citation. We
> evaluated every proven bound the reference implementation offers — four lower bounds and four
> upper bounds, in twelve configurations — against the complete exact-GED census of the five
> datasets where exact GED is computable, under the single unit cost model used throughout:
> **3,836,827 certified exact values, 46,774,932 bound evaluations, and no violation of any proven
> bound.**
>
> For the lower bound the result is stronger than a measurement. Blumenthal et al. prove that BRANCH
> and BRANCH-FAST coincide when edge edit costs are constant, which they are under our cost model;
> we confirm this on every one of the 3,836,827 certified pairs, with maximum absolute difference
> zero. The choice of BRANCH-FAST is therefore a cost decision between two provably identical
> bounds, not an extrapolation from a sample, and it holds at every graph size rather than only in
> the regime we could measure.
>
> For the upper bound we report a constraint. IPFP is the tightest upper bound we measured, and it
> also confirms the ordering its authors report. It costs 0.81 s per pair at n̄ = 30 — more than
> eight hundred times our pre-declared budget of 1 ms per pair, and roughly 9,700 core-hours over
> the Suite-2 census. We therefore report BIPARTITE, which meets the budget, and we state plainly
> that it is the loosest of the seven upper bounds measured and that this widens the bracket. The
> tighter methods and their costs are given in full, so the trade-off is the reader's to inspect.
>
> One further result bears on reproducibility. The reference implementation's local-search upper
> bounds initialise from a random node map with a non-reproducible seed by default, and under those
> defaults their value changes between runs on 91–94 % of pairs, by as much as ten edit operations.
> Every bound we report is computed under explicitly pinned options, listed with the results, under
> which the value is invariant across repetitions. We would encourage the same practice generally:
> a method name alone does not specify a graph edit distance computation.

*(≈ 290 words)*

## Disclosure — must accompany the fragment

Two costs of the cost-gated upper bound, both measured, both quantified in the report:

1. **The bracket becomes uninformative on 2 of 5 datasets** under BIPARTITE by our own pre-declared
   rule — ρ(Lev, UB) − ρ(Lev, exact) is −0.219 on Letter LOW and −0.177 on Letter MED, with
   bootstrap CIs excluding zero. Under BP-BEAM (deterministic), which misses the budget by 17 %, no
   dataset triggers it.
2. **BIPARTITE's relative error grows about ten times faster with graph size** than any alternative
   (AIDS, slope +0.294 per node against IPFP's +0.029). Since AE.1 is precisely about size scaling,
   this is adverse and must be surfaced by us rather than found by the reviewer.

If the reviewer pushes on the upper bound, the honest answer is that the budget was fixed before the
measurement and BP-BEAM missed it by 17 %; the trade-off is documented and the arm is available.

## Provenance

| Claim | Source |
|---|---|
| 3,836,827 certified · 46,774,932 evaluations · 0 violations | `data/analysis/validity.json` |
| BRANCH ≡ BRANCH_FAST, max diff 0.0, all 5 datasets | `validity.json` → `proven_orderings` |
| Equivalence is proven for constant edge costs | survey §5.2.4, `10.1007/s00778-019-00544-1` |
| IPFP 808,120 µs/pair at n̄ = 29.5; ≈ 9,742 core-h over Suite 2 | `data/timing/probe_n30__IPFP_MS.json`; 21,710,892 pairs (decision 27) |
| BIPARTITE loosest of seven — 1.095 vs IPFP 0.084 | `data/analysis/metrics.json` |
| 91–94 % varying at defaults, 0 % pinned | `data/determinism/*.json`, 60 probes |
| D13 fires on 2 of 5 | `data/analysis/bootstrap.json` → `d_rho_lev` |
| Error-vs-n slopes | `metrics.json` → `error_vs_n` |
| IPFP = *Pattern Recognition Letters* 87:38–46, 2017 | Crossref, verified |

**Not in this fragment, deliberately**: any claim that the selection transfers to n = 98 — it was
made at n ≤ 12 and item 2 of the disclosure is a reason to doubt it for BIPARTITE. That belongs to
T-05's bracket-width-versus-n measurement.
