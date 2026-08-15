# T-04 — response-letter fragment

**Draft material for `reviews/response_to_reviewers.tex`. Not final prose** — `review-answer`
writes the answers; this supplies the claims and their provenance. Serves **AE.4a** (the
requirement-modal owner), **AE.3**, **R1.1**, **R1.2a/b**, **R3.6a**.

---

## AE.4a / AE.3 — the choice of benchmark models, and the side-by-side comparison

We now compare against **eleven registered representations** rather than asserting IsalGraph's
properties in isolation, and the pool is structured rather than arbitrary. Four of them — the raw
adjacency triangle, graph6, nauty→graph6 and the AGM canonical code — emit **the same bit sequence**
and differ on exactly two orthogonal choices, raw bits versus 6-bit ASCII packing and incident
versus canonical labelling. **That is what lets us isolate canonicity as a variable at fixed
format**: holding graph6 fixed and changing only the labelling moves isomorphism invariance from
`4/50` to `50/50` and equal-`n` ρ from `0.539` to `0.974` on Letter LOW. The remaining members cover
the mining literature's two canonical families (CAM and M-DFSC, per Jiang, Coenen & Zito 2013), the
one `m`-scaling non-canonical format (sparse6), a kernel method (WL), and a trivial size baseline.

## R1.2a/b — uniqueness, and the comparison we now lead with

Restricting to graph pairs of equal order removes the size channel and leaves pure structure. There
the canonical representations score **0.97–1.00** against the non-canonical ones' **0.54–0.61** on
Letter LOW — a gap of **0.42–0.46** that the all-pairs view hides entirely. We report the equal-`n`
comparison as primary for this reason.

We also report a **completeness witness**: `K₃,₃` and the triangular prism are both connected and
3-regular on six vertices and are not isomorphic. The WL subtree kernel assigns them distance
**exactly 0.0000 at every number of refinement rounds** — 1-WL cannot separate two regular graphs of
the same degree and order — while every canonical member of the pool separates them. One six-node
figure carries the uniqueness argument.

## R1.1 — the proxy and the runtime comparison

Both are now measured against T-03's **certified exact** GED under a single unit cost model, on
certified-exact pairs only, rather than against an approximation.

On runtime we **language-match**: our previous figure compared a tuned C++ implementation against
interpreted baselines, which is the category error the reviewer identified. Every timing we now
report records the engine that produced it, and the competitor comparison runs **both arms in pure
Python** (GREC: min-DFS 1.03 ms/graph, IsalGraph 17.6 ms/graph).

## R3.6a — narrowing the message-length claim

The reviewer asked us to narrow the claim, and the measurement narrows it further than the request.
**IsalGraph is never the most compact representation on the small-graph suite**; the raw adjacency
matrix wins every dataset there, and on the three Letter sets IsalGraph is shorter on **0.0 %** of
graphs. The claim that survives is:

> IsalGraph is shorter than every other **string** serialisation and than the explicit-construction
> reference model; the raw adjacency matrix is shorter at these sizes, and the crossover is at
> `n ≈ 14` and low density.

The crossover is governed by **`m/n`, not by `n`**: on Mutagenicity (`n̄ = 27.9`, `m/n = 1.03`)
IsalGraph wins outright at 147.4 bits against the adjacency matrix's 300.0, while on Protein
(`n̄ = 31.9`, `m/n = 1.94`) — a *larger* dataset — it only ties.

## R3.6b — "strongly correlates", and the baseline we now print beside it

We have withdrawn the unqualified correlation claim. A trivial predictor — the absolute difference
in node count, with no representation at all — scores **ρ = 0.71–0.93** against exact GED on the
five small-graph datasets. IsalGraph exceeds it on **one** of the five, by 0.026, and on no dataset
by a margin that survives resampling: repeated 200-graph draws of the same dataset move ρ by up to
0.07. **Every correlation we report is now printed beside this baseline**, and the null is a
registered representation in the codebase so that a table without it cannot be produced.

---

## Provenance — one row per claim

| Claim | Artifact | Status |
|---|---|---|
| eleven backends, one bit sequence across the `n²` family | `src/isalgraph/competitors/`, `tests/unit/test_competitors_serial.py` | asserted in code over 7 fixtures + 300 random graphs + 8 boundary sizes |
| F3 `4/50 → 50/50`, equal-`n` ρ `0.539 → 0.974` | `competitors/README` §2 | scout, **reproduced** from `src/` |
| equal-`n` gap 0.42–0.46 | `corrected_rho_table.json` | **measured**, one draw, frozen conventions |
| WL distance exactly 0.0000 at `h = 1,2,3,5` | `reproduce --mode artefacts`, `tests/unit/test_wl_subtree.py` | **measured** |
| certified-exact GED reference | T-03, D6 cost model `[1,1,0,1,1,0]` | inherited |
| language-matched timings | `smoke_picasso_suite2.json`, engine `python` recorded in the header | **measured** |
| adjacency wins Claim A on Suite 1; 0.0 % on Letter | `competitors/README` §4.3, Suite-1 rows | scout, **reproduced** (Suite-1 rows are full-cohort) |
| Mutagenicity 147.4 vs 300.0; `m/n` crossover | `competitors/README` §4.3 | scout; **Suite-2 rows are draw-dependent — requote with the draw** |
| size null ρ 0.71–0.93; IsalGraph clears it on 1 of 5 | `corrected_rho_table.json` | **measured**, supersedes §4.1 |
| ρ moves up to 0.07 between draws | finding 14, corroborated by T-04's three-draw discrepancy | inherited + **corroborated** |

> ⚠ **Two things this fragment must not be edited into.** It does **not** claim IsalGraph clears the
> size null on two of five datasets — that figure came from a superseded composite table. And no ρ
> ordering here should be stated as significant until the graph-level bootstrap CIs exist (T-02 →
> T-06). Full list: `T-04-article-notes.md`, *"What is NOT claimable"*.
