# Wave contract — 2026-08-30-article-wl

> ## 🔴 COMPRESSION DIRECTIVE — author decision, 2026-08-30. Binds the compression wave.
>
> The merged manuscript measures **49 pages against a hard 35**. Three sections measured their own
> claim-free headroom with `\iffalse` and it totals **~1.3 p**, so prose trimming cannot close it.
> **Every claim stays in the body; the apparatus behind it moves to the uncapped supplementary PDF.**
>
> **What moves**
>
> - **The statistical apparatus.** §4.5/§4.6 protocol detail, §5's MRM block with its VIF figures and
>   exclusions, the per-dataset and per-competitor grids. Destinations S6 and S8.
> - **The greedy algorithm listing.** S9 already carries the exhaustive and the pruned canonical
>   searches; **extend it to the greedy encoder with the same figures and pseudocode**, so all three
>   sit together. The body keeps a prose description and a pointer.
>
> **What does NOT move, by author instruction**
>
> - 🔴 **Nothing that describes the method.** §3's exposition of the virtual machine, the instruction
>   set, displacement pairs, canonicalisation and the admissibility definition **stays in the body.**
>   §3 is therefore a *smaller* compression target than a page count alone would suggest.
> - 🔴 **The edit-path result and `fig:edit-path`.** 92.03 % against 52.26 %. This is the
>   instruction-space comparison against the minimum DFS code and it is protected regardless of cost.
> - 🔴 **The controlled cost-law experiment in §5.3.** 12 cells, ρ = +0.892, p = 0.0064, flat
>   search-free null. The only claim in the paper established by a controlled rather than an
>   observational experiment.
> - **Table 1 stays**, and this is load-bearing given the greedy listing is leaving: the defect a
>   reviewer found in the greedy algorithm was found by comparing the listing *against Table 1*. With
>   the listing in the supplement, the body must keep both Table 1 and a prose description precise
>   enough that the demand is still discharged from the body alone. Say in the prose what the guards
>   and the duplicate checks do, and which pointer each acts on.
>
> **Everything else is at the compression agent's discretion, subject to one rule: a claim never
> moves, is never compressed, and never loses the scope clause that qualifies it.** Where a table
> and a claim compete for the same space, the table moves.
>
> 🔴 **THE AUTHOR BIOGRAPHIES STAY. Corrected 2026-08-30 — I authorised this cut and had no
> standing to.** `backmatter.tex` carries the corresponding author's own instruction, dated
> 2026-08-27: *"DO NOT COMMENT THIS BLOCK OUT AGAIN to recover pages."* Separately, the user was
> offered "Author biographies dropped, 0.72 p" as a selectable option early in the session **and did
> not select it**; I then carried it into a later option description as though it were settled.
> And reinstating them is itself a demand (E12), so cutting them un-discharges something the paper
> is meant to answer in order to buy one page. Worth ~1 p measured. **Not available.**
>
> 🔴 **The citation plan COSTS pages and the ledger must carry it.** Measured clean-room: 7 new
> citations = **+1 p**, at roughly three printed lines per bibliography entry. The full plan takes
> the cited count from 27 to ~47, so budget **≈ +2 p across the wave**. It is worth paying — six of
> the Tier-1 items are attribution defects, and a submission naming `BRANCH_FAST`, `BIPARTITE`,
> GEDLIB, `grakel` and Benjamini–Hochberg while citing none of them is a worse problem than a page
> — but it is a cost, not a freebie.
>
> **Measured page-neutral, do not spend effort there**: §1, §2, §6, §7, the abstract, and the back
> matter. Each was measured by its own agent with `\iffalse` or a clean-room rebuild. **The gap
> closes in §3, §4 and §5 or it does not close.**

**Every agent in this wave reads this file first and treats it as frozen.** Where it disagrees
with anything else, ask the orchestrator (`main`) via `SendMessage`; do not resolve it yourself.

Base commit: `0e166dc344a4b29c79498a8252ec00844e21ac4b` (article repo, branch `master`).

---

## 1. What this wave does

The manuscript is a complete, green-building draft of the Pattern Recognition revision. Three
things are wrong with it and this wave fixes the first two.

1. **A second distance reference (Weisfeiler–Lehman) has been measured and is not in the paper.**
   Its artifacts are staged in the article directory and wired into nothing.
2. **The prose reads as machine-written in places, and it refers to the review process.** Both go.
3. **It is 44 pages against a hard 35.** *Not this wave's problem.* The user has decided:
   **content first, a dedicated compression pass after.** Write the best version of your section.
   Take trims that cost nothing — a redundant subsection preamble, a sentence that says what the
   next one says, a caption written at three times the length it needs. **Do not compress a claim,
   drop a scope clause, or delete a result to save space.** Record in your log every cut you
   considered and declined, with its page cost; the compression wave starts from that list.

Two structural cuts were authorised. **One of them turned out to be already done, so exactly one
remains:**

- ~~**Algorithm 1 (the `StringToGraph` listing) moves to supplementary S7.**~~ 🔴 **VOID, corrected
  2026-08-30.** The S2G listing already lives only in `supplementary/alg_s2g.tex`, `\input` by
  `s09_two_directions.tex:65`, and `03_method.tex:137` already points at `\supp{9}`. The body's
  only remaining listing is the greedy G2S algorithm, which **stays, without exception**. Note the
  destination was S9, never S7: `s07_algorithms_complexity.tex:25` warns against putting an
  algorithm float in S7, because §3.3.2 names the pruned listing "Algorithm S4" and that numbering
  holds only while S9 carries all four in order. **Nothing moves. The supplement is untouched.**
- **§4.2's graph-edit-distance machinery moves to S1/S2** — bake-off detail, the determinism
  finding, the `BP_BEAM_DET` sensitivity arm. The body keeps the cost model and the two bound
  choices with a one-clause justification each and a `\supp{}` pointer. S1 and S2 already contain
  the destination text, so this is **delete-and-point, not write-new**.

🔴 **Theorem 2.12's proof stays exactly where it is: a five-line sketch in §3.3.3, the full proof in
S11.** Corrected 2026-08-30. An earlier draft of this contract said "stays in the body", written on
the mistaken belief that the body held it. The user was asked whether to *move* the proof to the
supplement and declined the move; the proof was already there, so declining the move means **leave
it alone**. It is not a request to bring 0.803 p back into a manuscript that is 6–9 p over a hard
limit, and the same answer named page count as the binding concern. **Do not restore it to the
body.** What does stand: verify the theorem *statement* carries its fixed-directedness-class scope,
with the flag hypothesis in the statement rather than only in the proof.

---

## 2. Paths

| What | Where |
|---|---|
| Article sources (your worktree) | `<YOUR_WORKTREE>/review1/article/` |
| The plan | `/home/mpascual/research/code/IsalGraph/.claude/notes/review/plan/` — start at `prose.md` |
| **Not claimable** | `.claude/notes/review/tasks/T-06-article-notes.md` §10 |
| **Complexity red lines** | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-13-complexity/T-13-FRAMING.md` §7 |
| **T-06 results, both references** | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute/` — `README.md`, then `ged/REPORT.md` and `wl/REPORT.md` (**§0 of the WL report is a correction; read it first**) |
| Staged T-28 artifacts | `/home/mpascual/research/code/IsalGraph/docs/worklogs/T-28-artifacts/` (captions live here) |
| Supplementary sources | `<YOUR_WORKTREE>/review1/supplementary/` — **read-only this wave.** Material you move out of the body goes on your log's "→ supplement" list; the orchestrator lands it |

Build: `cd <YOUR_WORKTREE>/review1/article && make` · page count: `make pages` · warnings: `make warn`.

---

## 3. FROZEN — how the two distance references are framed

This is the user's own directive and it is the narrative spine of the revision. Every section that
touches distance follows it. **Do not paraphrase the logic into a different order.**

**Two references are reported, and they are differently scoped.**

- **Graph edit distance is the field's standard proxy for structural distance between two graphs.**
  It is what H3 names, it is what the literature uses, and that is why it is reported. It is not
  reported because it flatters us.
- **The Weisfeiler–Lehman subtree kernel distance is a second reference computed from the graphs'
  own structural features** — unnormalised colour-count multisets at `h = 2`,
  `sqrt(K(G,G) + K(H,H) - 2K(G,H))`, fitted once per dataset. It is exact at every size, so it
  needs no bracket, no ceiling and no regime split.
- **The representation distances are identical under both.** The same cached matrices enter both
  halves; only `d_ref` moves. Any difference between the two is therefore a property of the
  yardstick and not of the encoding. **Say this explicitly**; it is what makes the comparison an
  experiment rather than a second opinion.

**Under graph edit distance the result is negative, and the negative is stated first.**

`H3 is refuted against graph edit distance.` In those words, with the numbers intact. Then, and
only then, the reason: on these cohorts the reference is itself size-dominated — node-count
difference alone attains ρ = 0.71–0.997 against ground-truth GED, above 0.96 on seven of ten
Suite-2 datasets. **So this experiment does not settle the structural quality of any
representation, ours included.** State that as a limitation of the measurement, not as an excuse:
the same instrument convicts every competitor, and we built it.

**Under Weisfeiler–Lehman the same distances on the same pairs behave differently, and the honest
reading is that the baseline moved, not the representation.**

Mandatory, in the same paragraph: the arm's correlation **does not rise** — on three of the five
Suite-1 datasets it falls — and the reference's own size null drops from a median of 0.914 to
0.516. Without that sentence the paragraph reads as a rescue. **It is not a rescue of H3**: H3
names graph edit distance, and against graph edit distance the refutation stands.

**Then the instruction spaces.** Both distances are edit distances over codes, so each pair of
graphs induces a path of intermediate codes, and the question is how many of those intermediates
are codes of a graph at all. This is where IsalGraph and the minimum DFS code — our main
competitor — differ in kind rather than in degree, and it is measured, not argued. It is also the
property the correlation presupposes: a Levenshtein distance between codes is a distance between
graphs only if the codes between them denote graphs. Report it **beside** the correlation, never
in place of it.

---

## 4. The claim register — use these wordings

Frozen. They were measured, argued and in several cases retracted before reaching this form.

- **C14** — *"Holding every representation distance fixed and replacing only the reference, the
  canonical string's correlation exceeds its own node-count baseline on five of the five Suite-1
  datasets against a Weisfeiler–Lehman kernel, where it does so on one of five against exact graph
  edit distance. Its correlation does not rise — on three of the five it falls — and the
  reference's own size null drops from a median of 0.914 to **0.428 across those five datasets**.
  What changes is the baseline, not the representation."* **The last sentence is not optional.**

  🔴 **The frozen wording said "0.914 to 0.516" and its two medians have different denominators.**
  Corrected 2026-08-30, recomputed from `wl/data/t28_bootstrap_verdicts.json` and
  `t28_probe_point_estimates.json` (null per cell = arm ρ − excess, `isalgraph_pruned`,
  `all_pairs`). Four true values are in circulation for one quantity:

  | size null ρ(\|n_i − n_j\|, d_ref) | denominator | value |
  |---|---|---|
  | exact GED | the 5 Suite-1 datasets | **0.9139** |
  | WL kernel | **the same 5 datasets** | **0.4283** |
  | WL kernel | all 14 cells | 0.5160 |
  | WL kernel | 11 distinct datasets (`wl/REPORT.md` §3.1) | 0.5700 |

  Per-dataset Suite-1 nulls under WL: `aids` 0.2034 · `iam_letter_high` 0.4283 · `iam_letter_low`
  0.5696 · `iam_letter_med` 0.5160 · `linux` 0.1666. C14's first half is explicitly Suite-1, so the
  matched pairing is **0.914 → 0.428**. It is also the larger drop, so the correction favours
  neither side; it is simply the like-for-like comparison. If the 14-cell figure is needed for a
  wider claim, **name both denominators in the sentence.** Never write "0.914 to 0.516" unqualified.
- **C15** — *"Under that reference the canonical string outranks both canonically-labelled `nauty`
  serialisations on every one of the fourteen dataset cells measured, and within equal node counts
  **at n ≤ 20** (58 strata higher against 31 lower, sign test p = 0.0055 against each). **Above
  n = 20 the advantage does not hold**: it ties `nauty-graph6` (53 against 57, p = 0.78) and is
  outranked by `nauty-sparse6` (40 against 70, p = 0.0054). It is outranked by the gSpan minimum
  DFS code under every reference we tested."*
- **C16** — *"Under exact graph edit distance, no representation tested clears the node-count
  baseline on more than two of the five Suite-1 datasets, and under the proven lower bound none
  clears it on any of the nine Suite-2 datasets. Under the upper bound most clear on most. Which
  representation appears to track graph edit distance is therefore a property of which bound is
  read, not of the representation."*
- **C17** — 🔴 **DEFECTIVE AS FROZEN, corrected 2026-08-30.** The frozen wording claimed the three
  canonical codes clear their baseline on *"twelve of fourteen ... IsalGraph, the minimum DFS code
  and AGM CAM alike"* while the serialisations clear it on *"one and zero"*, with mean excess
  *"+0.148 against +0.125"*. **Neither `sec5-results` nor I can reproduce those counts**, and
  `"+0.148"` and `"one and zero"` appear nowhere in the primary data — only in `prose.md` and in
  this file. Recomputed twice independently (null per cell = arm ρ − its excess, from
  `t28_bootstrap_verdicts.json`; ρ from `t28_probe_point_estimates.json`; reference `wl`, view
  `all_pairs`, 14 cells):

  | arm | cells with positive excess | mean excess |
  |---|---:|---:|
  | `isalgraph_pruned` | 12 / 14 | **+0.1250** |
  | `min_dfs` | 12 / 14 — **the same twelve** | **+0.1494** |
  | `nauty_graph6` | 5 / 14 | −0.0406 |
  | `sparse6_nauty` | 2 / 14 | −0.0928 |
  | `agm_cam` | **disputed: 9/14 (−0.0432) or 14/14 (+0.0609)** | see below |

  Both canonical codes fail on the same two cells, `2/coil_del` and `2/protein`.

  **AGM CAM gets a clause, not a count.** The two computations differ because it refuses above
  `n = 12` by its own scope guard and completes on 6.15 % of Protein, so its ρ is conditioned on
  the graphs symmetric enough to finish; pairing it with an all-pairs null and pairing it with a
  restricted null give different, individually defensible answers. **State its scope limitation
  instead of a number.**

  **Write the family split on the four unambiguous arms**, keeping C17's structure, its conclusion
  and its non-distinctiveness clause: **IsalGraph is not distinctive here** — min-DFS clears the
  same twelve cells with a *larger* mean excess. The signed mean excess separates the two families
  more cleanly than the counts do.

- **C15** — ✅ **VERIFIED 2026-08-30 against `wl/data/t28_signtest_equal_n.json`. Write it as
  frozen.** `nauty_graph6` and `sparse6_nauty` at `n ≤ 20` are both 58 higher / 31 lower,
  p = 0.005545 (hence *"against each"*); above `n = 20`, `nauty_graph6` 53 / 57 at p = 0.775 and
  `sparse6_nauty` 40 / 70 at p = 0.005447. Under **exact GED** at `n ≤ 20` the arm leads both nauty
  arms 15 / 5 at p = 0.041, which confirms the red line that the nauty split is **`n`, not the
  reference**. One caveat: C15's closing *"outranked by the minimum DFS code under every reference"*
  is unambiguous on the dataset-level bootstrap (3 W / 2 T / 9 L), but the per-stratum sign test
  above `n = 20` is 42 / 59 at p = 0.11 — losing on the point estimate, not significant. Keep that
  sentence on the bootstrap, or scope it.
- **C3, C4, C5, C6, C8, C9** — already in the manuscript; see `prose.md` §5. Do not restate them
  more favourably than they came out.

Supporting counts, from `wl/REPORT.md`: paired bootstrap over **14 cells** (`suite2/mutagenicity`
timed out at 10 h and was not rerun; four of the fourteen are Suite-1/Suite-2 duplicates, so the
distinct-dataset count is **10** — say fourteen *cells*, never fourteen *datasets*). Verdicts:
vs `sparse6_nauty` 14 W / 0 T / 0 L · vs `nauty_graph6` 12 W / 0 T / 2 L · vs `agm_cam` 8 W / 4 T /
2 L on `all_pairs` and 2 W / 5 T / 7 L under `equal_n` · **vs `min_dfs` 3 W / 2 T / 9 L**.

Edit-path property (already drafted in §3.4, §5.4 and §6.2 at the base commit; verify, do not
re-derive): 92.03 % of the canonical string's intermediates admissible, 95 % CI [91.91, 92.16],
532,315 intermediates, against 52.26 % for the minimum DFS code, [52.00, 52.54], 246,220. Whole
paths clean on 80.53 % of draws against 38.47 %. Same 23,916 graph pairs, same metric, five
uniformly-drawn optimal alignments per pair, cluster bootstrap over pairs with 2,000 resamples.

---

## 5. Red lines — each is technically defensible and would still be wrong

A reviewer who checks finds every one of these. Authoritative lists: `prose.md` §5,
`T-06-article-notes.md` §10, `T-13-FRAMING.md` §7.

| Never write | Why |
|---|---|
| *"IsalGraph approximates graph edit distance after all"*, or any WL result offered as repairing H3 | H3 names graph edit distance. The WL measurement changes what the failure is attributable to, not whether it happened — and the ρ column **falls** on three of five |
| *"clears the size baseline on 5 of 5 datasets"*, unscoped | C14 is **Suite 1**, against the **WL kernel**. Over all fourteen cells it is 12 of 14. Both scopes, same sentence |
| *"clears the size baseline on 5 of 5 Suite-2 datasets"* | True under UB, false under LB — inverts on 7 of 10 |
| *"IsalGraph beats its competitors under the WL kernel"* | **min-DFS is not beaten under any of the eight references, in either band.** It is the competitor named as most important |
| *"it outranks both nauty serialisations under the WL kernel"* without `n ≤ 20` | The advantage dies above n = 20: a tie, and a significant loss to `nauty-sparse6` (40/70, p = 0.0054) |
| *"the nauty separation is visible under WL and not under GED"* | Measured and false. The split is `n`, not the reference |
| *"IsalGraph outranks min-DFS under the upper bound within equal n"* | The one cell of 64 where it leads, and its own lower bound reverses it on 6 of those 9 |
| A win claimed from the spectral λ-distance family | All four variants lose to min-DFS and clear the size null on 0, 2, 0 and 0 of 14. `spectral_esd` is the least size-dominated reference of the eight and the encoding tracks it **worst**. Report that — it is the evidence that the WL result is not reference-shopping |
| *"ρ ≈ 0.93 demonstrates structural fidelity"* | Mostly the size channel. This paper supplies the instrument that refutes it |
| *"most compact"* unqualified; *"most compact among representations admitting a metric"* | The scope is **canonical codes**, above `n ≈ 20`. The second is false in 0 of 122 strata |
| *"no existing method satisfies all four properties"*, softened or not | False against our own comparison table: both `nauty` serialisations carry every tabulated property and IsalGraph does not, because it rejects disconnected input |
| *"competitive with the best representations"* on distance | Best on **none** of 25 records |
| *"the exhaustive arm closes the gap to nauty-sparse6"* | Measured: it does not. 342.4 vs 336.0 at n = 40. Claimable at n = 20 |
| Any β₁ without β_size beside it; any coefficient from `aids_iam` or `coil_del`; `mutagenicity`'s β_lev | Unidentifiable or retracted |
| `43 s/graph`, `≈ 520×`, `≥ 6.8 core-hours` | Unprovenanced. **Retracted** |
| Any pre-registered result restated more favourably than it came out | Forfeits the protection for all of them |

**Two arithmetic rules that generate most of this project's defects.**

- **A ratio ships with its denominator in the same sentence.** Several figures here have two true
  values describing different comparisons. Where two are in circulation, print both and say which
  is which.
- **A second confirmation must come from a second computation.** If a figure reproduces a
  published one exactly, it shares a pipeline with it: the word is *reproduces*, not
  *corroborates*, and no independence may be claimed.

---

## 6. Style contract

Formal, concise, scientific. Academic *we*, active voice, short declarative sentences with the
number in them. **Invoke `/humanizer` over every passage you rewrite that runs past ~200 words**,
after the numbers are verified, so a rewrite cannot move one.

> ### 🔴 §1 carries NO verdicts and NO headline numbers — author instruction, 2026-08-27
>
> Recorded in `01_introduction.tex`'s own header (author comments 2 and 3): every verdict, every
> headline number and every conclusion was moved **out of the introduction**, and the header
> itemises where each one now lives. Each H-item states only what is hypothesised and names its
> answering section by `\ref`, never by a literal number. **Do not put a verdict back**, in the
> list or in a paragraph beside it.
>
> The one requirement that survives: §1 must not *imply* all four hypotheses are sustained. A
> neutral clause in the sentence introducing the list — *"Section 5 answers each, in the affirmative
> or otherwise"* — is enough. Understatement in an introduction carries no risk; the verdicts live
> in §5, §6, §7 and the abstract.
>
> **`prose.md` §1's thesis paragraph is number-dense and does not belong in §1. It belongs in the
> abstract.**

> ### 🔴 C12's frozen wording is amended — decided 2026-08-30
>
> C12 ends *"Re-implementing it is a project rather than a revision, and we state it as future
> work."* That names the review round in printed text. **Amended to: *"Re-implementing it is a
> project in its own right, and we state it as future work."*** The claim is unchanged —
> individualisation–refinement removes the dependence, and re-implementing it is out of scope.
> Lives in `07_conclusions.tex:130`; owned by wave C.

**Never appears in the printed text**, and three instances exist at the base commit which the
owning agent removes:

> reviewer · referee · *this revision* · *as requested* · *we were asked* · the review · feedback

Known instances: `04_experimental_design.tex:248` (*"the direction this revision is asked about"*),
`04_experimental_design.tex:449` (*"which is why this revision recomputes…"*),
`07_conclusions.tex:130` (*"a project rather than a revision"*). Say what the paper does, in the
present tense, as if it had always said it. Reviewer demands are discharged **substantively** and
never named.

**Do not write:**

- *"It is not X, it is Y"*, *"not merely A but B"*, *"this is not a caveat, it is a finding"*.
  State Y.
- Rule-of-three lists used for rhythm rather than content.
- Significance inflation: *groundbreaking*, *novel*, *pivotal*, *crucially*, *remarkably*,
  *importantly*, *notably*.
- Synonym cycling. The canonical string is the canonical string in every sentence.
- Vague attribution: *studies show*, *it is widely known*.
- Nominalisation: *"we perform an evaluation of"* → *"we evaluate"*.
- Copula avoidance: *leverages*, *utilises*, *facilitates* → *uses*, *reduces*, *extends*.
- Em-dash asides stacked two or three to a paragraph.
- Hedges carrying no information: *arguably*, *to some extent*, *it could be argued*.
- Participial-clause pile-ups and paragraph-opening *"By doing X, we…"* frames.
- A closing sentence that restates the paragraph it ends.

**Do write:** *"Node-count difference alone attains ρ = 0.71–0.997 against ground-truth graph edit
distance."* Quantify rather than characterise. **A scoped claim carries its scope in the same
sentence** — *"most compact of the canonical codes"* is fair; *"most compact"*, with the qualifier
deferred to a limitations section, is not, and the difference is exactly what gets checked.

Vary sentence length deliberately: the current draft's failure mode is a run of uniform 22-to-28
word sentences, each carrying one number.

---

## 7. Numbers

**Check every number against its source file before you type it, not after.** Five results in T-06
were retracted after being promoted, some twice.

🔴 **A registry-key trap that misattributes a number sixfold.** In
`benchmarks/real_data/eval_t06_figures/design.py` the key `isalgraph_exhaustive` is the **hybrid**
— exhaustive canonical with pruned fallback — and fits α = 4.71. The **true** exhaustive form is
the key `isalgraph_canonical`, `max_n = 12`, and fits α = 17.43. Name the mathematical form you
mean and check which key carries it.

🔴 **`isalgraph_pruned` is the primary arm** and is never shorter than the exhaustive canonical
string, so every compactness figure is a **conservative bound**. Say so; it is a stronger sentence
than the raw number.

---

## 8. Interfaces — frozen, shared, do not change unilaterally

**Labels are the interface between sections.** Renaming one breaks a peer's `\ref`.

- **Never rename or delete an existing `\label{}`.** If a float moves, its label moves with it.
- Existing labels in play: `sec:introduction` `sec:related` `sec:canon-lit` `sec:comparison`
  `sec:prior-work` `sec:method` `sec:instructions` `sec:state` `sec:alphabet` `sec:pairs`
  `sec:greedy` `sec:cost-model` `sec:canonicalization` `sec:exhaustive-canonical` `sec:pruning`
  `sec:invariance` `sec:topology` `sec:design` `sec:datasets` `sec:reference-ged`
  `sec:representations` `sec:bits` `sec:stats` `sec:implementation` `sec:results`
  `sec:res-completeness` `sec:res-compactness` `sec:res-cost` `sec:res-fidelity` `sec:discussion`
  `sec:tradeoff` `sec:method-contrib` `sec:limits` `sec:conclusion` ·
  `tab:representation-properties` `tab:instructions` `tab:datasets` `tab:representation-headtohead` ·
  `fig:worked-example` `fig:information-content` `fig:cost-law` `fig:edit-path` `fig:rho-vs-size` ·
  `alg:g2s` · `thm:invariant` `cor:metric` `lem:one-sided` `def:levenshtein` `def:ged`
  `def:admissible` · `eq:size-null` `eq:mrm`
- **Float label assignments for the new artifacts** (owner: the §5 agent; peers cite these and must
  not define them):
  - `fig:information-content` → now renders `fig4_information_content_edits.pdf`
  - `fig:rho-vs-size` → now renders `fig_rho_vs_size_wl_vs_ged_emphasis.pdf` (two panels: (a) WL,
    (b) GED). **The label does not change**, so every existing `\ref` keeps working.
  - `tab:representation-headtohead` → `tab_representation_headtohead.tex`, currently present in the
    article directory and `\input` by nothing
- Macros available: `\supp{n}` → "Section Sn of the supplementary material" · `\suppshort{n}` →
  "Sn" · `\pending{...}` renders red and **must not survive** · `\Sig` `\IsalGraph` `\Aut` `\wstar`
- **No `changes`-package markup (`\added`, `\deleted`, `\replaced`) in body files this wave.** The
  blue version declares wholesale-new sections once at the top instead.

🔴 **Two defects in the staged artifacts, for whoever wires them in.** The caption shipped with
`fig4_information_content_edits` references `Table~\ref{tab:representation-summary}` — **that label
exists nowhere in the manuscript** and will render as `??`. A shorter alternative caption is at
`docs/worklogs/T-28-artifacts/fig_ic.caption.tex` (≈ one third the length) and does not carry the
bad reference. The caption shipped with the WL-vs-GED figure runs ~700 words, about 0.8 p on its
own; it is a reference document, not a caption, and must be cut down to what a reader needs to read
the axes.

---

## 9. Working protocol

1. Work **only** inside your own worktree, on your own branch. Never touch another agent's files;
   everything outside your ownership set is read-only.
2. Build and measure with `make` / `make pages` / `make warn` in
   `<YOUR_WORKTREE>/review1/article/`. **Measure page spans; never convert from a words-per-page
   constant** — measured density runs 249 w/p in §2 against 321 in §1 of the same document.
   To measure your own section, read the start pages of your section and the next from `main.aux`
   (`grep -oE '\\newlabel\{sec:[a-z-]+\}\{\{[^}]*\}\{[0-9]+\}' main.aux`).
3. **`make` must exit 0 and `make warn` must show no undefined reference or citation** before you
   finish. A section that does not build is not delivered.
4. **Commit everything before you finish.** Uncommitted work does not exist to a merge.
5. **Write your log to `.claude/notes/2026-08-30-article-wl/<your-slug>.md` inside your worktree and
   commit it.** Required contents: what you changed and why, file-by-file · every number you typed
   with the file you checked it against · every claim-register wording you used · the measured page
   span of your section before and after · **the declined-cut list with page costs**, which is the
   compression wave's input · anything you could not verify · anything you assumed.
6. You cannot ask the user anything. `SendMessage` the orchestrator (`main`) and continue on a
   recorded assumption rather than blocking.
7. Do not push, rebase, merge, or run `git worktree` commands. Integration is the orchestrator's.
