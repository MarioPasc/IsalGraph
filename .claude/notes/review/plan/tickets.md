# Ticket board

**A brief index, not a specification.** Each row names the ticket in one line and lists the files an
agent must read before starting it. The *content* lives in those files — do not duplicate it here.

**Board: 92.1 days upper / 53.9 lower. Critical path: 27.0 days serial against 19.**
See [schedule](schedule.md). **T-25 closed and T-23 rescoped 2026-08-12. T-03 CLOSED 2026-08-13 —
the long pole is off the critical path. T-02 CLOSED 2026-08-13. T-27 opened.**

> ⚠ **T-01 changed the cohort size. The pair count is 21,710,892, not 40,024,242.**
> COIL-DEL contributes **3,900** graphs — the IAM split index's own definition, 100 classes × 39 —
> not the 7,200 `.gxl` files that ship beside it, of which 3,300 carry no class label
> (decision 27). Suite 2 is **16,370 graphs / 21,710,892 pairs**, so the extension is **3.1× graphs
> and 5.6× pairs**, not 3.7× and 10.3×. **`n_max = 98`, `n̄ = 31.68` and the density span 0.094–0.607
> are unchanged, so AE.1's evidence is untouched.** Nine of ten rows and all three discard ratios
> reproduced exactly, and Suite 1 reproduces `export_graphs.py` to the pair.
> Inherits: **T-05, T-06, T-13, T-18, T-20**. Record: [data](data.md) §1.3 and §7.
>
> Two further findings: **the size-biased discard is cohort-wide** — Letter discards at 1.23–1.32×
> and is 84 % of Suite 1 — and **LINUX carries no node or edge attribute at all**, which settles E6
> by measurement and hands T-18 its Tier-0 label column.

> ⚠ **T-02 found that neither bracket end was selected by measurement.** `IPFP` has **never been
> measured against exact GED** — [approx_ged](approx_ged.md) §2 says so in its own words —
> and `BRANCH_FAST` rests on **400 LINUX pairs at n̄ = 8.71**, licensing a regime to `n = 98`.
> ~~**T-27** runs the full proven-method grid against T-03's 3,897,911 certified exact values for
> ≈ 5 core-hours. **T-27 gates T-05.** Until it closes, both are *defaults*, not selections, and
> ρ(exact, LB) = 0.859 / ρ(exact, UB) = 0.522 may be quoted **only with "on 400 LINUX pairs"**~~
>
> ## ✅ T-27 CLOSED 2026-08-13 — T-05 is unblocked. Two warnings it must carry.
>
> **`BRANCH_FAST` / `BIPARTITE`, selected by measurement.** But the upper end is a *constraint*
> outcome — `BIPARTITE` is the **loosest** of seven upper bounds and wins only because the frozen
> M7 gate excludes every tighter one. **PI decision: frozen gate primary, tighter methods as a
> disclosed sensitivity arm.** Two consequences that are not optional for **T-05 and T-06**:
>
> 1. **D13 fires on 2 of 5 Suite-1 datasets under `BIPARTITE`**, removing rows from the F2 family
>    wherever it fires. `BP_BEAM_DET` would fire on none. Budget for a smaller `N_actual`.
> 2. **`BIPARTITE`'s relative error grows ~10× faster in `n`** than any alternative. The size-scaling
>    argument AE.1 attacks is carried by the bound that degrades fastest with size, so
>    `(UB − LB)/UB` versus `n` (approx_ged §3.1 item 3) is now **T-05's most important measurement**,
>    not a nice-to-have.
>
> **Also**: a method name without its options string is no longer a valid specification —
> GEDLIB's upper bounds vary on 91–94 % of pairs at library defaults.
> attached. Inherits: **T-05, T-06, T-20**.

> ⚠ **T-03 invalidated a premise that T-05, T-06 and T-22 all read.** GraphEdX's published GED uses
> **unit node costs**, not the zero node cost asserted in [gedlib](gedlib.md) §6 and
> [statistics](statistics.md) D6. Measured 4/4 unit, 0/4 zero. Anything derived from "the submission
> mixes IAM unit costs with GraphEdX topology-only costs" needs re-checking before it is printed.
> D6's *metric* argument is unaffected. Full record: `.claude/notes/review/tasks/T-03-design.md`
> amendment 4.

> ⚠ **T-07 found that `docs/references/` does not exist.** `corrections.md` §4 and `.claude/CLAUDE.md`
> both point at `docs/references/2512_10429v2.pdf` for reference **[28]**. The directory was deleted in
> `7d18f52` *"Initialize github pages site structure"*, along with `docs/references/Idea.pdf`.
> **Any ticket told to read [28] will find nothing there.** Recover with
> `git show a23acbf:docs/references/2512_10429v2.pdf > <dest>`, or use the archived copy in the T-07
> report's `sources/`. Both files corrected 2026-08-26.
>
> Favourable counterpart: **[29] is CC BY open access** (PMC12344769, DOI `10.1021/acs.jcim.5c00354`),
> not abstract-only as decision 9 implied. **But the PMC/BioC full text strips all three algorithm
> listings, every equation and every table body** — anything depending on [29]'s pseudocode needs the
> publisher PDF. Inherits: **T-08**.

> ⚠ **T-04 CLOSED 2026-08-15 and corrected five plan files. Read this before quoting any ρ.**
>
> 1. **`competitors/README` §4.1 and §4.2 are SUPERSEDED.** They are a composite of **three
>    different 200-graph draws** — most rows from `real_size_null.py`, AGM from `real_suite1.py`, WL
>    from `real_wl.py` — and differ from the folder's own `real_suite1.out` by up to **0.074**. Quote
>    `.claude/notes/2026-08-14-t04-competitors/corrected_rho_table.json` instead: one script, one
>    draw, one convention, eleven rows, both views. **Inherits: T-04a, T-06, T-17, T-20.**
> 2. **IsalGraph clears the size null on ONE of five datasets, not two.** Letter MED's `+0.007`
>    becomes `−0.044` on a different draw — a margin an order of magnitude below the between-draw
>    variability finding 14 already records. **The "two of five" claim was never robust to the draw.**
>    **Graph-level bootstrap CIs are now a precondition for printing any ρ.** Inherits: **T-02**,
>    T-06, T-20.
> 3. **README §4.2's "min-DFS wins every column" does not survive.** On the single-draw equal-`n`
>    table `isalgraph_canonical` takes Letter LOW (0.9987 vs 0.9956) and WL takes AIDS (0.4332 vs
>    0.3993). Inherits: T-17, T-20.
> 4. **finding 12 is refuted: grakel has no off-by-one.** `n_iter = k ≡ h = k`. The error was in our
>    own `scratch/backends.py::wl_features`. `wl_kernel_computer.py`'s `n_iter = 5` is **`h = 5`**,
>    not `h = 4`, and §4.1's WL row moves (Letter LOW 0.895 → 0.7792). Inherits: **T-06**.
> 5. **`nauty.md` §1 and README §6 trap 2 are wrong on both halves.** An inverted `canon_label`
>    **fails F3 loudly**, and `nx.is_isomorphic` **can never catch it** — any bijective relabelling is
>    isomorphic by construction. The trap is loud, and the prescribed guard is vacuous.
> 6. **`competitors/README` §4.3's sparse6 column is 6 bits high on every row** (the `':'`), and its
>    **five Suite-2 rows are not exactly reproducible** — the raw IAM GXL tree is gone from this
>    workstation and the recovered `.npz` enumerates differently. The five **Suite-1** rows are
>    full-cohort and remain exact.
> 7. **`graph6.md` §7's `convert_node_labels_to_integers(ordering="sorted")` does not pin the
>    labelling** — 290/300 disagreement with a sorted rebuild, and it made graph6 and sparse6
>    serialise *different* labellings.

> ## ⚠ T-05 closed 2026-08-15 and invalidated a measure this plan named. **T-06, T-18 and T-20
> inherit corrections.**
>
> 1. **`(UB − LB)/UB` alone inverts the AE.1 conclusion.** It is a ratio whose denominator grows with
>    `n`. Measured on all 21,710,892 pairs: the **absolute gap rises in 10/10 datasets** while the
>    relative width falls in 6/10 — **opposite signs in 4 of 10**. `approx_ged.md` §3.1 item 3 called
>    it "the single measurement that answers AE.1 most directly"; it is corrected in place. **Report
>    the absolute gap first.** **Inherits: T-06, T-20.**
> 2. **Class counts in `decisions.md` are RAW counts and two are false of the filtered cohort** —
>    Letter LOW retains **9 of 15**, GREC **17 of 22**, and **LINUX and AIDS-GraphEdX carry no class
>    label at all**. Any "Letter, 15 classes" sentence is false of what we analysed, and a
>    classification arm cannot run on those two. **Inherits: T-18, T-06.**
> 3. **`decisions.md` §6's 33.2 % orientation figure names a rate belonging to graphs ~3× larger than
>    those it was measured on**, and the two upper bounds move in **opposite** directions in `n`
>    (`BIPARTITE` 22.8 %→11.2 %, `IPFP_MS` 3.7 %→59.5 %). Any restatement must name method *and* size
>    range. **Inherits: T-20.**
> 4. **T-03's `ub_matrix` is run-dependent** (74–82 % of values change between runs), bounded to
>    exactly the **61,084 D11 censored interval upper ends**. Accepted unrepaired by PI decision, so
>    **stating it is obligatory** — otherwise re-running the reproduction script gives different
>    numbers, the R3.5a failure mode. **Inherits: T-20.**
> 5. **D14's "a few graphs" understates canonicalisation censoring at Suite-2 sizes** (5/10 random
>    `protein` and `coil_del` graphs exceed 15 s), and **its 300 s timeout cannot be enforced with a
>    Python signal** — `SIGALRM` does not interrupt the C++ engine; use a killed subprocess.
>    **Inherits: T-06.**
> 6. **`cohort_audit.py` can no longer re-derive the LINUX and AIDS-GraphEdX rows** on the current
>    tree — no single `--source` resolves both roots. The counts are confirmed by a *different*
>    program. **Inherits: T-06.**
>
> Full detail: [T-05 article notes](../tasks/T-05-article-notes.md).

> ⚠ **T-04 was scouted 2026-08-13 on the REAL cohort and several of its premises are wrong.**
> Every competitor was installed and run against Suite 1's **certified exact GED** (T-03) and
> Suite 2's IAM GXL. Evidence: **one file per competitor in
> [`competitors/`](competitors/README.md)**, whose §5 lists fifteen findings with owners. The ones
> that reach a printed number:
>
> 1. **The size null is unowned and it dominates.** `ρ(|n₁−n₂|, exact GED)` — count the nodes,
>    subtract, no representation at all — scores **0.899 / 0.909 / 0.926 / 0.713 / 0.799** on the
>    five Suite-1 datasets. **IsalGraph clears it on two of five, by ≤ 0.03**, and falls 0.24–0.54
>    below it on the other three. The manuscript's "ρ ≈ 0.93 on sparse IAM" reproduces (0.925) but
>    sits **0.026 above a baseline that needs no method**. Every printed ρ needs the null beside it,
>    and the **equal-`n` restriction** should be primary — there the canonical/non-canonical gap is
>    **0.42** and the claim is defensible. **Inherits: T-02, T-06, T-20.**
> 2. **gSpan's minimum DFS code beats IsalGraph on ρ on all five Suite-1 datasets**, by +0.047 to
>    +0.296, in both the all-pairs and equal-`n` views. AGM beats it on 3 of 4; **WL beats it on
>    LINUX and AIDS**. IsalGraph wins Claim A against min-DFS on 60–100 % of real graphs.
>    **Both halves must be stated. Inherits: T-17, T-20.**
> 3. **IsalGraph is shorter than the adjacency matrix on 0.0 % of Letter graphs** and never wins
>    Claim A on Suite 1. It wins on the **mean** at AIDS-IAM (85.3 vs 135.9 bits) and loses on the
>    **median** (60.2 vs 55.0). Print both. **Inherits: T-20.**
> 4. **AGM is not computable on Suite 2** — 100 % exact on Letter and LINUX, 99.6 % on Suite-1 AIDS,
>    **76 % on GREC**, **82 % on AIDS-IAM**. AGM runs on **Suite 1 only**, and
>    [preregistration](preregistration.md) §5's reduction rule has **no case** for a representation
>    computable on one suite and not the other. **Inherits: T-02's `N_max = 182`, T-17.**
> 5. **`canonical_string` breaks on Suite 2** — 342 ms/graph and 12/400 timeouts on AIDS-IAM against
>    `pruned`'s 18 ms and zero. **Suite 2 must use `pruned_canonical_string`. Inherits: T-06.**
>
> Also: **three min-DFS repositories tested, all three rejected** — including
> `kaviniitm/DFSCode`, which builds, claims exactly this, and is **not isomorphism-invariant**
> (46/90). Vendor nothing. **bliss/Traces stay cut** — the `pynauty` from-source build was rehearsed
> under gcc 12.2.0 and succeeded, so the insurance rationale has expired. And **ρ moved 0.07 between
> two independent 200-graph draws on AIDS**, which is direct support for [statistics](statistics.md)
> D2.

> ## ⚠ T-04a closed 2026-08-23. **`k = 3`, and three plan files named the wrong primary distance.
> T-06, T-17 and T-20 inherit corrections.**
>
> 1. **`padded_hamming` is primary for NOTHING.** [competitors/README](competitors/README.md) §3's
>    provisional column, [nauty](competitors/nauty.md) §3 and [agm](competitors/agm.md) §3 all named
>    it for `nauty_graph6` / `agm_cam`. Measured on the frozen draw it loses the F6 tie-break to
>    `levenshtein` by **68×** (0.0010 vs 0.0704 ms/pair) and **8.6×** respectively. All three files
>    corrected in place. **`levenshtein` is primary for all six surviving serialisations**;
>    `wl_subtree` takes `kernel`. **Inherits: T-06, T-17, T-20.**
> 2. **`k = 3`** — `adjacency`, `graph6`, `sparse6`, each failing F3 at **1/50**. `preregistration.md`
>    §7's `k` is settled; **T-06 owns applying it** (`N_actual = 182 − 15k − 8d + k·d − c`, **corrected 2026-08-17** — the form `182 − 15k − 8d` omits both the `+k·d` overlap term and `c`, and **under-counts `N_actual` by `3d` at `k = 3`**, which lowers the BH burden on every surviving test. See [preregistration](preregistration.md) §5–§5.3; `N_actual` is defined by **enumeration**, the closed form is a printed check).
> 3. **[gspan-mdfsc](competitors/gspan-mdfsc.md) §3's "best in the pool on all five Suite-1 datasets"
>    does not survive.** Under the selected distances `min_dfs` is best on **2 of 5** (Letter LOW,
>    Letter MED); `agm_cam` takes the other three. Over all 15 records: `min_dfs` 4, `agm_cam` 3,
>    `wl_subtree` 3, `sparse6_nauty` 3, `nauty_graph6` 2, **IsalGraph 0**. Inherits: **T-17, T-20.**
> 4. **`competitors/README` §7's Suite-2 LB range `−0.082 to −0.295` is stale** — corrected to
>    **`−0.289`**. The size null must be restricted to *each representation's own* pair set; on
>    Mutagenicity, where IsalGraph loses 14 graphs to the canonicalisation budget and every censored
>    graph is larger than every kept one, the whole-cohort null overstates by 0.118 (UB margin
>    **+0.078 → +0.196**). **A size null computed on a different pair set than the arm is not a
>    comparison.** Inherits: **T-06, T-20.**
> 5. **finding 13 is REFUTED for the shipped backends.** The n² family's `normalised()` reads
>    `sorted(nodes)`, so `nx.relabel_nodes(copy=True)` cannot leak insertion order through them. The
>    prescribed relabeller `fixtures.shuffled_copy` stays — its *stated reason* was wrong, not the
>    prescription. Inherits: **T-06.**
>
> Also: **F0 is measured at each backend's own budget** (`timeout_s = 2.0`,
> `max_projections = 50,000`, `search_nodes = 200,000`), **not** D14's 300 s. Every printed F0 must
> name its budget, and `SuiteScopeError` (a scope decision) must never be summed with
> `AGMBudgetExceeded` or `CanonicalizationTimeoutError` (budget outcomes).

> ## ⚠ T-09 closed 2026-08-25 and found a defect in the prose R3.7c points at. **T-11, T-20, T-24 and T-26 inherit.**
>
> 1. 🔴 **Remark 2.7 (`methodology.tex:462`) excludes half of its own search space.** It states that
>    *only* the uninserted-neighbour choice at each `V`/`v` contributes; **Definition 2.6 three lines
>    above defines `w*_G` over any starting node**, and `core/canonical.py` searches both. Measured on
>    a six-node example: six start nodes, six distinct strings of lengths **9, 10, 9, 11, 10, 10**,
>    one attaining the minimum. The schematic R3.7c asked for draws one subtree per start node and so
>    **contradicts the prose it illustrates**. Recorded as **E13** in [corrections](corrections.md) §3.
>    **Owner: T-11.** One clause; the rest of the remark is correct.
> 2. **`GraphToString.run_with_trace` is a replay, not a trace.** It runs the finished string back
>    through `StringToGraph`, so anything built from it shows a *decoder*. Any future figure or
>    analysis wanting the encoder's own behaviour must use `viz/encoder_trace.py`, which is pinned to
>    the frozen `core/graph_to_string.py` by test on 134,609 `(graph, start)` pairs. **Inherits: T-20, T-21.**
> 3. 🔴 **The graphical abstract was not regenerated and cannot be until T-06's numbers land in it.**
>    `graphical_abtract.pdf` panel (b) prints `Wins: 99.6 %`, `β = 0.537`, `R² = 0.947` and `14,108×` —
>    every one retired when T-06 withdrew Claim B at scale. T-09 built the panels for (a) and stopped.
>    **Inherits: T-24**, together with the misspelt filename (E12).
> 4. **[manuscript](manuscript.md) §2 under-counts this row.** It priced the worked example as one
>    0.75-page figure; it is **four independent panels plus the schematic**. **Inherits: T-26.**

**Read for every ticket**: [decisions](decisions.md) (do not re-litigate a signed decision) and
[demands](demands.md) (what the ticket is answering, and to whom).

---

## Board

| ID | Ticket | Depends | Days | Pri | **Read first** |
|---|---|---|---|---|---|
| ~~**T-01**~~ | ~~Data lock — audit tables, cohorts, merge splits, port surviving scripts into `tests/`~~ → **DONE 2026-08-13.** Re-derived, not ported: 15 of 16 scripts were gone and Suite 2 had no loader. `iam_gxl_loader.py` + `cohort_audit.py` + **34 tests**. **Suite 1 reproduces `export_graphs.py` exactly** (3,897,911 pairs). **Suite 2 = 16,370 graphs / 21,710,892 pairs / `n_max` 98** — COIL-DEL corrected 7,200 → **3,900** (decision 27). Nine of ten rows and all three discard ratios reproduced exactly. Four findings: the **size-biased discard is cohort-wide** (Letter 1.23–1.32×); **LINUX is unlabelled**, settling E6; the **density convention** matters (up to 27 %); **I-05 closed** at 1.19× | — | **done** | — | [data](data.md) §1, §7, [T-01 design](../tasks/T-01-design.md) |
| ~~**T-02**~~ | ~~Statistics lock — graph-level bootstrap, Mantel, pair-accounting ladder, and the frozen confirmatory family with its cardinality~~ → **DONE 2026-08-13.** Family enumerated and frozen at **`N_max = 197`** in three fixed-sequence families — F0 calibration gate 5, F1 bracket gate 10, F2 primary 182 — BH-FDR q = 0.05 within each; `N_actual = 182 − 15k − 8d`. **Four defects fixed in the locked protocol**: §9's exact-regime omnibus contradicted §4; two gates sat inside the family they gate; the labels row made the cardinality indeterminate; **D15 validated a 7.72 % subsample by drawing 94.4 % of a smaller dataset**. **D13 promoted to confirmatory**; ρ(Lev, UB) gets no primary rows. Raised **T-27** | T-01 | **done** | — | [preregistration](preregistration.md), [T-02 design](../tasks/T-02-design.md), [statistics](statistics.md) §12 |
| ~~**T-03**~~ | ~~Exact GED on Picasso~~ → **DONE 2026-08-13.** All five Suite-1 datasets: **3,897,911 pairs, 98.43 % certified exact, 1.57 % interval-censored, ≈ 2,081 core-h.** Both stages ran and **agree on their 22,051-pair overlap**. Three findings carried: the **exact solver changed** (`ANCHOR_AWARE_GED` is non-deterministic and non-exact), **GraphEdX uses UNIT node costs, not zero** (retracts a T-03 finding *and* contradicts [gedlib](gedlib.md) §6 / D6), and **censoring is hardware-dependent** | T-01 | **done** | — | [T-03 log](../tasks/../2026-08-12-exact-ged/summary.md), [exact_ged](exact_ged.md) §7 |
| ~~**T-04**~~ | ~~**Competitor backends** — `src/isalgraph/competitors/` in the IsalHG idiom: graph6, sparse6, nauty, AGM, **gSpan min-DFS**~~ → **DONE 2026-08-15.** **Eleven backends, 6 metrics, 383 tests**, 9,510 lines; ruff + `mypy --strict` clean; full suite **2,106 passed / 321 skipped**. **The reproduction gate closed bit-for-bit**: replaying each scout script's `Random(42)` stream reproduces `real_suite1.json` on **all 5 datasets × 8 rows at delta `0.00e+00`**. Oracles: AGM **327 graphs / 0 mismatches** vs the lex-min over all `n!`; min-DFS distinct codes **1/2/6/21/112** = OEIS A001349, **0 collisions**; budget **24/400 Mutagenicity, 0 elsewhere**. `pynauty` **builds from source under gcc 12.2.0 on Picasso**, byte-identical output (stop-condition 2 closed). **Five plan files corrected — see the header warning.** Carries: **the corrected ρ table supersedes `competitors/README` §4.1/§4.2**; **IsalGraph clears the size null on 1 of 5, not 2**; **bootstrap CIs are now a precondition for any printed ρ** | — | **done** | — | [T-04 design](../tasks/T-04-design.md), [article notes](../tasks/T-04-article-notes.md), [wave](../../2026-08-14-t04-competitors/summary.md) |
| ~~**T-04a**~~ | ~~**Metric feasibility** — every (representation × distance) cell on a fixed 200-graph sample; select each primary distance by the pre-declared rule~~ → **DONE 2026-08-23.** All **66 cells** measured on one frozen pooled draw (200 graphs, six node-count strata `[33,33,33,33,34,34]`, seed 42, `n` 2–83, suite split 51/149). **`k = 3`** over preregistration §4.1's seven-member Claim-B set: `adjacency`, `graph6`, `sparse6` have **no admissible distance**, each failing F3 at **1/50**. **`levenshtein` is primary for all six surviving serialisations, `kernel` for `wl_subtree`** — `padded_hamming` is primary for **nothing**, losing the F6 tie-break by **68×** on `nauty_graph6` and **8.6×** on `agm_cam`. Plus the **E1–E4 admissibility annex**: ψ = 0.0000 on all eleven draws for the seven canonical representations vs up to **1.148** for `sparse6`; **0 collisions** for six complete invariants (zero set ≡ VF2-certified isomorphic set) against **45 / 183,016** for WL; **0 axiom violations in 9,881,851 checks** over 467,180 triples. **Five findings carried: (1) `padded_hamming` primary column in `competitors/README` §3, `nauty.md` and `agm.md` is WRONG — corrected in place; (2) IsalGraph clears the size null on 1 of 5 Suite-1 datasets and is the best representation on 0 of 15 records (`min_dfs` 4, `agm_cam` 3, `wl_subtree` 3, `sparse6_nauty` 3, `nauty_graph6` 2) — `gspan-mdfsc.md`'s "best on all five Suite-1" corrected to 2 of 5; (3) the Suite-2 verdict FLIPS on 5/5 between LB and UB, all 15 paired differences excluding zero, because `ρ(\|Δn\|, LB) = 0.960–0.998` makes the lower bound very nearly the size null — degenerate by construction, not a defect in `BRANCH_FAST`; (4) `adjacency` beats IsalGraph significantly on 3/5 all-pairs and 0/5 equal-`n`, the F5-blindness trap, `ρ(d_adjacency, \|Δn\|) = 0.83–0.93`; (5) finding 13 REFUTED for the shipped backends** — the n² family's `normalised()` reads `sorted(nodes)`, so it is insertion-order invariant; the prescribed relabeller stays, its stated reason was wrong. **F0 is per-backend-budget, not D14's 300 s, and every printed F0 must name its budget** | ~~T-04~~ | **done** | — | [design](../tasks/T-04a-design.md), [annex protocol](../tasks/T-04a-admissibility-protocol.md), [article notes](../tasks/T-04a-article-notes.md), [letter](../tasks/T-04a-letter-fragment.md), [plan RESULT](competitors.md) §9 |
| ~~**T-05**~~ | ~~**Bounded GED via GEDLIB** — wire the bounds T-27 selects, pass the validation gates, run the calibration ladder, then all 21,710,892 Suite-2 pairs (≈ 0.57 core-h)~~ → **DONE 2026-08-15.** All **21,710,892** pairs bounded under D6, **≈ 2,140 core-h realised** (not 0.57). G1–G4 all passed and were re-verified with independent code: **0 bracket violations over all 21.7 M pairs**, **0** containment violations over 3,836,827 T-03-certified pairs, and G2 reproduces T-27 **element-wise on 10,807,845 pairs across three arms**, byte-identical. Ladder rungs 13–18 → **measured exact-GED ceiling `n = 17`** (up from 12). Certification spans **28.46 % → 0.03 %**, a factor of 949. ⚠ **The absolute gap `UB−LB` RISES with `n` in 10/10 datasets while `(UB−LB)/UB` falls in 6/10 — the measure this plan named would have inverted the AE.1 conclusion** (approx_ged §3.1 item 3, corrected). The `BP_BEAM_DET` arm fired 10/10: the frozen gate explains **63–88 %** of the widening at small `n` but only **35–51 %** at the disputed sizes. **Carries: §7.5 (`ρ(Lev,·)`) deferred in full → T-06** · **class counts false of the filtered cohort (Letter LOW 9/15, GREC 17/22, LINUX & AIDS-GraphEdX none) → T-18, T-06** · **T-03's `ub_matrix` run-dependent, accepted unrepaired → T-20 must state it** · **D14's censoring premise understated, and its 300 s timeout cannot be enforced with a Python signal → T-06** | ~~T-01~~, ~~T-03~~, ~~T-27~~ | 5–10 | ~~P0~~ **done** | [log](../tasks/T-05-design.md), [notes](../tasks/T-05-article-notes.md), [letter](../tasks/T-05-letter-fragment.md), [plan RESULT](approx_ged.md) |
| ~~**T-06**~~ | ~~**Full recompute** — all experiments, C++ engine, new cohorts, competitor columns, new statistics~~ → **DONE 2026-08-24.** Cohorts reproduce exactly (**16,370 / 21,710,892 / 5,350 / 3,897,911**). 155 encoding cells, **190 distance matrices**, 0 contract violations, reproduction gate at max \|Δ\| = **0.0000**, structural gate **190/190 joins exact**. **F0 fires 4 of 5 → majority branch: the 81 approximate-regime cells are DESCRIPTIVE. F1 gives `d = 7 of 10` (reported, not applied). `N_actual = 79`, enumeration = closed form, discrepancy 0, 79 of 79 cells with a p-value, BH rejects 75** (split **35 for / 34 against** — the count must never travel bare). 🔴 **Claim B is a clear disadvantage**: below its own `\|n_i−n_j\|` size null on **17 of 25 records, every one significantly**, including **4 of 5 Suite-1 datasets against *exact* GED** where no bracket argument applies; within-`n` ρ collapses 0.966 → 0.078. **Claim A is an advantage that GROWS with size** (20.4 % → **45.6 %** of strata; median gap −1.2 → **+242.1 bits**; **112/112 vs `min_dfs`** above n = 20) but is **net-negative pooled** — the "above n ≈ 20" scope is mandatory, and *"most compact admissible"* is **false** (true in **0 of 122** strata; `sparse6_nauty` dominates on both axes). **One clean positive: zero encoding collisions on 24,764,422 pairs.** **Findings other tickets must act on: (a)** the benchmark itself is size-dominated — `\|Δn\|` alone reaches ρ = **0.71–0.997** against ground-truth GED; **(b)** 🔴 **tier-3 MRM point estimates fall outside their own bootstrap intervals** ([statistics](statistics.md) §5) — **IsalSR and IsalHG inherit this**; **(c)** three pre-registration gaps found by executing it, all resolved conservatively ([preregistration](preregistration.md) RESULT); **(d)** `.claude/CLAUDE.md`'s suite reference state was stale by 3.5× (726 → **2,544**), corrected. **Debts:** `t06_completion` counts a censored graph as not completed (inert — §5.1 exempts the IsalGraph arm); E10's WL numbers predate the `h = 2` fix and need re-checking | T-02…T-05 | **done** | — | [log](../tasks/T-06-design.md), [framing](../tasks/T-06-FRAMING.md), [notes](../tasks/T-06-article-notes.md), [archive](/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute/) |
| ~~**T-07**~~ | ~~**Read [28] and [29]**; inherited/modified/new delta table **plus the sufficiency paragraph**; resolve D19~~ → **DONE 2026-08-26.** **Tab. 3 built: 9 rows × 4 columns, measured 0.67 p against a 0.70 p budget**, 0 overfull boxes; **sufficiency paragraph 145 words**, inside the 120–150 band. 🟢 **The headline is favourable and it is now evidence, not assumption: neither predecessor contains a single formal result** — `theorem`, `proof`, `lemma`, `proposition` each occur **0 times** in **both** [28] and [29], as do `complete invariant` and `graph edit distance`. [28]'s "canonical string" is *defined as* its greedy encoder's output on a matrix that presupposes an assumed vertex order, so it is canonical **per matrix, not per isomorphism class**; [29] argues relabelling invariance in **three sentences, one direction only, under an explicit assumption**, and measures no collision rate. **Theorem 2.12 is genuinely new.** Under an attribution rule frozen *before* the sources were read (`b300581`, generalisation counts as modification, never novelty): **3 inherited, 5 modified, 1 new** — the metric corollary folded into the theorem row rather than counted twice, so **the table understates**. **D19 RESOLVED, both halves**: [28]'s Transformer classification CONFIRMED but on a **synthetic** 3,000-sample set with **one** non-graph baseline and **no numbers printed in the text**; [29]'s is **LSTM *and* GRU** on **masked token prediction**, **not classification**. R3's record stays intact. **Findings other tickets must act on: (a)** 🔴 **T-14 now solely owns the R3.2 pre-emption** — the PI dropped the sequence-model row from Tab. 3, overruling [corrections](corrections.md) §4, so **no T-07 artifact discharges it**; the measured content is handed over in [notes](../tasks/T-07-article-notes.md) §4, and the strongest honest ground is that **neither predecessor ran a downstream graph-learning evaluation on a real benchmark**. **(b)** R3's phrase *"exhaustive shortest-then-lexicographic normalization"* is **half unverifiable** — the ordering criterion is dual-sourced, but **"exhaustive" cannot be sourced** from the CC-BY text (all three algorithm listings are stripped) and the public implementation enumerates the **starting heavy atom only**; Tab. 3 therefore prints no search-space row, though one would have favoured us. **(c)** `manuscript.md` §2's placement and 0.75 p price were stale → corrected to §2.3 / 0.67 p. **Debt:** settling (b) needs the publisher PDF; **nothing printed depends on it** | — | **done** | — | [design](../tasks/T-07-design.md), [notes](../tasks/T-07-article-notes.md), [letter](../tasks/T-07-letter-R3.1a.md), [archive](/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-07-prior-work-delta/) |
| **T-08** | **Related-work section** (§1.x) + bibliography to ≤ 55 | T-07 | 4–10 | P1 | [compliance](compliance.md), [manuscript](manuscript.md) §1 |
| ~~**T-09**~~ | ~~**Explanatory figures** — the canonical search-space schematic (**R3.7c, requested**) and the S2G/G2S worked example (**author preference, first page cut**). Both regenerate the graphical abstract~~ → **DONE 2026-08-25.** **Five figures**, `.pdf` + `.png`, on **one running example** chosen by enumerating every connected graph on 5–6 nodes: `n = 6`, `m = 6`, `\|Aut(G)\| = 1`, `w*_G = VVVnvPCPV`. The schematic answers R3.7c and draws **one subtree per start node**. The worked example became **four panels, not one** — S2G and G2S for the exhaustive *and* the pruned canonical form. **31 new tests; suite 2,583 / 321.** 🔴 **Remark 2.7 is wrong and the figure R3.7c asked for exposes it**: it says *only* the neighbour choice is searched over, but Definition 2.6 and `core/canonical.py` also search the **starting node** — six start nodes give six distinct strings (9, 10, 9, 11, 10, 10). **→ E13, owner T-11**, one clause. **Findings other tickets must act on: (a)** `GraphToString.run_with_trace` **replays** the finished string, so a naive G2S figure is the S2G figure with its mask flipped — the panels are built from a new instrumented mirror (`viz/encoder_trace.py`) pinned to the frozen encoder on **134,609 (graph, start) pairs, 0 mismatches**, `core/` untouched; **(b)** the **graphical abstract was NOT regenerated** — panel (b) still prints `99.6 %`, `β = 0.537`, `R² = 0.947`, `14,108×`, all retired by T-06 → **T-24**; **(c)** `manuscript.md` §2 priced this as **one** 0.75-p figure and it is now **four panels + the schematic** → **T-26** re-prices; **(d)** the CDLL pointer arrows drew **only their heads** (axis-unit tail vs point-space shrink) — fixed, and every figure calling `draw_cdll_ring` regenerated. **Debt:** the pruned canonical string is emitted by **no greedy run**, which is small-scale evidence that Definition 2.6's neighbour branch is load-bearing — one sentence for **T-20** if it wants it | — | **done** | — | [design](../tasks/T-09-design.md), [notes](../tasks/T-09-article-notes.md), [letter](../tasks/T-09-letter-fragment.md), [archive](/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-09-explanatory-figures/) |
| **T-11** | **Manuscript defects** — Alg. 2, Example 2.3, equivariance→invariance, **and E7's float fix, which must precede any trim** | — | 2 | P1 | [corrections](corrections.md) §3 |
| **T-12** | **Claim scoping** — B1…B6, and E5/E6 | T-06 | 2 | P1 | [corrections](corrections.md) §1 |
| **T-13** | **Complexity section** — `P(M)` recomputation, four costed operations, three-way separation, the `\|Aut(G)\|` worst case | — | 2 | P1 | [corrections](corrections.md) §5, [data](data.md) §4 |
| **T-14** | **Response letter** — assembles the fragments each ticket emits; **not written from scratch at the end** | all | 3 | **P0** | [manuscript](manuscript.md) §4, [demands](demands.md) |
| **T-15** | **Page trim to 35** + supplementary split | all | 2 | **P0** | [manuscript](manuscript.md) §3, [compliance](compliance.md) §7 |
| **T-17** | **AE.3 comparison table** as a paper artifact — properties, strengths, limitations of each, on R1.2's five axes. **Rows populated from T-04's measurements, not asserted** | T-04, T-07 | 2–3 | **P0** — the AE endorsed this in their own voice | [competitors](competitors.md), [demands](demands.md) AE.3/R1.2b |
| **T-18** | **Labels** — tiered; Tier 0 not optional, Tier 1 recommended | T-05, T-06 | 0.3–1 | **P0** (Tier 0) | [labels](labels.md) |
| **T-19** | **Bibliography recency and venue audit** — classify all 43 references; add **≥ 4 from 2025–26**, ≥ 3 at PR venues other than the PR journal, **self-citations excluded** | T-08 | 1–2 | **P0** — EiC checks independently | [compliance](compliance.md) §2–§4 |
| **T-20** | **Manuscript rewrite** — §3.1, §3.2, §3.3, §4, §5, abstract. The largest single writing task | T-06 | 5–7 | **P0** | [manuscript](manuscript.md) §1, [statistics](statistics.md), [data](data.md) |
| **T-21** | **Implementation, reproducibility and artifact release** — C++ engine and GEDLIB in §3.3; versions; the `-march` and non-rsyncing-`.so` traps; data-availability statement | T-06 | 1–2 | P1 | [compliance](compliance.md) §8, [gedlib](gedlib.md) |
| **T-22** | **Formal-statement audit** — restate Thm 2.12 within a fixed directedness class, move the flag hypothesis into the statement, **re-verify all three proof steps**, propagate to **Cor. 2.13** | — | 1–2 | **P0** | [corrections](corrections.md) §2, [statistics](statistics.md) D6 |
| ~~**T-23**~~ | ~~Clear the Picasso `fscratch` file-count quota~~ → **RESCOPED 2026-08-12, no longer blocking.** T-03 + T-05 output is **30 files** (0.0075 % of the hard limit); the pressure is the GEDLIB **build tree** (50–90k files), pruned after `build_ext`. Folded into T-05's environment setup | — | 0.1 | P2 | [gedlib](gedlib.md) §2, [exact_ged](exact_ged.md) §5.1 |
| **T-24** | **Submission package and Elsevier compliance** — source files, AI declaration, biographies, acknowledgements, highlights, graphical abstract (**fix the misspelt filename**), competing-interest and data-availability statements | T-15 | 1 | **P0** | [compliance](compliance.md) §8 |
| ~~**T-25**~~ | ~~Restore validation gate 2, or retire it on the record~~ → **CLOSED 2026-08-12 by option A.** `ged_bounds.py` written and **tracked in the repo**; gate 2 executable and **passing** (0 violations / 400 LINUX pairs); 35 unit tests. Two findings carried to T-05: the upper bound is **not symmetric**, and **the retired H4 numbers do not reproduce** | — | **done** | — | [exact_ged](exact_ged.md) §4 |
| **T-26** | **Bibliography-slot and page-budget reconciliation** — the two arithmetics the EiC checks independently and no other ticket owns end to end. **Runs after T-08 and T-19, before T-15** | T-08, T-19 | 0.5 | **P0 — EiC pass/fail** | [compliance](compliance.md) §2, [manuscript](manuscript.md) §2–§3 |
| **T-28** | **Alternative similarity references for §5.4** — §5.4 reports **0 win / 1 tie / 24 loss** against GED and the arm sits **below its own `|n_i − n_j|` size null on 17 of 25 records**, which the PI judges a rejection risk (2026-08-29). Add the **WL subtree kernel distance** and the **spectral λ-distance** as alternative REFERENCES. **The representation distances do not move**: T-04a's selections (`levenshtein` for the six serialisations, `kernel` for `wl_subtree`) are reused byte-for-byte from the T-06 cache, so no encoder runs and there is no distance stage. **Probe result (point estimates, 15 cells): under the WL kernel the arm beats `agm_cam` 15/15, `sparse6_nauty` 15/15 and `nauty_graph6` 12/15, and **against `min_dfs` the paired graph-level bootstrap is a TIE, not a loss** — the difference's 95 % interval covers zero on every cell measured so far (LINUX: −0.0235 [−0.1470, +0.0923] and −0.0413 [−0.1326, +0.0508]), where under *exact* GED the same comparator is a significant loss (−0.1691 [−0.2785, −0.0784], p = 0.002). So under the WL reference the arm is beaten by **nothing**: three significant wins and one tie. It also clears its size null on 12/15 against 8/25 under GED** — because the size null against the WL kernel runs 0.16–0.87 where against GED it reaches **0.9971**. The pre-declared primary spectral variant (normalised L, zero-padded) **fails**: 0/15 wins, 0/15 clearing the null, since `tr(L_sym) = n` makes the padded Euclidean distance a size proxy. **T-28b** (IsalChem p.7949 fallback) concluded **do not implement** — all seven fingerprint metrics are size-dominated by the same mechanism. 🔴 **Guard added**: family membership was decided by representation alone, so any new reference would have entered the pre-registered family as a `B1a` row and inflated `N_actual` past 79 with no error raised | T-04a, T-06 | **in progress** | P0 | [design](../tasks/T-28-design.md), [worklog](../../../../docs/worklogs/T-28-metric-distances.md), [fallback](../tasks/T-28b-isalchem-fallback.md) |

| ~~**T-27**~~ | ~~GED bound bake-off — select both bracket ends by measurement~~ → **DONE 2026-08-13.** **60 cells, 46,774,932 bound evaluations, 0 M4 violations, ≈ 7 core-h.** **LB = `BRANCH_FAST` (5 of 5); UB = `BIPARTITE` (5 of 5, by elimination).** Four findings carry: **`BRANCH` ≡ `BRANCH_FAST` is PROVEN under D6** (survey §5.2.4) and measured identical on all 3,836,827 certified pairs — decision 11 upheld on a theorem, not on 400 LINUX pairs; **GEDLIB's UBs vary on 91–94 % of pairs at defaults** (`RANDOM`/`REAL`), 0 % pinned, so **a method name without options is not a specification**; **`BIPARTITE` trips D13 on 2 of 5 datasets** (Letter LOW −0.219, MED −0.177) where `BP_BEAM_DET` trips none; **its error grows ~10× faster in `n`** (AIDS slope +0.294/node vs `IPFP_MS` +0.029). **`HED` resolved** — LB-only by design, usable with `--edge-set-distances OPTIMAL`, loosest in the grid, confirming `BED ≥ HED` | T-03 | **done** | — | [REPORT](/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-27-ged-bound-bakeoff/REPORT.md), [article notes](../tasks/T-27-article-notes.md), [approx_ged](approx_ged.md) §2 RESULT |

**Retired**: ~~T-01b~~ (new-dataset audit — **superseded**: its numbers are among the unverified ones,
see the header warning) · ~~T-10~~ (merged into T-09) ·
~~T-16~~ (`wl_pruned_canonical` — **rejected**, [decisions](decisions.md) §2).

---

## Dependency structure

**Critical path** — **T-27 → T-05 → T-06 → T-20 → T-15 → T-24**, with T-14 accruing throughout.
**T-01, T-02, T-03 are all done and off the path.** **T-27 is the only remaining gate on T-05**: it
decides which bound T-05 computes. It is 1–2 days and can start immediately.

**T-02 is closed and no longer gates T-06** — but three of its parameters are still open and each has
a named owner: ~~`k` (T-04a)~~ — **`k = 3`, settled 2026-08-23; T-06 applies it** — `d` (T-06's own F1
run), and the primary bound at each end (T-27, `BRANCH_FAST` / `BIPARTITE`). See
[preregistration](preregistration.md) §7.

**Parallel off it** — T-04 → T-04a → T-17 · T-07 → T-08 → T-19 → T-26 · T-22 · T-13 · T-09 · T-11.

~~**T-04a gates T-06's distance matrices**, so it is on the path for everything downstream of the
competitors even though it is half a day.~~ **Gate released 2026-08-23**: T-06 computes its
production matrices under `levenshtein` for all six admissible serialisations, `kernel` for
`wl_subtree`, and none at all for `adjacency`, `graph6` and `sparse6`.

**Ordering constraints that cost rework if violated**: [manuscript](manuscript.md) §5.

---

## Closing a ticket

**Use the `review-close` skill.** It is the counterpart to `review-ticket`: that one drives a ticket
to completion, this one writes it up. It standardises the board entry, the plan-file RESULT section,
the article notes and the letter fragment — and, most importantly, it enforces the rule that cost
T-03 real time to learn:

> **A finding that contradicts a plan file must be written INTO that file, not only into the ticket
> log.** The log is for whoever audits the ticket; the plan files are the instruction set for whoever
> runs the next one. A correction that lives only in the log is one the next agent will not read.

It also names the **inherited-premise trap** — configuring a check from a plan assertion, getting a
clean one-sided result, concluding something about the *data*, then "independently verifying" it with
a second script that shares the same assertion. That is how T-03 briefly concluded GraphEdX's matrix
was approximate when the premise about its cost model was what was wrong.

## Response-letter fragments

**Every ticket emits its response fragment when it closes.** T-14 assembles, harmonises the register
and writes part 0 — three days is enough for that and is not enough for writing 41 answers from
scratch. [demands](demands.md) is the index; an empty fragment cell is a visible hole.
