# Corrections — claim scoping, manuscript defects, formal audit

**Owners**: T-11 (text defects), T-12 (claim scoping), T-22 (formal statements), T-07 (prior-work delta)
**Serves**: R3.1a/b, R3.3a/b/c, R3.4a/c, R3.6a/b, R3.7a/e, E1–E12
**Status**: all accepted. **Ordering constraint: T-06 (numbers) → T-12 (scoping)** — every scoped
claim quotes a number T-06 re-derives.

Related: [manuscript](manuscript.md) (where each edit lands) · [demands](demands.md) ·
[statistics](statistics.md) · [approx_ged](approx_ged.md)

---

## 1. Claim scoping — B1…B6

| # | Change | Sites |
|---|---|---|
| **B1** | **Scope G2S**: undirected **connected**; directed **root reaching all nodes**. State the asymmetry — **S2G total, G2S partial** | `main.tex:106–108`, `introduction.tex:33`, `:45–46`, `conclusion.tex:74` |
| **B2** | **Directedness**: the flag is **external metadata**; **restate Theorem 2.12 within a fixed directedness class**; move the "deterministic given `w` and the flag" hypothesis from the proof into the statement | §2.3.3 |
| **B3** | "GED **standard** construction" → "**explicit-construction reference model**"; real serializations supplied beside it | §3.2.3 |
| **B4** | Propagate the results section's **conditional** framing to abstract and conclusion. Numbers re-derived in T-06 | abstract, §5 |
| **B5** | Limitations: the `n` ceiling **with its cause**; exponential worst case; **no sequential or downstream task** | §5 |
| **B6** | Unify the four-properties claim; attach it to the T-17 comparison table; **soften "no existing method"** | §1, §5 |

**B1's body is already correct** (`methodology.tex:277`, `:352–358`, `:438`) — this is a scoping pass
over four abstract/intro/conclusion sites, ≈ 2 h, 0 net pages.

**B6 must unify as well as soften.** The absolute claim appears twice with **different property sets**:
`introduction.tex:33` (compact / reversible / structure-preserving / canonicalisable) vs
`conclusion.tex:74` (universal validity / reversibility / canonical completeness).

**B2 — use the exact witness, never a rate without its window.** A single undirected edge and a
single directed arc both canonicalise to `"V"`. That witness is exact and needs no enumeration. If a
collision *rate* is quoted, its enumeration window must be quoted with it: over labeled edge sets with
`n ≤ 4` and ≤ 4 edges, 63 of 441 collide; over distinct canonical strings in that window, 6 of the 7
undirected classes are also produced by some directed graph. **Two measurements of one phenomenon —
do not merge them.**

---

## 2. T-22 — the formal-statement audit

**R3's premise is half wrong, and the real defect is worse than described.**

Theorem 2.12 (`methodology.tex:628–637`) reads only *"Let G and H be finite, simple, connected graphs.
Then w*_G = w*_H ⟺ G ≅ H."* — it **does not mention the `directed` flag**. Only the **proof** does
(`:643–644`: "The decoder S2G is a deterministic function of `w` **and the directed flag**").

**A load-bearing hypothesis lives in the proof and not in the statement.** T-22's restatement is
therefore a **factual correction**, not a stylistic one, and re-checking the proof is forced by it.

Scope:

1. Restate Thm 2.12 within a fixed directedness class; move the flag hypothesis into the statement.
2. **Re-verify all three proof steps.**
3. **Propagate to Corollary 2.13** — its proof derives identity of indiscernibles "directly from
   Theorem 2.12" (`methodology.tex:738–740`), and Cor. 2.13 is [statistics](statistics.md) **D6's
   lead justification**, which carries the entire T-03 recompute. **Auditing the corollary is a
   prerequisite for the argument justifying T-03.**
4. Add a directedness-collision regression to `tests/property/` — **unasked-for; cheap; drops first
   if T-22 overruns.** Cost of dropping: hours, no manuscript content.

---

## 3. Manuscript defects — all accepted

| ID | Defect | Fix | Owner |
|---|---|---|---|
| **R3.4a** | Alg. 2 `C`/`c` **guards and duplicate checks** reversed vs Table 1 | rewrite `methodology.tex:321–336` to match `graph_to_string.py:208–238` | T-11 |
| **R3.4c** | `n^{9.0}` at `conclusion.tex:50` has no source; `:50` vs `:68` disagree; `n^{4.9}` called "super-polynomial" | all exponents re-derived in T-06; three-way separation in T-13 | T-06, T-13 |
| **R3.7e** | "breaks permutation equivariance" | → **invariance**. `M → P M Pᵀ` *is* equivariance; invariance is what breaks | T-11 |
| **E1** | density never computed; no node count reported | [data](data.md) §1 | T-01, T-20 |
| **E2 / F2** | 473,147-pair gap; LINUX 3,916 vs 1,685 | **cause: within-split GED coverage**, not filtering. Fixed by T-03 | T-03 |
| **E3** | fits declared `n = 3–20`, greedy data to 50 | re-derived | T-06, T-20 |
| **E4** | a fourth node range (`n = 3–11`) appears | cross-referenced | T-20 |
| **E5** | abstract self-contradiction — `:106` "any finite, simple graph" vs `:114` "any connected graph" | B1 | T-12 |
| **E6** | "labels present in all five datasets" — **false for LINUX**, two sites: `conclusion.tex:70`, `:81` | corrected. **T-12 owns the edit** (was claimed by three tickets at once) | T-12 |
| **E7** | algorithms float to pp. 33–35, after the references | relax `\floatpagefraction{1}` / `\textfraction{.001}` (`main.tex:66–67`) and place algorithms near their discussion. **Must run BEFORE T-15** — it changes pagination, and it is the single largest page recovery available (~2 p) | **T-11** |
| **E8** | a draft self-correction is printed in Example 2.3 | delete; `[0,2,1]` is right | T-11 |
| **E9** | 13 dead entries in `cas-refs.bib` (56 defined, 43 cited) | prune, so the 35–55 count cannot be miscounted from the file | T-08 |
| **E11** | generative-AI declaration commented out | restore; Elsevier compliance | **T-24** |
| **E12** | orphaned figure PDFs; **`graphical_abtract.pdf` misspelt**, referenced under that spelling at `main.tex:131` | rename and re-reference | **T-24** |
| ~~**D19**~~ | ~~[28] Transformer / [29] LSTM claims unverified~~ → **RESOLVED 2026-08-26 (T-07), both halves.** **[28]**: Transformer classification CONFIRMED — but on a **synthetic** 3,000-sample 3-class set, **one** non-graph baseline, **no numeric result printed anywhere in the text**. **[29]**: CONFIRMED but under-described — **LSTM *and* GRU**, and the task is **masked token prediction** over ZINC, **not classification**. R3's record stays intact: both claims are true in kind. See §4 and [T-07 notes](../tasks/T-07-article-notes.md) | T-07 |
| **E13** | **Remark 2.7 (`methodology.tex:462`, `\label{rem:search-space}`) excludes half of its own search space.** It reads *"Only the identity of the uninserted neighbour chosen at each `V`/`v` step contributes to the search space"*, but Definition 2.6 three lines above defines `w*_G` over **any starting node** and `core/canonical.py` searches both. Found by T-09 while drawing the figure R3.7c asked for, which shows one subtree per start node and so **contradicts the prose it illustrates** | replace with: *"Two things are searched over: the starting node, and the identity of the uninserted neighbour chosen at each `V`/`v` step. The priority order […] and the minimum-displacement pair ordering […] are intrinsic to the algorithm definition and are not branched over."* One clause, no page cost. **The rest of the remark is correct and is what R3.7c is about** | **T-11** |

**E13, measured, not argued**: on T-09's running example (`n = 6`, `|Aut(G)| = 1`), greedy
`G2S` from the six starting nodes gives strings of length **9, 10, 9, 11, 10, 10** — six
distinct strings, one of which attains `w*_G`. The starting node is the outer loop of the
search, not a free choice, and Remark 2.7's word *"Only"* denies it. E8 (Example 2.3's
printed self-correction) sits four hundred lines earlier in the same file and is also T-11's;
**do both in one pass**. Evidence: [T-09 article notes](../tasks/T-09-article-notes.md) §1.

**R3.4a, confirmed against the code**: Table 1 defines `C` = primary→secondary, `c` =
secondary→primary. Algorithm 2's `C` guard tests `(ṽ₂,ṽ₁) ∈ E` and duplicate-checks
`(ℓ₂,ℓ₁) ∉ E(G_out)` while *adding* `(ℓ₁,ℓ₂)`. The implementation guards
`tent_sec_in in neighbors(tent_pri_in)` and checks `tent_sec_out not in neighbors(tent_pri_out)`,
then adds `(tent_pri_out, tent_sec_out)` — **both** guard and duplicate check match Table 1.
**Pseudocode wrong, implementation right.** The reviewer spotted only the guards; we fix both.

**R3.4c is three-way, not two-way**: `results.tex:88`/`:107`/`:239` give α = 4.9;
`conclusion.tex:50` gives **n^9.0** *and* n^4.5; `:68` gives n^4.9; `:80` calls the fitted curve
"super-polynomial". The reviewer named two of the three.

---

## 4. T-07 — the [28] / [29] delta

**[29] is published and already cited**: `ThurnhoferHemsi:2025` — Thurnhofer-Hemsi, García-Aguilar,
Fernández-Rodríguez, López-Rubio, *Representation of Molecules by Sequences of Instructions*,
**J. Chem. Inf. Model. 65(15):7936–7955, 2025**. Write the table from the **paper**; use
`github.com/icai-uma/IsalChem` as the implementation cross-check. D19's [29] half is directly
resolvable rather than inferred.

> ## ⚠ CORRECTED 2026-08-26 (T-07) — the in-repo path is dead, and [29] is open access
>
> **`docs/references/` does not exist.** The directory was deleted in `7d18f52` *"Initialize github
> pages site structure"*, taking `docs/references/2512_10429v2.pdf`, `docs/references/Idea.pdf` and
> `docs/original_code_and_files/2512.10429v2.pdf` with it. `.claude/CLAUDE.md` "Key References"
> carries the same dead path. **This would have blocked T-07.**
>
> **Recover with** `git show a23acbf:docs/references/2512_10429v2.pdf > <dest>` — 12 pp., verified
> as arXiv:2512.10429v2. An archived copy now lives in the T-07 report's `sources/`.
>
> **Better than the plan predicted: [29] is CC BY open access**, not abstract-only. Full text at
> **PMC12344769**, DOI `10.1021/acs.jcim.5c00354`, PMID 40720985, retrievable from the NCBI BioC
> endpoint. **Caveat that matters**: the BioC conversion strips all three algorithm listings, every
> equation and every table body. Anything that depends on [29]'s pseudocode needs the publisher PDF.

~~**[28]** is the preprint, PDF in-repo at `docs/references/2512_10429v2.pdf`, and is **permanently
arXiv-only**.~~ [28] is the preprint and is **permanently arXiv-only**; see the recovery command above.

### 4.1 R3.1a has two conjuncts — deliver both

> "The paper should provide a detailed side-by-side comparison that identifies which components are
> inherited, modified, or genuinely new, **and explain why the combined extension constitutes a
> sufficiently substantive contribution.**" (`mail.txt:86`)

**(i) The delta table.** The entire existing prior-work comparison is **two sentences**
(`introduction.tex:52–53`).

> ## ⚠ CORRECTED 2026-08-26 (T-07) — the predicted row assignment was too generous to "inherited"
>
> The prediction below assigned **five** components to *inherited*. Measured against both papers and
> the reference implementation, only **three** survive. **The conclusion is unchanged and slightly
> strengthened**: the completeness theorem is still the single new component, and it is now
> evidenced rather than assumed.
>
> | Predicted | Measured | Why it moved |
> |---|---|---|
> | CDLL — inherited | **modified** | [29]'s list holds **hydrogen atoms** as free-valence placeholders, not molecule vertices (`29.txt:39`, confirmed in `datastructures.py:118–119`). A graph with no hydrogens has no container. [28] has **no linked list at all** — `grep "linked list"` = 0. |
> | two-pointer VM — inherited | **inherited** ✓ | dual-sourced, `29.txt:127` and `isalchemstate.py:42–43` |
> | alphabet — inherited | **modified** | [29]'s tokens are 1–2 characters fusing opcode with element and bond order; ours are nine label-free single characters. [28]'s is five symbols, none of which creates a node. |
> | incremental construction — inherited | folded into *execution state* | not a separable row once the container differs |
> | normalisation — inherited | **inherited** ✓, but **only the ordering criterion** | see the box below |
> | generic-topology redesign — modified | **modified** ✓, split across *domain*, *execution state*, *alphabet* | one row could not carry it |
> | completeness theorem — new | **new** ✓ | and now evidenced: `theorem`, `proof`, `lemma`, `proposition` each occur **0 times** in **both** predecessors |
>
> **Final table: 9 rows — 3 inherited, 5 modified, 1 new.**
>
> ### 🔴 On "exhaustive shortest-then-lexicographic normalization"
>
> That phrase is **R3's** (`mail.txt:86`), not the plan's, and it is **half verifiable**:
>
> - **shortest-then-lexicographic** — CONFIRMED, dual-sourced. `29.txt:141` and `compress()`'s own
>   docstring at `isalchemutilities.py:486–488`.
> - **exhaustive** — **not supportable from any source available to us.** [29] asserts it
>   (`29.txt:145`, *"exploring all possible options in steps 7 and 9 of Algorithm 2"*) but the CC-BY
>   text strips Algorithm 2's body, so what those steps range over cannot be read; and the public
>   implementation enumerates the **starting heavy atom only**, with one greedy `set.pop()` for
>   neighbour choice and zero backtracking (`isalchemutilities.py:491–496`; `grep -ci
>   "backtrack\|branch\|prune\|itertools\|permutation"` = 0).
>
> **Tab. 3 therefore attributes only the ordering criterion and says nothing about [29]'s search
> space.** A "search space" row would have favoured us and was rejected for being unsourceable.
> Settling it needs the publisher PDF; **nothing printed depends on it.**

~~The table documents CDLL, two-pointer VM, alphabet, incremental
construction, normalisation as inherited; generic-topology redesign as modified; the completeness
theorem as new.~~

**(ii) The sufficiency argument — one paragraph closing §2.x, ~120–150 words, ≈ 0.1 page.** This was
**unowned in every document** until 2026-08-12. It re-orders facts T-07 already gathers — the
completeness theorem (`methodology.tex:628–637`) is the new result; generic topology replaces [28]'s
fixed node ordering and [29]'s molecular restriction; scope extends to unlabeled, unbounded-degree
graphs. **No new investigation.**

> **Why (ii) cannot be dropped.** The delta table will document that **both predecessors ran a
> sequence model and this paper does not** — which is R3.2's exact argument. Delivering (i) without
> (ii) hands R3 the conclusion that the extension is *less* substantive: **the artifact becomes
> evidence against us.**
>
> ## ⚠ OVERRULED 2026-08-26 by PI decision — the row is NOT in Tab. 3
>
> Tab. 3 ships **architectural only**. The sequence-model concession is made **once**, on its own
> terms, in **§6.3** ([prose](prose.md) §2's red line), and the sufficiency paragraph stands on the
> theorem alone.
>
> 🔴 **No T-07 artifact now discharges the pre-emption.** If §6.3 does not carry it, the demand falls
> between two tickets and nothing on the board catches it. **T-14 owns it**, and T-07 handed over the
> *measured* content so it can be written from fact rather than from R3's paraphrase —
> [T-07 article notes](../tasks/T-07-article-notes.md) §4 and
> [REPORT](/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-07-prior-work-delta/REPORT.md) §5.
>
> **The measured content, which R3's phrasing under-describes in both directions**: [28]'s Transformer
> experiment is 3-way classification on a **synthetic** 3,000-sample set with **one** non-graph
> baseline and **no numbers printed in the text**; [29]'s is **LSTM *and* GRU** on **masked token
> prediction** over ZINC, not classification. **Neither predecessor ran a downstream graph-learning
> evaluation on a real benchmark**, which is the strongest honest ground §6.3 has.

~~**Pre-empt the reading inside the table itself.** Write the row as a stated scope decision —
*sequence-model evaluation: present in [28] and [29] on their respective domains; deliberately out
of scope here, where the contribution is the canonicalisation result; designated as the next
study* — rather than leaving the reader to notice the gap.~~

This paragraph is also the home for **AE.5**'s only unowned clause — R3's preamble at `mail.txt:83`
names "**rationale**" among four targets requiring clarification. Novelty → R3.1, details → R3.4,
interpretation → R3.5/R3.6; **rationale lands here at no marginal cost.**

---

## 5. T-13 — the complexity section

`P(M)` recomputation is *derivable* from `methodology.tex:293–295` but never stated, and grepping
`complexity|O(|Θ|polynomial|worst-case` across the manuscript returns **five hits, all qualitative**.
**There is no complexity analysis of G2S or the canonical search anywhere** — yet `main.tex:114–115`
claims G2S runs "in time **polynomial** in the number of nodes". **The requested accounting is the
missing support for something the abstract already asserts, so it cannot be trimmed.**

Deliverables, all argument, no measurement:

1. **`P(M)` is recomputed per frame**, stated explicitly.
2. **Cost the four named operations**: pair scanning, pointer walking, neighbour checks, canonical
   backtracking.
3. **Three-way separation** (R3.7d): theoretical complexity / worst-case search behaviour / empirical
   runtime scaling, never conflated.
4. **The characterised worst case replaces "exponential"**: cost is governed by `|Aut(G)|`, not size
   or density, and the current triplet pruning key is **provably coarser than 1-WL** (2.4–2.6× fewer
   classes, measured). That is a stronger and more honest statement than an unqualified
   "exponential", and it costs hours instead of days.
5. **Automorphism pruning is future work, not this revision.** Individualisation–refinement with
   automorphism detection is what nauty/bliss/Traces do and is the actual fix; re-implementing it is
   a project. State it as future work and cite nauty — already vendored as a competitor, so the
   citation is free.

**Page cost ≈ 0.5 page, additive, in a 35/35 document — a budget item, not a cut candidate.**

---

## 6. R3's own factual slips — recorded neutrally

These strengthen our wording, not our score, and are corrected in passing without emphasis:

- R3.3b says Theorem 2.12 "states" the flag hypothesis. **It does not — the proof does**, which is a
  worse defect than described.
- R3.4a identifies reversed guards but not the equally reversed **duplicate checks**.
- R3.4c names two contradictory exponents; there are **three**.
- R1.3 says "IAM and LINUX are unlabeled". **IAM Letter carries class-defining `(x, y)` coordinates**
  — see [labels](labels.md) C1.

---

## 7. §4 RESULT — T-07, closed 2026-08-26

**Both artifacts built and measured. No compute, no code, no test-suite impact.**

| Artifact | Where | Measured |
|---|---|---|
| **Tab. 3**, 9 rows × 4 columns, grouped inherited / modified / new | `artifacts/tab3_prior_work_delta.tex` | **368.4 pt = 0.67 p** against a 0.70 p budget; 0 overfull boxes, 0 undefined control sequences |
| **Sufficiency paragraph**, R3.1a(ii) + AE.5 "rationale" | `artifacts/sufficiency_paragraph.tex` | **145 words**, inside the 120–150 band; stands on the theorem, no R3.2 defence |

Archive: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-07-prior-work-delta/`
— `REPORT.md`, `PROVENANCE.md` (per-cell anchors), `VERIFICATION.md`, `sources/`, `inventories/`.

### The headline, and it is favourable

**Neither predecessor contains a single formal result.** `theorem`, `proof`, `lemma` and
`proposition` each occur **0 times** in **both** [28] and [29]; `complete invariant` occurs 0 times
in both; `graph edit distance` occurs 0 times in both.

- **[28]**: its "canonical string" is *defined as the greedy encoder's output* on an adjacency matrix
  that is itself a function of an assumed total vertex order (`28.txt:89`). Canonical **per matrix**,
  not per isomorphism class.
- **[29]**: relabelling invariance argued in **three sentences, one direction only**, resting on an
  explicit assumption (`29.txt:145`). The converse is never stated, and **no experiment measures a
  collision rate**, so the property is not even claimed empirically.

**Theorem 2.12 is genuinely new, and this is now evidence rather than assumption.**

### Composition, under a rule frozen before the sources were read

The attribution rule and its five tie-breaks — all resolving toward the conservative reading, with
*generalisation is modification, never novelty* — were committed as `b300581` **before** any
inventory was read. Result: **3 inherited, 5 modified, 1 new.** The metric corollary was folded into
the theorem row rather than counted as a second novelty. **The table understates.**

### Two rows deliberately absent

- **Normalisation search space.** Would have favoured us; rejected as unsourceable. See §4's box.
- **Sequence-model evaluation.** PI decision; see the OVERRULED box in §4.1. **T-14 owns the
  pre-emption now** — this is the one live debt T-07 leaves.

### Standing requests answered

§4 asked T-07 to *"write the table from the paper; use `github.com/icai-uma/IsalChem` as the
implementation cross-check"*. **Done, and the cross-check earned its cost**: the code independently
confirmed the hydrogen-only container and the shortest-then-lexicographic criterion, and it is what
revealed that "exhaustive" cannot be sourced. Governance rule used: **the paper sets the cell, the
code corroborates**; where they disagree the disagreement is reported, never averaged.
