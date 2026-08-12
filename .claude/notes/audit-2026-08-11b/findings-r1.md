# Findings — Reviewer #1 · audit-2026-08-11b

**Agent**: `audit-r1` · **Demands owned**: R1.1, R1.2a, R1.2b, R1.3a, R1.3b, R1.3c, R1.3d · **Date**: 2026-08-11
**Sources readable**: manuscript **yes** · letter **yes** · plan **yes**

> **Provenance note.** The agent's own write of this file was blocked by a harness hook. The
> orchestrator persisted the content and added the **Phase-5 verification stamps** marked
> `[ORCH-VERIFIED]`, each re-checked against the manuscript source independently of the agent.

## Verdict table

| ID | Modal | Verdict | One line |
|---|---|---|---|
| R1.1 | SUGGESTION | **COVERED** | Requirement owner is AE.4a, not R1.1; competitors safe |
| R1.2a | REQUIREMENT (defect statement) | **COVERED** | AGM/gSpan absent from every `.tex`; T-08 owns it |
| R1.2b | SUGGESTION | **COVERED** | Five axes cost ~zero over the AE.3 table already required |
| R1.3a | CLARIFICATION | **COVERED** | Density never computed; stratification is AE.1 work |
| R1.3b | **PREMISE — no modal** | **COVERED** | Premise licenses nothing; work rides R1.3c/R1.2b/AE.4b |
| R1.3c | SUGGESTION | **COVERED** | One §4 paragraph — the only absent clause in R1.3 |
| R1.3d | SUGGESTION | **COVERED** | Already at `conclusion.tex:70–71`, `:81` |

**Nothing in R1's slice is unowned.** Two excess items, both **sub-ticket** granularity:
**bliss/Traces** (1 d) and **`labels.md` Tier 2** (0.5 d). **1.5 days recoverable**, neither on the
critical path.

---

## R1.1 — Alternative approaches in a similar problem setting

**Operative clause** (`mail.txt:75`), verbatim:

> "A more informative evaluation would compare the proposed methods against alternative approaches that address a similar problem setting."

**Modal**: "would compare" → **SUGGESTION**.

**Full comment spans** `mail.txt:75`. **What the rest of it is doing**: the first two sentences are a
**separate factual objection**, not justification — "this comparison appears somewhat unfair, as the
objectives and underlying assumptions of the two approaches differ significantly." A published
comparison called unfair is a defect report, and cut guard #1 protects its fix regardless of the
closing modal. **R1.1 therefore contains two asks, not one.**

### Q3 — Already in the manuscript?

| Clause of the ask | State | Evidence |
|---|---|---|
| Fig. 2 compares encode+Levenshtein against per-pair GED | objection **accurate** | `results.tex:216–219` — "measuring the end-to-end speedup of the \IsalGraph{} pipeline (encoding plus pairwise Levenshtein computation) over exact GED" |
| Presented as a headline result, unqualified | yes | `results.tex:230–231`, `:233` — "over $14{,}000\times$ at $n=11$" |
| Any comparator representation | **absent entirely** | `results.tex:209–251` contains only the three IsalGraph variants (`computational_experiments.tex:110–127`) |

**Residual gap**: (i) re-plot so per-graph and per-pair costs stop sharing an axis, plus two
sentences of framing; (ii) at least one comparator representation with a measured curve.

### Q4 — Explicit `or`? No. R3.6a's cheap branch exists but cannot carry the build alone.

### Q5 — Argument or measurement?

**Split.** Clause (i) is argument — encoding is paid once per graph, GED once per pair; there is no
shared denominator. Clause (ii) cannot be argued.

### What the plan currently commits to

`plan.md:103`; `plan.md:429–430` — "(b) Retires **R1.1** (proxy half)", "(c) Retires **R1.1** +
**D16** — the per-graph/per-pair category error". Both clauses owned. `plan.md:371` gives the set
that actually enters: **graph6, sparse6, nauty-graph6, AGM, gSpan min-DFS, IsalGraph**.
T-04 = 3–8 d (`plan.md:545`), T-04a = 0.5–1 d (`plan.md:565`).

### Q6 — Second customer?

| Driver | Line | Text |
|---|---|---|
| **AE.4a** | `:66` | "…including **the choice of benchmark models**…" |
| **AE priority** | `:67` | "**Please address these concerns thoroughly**…" → converts AE.4a to **REQUIREMENT** |
| **AE.3** | `:63–64` | "properties, strengths, and limitations of each" |
| **R1.2b** | `:77` | efficiency and scalability axes need measured numbers |
| **R3.6a** | `:109` | expensive branch of an explicit `or` |

`:67`'s antecedent covers `:59–66`, so **AE.4a inherits a requirement modal**; `:66`'s "Both
reviewers also ask for" has no antecedent other than R1.1 and R3.6a.

### Verdict: **COVERED**

**Why**: R1.1's "would compare" **cannot** carry six backends alone — it is a suggestion. It does not
have to. **AE.4a is the requirement-modal owner.** Cutting the competitors would leave a requirement
the Area Editor singled out for emphasis with no owner.

**Effort**: plan **3.5–9 d** (T-04 + T-04a + share of T-06); component sum at `plan.md:381–388`
reconciles to ≈ 5.5–6.5 d. · Proportionate to **R1.1 alone**: **2.5–3.5 d** — re-plot Fig. 2 (~0.5 d
inside T-20) + gSpan min-DFS (2–3 d, `plan.md:387`). The delta is what AE.4a / AE.3 / R1.2b buy.

**Excess component — bliss/Traces, 1 day** (`plan.md:384`, "0.5 d each"): absent from the
`ReprBackend` list at `plan.md:371`; functionally duplicates nauty (all three emit a canonical
labelling serialised to graph6, differing in *speed*, not representation); adds no table row; named
by no reviewer or editor. Unlike nauty — a dependency of the graph6 and AGM rows
(`plan.md:374–375`) — it has no such role. **Cut order item 1. Saves 1 day, costs nothing.**

This falsifies `plan.md:1008–1009` ("Nothing below this line is cuttable: every remaining ticket is
the sole owner of at least one numbered demand") at **component** granularity.

**Assumptions made**: that `:66`'s "choice of benchmark models" means comparison representations, not
datasets — `:66` covers datasets in the adjacent clause. **`audit-editors` owns AE.4a and must
confirm; this verdict depends on it.** If it means datasets, R1.1 drops to a lone suggestion.

---

## R1.2a — AGM and gSpan not discussed

**Operative clause** (`mail.txt:77`), verbatim:

> "the paper does not adequately position itself with respect to existing graph canonicalization methods. For example, canonical adjacency matrix representations used in Apriori-based Graph Mining (AGM) and depth-first search (DFS) codes employed by gSpan are not discussed."

**Modal**: "does not adequately position itself" / "are not discussed" → defect statement =
**REQUIREMENT**. Reinforced by AE.2 (`:62`, "additional references are needed").

### Q3 — Already in the manuscript?

| Clause of the ask | State | Evidence |
|---|---|---|
| AGM cited or discussed | **absent** | zero hits across all six `.tex` and `cas-refs.bib` `[ORCH-VERIFIED]` |
| gSpan / DFS codes | **absent** | zero hits `[ORCH-VERIFIED]` |
| Canonicalization positioned at all | **absent** | `introduction.tex:18–33` is the entire related-work discussion: SMILES `:19`, SELFIES `:21`, KG embedding `:23`, DeepWalk/node2vec `:26`, MPNN/GNN `:27`, graph transformers `:28`, GraphRNN `:29`, VAEs `:30`, deep matching `:31`. **No canonicalization method.** |
| Claims resting on the absence | **absolute, twice** | `introduction.tex:33` — "No existing method is simultaneously compact, reversible, structure-preserving, and canonicalisable for arbitrary graphs." `[ORCH-VERIFIED verbatim]` · `conclusion.tex:74` — "\IsalGraph{} is the first to do so" `[ORCH-VERIFIED verbatim]` |

**Residual gap**: two paragraphs plus 5–6 references. No experiment.

### Q5 — Argument entirely. Citation and discussion.

### What the plan currently commits to

`plan.md:104` → T-04, T-08. T-08 = 4–10 d (`plan.md:549`); `manuscript.md:24` (new §1.x);
`plan.md:471` budgets 5–6 bibliography entries; `competitors.md:135` records the fallback — "R1.2 is
answered by citation and by the AE.3 table either way".

### Q6 — Second customer? AE.2 and R3.1b need the same section.

### Verdict: **COVERED**

**Effort**: R1.2a's share of T-08 ≈ **0.5 d**. Proportionate: the same.

**Traceability defect (not an effort defect)**: `plan.md:104` books **T-04** (3–8 d) against R1.2a.
R1.2a asks for *discussion*; the vendoring is R1.1 / AE.4a work. If T-04 slips, a §0.5 reader would
wrongly conclude R1.2a is unanswered. **Recommend demoting that T-04 reference to a note.**

---

## R1.2b — Conceptual difference and the five axes

**Operative clause** (`mail.txt:77`), verbatim:

> "It would be helpful for the authors to clarify how the proposed approach differs conceptually from these existing representations and what advantages it offers in comparison. Specifically, does the proposed graph-string representation provide benefits in terms of uniqueness, expressiveness, computational efficiency, scalability, or downstream learning performance? A more thorough comparison with established graph canonicalization techniques would help better contextualize the contribution and novelty of the work."

**Modal**: "It would be helpful", "would help" → **SUGGESTION**.

**Q1 detail that matters**: the axis list is a **question containing a disjunction** —
"…scalability, **or** downstream learning performance?". R1 asks whether a benefit exists on *any*
axis. `plan.md:105` reads it as five printed rows — a legitimate but strengthening reading.

### Q3 — Already in the manuscript?

| Axis | State | Evidence |
|---|---|---|
| uniqueness | claimed, compared to nothing | `conclusion.tex:20–23` |
| expressiveness | unquantified | — |
| computational efficiency | measured only against GED | `results.tex:230–240` |
| scalability | stated as a limitation | `conclusion.tex:68` |
| downstream learning | absent; prospective only | `conclusion.tex:76` — "**may enable**" `[ORCH-VERIFIED]` |

### Q5 — Mostly argument.

Efficiency and scalability are read off T-04 / T-04a / T-05 outputs produced for R1.1 and AE.1;
expressiveness is `labels.md` C2 at "one pass over the 19,670 canonical strings T-06 already
produces. Minutes." (`labels.md:179`); downstream learning is two words.

### Q6 — Second customer?

**AE.3 endorses the same table in the Area Editor's own voice** (`:63–64`, requirement via `:67`) and
asks for "properties, strengths, and limitations of each" — which requires a multi-axis table
independently. R3.1b needs it to license retiring `introduction.tex:33` / `conclusion.tex:74`.

### Verdict: **COVERED**

**Effort**: T-17 = 2–3 d (`plan.md:564`); **marginal cost of R1.2b's five axes over the AE.3 table
≈ 0**. Protected by cut guard #4 (claim scoping) — T-17 "licenses the softening of
`introduction.tex:33` / `conclusion.tex:74`", both verified verbatim.

**Why not OVER**: a five-axis measured table is not disproportionate to "It would be helpful",
because **R1.2b is not paying for it — AE.3 is**.

---

## R1.3a — Is edge density sufficient?

**Operative clause** (`mail.txt:79`), verbatim:

> "As such, it is unclear whether edge density alone is sufficient to explain the observed decline in performance."

**Modal**: "it is unclear whether" → **CLARIFICATION**.

### Q3 — Already in the manuscript? R1's premise is correct.

| Clause of the ask | State | Evidence |
|---|---|---|
| A density attribution exists | yes | `conclusion.tex:69` — "The correlation between Levenshtein distance and GED degrades substantially **as density increases**: Spearman $\rho$ drops from $0.934$ on sparse graphs (IAM LOW) to $0.349$ on the densest benchmark (AIDS)." `[ORCH-VERIFIED verbatim]` |
| Mechanism given | qualitative only | `results.tex:199–202` |
| **Density actually reported** | **NO — the paper reports mean edge *count*** | `results.tex:192–195` — "higher **mean edge counts** coincide with lower $\rho$ values" `[ORCH-VERIFIED verbatim]`; `computational_experiments.tex:40`, `:47`, `:53` give only $\bar m$. **No density figure appears anywhere.** |
| Evidence beyond 5 dataset points | absent | `results.tex:192–202` is the whole argument |

**Residual gap**: the manuscript **argues from $\bar m$ while claiming "density"**. AIDS is filtered
to $n \le 12$ (`computational_experiments.tex:53` `[ORCH-VERIFIED]`), so $\bar m = 10.70$ is a density
of roughly $10.7/\binom{12}{2} \approx 0.16$ — exactly R1's "relatively modest". **This is our
defect. Cut guard #1 applies.**

### Q5 — Argument or measurement?

The paper's own table (`results.tex:151`) gives LINUX at $\bar m = 8.35$, $\rho = 0.433$ — which
refutes the *label* hypothesis, not the density one. Five dataset-level points cannot separate size
from density from domain. **A within-dataset stratification is the right instrument; argument cannot
substitute here.**

### What the plan currently commits to

`plan.md:106`; `plan.md:857` locks stratification "by **node count** and **true density**, within and
across datasets | **AE.1, R1.3, E1**"; `plan.md:868–872` — "report true density per dataset
(**currently uncomputable from the paper** — E1); **stratify AIDS pairs by density** … **This can
refute `conclusion.tex:30–36`.** Run it early."

### Q6 — Second customer? AE.1 (`:60`, "should be clear" — REQUIREMENT), E1, R3.5b.

### Verdict: **COVERED**

**Does it exceed `:79`?** No. The plan does exactly (a) report density, (b) test within a dataset,
(c) hedge or withdraw. No new subsection, dataset or figure is booked against R1.3a.

**Effort**: **hours, not days** — a `groupby` over correlations T-06 computes regardless. No
ticket-board line is attributable to R1.3a alone.

**Assumptions made**: the ~0.16 uses $n_{\max} = 12$, an upper bound on $n$, so it is a *lower* bound
on mean density; R1's point survives either way. The manuscript reports no $\bar n$ (defect E1).

---

## R1.3b — Label loss as the cause · **the sharpest test in this slice**

**Clause** (`mail.txt:79`), verbatim and **declarative**:

> "Consequently, the performance degradation on AIDS may come from the loss of label information rather than structural complexity alone."

**Modal**: "may come from" → **none. This is a PREMISE**, supporting the operative clause two
sentences later (R1.3c).

**A declarative premise licenses no work on its own. The floor the letter requires for R1.3b as an
independent item is zero.** Everything booked against it must earn its place from R1.3c, R1.2b,
AE.4b or a cut guard — and, checked item by item, **all of Tier 0–1 does**.

### Q3 — The premise collapses against the manuscript, three ways, free

| Element | State | Evidence |
|---|---|---|
| Label discarding disclosed | **three times** | `computational_experiments.tex:30–31` — "In all cases, node and edge attributes are discarded; the \IsalGraph{} encoding operates solely on graph topology." `[ORCH-VERIFIED verbatim]`; `conclusion.tex:70`, `:81` |
| **AIDS GED ground truth is itself topology-only** | **refutes the premise outright** | `computational_experiments.tex:52` — AIDS is the "**topology-only variant** from~\cite{jain2024graphedx}" `[ORCH-VERIFIED verbatim]`; `:55–56` — "LINUX and AIDS use **topology-only costs** (zero for node operations, unit for edge operations)" `[ORCH-VERIFIED verbatim]` |
| Counter-example in our own results | yes | `results.tex:151` — LINUX (unlabeled) already at 0.433; IAM Letter, whose class-defining $(x,y)$ coordinates we also discard, at **0.934** |
| **A false sentence — ours** | yes | `conclusion.tex:70` — labels "**present in all five benchmark datasets**" `[ORCH-VERIFIED verbatim]`; `conclusion.tex:81` — "which are present in all five datasets used here" `[ORCH-VERIFIED verbatim]`. False for LINUX. |

### Q5 — Argument replaces measurement, completely

For AIDS, **both sides of the correlation are topology-only**. A variable absent from both arguments
of a correlation cannot have determined its value. **Label loss is arithmetically incapable of
explaining $\rho = 0.349$** — a proof, not weak evidence. Cost: two sentences. Plus the free
counter-example: the dataset losing the most semantically load-bearing attributes has the **highest**
$\rho$.

### Q6 — Second customer, tier by tier

(`labels.md:25–28`; Tier 0 ≈ 2 h, Tier 1 ≈ 3 h, Tier 2 0.5 d + 0.3 core-h, Tier 3 declined)

| Tier | Item | Licensed by | Survives? |
|---|---|---|---|
| 0 | C1 rebuttal prose | **R1.3c** — this paragraph *is* the absent clause | yes |
| 0 | C3 label column + E6 fix | **AE.4b** (`:66`, requirement via `:67`) **+ cut guard #1** | yes, twice |
| 0 | C4 future-work concreteness | **R1.3d** | yes |
| 1 | C2 collision count | **R1.2b** expressiveness + **AE.3**; `labels.md:161–168` says so explicitly | yes |
| 2 | L1–L3 logged label-aware GED | **nobody** — `labels.md:27` calls it "Round-2 insurance" | **no** |
| 3 | results subsection | declined by default | — |

### Verdict: **COVERED**

**Why**: no tier rests on the premise. Tier 1's driver is named correctly as R1.2b / AE.3 rather than
R1.3 — the exact discipline this audit exists to enforce. **The plan gets R1.3b right.**

**Effort**: Tier 0–1 ≈ **5 hours**, and it would be the same 5 hours if R1.3b had never been written.

**Two structural flags:**

1. **`plan.md:107` gives a premise its own budget line.** Recommend §0.5 mark it
   `PREMISE — served by R1.3c / R1.2b / AE.4b`, with no separate ticket allocation.
2. **Tier 2 is the only item in this slice with no demand-side driver.** Already first on the cut
   list (`plan.md:997`) and deferred to the 2026-08-18 PI decision. Recorded so that decision is
   made knowing **no reviewer asked for it** — `labels.md:35–38`'s case is option value, not
   coverage.

**Assumptions made**: LINUX is unlabeled — verified indirectly (R1 asserts it at `:79`; no attribute
description at `computational_experiments.tex:44–48`, whereas AIDS's molecular content is named at
`:50–52`). The raw dataset was not opened. `conclusion.tex:70` is wrong about IAM Letter regardless,
so E6 survives either way.

---

## R1.3c — Discuss the limitation and its impact

**Operative clause** (`mail.txt:79`), verbatim:

> "A more thorough discussion of this limitation, along with its impact on the reported results, would strengthen the paper."

**Modal**: "would strengthen" → **SUGGESTION**. The operative noun is **discussion**.

### Q3 — Already in the manuscript?

| Clause of the ask | State | Evidence |
|---|---|---|
| "discussion of this limitation" | **already satisfied** | `conclusion.tex:70` — "The current formulation operates on graph topology only; node and edge labels … are discarded during encoding." `[ORCH-VERIFIED]` · `:71` — "a **prerequisite** for applications in domains such as molecular chemistry and program analysis" `[ORCH-VERIFIED]` · also `:81` |
| "**its impact on the reported results**" | **absent** | §5 discusses labels; `results.tex:192–206` interprets AIDS by density. **Nothing joins them.** |

**Residual gap**: **one paragraph.**

### What the plan currently commits to

`plan.md:108` — "R1.3 asks for a **discussion**, not an experiment — the missing piece is the
*connection*"; `labels.md:113–114`.

### Verdict: **COVERED**

The README's over-scoping correction did **not** overshoot: no label experiment, no results
subsection, Tier 3 declined. Effort: one paragraph inside T-20's §4/§5 rewrite.

**Correction to the README lesson** (`README.md:136–154`): it correctly cites `conclusion.tex:70–71`
and `:81` as already satisfying two asks, but does **not** note that both sentences are **factually
wrong** — "present in all five benchmark datasets". Its recommended "point at it" response would
point at a false sentence. `labels.md:25`, `:190–192` catch this (E6); the README does not. **Anyone
working from the README alone reproduces the error.**

---

## R1.3d — Labels as future work

**Operative clause** (`mail.txt:79`), verbatim:

> "Especially if incorporating label information could be applicable and a promising direction for future work."

**Modal**: "could be … promising" → **SUGGESTION**, the softest register in the letter.

**Already satisfied twice**: `conclusion.tex:71` and `:81` — "A second open problem is the
incorporation of node and edge labels … a labelled variant of \IsalGraph{} would permit direct
comparison with GED under non-uniform cost functions." `[ORCH-VERIFIED verbatim]`

`:81` describes the label-aware GED experiment *verbatim* as future work — **the strongest available
argument for not running it inside 20 days.**

**Residual gap**: concreteness only. Plan: `plan.md:109`; `labels.md:194–205`, with the correct guard
"**Do not write that row until T-07 confirms it.**"

### Verdict: **COVERED** — ≈ **0.5 hour** inside T-12.

---

## Notes for the orchestrator

**Cross-voice overlaps — merge to one row:**

- **R1.1 ↔ AE.4a** (`:66` + `:67`, requirement), **R3.6a**, **AE.3**, **R1.2b**. **R1.1 is not the
  requirement-modal owner of the competitors — AE.4a is.** Reconcile with `audit-editors`.
- **R1.2a ↔ AE.2** (`:62`), **R3.1b** — one T-08.
- **R1.2b ↔ AE.3** (`:63–64`) — one T-17, two drivers; AE.3 is the requirement, R1.2b supplies axes.
- **R1.3a ↔ AE.1** (`:60`), **E1** — the density stratification is AE.1's. Do not charge it twice.
- **R1.3b (C3) ↔ AE.4b** (`:66`) — one table column.

**Priority statement**: `mail.txt:67` — "Please address these concerns thoroughly" — antecedent
`:59–66`. **This is what converts AE.4a and AE.4b from soft to requirement.** Everything expensive in
this slice hangs off that sentence, not off any R1 modal.

**Could not verify**: whether LINUX carries labels (indirect evidence only); whether `:66`'s
"benchmark models" means representations or datasets (`audit-editors`' row).

**Factual errors in R1's own premises** — these bear on how the response is worded and often
strengthen our position, but they are never a scoring move:

1. **R1 is wrong that IAM is unlabeled.** IAM Letter carries continuous $(x,y)$ coordinates — the
   attributes defining the fifteen letter classes. **This strengthens our position**: the benchmark
   losing the most load-bearing attributes has the highest $\rho$. `labels.md:157–159` prescribes the
   right tone.
2. **R1 is right that AIDS's density is modest**, and right that we never established it. Our defect.
3. **Our own text is wrong about labels** (`conclusion.tex:70`, `:81`). Owner: T-18 Tier 0 / E6.

**Terminology hazard for the response letter** (`methodology.tex:430–431`): the manuscript uses "node
**labelings**" in the graph-isomorphism sense (vertex numbering) — "a *labeling-independent*
representation". If both senses share a paragraph of the response letter, R1 will read a
contradiction where none exists.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-11 | Created by `audit-r1` for audit-2026-08-11b; persisted by the orchestrator after a harness hook blocked the agent's own write. Phase-5 stamps `[ORCH-VERIFIED]` added by the orchestrator against the manuscript source. |
