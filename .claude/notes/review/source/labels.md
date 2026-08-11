# Labels and attributes — R1.3, AE.4b

**Status**: **v2.0, 2026-08-11.** Rescoped after author review: v1.0 proposed a new experimental
section and that was **overreach**. Owner: **T-18**.

Answers R1.3, AE.4's "fully labeled, vs. partially-labeled" sub-issue, and the *expressiveness* row
of the T-17 comparison table.

---

## 0. PI decision — is the effort worth it?

**This document is a proposal, not a lock.** Sections 3 and 4 describe work of very different cost
and very different value, and the trade is a judgement about how much of a 20-day revision to spend
on the softest-modal comment in the review. That call belongs to the PI.

The evidence needed to make it is in §1 (**the manuscript never claimed label handling** — verified
against the sources), §2 (**R1.3 asks for a discussion, not an experiment**) and §5 (**a labelled
variant is a different paper**).

### The four tiers

| Tier | Content | Effort | What it buys | What it risks |
|---|---|---|---|---|
| **0 — not optional** | **C1** rebuttal · **C3** label column + **E6 correction** · **C4** future work made concrete | ~2 h of writing | R1.3 is answered; AE.4b is answered; a **false sentence** (`conclusion.tex:70`, `:81` — labels are *not* in all five datasets) is fixed | nothing. C3 is a factual correction we owe regardless of R1.3 |
| **1 — recommended** | + **C2** topological collision count | ~3 h: one pass over strings T-06 already emits, plus two table columns and a sentence | **Its driver is R1.2 / AE.3, not R1.3** — the *expressiveness* row of the comparison table the Area Editor endorsed needs a number, not the phrase "blind to labels". It also happens to answer R1's concrete scenario with a measurement | a large fraction is an uncomfortable result. It is still better found by us than by R3 in round 2 |
| **2 — proposed** | + **L1–L3** computed and stored, **not written up** | ~0.5 d wiring + **0.3 core-hours** compute | Round-2 insurance. If R1 presses, the answer already exists and does not need a new run under a shorter deadline | half a day on the critical path, for material that may never be used |
| **3 — declined by default** | Promote L1–L3 into a results subsection | ~1 d + **≈ 1 page** | Quantifies the label-blindness cost in ρ | Costs a page we do not have (`manuscript.md` §3); introduces a second cost model into a revision whose headline statistical fix is **one** cost model (D6), so it must carry the §4 disclaimer or it reads as backsliding on R3.5b |

### Recommendation

**Tier 2**: commit Tiers 0–1 to the manuscript, compute and store L1–L3 without writing them up, and
revisit Tier 3 only if the page budget survives T-15 or if R1 presses in round 2.

The reasoning is asymmetric cost. Tiers 0–1 are hours and answer the comment. Tier 2 is half a day
and buys an option that is worth far more in round 2 than it costs now, because the compute will
already be configured and the alternative is re-running GED under a second cost model against a
shorter clock. Tier 3 is the only tier that competes for pages, and it is the only one that can wait.

**The counter-case, stated fairly**: R1.3's modal is "would strengthen"; the manuscript already names
this as future work twice; and every hour here is an hour not spent on T-20, which has no owner
outside this revision and touches five sections. A PI who reads the review as "answer it in prose and
spend the time on the recompute" is making a defensible call, and Tier 0 alone is a complete, honest
answer to R1.3.

**Decision needed by 2026-08-18** — before T-06 launches, because L1–L3 must be configured into the
same run rather than bolted on afterwards.

---

## 1. The framing that decides everything: we never claimed labels

Verified against the uncommented `.tex` sources, 2026-08-11:

| Location | Text |
|---|---|
| `computational_experiments.tex:30` | "In all cases, node and edge attributes are **discarded**" |
| `conclusion.tex:70` | "The current formulation operates on **graph topology only**; node and edge labels … are discarded during encoding" |
| `conclusion.tex:71` | "Extending IsalGraph to attributed graphs is therefore a **prerequisite** for applications in domains such as molecular chemistry and program analysis" |
| `conclusion.tex:81` | "A second open problem is the incorporation of node and edge labels … **a labelled variant of IsalGraph would permit direct comparison with GED under non-uniform cost functions**" |

**The manuscript states three times that labels are discarded and twice names their incorporation as
future work.** R1.3 is therefore not "you claimed X and failed to deliver". It is a scope observation
plus a causal hypothesis about one number.

Two consequences that shape the whole response:

1. **`conclusion.tex:81` already anticipates the experiment.** "A labelled variant … would permit
   direct comparison with GED under non-uniform cost functions" is, verbatim, the label-aware GED
   study. The paper already classifies it as future work requiring a variant that does not exist.
   Building it now would be executing our own stated future work inside a 20-day revision.
2. **⚠ Terminology trap for the response letter.** `methodology.tex:430–431` uses "node **labelings**"
   to mean **vertex numbering** — "different node labelings may produce distinct strings … a
   *labeling-independent* representation". That is the graph-isomorphism sense, unrelated to semantic
   labels. The two senses must never appear in the same paragraph of the letter or the reviewer will
   read a contradiction where none exists.

---

## 2. What R1.3 actually asks for

> A more thorough **discussion** of this limitation, along with its impact on the reported results,
> would strengthen the paper. Especially if incorporating label information could be applicable and
> **a promising direction for future work**. (`mail.txt:79`)

A discussion and a future-work statement. Not an experiment. Compare the modal against R3's
"should be described" / "should be narrowed" — this is the softest register in the review after
R3.2.

**And the comment opens with the complaint it is actually making**, which is not about labels at all:

> Moreover, **the discussion of the experimental results is rather overlooked.** (`mail.txt:79`)

Everything that follows is the illustration. The argument runs: you attribute the AIDS degradation to
density → AIDS's density is modest, so density alone may not explain it → and here is an alternative
you never considered. **R1 is asking us to defend or withdraw a causal claim**, and uses labels as
evidence that we never examined it.

### R1's three asks — two are already satisfied

| Ask | State in the submitted manuscript |
|---|---|
| Discuss the label limitation | **Already there** — `conclusion.tex:70–71`, `:81` |
| Labels as a promising future direction | **Already there** — `conclusion.tex:81` |
| **"along with its impact on the reported results"** | **Absent everywhere** |

`reviewer-1.md` states it exactly:

> What is **not** present anywhere is R1's actual argument: that label loss, rather than density, may
> **cause the AIDS result**. The limitation is filed under future work; it is never connected to the
> interpretation of the AIDS number.

**The literal deliverable is one paragraph in §4.** The limitation lives in §5, the AIDS
interpretation lives in §4, and nothing joins them.

### Which half costs work

| Half of R1.3 | Answerable how | Cost |
|---|---|---|
| **Labels cause the AIDS number** | **Prose** — the argument in C1 is decisive and needs no measurement | free |
| **Density does not explain it** | **Measurement** — the paper never computes density (E1), so there is nothing to defend the claim with | already budgeted under **AE.1**, not attributable to R1.3 |

AE.4b is separate and lighter still: "differences in information and structure in the graph datasets
used (e.g., fully labeled, vs. partially-labeled)" is answered by **a column in the dataset table**.

---

## 3. Tiers 0–1 — the manuscript deliverables

Four items. Total cost: one table column, one paragraph in §4, one paragraph in §5.
**C1, C3 and C4 are Tier 0 and are not optional; C2 is Tier 1 and is recommended** (§0).

### C1 — The rebuttal, stated plainly — **two arguments, both free**

**Logical.** Both sides of the reported correlation are topology-only: the GraphEdX AIDS matrix is
the topology-only release (`computational_experiments.tex:52`) and the string encodes topology. ρ on
AIDS is a functional of the topology-only graph collection alone, so **label loss cannot have
determined it**. It survives the recompute unchanged, because D6 is also topology-only.

**Empirical, and it comes out of the submitted paper's own Table 3.** R1's premise that "IAM and
LINUX are unlabeled" is wrong for IAM (D7): IAM Letter carries continuous `(x, y)` node coordinates
— **the very attributes that define the fifteen letter classes** — and we strip them at
`iam_letter_loader.py:4`, `:60`. So:

| Dataset | What is discarded | ρ (pruned canonical) |
|---|---|---:|
| IAM Letter | `(x, y)` coordinates — **class-defining** | **0.93** |
| AIDS | atom and bond types | **0.349** |

**The benchmark that loses the most semantically load-bearing attributes has the highest
correlation.** If label loss drove the degradation the ordering would be reversed. This is a
counter-example, not merely an absence of evidence, and it costs nothing to state.

It is a rebuttal, not a dismissal — the paragraph continues into C4, and the density half of R1.3 is
answered by measurement in `plan.md` §8.

> **Tone.** R1's factual slip about IAM is corrected in passing and without emphasis — "IAM Letter in
> fact carries continuous node coordinates, which we also discard; this strengthens rather than
> weakens the point" — never as a scoring move.

### C2 — Topological collision count *(the one measurement)* — **driven by R1.2 / AE.3, not by R1.3**

> **Whose ask this really is.** R1.3 is fully answered by C1, C3 and C4 without it. C2's customer is
> **R1.2** — *"does the proposed graph-string representation provide benefits in terms of uniqueness,
> **expressiveness**, computational efficiency, scalability, or downstream learning performance?"* —
> and the **AE.3** comparison table the Area Editor endorsed in their own voice, whose
> *expressiveness* row otherwise reads "blind to labels" with no number behind it. It happens to also
> answer R1's concrete scenario, which is why it sits in this file.

R1's concrete scenario is: identical topology, different chemistry, therefore indistinguishable. The
canonical string is a complete invariant of topology within a directedness class (Thm 2.12), so two
graphs are indistinguishable to IsalGraph **iff** they share a canonical string. Per dataset:

- `G` retained graphs, `S` distinct canonical strings;
- fraction of graphs sitting in a collision class whose members differ in label multiset —
  **R1's scenario, counted**;
- one worked example: two molecules, same string, different formula.

**Cost**: one pass over the 19,670 canonical strings T-06 already produces. Minutes.
**Placement**: two columns in the dataset-properties table plus one sentence. No new section.

Decisive either way. Small fraction → the limitation is bounded and we say so with a number. Large
fraction → we have quantified a real expressiveness ceiling, which is a stronger paper than one that
concedes it in prose. **Always state the corpus the fraction is measured over** — the same discipline
`.claude/CLAUDE.md` invariant 6 imposes on the directedness collision rate.

### C3 — A label column in the dataset table

`none` / `categorical` / `continuous` / `categorical + continuous`, per dataset. Answers AE.4b
directly and fixes **E6** in the same stroke: `conclusion.tex:70` and `:81` claim labels are "present
in all five benchmark datasets", which is **false for LINUX**. Any response quoting that sentence
quotes a false one.

### C4 — Future work, with substance — **already written, only needs concreteness**

R1's closing ask is **already satisfied** by `conclusion.tex:70–71` and `:81`. The work here is not to
add a future-work statement but to make the existing one concrete, and to *point at it* from the §4
paragraph so the reviewer sees that it was there.

Replace the current one-line statement with the concrete extension: a labelled alphabet `Σ × L`,
under which S2G stays total and the canonical construction is unchanged. **Conditional on T-07**: if
`github.com/icai-uma/IsalChem` confirms a labelled instruction alphabet, cite it as precedent and add
a row to the R3.1 delta table — *labelled instructions: present in [29] for molecules; deliberately
not carried into IsalGraph's generic-topology redesign; the designated extension*. **Do not write
that row until T-07 confirms it.**

---

## 4. Tier 2 — logged during T-06, paper inclusion decided later

Per the standing rule that we log everything the runs can cheaply produce and decide afterwards. None
of the following is committed to the manuscript; all of it is cheap enough that not having it would
be the more expensive mistake. **Subject to the §0 PI decision, due 2026-08-18.**

| # | Logged quantity | If it is ever needed |
|---|---|---|
| L1 | **Label surplus** `Λ = GED_lab − GED_topo`, bracketed, on the labeled datasets. Provably `Λ ≥ 0`: topology-only is the labeled problem with substitution costs set to zero, a relaxation over the identical set of edit paths | quantifies the blind fraction of a labeled task if a reviewer presses in round 2 |
| L2 | `ρ(d_Lev, GED_lab)` beside `ρ(d_Lev, GED_topo)`, same pairs, same graph-level resamples | the direct measure of what label-blindness costs, in ρ |
| L3 | Bracket for Λ: `max(0, LB_lab − UB_topo) ≤ Λ ≤ UB_lab − LB_topo`. The `max(0, ·)` is not cosmetic — it is where the proven sign enters | reporting a negative lower bound for a provably non-negative quantity is the kind of detail R3 checks |

**Implementation**: one extra `set_edit_cost` call. `CONSTANT` with
`edit_cost_constant = [1, 1, 1, 1, 1, 1]` (unit substitution) instead of D6's `[1, 1, 0, 1, 1, 0]`.
The discrete label metric is a metric, so `GED_lab` stays a metric and `statistics.md` D6's
justification 1 survives unchanged.

**Letter is a separate arm if L1–L3 are ever run.** IAM Letter carries continuous `(x, y)`
coordinates (D7 — R1 is wrong that IAM is unlabeled), for which a discrete substitution cost is
meaningless. It would use the published `LETTER` model with a Euclidean node-substitution cost, in
its own row, never pooled.

> **If any of L1–L3 reaches the paper, this sentence goes with it**: the primary correlation study is
> topology-only under one cost model (D6) across all ten datasets — that is what retires R3.5b — and
> the label arm is a separate, explicitly labelled diagnostic in which heterogeneity is the object of
> study rather than a defect in a pooled figure. Without it, a reader just told "one cost model
> throughout" reads the label arm as backsliding.

**Cost of logging all three: ≈ 0.3 core-hours**, against T-03's 1,000–1,650.

---

## 5. Why we do not build a labelled variant

R1's request, taken literally, is not satisfiable inside this revision, and the reason is structural
rather than budgetary.

1. **The alphabet is nine characters.** `Σ = {N,n,P,p,V,v,C,c,W}` (`methodology.tex:72`). Nothing in
   it carries a label. Supporting labels means `Σ × L` or new instructions — a different instruction
   set, hence a different S2G, a different G2S, a different canonical form and a different
   completeness proof.
2. **It changes the paper's compactness result.** `B_Isal(w) = L log₂ 9`
   (`computational_experiments.tex:157–160`) depends on `|Σ| = 9`. A labelled alphabet changes the
   bit accounting, so Claim A would have to be re-derived and re-measured.
3. **Theorem 2.12 would have to be reproved.** The complete-invariant result is stated over the
   current alphabet and the structural triplet. It is also the contribution R3.1 identifies as the
   paper's principal new one — reopening it in revision is the last thing this manuscript should do.
4. **The paper already says so.** `conclusion.tex:71` calls attributed-graph support a
   *prerequisite* for molecular applications, and `:81` an *open problem*. We are not conceding
   something new; we are pointing at a position the submission already holds.

**Response-letter framing**: R1's suggestion is correct and is the natural next step, which is why
the submission already names it as future work in two places. The revision makes it concrete
(C4) and quantifies what it would buy (C2), rather than attempting a new instruction set under a
21-day deadline.

---

## 6. Where the density half went

v1.0 also proposed a label × density decomposition. **That belongs to AE.1 and R1.3a, not here**, and
it now lives in `statistics.md` §7 (stratification by node count, density, mean degree, symmetry) and
`plan.md` §8 (the within-AIDS density stratification, which can refute `conclusion.tex:30–36`).

The one thing worth keeping from it: the locked cohort **breaks the label × density confound by
construction**, and that is worth one sentence in §4 because the submitted study could not do it.
Categorical labels now appear at density 0.094 (Mutagenicity) and 0.328 (COIL-DEL); the unlabeled
dataset sits at 0.255 (LINUX), between them. In the submission, AIDS was simultaneously the densest,
the only categorically-labeled and the worst-correlating benchmark, so no separation was possible.

---

## 7. Acceptance criteria

1. **C1** appears first in the R1.3 response — R1's objection to the AIDS number is answered on its
   own terms before anything else is offered.
2. **C2** reported for all ten datasets, corpus size stated beside every fraction.
3. **C3** in the dataset table; **E6 corrected** in `conclusion.tex:70` and `:81`.
4. **C4** written; the [29] precedent row added **only if T-07 confirms it**.
5. **L1–L3 computed and stored** with the T-06 artifacts, whether or not they are used.
6. The two senses of "labeling" never share a paragraph (§1).

---

## 8. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v1.0 | Created to close `gap-audit.md` GAP-2 |
| 2026-08-11 | **v2.1** | Re-read R1.3 in full after the author asked why labels are being addressed at all. **Two of R1's three asks are already satisfied** by `conclusion.tex:70–71` and `:81`; the only gap is the *"impact on the reported results"* clause, which is one paragraph in §4. The comment's opening sentence — *"the discussion of the experimental results is rather overlooked"* — shows the target is the **density attribution**, with labels as the illustration. C1 gains a second, **empirical** arm: IAM Letter discards the class-defining `(x, y)` coordinates and correlates at ρ ≈ 0.93 while AIDS discards atom types and correlates at 0.349, so the ordering is the reverse of what label loss predicts. **C2's driver reassigned from R1.3 to R1.2 / AE.3**, where the *expressiveness* row needs a number; C4 marked as already-written |
| 2026-08-11 | **v2.0** | **Rescoped after author review.** v1.0 proposed a new experimental section for a comment that asks for a *discussion*; R1.3's modal and `conclusion.tex:81`'s existing future-work statement both establish that as overreach. Committed set reduced to four low-cost items (C1–C4); the label-aware GED arm demoted to **logged, inclusion decided later** (L1–L3); the density material moved to `statistics.md` §7 and `plan.md` §8 where its drivers (AE.1, R1.3a) live. New §1 — the manuscript never claimed label handling, verified against the sources — and §5, why a labelled variant is a different paper |
