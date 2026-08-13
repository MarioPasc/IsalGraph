# Labels and attributes — R1.3, AE.4b

**Owner**: T-18 · **Serves**: R1.3a/b/c/d, AE.4b, R1.2b's *expressiveness* axis, E6
**Status**: tiered; **PI decision on effort due 2026-08-18** (S-d), before T-06 launches.

Related: [statistics](statistics.md) · [demands](demands.md) · [corrections](corrections.md) (E6) ·
[schedule](schedule.md) (Tier 2 is cut candidate #1)

---

## 1. The framing that decides everything: we never claimed labels

Verified against the uncommented `.tex` sources:

| Location | Text |
|---|---|
| `computational_experiments.tex:30` | "In all cases, node and edge attributes are **discarded**" |
| `conclusion.tex:70` | "The current formulation operates on **graph topology only**" |
| `conclusion.tex:71` | "Extending IsalGraph to attributed graphs is therefore a **prerequisite** for applications in … molecular chemistry and program analysis" |
| `conclusion.tex:81` | "A second open problem is the incorporation of node and edge labels … **a labelled variant … would permit direct comparison with GED under non-uniform cost functions**" |

**R1.3b is a premise, not a demand** — "the performance degradation on AIDS **may come from** the loss
of label information" is declarative and the letter contains no imperative there. It licenses no work
of its own; every deliverable below is owned by **R1.3c**, **R1.2b** or **AE.4b**.
**AE.4b is the requirement-modal owner** and is why Tier 0 is not optional.

---

## 2. The four tiers

| Tier | Content | Effort | What it buys | What it risks |
|---|---|---|---|---|
| **0 — not optional** | **C1** rebuttal · **C3** label column + **E6 fix** · **C4** future work made concrete | ~2 h of writing | R1.3 answered; AE.4b answered; a **false sentence** corrected | nothing. C3 is a factual correction we owe regardless |
| **1 — recommended** | + **C2** topological collision count | ~3 h: one pass over strings T-06 already emits, plus two columns and a sentence | its driver is **R1.2 / AE.3**, not R1.3 — the *expressiveness* row of the AE-endorsed table needs a number, not the phrase "blind to labels" | a large fraction is an uncomfortable result. Better found by us than by R3 in round 2 |
| **2 — proposed** | + **L1–L3** computed and **stored, not written up** | ~0.5 d wiring + **0.3 core-hours** | round-2 insurance: if R1 presses, the answer exists and needs no new run under a shorter deadline | half a day for material that may never be used. **No reviewer asked for it** |
| **3 — declined by default** | promote L1–L3 into a results subsection | ~1 d + **≈ 1 page** | quantifies the label-blindness cost in ρ | costs a page we do not have; introduces a **second cost model** into a revision whose headline fix is **one** cost model, so it reads as backsliding on R3.5b unless carefully framed |

**Recommendation: Tiers 0–1 committed; Tier 2 conditional on T-03 landing early; Tier 3 declined.**
Tiers 0–1 are hours and answer the comment. Tier 2's half day sits one day before T-06 launches on a
board that does not fit — see [schedule](schedule.md).

**Counter-case, stated fairly**: R1.3's modal is "would strengthen"; the manuscript already names this
as future work twice; and every hour here is an hour not spent on T-20. **A PI who reads the review as
"answer it in prose and spend the time on the recompute" is making a defensible call, and Tier 0 alone
is a complete, honest answer to R1.3.**

---

## 3. Tier 0–1 deliverables

Total cost: one table column, one paragraph in §4, one paragraph in §5.

### C1 — The rebuttal, two arguments, both free

**Logical.** Both sides of the reported correlation are topology-only: the GraphEdX AIDS matrix is the
topology-only release (`computational_experiments.tex:52`) and the string encodes topology. ρ on AIDS
is a functional of the topology-only graph collection alone, so **label loss cannot have determined
it**. It survives the recompute unchanged, because D6 is also topology-only.

**Empirical, out of the submitted paper's own Table 3.** R1's premise that "IAM and LINUX are
unlabeled" is **wrong for IAM**: IAM Letter carries continuous `(x, y)` node coordinates — the very
attributes that define the fifteen letter classes — and we strip them at `iam_letter_loader.py:4`, `:60`.

| Dataset | What is discarded | ρ (pruned canonical) |
|---|---|---:|
| IAM Letter | `(x, y)` coordinates — **class-defining** | **0.93** |
| AIDS | atom and bond types | **0.349** |

**The benchmark that loses the most semantically load-bearing attributes has the highest
correlation.** If label loss drove the degradation the ordering would be reversed. A counter-example,
not merely an absence of evidence, and it costs nothing to state.

> **Tone.** R1's factual slip about IAM is corrected in passing and without emphasis — "IAM Letter in
> fact carries continuous node coordinates, which we also discard; this strengthens rather than
> weakens the point" — never as a scoring move.

It is a rebuttal, not a dismissal: the paragraph continues into C4, and the density half of R1.3 is
answered by measurement ([statistics](statistics.md) §8).

### C2 — Topological collision count *(the one measurement)*

R1's concrete scenario: identical topology, different chemistry, therefore indistinguishable. The
canonical string is a complete invariant of topology **within a directedness class** (Thm 2.12), so
two graphs are indistinguishable to IsalGraph **iff** they share a canonical string. Per dataset:

- `G` retained graphs, `S` distinct canonical strings;
- fraction of graphs in a collision class whose members differ in label multiset — **R1's scenario,
  counted**;
- one worked example: two molecules, same string, different formula.

**Cost**: one pass over the **16,370** canonical strings T-06 already produces (~~19,670~~, corrected
by T-01's cohort re-derivation — [data](data.md) §1.3). Minutes.
**Placement**: two columns in the dataset-properties table plus one sentence. No new section.

Decisive either way: a small fraction bounds the limitation with a number; a large fraction quantifies
a real expressiveness ceiling, which is a stronger paper than one conceding it in prose.
**Always state the corpus the fraction is measured over.**

### C3 — A label column in the dataset table

`none` / `categorical` / `continuous` / `categorical + continuous`, per dataset. Answers AE.4b
directly and fixes **E6** in the same stroke: `conclusion.tex:70` and `:81` claim labels are "present
in all five benchmark datasets", which is **false for LINUX**. Any response quoting that sentence
quotes a false one.

> ✅ **MEASURED 2026-08-13 (T-01) — this deliverable's data already exists.**
> `cohort_audit.py` records the node and edge attribute names present in each source file before the
> topology-only loader discards them; the table is [data](data.md) §1.5, machine-readable in
> `results/cohort_audit/suite2.json`. **LINUX has neither node nor edge attributes**, confirmed by
> parsing rather than by reading the dataset documentation. Protein carries three node and five edge
> attributes, AIDS (IAM) five and one, GREC three and three.
>
> **C3 is now a transcription job, not a measurement.** T-18 maps the measured attribute names onto
> the four-way `none / categorical / continuous / both` vocabulary and prints the column. Note that
> **AIDS (GraphEdX)** enters already stripped upstream, so its row states the release, not the
> underlying chemistry.

### C4 — Future work, with substance

R1's closing ask is **already satisfied** at `conclusion.tex:70–71` and `:81`. The work is not to add
a future-work statement but to make the existing one concrete and to *point at it* from the §4
paragraph so the reviewer sees it was there.

Replace the one-line statement with the concrete extension: a labelled alphabet `Σ × L`, under which
S2G stays total and the canonical construction is unchanged. **Conditional on T-07**: if [29]
confirms a labelled instruction alphabet, cite it as precedent and add a delta-table row — *labelled
instructions: present in [29] for molecules; deliberately not carried into IsalGraph's
generic-topology redesign; the designated extension*. **Do not write that row until T-07 confirms it.**

---

## 4. Tier 2 — logged during T-06, inclusion decided later

Not committed to the manuscript. **Subject to the S-d decision.**

| # | Logged quantity | If it is ever needed |
|---|---|---|
| **L1** | **Label surplus** `Λ = GED_lab − GED_topo`, bracketed, on the labeled datasets. Provably `Λ ≥ 0`: topology-only is the labeled problem with substitution costs set to zero — a relaxation over the identical set of edit paths | quantifies the blind fraction of a labeled task if R1 presses in round 2 |
| **L2** | `ρ(d_Lev, GED_lab)` beside `ρ(d_Lev, GED_topo)`, same pairs, same graph-level resamples | the direct measure of what label-blindness costs, in ρ |
| **L3** | Bracket for Λ: `max(0, LB_lab − UB_topo) ≤ Λ ≤ UB_lab − LB_topo`. The `max(0, ·)` is **not cosmetic** — it is where the proven sign enters | reporting a negative lower bound for a provably non-negative quantity is the kind of detail R3 checks |

**Implementation**: one extra `set_edit_cost` call — `CONSTANT` with `[1, 1, 1, 1, 1, 1]` (unit
substitution) instead of D6's `[1, 1, 0, 1, 1, 0]`. The discrete label metric is a metric, so
`GED_lab` stays a metric and D6's justification 1 survives unchanged.

**Letter is a separate arm if L1–L3 are ever run.** IAM Letter's continuous `(x, y)` coordinates make
a discrete substitution cost meaningless; it would use the published `LETTER` model with a Euclidean
node-substitution cost, in its own row, **never pooled**.

> **If any of L1–L3 reaches the paper, this sentence goes with it**: the primary correlation study is
> topology-only under one cost model (D6) across all ten datasets — that is what retires R3.5b — and
> the label arm is a separate, explicitly labelled diagnostic in which heterogeneity is the object of
> study rather than a defect in a pooled figure. Without it, a reader just told "one cost model
> throughout" reads the label arm as backsliding.

**Cost of logging all three: ≈ 0.3 core-hours**, against T-03's 1,000–1,650.

---

## 5. Why we do not build a labelled variant

A labelled variant is **a different paper**. It requires a new alphabet, a re-proved canonicalisation
result, a re-run of every experiment under a second cost model, and its own claim scoping — in a
round whose opening comment (R3.1) asks whether the present contribution is substantive enough. The
manuscript already names it as the designated next study; that is the honest position and it is the
one `conclusion.tex:81` already takes.

---

## 6. Acceptance criteria

1. Tier 0 is in the manuscript: C1 paragraph, C3 column, C4 concrete future work, **E6 corrected at
   both sites** (`conclusion.tex:70` and `:81`).
2. If Tier 1 runs, the collision fraction is printed **with the corpus it is measured over**.
3. If Tier 2 runs, L1–L3 are logged with the `max(0, ·)` floor and are **not** written up without the
   §4 framing sentence.
4. No response text quotes "labels present in all five benchmark datasets".
