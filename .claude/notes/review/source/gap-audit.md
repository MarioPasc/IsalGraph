# Gap audit of the revision plan — 2026-08-11

**Scope**: every demand in `mail.txt` (EiC, Area Editor, R1, R3) and every self-found defect
(`verified-discrepancies.md` D1–D20, E1–E12), checked against `plan.md` v0.5, `data.md` v1.0 and
`statistics.md` v2.0. This file records **what is missing and what is wrong**. Fixes land in
`plan.md` §0.5 (traceability matrix), §7 (tickets) and §13 (schedule), and in the three companions
`labels.md`, `manuscript.md`, `statistics.md`.

> **Before using this file to scope work, read `README.md` §"Read this before scoping any response —
> the R1.3 lesson".** This audit was built to find *under*-coverage. The R1.3 episode showed the
> opposite error is just as expensive: a comment whose bulk discusses one topic was scoped as a new
> experimental section when its operative clause asked for a paragraph, and two of its three asks
> were already satisfied in the submitted manuscript. That file carries the six-question test and
> names the comments where the same error is most likely — **R3.5b and R3.6a first**.

Two verdict classes:

- **GAP** — a demand with no owner: no decision, no ticket, no manuscript artifact.
- **FLAW** — a decision that *is* locked but is wrong, unsupported, or infeasible as written.

Nothing here is a matter of taste. Every item either changes what gets computed, changes a number
that will be printed, or is a compliance requirement the EiC checks independently of the reviewers.

---

## 0. Summary

| Class | Count | Severity |
|---|---:|---|
| **GAP** — demands with no owner | **10** | 3 blocking, 5 major, 2 minor |
| **FLAW** — locked decisions that are wrong or infeasible | **15** | 1 blocking, 6 major, 8 minor |

## 0.1 Resolution log — author review, 2026-08-11

| Item | Resolution |
|---|---|
| **MF1** | **Confirmed, and it blocks nothing.** Re-measured with `audit_dropped.py`: the cohort, its counts and its pair totals are unchanged, and the COIL-RAG / Fingerprint / Web drop decisions all survive on connected-set numbers (COIL-RAG kept 7,100, n̄ 3.02, density 0.936; Fingerprint kept 2,056, n̄ 5.03). The correction is to *descriptions*, not to *data*. **S-a is a wording decision** |
| **MF6 / MF7** | **Superseded by `competitors.md`.** The padded-Hamming convention was a unilateral call and is withdrawn; **T-04a** now selects each representation's distance by measurement, with the rule fixed in advance and ties broken on cost — never on correlation with GED |
| **MF8** | **Resolved — T-16 rejected** (author). No reviewer asked for `wl_pruned_canonical`; it was an IsalSR/IsalHG transfer. The WL *measurement* is retained inside T-13, where it answers R3.7d. 3–4 days returned |
| **GAP-2** | **Rescoped.** `labels.md` v1.0 proposed a new experimental section; verified against the sources, the manuscript **never claimed label handling** and R1.3 asks for a *discussion*. Now four costed tiers with the effort call referred to the PI, due 2026-08-18 |
| **MF9** | Accepted — E7 moved to T-11, ahead of the trim |
| **MF16** | Accepted — Friedman/CD restricted to the ten-dataset regime |
| **J5 / J6 / J7** | D13, D14 and D15 were reported unclear; rewritten in `statistics.md` with plain-language framing and worked numbers. The decisions themselves are unchanged |
| **MF17** *(new)* | See below |

### MF17 · `data.md` §2.2.1's Fingerprint row is unreproducible — **minor**

Surfaced while re-checking the drop decisions. §2.2.1 reports Fingerprint at **67.2 %** retention with
a discarded mean of **11.56** nodes. Measured: **51.4 %** (2,056 of 4,000 — which is what §2.1 itself
reports) and **5.98**. The internal check holds on the measured values:
`(2056 × 5.03 + 1944 × 5.98) / 4000 = 5.49`, matching the raw mean.

**Every other row in that table reproduces exactly** — GREC, Mutagenicity, Protein and AIDS-IAM to
the decimal; Letter LOW/HIGH differ only by the `n ≥ 1` vs `n ≥ 2` threshold. Origin of the
Fingerprint figures unknown. **Not load-bearing** — Fingerprint is dropped — but §2.2.1 is the table
that justifies GREC's inclusion and quantifies the size-bias problem, so it carries a warning and no
number from that row may be quoted.

---

The three blocking items, in order of when they bite:

1. **MF11 / T-23** — the Picasso `fscratch` **file-count quota is already exceeded** with a 7-day
   grace that expires ≈ 2026-08-18. T-03 checkpoints frequently; a run started near the hard limit
   fails partway. Nothing in the plan owns clearing it, and T-03 is the long pole.
2. **MF1** — `data.md` §2.1's size statistics are **raw-set numbers presented as connected-set
   numbers**, and the error is load-bearing: decision 12 (drop TUDataset) is justified by
   "Mutagenicity already reaches n = 417", but the 417-node graph is **disconnected and discarded**.
   The retained maximum is **98**.
3. **GAP-5 / T-20** — **no ticket rewrites the manuscript.** T-06 computes, T-14 writes the letter,
   T-15 trims pages. Sections 3.1, 3.2, 3.3, 4 and 5, the abstract, the highlights and the graphical
   abstract all change completely and have no owner.

---

## 1. GAPS — demands with no owner

### GAP-1 · AE.3's comparison table is not a manuscript artifact — **major**

AE.3 is the one request the Area Editor singles out and endorses in their own voice:

> reviewer 3 has asked for a detailed side-by-side comparison of **existing graph representations**
> with the proposed one, which fairly and completely identifies the **properties, strengths, and
> limitations of each** — this will help focus the presentation of work in the paper, and clarify
> the contribution of the work. (`mail.txt:63–64`)

`plan.md` §4.3 row (d) maps AE.3 to the **[28]/[29] delta table**. That is R3.1 and R3.7b, a
different object: it decomposes *our own* prior work into inherited / modified / new. AE.3 asks for
a comparison against the **field**. The plan's §4.2 table is an *engineering* table — it carries an
"Effort" column in developer-days — and is not a paper artifact.

R1.2 names the axes the table must carry:

> does the proposed graph-string representation provide benefits in terms of **uniqueness,
> expressiveness, computational efficiency, scalability, or downstream learning performance**?
> (`mail.txt:77`)

The fifth axis is the one we lose: §6 declines the sequential-model experiment, so the honest entry
is "not evaluated". **That must be a printed row, not an omission** — R1 asked the question directly
and a table that silently drops the column reads as evasion.

→ **T-17**. Owns the manuscript table, its axes, and the softening of `introduction.tex:33` /
`conclusion.tex:74` that it licenses (plan §10 B6).

### GAP-2 · The label confound has no experiment — **major**

This is AE.4's second named sub-issue ("fully labeled, vs. partially-labeled", `mail.txt:66`) and
the whole of R1.3, which `reviewer-1.md` calls "the single strongest criticism in the review".

`plan.md` §8 answers only the **density** half. Its answer to the label half is a rebuttal:

> the GraphEdX GED is itself topology-only, so both sides of the correlation are label-blind and a
> label-loss mechanism cannot explain that number.

The rebuttal is sound **for the submitted number** and must be kept. It is not sufficient, for three
reasons the plan does not address:

1. Decision 3 + T-03 **recompute all GED ourselves**. The "GraphEdX is topology-only" premise stops
   being a property of someone else's data and becomes *our* modelling choice, which we then have to
   defend rather than cite.
2. The locked cohort adds **five IAM datasets that all carry labels** — GREC, AIDS-IAM, COIL-DEL,
   Mutagenicity, Protein. The plan never says what happens to those labels. Stripping them silently
   on five new datasets, in a revision responding to a label criticism, is the worst available
   option.
3. GEDLIB ships the **published per-dataset IAM cost models** with non-zero substitution costs
   (`CHEM_1`, `CHEM_2`, `PROTEIN`, `LETTER`, `GREC_1/2`, `FINGERPRINT` — verified installed,
   `data.md` §7.5). A **label-aware GED** is therefore available at the cost of one extra
   `set_edit_cost` call. The plan mentions these models only as a §7.3 sensitivity analysis and
   never connects them to R1.3.

The decisive experiment costs ≈ 0.3 core-hours. Design in **`labels.md`**.

→ **T-18**.

### GAP-3 · EiC.a2 and EiC.b have no owner and no acceptance criterion — **major**

> Missing references from last and current year most probably would mean you are missing the state
> of the art and the revision process can be delayed being asked to update it. (`mail.txt:126`)
> please make sure you cite RECENT work from the field of pattern recognition not only the Pattern
> Recognition journal. (`mail.txt:128`)

Current state (`00-editor-and-decision.md`): the newest third-party citations are **2024**;
**nothing from 2025 or 2026 except the authors' own [28] and [29]**. `plan.md` §5.4 allocates
"2–3 slots" for this and calls it "the weakest current position". A slot allocation is not a ticket:
no one is assigned, no search protocol exists, and there is no acceptance criterion.

`00-editor-and-decision.md` also records an explicit open item that the plan never picks up:

> **I did not audit venue composition of the bibliography** — establishing which of the 43
> references count as "pattern recognition" is a judgement call, not a lookup. Flagged as an open
> item.

→ **T-19**. Acceptance criteria and the venue audit are specified there.

### GAP-4 · The five axes of R1.2 are never committed to — **minor** (subsumed by T-17)

Covered by GAP-1. Recorded separately because R1.2 is the comment, AE.3 is the endorsement, and the
response letter must answer both by pointing at one table.

### GAP-5 · No ticket rewrites the manuscript — **blocking**

The ticket board covers computation (T-01…T-06, T-16), reading (T-07), related work (T-08), two
figures (T-09, T-10), error fixes (T-11), claim scoping (T-12), the complexity section (T-13), the
letter (T-14) and the page trim (T-15).

Nothing owns the prose that changes because the numbers changed:

| Section | Why it must be rewritten |
|---|---|
| §3.1 Benchmark Datasets (`computational_experiments.tex:14–58`) | 5 datasets → 10; new property table with n̄, density, connectivity retention, discarded-subset statistics |
| §3.2 Evaluation Protocol (`:90–233`) | the entire statistical protocol is replaced (`statistics.md` D1–D15) |
| §3.3 Implementation (`:234–`) | the C++ engine and GEDLIB did not exist at submission; every timing now comes from a different implementation than the one described |
| §4 Results (all of `results.tex`) | every number is re-derived |
| §5 Conclusion (all of `conclusion.tex`) | every number, plus B1–B6 |
| Abstract (`main.tex:106–126`) | B1, B4, B6, §6 |
| Highlights, graphical abstract (`main.tex:129–141`) | submitted separately; they restate the claims being scoped down |

→ **T-20**, spec in **`manuscript.md`**.

### GAP-6 · The Implementation section, the engine, and the artifact release — **major**

R3's opening credits the "**open implementation**" as a strength (`mail.txt:83`). The revision
changes it beyond recognition: a C++ engine that postdates submission, a GEDLIB dependency, five new
datasets, six competitor backends, a new statistical stack. `statistics.md` §8 requires "software
and library versions, including GEDLIB" and no one captures them.

Two specific hazards:

- **Every reported timing changes meaning.** The submitted numbers were produced in pure Python; the
  revision reports C++ numbers. Reporting a 23×–1025× faster implementation without saying so, in a
  paper whose §4.2 is entirely about runtime, is the kind of thing R3 catches.
- **The `.so` does not rsync** (`.claude/CLAUDE.md`) and `-march=native` produces SIGILL on part of
  Picasso. Reproducibility instructions that omit this do not work.

→ **T-21**.

### GAP-7 · No formal audit of Theorem 2.12 and Corollary 2.13 — **major**

Plan §10 B2 correctly requires restating Theorem 2.12 within a fixed directedness class and moving
the "deterministic given `w` and the flag" hypothesis out of the proof
(`methodology.tex:643–644`) into the statement. Nobody owns the consequences:

- the proof runs `methodology.tex:639–726` in three steps; changing the hypothesis means
  **re-checking that all three still go through**;
- **Corollary 2.13** (`methodology.tex:728`) asserts the graph distance is an isomorphism-invariant
  metric. It inherits the restriction and no document says so. This matters more than it looks:
  `statistics.md` D6 justification 1 uses Corollary 2.13 to argue the GED reference must be a metric.
  If the corollary is silently weaker than stated, that argument weakens with it;
- `codebase-pointers.md` proposes a directedness-collision regression in `tests/property/`; it is in
  no ticket.

→ **T-22**.

### GAP-8 · The `fscratch` quota blocker is unowned — **blocking**

`data.md` §7.5, verbatim:

> **Still above the 250k soft quota with a 7-day grace.** T-03 checkpoints frequently and a run near
> the hard limit will fail partway, so free more before launching it.

305.8k files against a 250k soft quota and a 400k hard limit. The grace expires ≈ **2026-08-18**,
which is inside the revision window and before T-03 can plausibly finish. The failure mode is
`shutil.Error: [Errno 122] Disk quota exceeded` mid-run, which reads as a code fault.

→ **T-23**, P0, **must precede T-03**.

### GAP-9 · Submission package and Elsevier compliance — **major**

The plan tracks the page limit and the AI declaration. It does not track the rest of what a
resubmission has to carry:

| Item | State | Source |
|---|---|---|
| **Source files, not PDF** | LaTeX present; no ticket owns assembling the package | `mail.txt:22` |
| Generative-AI declaration | commented out at `main.tex:198–202` | E11 |
| Author biographies + photos | commented out at `main.tex:225–245`; Pattern Recognition requires them | `00-editor-and-decision.md` |
| Acknowledgements (funders, SCBI, NVIDIA) | commented out at `main.tex:175–177` | same |
| Highlights, graphical abstract | uploaded separately; restate claims being scoped down | `main.tex:129–141` |
| `graphical_abtract.pdf` | **misspelt filename**, referenced under that spelling at `main.tex:131` | E12 |
| Declaration of competing interest, data availability | not present, not tracked | Elsevier standard |

→ **T-24**, checklist in `manuscript.md` §5.

### GAP-10 · The response letter has no architecture and no disclosure policy — **major**

T-14 is "Response letter, 3 days, depends on all". For 41 numbered demands plus 12 self-found
defects, drafted at the end of a 20-day window, that is a plan to run out of time.

The unmade decision underneath it: **do we disclose E1–E12 to the reviewers?** These are defects
nobody caught — the 473,147-pair gap (E2), density never computed (E1), "labels present in all five
datasets" being false (E6), the printed draft self-correction (E8).

**Recommendation: disclose, in a short dedicated section.** The recompute changes all of these
numbers anyway; R3 checked thirteen of thirteen checkable claims and will check again; and a
volunteered correction is the cheapest credibility we can buy in round 2. But it is the authors'
call — recorded in `manuscript.md` §4 as **PROPOSED, needs sign-off**.

→ folded into **T-14**, spec in `manuscript.md` §4.

---

## 2. FLAWS — locked decisions that are wrong or infeasible

### MF1 · `data.md` §2.1 reports raw-set statistics as connected-set statistics — **blocking**

**Verified independently today** by re-parsing every GXL file
(`scratchpad/audit_recheck.py`, filter `min_nodes = 2` + `nx.is_connected`, matching
`dataset_filter.py::filter_graphs`):

| Dataset | kept | n̄ kept | n max **kept** | n̄ **raw** | n max **raw** | `data.md` §2.1 prints |
|---|---:|---:|---:|---:|---:|---|
| Letter LOW | 1,180 | 4.07 | **7** | 4.68 | 8 | n mean 4.68, n max 8 |
| GREC | 650 | 11.45 | **24** | 11.51 | 24 | n mean 11.51, n max 24 |
| AIDS-IAM | 1,811 | 14.02 | **85** | 15.69 | 95 | n mean 15.69, n max 95 |
| Mutagenicity | 4,040 | 28.53 | **98** | 30.32 | **417** | n mean 30.32, n max **417** |
| Protein | 569 | 31.68 | **96** | 32.63 | 126 | n mean 32.63, n max 126 |

The identity is exact, not approximate. For Mutagenicity,
`(4040 × 28.53 + 297 × 54.70) / 4337 = 30.32` — §2.1's number to the decimal. Same for Letter LOW
(4.679 → 4.68), GREC (11.51), Protein (32.64 → 32.63).

**Conclusion: §2.1's `n med`, `n mean`, `n p90`, `n p99`, `n max`, `m mean` and `density` columns are
computed over the RAW set, while its `N conn` and `ret. %` columns are over the connected set.** The
table header claims otherwise. `data.md` §0's numbers are correct and reproduce exactly.

**Three locked statements inherit the error:**

1. **Decision 12** (`plan.md` §0) — "TUDataset dropped — **Mutagenicity already reaches n = 417**".
   The 417-node graph is disconnected, so `filter_graphs` removes it. The retained ceiling is **98**.
   The decision may well survive (98 vs 12 is still an 8.2× extension), but **its stated
   justification does not**, and re-affirming it is an author decision, not a silent edit.
2. **`data.md` §2.3** — "This cohort takes the maximum node count from **20 → 417**". Correct value:
   20 → **98**.
3. **`data.md` §4.2** — "Mutagenicity heavy tail (median n = 27, max n = **417**)" as the explanation
   for the 6.7× timing spread. The encoded corpus stops at 98.

→ fixed in `data.md`; decision 12's rationale rewritten in `plan.md`; re-affirmation flagged.

### MF2 · The calibration regime does not overlap the inference regime — **major**

`plan.md` §3.3 and `statistics.md` §6 license the large-n study by calibrating the GED bounds
"where exact GED exists". Two facts make that weaker than it reads:

- exact GED exists only for **n ≤ 12** (`data.md` §3);
- the bounds were validated on **n = 3–9** only (`data.md` §5: "Validation on real pairs, n = 3–9").

The inference regime runs to **n = 98**, with Suite-2 means of 11.45 – 31.68. So the calibration
sample tops out **three to ten times below** the population it certifies, and the plan's own warning
against extrapolation —

> Bipartite GED's error is known to grow with graph size, so a declining ρ at large n would be
> uninterpretable (`plan.md` §3.3)

— applies to the calibration itself.

Bracket **validity** is not at risk: `LB ≤ GED ≤ UB` is proven at every n. **Tightness** is, and
tightness is what the argument rests on.

**Fix — three parts, all cheap, specified in `statistics.md` §6.1:**

1. **Size-stratified calibration ladder.** Run `ANCHOR_AWARE_GED` on a small stratified sample at
   each n from 3 to the feasible ceiling, with a per-pair time budget. Open question 16 already
   proposes benchmarking `ANCHOR_AWARE_GED` against `networkx` A*; this wires it into the design
   instead of leaving it as a nice-to-have. Every node the exact solver buys widens the calibration.
2. **Regress bracket width and ρ-gap on n**, and report the extrapolation with its uncertainty
   rather than asserting transfer.
3. **Report relative bracket width `(UB − LB)/UB` as a function of n across all of Suite 2.** This
   needs no exact GED, is computable on all 40 M pairs, and is the strongest evidence available that
   the reference does not degrade with size. It is the one measurement that directly answers AE.1.

### MF3 · The primary large-n inference rule has no decision threshold — **major**

`statistics.md` §6 fixes a pre-declared rule for the calibration gate (CI excludes 0 **and** point
estimate > 0.05). `plan.md` §7.3's large-n rule has no such rule:

> If ρ(Lev, LB) and ρ(Lev, UB) **agree**, the conclusion is robust … If they **disagree**, the
> bracket is too wide to support a claim at that size and we say so.

"Agree" is undefined. Since this rule governs every number above n = 12 — the entire answer to AE.1
— leaving the threshold to be chosen after seeing the estimates is exactly the practice the
pre-registration in `statistics.md` §6 exists to prevent.

→ **D13** in `statistics.md`: threshold fixed in advance, symmetric with §6.

### MF4 · Censored encodings have no analysis rule — **major**

`data.md` §4.3 measures it: pruned canonicalisation on Mutagenicity graph 3703 (n = 98) **does not
finish in four minutes**, and `|Aut(G)| > 20,000` is the mechanism (§4.4). The locked timeout is
300 s (open question 13). So **some Suite-2 graphs will have no canonical string.**

`statistics.md` D12 requires reporting censoring rates per stratum. It never says what the analysis
*does* with a censored graph. The default — drop it — deletes the graph and every pair containing
it, and it deletes preferentially the **high-|Aut| graphs**, which is precisely the population whose
behaviour the scalability claim is about.

This is the same selection-bias structure as the connectivity discard, which `plan.md` open question
15 flags carefully. The encoding discard is not flagged anywhere.

→ **D14** in `statistics.md`: analysis rule, sensitivity arm, and mandatory reporting.

### MF5 · The bootstrap and Mantel budgets are off by two to three orders of magnitude — **major**

Locked: `statistics.md` D2 — graph-level cluster bootstrap, **2,000 replicates**; D3 — Mantel,
**9,999 permutations**. Budgeted: `plan.md` §2 "Graph-level bootstrap + Mantel — 4–8 core-hours";
`data.md` §7 "5–10".

The arithmetic on the two largest datasets:

| Dataset | graphs | pairs | bootstrap: 2,000 × pairs | Mantel: 9,999 × pairs |
|---|---:|---:|---:|---:|
| COIL-DEL | 7,200 | 25,916,400 | **5.2 × 10¹⁰** | **2.6 × 10¹¹** |
| Mutagenicity | 4,040 | 8,158,780 | 1.6 × 10¹⁰ | 8.2 × 10¹⁰ |
| IAM Letter HIGH | 2,059 | 2,118,711 | 4.2 × 10⁹ | 2.1 × 10¹⁰ |

And that is **one cell**. The confirmatory family (`statistics.md` §10) is IsalGraph versus **each**
competitor, per dataset, over two GED references — with six competitors and ten datasets that is
~120 cells for Claim B, each needing its own bootstrap, plus D7's difference-of-ρ on the *same*
resamples. Spearman also requires **re-ranking inside each replicate**; ranks cannot be precomputed
once and reused, so the per-replicate cost is `O(p log p)`, not `O(p)`.

Nothing in the three documents states the algorithm or the approximation. A protocol whose compute
is under-budgeted by 10²–10³× will be discovered in week 3, on the critical path.

→ **D15** in `statistics.md`: explicit policy — replicate counts scaled per dataset, an
`m`-out-of-`n` graph bootstrap where warranted, within-replicate pair subsampling with a stated
seed, and the resulting compute budget written down.

### MF6 · Hamming distance is undefined for most competitor pairs — **major** · ✅ SUPERSEDED by `competitors.md` / T-04a (§0.1)

`plan.md` §4.2 assigns **Hamming** as the distance for graph6, sparse6, nauty-canonical graph6 and
the adjacency matrix. graph6 encodes `n` in its header and packs `n(n−1)/2` bits, so **two graphs
with different node counts produce strings of different length and Hamming distance is undefined.**

In Suite 1 the node counts run 2–12; in Suite 2, 2–98. Equal-`n` pairs are a small minority. The
comparison the Area Editor explicitly endorsed would therefore be reported on a metric that does not
exist for most of the data — and `plan.md` §4.3's stated expectation ("Hamming on non-canonical
graph6 should correlate poorly … report it either way") would be recording an artefact of
undefinedness as a finding.

→ **fixed in `plan.md` §4.2**: distances redefined per competitor, with the padding convention
stated (embed both adjacency matrices in a common `max(n₁,n₂)` frame, which is exactly the
node-insertion operation the D6 cost model charges) and Levenshtein used wherever the strings are
variable-length.

### MF7 · sparse6 is the strongest competitor and is not identified as such — **major**

Experiment (a) claims compactness. `sparse6` exists specifically to encode **sparse** graphs
compactly, and IsalGraph's compactness advantage is claimed on **sparse** graphs. sparse6 is
therefore the head-to-head competitor for the paper's Claim A, not a make-weight — and `plan.md`
§4.2 lists it in one cell alongside graph6 with no comment.

Two things must be locked before T-06:

1. **The bit accounting must be like-for-like.** `B_Isal(w) = L log₂ 9`
   (`computational_experiments.tex:157–160`) is an *entropy* bound on the symbol stream, not a byte
   count. graph6/sparse6 produce actual printable-ASCII bytes with 6 bits used per byte. Comparing
   an entropy bound against a wire format flatters us. Report **both** conventions for **every**
   method, and say which is primary.
2. **A losing result is publishable and must be pre-committed.** If sparse6 beats IsalGraph on bits
   for sparse graphs, that is the expected outcome — sparse6 is a bit-packed format with no
   reversible-edit-distance property — and the contribution is then correctly stated as
   *canonical **and** edit-distance-compatible*, not *shortest*. Deciding this after seeing the
   numbers is not an option.

### MF8 · T-16 has no publication decision — **major** · ✅ RESOLVED: T-16 rejected (§0.1)

T-16 builds `wl_pruned_canonical`, a new C++ algorithm, at 3–4 days on a 20-day budget. Neither the
ticket nor §7.2 says whether it appears in the paper.

Both answers have consequences the plan has not absorbed:

- **If it is reported**, it is a *new algorithm introduced during revision*. It changes Table 2,
  Table 3, Figure 2 and Figure 4, needs its own claim scoping, and lands in a review round whose
  first comment (R3.1) is about whether the contribution is substantive enough — inviting "is this a
  contribution or a patch?".
- **If it is not reported**, it is 3–4 days of C++ off the critical path in the tightest window of
  the project.

There is a third option the plan does not consider and which is strictly better than either: **use
WL as a measurement, not as a shipped algorithm.** `data.md` §4.4 already has the number — 1-WL
yields 66 classes against the structural triplet's 28 on the graph that hangs, 2.4× finer. That is
one sentence and one table row in the complexity section (T-13), it directly answers R3.7d's request
for a *characterised* worst case, and it costs hours rather than days.

→ recorded as a decision needing sign-off in `plan.md` §7.2, with the measurement-only path
recommended.

### MF9 · E7 is assigned to the page trim and must precede it — **minor**

E7 is float placement: `\floatpagefraction{1}` and `\textfraction{.001}` (`main.tex:66–67`) defer
all three algorithms past the bibliography to pp. 33–35. `plan.md` §9 assigns it to **T-15**, the
page trim.

Fixing float placement **changes pagination**, so it must run **before** the trim, not as part of
it. It also recovers up to three near-empty float pages — which is the single largest page saving
available and materially changes what T-15 has to cut.

→ moved to **T-11**, with the recovered-page estimate feeding `manuscript.md` §3.

### MF10 · The R3.2 decline is undercut by the R3.1 table — **major**

§6 declines the sequential-model experiment and mitigates with §4's comparison. The interaction with
R3.1 is not noted anywhere:

**T-07 produces a table documenting that [28] contains a Transformer classification experiment and
[29] contains an LSTM experiment.** That table is the deliverable for R3.1 and AE.3. It is also, in
the same document, the clearest possible statement that both predecessors evaluated a sequence model
and this paper does not — which is R3.2's exact argument, in our own table.

The decline can still be right. But it needs (i) the two-sentence framing that pre-empts the
reading, and (ii) a **contingency with a decision date**, not a "residual risk" paragraph. Every
ingredient for a minimal classification experiment already exists: canonical strings for 19,670
graphs across ten datasets, **all of which carry class labels** (Letter 15 classes, GREC 22,
COIL-DEL 100, Mutagenicity 2, Protein 6, AIDS 2), and an RTX 4060 that trains a small
character-level Transformer on ~10⁴ short strings in minutes.

→ `plan.md` §6 gains a contingency with a **go/no-go date of 2026-08-22** and a fixed minimal scope.
Still declined by default — this only makes the fallback executable if T-03 finishes early.

### MF11 · There is no schedule — **major**

Ticket durations are given as ranges. Summed at the upper bound: **76.5 days** of work in a
**20-day** window. The plan states a critical path (T-01 → T-01b → T-03/T-05 → T-06 → T-14) but
never lays it on a calendar, and three costs are entirely unbudgeted:

- **Picasso queue time.** The user's own environment notes record offline nodes. T-03 is a 64–128
  core job with a `1-00:00:00` limit; queue wait is not zero and is not in the plan.
- **The quota blocker** (GAP-8), which gates T-03.
- **Rework after a refutation.** `statistics.md` §5 and `plan.md` §8 both say, correctly, that the
  MRM and the AIDS density stratification **can refute the paper's central claims**, and both say
  "run it in week 1". Neither reserves time for the rewrite that a refutation forces.

→ **`plan.md` §13**: a dated calendar with named gates, and an explicit statement of what gets cut
if T-03 slips.

### MF12 · `data.md` Q4 is marked open and is closed — **minor**

`data.md` §0 states the reconciliation is closed; §2.2 says "**Must be reconciled before any count is
reprinted**" and Q4 still lists it as open. Confirmed closed today: `filter_graphs` defaults to
`min_nodes = 2, require_connected = True` (`dataset_filter.py:37–43`), and applying it reproduces
**1,180 / 1,253 / 2,059 / 650 / 1,811 / 7,200 / 4,040 / 569** exactly.

### MF13 · "Drop the vacuous `n_max`" is wrong for AIDS — **minor**

T-01 says "drop the vacuous `n_max`". `n_max = 12` is vacuous for IAM Letter and LINUX and **is not
vacuous for AIDS-GraphEdX**: 819 connected → 769 retained (`data.md` §0). Suite 1 is *defined* by
`n ≤ 12`; the filter is kept there and dropped only for Suite 2. Also `filter_graphs` takes `n_max`
as a **required positional argument**, so "dropping" it is an API change, not a config edit.

### MF14 · The GraphEdX validation gate has no cost-model configuration — **minor**

`plan.md` §3.2 and §7.1 gate T-03 on reproducing ~500 within-split AIDS pairs against the published
GraphEdX matrix. GraphEdX uses **topology-only** costs (zero node operations), so the gate must run
GEDLIB with `edit_cost_constant=[0, 0, 0, 1, 1, 0]` — **not** the D6 production model
`[1, 1, 0, 1, 1, 0]`. §7.3's production-assignment table has no row for it, and running the gate
under the production model would produce a guaranteed mismatch that looks like a solver bug.

### MF15 · Fischer et al. 2015 does not satisfy EiC.b — **minor**

`plan.md` §7.3 asserts that reporting `HED` "serves EiC.b's 'cite recent work from the field of
pattern recognition' **directly**". Fischer et al. is *Pattern Recognition* 48(2), **2015**. EiC.a
asks for "references from **last and current year**". Venue fit is satisfied; **recency is not**, and
recency is the half we currently fail. Cite it for venue balance; do not count it against EiC.a2.

### MF16 · Friedman/CD on five datasets is underpowered — **minor**

`statistics.md` D8 and §4 lock Friedman + Wilcoxon–Holm + critical-difference diagrams, "two
separate diagrams — one for the exact-GED regime, one for the approximate". The exact regime has
**five** datasets. Demšar (*JMLR* 7:1–30, 2006) develops the CD diagram for `N` datasets with `N`
comfortably larger; at `N = 5` the Friedman statistic is conservative and the critical difference is
wide enough to separate almost nothing.

→ report the exact-regime comparison **descriptively** (per-dataset ρ with bootstrap CIs, D7
differences), reserve the omnibus and the CD diagram for the ten-dataset approximate regime, and say
so rather than printing an underpowered omnibus.

---

## 3. What is already correct and complete

Recorded so the audit is not read as a verdict on the plan as a whole. These are covered, locked and
methodologically sound as written:

- **R3.5a/b/c** — the pair-accounting ladder, one cost model (D6), graph-level bootstrap and Mantel.
  D6's four-part justification, and especially the metric/pseudometric argument, is the strongest
  single passage in the package.
- **R3.4a, R3.4c, R3.7e, D1, D2, D20, E5, E6, E8** — every factual correction has an owner and a fix.
- **AE.1 size scaling** — the Suite 1 / Suite 2 split, and the reframing that the ceiling belongs to
  the *reference measurement* rather than to IsalGraph, is the correct answer and it is measured
  rather than argued.
- **The GEDLIB lock (§7.3)** — proven-bound reasoning, the accessor-capability matrix, and the
  three validation gates. The silent-failure traps (`get_lower_bound()` returning 0.00 on
  upper-bound methods) are documented at exactly the level that prevents a corrupted matrix.
- **H4** — measuring that the *lower* bound tracks exact GED better than the upper bound
  (ρ 0.966 vs 0.840) before committing compute is the plan working as intended.
- **`statistics.md` D4 (MRM)** — pre-empting the size-confound attack, with the interpretation fixed
  in advance and an explicit statement that it can refute Claim B.

---

## 4. Change log

| Date | Change |
|---|---|
| 2026-08-11 | Created. 10 GAPs, 16 FLAWs. MF1 verified by independent re-derivation of the full IAM cohort (`scratchpad/audit_recheck.py`); MF12 closed by inspection of `dataset_filter.py:37–43` |
| 2026-08-11 | **Author review.** §0.1 resolution log added. MF1 confirmed but shown to **block nothing** (`audit_dropped.py`); MF6/MF7 superseded by `competitors.md` and T-04a; MF8 resolved by rejecting T-16; GAP-2 rescoped and referred to the PI as four costed tiers; **MF17 added** — `data.md` §2.2.1's Fingerprint row is unreproducible |
