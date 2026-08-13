# Response-letter fragment — AE.1, and the cohort half of R3.5a / AE.4b

**Emitted by**: T-01, 2026-08-13 · **Assembled by**: T-14 · **Register**: academic "we", active voice

---

## AE.1 — the impact of graph size must be clear

> *"The impact of the graph size should be made clear."* (`mail.txt:59–60`)

**Draft response.**

We have extended the study from five datasets to ten, all from the IAM Graph Database, and we report
the size evidence in two regimes rather than one. Under the connectivity and minimum-size filter the
revised cohort holds **16,370 graphs and 21,710,892 pairs**, against 5,350 and 3,897,911 in the
submitted version. The largest retained graph has **98 nodes** against 12, the largest per-dataset
mean rises from 10.6 to **31.7 nodes**, and the density range widens from 0.218–0.607 to
**0.094–0.607**.

The two regimes exist because the *reference*, not the encoding, imposes the ceiling. Exact graph
edit distance costs 36.9 s per pair at twelve nodes and grows by roughly a factor of five per added
node, so no public benchmark supplies it beyond that scale. IsalGraph's canonical encoder has no such
ceiling: 122 µs at twenty nodes, 3.9 ms at a mean of thirty-two, and no timeout to ninety-six. Below
twelve nodes we therefore compare against exact GED; above it we compare against a **proven bracket**
whose two ends are published lower and upper bounds, and we report the relative bracket width as a
function of node count so that a reader can separate degradation of the representation from
degradation of the reference.

Every count above is re-derived by a script released with the paper, and both sides of every filter
are reported. The connectivity requirement discards graphs that are on average larger than those it
keeps — by 1.92× on Mutagenicity, 2.27× on AIDS, 1.58× on Protein, and by 1.23–1.32× on the three
Letter sets already used in the submitted version. Two datasets show no such bias (GREC 1.01×) or the
reverse (AIDS from GraphEdX, 0.95×). We give the retained and discarded means and maxima side by side
in Table X and state the resulting scope limitation in Section 5: the reported mean size of about
thirty-two nodes is the mean of the connected subsample.

*(~300 words. Trim target under page pressure: the encoder timing sentence, which duplicates §4.2.)*

---

## AE.4b — fully labeled versus partially labeled datasets

The dataset table gains a **label-content column**, populated by parsing the source files rather than
from the dataset documentation: none, categorical, continuous, or both. It records that IAM Letter
carries continuous `(x, y)` node coordinates, that Mutagenicity carries a categorical node attribute
and a categorical edge attribute, that Protein carries three node and five edge attributes, and that
**LINUX carries none**. All are discarded by the topology-only encoder, which the paper states, and
the column makes the extent of that discarding visible per dataset instead of in a single sentence.

**This corrects a false statement in the submitted version** (self-found defect E6): Sections 5 and 6
assert that labels are present in all five benchmark datasets, and LINUX has neither node nor edge
attributes. Owner: T-18 / T-12.

---

## R3.5a — the first rungs of the pair-accounting ladder

`raw → connected` is measured per dataset with its retention percentage (51.4 %–100 %), together with
the discard broken down by reason: below minimum size, above the node ceiling, disconnected. The
later rungs (`GED-available → GED > 0 → Lev > 0 → analysed`) are filled by T-05 and T-06; the ladder's
definition is frozen in [statistics](../plan/statistics.md) §10, and T-03's fragment
(`T-03-letter-R3.5a.md`) covers the exact-GED half.

---

## One disclosure we make rather than be caught on

COIL-DEL ships 7,200 graph files, of which the benchmark's own split index defines **3,900** — one
hundred classes of thirty-nine graphs each. We use the 3,900. The remaining files carry no class
label, and including them would enlarge the pair count while making 46 % of the largest dataset
unclassifiable. **The earlier version of this manuscript's cohort figures used the larger
enumeration**; the correction reduces the reported pair count and changes no size, density or
correlation claim.

---

## Provenance

| Claim | Source artifact |
|---|---|
| 16,370 graphs / 21,710,892 pairs | `results/cohort_audit/suite2.json`; [data](../plan/data.md) §1 |
| 5,350 / 3,897,911 submitted | `export_graphs.py` asserted cohort; reproduced by `cohort_audit.py` |
| `n_max = 98`, `n̄ = 31.68`, density 0.094–0.607 | [data](../plan/data.md) §1 |
| Discard ratios 1.92 / 2.27 / 1.58 / 1.32 / 1.31 / 1.23 / 1.01 / 0.95 | [data](../plan/data.md) §1.4 |
| Label content per dataset; LINUX unlabelled | [data](../plan/data.md) §1.5 |
| COIL-DEL 3,900 = 100 × 39, 3,300 unlabelled | [data](../plan/data.md) §1.3; decision 27 |
| Exact GED 36.9 s/pair at n = 12; ×5 per node | [data](../plan/data.md) §4 |
| Encoder 122 µs at n = 20, 3.9 ms at n̄ = 32, 1.1 s at n = 96 | [data](../plan/data.md) §4 |
