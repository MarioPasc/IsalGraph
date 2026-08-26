# T-07 — article notes

**Closed** 2026-08-26. Ordered by consequence: items that change what the paper may claim first.
Every number names whether it was **measured by T-07**, **inherited from a plan file**, or
**predicted and never checked**.

Archive:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-07-prior-work-delta/`

---

## 1. The contribution claim is safe, and it is now evidenced — **T-20 (§2.3, §1), T-14 (R3.1a)**

**Measured by T-07**, over the complete text of both predecessors:

| Term | [28] | [29] |
|---|---|---|
| `theorem` | **0** | **0** |
| `proof` | **0** | **0** |
| `lemma` | **0** | **0** |
| `proposition` | **0** | **0** |
| `complete invariant` | **0** | **0** |
| `graph edit distance` | **0** | **0** |

**Neither predecessor contains a single formal result.** More precisely:

- **[28]** — its canonical string is *defined as the greedy encoder's output* (`28.txt:150`,
  `:165–166`) on an adjacency matrix which by the paper's own construction presupposes *"a complete
  order … on the N elements (vertices) in V"* (`28.txt:89`). It is canonical **per matrix**, not per
  isomorphism class. The paper never claims otherwise — and never states the restriction either.
- **[29]** — relabelling invariance is argued in **three sentences of prose, in one direction only**,
  resting on an explicit assumption: *"The obtained molecular graph may be assumed to be unique for a
  given molecular species."* (`29.txt:145`). The converse is never stated in any notation, and **no
  experiment measures uniqueness or a collision rate** — so the property is not even claimed
  empirically.

**What this licenses**: *"We prove what [29] asserted"* is defensible. **What it forbids**:
*"[29] proves X and we extend it"* would be rejected by anyone who has read [29].

**Where it lands**: the §2.3 sufficiency paragraph, already drafted at
`artifacts/sufficiency_paragraph.tex` (145 words).

---

## 2. Tab. 3 — composition, and why it understates — **T-20 (§2.3)**

**Measured.** Nine components: **3 inherited, 5 modified, 1 new.**

| Verdict | Components |
|---|---|
| **Inherited** | traversal pointers; every string decodes; normalisation criterion |
| **Modified** | execution state; instruction alphabet; encoder; locality reference distance; domain |
| **New** | isomorphism invariance of the canonical form |

The attribution rule was frozen and committed as **`b300581` before any source was read**, with five
tie-breaks all resolving toward the conservative reading — chiefly *generalisation is modification,
never novelty*. Two consequences worth defending out loud if challenged:

1. **The metric corollary (Cor. 2.13) was folded into the theorem row**, not given a second "new".
   Printing a result and its corollary as two novelties is the inflation R3.1a is probing.
2. **No complexity row**, although [28] has only string-length asymptotics and [29] has **no
   complexity analysis at all**. The numbers belong to T-13 and would create a second place to keep
   in sync.

**Reproduction parameters for the page measurement** — a page figure is a property of the setup, not
of the table: `elsarticle[review,times,number]`, `\textwidth` 345 pt, `\textheight` 550 pt,
`\footnotesize`, `\arraystretch 1.15`, `\linespread{1}`. **368.4 pt = 0.67 p.**

> 🔴 **`\linespread{1}` inside the float is load-bearing and must not be removed.** The `review`
> class option sets `\@blstr{1.5}` (`elsarticle.cls:73`), stretching every table line by 1.5×.
> Without the reset the same table is **over a full page**. Measured, not estimated.

---

## 3. D19 — RESOLVED, both halves — **T-14 (R3.1, R3.2), T-20**

`verified-discrepancies.md` recorded D19 as **UNVERIFIED** because nobody had opened the sources.
Both of R3's content claims are now checked against the primary text.

### D19a — [28]'s Transformer classification: **CONFIRMED, and R3's phrase over-sells it**

Encoder-only Transformer, 3-way graph classification, 8 configurations
(`M ∈ {2,3} × H ∈ {4,16} × F ∈ {128,256}`), 10-fold CV, Adam lr 0.001, batch 64, 100 epochs,
4 × A100 40 GB. **All measured by T-07 from `28.txt`.**

- Dataset is **synthetic and purpose-built**: *"The synthetic dataset that has been tailored to test
  our approach is detailed next."* (`28.txt:847`). 3,000 samples, 1,000 per class, geometric graphs
  of roughly 12 nodes. **No public benchmark.**
- **One baseline**: row-major binary flattening of the adjacency matrix. **No graph-learning baseline
  at all**, despite the introduction surveying GCN/GAT/GIN.
- **No numeric result appears anywhere in the text.** Eight figures, no results table, no error
  bands, no seed, no significance test.
- The abstract's speed claim is bounded by the plotted axis spans at **under ~1.5 %**, and the paper
  concedes *"The differences are not very significant,"* (`28.txt:926`).

### D19b — [29]'s "LSTM model": **CONFIRMED but materially under-described**

- **Two families, not one**: *"the LSTM and GRU models have been employed, with embedding dimensions
  4, 8, 16, and 32; and hidden sizes 8, 16, 32, and 64."* (`29.txt:371`) — 16 configurations per
  family per notation.
- **The task is not classification.** It is **masked / random-position token prediction**: *"The task
  is to predict a token, randomly chosen from a string representing a ZINC molecule in the notation
  at hand."* (`29.txt:371`). 10,000 ZINC molecules, 80/20 split, 1,000 epochs, against SMILES,
  SELFIES and InChI.

**Verdict on the reviewer, stated neutrally**: both claims are true in kind, so **R3's accuracy
record is intact** and should be treated as such in the letter.

---

## 4. 🔴 Handoff to T-14 — the pre-emption T-07 no longer discharges — **T-14 (§6.3, R3.2)**

**PI decision 2026-08-26**: the sequence-model row is dropped from Tab. 3, which **overrules**
`corrections.md` §4's instruction to *"pre-empt the reading inside the table itself"*. Tab. 3 is
architectural only.

**Nothing T-07 produced now discharges that pre-emption.** §6.3 must carry it. The measured content,
so it can be written from fact rather than from R3's paraphrase:

| | [28] | [29] IsalChem |
|---|---|---|
| Model | encoder-only Transformer, 8 configs | **LSTM *and* GRU**, 16 configs each |
| Task | 3-way graph classification | **masked / random-position token prediction** |
| Data | **synthetic**, 3,000 samples, ~12-node graphs | 10,000 ZINC molecules, 80/20 |
| Baselines | **one**, row-major binary flattening; no graph baseline | SMILES, SELFIES, InChI |
| Reported | **no numbers in text**; 8 figures, no seed, no test | loss / correct % / validity tables; no seed, no repeats, no test set, no significance test |

**The strongest honest ground for §6.3**: *neither predecessor ran a downstream graph-learning
evaluation on a real benchmark.* Replicating either would not answer R3.2 as posed.

**Do not** import this into the §2.3 sufficiency paragraph — `prose.md` §2's red line, and the
paragraph as drafted is clean of it (checked).

---

## 5. What is NOT claimable

Read this before writing anything into §2.3 or the letter.

1. **Do not claim [29]'s normalisation searches only the starting atom.** R3's phrase *"exhaustive
   shortest-then-lexicographic normalization"* is **half verifiable**. The ordering criterion is
   dual-sourced and confirmed. **"Exhaustive" is not sourceable**: [29] asserts it (`29.txt:145`) but
   the CC-BY text **strips all three algorithm listings**, so what Algorithm 2's steps 7 and 9 range
   over cannot be read. The public implementation enumerates the starting heavy atom only, with one
   greedy `set.pop()` and zero backtracking — **but the code is not the paper**, and T-07's frozen
   rule is that the paper governs the printed cell. Settling this needs the publisher PDF.
   **Nothing printed depends on it.**
2. **Do not print any criticism of [29]'s implementation.** T-07 found a real defect in the IsalChem
   repository's `compare()` — every `return False` is nested inside `if verbose:`, so the default
   call returns `True` unconditionally (`isalchemutilities.py:556–581`). It is a **co-authored prior
   work**, it has no bearing on Tab. 3, and it is out of scope. PI note only; see `REPORT.md` §4.
3. **Do not quote [28]'s equation (1).** It defines `M_G(i,j) = 0` when an edge is present — the
   complement of the standard convention, contradicting its own equation (7), its encoder and its
   §3.1. Verified on the PDF, not a `pdftotext` artefact.
4. **Do not attribute a node-count ceiling or a disconnected-graph restriction to [29].** The paper
   is **silent** on both. The three facts that *do* carry the "generic topology redesign" claim are:
   the container is a hydrogen list; the seed state is H₂; insertion degree is a per-element
   constant.
5. **Do not cite [29] for a correlation statistic.** It reports **none** — `pearson`, `spearman` and
   `p-value` each occur 0 times — yet its Discussion says Levenshtein *"strongly correlates with
   chemical similarity"* (`29.txt:424`). Cite only the qualitative monotone trend, and prefer the
   abstract's hedged wording.
6. **Do not cite [29] as a complexity comparison point.** It states no bound of any kind, including
   for its own normalisation.
7. **The struck agent claim**: an extraction agent reported that IsalChem *"silently downgrades
   aromatic bonds to single"*. **`grep -ci aromatic` over the whole repository returns 0.** The claim
   was struck and appears in no artifact. Recorded so it cannot re-enter.

---

## 6. Provenance and reproduction

| Item | Provenance |
|---|---|
| [28] full text | `git show a23acbf:docs/references/2512_10429v2.pdf`, then `pdftotext -layout`. 12 pp., 1,034 lines. Archived in `sources/` |
| [29] full text | NCBI BioC REST, PMC12344769, CC BY. 212 passages. Archived as `29_bioc.json` + `29.txt` |
| IsalChem source | `github.com/icai-uma/IsalChem`, shallow clone, last push 2025-12-17, 2,499 lines |
| Attribution rule | frozen at `b300581`, **before** any inventory was read |
| Verification | **33 quotes re-checked by the orchestrator, all HIT; 18 absence claims re-counted, all confirmed; 1 agent claim struck.** `VERIFICATION.md` |
| Per-cell anchors | `PROVENANCE.md` |
| Compute | **none**. No cluster, no code, no test-suite impact. T-07's two commits touch one file |

**One number is inherited, not measured by T-07**: the `24,764,422` collision-free pairs cited in the
sufficiency paragraph is **T-06's**, registered as claim C1 in `prose.md`'s frozen claim register
(3,424,764 Suite-1 certified + 21,339,658 Suite-2 at `LB > 0`). If T-06's figure moves, this
paragraph moves with it.
