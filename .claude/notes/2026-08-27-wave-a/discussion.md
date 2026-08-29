# discussion — §6 Discussion and limitations

## What the section now says

§6.1 opens on C10 and then makes both of its clauses checkable *inside this manuscript*, which
the frozen wording alone does not: the best compressor is raw `sparse6`, named in §2.1 as failing
the invariance filter at 1 of 50 relabelings, and the best correlator is the Weisfeiler–Lehman
map, which has no message length. Then the like-for-like reading — more compact than min-DFS and
nauty-graph6 above n ≈ 20, correlation bracket-dependent — and then the concession, in the same
paragraph and in three sentences: nauty-sparse6 is shorter at every size above 20, correlates
better on the same strata, and holds the metric-admitting frontier alone. The second paragraph
states the categorical differentiator once: the string is a program, every prefix builds a
subgraph, |Σ| = 9 independent of n against a sparse6 index width of ⌈log₂ n⌉. §2.2 forward-
references this paragraph and now finds it.

§6.2 is the recommendation: report the |n_i − n_j| baseline beside any GED-correlation claim, and
decompose within fixed n. It rests **only** on the field-level negative (the benchmarks are
size-dominated, identically for all seven arms) and never claims our own limitation away.

§6.3 carries seven limitations, each with its cause. The R3.2 concession is one flat sentence
followed by the two predecessors' actual experimental setups, stated neutrally.

## Number provenance — every number I wrote

| Number as printed | Source file | Where in it |
|---|---|---|
| 1 of 50 relabelings | `article/02_related_work.tex` | line 67, and `tab_representation_properties.tex` caption |
| n ≈ 20 (compactness scope) | `article/05_results.tex` | line 126, C2's frozen scope |
| n = 20 (nauty-sparse6 shorter above) | `article/05_results.tex` | lines 139–141 |
| \|Σ\| = 9 | `article/tab_instruction_set.tex` | 9 distinct symbols {N,n,P,p,V,v,C,c,W}; `01_introduction.tex:38` says "nine instructions" |
| ⌈log₂ n⌉ | sparse6 format spec, `\cite{mckay2022graph6}` | format property, not our measurement |
| ρ = 0.71–0.997 | `article/05_results.tex` | lines 339–341 (C5) |
| ρ = 0.9278 / ρ = 0.9139 | `article/05_results.tex` | inline block 1, IAM Letter LOW row |
| n ≈ 40 | `article/05_results.tex` | line 398 (C6) |
| n = 98 | `article/04_experimental_design.tex` | lines 136–141, "the retained ceiling is 98 nodes" |
| n = 12 | `article/04_experimental_design.tex` | line 138, "Below n = 12 the reference is exact" |
| \|Aut(G)\| = 10⁸ | `article/05_results.tex` | lines 233–234, "100 % above 10⁸" |
| 3,000 graphs, ~12 nodes, one non-graph baseline | `T-07-article-notes.md` | §4 table, [28] column |
| 10,000 ZINC molecules, SMILES/SELFIES/InChI | `T-07-article-notes.md` | §4 table, [29] column |
| LOW / MED levels of IAM Letter | `article/05_results.tex` | inline block 2 |

Nothing was inferred from another number or taken from memory. Two numbers the brief offered were
**not** written — see Decisions.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| R3.1b | `sec:tradeoff`, `sec:limits` | *"That representation holds the metric-admitting frontier alone."* + *"Both canonicalised serialisations accept the input we reject."* |
| R3.6b | `sec:tradeoff` | *"Its correlation against them is bracket-dependent, indistinguishable under the lower bound and weaker under the upper."* |
| AE.1 | `sec:limits` item 1 | *"The node ceiling of this evaluation belongs to the reference distance and not to the encoding."* |
| R1.3c | `sec:limits`, the condition ¶ and the labels ¶ | *"…the canonical string stops paying its way against the size baseline between the LOW and MED levels of IAM Letter."* + *"what the discard costs is not measured there at all"* |
| R3.7a (1) | `sec:limits` item 1 | as AE.1 above, with its cause |
| R3.7a (2) | `sec:limits` item 2 | *"Canonicalisation carries no polynomial guarantee… Every graph we measured above \|Aut(G)\| = 10⁸ exhausted the search budget."* |
| R3.7a (3) / R3.2 | `sec:limits` item 3 | *"We evaluate no sequential model and no downstream pattern-recognition task…"* then the two setups, then *"Neither is a graph-learning evaluation on an established benchmark, so reproducing either would leave the question open."* |

## Measured

- **Section length: 2.182 p** against a 1.80 p target — **+0.38 p, and +0.08 p outside the ±0.30
  tolerance.** Method: isolated document (`scratchpad/measure/measure.tex`) = preamble +
  `06_discussion.tex` only, with `\newlabel` and `\bibcite` seeded from the real build so every
  `\ref` and `\cite` sets at its true printed width; probe reads `\thepage` / `\the\pagetotal`
  after the last line; length = (page − 1) + pagetotal/`\textheight`, `\textheight` = 550.27614 pt.
  The section has no floats, so an isolated span is exact and no peer float can contaminate it.
- **The section sits on a page-break plateau.** `\raggedbottom` is set, and two successive passes
  each removing ~30 words left `\pagetotal` at exactly 100.0 pt — 2.182 p both times. Word-level
  trimming is spent; the next reduction has to be a whole paragraph. Recorded in the file header.
- Compile: **clean. 0 undefined citations, 0 undefined references, 0 errors, 0 overfull boxes over
  5 pt.** Citation figure taken **after** running `bibtex` in my out directory, per the CONTRACT's
  build trap. Document builds at 40 pages (peers live).
- Humanizer: run last, after the numbers were verified. 8 sentence splits, no lexical
  substitutions. Median sentence length 27 → 16 (band 10–24), CV 0.469 → 0.509, 681 → 674 prose
  tokens. Ledger re-checked by whitespace-insensitive diff: **numbers identical, citations and
  cross-references identical.**

## Decisions and assumptions

1. **I did not write ρ = 0.26.** The §6.2 brief and two coordinator messages give the contrast as
   "ρ = 0.93 against ρ = 0.26". 0.26 is not printed in §5 and I could not trace it to a file I
   opened, so under the CONTRACT's no-number-from-memory rule I substituted a contrast that is
   printed in §5.4's own inline block and is therefore checkable by a reviewer on the facing page:
   ρ = 0.9278 for the canonical string on IAM Letter LOW against ρ = 0.9139 for the baseline. It
   demonstrates the same point — a headline correlation that survives only until the baseline is
   printed beside it. **If 0.26 has a source, it is the stronger illustration and should replace
   mine.**
2. **C10's referents are named rather than left abstract.** The frozen sentence's two clauses point
   at raw `sparse6` and `wl_subtree` (T-06-FRAMING §9.4), and neither is a row of
   `tab:representation-properties`. Left abstract, "the most compact serialisation admits no
   metric" reads as contradicting the table, where nauty-sparse6 carries `metric ✓`. I anchored
   clause 1 on §2.1's own statement that raw graph6/sparse6 fail the invariance filter at 1 of 50
   relabelings, and clause 2 on the WL map's absence from the compactness comparison.
3. **I added a limitation the brief's seven do not list**, because §5.4 promises it: *"…and
   Section~\ref{sec:limits} states the condition that follows."* Without it that forward reference
   dangles. It is the coordinator's negative 2, written as degradation under rising distortion.
4. **Items 5 and 6 point back to §4 instead of reprinting its figures.** §4 already carries the
   size-biased discard and "the loosest of the seven we measured, by 6.7×" with the +0.294/node
   slope. Reprinting would have cost ~40 words and risked a second, differing ratio (see below).
5. `\GTS` is a math-mode macro; used as `$\GTS$`, matching §3.

## For the orchestrator

**Three things I think are wrong outside my file. I changed none of them.**

1. 🔴 **§5.2's closing sentence contradicts T-06-FRAMING §9.1, and the error is against us.** It
   reads: *"IsalGraph leads the codes that carry their own canonical form and trails the ones that
   borrow a canonical labelling from `nauty`."* Plural — both nauty forms. But §9.1's dominance
   matrix (sign test over (dataset, n) strata at n > 20, α = 0.05) records `isalgraph_pruned` vs
   `nauty_graph6` as **WIN** on compactness, and §9.3 states the weak-dominance result explicitly:
   *"IsalGraph weakly dominates `nauty_graph6`: wins compactness, ties correlation. That is a
   clean, defensible, unqualified statement about one named competitor."* §5.2's own evidence is
   only about nauty-sparse6 (*"nauty-sparse6 is shorter than IsalGraph at every size above 20"*);
   the plural generalisation overshoots it. **My §6.1 follows the source and says we are more
   compact than nauty-graph6**, so as it stands §5.2 and §6.1 disagree in print. One of the two
   has to move and I believe it is §5.2's.
2. ⚠ **`tab_representation_properties.tex`'s caption describes a column the tabular does not
   have.** The caption defines *"`bits` is whether a message length is defined at all"*, but the
   header row is representation / family / ψ / complete / collis. / revers. / disconn. / metric /
   n_max / compl. / learning — no `bits` column. My §6.1 leans on exactly that property for the WL
   map, so a reviewer following the caption will look for the column and not find it.
3. ⚠ **The selected upper bound is `BIPARTITE`, and several plan files still say `IPFP`.** §4 and
   `approx_ged.md:142` (T-27) agree on BIPARTITE; `approx_ged.md:19/:81/:128` and the repo's
   `.claude/CLAUDE.md` GEDLIB section still name IPFP as the tight upper bound. Also note two
   different looseness ratios are in circulation — §4 prints **6.7×**, `approx_ged.md:155` gives
   **13×** (mean relative error 1.095 vs 0.084). I printed neither and pointed at §4 instead, but
   they should be reconciled before the response letter quotes one.

**On length.** 2.182 p, +0.08 p outside tolerance, and I stopped there deliberately: the section is
on a `\raggedbottom` plateau where word cuts do not move the page, and everything discretionary is
already gone. Cutting further means dropping a demand or a whole paragraph. The two candidates, if
you need the page: the §6.2 IAM Letter demonstration (~35 words, but it is the section's only
concrete illustration of the recommendation) or §6.3's condition paragraph (~40 words, but §5.4
forward-references it). I recommend neither and would take the 0.08 p.

**On the two corrections you sent.** Both landed. My §6.2 did contain the banned sentence
(*"Together they place the negative result in the benchmark rather than in the encoding"*) and it
is gone; §6.2 now rests only on the field-level negative. I have recorded the reasoning as a
comment block above §6.2 so the next agent to touch it does not reintroduce it from the old brief.
My draft never contained the thesis paragraph's "where structure varies and size does not" clause
— I checked by grep before and after.
