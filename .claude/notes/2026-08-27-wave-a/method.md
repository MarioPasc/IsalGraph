# wave-method — Section 3, Method

## What the section now says

The section opens with one roadmap paragraph (the four submitted subsection preambles are
merged into it) and fixes notation: `n = |V|`, `m = |E|`, `M` = the encoder's running output
node count. The submitted text used `M` for both the output node count and the edge count;
that collision is gone.

§3.1 defines the interpreter state as a triple `(G, L, pi)`, states that CDLL indices and graph
node indices are different index spaces, gives the nine-symbol alphabet with Table 1, and states
the two properties used later: `V`/`v` do not advance the pointer they insert through, and every
string over Sigma decodes to a valid graph. `S2G` follows as Algorithm 1, `Theta(|w|)`, taking
directedness as a second argument — which is where §3.3.3 later lands.

§3.2 defines displacement pairs with the full three-component sort key, then the greedy encoder
as Algorithm 2. The `C` and `c` branches are rewritten against `graph_to_string.py` (R3.4a) and
a short prose paragraph states the correction. The listing now names `choose(C_i)` explicitly as
the one open decision, which is what §3.3 searches over.

§3.2.3 is new. It names three objects and keeps them named: theoretical cost, worst-case search
behaviour, empirical scaling. The four operations are costed with pointer walking as the dominant
`O(m n^3)` term; `P(M)` is stated as recomputed at every frame with the 12-distinct-`M`
measurement; the one-sided displacement lemma makes `|w| = m + sum(|a|+|b|)` exact; the invariance
floor proposition and its two corollaries replace the word "exponential" with a characterisation;
and the fitted exponent is labelled a cohort property, not a complexity result.

§3.3 defines both canonical forms, corrects the false "Only" of the submitted Remark 2.7 in prose,
carries the six start-node measurement the removed schematic used to carry, states that the pruned
form is never shorter than `w+_G` and therefore that §5.2's compactness figures are conservative,
and restates the completeness theorem within a fixed directedness class with the flag hypothesis
in the statement rather than in the proof. §3.4 defines both distances and hands the fidelity
question to §5 without claiming an answer.

## Number provenance — every number I wrote

| Number as printed | Source file | Where in it |
|---|---|---|
| `1,109,460` frames | `results/reports/T-13-complexity/REPORT.md` | §3, "Measured over 1,109,460 frames" |
| `12` distinct values of `M` | same | §3, "Only 12 distinct values of `M` occur" |
| `25.5x` to `108.6x` | same | §3, "25.5× (n=6) to 108.6× (n=10)" |
| `178,886` (graph, start) pairs, no mismatch | same | §4, "178,886 (graph, start) pairs, 0 mismatches" |
| scan depth `2.1 %` of worst case | same | §4 table, pair scanning row |
| first pair accepted in `26.45 %` of frames | same | §4 table, pair scanning row |
| `215,270` frames, 0 exceptions | same | §4, one-sided displacement lemma |
| `0.238` vs `1.480` movement characters | same | §4, "insertion frames average 0.238 ... chord frames' 1.480" |
| `16,370` cohort + `664` constructed graphs, 0 violations | same | §6, "Gate G1" |
| `O(m n^2 log n)`, `O(m n^2)`, `O(m n^3)`, `O(m Delta)` | same | §4 table, per-encode column |
| `m` frames = `n-1` insertions + `m-n+1` chords | same | §4, opening line |
| alpha `2.04 [1.88, 2.20]` greedy | `T-13-complexity/tables/tab_t13_scaling_exponent.tex` | `IsalGraph$_{greedy}$` row |
| alpha `3.15 [2.18, 4.01]` pruned | same | unadorned `\textsc{IsalGraph}` row |
| alpha `17.43 [14.67, 19.35]` exhaustive | same | `IsalGraph$^{\ast}$` row = key `isalgraph_canonical` |
| min-DFS `2.08`, nauty-graph6 `3.24`, AGM CAM `4.89` | same | corresponding rows |
| lengths `9, 10, 9, 11, 10, 10`; one attains `w+_G` | measured live this session, engine `cpp` | `GraphToString(g).run(v)` for v in 0..5 on `RUNNING_EXAMPLE_EDGES` |
| `18,461` completed; pruned shorter in 0 cases | `.claude/notes/2026-08-25-t06-exhaustive/log.md` | §"Two minimality invariants", table |
| `7,316` of `18,461` (`39.6 %`) | same | same table, third row |
| `5,350` Suite-1 graphs, never shorter | `T-09-explanatory-figures/figures/fig_worked_example_g2s_pruned.caption.tex:4-6` | caption text |
| `63.9 %` at `n = 12`, median `+1` symbol | `.claude/notes/review/tasks/T-06-POSITIONING.md:213,215` | per-`n` breakdown table |
| nine-character alphabet | `tab_instruction_set.tex` | the table itself |

No number came from memory, from another number, or from a peer's draft.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| R3.4a | Algorithm 2 `C`/`c` branches + `\ref{sec:greedy}` prose | *"In the previous version of this listing both guards, and both duplicate checks, were stated on the transposed pair, which contradicts Table 1 ... We have corrected both branches against the reference implementation, which is the definition of record."* |
| R3.4b | `\ref{sec:cost-model}` | *"P(M) is recomputed at every frame, at all three call sites in the reference implementation, Theta(M^2 log M) each; nothing is memoised."* |
| R3.4c | `\ref{sec:cost-model}`, empirical scaling | *"A fitted exponent ... is a property of that cohort and not a bound on the algorithm, and two fits from two cohorts conflict only if either is read as one."* |
| R3.7d | `\ref{sec:cost-model}` opening + three `\paragraph` headings | *"Three questions are answered by three different objects and we name them here so that they are not conflated later."* |
| R3.3b | `\ref{thm:invariant}` statement | *"...that are either both undirected or both directed"* — in the statement, not the proof |
| R3.3c | `\ref{sec:invariance}`, after the proof | *"The flag is therefore external metadata that must accompany the string ... deduplicating a corpus that mixes the two classes must key on the pair (flag, string)."* |
| R3.7c (partial) | `\ref{sec:exhaustive-canonical}` | *"Figure S1 draws the resulting search space as six start-node subtrees with the canonical path highlighted."* |
| E13 | `\ref{sec:exhaustive-canonical}`, "What is searched over" | *"The search ranges over two things: the starting node ... and the identity of the uninserted neighbour"* |
| E8 | — | Example 2.3 cut outright; the printed draft self-correction is gone |
| P2 | `\ref{sec:pruning}` | *"...every compactness figure in Section 5.2 is a conservative lower bound on what the exhaustive form achieves."* |
| B2 | `\ref{sec:pairs}` + `\ref{sec:invariance}` | Sort key printed with all three components; the directedness witness is quoted with no collision rate, so no rate travels without its window |

## Measured

- **Section length: 12.23 p** against the 10.4 p target — **1.83 p over**. (12.26 p before the
  humanizer pass, which came out one line shorter.)
  Method: `\pdfsavepos` probes at the first and last token of the section, read at shipout;
  fraction of the start page below the probe, plus whole pages, plus fraction of the end page
  above it, against `\textheight = 550.27614 pt` and a 4.3 cm top margin. Cross-checked against
  the `.aux` page numbers of `sec:method` (p.8) and `sec:design` (p.22). Probes removed afterwards;
  `grep` confirms zero remain.
- Whole document: **30 p**, plain `latexmk` with no manual bibtex step, after the orchestrator
  refreshed the shared `main.bbl` to 25 entries.
- Compile: clean. **0 errors, 0 undefined references, 0 undefined citations, 0 font warnings,
  no "float too large for page"**; worst overfull box ~1 pt, well under the 5 pt bar. Both
  algorithm listings place correctly; the `\linespread{1}\small\selectfont` resets are untouched.
- **The font warning the orchestrator traced to this file is fixed.** Cause:
  `$d_{\IsalGraph}$` inside `cor:metric`, whose body `\theoremstyle{plain}` typesets in italic;
  `\IsalGraph` expands to `\textsc`, and Times has no small-caps-italic face
  (`OT1/txr/m/scit`). Fix: `d_{\textup{\IsalGraph}}` in that one spot, which renders identically
  to the six upright occurrences elsewhere. Document-wide font warnings are now 0.
- **No `changes` markup in this file.** The nine `\added{}` spans I had written are removed on
  instruction; the header comment block now inventories the checkable spans for the dedicated
  markup pass, quoted by opening words so that pass need not re-derive them.
- **Algorithm 1 costs 0.79 p**, measured by `\iffalse`-ing the float and re-measuring
  (12.26 -> 11.47 p), then restoring it.

## What changed in Algorithm 2, and what it now matches

`src/isalgraph/core/graph_to_string.py`, the frozen Python reference.

**The `C` branch.** Code, lines 209-212:

```python
if tent_sec_in in self._input_graph.neighbors(
    tent_pri_in
) and tent_sec_out not in self._output_graph.neighbors(tent_pri_out):
    self._output_graph.add_edge(tent_pri_out, tent_sec_out)
```

The guard is `(v~1, v~2) in E` and the duplicate check is
`(val(l~1), val(l~2)) not in E(G_out)`; the inserted arc is `(val(l~1), val(l~2))`.

Submitted listing: guard `(v~2, v~1) in E`, duplicate check
`(val(l~2), val(l~1)) not in E(G_out)`, inserted arc `(val(l~1), val(l~2))`.
**Both tests were on the transposed pair relative to the arc actually inserted**, and relative
to Table 1, which defines `C` as "Edge insertion (primary -> secondary)".

**The `c` branch.** Code, lines 224-229:

```python
if (
    self._input_graph.directed()
    and tent_pri_in in self._input_graph.neighbors(tent_sec_in)
    and tent_pri_out not in self._output_graph.neighbors(tent_sec_out)
):
    self._output_graph.add_edge(tent_sec_out, tent_pri_out)
```

Guard `(v~2, v~1) in E`, duplicate check `(val(l~2), val(l~1)) not in E(G_out)`, inserted arc
`(val(l~2), val(l~1))`. The submitted listing had `(v~1, v~2)` and `(val(l~1), val(l~2))` —
the same transposition, in the opposite direction.

**So the two branches had their index pairs swapped with each other.** Both guards and both
duplicate checks are now as the implementation evaluates them. The inserted arcs were already
correct in both branches and are unchanged.

**One further change, not requested but implied.** The submitted listing wrote the `V`/`v` guard
as `exists c in N_G(v~1) with c not in dom(iota)`, which hides that the encoder picks a specific
candidate. `_find_new_neighbor` (`graph_to_string.py:344-347`) returns the **first** such
neighbour in `set` iteration order. The listing now names the candidate set `C_i` and the
selection `choose(C_i)`, so the single decision §3.3 searches over is visible in the listing that
defines it. Net line count unchanged.

## Decisions and assumptions

1. **`def:pairs` was already correct in the submitted manuscript.** I was told to check it against
   invariant 5. Submitted text: sorted by `(|a|+|b|, |a|, a, b)` lexicographically. Code
   (`graph_to_string.py:62`): `key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0]), pair)`.
   These are the same order. **B2 is a historical code bug that never reached the manuscript**;
   there was nothing to correct, and I did not invent a correction. I print the key as
   `(|a|+|b|, |a|, (a,b))` to match the invariant's form exactly.
2. **No collision rate is printed.** The directedness argument uses only the exact witness — a
   single undirected edge and a single directed arc both canonicalise to `V`. Quoting any rate
   would require its enumeration window, and the witness needs no enumeration, so the rate is
   simply absent. This is the safest discharge of R3.3c and cannot violate B2.
3. **The submitted §2.4's "strong correlation" claim is removed.** It said the Spearman
   correlation "is high on sparse graphs and moderate on denser graphs". H3 is refuted at scale,
   and that sentence is the R3.6b overclaim. §3.4 now hands the question to §5 and §6 without
   asserting an answer. No demand assigned this to me; leaving it would have contradicted §5.
4. **Notation collision fixed.** The submitted text used `M` for the output node count in the
   displacement-pair definition and for the edge count in the string-length remark. Now `n`, `m`,
   `M` throughout.
5. **The theorem is stated for both canonical forms.** The submitted proof covered only `w*_G`
   (pruned), yet §5 reports an exhaustive arm. Step 1 is the only place the pruning rule enters,
   so one added clause covers `w+_G` too.
6. **Algorithm 1 was restored after measurement.** See below.
7. **Markup removed on instruction.** I had wrapped nine checkable spans in `\added{}`. The
   orchestrator's rule is that a dedicated later pass owns `changes` markup, so the wrappers are
   gone and the header block lists the spans instead. §3.2.3 is new in its entirety and is
   deliberately *not* listed for marking — review-procedure §2.3 declares whole-new sections once,
   at the top of the blue version.
8. **Humanizer ran last**, after every number was verified, as the contract requires. It made 16
   prose-only edits and reported every number byte-identical; I re-checked the nine
   highest-risk figures in the file afterwards and they are unchanged.
9. **Property vocabulary checked against Section 1.** I use *compactness* in Section 1's sense and
   *local* only descriptively, never "structure-preserving". Section 3 does not enumerate the
   four-property set, so there is nothing to re-name.

## For the orchestrator

**The 10.4 p target for §3 appears to double-count E7, and I think the plan is wrong here.**

`prose.md` §3's own itemised cut table nets to **−1.35 p**, not −3.6 p. Its own "Net" row says so,
and adds *"plus E7's share of the float recovery, which lands mostly here."* But **E7 was already
fixed in `main.tex` before this wave started** — the file says *"E7 FIXED HERE BY OMISSION ...
Both lines are deleted"* — so the 13-page empty scaffold I measured against already contains
E7's recovery. It cannot be credited to §3 a second time.

Arithmetic: 14.0 submitted − 0.70 (Alg. 3) − 0.50 (Example 2.3, remarks) − 0.40 (§2.4 thinned)
− 0.30 (preambles) − 0.75 (proof) + 0.80 (cost model) = **12.15 p**. I measured **12.26 p**,
which is within 0.11 p of the plan's own itemised prediction. **Every itemised cut was made.**
The 10.4 p figure in the §3 section map is reachable only by counting E7 twice.

**What I did not do, and why.** `prose.md` §10 pre-declares the cut order: Tab. 5 -> S5 first,
then Alg. 1 -> S7. **Tab. 5 is `tab:bit-overhead`, which lives in §4 and is not mine.** §10.4 also
says to keep Alg. 1 unless the page count forces the cut, and to cut it *after* Tab. 5. So I
restored Alg. 1 rather than take a cross-section decision unilaterally.

**The option, costed rather than estimated: Alg. 1 -> S7 is worth exactly 0.79 p**, measured, not
guessed (§3 goes 12.26 -> 11.47 p; document 30 -> 29 p). It is a two-line change I can make in
one edit if you want it. **Note that even with it, §3 lands at 11.47 p, still 1.07 p over 10.4** —
so cutting Alg. 1 does not solve the stated problem, it only reduces it. If §3 must reach 10.4 p,
something demand-bearing has to go, and I would want that decision from you rather than from me.

**My recommendation**: accept §3 at 12.26 p and re-baseline the §3 row of the section map against
the itemised table, because the document is at 30 p with a 35 p ceiling and the margin is real.
If the total later needs the page, Alg. 1 -> S7 is the cheapest 0.79 p on the board and it is
ready.

**One more thing worth your time.** The exponent table's row *keys* do not mean what their names
suggest, and this is a live trap for §5. In
`benchmarks/real_data/eval_t06_figures/design.py`, the key `isalgraph_exhaustive` is
*"IsalGraph (exhaustive canonical, **pruned fallback**)"* — a hybrid, alpha = 4.71 — while the key
`isalgraph_canonical` is the true exhaustive form, `max_n = 12`, **alpha = 17.43**. `prose.md`
§10.1's mapping is correct as written because it speaks of the mathematical forms, but anyone
reading the table by key name will attribute 17.43 to the wrong arm. I printed 17.43 for the
exhaustive canonical form, which is right.
