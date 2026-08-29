# wave-conclusions — Section 7, Conclusions and future work

## What the section now says

Five paragraphs, continuous prose, no floats, no `changes` markup.

It opens on the frozen property set — validity, reversibility, canonicity, compactness — names
the scope of the first two (decoding total; encoding defined for connected undirected simple
graphs and for directed graphs in which some node reaches every other), and states canonicity
within a fixed directedness class, with the metric corollary as its consequence. "Locality" is
absent from the set by design: it is H3's measured quantity, not a property we claim.

It then reports H1–H4 by verdict in Section 5's order. H1 holds by census. H2 holds where it is
scoped and fails where it is not, with both halves of the compactness claim in the same sentence.
H3 and H4 are refuted and neither is softened: H3's refutation is located in the benchmark and
generalises to every representation tested; H4's is replaced by the |Aut(G)| statement in the form
§5.3 requires — size sets the frame count, |Aut(G)| governs the branching at fixed size — and is
called out as the paper's only controlled experiment and as predictive.

Two future-work items close it. Labels: the Σ × L product alphabet, with the label attached to the
instruction that creates the object it belongs to, so the alphabet grows and the symbol count does
not, with IsalChem as the precedent; the E6 correction is carried in the same paragraph. And the
canonicalisation search: the invariance ceiling, the measured headroom below it on the cohort, and
C12 verbatim on individualisation–refinement.

The three mandated deletions are all absent: no `n^{9.0}`, no "super-polynomial", no "labels
present in all five benchmark datasets". No fitted exponent and no declared node range appear
either, which is how E3 and E4 are discharged.

## Number provenance — every number I wrote

Rule applied: no number appears here that is not already in `05_results.tex`. Verified by script
over every numeric token in the prose.

| Number as printed | Source file | Where in it |
|---|---|---|
| `n \approx 20` | `05_results.tex` | §5.2 ¶1, *"the most compact above $n \approx 20$"* |
| `17` of `25` records | `05_results.tex` | §5.4 ¶1, *"Over all $25$ records the canonical string falls below its own size null on $17$"* (phrasing matched to `01_introduction.tex` H3) |
| `n \approx 40` | `05_results.tex` | §5.4 ¶9, *"Within a fixed node count, above $n \approx 40$…"* |
| `\rho = 0` | `05_results.tex` | §5.4 ¶9, same sentence |
| `99.939\,\%` | `05_results.tex` | §5.3 closing ¶ — traces to `results/reports/T-13-complexity/REPORT.md` |
| `16{,}370` | `05_results.tex` | §5.3 closing ¶, *"of the $16{,}370$ cohort graphs"* — same source |
| `41.869\,\%` | `05_results.tex` | §5.3 closing ¶, *"and the incumbent key on $41.869\,\%$"* — same source |
| `1`-WL | `05_results.tex` | §5.3 closing ¶ (verb "attains" taken from there) |

Not printed, deliberately: the per-dataset resolution rates (Mutagenicity 14.50 %, COIL-DEL
10.33 %, Protein 10.54 %, Letter LOW 100 %, `T-13-complexity/REPORT.md:208`). They are not in
Section 5, so the concentration is stated in words and not in numbers. The constructed-ladder
figures 39/132 and 23/132 appear nowhere — they are a different population from the cohort pair.

## Demands discharged

| Demand | Where it landed | The sentence that discharges it |
|---|---|---|
| R1.3d | ¶4 (labels) | *"A product alphabet $\Sig \times L$ attaches a label to the instruction that creates the object it belongs to, so the alphabet grows and the symbol count does not; \textsc{IsalChem} takes this route for molecules~\cite{ThurnhoferHemsi:2025}."* |
| R3.6b | ¶2, ¶3 | *"H2 holds where it is scoped and fails where it is not"* — and every compactness and fidelity claim carries its scope in the same sentence. Nothing "strongly correlates" |
| R3.7e | ¶1, ¶2, ¶3 | The four properties named once in §1's set and order; *"the canonical string is a complete invariant"* (never equivariant); no broad unscoped statement survives |
| E3 | deletion | No fitted exponent appears; the exponent question is dissolved in §3.2.3/§5.3 |
| E4 | deletion | No node range is declared in this section |
| E6 | ¶4 | *"the benchmarks are not uniformly labelled either: LINUX carries no node or edge attribute (Table~\ref{tab:datasets})"* |
| B6 | ¶1 | Property set is exactly `{validity, reversibility, canonicity, compactness}`; locality is not in it |

## Measured

- **section length: 1.149 p** against a 1.00 p target — **+0.149 p**, inside the contract's
  ±0.3 p. 405 raw words, ≈ 352 raw w/p. Measured with a `\pdfsavepos` instrument at the section
  bounds (`length = Δpage + Δy/\textheight`, `\textheight = 550.27614 pt`), in a **standalone
  harness** that inputs `preamble` and this file only. The instrument is removed; the method is
  recorded in the file header.
- **In-document measurement is unreliable while peers are live, and I hit it.** The same
  instrument read 1.193 p, then 1.196 p, then **1.277 p** on `main.tex` within minutes and with
  no edit of mine. A peer filling §6 moved the page break; once START and END straddle an
  intervening float page the `Δpage` term counts a page the section does not occupy. The
  isolated figure is the stable one and is what I report.
- **compile: clean.** 0 undefined citations, 0 undefined references, 0 errors, 0 overfull boxes
  over 5 pt anywhere in the document, none at all inside this section's log segment. The
  undefined-citation count was taken **after** a `bibtex` run in the outdir, per the CONTRACT trap.
- whole document: 40 pages at the time of the final build (peers still writing; §6 was still
  TODO-only when I started and had content by the end).
- humanizer: run after the numbers were verified. Participial clauses 2.61 → **0.00**;
  sentence-length CV 0.524; every rhetorical, typographic and attribution band `ok`.

## Decisions and assumptions

1. **I could not reach 1.00 p without deleting mandated content, and I did not delete any.**
   The floor is arithmetic: four property names with their scope (~60 w), four verdicts with H3's
   field-level half and H4's controlled/predictive half (~200 w), Σ × L with IsalChem and the E6
   correction (~70 w), the invariance ceiling with both resolution numbers and C12 verbatim
   (~115 w). That is ~405 words = 1.149 p at this section's measured density. I cut the opening
   restatement of what IsalGraph is (§1 already says it, −13 w) and every other word I could find.
   **The last 0.15 p is available only from these three, in the order I would spend them:**
   (a) the metric corollary clause, −9 w ≈ 0.026 p — loses a genuine result;
   (b) the four `\ref` pointers into §5, −12 tokens ≈ 0.035 p — loses checkability in the
   direction a reviewer checks;
   (c) *"so the alphabet grows and the symbol count does not"*, −11 w ≈ 0.031 p — makes R1.3d's
   extension gestured at rather than concrete.
   All three together buy ~0.09 p and land at ~1.06. **I recommend taking none of them and
   spending 0.15 p elsewhere on the ladder.**
2. **"rooted directed graphs" was rejected.** The word *root* appears nowhere in the manuscript,
   so it would have been an undefined term and a synonym for §1's phrasing. Replaced with §1's own
   *"directed graphs in which some node reaches every other"*, at +5 words.
3. **Ceiling, not floor.** §3 names the proposition *Invariance floor* while §5.3 calls the same
   object *"the invariance ceiling of Proposition~\ref{prop:invariance-floor}"*. I follow §5.3,
   because that is the sentence this paragraph continues, and avoid the word *floor* entirely so
   the metaphor does not invert mid-paragraph.
4. **Two defects of my own, found during the final number check, both fixed:**
   - *"no two non-isomorphic graphs in the cohort share a canonical string"* **overstated §5.1**.
     The census certifies non-isomorphism from GED > 0, and §5.1 is explicit that Suite 2 leaves
     outside it the pairs the lower bound could not separate. Now: *"no pair certified
     non-isomorphic in the cohort shares a canonical string."*
   - The `n ≈ 40` sentence **had lost its scope clause**. §5.4 and §1 both write *"Within a fixed
     node count, above $n \approx 40$…"*, and the measurement is within-stratum. Restored.
5. **Hedges measure 0.00 and I did not add any.** The humanizer flags this LOW, but `01_introduction.tex`
   measures 0.00 on the same metric, so zero hedges is the author baseline here, and the project's
   prose contract bans hedges that carry no information. Deviation recorded rather than applied.
6. **Nominalizations measure 82 per 1,000 against §1's 45 and the skill's ≤ 60.** I reduced what I
   could and stopped. The residue is the four frozen property names in one sentence plus fixed
   terminology — *canonicalisation*, *automorphism detection*, *individualisation–refinement*,
   *partition*, *invariant* — which the prose contract forbids cycling. Fixing the metric would
   mean renaming the paper's terms.

## For the orchestrator

- **The section is 0.149 p over a 1.00 p target and I am reporting it rather than cutting a
  demand.** Decision 1 above lists the only three cuts left and what each costs. Your call.
- **`prose.md` §4's §7 brief asks for the per-dataset resolution rates** (Mutagenicity 14.5 %,
  COIL-DEL 10.3 %, Protein 10.5 %, against 100 % on Letter LOW) **that the same file's header rule
  forbids**, because they are not in Section 5. I followed the header rule and stated the
  concentration in words. If you want the numbers, they need a home in §5.3 first.
- **§5.1 and any conclusion drawn from it are not interchangeable.** The census is over
  *GED-positive* pairs, and Suite 2 leaves the lower-bound-inseparable pairs outside it. Any
  section restating H1 as "no two non-isomorphic graphs collide" overstates it. Worth checking
  `00_abstract.tex` for the same slip — I do not own that file.
- **Measuring a section inside `main.tex` while three agents write is unsound.** I logged three
  different readings of an unchanged section within minutes. The isolation harness in this
  section's file header is peer-proof and costs one extra build; recommend it for wave B.
- I made no claim that the representation does better where structure rather than size varies,
  and H3 carries no hedge — per your mid-task correction, which the IAM Letter control
  (0.9278 → 0.6660 against a flat size null) settles against `prose.md` §1's thesis clause.
