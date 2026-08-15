# nauty canonical labelling → graph6

**Verdict**: **RUN — this is the fair canonical serialisation, and the pool's most important
control.** It is graph6 with the one variable changed. Installs and builds cleanly.
**bliss / Traces stay cut** (§8) — the insurance rationale that kept them alive is now void.

**Role**: [competitors](../competitors.md) §2 row 3 · serves **AE.4a**, **AE.3**, **R1.2b**
**Evidence**: measured on this workstation, 2026-08-13,
`scratchpad/competitors/{probe,sweep,scale,stability}.py`, plus a from-source build rehearsal.
Cross-refs: [graph6](graph6.md), [agm](agm.md) (the other lex-order canonical form), [README](README.md) §2.

---

## 1. Reproducibility — is anything blocking?

**No. Verified from source, which is the state Picasso will be in.**

```bash
pip install pynauty                     # 2.8.8.1
```

| Check | Result |
|---|---|
| PyPI sdist bundles nauty | **yes** — `pynauty-2.8.8.1/src/nauty2_8_8/`, 256 nauty source files. **No network access beyond pip** |
| Builds from source, no cached wheel | **yes** — `pip install --no-binary :all: --no-cache-dir pynauty`, exit 0 |
| Toolchain | **gcc 12.2.0**, which is the module CLAUDE.md pins for Picasso (`module load gcc/12.2.0`) |
| Post-build smoke test | `pynauty.canon_label(Graph(3, adjacency_dict={0:[1],1:[2]}))` → `[0, 2, 1]` |
| Encode cost | **0.042–0.351 ms** per graph over the whole cohort profile range |

**API actually used** (not the one most tutorials show):

```python
import pynauty
pg  = pynauty.Graph(n, directed=False, adjacency_dict={v: list(nbrs)})
lab = pynauty.canon_label(pg)        # canonical order: lab[i] is the OLD vertex at NEW position i
cert = pynauty.certificate(pg)       # bytes; equal iff isomorphic
grp  = pynauty.autgrp(pg)            # generators, orbits, |Aut| as (mantissa, exponent)
```

> ## ⚠ CORRECTED 2026-08-15 by T-04 — wrong on both halves; the inversion itself is real
>
> **The inversion is real and the fix is right**: `canon_label` gives, for each new position, the
> old vertex, so relabelling needs `pos = {old: new for new, old in enumerate(lab)}`. Everything
> below about *that* stands. The two claims about its **symptom** and its **guard** do not.
>
> **1. The inverted labelling does not "pass an invariance test". It fails F3 loudly.** Because
> `lab_{G^τ}` depends on `τ`, the wrong-direction image is `G^{τ π_G⁻¹ τ}`, which varies with the
> relabelling. Measured by track B on the fixtures (15/19/5/13 distinct codes) and by the
> orchestrator independently: **non-invariant on every connected trial**, 22/22 at `n = 8`.
>
> **2. `nx.is_isomorphic(G, relabelled)` cannot catch it, ever.** Any bijective relabelling of a
> graph is isomorphic to it *by construction*, so the assertion is vacuous for this fault. Measured:
> **`True` on 100 % of deliberately inverted cases.** It also costs 6.7 ms at `n = 96` against
> 0.33 ms for the relabelling itself — a 20× tax on a step whose published cost is 0.042–0.351 ms.
>
> **What this changes**: the inversion is a *loud* failure, not a silent one, so the risk this file
> and [README](README.md) §6 item 2 flag as the second-quietest trap is **not quiet at all**. The
> prescribed guard should be an unconditional `O(n+m)` bijection-and-edge-count check, which
> actually *proves* the relabelling is a permutation of the right graph; `nx.is_isomorphic` is worth
> keeping only for a different fault — a wrong networkx-label → pynauty-index map.
>
> **What survives**: the inversion formula, the from-source build, `pynauty.autgrp` for `|Aut(G)|`,
> the certificate caveat, and every measured number in this file.

> ~~**`canon_label` returns the inverse of what you probably want.** It gives, for each new position,
> the old vertex. To relabel you need `pos = {old: new for new, old in enumerate(lab)}`. Getting
> this backwards produces a *different but still deterministic* labelling — it will pass an
> invariance test and be wrong. Assert `nx.is_isomorphic(G, relabelled)` on every encode.~~

---

## 2. What the representation looks like

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`, `|Aut(G)| = 4`.

```
graph6(G)                          'ElCW'      -- the labelling it was handed
nauty->graph6(G)                   'E@ro'      -- the canonical labelling
nauty->graph6(G relabelled)        'E@ro'      <-- INVARIANT
nauty->graph6(H = G minus 0-3)     'E@po'
pynauty.certificate(G) == certificate(G')      True
```

| Test | graph6 | **nauty→graph6** |
|---|---|---|
| 200 relabellings of the running example | 122 distinct | **1** |
| F3 sweep: 40 graphs × 25 relabellings | **0 / 40** invariant | **40 / 40** invariant |

| Property | nauty→graph6 | IsalGraph pruned |
|---|---|---|
| Reversible | **yes**, to an isomorphic copy | yes, up to isomorphism |
| Isomorphism-invariant | **yes** (40/40) | **yes** (40/40) |
| Complete invariant | **yes** | yes, within a directedness class |
| Fixed finite alphabet | yes, 64 printable ASCII | yes, `\|Σ\| = 9` |
| Length | deterministic, `1 + ⌈n(n−1)/12⌉` | data-dependent |
| Handles disconnected | **yes** | no |
| Worst case | exponential (Miyazaki 1997 families); **not observed on this cohort** | exponential in density; **observed**, see §5 |
| Encode cost | 0.042–0.351 ms | 0.01–1.02 ms (pruned); `canonical` times out above `n ≈ 30` |

**Cite**: McKay & Piperno, *Practical graph isomorphism, II*, **J. Symb. Comput. 60:94–112, 2014**,
doi:10.1016/j.jsc.2013.09.003. For the exponential worst case: Miyazaki, *The complexity of
McKay's canonical labeling algorithm*, in **Groups and Computation II, DIMACS 28:239–256, 1997**.
Both are missing from `cas-refs.bib` and both belong there
([reviewer-1](../../source/reviewer-1.md) names the gap).

---

## 3. Which distance does it accept?

Because the code is canonical, every candidate is at least *meaningful*.

| Candidate | F1 (well-defined) | Note |
|---|---|---|
| Hamming | **100 % of equal-`n` pairs**, 0 % otherwise | length is a function of `n` alone |
| padded Hamming | 100 % | the canonical triangles embed in a common `max(n₁,n₂)` frame |
| Levenshtein | 100 % | measured below |

Measured over 120 one-edit pairs (unit GED = 1) against 120 random same-`n` pairs, `n ∈ [6,12]`:

| | median Lev, GED = 1 | median Lev, random | **separation** | max, GED = 1 | median length |
|---|---:|---:|---:|---:|---:|
| nauty→graph6 | 5.0 | 6.0 | **0.83** | 10 | 7 |

**On the real cohort, against certified exact GED** (200-graph sample, Levenshtein):

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **nauty→graph6, all pairs** | 0.677 | 0.663 | 0.639 | 0.538 | 0.460 |
| **nauty→graph6, equal-`n`** | **0.974** | **0.969** | 0.682 | 0.261 | 0.186 |
| graph6, equal-`n` (same format, no canonicalisation) | 0.539 | 0.430 | 0.447 | 0.286 | 0.171 |
| AGM CAM, all pairs (the other canonical form) | 0.911 | 0.920 | 0.892 | 0.798 | — |

> **Pre-committed outcome 2 in [competitors](../competitors.md) §4 is CONFIRMED, and it is the most
> useful negative result in this folder.** A single unit edit moves the canonical graph6 string by a
> **median of 5 characters out of 7** — 83 % of the distance to a completely unrelated graph — and
> on real data nauty→graph6 scores **0.46–0.68**, the lowest of the three canonical serialisations,
> against AGM's 0.80–0.92 on the same graphs with the same distance. nauty's canonical form is a
> *unique* representative, not a *stable* one: refinement can hand two graphs that differ by one
> edge two wholly different orderings.
>
> **Two refinements the real data forces.**
> 1. **Name nauty, not "canonical forms".** AGM is also canonical and is ~0.25 better in ρ. The
>    instability is a property of *nauty's* refinement, not of canonicity.
> 2. **The equal-`n` row is where canonicalisation pays.** On Letter LOW, holding the format fixed
>    and changing only the labelling moves equal-`n` ρ from **0.539 to 0.974**. That single
>    comparison — same bits, same distance, one variable — is the cleanest evidence in the folder
>    that a representation must be canonical before an edit distance on it means anything, and it
>    is invisible in the all-pairs view because the size channel floats everything
>    ([README](README.md) §4.2).
>
> This separates the two properties the paper conflates. **Canonical** and
> **edit-distance-compatible** are independent, and nauty is the existence proof that the first does
> not imply the second. R1.2's uniqueness axis and its "what advantages does it offer" clause are
> answered by this single row.

By §3.4's rule nauty→graph6 does have admissible distances (F1/F2/F3/F4 all pass), so it **stays in
the running comparison** and its poor separation is reported as a result rather than an exclusion.
Primary distance: **padded Hamming**, the cheapest that is defined on 100 % of pairs.

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes, and it is byte-for-byte identical to graph6** — same format, same payload length
`n(n−1)/2`, same `6 · (1 + ⌈n(n−1)/12⌉)` entropy bound. Canonicalisation permutes the bits; it does
not change how many there are.

> **Do not print graph6 and nauty→graph6 as two Claim A rows with the same numbers and no
> explanation.** State once that the four `n²` members of the pool — adjacency, graph6,
> nauty→graph6, AGM CAM — have the *same* information content by construction and differ only in
> **which labelling** they commit to, then give one row and a footnote. A reviewer who notices two
> identical columns will assume a copy-paste error. See [README](README.md) §2.

Measured entropy-bound bits: identical to the [graph6](graph6.md) §4 column at every profile.

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned, and it is the pool's centre of gravity.** [competitors](../competitors.md) §1 already
says we need canonical relabelling "to make the graph6 comparison fair rather than a strawman" —
this file measures how much that mattered: separation moves 1.00 → 0.83, and invariance 0/40 →
40/40, with the format held constant.

**IsalGraph's advantage over nauty→graph6, audited axis by axis:**

| Axis (R1.2b) | Winner | Measurement |
|---|---|---|
| Uniqueness | **tie** | both 40/40 invariant, both complete invariants |
| Expressiveness | **nauty** | disconnected, isolated vertices, vertex colours; IsalGraph raises `DisconnectedGraphError` |
| Computational efficiency | **nauty**, decisively | 0.35 ms at `n = 32` versus IsalGraph `canonical_string` **timing out at 10 s** from `n = 30, m = 60` upward. `pruned` is 1.02 ms and survives |
| Scalability | **nauty** | see above; also nauty is the reference implementation of a 40-year-old, heavily optimised C library |
| Downstream learning | **not evaluated** | R1.2b's fifth axis, reported as such |
| Bits | **tie** | identical by construction |
| **Edit-distance compatibility** | **IsalGraph** | separation 0.73 vs 0.83 — but *both are weak*; [gspan-mdfsc](gspan-mdfsc.md) beats both at 0.32 |

> **The foundational advantage over nauty is narrow and must be stated narrowly.** It is not speed,
> not compactness, not robustness. It is that IsalGraph's canonical form is a **string over a fixed
> 9-symbol alphabet whose symbols are graph-construction operations**, so a substring is a partial
> construction and an edit is an operation edit. nauty's output is a permutation; graph6 is a
> packing of it; neither has an interpretation at the level of a single symbol. That is the property
> the paper should claim, and §3's separation numbers are its evidence — including the part that
> says the advantage over nauty is 0.73 vs 0.83, which is small.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Yes.** `pip install pynauty`, sdist bundles nauty 2.8.8, **from-source build verified** with gcc 12.2.0 |
| 2 | Representation | canonical relabelling → graph6. **40/40 invariant**, complete invariant |
| 3 | Distance | Hamming / padded Hamming / Levenshtein all defined. **Primary: padded Hamming.** Real ρ **0.46–0.68** all-pairs, **0.19–0.97** equal-`n` — lowest of the three canonical forms |
| 4 | Claim A? | **Yes**, identical bit count to graph6 by construction — print one row, not two |
| 5 | Scope | **In, and central.** Isolates canonicity as a variable at fixed format |
| — | IsalGraph advantage | **Narrow**: only edit-distance compatibility (0.73 vs 0.83) and the operational alphabet. nauty wins efficiency, scalability, expressiveness |

---

## 7. For the integration agent

- Pin `pynauty == 2.8.8.1`. Build it on Picasso as part of environment setup, like the C++ engine —
  **it will not rsync**, for the same reason.
- File-count budget: the pynauty build tree is small next to GEDLIB (hundreds, not tens of
  thousands of files) so the `fscratch` inode quota in CLAUDE.md is not at risk here. Delete
  `src/nauty2_8_8/` build artefacts anyway.
- ~~**Assert `nx.is_isomorphic(G, relabelled)` on every encode.** The `canon_label` inversion trap in
  §1 produces a deterministic wrong answer that passes F3.~~
  **CORRECTED 2026-08-15 by T-04 — see the block in §1.** The inverted labelling **fails F3**, and
  `nx.is_isomorphic` **cannot catch it**: any bijective relabelling is isomorphic by construction,
  so the assertion was `True` on 100 % of inverted cases. Assert an `O(n+m)` bijection-and-edge-count
  check instead, which proves the relabelling is a permutation of the right graph; keep
  `nx.is_isomorphic` only for a wrong networkx-label → pynauty-index map.
- Reuse `pynauty.autgrp` to get `|Aut(G)|` — [corrections](../corrections.md) §5 / T-13 needs it for
  the complexity section's worst case, and it is free once this backend exists.
- `pynauty.certificate()` is *not* a substitute for the graph6 route in a comparison table: it is a
  padded machine-word bit matrix, so its length is a function of the word size, not of the graph.
  Use it only for the F3 assertion.

---

## 8. bliss and Traces stay cut — the counter-case has expired

[competitors](../competitors.md) §2 cuts bliss/Traces (decision S-g, 1.0 d) and records one
counter-case: *"they are cheap insurance if `pynauty` fails to build on Picasso, which would
otherwise take the graph6 and AGM rows down with it."*

**That insurance is no longer needed.** The from-source build was rehearsed in a clean environment
on the same gcc version Picasso pins, and it succeeded (§1). The remaining rationale for the cut
stands unchanged: bliss and Traces emit a canonical labelling serialised to graph6 exactly as nauty
does, so they differ in **speed, not representation**, produce no distinct table row, and were
requested by nobody.

**One correction to the cut's wording**: the note says a `pynauty` failure "would take the graph6
row down with it". It would not — graph6 needs only `networkx`. It would take **nauty→graph6** down,
and it would take AGM down only if AGM were implemented on top of nauty's labelling, which
[agm](agm.md) shows it cannot be.
