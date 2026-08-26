# T-07 — response-letter fragment: R3.1a(i), R3.1a(ii), AE.5 ("rationale")

**Not final letter prose.** Raw material for `review-answer`, with every claim sourced.
Owner of the final text: T-14. Written 2026-08-26.

---

## The comment

> "The manuscript appears to combine and extend two closely related prior frameworks. The earlier
> preprint [28] already introduced a universally valid, reversible, compact instruction-string
> representation of ordered adjacency matrices, together with locality claims and a Transformer-based
> classification experiment. IsalChem [29] already introduced the circular-list/two-pointer
> virtual-machine architecture, incremental graph construction, exhaustive shortest-then-lexicographic
> normalization, and Levenshtein-based locality. […] The paper should provide a detailed side-by-side
> comparison that identifies which components are inherited, modified, or genuinely new, and explain
> why the combined extension constitutes a sufficiently substantive contribution."
> — `mail.txt:86`

---

## What we did

**Read both predecessors in full and built the comparison from the papers themselves**, using the
public IsalChem implementation as an independent cross-check. New **§2.3 "Relation to the authors'
prior work"** carries **Table 3** (9 components × 4 columns, grouped inherited / modified / new) and
a closing paragraph answering the second conjunct.

**Concede R3's framing where it is right.** The circular-list/two-pointer architecture *is*
inherited, and we now say so in a table rather than in the two sentences the submission offered
(`introduction.tex:52–53`). Of nine components, **three are inherited and five are modified** — the
comparison is not flattering to us by construction, and it was built under an attribution rule fixed
before we read the sources, under which any generalisation of a predecessor's component counts as
*modified* rather than as novelty.

---

## The substantive point, and it is checkable

**Neither predecessor states or proves that its normal form separates isomorphism classes.**

| Term | [28] | [29] |
|---|---|---|
| `theorem` / `proof` / `lemma` / `proposition` | **0** | **0** |
| `complete invariant` | **0** | **0** |

- [28]'s canonical string is *defined as* the output of its greedy encoder, applied to an adjacency
  matrix that presupposes *"a complete order … on the N elements (vertices) in V"*. It is canonical
  per matrix, not per isomorphism class.
- [29] argues relabelling invariance in three sentences, **in one direction only**, resting on the
  stated assumption that *"The obtained molecular graph may be assumed to be unique for a given
  molecular species."* The converse is never stated, and no experiment measures a collision rate.

**Theorem 2.12 and its proof are therefore the new object**, and §2.3's closing paragraph rests on
that rather than on an accumulation of smaller differences.

**Tone note for the final letter**: this must read as a factual delta, never as criticism of the
predecessors. Both are the present authors' own prior work; the point is what this paper adds, not
what they lacked.

---

## Two places where R3's characterisation needs a gentle correction

Both are worth making because R3 checked thirteen of thirteen checkable claims in round one, so
precision here is respected rather than resented.

1. **"exhaustive shortest-then-lexicographic normalization"** — the shortest-then-lexicographic
   criterion is exactly right and we adopt it unchanged. **We cannot confirm "exhaustive"** and
   deliberately claim nothing about [29]'s search space in Table 3.
2. **"an LSTM model"** — [29] in fact evaluates **LSTM and GRU**, and the task is **masked token
   prediction**, not classification. Relevant to R3.2 rather than to R3.1a; **T-14 owns it in §6.3.**

---

## Provenance

| Claim in this fragment | Source |
|---|---|
| 9 components; 3 inherited / 5 modified / 1 new | `artifacts/tab3_prior_work_delta.tex`; per-cell anchors in `PROVENANCE.md` |
| Zero formal results in either predecessor | orchestrator-verified absence counts, `VERIFICATION.md` |
| [28]'s assumed vertex order | `28.txt:89` |
| [28]'s canonical = greedy output | `28.txt:150`, `:165–166` |
| [29]'s one-directional argument and its assumption | `29.txt:145` |
| [29]'s shortest-then-lex criterion | `29.txt:141`; corroborated `isalchemutilities.py:486–488` |
| "exhaustive" unverifiable | `29.txt:145` + all three algorithm listings stripped from the CC-BY text |
| [29] uses LSTM **and** GRU; token prediction | `29.txt:371` |
| Sufficiency paragraph, 145 words | `artifacts/sufficiency_paragraph.tex` |
| Table measured at 0.67 p | `elsarticle[review]` harness, `REPORT.md` §1 |
| `24,764,422` collision-free pairs | **T-06**, not T-07 — `prose.md` claim register C1 |

**Not in this fragment, deliberately**: any criticism of [29]'s public implementation, and any
statement about [29]'s normalisation search space. See `T-07-article-notes.md` §5.
