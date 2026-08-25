# T-09 response-letter fragment

Answers **R3.7c**. Draft prose for **T-14** to place; not final letter text.

---

## R3.7c — "would benefit from a schematic of the canonical search space"

> We have added the schematic (new Figure ~2). It is drawn by the same enumerator the
> canonicalisation uses, and a test asserts that the enumerator reproduces
> `canonical_string` on six graph families, so the figure cannot drift from the algorithm it
> depicts.
>
> Preparing it exposed an error in the prose the figure accompanies. Remark 2.7 stated that
> *only* the identity of the uninserted neighbour chosen at each `V`/`v` step contributes to
> the search space. Definition 2.6 defines `w*_G` as the shortest string encodable from
> **any starting node**, and the implementation searches both. On the six-node example the
> figure now uses, the six starting nodes give six distinct strings of lengths 9, 10, 9, 11,
> 10 and 10; only one attains the minimum. The remark has been corrected to name both
> branch sources, and the figure shows one subtree per starting node accordingly. What the
> remark was there to say — that the displacement ordering `P(M)` and the priority
> `V ≻ v ≻ C ≻ c` are intrinsic to the algorithm and are never branched over — is unchanged
> and is the distinction the figure draws.
>
> We have also added a two-panel worked example of the two conversions on the same graph
> (new Figures ~3 and ~4). The reviewer did not request these; we include them because the
> round-trip property is the paper's foundational claim and it is far easier to check
> visually than to follow through Example 2.3's prose, which they replace. Should the page
> limit require it, these two are the first figures we will withdraw.

---

## Provenance

Every claim above, with the artifact that produced it.

| Claim | Source |
|---|---|
| Schematic exists; drawn by the canonicalisation's own enumerator | `src/isalgraph/viz/search_tree.py::canonical_search_tree_figure` |
| Enumerator reproduces `canonical_string` on six families | `tests/viz/test_search_tree.py::test_enumerator_agrees_with_canonical_string` |
| Remark 2.7 excludes the start node; Definition 2.6 includes it | `methodology.tex:462` vs `:456`; `src/isalgraph/core/canonical.py` |
| Six start nodes → lengths 9, 10, 9, 11, 10, 10; one attains the minimum | [T-09 article notes](T-09-article-notes.md) §1; `tests/viz/test_worked_example.py` |
| Worked-example panels replace Example 2.3's prose trace | [manuscript](../plan/manuscript.md) §3.1 recovery table |
| Figures 3 and 4 answer no reviewer demand | [decisions](../plan/decisions.md) S-g; [manuscript](../plan/manuscript.md) §3.2 rank 11 |

---

## Not to be written into the letter

- The graphical abstract. T-09 did not regenerate it, and its panel (b) still carries
  numbers T-06 retired. **T-24** owns it; see [T-09 article notes](T-09-article-notes.md) §6.
- The CDLL pointer-arrow rendering defect (article notes §5). It is a defect in figure code
  written *for this revision*, not in anything the reviewers saw. Reporting it invites a
  question about figures that were never wrong.
- Any page-count promise for the new figures. **T-26** owns that arithmetic and the
  inventory in `manuscript.md` §2 is stale on this row.
