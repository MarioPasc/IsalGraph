# Adjacency matrix — the `n²` reference point

**Verdict**: **RUN — as the reference point, not as a rival.** It is the one representation every
reviewer already has in mind, it is the floor for the `n²` family, and it is free.

**Role**: [competitors](../competitors.md) §2 row 4 · serves **AE.4a**, **AE.3**, **R3.6a**
**Evidence**: measured on this workstation, 2026-08-13.
Cross-refs: [graph6](graph6.md) (the same bits, packed), [agm](agm.md) (the same bits, canonically
permuted), [README](README.md) §2.

---

## 1. Reproducibility — is anything blocking?

**Nothing.** It is `nx.to_numpy_array`, or nine lines of pure Python for the upper triangle. There
is no package to install, no version to pin, and no failure mode. Effort: **trivial**, exactly as
[competitors](../competitors.md) §2 estimates.

---

## 2. What the representation looks like

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`.

```
strict upper triangle, column-wise:  '101001000100111'      15 bits
same for H = G minus edge (0,3):     '100001000100111'
```

The convention matters and must be fixed once: **strict upper triangle read column-wise**,
`a(0,1) a(0,2) a(1,2) a(0,3) …`. This is byte-identical to graph6's payload and to AGM's CAM code
reading order, which is what makes [README](README.md) §2's family argument work. Choosing
row-major instead would break the correspondence for no benefit.

| Test | Result |
|---|---|
| 200 relabellings of the running example | **122 distinct bit strings** |
| F3 sweep: 40 graphs × 25 relabellings | **0 / 40 invariant** |

| Property | adjacency | IsalGraph pruned |
|---|---|---|
| Reversible | **yes**, exactly | yes, up to isomorphism |
| Isomorphism-invariant | **no** (0/40) | **yes** (40/40) |
| Complete invariant | no | yes, within a directedness class |
| Alphabet | `{0,1}` — the only binary member of the pool | `\|Σ\| = 9` |
| Length | **`n(n−1)/2` exactly**; a function of `n` alone | data-dependent |
| Handles disconnected | **yes** | no |
| Handles isolated vertices | **yes** | n/a |
| Encode cost | `Θ(n²)`, microseconds | 0.01–1.02 ms (pruned) |

---

## 3. Which distance does it accept?

This is the **only** representation in the pool for which the padded-Hamming convention in
[competitors](../competitors.md) §3.2 is natural rather than improvised: the code is a positional
bit vector indexed by vertex pairs, so embedding two graphs in a common `max(n₁,n₂)` frame is
well defined and its justification — that the padding coincides with the node insertions D6
charges — actually holds.

| Candidate | F1 | Note |
|---|---|---|
| Hamming | 100 % of equal-`n` pairs, 0 % otherwise | length is `n(n−1)/2` |
| **padded Hamming** | **100 %** | **the only pool member where this is principled** |
| Levenshtein | 100 % | defined but semantically odd on a positional vector |

Measured, 120 one-edit pairs (unit GED = 1, edited copy randomly relabelled) vs 120 random same-`n`
pairs:

| | median Lev, GED = 1 | median Lev, random | **separation** | median Ham, GED = 1 | median Ham, random |
|---|---:|---:|---:|---:|---:|
| adjacency | 12.0 | 13.0 | **0.92** | 17.0 | 18.0 |

Separation **0.92**: a one-edit pair is nearly indistinguishable from an unrelated one, because the
labelling dominates the content. **Fails F3, takes no primary distance, excluded from the running
Claim B comparison** and reported in the supplementary grid with the reason printed.

> Note the ordering across the three non-canonical formats: graph6 1.00, adjacency 0.92,
> sparse6 0.88. None carries usable signal, and the small differences are packing artefacts, not
> structure. Report them as one finding, not three.

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes, and it is the natural denominator.** [competitors](../competitors.md) §5 already assigns it
the entropy bound `n(n−1)/2` — no alphabet argument needed, since the symbols *are* bits.

| Convention | Value |
|---|---|
| Entropy bound | **`n(n−1)/2`** |
| Realised bytes | `8 · ⌈n(n−1)/16⌉` if packed two-per-byte… — **do not** invent a packing. Use `⌈n(n−1)/16⌉` bytes, i.e. the triangle packed 8 bits to a byte, and say so |

Measured entropy-bound bits, median of 5 per Suite-2 profile:

| Profile | `n` | `m` | **adjacency** | graph6 | sparse6 | min-DFS | IsalGraph pruned | GED construction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 4 | 3 | **6** | 12 | 24 | 12 | 13 | 18 |
| LINUX | 9 | 8 | **36** | 42 | 66 | 64 | 38 | 80 |
| AIDS (IAM) | 14 | 15 | 91 | 102 | 102 | 120 | **82** | 148 |
| COIL-DEL | 22 | 54 | **231** | 240 | 348 | 540 | 418 | 615 |
| Mutagenicity | 29 | 30 | 406 | 414 | 222 | 300 | **181** | 358 |
| Protein | 32 | 61 | 496 | 504 | **396** | 610 | 533 | 702 |

> **The adjacency matrix beats IsalGraph on 7 of the 10 cohort profiles.** This is the single most
> load-bearing number in the folder and it must reach the paper. It is not a defeat: `n(n−1)/2` is
> minimal *when the graph is dense and small*, and Suite 1 is exactly that regime — `n̄` between
> 4.07 and 11.03, density 0.218–0.607. The submitted manuscript compared only against the
> author-defined GED construction (`B_GED`, right-hand column), which the adjacency matrix also
> beats everywhere. **R3.6a's "narrow the claim accordingly" applies to us at least as much as the
> reviewer knew.**
>
> The claim that survives measurement: *IsalGraph's message length is governed by `m`, so it is
> shorter than every `n²` serialisation once the graph is large and sparse — the regime AE.1 asked
> us to extend into — and longer when it is small or dense.*

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned as the reference point.** R1.2 asks how IsalGraph "differs conceptually from existing
representations"; the adjacency matrix is the representation the answer has to start from, and
`introduction.tex:11–17` already cites it. Its inclusion costs nothing and its omission would be
conspicuous.

**IsalGraph's advantage**: canonicity (40/40 vs 0/40), completeness as an invariant, and `m`-scaling
instead of `n²`-scaling — real, and worth exactly the three profiles where it wins.

**What the adjacency matrix has that IsalGraph does not**: exactness, universality (disconnected,
isolated, directed, weighted, labelled — all without ceremony), a length that is a closed form, a
decode with no search, and **fewer bits on 7 of 10 profiles**.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Yes, trivially.** No dependency at all |
| 2 | Representation | `n(n−1)/2` positional bits; **not** isomorphism-invariant (0/40) |
| 3 | Distance | **padded Hamming** is principled here and only here; separation **0.92**, no GED signal |
| 4 | Claim A? | **Yes — the denominator.** `n(n−1)/2` bits; beats IsalGraph on **7 of 10** cohort profiles |
| 5 | Scope | **In, as the reference point.** Omitting it would be conspicuous |
| — | IsalGraph advantage | **Canonicity yes; compactness only at low density** |

---

## 7. For the integration agent

- Fix the reading order to **strict upper triangle, column-wise**, and assert it against
  `nx.to_graph6_bytes` on a few graphs — the two must agree bit for bit, and that assertion is what
  keeps [README](README.md) §2's family claim true in code rather than in prose.
- Report `n(n−1)/2` for the entropy bound. **Do not** report the realised byte count as
  `len(str) * 8` — the string `'101001…'` is a debugging view, not a serialisation, and counting it
  as 8 bits per character would inflate the adjacency matrix by 8× and hand us a baseline we beat
  for free. This is the single easiest way to produce a wrong Claim A table.
- The padded-Hamming frame must be built from the **canonical** labelling when the pair comes from
  a canonical backend, and from the incident labelling otherwise. Mixing the two silently compares
  different things.
