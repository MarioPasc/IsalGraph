# Work log — track C (`t27-literature`), wave `2026-08-13-t27-bakeoff`

Base commit `4a31817`. Branch `worktree-agent-acea836daa7d77780`.

## Mission

Produce a verified citation, complexity and claimed-tightness table for the nine GED bound methods
T-27 measures, plus GEDLIB, and state the published tightness ordering as testable predictions
separating proven dominance from empirical observation.

## Files changed

| File | Change |
|---|---|
| `.claude/notes/review/tasks/T-27-literature.md` | created — the deliverable |
| `.claude/notes/2026-08-13-t27-bakeoff/literature.md` | created — this log |

Nothing else was touched. No code, no plan file, no `tickets.md`, no `CONTRACTS.md`. No `ssh`,
`rsync`, `sbatch`, `pip install`, build or test run. Nothing written under the repo's `scratchpad/`
(paper PDFs were downloaded to the session scratchpad under `/tmp`, outside the repo).

## What I verified and how, per citation

Verification order was: (1) GEDLIB's own method headers, to establish *which paper the code we ran
actually implements* — this is the authority for attribution and it settled two of the three
questions the brief raised; (2) Crossref, for publisher-deposited bibliographic fields; (3) the
primary full texts, for complexity, proof status, tightness and determinism.

### Step 1 — GEDLIB source, `dbblumenthal/gedlib` `master`

Fetched `src/methods/{branch,branch_fast,branch_tight,star,hed,ipfp,bipartite,refine,bp_beam,
ls_based_method,lsape_based_method}.hpp` from `raw.githubusercontent.com` and read the `@details`
Doxygen block of each, which names the paper and DOI the class implements plus the full option
table with defaults.

This is where the single most consequential finding came from: **`LSBasedMethod` — the base class of
`IPFP`, `REFINE` and `BP_BEAM` — defaults `--initialization-method` to `RANDOM` and `--randomness`
to `REAL`.** That is the documented explanation of the brief's `IPFP` anomaly (UB 3.00 on a 4-node
instance of true GED 1.00, disagreeing across machines). It is not a build difference and not a
GEDLIB bug. Relayed to `main` as it bears directly on track A's harness.

### Step 2 — Crossref, `api.crossref.org/works/<doi>`

Queried each DOI and read authors, title, container-title, volume, issue, page, year, type,
publisher. Two records are defective and are flagged in the deliverable:

- `10.1007/978-3-319-11656-3_11` (C6, BP_BEAM) — **corrupt**: `container-title` reports "Advanced
  Information Systems Engineering" and the ISBNs of LNCS 7908. Resolved from the printed page instead.
- `10.1007/978-3-030-20081-7_2` (C7, GEDLIB) — not resolvable by direct DOI lookup; found via
  bibliographic query, then completed from the Springer chapter page.

### Step 3 — primary full texts

| Citation | Read | Source |
|---|---|---|
| C2 Zeng et al. 2009 | ✅ full | `http://www.vldb.org/pvldb/vol2/vldb09-568.pdf` (PVLDB open access), `pdftotext` |
| C4 Bougleux et al. 2017 | ✅ full | authors' accepted manuscript, `bougleux.users.greyc.fr/articles/ged-prl.pdf` |
| C5 Riesen & Bunke 2009 | ✅ full | archived publisher PDF via Wayback |
| C6 Riesen et al. 2014 | ✅ full | archived Springer PDF via Wayback |
| C7 GEDLIB 2019 | ✅ metadata | Springer chapter page, "Cite this paper" block |
| C8 Blumenthal et al. 2020 | ✅ full | authors' OA PDF, `bougleux.users.greyc.fr/articles/vldbj19comparing-heuristics-ged.pdf` |
| C1 Blumenthal & Gamper 2018 | ❌ | no OA copy located; no arXiv record. Claims taken from C8 and attributed to C8. |
| C3 Fischer et al. 2015 | ❌ | Elsevier paywall; see below |

C2, C5, C6 and C8 were the sources for every complexity, proof-status and tightness statement I
attribute to a primary paper. Where only C8 was readable (C1, C3), the deliverable says so in the
row rather than implying the primary source was consulted.

Three `read-paper` agents were run in parallel for C3, C4 and C5+C6, each given a fixed list of
questions and instructed to mark anything not visible in the text as UNVERIFIED rather than fill it
from memory. All three complied; the C3 agent correctly returned a failure rather than a guess.

### Findings that changed what the deliverable says

1. **`IPFP` is in *Pattern Recognition Letters* 87:38–46, 2017 — a *Pattern Recognition* family
   journal, and a special issue on graph-based pattern recognition.** The brief asked me to flag
   this prominently if true. It is true, and it means the EiC venue-fit argument no longer rests on
   HED alone: C3 (*Pattern Recognition*), C4 and C9 (*PRL*) give three family citations, one of
   them the manuscript's own venue.
2. **`REFINE` is Zeng's method, not GEDLIB's.** `refine.hpp` says so explicitly, and it is the
   *same paper* as `STAR` — one citation covers two methods. GEDLIB's header additionally
   attributes K-Refine to Zeng, which C8 Table 4 contradicts (K-REFINE is Boria et al.); immaterial
   for us since `--max-swap-size` defaults to 2.
3. **`BP_BEAM` is Riesen, Fischer & Bunke 2014, LNAI 8774:117–128 — not "Neuhaus & Riesen".** The
   plan's attribution is wrong.
4. **`BRANCH` and `BRANCH_FAST` are provably *equal*, not merely ordered, under our cost model.**
   C8 §5.2.4: "BRANCH-FAST's lower bound is never tighter than the one computed by BRANCH. For
   constant edge edit costs, BRANCH and BRANCH-FAST are equivalent." D6 has constant edge edit
   costs. This is stronger than the brief anticipated and makes P1 an exact elementwise equality
   over 3.9 M pairs with zero tolerance — the cheapest and sharpest harness validation available.
5. **`BED ≥ HED` is PROVEN, not empirical** — answering `main`'s question directly. C8 §8.1: HED
   sums row and column minima of the same LSAPE instance BRANCH solves, and `LB ≤ LSAPE(C)`. A
   violation in our data is therefore a harness bug, not a finding.
6. **`STAR` carries a cost-model precondition no other method has** — uniform `c_V` and `c_E`
   (C8 §5.2.6, Table 4). D6 makes substitutions free, so the precondition holds only because our
   graphs are effectively unlabeled. If any dataset carries genuine labels, STAR's Lemma 4.2
   guarantee lapses and its "lower bound" may exceed the exact GED.
7. **`C4` does not claim IPFP is the tightest upper bound.** GNCCP beats IPFP on 3 of C4's 4 GREYC
   datasets and Neuhaus–Bunke beats it on MUTAG-20/50 under one cost regime. The strong claim is
   C8 §9.6's, made with a uniform reimplementation *and multi-start*. A single-start IPFP losing on
   our data would not contradict C8.
8. **C6 and C8 disagree about BP-Beam's node ordering.** C6 §5 says the swap order is **fixed** and
   lists variable ordering as future work; C8 §7.2.3 says the method "starts by producing a random
   ordering". The randomisation is GEDLIB's/C8's, not C6's — recorded that way.
9. **`BIPARTITE` is not tie-free.** C5's Algorithm 2 line 11 selects "an **arbitrary** uncovered
   zero". Distinct optimal LSAPE solutions have equal assignment cost but different induced GED
   upper bounds. C5 never discusses it. The `BIPARTITE` column is reproducible only against a fixed
   `--lsape-model` and GEDLIB build.

## Corrections found to `approx_ged.md` §5 (and §2)

Recorded, **not applied** — plan files are the orchestrator's. Full text in deliverable §6.

1. "Bougleux et al., 2017 (IPFP — our UB)" has **no venue, volume or pages**. → *Pattern Recognition
   Letters* **87**:38–46, 2017, `10.1016/j.patrec.2016.10.001`, special issue.
2. "Zeng et al., *VLDB* 2009" → *Proceedings of the VLDB Endowment* **2**(1):25–36, 2009,
   `10.14778/1687627.1687631`. "VLDB 2009" reads as the conference; it is the journal. §5 lists Zeng
   only against `STAR`, but the same paper is also the source of **`REFINE`**.
3. "Blumenthal et al., GbRPR 2019" lacks its volume → **LNCS 11510:14–24**, eds. Conte, Ramel &
   Foggia, Springer Cham.
4. §2 lists `BP_BEAM` as "**Neuhaus & Riesen**" with no year or venue. **Wrong** → Riesen, Fischer &
   Bunke, ANNPR 2014, LNAI **8774**:117–128, `10.1007/978-3-319-11656-3_11`.
5. **§5 omits C8 entirely.** Every complexity in §2's tables and both dominance claims in its
   reason 1 (`BED ≥ LED`, `BED ≥ HED`) come from the *VLDB Journal* survey, not from C1. Without C8
   in the bibliography those numbers are uncited.
6. §2's complexities `O(n²Δ² + n³)` (BRANCH_FAST) and `O(n²Δ³ + n³)` (BRANCH) are a defensible
   simplification of C8's forms but **drop the `Δ_max log Δ_max` sorting term** and collapse
   `Δ_min`/`Δ_max`. If a complexity reaches the manuscript, quote C8's form and cite C8 §5.2.3/§5.2.4.

### Additional corrections to `gedlib.md` §5 — for `main` to propagate alongside the above

Prompted by `main`'s mid-task message, and confirmed independently against C8 §8.1 and `hed.hpp`:

1. **`HED`'s `get_upper_bound() = inf` is correct behaviour, not a defect.** C8 §8.1: HED's bound
   "does not correspond to a feasible LSAPE solution, because of which HED does not compute an upper
   bound for GED"; C8 Table 4 lists HED as `upper bound: no`. §4's "Trap 2" table should move `HED`
   into the **lower-bound** row rather than listing it as an anomaly, and §5's "unresolved, do not
   use yet" is now wrong on both counts.
2. The "needs explicit method options" guess was right, and the reason is worth recording: the
   option is `--edge-set-distances OPTIMAL` (default `HED`), and the default degenerates because
   **D6 makes edge substitution free**. It is a cost-model interaction, not a library defect, and it
   would not reproduce under the IAM per-dataset cost models.
3. `main`'s half-integer question is answered from the source: two independent halvings compose.
   BRANCH's LSAPE instance already carries edge costs at ½ (C8 §5.2.3, each edge charged at both
   endpoints), and HED halves again over row and column minima (C8 §8.1). Granularity is therefore
   **0.25**, which is why 0.50, 1.25 and 1.75 appear.
4. `main` asked me to check HED's determinism from the source rather than infer it from the method
   family. Done: `hed.hpp` exposes only `--threads`, `--lsape-model` and `--edge-set-distances`.
   **Deterministic** — it is a closed-form sum over row and column minima, not a search.

### One item for `statistics.md` D6

D6 is presented as universally applicable. It is not, for one method: `STAR` requires **uniform**
`c_V` and `c_E`, and D6 satisfies that only for effectively unlabeled graphs (finding 6 above).

## Deviations from this brief

None material. Two notes:

- The brief asked for "one row per method". I wrote a compact master citation table (§1) plus a
  per-method detail block (§2–§4), because complexity, proof status, tightness and determinism do
  not fit legibly in one row and the proven/empirical distinction needed room to be unambiguous.
- I spawned three `read-paper` subagents for the primary texts I had not yet read, rather than
  reading them in-thread. This is the context-firewall pattern; each was given a fixed question list
  and an explicit instruction to return UNVERIFIED rather than recall.

## Left UNVERIFIED, and why

| Item | Why |
|---|---|
| **C3 (Fischer, HED) full text** | Elsevier paywall. Tried: `doi.org` → `linkinghub.elsevier.com`; ScienceDirect (403 bot wall); Unpaywall/OpenAlex, which list exactly one OA location — BORIS (Univ. Bern, `10.48620/99172`), whose PDF bitstream returns `401` via the REST API and an anti-bot wall via the UI despite metadata marked `open.access`; FHNW `irf.fhnw.ch` (record present, zero bitstreams); arXiv, CORE, fatcat, Wayback, scholar.archive.org — no copy. Pirate mirrors not used. **Verified from the abstract/highlights**: the "quadratic time" claim and the framing of BP as "cubic time". **Unverified**: exact complexity expression and section; whether the lower-bound property is a numbered theorem; tightness table/figure numbers; determinism. Remaining route: UMA library / interlibrary ScienceDirect access. |
| **C1 (Blumenthal & Gamper, TKDE 2018) full text** | No OA copy located; no arXiv record for the title. The bibliographic record is fully Crossref-verified. All C1-derived claims are attributed to **C8**, the survey by the same first author, not to C1. |
| **C10 LNCS volume** | Same corrupt-Crossref problem as C6. Optional reference — IBP-Beam is off by default (`--num-orderings 1`) — so not pursued further. |
| **C11 (Leordeanu et al., NeurIPS 2009)** | Transcribed from C4's bibliography, not from a NeurIPS proceedings page. Optional; cite only if the manuscript explains what IPFP is rather than merely using it. |
| **`BRANCH_TIGHT` complexity exponents** | Read via `pdftotext` from C8's two-column PDF, where superscripts detach from their base. Reconstructed as `O(N³Δ_max² + I(N²Δ_max³ + N³))`; the shape is certain, the exponents should be confirmed against the typeset PDF before appearing in the manuscript. |
| **`BP_BEAM` composed complexity** | C8 §7.2.3's per-component costs are quoted verbatim; the final composed expression was truncated by the extraction. C6 states none at all. |

Nothing was filled in from memory anywhere in the deliverable.

## Open issues for the orchestrator

1. **Track A must pin the LS-GED initialiser.** `IPFP`, `REFINE` and `BP_BEAM` are non-deterministic
   under GEDLIB defaults (`--initialization-method RANDOM`, `--randomness REAL`). Pinning
   `--initialization-method BIPARTITE` additionally converts two empirical comparisons into proven
   per-pair inequalities (deliverable P11) — free harness validation for one option string.
2. **Run P1 (`BRANCH` = `BRANCH_FAST`, exact equality) before any tightness number.** Zero
   tolerance, 3.9 M pairs. If it fails, nothing downstream is trustworthy.
3. **The `HED` cell must record `--edge-set-distances OPTIMAL` in `meta.options`.** The default and
   the pinned setting are different estimators and a reader cannot tell them apart from the number.
4. **The write-up must state the IPFP configuration prominently.** C8's "always tightest" was
   measured with multi-start; a single-start result that loses is not a contradiction, but a
   reviewer will read it as one unless the configuration is on the page.
5. **`STAR`'s uniform-cost precondition** should reach both the T-27 write-up and `statistics.md` D6.
6. Whether to pursue C3's full text through UMA interlibrary access. HED's row currently stands on
   C8; that is defensible and correctly attributed, but the manuscript must not say "Fischer et al.
   prove …" until someone reads the theorem.
