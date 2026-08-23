# Track C — the audit path: supplementary table, selection table, `k`

**Written by the orchestrator**, not by the track. Track C's own `.md` write was blocked by a tool
policy in its environment; the content below is its report, verified by me against the diff and by
re-running every acceptance check myself. Where I ruled on something it raised, the ruling is marked.

**Commits**: `0bfdf4e`, `0059be6`, `b0c4773` on `ticket/T-04a`. **Base**: `7e96f4a`.

---

## 1. What changed

| File | |
|---|---|
| `src/isalgraph/competitors/report.py` | new, stdlib-only, 866 → 903 lines |
| `tests/unit/test_competitors_report.py` | new, 43 tests |

Constants defined **here and nowhere else**: `CLAIM_B_COMPARATORS`, `N_MAX = 182`,
`ROWS_PER_REPRESENTATION = 15`, `ROWS_PER_SUITE = {suite1: 5, suite2: 10}`.

Four artifacts: `supplementary_grid.csv` (all 66 cells), `selection.md`, `f5_table.md`, `k.json`.

**Beyond the brief, and kept**: `report.py` **recomputes** the selection from the cells under
`competitors.md` §3.4 and prints a *Recomputed* column plus a disagreements section, so a defect in
track A's selector surfaces instead of being adopted. An audit path that adopts the thing it audits
is not an audit path.

## 2. Verification I ran myself

| Check | Result |
|---|---|
| `pytest tests/unit/test_competitors_report.py -q` | **43 passed** |
| `ruff check` (both files) | clean |
| `mypy src/isalgraph/competitors/report.py` | clean |
| forbidden-import probe (no `grid`, `f5`, `numpy`, `scipy`, `networkx`) | `FORBIDDEN IMPORTS: []`, exit 0 |
| per-commit file list | only its two files, all three commits |

## 3. The three things it raised, and my rulings

**(1) A comparator absent from `primary_distance` → CHANGED to abort.** Track C initially treated a
missing key as `null`, which raises `k`. That conflates two cases the pre-registration charges
differently: *no admissible distance* is −15 and keeps the Claim A rows, while *not computable at
all* is −25 and §5 says it "is recorded separately". Folding an absent backend into `k` undercharges
it by 10 and mislabels it. It cannot arise here — all eleven backends are registered and available,
checked — so the tool now refuses to guess and exits 2 naming the backend and the charge. Four tests
cover it, including `test_the_abort_never_increments_k`.

**(2) A suite with `attempted == 0` is not charged — ACCEPTED, and it is live.** CONTRACTS §2 permits
a zero-count suite and `frac` is then `0.0`, which would charge 5 or 10 rows for a property of the
**draw** rather than of the representation. `S200` draws **zero LINUX graphs** — LINUX is 89 graphs
at `n ∈ [4,10]`, competing against thousands in the strata it falls in — so this is a real path.
Recorded in the design note changelog as track C's divergence.

**(3) A rendering defect that every content assertion passed through.** The size-null label
`` `|n1 - n2|` `` split its markdown cell on the pipe, silently adding columns to the table. Fixed to
`abs(n1 - n2)`, absence reasons moved to a footnote, and two tests now assert every row matches its
header width. Same class as the three defects this ticket was opened to fix: a plausible artifact and
no error.

## 4. The arithmetic, on fixtures

The 66-cell fixture run produced `k = 2` (`sparse6`, `adjacency`), two `partial` entries
(`agm_cam` / suite2 / 10 rows, `min_dfs` / suite1 / 5 rows), and
`n_actual_f2_before_d = 182 − 15·2 − 15 = 137`.

**The trap it flagged itself**: counting `sparse6_nauty`, `isalgraph_canonical`, `isalgraph_pruned`
or `size_null` into `k` would have given `k = 4` and 107 — **a 30-row error in the FDR denominator**.
`sparse6` is both a `k` member and sub-unity on suite2 and is charged **once**, via `k` only; the
`k → partial` precedence is what prevents the double-count.

These are fixture numbers. The real ones come from the `S200` run and are recorded in the ticket's
closing report.
