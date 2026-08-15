# Wave 1 — orchestrator verification

**A log is a claim, not evidence.** Every number below was re-run by the orchestrator, in the
agent's own worktree or in the main checkout, not read from the agent's log. Where the
orchestrator's independent check used a different method than the agent's, that is stated.

Base commit: `9d2291b87b37c9b4fbee9ce994f779dfd69db01c`.

---

## Track A — `competitor-serial` · verdict **ACCEPT**

Branch `worktree-agent-a6e805f2cd1aa441f`, 3 commits, `git status --porcelain` empty.
`git diff --stat` matches the log's file list exactly: three backends, one test file, one work
log, **1,720 insertions, nothing outside its ownership**. The `sitecustomize.py` / `wtpy` shim
is present in the worktree and correctly uncommitted.

| Check | Method | Result |
|---|---|---|
| Its own suite | re-run: `./wtpy -m pytest tests/unit/test_competitors_serial.py` | **143 passed, 5 xfailed** — matches the log |
| Criterion 1, running example | orchestrator, direct encode | `adjacency` `'101101000100011'` / `'101001000100011'`, `graph6` `'ElCW'` / `'EhCW'`, `sparse6` `'EaWIzR'` / `'EaYms'` — **all exact** |
| Criterion 6, family identity | orchestrator unpacked graph6's payload independently | **7/7 fixtures agree bit for bit** |
| sparse6 `':'` accounting | orchestrator | `wire = b':EaWIzR'`, `entropy = 36.0 = 6·7 − 6`; delta against `6·len` is **exactly 6** |

The five `xfailed` are the `bits.py` defect the agent carried as `xfail(strict=True)` rather
than fixing in a file it does not own. That is the correct behaviour and it is why the defect
could not be missed.

### Its four findings, each re-derived independently

1. **`bits.py` halved every adjacency and AGM `realised_bits`. CONFIRMED, fixed by me.**
   The frozen `8·⌈n(n−1)/16⌉` is `8·⌈T/8⌉` because `n(n−1) = 2T`; the code read the 16 as a
   word size. Measured: `n=6` gave 8 bits for a 15-bit triangle, `n=98` gave 2,384 where 4,760
   is right. `entropy_bits` unaffected, so **Claim A's headline table never moved**; the
   realised-bytes column did. Fixed in `c33fd13`.
   → **It also exposed that design criterion 4 was never satisfiable**:
   `realised_bits < len(e.text)` cannot hold when packing `T` bits into bytes costs ≥ `T` bits.
   It passed only because of the defect. Criterion amended to `realised_bits < 8·len(text)`
   and `entropy ≤ realised < entropy + 8`.

2. **sparse6 Claim A is exactly 6 bits below README §4.3, on all ten datasets. CONFIRMED.**
   §4.3 was computed as `6·len(code)` **including** the `':'`; §4.2 then froze
   `6·len(wire) − 6`, deliberately, because the prefix is framing and not payload. The two are
   both defensible and they differ by exactly one character. **The frozen convention wins**;
   §4.3's sparse6 column is restated −6 bits under gate 1b.

3. **`nx.convert_node_labels_to_integers(ordering="sorted")` does not pin the labelling.
   CONFIRMED, independently.** The agent measured 260/300; the orchestrator measured
   **290/300** disagreement against a genuine sorted rebuild, on its own scrambling. It
   renames node *values* and leaves insertion order, and `to_graph6_bytes` re-derives labels
   from insertion order. `graph6.md` §7 prescribes it, CONTRACTS.md repeated it, and both are
   wrong. Worse, it made `graph6` and `sparse6` serialise **different** labellings, which is
   what the scout's code did. Relayed to track B mid-flight, because its criterion 7 compares
   AGM's identity permutation against A's reading order.

4. **F3's Letter successes are not a sampling artefact. CONFIRMED, and sharper than reported.**
   The agent enumerated all `n!` relabellings and found the 20-draw harness returns the same
   counts. The orchestrator enumerated **every connected graph on `n = 2…6`**: exactly **5**
   are invariant under every relabelling, and **all 5 are complete graphs**. The reason is a
   theorem, not a sample — the strict upper triangle is relabelling-invariant iff the
   adjacency matrix is constant off-diagonal. **So F3 for the non-canonical formats measures
   the fraction of complete graphs in the draw**, which is why Letter (many `K₂`/`K₃`) scores
   4–9/50 and LINUX/AIDS score 0/50. That is a far better statement than the folder's, and it
   belongs in the paper's F3 caption.

---

## Track C — `competitor-mining` · verdict **ACCEPT**

Branch `track-C-mining`, 4 commits, `git status --porcelain` empty.
`git diff --stat` matches: two backends, two test files, one work log, **2,508 insertions,
nothing outside its ownership**. `benchmarks/real_data/eval_setup/wl_kernel_computer.py` is
**untouched**, as instructed.

| Check | Method | Result |
|---|---|---|
| Its own suite | re-run: `./wtpy -m pytest tests/unit/test_min_dfs.py tests/unit/test_wl_subtree.py` | **75 passed, 1 skipped** — matches |
| Criterion 2, the corrected WL identity | orchestrator, fresh instances per `h` | `ours(h=1) = 2.0`, **`ours(h=2) = 5.830951894845301`**, `ours(h=3) = 7.211102550927978` — identical to the wave-0 values, so grakel's `n_iter = k ≡ h = k` holds in the shipped code |
| Criterion 7, the completeness witness | orchestrator | `d = 0.0` **exactly** at `h = 1, 2, 3, 5`; min-DFS separates them |
| Criterion 4, complete invariant | orchestrator enumerated every connected graph `n = 2…6` | distinct codes **`[1, 2, 6, 21, 112]`** = OEIS A001349, **zero collisions** |
| min-DFS length | orchestrator | `\|code\| = m = 7` on the running example, tuple-level symbols |

### Its three non-closures, all diagnosed and none tuned

1. **Criterion 6, the `kaviniitm` gate** — the binary is not in the repo and the agent did not
   clone it (correct: "vendor nothing"). Instead it re-verified all seven archived K1
   counterexamples independently — each is a *valid* DFS code strictly larger than the
   minimum — shipped the gate parameterised with **K2 first**, and proved the gate can fail
   using a greedy no-branch candidate. The differential re-runs under `KAVIN_DFSCODE_BIN`.
   **Accepted**: the reusable artefact is the gate, and it exists and can fail.

2. **Criterion 8, WL's false zeros** — Letter 0/0/0 and LINUX 1 reproduce; **AIDS is 11, not
   ≈6**. Two causes, both quantified by the agent: its denominator is all 19,900 pairs against
   the folder's 15,686 certified ones (rescaling to ≈8.7), plus the corrected WL. Consistent
   with W0-1 and with the brief's instruction to treat that figure as a prior.

3. **Criterion 11, Claim A** — 9 of 10 exact; **Protein 620.0 vs 615.0**. Cause: a different
   400-graph draw. `real_suite2.py` loads Suite 2 from raw IAM GXL, which no longer exists on
   this workstation; the cohort came back as exported `.npz` from Picasso, whose enumeration
   order differs. See the open question below.

### The E10 measurement, which is the deliverable T-06 inherits

`h = 2` versus `h = 5` over 60 graphs / 1,770 pairs: feature dimension grows **4.9× / 17.8× /
24.1×** (Letter LOW / LINUX / AIDS), Spearman between the two distance vectors 0.86–0.91, and
**11.2 % / 16.6 % / 13.6 %** of pair orderings flip. The decisive number: `frac(d = 0)` is
**identical** at both — three extra refinements separate **zero** additional pairs. That
argues for `h = 2` on cost alone, **without touching ρ**, which is exactly the F5-blind form
the selection rule requires.

### Two more defects in the orchestrator's `smoke.py`, both real, both fixed

- **`_f3` caught only `CompetitorError`.** min-DFS raises a plain `ValueError` on a
  disconnected graph, so it escaped and would have aborted the whole smoke run — every other
  backend's numbers lost to one graph the cohort filter was supposed to exclude.
- **`_f3` fitted a `VectorBackend` one graph at a time** — the per-batch trap at batch size
  one. Harmless for a WL whose features are fit-independent, silent corruption for any future
  `VectorBackend` that is not.

Both fixed in `541320f`.

---

## Open, for the batched escalation

- **Criterion 1 cannot be met "exactly" on the five Suite-2 Claim A rows.** They are medians
  over a **400-graph draw**, and the draw is not reproducible: `real_suite2.py` sampled from
  the raw IAM GXL tree, which is gone from this workstation, and the recovered `.npz`
  enumerates in a different order. Coarse statistics survive it — track A hit 10/10 on
  `adjacency` and `graph6`, whose bits depend only on `n` — and finer ones do not: min-DFS's
  `m·2⌈log₂ n⌉` missed Protein by 5 bits. **The five Suite-1 rows are full-cohort and remain
  exactly reproducible.**
- **sparse6's −6 bits** (track A finding 2): frozen convention versus §4.3 as printed.
