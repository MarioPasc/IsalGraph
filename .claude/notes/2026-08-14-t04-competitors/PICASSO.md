# T-04 on Picasso — environment, and what the smoke actually gates

**Orchestrator only.** No subagent ssh'd, rsync'd or submitted anything. Design note §7.

Measured 2026-08-15 from `picasso3` (login node). Verification script:
`scratch_t04/picasso_verify.py`, mirrored from
`.claude/notes/2026-08-14-t04-competitors/picasso_verify.py`.

---

## 1. Stop-condition 2 is closed — `pynauty` builds from source under gcc 12.2.0

This was the one thing the Picasso run exists to gate (design §7): a failure would take
`nauty_graph6`, `sparse6_nauty` **and** AGM's orbit pruning down together, and would change
`k` in `N_actual = 182 − 15k − 8d`.

```
module load gcc/12.2.0          # gcc (GCC) 12.2.0 -- the login-node default is 7.5.0
pip install --no-binary :all: --no-cache-dir pynauty==2.8.8.1
  -> Building wheel for pynauty (pyproject.toml): finished with status 'done'
  -> Successfully installed pynauty-2.8.8.1
```

**Built, not downloaded**, and the results are **byte-identical to the workstation**:

| | Picasso (from source, gcc 12.2.0) | Workstation (cp311 wheel) |
|---|---|---|
| `nauty_graph6` `G` | `'E@ro'` | `'E@ro'` |
| `nauty_graph6` `H` | `'E@po'` | `'E@po'` |
| `\|Aut(G)\|` | `4.0` | `4.0` |
| K₃,₃ | `'Es\o'` | `'Es\o'` |
| prism | `'E{Sw'` | `'E{Sw'` |

The `canon_label` inversion guard — `pos = {old: new for new, old in enumerate(lab)}` then
`assert nx.is_isomorphic(G, relabelled)` — was exercised on every one of those and passes.
That guard is the one that matters: inverting `canon_label` yields a *different but still
deterministic* labelling, which **passes F3** and is wrong.

`module load` needs a **login shell** over ssh (`ssh picasso 'bash -lc "..."'`); a plain
`ssh picasso 'module load ...'` silently does nothing and leaves you on gcc 7.5.0.

---

## 2. ⚠ `grakel` does not work on Picasso, and it is not worth fixing there

`isalgraph`'s Picasso env carries **numpy 2.4.6** (the workstation has 1.26.4). grakel 0.1.10
does not support numpy ≥ 2, in two distinct ways:

| Install route | Failure |
|---|---|
| PyPI wheel | `ImportError: numpy.core.multiarray failed to import` — the Cython extension is built against the numpy-1 ABI |
| **rebuilt from source** (`--no-binary grakel`, gcc 12.2.0) | `ImportError: cannot import name 'ComplexWarning' from 'numpy'` — a **source-level** incompatibility; `ComplexWarning` moved to `numpy.exceptions` in numpy 2 |

So the break is not an ABI mismatch a rebuild fixes; grakel 0.1.10 is simply numpy-1 code.

**Decision: accept it and state it.** Three reasons:

1. **It is not what the Picasso run gates.** Design §7 names `pynauty`'s from-source build,
   and that passes.
2. **The WL backend does not need grakel.** Track C ships **two** implementations, and the
   shared-vocabulary one is pure Python with no numpy dependency. grakel is a *cross-check*,
   and it is a workstation-side one: both implementations already agree there to four
   decimals on all five Suite-1 datasets, at `h = 2` and `h = 3`
   ([WAVE0-FINDINGS](WAVE0-FINDINGS.md) W0-1).
3. **The alternative is worse.** Pinning `numpy < 2` in that env would reach straight into
   T-05's live GED work, and a separate env costs 20–40k inodes against an `fscratch` file
   quota already at **226.7k / 250.0k** (hard limit 400.0k).

> **Inherited by T-06.** Any WL number computed *on the cluster* must come from our own
> implementation, or from a dedicated env with `numpy < 2`. Reaching for grakel there will
> fail at import, loudly — which is the good case — but the reason belongs on record now.

---

## 3. What else is installed and verified

| Item | Picasso | Workstation | Note |
|---|---|---|---|
| Python | 3.11.15 | 3.11.15 | ✔ |
| `networkx` | 3.6.1 | 3.6.1 | ✔ |
| `numpy` | **2.4.6** | **1.26.4** | differs; only `.npz` reading depends on it here |
| `scipy` | 1.17.1 | 1.17.1 | ✔ |
| `rapidfuzz` | 3.14.5 | 3.14.5 | ✔ — symbol-level `1`, character-level `4`, verified both sides |
| `pynauty` | 2.8.8.1 **from source** | 2.8.8.1 wheel | ✔ byte-identical output |
| `grakel` | 0.1.10, **unusable** | 0.1.10 (stale `__version__` string reads 0.1.8) | §2 |
| `isalgraph` | **not installed in that env** | editable, engine `cpp` | see §5 |

---

## 4. Cohorts — all ten, on Picasso, with the locked counts

Mirrored into the layout `datasets.py` expects, under
`$ISALGRAPH_COHORT_ROOT = fscratch/datasets/isalgraph`:

```
exported/         iam_letter_low iam_letter_med iam_letter_high linux aids
exported_suite2/  grec aids_iam coil_del mutagenicity protein
```

Verified counts: Letter LOW **1180** · MED **1253** · HIGH **2059** · LINUX **89** ·
AIDS **769** · GREC **650** · AIDS-IAM **1811** · COIL-DEL **3900** · Mutagenicity **4040** ·
Protein **569**. GREC 650 and AIDS-IAM 1811 reproduce T-01's retained counts exactly.

> **These are also how Suite 2 came back.** The raw `IAM_Database/extracted` GXL tree is
> **no longer on the workstation** — the design note assumed it was — and the exported CSR
> `.npz` files were recovered from here on 2026-08-15. Without that, five of the ten Claim A
> rows and the whole AGM ceiling table would have been unreachable locally.

---

## 5. Open for wave 2 — the smoke session itself

The loginexa run (design §8 criterion 8) still needs, and none of it is on the critical path
until the three tracks merge:

1. **`isalgraph` importable in the Picasso env.** The repo is at `fscratch/repos/IsalGraph`
   but is not `pip install -e`'d into `conda_envs/isalgraph`. For the competitor backends a
   `PYTHONPATH` is enough — they are pure Python. For the **reference arm** it is not: the
   C++ engine must be built there (`-march=x86-64-v3`, never `native`), or the arm must be
   run with `Budget(timeout_s=None)`, since `timeout_s` is a `cpp`-only parameter and the
   backend now **refuses** a budget it cannot enforce rather than dropping it.
2. **`.claude/loginexa.yaml`** — does not exist yet; written on first invocation of the
   `test-picasso-loginexa` skill.
3. The run itself: one interactive loginexa session, 30-minute wallclock, **no queue**, so
   SCBI's 2-hour job floor does not apply. `smoke.py` for all eleven backends against one
   Suite-1 and one Suite-2 dataset, then each agent gets its JSON slice by `SendMessage`.

**Quota watch**: `fscratch` is at **0.47 TB / 1.40 TB** space but **226.7k / 250.0k files**
(hard 400.0k). The limit that bites is the file count, and it is 91 % of the soft quota
already. Nothing in T-04 should create a build tree there.
