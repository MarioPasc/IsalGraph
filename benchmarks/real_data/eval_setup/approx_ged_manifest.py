"""Write ``manifest.json`` and ``PROVENANCE.md`` for the published APPROX_GED tree.

T-05 design §3.1 requires one manifest covering every published data file and one human-readable
provenance note beside it. Acceptance criterion 5 additionally requires that the manifest record the
frozen options string for every role, because GEDLIB's upper bounds change on 91.5-93.6 % of pairs
between runs at library defaults: a value without its options string is not a reproducible
measurement.

The manifest is built by *reading the files*, never by restating the design. Where a file's own
metadata disagrees with the frozen specification the manifest records both and flags the file, so a
mismatch surfaces here rather than in a reviewer's re-run.

Usage::

    python -m benchmarks.real_data.eval_setup.approx_ged_manifest --root <APPROX_GED dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "t05-approx-ged-1"

KEYS = [
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids_graphedx",
    "grec",
    "aids_iam",
    "coil_del",
    "mutagenicity",
    "protein",
]

# The frozen specification -- T-05 design §1, CONTRACTS §3. The options string is part of the
# method name; a run that does not record it verbatim is invalid.
ROLE_SPEC: dict[str, dict[str, str]] = {
    "LB": {
        "role": "lb",
        "method": "BRANCH_FAST",
        "options_string": "--threads 1",
        "accessor": "get_lower_bound",
        "scope": "all 21,710,892 Suite-2 pairs",
        "status": "primary lower bound",
    },
    "UB": {
        "role": "ub",
        "method": "BIPARTITE",
        "options_string": "--threads 1",
        "accessor": "get_upper_bound",
        "scope": "all 21,710,892 Suite-2 pairs",
        "status": "primary upper bound",
    },
    "UB_SENSITIVITY": {
        "role": "ubs",
        "method": "BP_BEAM",
        "options_string": (
            "--threads 1 --randomness PSEUDO "
            "--initialization-method BIPARTITE --initial-solutions 1"
        ),
        "accessor": "get_upper_bound",
        "scope": "all 21,710,892 Suite-2 pairs",
        "status": "disclosed sensitivity arm, full cohort",
    },
    "UB_TIGHT": {
        "role": "ubt",
        "method": "IPFP",
        "options_string": "--threads 1 --randomness PSEUDO --initial-solutions 10",
        "accessor": "get_upper_bound",
        "scope": "the §1.1 size-stratified 28,000-pair subsample",
        "status": "disclosed sensitivity arm, sampled",
    },
}

COST_MODEL = {
    "name": "D6",
    "edit_cost_constant": [1, 1, 0, 1, 1, 0],
    "order": ["node_ins", "node_del", "node_rel", "edge_ins", "edge_del", "edge_rel"],
    "gedlib_edit_cost": "CONSTANT",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(z: Any) -> dict[str, Any]:
    if "metadata" not in z.files:
        return {}
    try:
        return json.loads(str(z["metadata"]))
    except (ValueError, TypeError):
        return {}


def describe_dense(path: Path, role_dir: str) -> dict[str, Any]:
    """Describe one dense role matrix, measuring rather than trusting its metadata."""
    z = np.load(path, allow_pickle=False)
    meta = _meta(z)
    m = z["ged_matrix"]
    n = int(m.shape[0])
    off = ~np.eye(n, dtype=bool)
    iu = np.triu_indices(n, k=1)
    rec: dict[str, Any] = {
        "file": f"{role_dir}/{path.name}",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "dataset": path.stem,
        "n_graphs": n,
        "n_pairs": int(iu[0].size),
        "keys": sorted(z.files),
        "measured": {
            "symmetric": bool(np.array_equal(m, m.T)),
            "diagonal_zero": bool(not np.count_nonzero(np.diag(m))),
            "all_finite": bool(np.isfinite(m).all()),
            "min": float(m.min()),
            "max": float(m.max()),
            "n_zero_offdiag": int(np.count_nonzero(m[off] == 0)),
            "zero_offdiag_fraction": float(np.count_nonzero(m[off] == 0) / max(int(off.sum()), 1)),
        },
        "recorded": {
            k: meta.get(k)
            for k in (
                "method",
                "options_string",
                "accessor",
                "cost_model",
                "role",
                "code_commit",
                "computed_utc",
                "seconds_total",
                "n_certified",
                "certification_rate",
                "slurm_job_id",
                "schema_version",
            )
        },
    }
    if "certified_mask" in z.files:
        cm = z["certified_mask"]
        rec["measured"]["certification_rate"] = float(cm[iu].mean())
        rec["measured"]["certified_mask_diagonal_true"] = bool(np.all(np.diag(cm)))

    spec = ROLE_SPEC[role_dir]
    mismatches = [
        f"{field}: recorded {meta.get(field)!r} != frozen {spec[field]!r}"
        for field in ("method", "options_string")
        if meta.get(field) != spec[field]
    ]
    rec["conforms_to_frozen_spec"] = not mismatches
    if mismatches:
        rec["spec_mismatches"] = mismatches
    if meta.get("slurm_job_id") in (None, "", "None"):
        rec.setdefault("provenance_gaps", []).append(
            "slurm_job_id absent -- CONTRACTS §4 lists it in the metadata schema"
        )
    return rec


def describe_flat(path: Path, label: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = _meta(z)
    rec: dict[str, Any] = {
        "file": label,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "keys": sorted(z.files),
        "recorded": {
            k: meta.get(k)
            for k in (
                "method",
                "options_string",
                "accessor",
                "cost_model",
                "seed",
                "bin_edges",
                "n_per_bin",
                "code_commit",
                "computed_utc",
                "schema_version",
            )
        },
    }
    for k in ("value", "exact", "n_nodes", "pair_i"):
        if k in z.files:
            rec["rows"] = int(np.asarray(z[k]).shape[0])
            break
    if "value" in z.files:
        v = np.asarray(z["value"], dtype=np.float64)
        rec["measured"] = {
            "all_finite": bool(np.isfinite(v).all()),
            "min": float(v.min()),
            "max": float(v.max()),
        }
        if "value_fwd" in z.files and "value_rev" in z.files:
            rec["measured"]["value_equals_min_fwd_rev"] = bool(
                np.array_equal(v, np.minimum(z["value_fwd"], z["value_rev"]))
            )
    return rec


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for role_dir in ("LB", "UB", "UB_SENSITIVITY"):
        for key in KEYS:
            p = root / role_dir / f"{key}.npz"
            if p.exists():
                files.append(describe_dense(p, role_dir))
    for name, label in (
        ("subsample.npz", "UB_TIGHT/subsample.npz"),
        ("subsample_pairs.npz", "UB_TIGHT/subsample_pairs.npz"),
    ):
        p = root / "UB_TIGHT" / name
        if p.exists():
            files.append(describe_flat(p, label))
    for p in sorted((root / "ladder").glob("rung_*.npz")):
        files.append(describe_flat(p, f"ladder/{p.name}"))
    for key in KEYS:
        p = root / "exported_suite2" / f"{key}.npz"
        if p.exists():
            files.append(describe_flat(p, f"exported_suite2/{p.name}"))

    ladder: dict[str, Any] = {}
    lm = root / "ladder" / "manifest.json"
    if lm.exists():
        d = json.loads(lm.read_text())
        ladder = {
            k: d.get(k)
            for k in ("exact_ged_ceiling", "truncated_at_rung", "truncate_below", "seed", "note")
        }

    gates: dict[str, Any] = {}
    for gp in sorted((root / "gates").glob("*.json")):
        try:
            gates[gp.stem] = json.loads(gp.read_text())
        except ValueError:
            gates[gp.stem] = {"unreadable": True}

    dense = [f for f in files if f["file"].split("/")[0] in ROLE_SPEC and "n_graphs" in f]
    return {
        "schema_version": SCHEMA_VERSION,
        "ticket": "T-05 -- bounded GED over Suite 2",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_code_commit": code_commit(),
        "cost_model": COST_MODEL,
        "role_specification": ROLE_SPEC,
        "cohort": {
            "filter": {
                "min_nodes": 2,
                "require_connected": True,
                "n_max": None,
                "splits_merged": True,
                "enumeration": "split_index",
            },
            "n_datasets_expected": len(KEYS),
            "n_pairs_expected": 21_710_892,
            "n_pairs_present": sum(f["n_pairs"] for f in dense if f["file"].startswith("LB/")),
        },
        "ladder": ladder,
        "gates": gates,
        "counts": {
            "dense_matrices": len(dense),
            "files_total": len(files),
            "spec_mismatches": sum(1 for f in files if f.get("spec_mismatches")),
            "provenance_gaps": sum(1 for f in files if f.get("provenance_gaps")),
        },
        "files": files,
    }


PROVENANCE_TEMPLATE = """# APPROX_GED -- provenance

Produced by IsalGraph revision ticket **T-05**, "bounded GED over Suite 2", for the *Pattern
Recognition* major revision (PR-D-26-03293). Generated {generated_utc} from repository commit
`{commit}`.

Machine-readable counterpart: `manifest.json`, which carries a sha256 and the measured shape,
symmetry, zero fraction and certification rate of **every** file listed below. Prefer it over this
note for any number.

## What is here

{role_table}

Cost model **D6** throughout: `CONSTANT` edit cost, `[1, 1, 0, 1, 1, 0]` over
`[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]` -- node and edge insertion and
deletion cost 1, substitutions are free. One model for every dataset, which is what reviewer comment
R3.5b asked for.

**The options string is part of the method name.** GEDLIB's upper bounds change on 91.5-93.6 % of
pairs between runs at library defaults, so a value quoted without its string is not reproducible.
Every string above is pinned and recorded in each file's own `metadata`.

## Schema

Every dense file carries the ten keys of `GED_PRECOMPUTED/extended_merged_exact_ged/computed/*.npz`,
so one loader reads exact and bounded files alike: `ged_matrix`, `lb_matrix`, `ub_matrix`,
`certified_mask`, `seconds_matrix`, `node_counts`, `edge_counts`, `graph_ids`, `labels`, `metadata`.

- `ged_matrix` is **this directory's own value** -- the lower bound in `LB/`, `BIPARTITE` in `UB/`,
  `BP_BEAM_DET` in `UB_SENSITIVITY/`. It is always a bound, **never an interpolation**; there is no
  midpoint anywhere in this tree.
- `lb_matrix`, `ub_matrix` and `certified_mask` are the **same arrays in all three files**, so the
  bracket and the certification rate travel with any single file.
- `certified_mask` is `lb_matrix == ub_matrix` at `1e-9`. It is a **derived** statement -- a proven
  lower bound of `k` and an exhibited edit path of cost `k` together prove GED = `k` -- and is never
  sourced from a backend's self-report, because `ANCHOR_AWARE_GED` was measured issuing a false
  optimality certificate.

## Reading these files without corrupting your analysis

- **GED is legitimately 0** for isomorphic graphs -- 15.5 % of Letter LOW. Do not assert
  `0 < value < inf` per pair. The silent-zero failure has the shape of a matrix that is >= 99 %
  zero off-diagonal, and that is what the gate checks.
- **Censored exact-GED pairs carry `inf`, not `NaN`.** A filter written as `np.isnan(...)` passes
  them straight through, and `inf <= x` is False while raising nothing. Select on `certified_mask`
  and filter with `np.isfinite`.
- **`aids_graphedx` (819 graphs) is not Suite 1's `aids` (769).** Suite 1 applies `n_max = 12`. The
  769 are a strict subset of the 819, so the two join on `graph_ids` -- **never positionally**.

## Validation

**G2, the strong gate.** On the four datasets whose Suite-2 cohort is identical to Suite 1
(`iam_letter_{{low,med,high}}`, `linux`), these values reproduce the T-27 bake-off census
**element-wise**: {g2_pairs} pairs, byte-identical sha256 on the value array, across
`BRANCH_FAST`, `BIPARTITE` **and** `BP_BEAM_DET`. One comparison covers loader, cost model, options
string, symmetrisation and pair ordering.

**G3, bracket validity.** Zero violations of `LB <= UB` across the cohort, and zero violations of
`lb <= exact <= ub` against T-03's certified exact values on the overlapping cohorts, joined on
`graph_ids`.

**G4, structural.** Every dense matrix symmetric to machine precision, zero-diagonal, finite,
non-negative, with off-diagonal zero fraction below 0.99 and `certified_mask` true on the diagonal.

Upper bounds are computed in **both orientations and minimised**, because `BIPARTITE`, `BP_BEAM` and
`IPFP` build an edit path from a directed assignment and are not symmetric. `BRANCH_FAST` is
symmetric; that was measured on 9,406 pairs across five datasets and two size strata rather than
assumed, and found identically equal.

## Known limitations -- read before quoting a timing or an interval

1. **`seconds_matrix` is realised wall time under a parallelisation now known to be pathological,
   not a per-pair cost of the method.** The process pool used by the production campaigns is
   negative-scaling: on identical work, 1 worker took 36 core-seconds, 4 took 212, 15 took 928 and
   32 took 5,260. Datasets differ in the worker count their run used, so **these timings are not
   comparable across datasets**, and any conclusion of the form "method X costs Y per pair" drawn
   from them is an artefact of the pool. `--workers 1` is the measured optimum.
2. **The exact-GED ceiling is a measurement, not an assertion.** The ladder truncates at the first
   rung certifying below 25 %: rung 17 certified {r17}, rung 18 certified {r18}, so the ceiling is
   **n = {ceiling}**. Per-rung quantities are conditioned on certified pairs, and the certified
   subset becomes a more biased sample of the rung as `n` grows -- at rung 18 it is the {r18} of
   pairs that A* could finish inside 1,200 s.
3. **The ladder is six datasets, not ten, and its composition shifts across rungs.** Letter and
   LINUX cap at `n <= 10` and contribute at no rung; neither AIDS cohort has a 14-node connected
   graph. So a bare rung-to-rung slope conflates a size effect with a provenance effect, and the
   provenance shift is forced -- it is which real datasets contain connected graphs of each size.
4. **Size and provenance are confounded across the size range** and no sampling design removes it:
   the small bins are overwhelmingly Letter and the large bins overwhelmingly Mutagenicity and
   COIL-DEL, while density moves with provenance over the same range. The size-scaling curve is
   therefore reported **within dataset** as primary; any pooled curve is descriptive only.
5. **`UB_TIGHT/` is a size-stratified sample, not a random sample of Suite-2 pairs.** It
   deliberately over-weights large `n`. Every figure from it is reported per bin and never pooled
   into a cohort-level mean.
6. **T-03's `ub_matrix` is run-dependent** and was accepted rather than repaired (PI decision
   2026-08-15). Its default options left `IPFP` on GEDLIB's `--randomness REAL`, so 74-82 % of its
   upper-bound values change between runs. The exposure is bounded and verified: `ub_matrix` equals
   `ged_matrix` on all 234,258 certified AIDS and 3,870 certified LINUX pairs, so it is **exactly
   the 61,084 D11 censored-interval upper ends**. Those upper ends are heuristic and will not
   reproduce; the lower ends are unaffected. This does not affect any file in *this* directory.
7. **`slurm_job_id` is absent from the dense files' metadata** although CONTRACTS §4 lists it. Job
   identifiers are recoverable from `run_reports/`. Recorded because provenance that looks
   checkable but is not is worse than provenance that is plainly missing.

## Not in this directory

`rho(Lev, ·)` -- the correlation between IsalGraph's Levenshtein distance and these bounds -- is
**not** computed here. It needs a canonical string per Suite-2 graph, and only five of the ten
datasets have ever been canonicalised (all of them Suite-1 cohorts at `n <= 12`). It is deferred in
full to ticket **T-06** by PI decision 2026-08-15, under the protocol decision **D14** already
fixes: a 300 s canonicalisation timeout, a greedy-min fallback for a censored graph rather than a
drop, affected pairs flagged, and censoring reported per symmetry stratum.

> One measured caution for whoever runs it: **a Python signal-based timeout does not interrupt the
> C++ engine.** Signal handlers run only between bytecode instructions, so `SIGALRM` stays queued
> for the whole duration of a native call and D14's budget silently fails to apply. Enforce it with
> a killed subprocess.
"""


def write_provenance(root: Path, manifest: dict[str, Any]) -> Path:
    lad = manifest.get("ladder", {})
    rungs = {}
    for p in sorted((root / "ladder").glob("rung_*.npz")):
        try:
            z = np.load(p, allow_pickle=False)
            rungs[int(p.stem.split("_")[1])] = float(np.asarray(z["certified"]).mean())
        except (ValueError, KeyError, OSError):
            continue
    pct = lambda n: f"{rungs[n]:.1%}" if n in rungs else "n/a"  # noqa: E731
    n_by_dir = {
        d: sum(1 for f in manifest["files"] if f["file"].startswith(d + "/"))
        for d in ("LB", "UB", "UB_SENSITIVITY", "ladder", "exported_suite2")
    }

    # Generated from ROLE_SPEC so the published table cannot drift from the frozen specification.
    rows = ["| Directory | Contents | Method | Options string, verbatim |", "|---|---|---|---|"]
    for d in ("LB", "UB", "UB_SENSITIVITY"):
        s = ROLE_SPEC[d]
        rows.append(
            f"| `{d}/` | {n_by_dir[d]} dense matrices | `{s['method']}` | `{s['options_string']}` |"
        )
    s = ROLE_SPEC["UB_TIGHT"]
    rows.append(
        f"| `UB_TIGHT/` | the 28,000-pair size-stratified subsample "
        f"| `{s['method']}` | `{s['options_string']}` |"
    )
    rows.append(
        f"| `ladder/` | exact-GED calibration rungs, {n_by_dir['ladder']} of them "
        f"| `networkx.graph_edit_distance` | 1,200 s per-pair budget |"
    )
    rows.append(
        f"| `exported_suite2/` | the input graphs, CSR, {n_by_dir['exported_suite2']} files "
        f"| -- | -- |"
    )
    rows.append("| `gates/` | validation gate results | -- | -- |")

    text = PROVENANCE_TEMPLATE.format(
        generated_utc=manifest["generated_utc"],
        commit=manifest["generator_code_commit"],
        role_table="\n".join(rows),
        g2_pairs="10,807,845",
        ceiling=lad.get("exact_ged_ceiling", "n/a"),
        r17=pct(17),
        r18=pct(18),
    )
    out = root / "PROVENANCE.md"
    out.write_text(text)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="the published APPROX_GED directory")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    manifest = build(root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    c = manifest["counts"]
    print(
        f"manifest.json: {c['files_total']} files, {c['dense_matrices']} dense matrices, "
        f"{c['spec_mismatches']} spec mismatches, {c['provenance_gaps']} provenance gaps"
    )
    print(
        f"  cohort pairs present: {manifest['cohort']['n_pairs_present']:,} / "
        f"{manifest['cohort']['n_pairs_expected']:,}"
    )
    if not args.manifest_only:
        print(f"wrote {write_provenance(root, manifest)}")
    return 1 if c["spec_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
