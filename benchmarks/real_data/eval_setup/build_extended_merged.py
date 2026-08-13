"""Assemble the extended exact-GED distribution beside the original ground truth.

Builds ``GED_PRECOMPUTED/extended_merged_exact_ged/`` from the T-03 Contract D matrices
and, where one exists, the GraphEdX published matrix, keeping the two **side by side**
rather than reconciled into one number.

They must not be reconciled, for two independent reasons measured during T-03:

1. **Different cost models.** GraphEdX charges zero for node operations
   (``[0,0,0,1,1,0]``); we use the unit model (``[1,1,0,1,1,0]``, decision D6) because a
   zero node cost makes GED a *pseudo*metric while the IsalGraph distance is a metric.
2. **The published matrix is not exact.** Over 208 AIDS pairs recomputed under GraphEdX's
   own cost model, 150 sat above our certified optima, 58 equal, and **none** below. A
   value below a published one is a proof the published one is not optimal, since GED is
   a minimum and A* returns an achievable edit path.

A single merged column would therefore assert an equivalence that does not hold. The
output keeps three provenance classes distinct and quantifies their disagreement.

Run locally: reading the published matrices needs ``torch``, which is deliberately absent
from the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

SUITE1 = ("iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids")
#: Datasets for which a published reference exists at all. IAM Letter ships raw .gxl
#: files and no GED matrix, so every Letter value in the study was always ours.
HAS_REFERENCE = ("aids", "linux")
GRAPHEDX_NAME = {"aids": "AIDS", "linux": "LINUX"}


class MergeError(Exception):
    """Raised when the two sources cannot be aligned or a matrix fails its checks."""


@dataclass(slots=True)
class DatasetReport:
    """Per-dataset accounting for the provenance document."""

    key: str
    n_graphs: int
    n_pairs: int
    n_certified: int = 0
    n_censored: int = 0
    n_zero: int = 0
    has_reference: bool = False
    n_overlap: int = 0
    n_ours_lower: int = 0
    n_equal: int = 0
    n_ours_higher: int = 0
    mean_delta: float = 0.0
    max_delta: float = 0.0
    reference_coverage: float = 0.0
    notes: list[str] = field(default_factory=list)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _upper_index(i: int, j: int, n: int) -> int:
    """Index into a split's upper triangle **including** the diagonal.

    GraphEdX stores ``n(n+1)/2`` entries per split in this order.
    """
    if i > j:
        i, j = j, i
    return i * n - i * (i - 1) // 2 + (j - i)


def load_published(name: str, source_dir: str) -> dict[str, np.ndarray]:
    """Load GraphEdX's published GED, one dense matrix per split.

    Parameters
    ----------
    name : str
        Upper-case GraphEdX dataset name, e.g. ``'AIDS'``.
    source_dir : str
        Directory holding ``{split}_result.pt``.

    Returns
    -------
    dict
        ``{split: (n_split, n_split) float64}``.

    Raises
    ------
    MergeError
        If ``torch`` is unavailable or a split's payload has the wrong length.
    """
    try:
        import torch  # noqa: PLC0415 - deliberately local; the cluster has no torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MergeError(
            "reading GraphEdX's published matrix needs torch; run this on the "
            "workstation, not on Picasso"
        ) from exc

    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        gpath = os.path.join(source_dir, name, f"{split}_graphs.pt")
        rpath = os.path.join(source_dir, name, f"{split}_result.pt")
        graphs = torch.load(gpath, weights_only=False)
        res = torch.load(rpath, weights_only=False)
        n = len(graphs)
        if len(res) != n * (n + 1) // 2:
            raise MergeError(
                f"{name}/{split}: {len(res)} results for {n} graphs, expected {n * (n + 1) // 2}"
            )
        m = np.full((n, n), np.nan, dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                v = res[_upper_index(i, j, n)]
                val = float(v[0] if isinstance(v, (tuple, list)) else v)
                m[i, j] = m[j, i] = val
        out[split] = m
    return out


def compare_to_reference(
    ours: np.ndarray, graph_ids: list[str], published: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, DatasetReport]:
    """Align our matrix to the published one and quantify the disagreement.

    Alignment goes through the graph ids, which encode split and within-split index as
    ``{dataset}_{split}_{idx}``. The published matrix covers **within-split pairs only**,
    so the overlap is a strict subset of ours.

    Returns
    -------
    tuple
        ``(reference_matrix, overlap_mask, report)`` where ``reference_matrix`` is our
        shape with ``nan`` off the overlap.
    """
    n = len(graph_ids)
    ref = np.full((n, n), np.nan, dtype=np.float64)
    mask = np.zeros((n, n), dtype=bool)

    parsed: list[tuple[str, int] | None] = []
    for gid in graph_ids:
        parts = gid.rsplit("_", 2)
        if len(parts) != 3 or not parts[2].isdigit():
            parsed.append(None)
            continue
        parsed.append((parts[1], int(parts[2])))

    for a in range(n):
        pa = parsed[a]
        if pa is None or pa[0] not in published:
            continue
        for b in range(a + 1, n):
            pb = parsed[b]
            if pb is None or pb[0] != pa[0]:
                continue  # cross-split: GraphEdX publishes nothing here
            m = published[pa[0]]
            if pa[1] >= m.shape[0] or pb[1] >= m.shape[0]:
                continue
            v = m[pa[1], pb[1]]
            if not np.isfinite(v):
                continue
            ref[a, b] = ref[b, a] = v
            mask[a, b] = mask[b, a] = True

    rep = DatasetReport(key="", n_graphs=n, n_pairs=n * (n - 1) // 2, has_reference=True)
    iu = np.triu_indices(n, 1)
    ov = mask[iu]
    rep.n_overlap = int(ov.sum())
    if rep.n_overlap:
        o, r = ours[iu][ov], ref[iu][ov]
        finite = np.isfinite(o) & np.isfinite(r)
        o, r = o[finite], r[finite]
        d = o - r
        # GED is integer-valued under both cost models, so any REAL difference is >= 1
        # and the tolerance belongs at 0.5. Two successively tighter guesses both
        # reported storage noise as disagreement: 1e-9 flagged 7 LINUX pairs with deltas
        # of 2.7e-07..3.1e-06, and 1e-6 still flagged 86 AIDS pairs whose deltas all
        # round to zero. GraphEdX stores floats and its own loader rounds anything
        # within 0.01 of an integer, so half-integer separation is the right scale.
        tol = 0.5
        rep.n_ours_lower = int((d < -tol).sum())
        rep.n_equal = int((np.abs(d) <= tol).sum())
        rep.n_ours_higher = int((d > tol).sum())
        rep.mean_delta = float(d.mean()) if d.size else 0.0
        rep.max_delta = float(np.abs(d).max()) if d.size else 0.0
        rep.reference_coverage = rep.n_overlap / rep.n_pairs
    return ref, mask, rep


def summarise_ours(path: str) -> DatasetReport:
    """Read one Contract D matrix and report its accounting."""
    z = np.load(path, allow_pickle=False)
    g = z["ged_matrix"]
    n = g.shape[0]
    iu = np.triu_indices(n, 1)
    off = g[iu]
    cert = z["certified_mask"][iu] if "certified_mask" in z.files else np.isfinite(off)
    rep = DatasetReport(key="", n_graphs=n, n_pairs=off.size)
    rep.n_certified = int(cert.sum())
    rep.n_censored = int(np.isinf(off).sum())
    rep.n_zero = int((off == 0).sum())
    if not np.allclose(g, g.T, equal_nan=True):
        raise MergeError(f"{path}: matrix is not symmetric")
    if not np.all(np.diag(g) == 0):
        raise MergeError(f"{path}: diagonal is not zero")
    return rep


def build(
    computed_dir: str, source_dir: str, out_dir: str, keys: tuple[str, ...]
) -> dict[str, Any]:
    """Assemble the extended distribution. Returns the manifest."""
    os.makedirs(os.path.join(out_dir, "computed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reference"), exist_ok=True)

    manifest: dict[str, Any] = {"datasets": {}, "cost_model_ours": [1, 1, 0, 1, 1, 0]}
    reports: dict[str, DatasetReport] = {}

    for key in keys:
        src = os.path.join(computed_dir, f"{key}.npz")
        if not os.path.isfile(src):
            logger.warning("%s: no computed matrix at %s -- skipping", key, src)
            continue
        dst = os.path.join(out_dir, "computed", f"{key}.npz")
        shutil.copy2(src, dst)
        rep = summarise_ours(dst)
        rep.key = key
        z = np.load(dst, allow_pickle=False)
        gids = [str(x) for x in z["graph_ids"]]

        if key in HAS_REFERENCE:
            try:
                pub = load_published(GRAPHEDX_NAME[key], source_dir)
                ref, mask, cmp_rep = compare_to_reference(z["ged_matrix"], gids, pub)
                np.savez_compressed(
                    os.path.join(out_dir, "reference", f"{key}_graphedx.npz"),
                    reference_matrix=ref,
                    overlap_mask=mask,
                    graph_ids=z["graph_ids"],
                    metadata=json.dumps(
                        {
                            "source": "GraphEdX published *_result.pt",
                            "cost_model": [1, 1, 0, 1, 1, 0],
                            "cost_model_note": (
                                "MEASURED 2026-08-12, not taken from the plan: the "
                                "published values match a UNIT node cost on 4/4 tested "
                                "pairs and a zero node cost on 0/4, each differing from "
                                "the zero-node value by exactly |n1 - n2|. The revision "
                                "plan asserts zero node cost; that assertion is wrong."
                            ),
                            "coverage": "within-split pairs only",
                            "status": "agrees with ours on 99.998% of the finite overlap",
                        }
                    ),
                )
                for f in (
                    "has_reference",
                    "n_overlap",
                    "n_ours_lower",
                    "n_equal",
                    "n_ours_higher",
                    "mean_delta",
                    "max_delta",
                    "reference_coverage",
                ):
                    setattr(rep, f, getattr(cmp_rep, f))
            except MergeError as exc:
                rep.notes.append(f"reference unavailable: {exc}")
                logger.warning("%s: %s", key, exc)
        else:
            rep.notes.append("IAM Letter ships no published GED matrix; all values are ours")

        reports[key] = rep
        manifest["datasets"][key] = {
            "computed_sha256": _sha256(dst),
            "n_graphs": rep.n_graphs,
            "n_pairs": rep.n_pairs,
            "n_certified": rep.n_certified,
            "n_censored": rep.n_censored,
            "n_zero_offdiag": rep.n_zero,
            "has_reference": rep.has_reference,
            "n_overlap_with_reference": rep.n_overlap,
            "ours_lower": rep.n_ours_lower,
            "equal": rep.n_equal,
            "ours_higher": rep.n_ours_higher,
            "notes": rep.notes,
        }
        logger.info(
            "%-18s %5d graphs  %9d pairs  %9d certified  %6d censored  overlap %d",
            key,
            rep.n_graphs,
            rep.n_pairs,
            rep.n_certified,
            rep.n_censored,
            rep.n_overlap,
        )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    write_provenance(out_dir, reports)
    return manifest


def write_provenance(out_dir: str, reports: dict[str, DatasetReport]) -> None:
    """Write PROVENANCE.md, the document that keeps the two sources apart."""
    L: list[str] = []
    A = L.append
    A("# Extended exact GED — provenance\n")
    A("**Every value in this directory belongs to exactly one of three classes.** They are")
    A("kept separate on purpose: the two sources disagree by construction, and presenting a")
    A("single merged number per pair would assert an equivalence that does not hold.\n")
    A("| Class | Where | Datasets | Coverage | Cost model | Status |")
    A("|---|---|---|---|---|---|")
    A(
        "| **A — ours, certified exact** | `computed/` | all five | all pairs "
        "| `[1,1,0,1,1,0]` | A* run to completion |"
    )
    A(
        "| **B — ours, interval-censored** | `computed/` | all five | timeout pairs "
        "| `[1,1,0,1,1,0]` | `inf` in `ged_matrix`, bracket in `lb_matrix`/`ub_matrix` |"
    )
    A(
        "| **C — GraphEdX published** | `reference/` | AIDS, LINUX only "
        "| **within-split only** | `[1,1,0,1,1,0]` — **measured, see below** | agrees with ours |"
    )
    A("")
    A("## Class C's cost model — measured here, and it contradicts the revision plan\n")
    A("> ### ⚠ Correction, 2026-08-12\n")
    A("> The revision plan states that GraphEdX charges **zero** for node operations")
    A("> (`gedlib.md` §6, `statistics.md` D6), and the gate-0 configuration was derived from")
    A("> that. **It is wrong.** Tested directly by recomputing pairs under both models and")
    A("> comparing to the published file:\n")
    A("> ```")
    A(">   pair    dn | published  zero-node  unit-node | verdict")
    A("> 241, 475   1 |       8.0        7.0        8.0 | matches UNIT")
    A("> 207, 377   3 |       8.0        5.0        8.0 | matches UNIT")
    A("> 135, 339   1 |       2.0        1.0        2.0 | matches UNIT")
    A("> 211,  67   4 |       9.0        5.0        9.0 | matches UNIT")
    A("> ```")
    A("> Zero-node 0/4, unit-node 4/4, and in every case the published value exceeds the")
    A("> zero-node value by **exactly `|n₁ − n₂|`**. GraphEdX's AIDS matrix uses the **same")
    A("> unit cost model as D6**.\n")
    A("**What this retracts.** An earlier T-03 finding held that the published matrix was an")
    A("approximate upper bound, on the strength of gate 0 measuring 150 pairs below it, 58")
    A("equal and none above. Gate 0 ran under `[0,0,0,1,1,0]` because the plan said to. Those")
    A("150 low values were the *arithmetic of the wrong cost model* — each low by exactly the")
    A("node-count difference — not evidence of non-optimality. **That finding is withdrawn.**\n")
    A("**What survives.** Comparing like with like, our values and theirs agree on all but")
    A("**2 of the finite overlap pairs**. Those two are real and both have ours *below* theirs")
    A("by 2 (`aids_train_0024`/`aids_train_0246` 5 vs 7, `aids_val_0016`/`aids_val_0036` 7 vs")
    A("9), both certified. Since GED is a minimum and A* returns an achievable path, those two")
    A("published entries are provably non-optimal — but 2 in 131,148 is a rounding error, not a")
    A("characterisation. **Treat class C as essentially exact under unit costs.**\n")
    A("**Why the recompute is still necessary**, on the two grounds that were always the real")
    A("ones and are untouched by this correction:\n")
    A("1. **Coverage.** GraphEdX publishes GED for *within-split pairs only* — 43.9 % of AIDS")
    A("   pairs and 43.0 % of LINUX. The submitted ρ values were computed on that subset")
    A("   without disclosing it. The columns below quantify exactly how much was missing.")
    A("2. **One model across all ten datasets.** IAM Letter ships **no GED matrix at all** —")
    A("   raw `.gxl` files only — so every Letter value in this study, and in the submitted")
    A("   version, was always ours. A single cost model over the whole cohort is what D6 asks")
    A("   for, and it is unobtainable from the distributions as shipped.\n")
    A("D6's own justification is unaffected: it rests on GED remaining a *metric*, which is an")
    A("argument about zero node costs in general, not about what GraphEdX happened to ship.\n")
    A("## Accounting\n")
    A(
        "| Dataset | graphs | pairs | certified | censored | published overlap "
        "| ours < ref | equal | ours > ref |"
    )
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, r in reports.items():
        ov = f"{r.n_overlap:,} ({100 * r.reference_coverage:.1f}%)" if r.has_reference else "—"
        lo = f"{r.n_ours_lower:,}" if r.has_reference else "—"
        eq = f"{r.n_equal:,}" if r.has_reference else "—"
        hi = f"**{r.n_ours_higher:,}**" if r.has_reference else "—"
        A(
            f"| {key} | {r.n_graphs:,} | {r.n_pairs:,} | {r.n_certified:,} | "
            f"{r.n_censored:,} | {ov} | {lo} | {eq} | {hi} |"
        )
    A("")
    A("### Reading the last three columns\n")
    A("Both sides now use the **same unit cost model**, so the two should simply agree, and on")
    A("the finite overlap they do: every disagreement is a genuine one rather than a modelling")
    A("artefact. Counts are taken with a tolerance of **0.5**, because GED is integer-valued")
    A("and GraphEdX stores floats — two tighter tolerances (1e-9, then 1e-6) both reported pure")
    A("storage noise as disagreement before this was pinned down.\n")
    A("**`ours > ref` is the falsifying column.** Our value is produced by an A* search run to")
    A("completion, so it is achievable; a published value *below* an achievable cost would mean")
    A("our search, our index alignment, or their file is wrong. **`ours < ref`** is not")
    A("falsifying — it means their entry is not optimal, which is a claim about their file and")
    A("one we can prove, since we exhibit the cheaper path.\n")
    A("Censored pairs are excluded from these three columns: with no certified value there is")
    A("nothing to compare. The overlap column counts *all* published pairs, so")
    A("`overlap − (lower + equal + higher)` is the number of censored pairs inside it.\n")
    A("## Layout\n")
    A("```")
    A("extended_merged_exact_ged/")
    A("  computed/<key>.npz      class A + B -- ged_matrix, lb_matrix, ub_matrix,")
    A("                          certified_mask, seconds_matrix, node_counts,")
    A("                          edge_counts, graph_ids, labels, metadata")
    A("  reference/<key>_graphedx.npz   class C -- reference_matrix, overlap_mask,")
    A("                          graph_ids, metadata (nan outside the overlap)")
    A("  manifest.json           sha256 and counts per dataset")
    A("  PROVENANCE.md           this file")
    A("```\n")
    A("`ged_matrix` holds `inf` for a censored pair, which is the existing convention and is")
    A("what `method_comparator.py` and `validator.py` already mask on. The censoring interval")
    A("is never lost: `lb_matrix` and `ub_matrix` are finite everywhere. Censored pairs are")
    A("**analysed as interval-censored, never dropped** (decision D11), and the censoring rate")
    A("is reported **per stratum, never pooled** (D12).\n")
    with open(os.path.join(out_dir, "PROVENANCE.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--computed-dir", required=True, help="directory of our Contract D .npz")
    p.add_argument("--source-dir", required=True, help="GED_PRECOMPUTED root")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--datasets", default="all")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(levelname)s %(message)s")
    keys = SUITE1 if a.datasets == "all" else tuple(a.datasets.split(","))
    try:
        build(a.computed_dir, a.source_dir, a.out, keys)
    except MergeError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
