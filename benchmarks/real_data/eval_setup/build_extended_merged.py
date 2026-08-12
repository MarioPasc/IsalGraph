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
        # GraphEdX stores its values as floats, and their loader itself rounds anything
        # within 0.01 of an integer. A 1e-9 tolerance therefore reports pure storage
        # noise as disagreement: on LINUX it flagged 7 pairs whose deltas were all
        # between 2.7e-07 and 3.1e-06. GED under both models is integer-valued, so any
        # real difference is at least 1 and 1e-6 separates the two cleanly.
        tol = 1e-6
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
                            "cost_model": [0, 0, 0, 1, 1, 0],
                            "coverage": "within-split pairs only",
                            "status": "approximate upper bound, not exact GED",
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
        "| **within-split only** | `[0,0,0,1,1,0]` | **approximate upper bound** |"
    )
    A("")
    A("## Why class C is not ground truth\n")
    A("Two independent reasons, both measured during T-03.\n")
    A("**The cost model differs.** GraphEdX charges zero for node operations. With zero node")
    A("cost, inserting an isolated vertex is free, so two non-isomorphic graphs can sit at")
    A("distance 0 and GED is only a *pseudo*metric — while the IsalGraph graph distance is a")
    A("metric. Validating a metric against a pseudometric reference is incoherent, which is")
    A("decision D6.\n")
    A("**The published values are not optimal.** Recomputing 208 AIDS pairs under GraphEdX's")
    A("*own* cost model gave **150 below** the published value, **58 equal**, and **none above**.")
    A("GED is a minimum and an A* search returns an *achievable* edit path, so a lower value is")
    A("a proof that the published one is not the minimum. For AIDS train pair (76, 211) the")
    A("published matrix gives **11** while we exhibit a path of cost **6**. The strictly")
    A("one-sided discrepancy is what identifies the reference, rather than our solver, as the")
    A("source: a faulty solver errs in both directions.\n")
    A("**IAM Letter has no published GED matrix at all** — the distribution ships raw `.gxl`")
    A("files. Every Letter value in this study, and in the submitted version, was always ours.\n")
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
    A("### Reading the last three columns — the direction depends on which cost model\n")
    A("This table compares **our D6 values** against **GraphEdX's zero-node-cost values**, so")
    A("the two differ for two reasons at once and the expected direction is *ours ≥ theirs*:")
    A("we charge 1 per node insertion or deletion where they charge 0, so any pair whose graphs")
    A("differ in order costs us an extra `|n₁ − n₂|`. **`ours > ref` is therefore expected and")
    A("benign here**, and its size tracks how often the two graphs differ in order.\n")
    A("The falsifying direction in *this* table is **`ours < ref`**: under a strictly cheaper")
    A("cost model our value can never fall below theirs. A non-zero count there would mean our")
    A("solver, our alignment, or their file is wrong.\n")
    A("> **Do not read this table as the solver check.** That check is gate 0, which recomputes")
    A("> under GraphEdX's *own* cost model, where the inequality reverses: ours ≤ theirs, and")
    A("> `ours > ref` would be the falsifying column. Gate 0 measured **150 lower, 58 equal, 0")
    A("> higher** over 208 AIDS pairs — one-sided in the direction that indicts the reference,")
    A("> not the solver. The two tables answer different questions and their expected")
    A("> directions are opposite.\n")
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
