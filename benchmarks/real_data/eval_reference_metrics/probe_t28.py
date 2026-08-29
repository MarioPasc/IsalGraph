"""T-28 fast probe: point-estimate Spearman rho under the new reference metrics.

No bootstrap, no permutation, no confidence intervals. This answers one question
as quickly as the cached artifacts allow -- *does the canonical string track the
Weisfeiler-Lehman kernel distance or the spectral lambda-distance better than the
competing representations do?* -- so that the full pre-registered campaign is
launched already knowing what it will find.

Every representation distance is read from the T-06 cache and is **not**
recomputed, so the T-04a primary-distance selections are preserved by
construction: ``levenshtein`` for the six serialisations, ``kernel`` for
``wl_subtree``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral import cohort_spectra, spectral_distance_matrix  # noqa: E402

SUITE1 = ("aids", "iam_letter_low", "iam_letter_med", "iam_letter_high", "linux")
SUITE2 = (
    "aids_graphedx",
    "aids_iam",
    "coil_del",
    "grec",
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "mutagenicity",
    "protein",
)
ARM = "isalgraph_pruned"


def _load_graphs(archive: Path, suite: str, dataset: str) -> dict[str, Any]:
    sub = "exported" if suite == "suite1" else "exported_suite2"
    with np.load(archive / "data" / sub / f"{dataset}.npz", allow_pickle=True) as z:
        return {
            "graph_ids": np.asarray(z["graph_ids"]).astype(str),
            "n_nodes": np.asarray(z["n_nodes"]),
            "edge_offsets": np.asarray(z["edge_offsets"]),
            "edges": np.asarray(z["edges"]),
        }


def _align(
    matrix: npt.NDArray[Any], src: npt.NDArray[Any], target: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Reorder a square matrix from *src* id order onto *target* id order."""
    pos = {gid: i for i, gid in enumerate(src)}
    idx = np.array([pos[g] for g in target], dtype=np.intp)
    return matrix[np.ix_(idx, idx)]


def _load_distance(
    archive: Path, suite: str, dataset: str, rep: str, metric: str, target: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]] | None:
    path = archive / "data/source/T06/distances" / suite / f"{dataset}__{rep}__{metric}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        src = np.asarray(z["graph_ids"]).astype(str)
        dm = np.asarray(z["distance_matrix"], dtype=np.float64)
        mask = np.asarray(z["defined_mask"], dtype=bool)
    if list(src) != list(target):
        dm, mask = _align(dm, src, target), _align(mask, src, target)
    return dm, mask


def _load_ged(archive: Path, suite: str, dataset: str, target: npt.NDArray[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if suite == "suite1":
        ged_dir = "data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
        p = archive / ged_dir / f"{dataset}.npz"
        if p.exists():
            with np.load(p, allow_pickle=True) as z:
                src = np.asarray(z["graph_ids"]).astype(str)
                m = np.asarray(z["ged_matrix"], dtype=np.float64)
                has_cert = "certified_mask" in z
                cert = np.asarray(z["certified_mask"], dtype=bool) if has_cert else None
            same = list(src) == list(target)
            out["exact"] = m if same else _align(m, src, target)
            if cert is not None:
                out["_exact_mask"] = cert if same else _align(cert, src, target)
    else:
        p = archive / "data/source/APPROX_GED/LB" / f"{dataset}.npz"
        if p.exists():
            with np.load(p, allow_pickle=True) as z:
                src = np.asarray(z["graph_ids"]).astype(str)
                for key, field in (("lb", "lb_matrix"), ("ub", "ub_matrix")):
                    if field in z:
                        m = np.asarray(z[field], dtype=np.float64)
                        out[key] = _align(m, src, target) if list(src) != list(target) else m
    return out


def run_dataset(
    archive: Path, suite: str, dataset: str, reference_root: Path | None = None
) -> list[dict[str, Any]]:
    """Return one record per (representation, reference, view).

    Args:
        archive: Root of the isalgraph artifact archive.
        suite: Suite key.
        dataset: Dataset key.
        reference_root: When given, references are READ from this tree in the
            dense CONTRACTS section 4 schema instead of being computed here.
    """
    graphs = _load_graphs(archive, suite, dataset)
    ids = graphs["graph_ids"]
    n_nodes = np.asarray(graphs["n_nodes"], dtype=np.int64)
    g = len(ids)

    # --- discover which representations were cached for this cell -----------
    dist_dir = archive / "data/source/T06/distances" / suite
    reps: dict[str, str] = {}
    for path in sorted(dist_dir.glob(f"{dataset}__*.npz")):
        stem = path.stem
        _, rep, metric = stem.split("__", 2)
        if metric == "size_null":
            continue
        reps[rep] = metric

    # --- references ---------------------------------------------------------
    refs: dict[str, npt.NDArray[Any]] = {}
    ref_masks: dict[str, npt.NDArray[Any]] = {}
    ged = _load_ged(archive, suite, dataset, ids)
    for key, val in ged.items():
        if key.startswith("_"):
            continue
        refs[key] = val
    if "_exact_mask" in ged:
        ref_masks["exact"] = ged["_exact_mask"]

    if reference_root is not None:
        # Read the BUILT production matrices, so the probe and the campaign are
        # measuring the same objects rather than two implementations of the same
        # description. This is also how spectral_esd reaches the probe.
        for path in sorted((reference_root / suite).glob(f"{dataset}__*.npz")):
            key = path.stem.split("__", 1)[1]
            with np.load(path, allow_pickle=True) as z:
                src = np.asarray(z["graph_ids"]).astype(str)
                mat = np.asarray(z["distance_matrix"], dtype=np.float64)
            refs[key] = mat if list(src) == list(ids) else _align(mat, src, ids)
    else:
        # WL reference: byte-identical to the wl_subtree arm's cached kernel matrix.
        wl = _load_distance(archive, suite, dataset, "wl_subtree", "kernel", ids)
        if wl is not None:
            refs["wl"] = wl[0]

        # Spectral references, computed here.
        variants = (("norm", "spectral"), ("comb", "spectral_comb"), ("adj", "spectral_adj"))
        for variant, key in variants:
            spectra = cohort_spectra(
                n_nodes, graphs["edge_offsets"], graphs["edges"], variant=variant
            )
            refs[key] = spectral_distance_matrix(spectra)

    # --- pair views ---------------------------------------------------------
    iu = np.triu_indices(g, k=1)
    dn = np.abs(n_nodes[iu[0]] - n_nodes[iu[1]]).astype(np.float64)
    equal_n = dn == 0.0

    records: list[dict[str, Any]] = []
    for rep, metric in sorted(reps.items()):
        loaded = _load_distance(archive, suite, dataset, rep, metric, ids)
        if loaded is None:
            continue
        dm, defined = loaded
        x_all = dm[iu]
        d_ok = defined[iu]
        for ref_key, ref_mat in refs.items():
            y_all = ref_mat[iu]
            base = d_ok & np.isfinite(x_all) & np.isfinite(y_all)
            if ref_key in ref_masks:
                base = base & ref_masks[ref_key][iu]
            for view, vmask in (("all_pairs", base), ("equal_n", base & equal_n)):
                k = int(vmask.sum())
                if k < 50:
                    continue
                x, y = x_all[vmask], y_all[vmask]
                if np.all(x == x[0]) or np.all(y == y[0]):
                    continue
                rho = float(stats.spearmanr(x, y).statistic)
                rec: dict[str, Any] = {
                    "suite": suite,
                    "dataset": dataset,
                    "representation": rep,
                    "metric": metric,
                    "reference": ref_key,
                    "view": view,
                    "rho": rho,
                    "n_pairs": k,
                    "n_graphs": g,
                }
                if view == "all_pairs":
                    rec["size_null_rho"] = float(stats.spearmanr(dn[vmask], y).statistic)
                records.append(rec)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_archive = Path("/home/mpascual/research/data/isalgraph_archive")
    ap.add_argument("--archive", type=Path, default=default_archive)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", default="all")
    ap.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="read built reference matrices from here instead of computing them",
    )
    args = ap.parse_args()

    cells = [("suite1", d) for d in SUITE1] + [("suite2", d) for d in SUITE2]
    if args.datasets != "all":
        want = set(args.datasets.split(","))
        cells = [c for c in cells if c[1] in want]

    rows: list[dict[str, Any]] = []
    for suite, dataset in cells:
        try:
            got = run_dataset(args.archive, suite, dataset, args.reference_root)
        except Exception as exc:  # noqa: BLE001 - probe: report and continue
            print(f"[skip] {suite}/{dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows.extend(got)
        print(f"[ok] {suite}/{dataset}: {len(got)} records", file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {len(rows)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
