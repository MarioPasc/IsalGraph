"""Build ``SUMMARY.md`` and ``manifest.json`` for the T06_exhaustive campaign.

Reads the new arm's encodings from ``T06_exhaustive/`` and the pruned arm's from
``T06/``. **``T06/`` is opened read-only and never written**: it is the
pre-registered record.

Three things this reports that a plain completion rate does not.

**Per node-count band.** The completion distribution is heavy-tailed, so one
pooled rate over a cohort that is 65 % at ``n <= 12`` says almost nothing about
the graphs where the exhaustive form is expensive. The bands are the ones the
ceiling measurement used.

**Fallback by tier, not one censoring rate.** A ``pruned``-tier row is still a
canonical form and stays inside the completeness theorem; a ``greedy``-tier row
does not. A rate conflating the two is not interpretable.

**The saving is paired.** Symbol counts are compared per graph on the
``graph_ids`` join, never as a difference of two cohort medians -- the two arms
censor different graphs, so the marginal medians are taken over different sets.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

ARM = "isalgraph_exhaustive"
BASELINE = "isalgraph_pruned"
SEED = 42

SUITE1 = ("linux", "aids", "iam_letter_low", "iam_letter_med", "iam_letter_high")
SUITE2 = (
    "linux",
    "grec",
    "protein",
    "aids_graphedx",
    "iam_letter_low",
    "iam_letter_med",
    "aids_iam",
    "iam_letter_high",
    "coil_del",
    "mutagenicity",
)

#: Node-count bands. The first three are the ceiling measurement's bands; the
#: last is everything the 60 s probe never reached.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("n<=12", 0, 12),
    ("n 13-20", 13, 20),
    ("n 21-26", 21, 26),
    ("n>26", 27, 1 << 30),
)

#: The node counts the PI asked for by name.
HEADLINE_NODE_COUNTS: tuple[int, ...] = (20, 40)

#: Statuses that mean the graph produced a usable encoding. ``censored`` counts
#: as completed: D14 retains it with a substitute string rather than dropping it.
USABLE = ("ok", "censored", "fallback")


@dataclass(frozen=True)
class CellRow:
    """One ``(suite, dataset)`` cell's provenance and outcome."""

    suite: str
    dataset: str
    representation: str
    path: str
    n_graphs: int
    n_ok: int
    n_censored: int
    n_error: int
    fallback_tiers: dict[str, int]
    engine: str
    build_hash: str
    src_commit: str
    code_commit: str
    seed: int
    budget_s: float
    generated_utc: str


def _git_head(repo: Path) -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def _load(path: Path) -> dict[str, Any] | None:
    """Read one encodings ``.npz`` into plain arrays, or ``None`` if absent."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as handle:
        out = {key: handle[key] for key in handle.files if key != "metadata"}
        out["metadata"] = json.loads(str(handle["metadata"]))
    return out


def _tiers(metadata: dict[str, Any]) -> dict[str, int]:
    """Parse the ``fallback_tiers={...}`` tally out of ``metadata.notes``."""
    notes = str(metadata.get("notes", ""))
    if "fallback_tiers=" not in notes:
        return {}
    blob = notes.split("fallback_tiers=", 1)[1].strip()
    try:
        return dict(json.loads(blob))
    except (TypeError, ValueError):
        return {}


def cell_row(suite: str, dataset: str, path: Path, data: dict[str, Any]) -> CellRow:
    """Build the manifest row for one cell."""
    status = data["status"].astype(str)
    meta = data["metadata"]
    return CellRow(
        suite=suite,
        dataset=dataset,
        representation=ARM,
        path=str(path),
        n_graphs=int(status.size),
        n_ok=int((status == "ok").sum()),
        n_censored=int((status == "censored").sum()),
        n_error=int((status == "error").sum()),
        fallback_tiers=_tiers(meta),
        engine=str(meta.get("isalgraph_engine", "")),
        build_hash=str(meta.get("isalgraph_build_hash", "")),
        src_commit=str(meta.get("src_commit", "")),
        code_commit=str(meta.get("code_commit", "")),
        seed=int(meta.get("seed", SEED)),
        budget_s=float(meta.get("encode_budget_s", -1.0)),
        generated_utc=str(meta.get("generated_utc", "")),
    )


def _band_mask(node_counts: np.ndarray, low: int, high: int) -> np.ndarray:
    return (node_counts >= low) & (node_counts <= high)


def band_stats(new: dict[str, Any], old: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Completion, fallback tiers and paired symbol saving, per band.

    Args:
        new: The exhaustive arm's arrays.
        old: The pruned arm's arrays, or ``None`` when the baseline cell is
            missing. The saving is then reported as ``None`` rather than 0.

    Returns:
        Band name -> statistics.
    """
    node_counts = np.asarray(new["node_counts"], dtype=np.int64)
    status = new["status"].astype(str)
    fallback = np.asarray(new["fallback_used"], dtype=bool)
    length = np.asarray(new["length"], dtype=np.int64)

    paired_new = paired_old = None
    if old is not None:
        # Join on graph_ids -- the two arms may censor different graphs, so a
        # positional join would silently compare different cohorts.
        new_ids = new["graph_ids"].astype(str)
        old_ids = old["graph_ids"].astype(str)
        order = {gid: i for i, gid in enumerate(old_ids)}
        idx = np.array([order.get(gid, -1) for gid in new_ids], dtype=np.int64)
        old_len = np.asarray(old["length"], dtype=np.int64)
        old_status = old["status"].astype(str)
        ok = (
            (idx >= 0)
            & np.isin(status, USABLE)
            & np.isin(np.where(idx >= 0, old_status[idx], "error"), USABLE)
        )
        paired_new = np.where(ok, length, -1)
        paired_old = np.where(ok, np.where(idx >= 0, old_len[idx], -1), -1)

    out: dict[str, dict[str, Any]] = {}
    for name, low, high in BANDS:
        mask = _band_mask(node_counts, low, high)
        n = int(mask.sum())
        if n == 0:
            continue
        usable = mask & np.isin(status, USABLE)
        entry: dict[str, Any] = {
            "n_graphs": n,
            "n_complete": int(usable.sum()),
            "completion_rate": float(usable.sum() / n),
            "n_fallback": int((mask & fallback).sum()),
            "fallback_rate": float((mask & fallback).sum() / n),
            "n_error": int((mask & (status == "error")).sum()),
            "median_symbols": float(np.median(length[usable])) if usable.any() else None,
        }
        if paired_new is not None and paired_old is not None:
            good = mask & (paired_new >= 0) & (paired_old > 0)
            if good.any():
                ratio = paired_new[good] / paired_old[good]
                entry["n_paired"] = int(good.sum())
                entry["median_saving_pct"] = float(100.0 * (1.0 - np.median(ratio)))
                entry["mean_saving_pct"] = float(100.0 * (1.0 - ratio.mean()))
                entry["n_exhaustive_longer"] = int((paired_new[good] > paired_old[good]).sum())
                entry["n_strictly_shorter"] = int((paired_new[good] < paired_old[good]).sum())
        out[name] = entry
    return out


def headline_bits(cells: list[tuple[str, str, dict[str, Any]]], node_count: int) -> dict[str, Any]:
    """Median entropy bits at exactly *node_count*, pooled over every cell.

    Args:
        cells: ``(suite, dataset, arrays)`` for every loaded cell.
        node_count: The exact ``n`` to select.

    Returns:
        The median and the sample size it was taken over.
    """
    values: list[float] = []
    for _suite, _dataset, data in cells:
        node_counts = np.asarray(data["node_counts"], dtype=np.int64)
        status = data["status"].astype(str)
        bits = np.asarray(data["entropy_bits"], dtype=np.float64)
        mask = (node_counts == node_count) & np.isin(status, USABLE) & np.isfinite(bits)
        values.extend(float(v) for v in bits[mask])
    if not values:
        return {"n": node_count, "n_graphs": 0, "median_entropy_bits": None}
    return {
        "n": node_count,
        "n_graphs": len(values),
        "median_entropy_bits": float(np.median(values)),
    }


def _pooled(per_cell: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Sum the per-cell band statistics into one cohort-level table."""
    out: dict[str, dict[str, Any]] = {}
    for name, _low, _high in BANDS:
        rows = [cell[name] for cell in per_cell if name in cell]
        if not rows:
            continue
        n = sum(r["n_graphs"] for r in rows)
        complete = sum(r["n_complete"] for r in rows)
        entry: dict[str, Any] = {
            "n_graphs": n,
            "n_complete": complete,
            "completion_rate": complete / n if n else None,
            "n_fallback": sum(r["n_fallback"] for r in rows),
            "fallback_rate": sum(r["n_fallback"] for r in rows) / n if n else None,
            "n_error": sum(r["n_error"] for r in rows),
        }
        savings = [r["median_saving_pct"] for r in rows if "median_saving_pct" in r]
        weights = [r["n_paired"] for r in rows if "median_saving_pct" in r]
        if savings:
            entry["n_paired"] = sum(weights)
            entry["median_saving_pct_weighted"] = float(np.average(savings, weights=weights))
            entry["n_exhaustive_longer"] = sum(r.get("n_exhaustive_longer", 0) for r in rows)
            entry["n_strictly_shorter"] = sum(r.get("n_strictly_shorter", 0) for r in rows)
        out[name] = entry
    return out


def collect(new_root: Path, old_root: Path) -> dict[str, Any]:
    """Walk every cell and assemble the whole report payload."""
    rows: list[CellRow] = []
    per_cell_bands: list[dict[str, dict[str, Any]]] = []
    loaded: list[tuple[str, str, dict[str, Any]]] = []
    missing: list[str] = []
    per_cell_detail: dict[str, dict[str, Any]] = {}

    for suite, datasets in (("suite1", SUITE1), ("suite2", SUITE2)):
        for dataset in datasets:
            path = new_root / "encodings" / suite / f"{dataset}__{ARM}.npz"
            data = _load(path)
            if data is None:
                missing.append(f"{suite}/{dataset}")
                continue
            baseline = _load(old_root / "encodings" / suite / f"{dataset}__{BASELINE}.npz")
            rows.append(cell_row(suite, dataset, path, data))
            bands = band_stats(data, baseline)
            per_cell_bands.append(bands)
            per_cell_detail[f"{suite}/{dataset}"] = bands
            loaded.append((suite, dataset, data))

    return {
        "cells": [asdict(row) for row in rows],
        "missing_cells": missing,
        "bands_pooled": _pooled(per_cell_bands),
        "bands_per_cell": per_cell_detail,
        "headline": [headline_bits(loaded, n) for n in HEADLINE_NODE_COUNTS],
        "n_cells": len(rows),
    }


def _fmt(value: float | None, spec: str = ".1f") -> str:
    """Format a possibly-absent number. ``None`` renders as ``--``, never 0."""
    return "--" if value is None else format(value, spec)


def render_summary(payload: dict[str, Any], repo: Path) -> str:
    """Render ``SUMMARY.md``."""
    cells = payload["cells"]
    budgets = sorted({row["budget_s"] for row in cells})
    engines = sorted({row["engine"] for row in cells})
    hashes = sorted({row["build_hash"] for row in cells})
    commits = sorted({row["src_commit"] for row in cells})

    lines: list[str] = []
    add = lines.append
    add("# T06_exhaustive -- campaign summary")
    add("")
    add(
        "The T-06 IsalGraph arm re-encoded with `canonical_string` (the true `w*_G`) "
        "in place of the length-suboptimal `pruned_canonical_string`."
    )
    add("")
    add("## What ran")
    add("")
    add(f"- **Cells**: {payload['n_cells']} of 15 `(suite, dataset)` pairs.")
    add(f"- **Budget**: {', '.join(_fmt(b, '.0f') for b in budgets)} s per graph.")
    add(f"- **Engine**: {', '.join(engines)}; build hash {', '.join(hashes)}.")
    add(f"- **Seed**: {SEED}. **src_commit**: {', '.join(c[:12] for c in commits)}.")
    add(f"- **Repo HEAD at report time**: `{_git_head(repo)[:12]}`.")
    add("")
    add(
        "> The budget is recorded in every cell's `metadata.encode_budget_s`. "
        "A censoring rate is a property of its budget and is not quotable without it."
    )
    add("")
    if payload["missing_cells"]:
        add(f"**Cells that did NOT run**: {', '.join(payload['missing_cells'])}")
    else:
        add("**Every cell completed.** No cell failed.")
    add("")
    add("## Completion, fallback and saving, per node-count band")
    add("")
    add(
        "| band | graphs | complete | fallback | median saving vs pruned | "
        "strictly shorter | longer |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|")
    for name, _low, _high in BANDS:
        row = payload["bands_pooled"].get(name)
        if row is None:
            continue
        add(
            f"| `{name}` | {row['n_graphs']:,} | "
            f"{row['n_complete']:,} ({100 * row['completion_rate']:.2f} %) | "
            f"{row['n_fallback']:,} ({100 * row['fallback_rate']:.2f} %) | "
            f"{_fmt(row.get('median_saving_pct_weighted'), '.2f')} % | "
            f"{row.get('n_strictly_shorter', 0):,} | "
            f"{row.get('n_exhaustive_longer', 0):,} |"
        )
    add("")
    add(
        "Saving is **paired per graph on the `graph_ids` join**, never a difference "
        "of two cohort medians: the two arms censor different graphs, so their "
        "marginal medians are taken over different sets."
    )
    add("")
    add("## Fallback tiers")
    add("")
    tiers: dict[str, int] = {}
    for row in cells:
        for tier, count in row["fallback_tiers"].items():
            tiers[tier] = tiers.get(tier, 0) + count
    if tiers:
        add("| tier | graphs | what it means |")
        add("|---|---:|---|")
        meaning = {
            "pruned": "still a canonical form; the row stays inside the completeness theorem",
            "greedy": "greedy-min; **not** canonical, outside the completeness theorem",
        }
        for tier, count in sorted(tiers.items()):
            add(f"| `{tier}` | {count:,} | {meaning.get(tier, '')} |")
    else:
        add("No graph needed a fallback at this budget.")
    add("")
    add("## Headline -- median entropy bits")
    add("")
    add("| n | this arm | pruned arm | nauty sparse6 |")
    add("|---|---:|---:|---:|")
    reference = {20: (136, 144), 40: (349, 336)}
    for entry in payload["headline"]:
        n = entry["n"]
        pruned, sparse6 = reference.get(n, (None, None))
        add(
            f"| {n} | {_fmt(entry['median_entropy_bits'], '.1f')} "
            f"({entry['n_graphs']:,} graphs) | {pruned} | {sparse6} |"
        )
    add("")
    add("## Per-cell detail")
    add("")
    add("| cell | graphs | ok | censored | error | budget |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in cells:
        add(
            f"| `{row['suite']}/{row['dataset']}` | {row['n_graphs']:,} | "
            f"{row['n_ok']:,} | {row['n_censored']:,} | {row['n_error']:,} | "
            f"{row['budget_s']:.0f} s |"
        )
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path("/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source")
    ap.add_argument("--new-root", type=Path, default=root / "T06_exhaustive")
    ap.add_argument("--old-root", type=Path, default=root / "T06")
    ap.add_argument("--repo", type=Path, default=Path("/home/mpascual/research/code/IsalGraph"))
    args = ap.parse_args(argv)

    payload = collect(args.new_root, args.old_root)
    manifest = args.new_root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    LOGGER.info("wrote %s (%d cells)", manifest, payload["n_cells"])

    summary = args.new_root / "SUMMARY.md"
    summary.write_text(render_summary(payload, args.repo))
    LOGGER.info("wrote %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
