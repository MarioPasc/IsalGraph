"""Is the pruned canonical form ever shorter than the exhaustive one?

``canonical_pruned.py`` restricts each V/v branch to candidates sharing the
maximum structural triplet, and its own docstring records that the result "may
produce longer strings on some graphs". T-06's main arm is that pruned form,
so every compactness figure in the paper is computed on a length-suboptimal
canonical string. This script measures the gap where both arms exist.

It reads the archived Suite-1 encoding cells rather than re-encoding, so it
costs seconds and cannot disagree with the campaign. Suite 1 is the only place
both arms exist: ``isalgraph_canonical`` carries ``SUITE1_ONLY``, so above
``n = 12`` there is nothing to compare against.

Two things it checks, and the first is the one that matters:

**Direction.** The pruned form must never be *shorter* -- the exhaustive search
minimises over a strict superset of the pruned one's candidates. A single
counter-example would mean one of the two implementations is wrong, so this is
a free correctness check on 5,350 graphs.

**Magnitude.** How much the paper's arm pays, per node count. Monotone in ``n``
to the A* ceiling; it is not measurable above it and must not be extrapolated.

Usage::

    python .claude/notes/review/tasks/t06_pruned_vs_exhaustive.py [ENCODINGS_DIR]
"""

from __future__ import annotations

import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np

DEFAULT_ROOT = (
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06/encodings"
)

ALPHABET_SIZE = 9


def collect(root: str) -> dict[int, list[tuple[int, int]]]:
    """Return ``{n: [(pruned length, exhaustive length), ...]}`` for Suite 1.

    Args:
        root: The ``encodings/`` directory.

    Returns:
        Paired symbol counts per node count, over graphs both arms encoded.

    Raises:
        FileNotFoundError: If no pruned Suite-1 cell exists under *root*.
    """
    paths = sorted(glob.glob(f"{root}/suite1/*__isalgraph_pruned.npz"))
    if not paths:
        raise FileNotFoundError(f"no suite1 isalgraph_pruned cells under {root}")
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for path in paths:
        pruned = np.load(path, allow_pickle=True)
        exhaustive = np.load(path.replace("_pruned", "_canonical"), allow_pickle=True)
        position = {gid: i for i, gid in enumerate(exhaustive["graph_ids"])}
        pairs = [(i, position[g]) for i, g in enumerate(pruned["graph_ids"]) if g in position]
        pi = np.fromiter((a for a, _ in pairs), dtype=int, count=len(pairs))
        ci = np.fromiter((b for _, b in pairs), dtype=int, count=len(pairs))
        keep = (pruned["status"][pi] == "ok") & (exhaustive["status"][ci] == "ok")
        pi, ci = pi[keep], ci[keep]
        for n, a, b in zip(
            pruned["node_counts"][pi], pruned["length"][pi], exhaustive["length"][ci], strict=True
        ):
            out[int(n)].append((int(a), int(b)))
    return dict(out)


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: ``sys.argv``; the optional first argument overrides the archive
            root.

    Returns:
        ``0`` when the pruned form is never shorter, ``1`` otherwise -- a
        non-zero exit here is a correctness failure, not a reporting one.
    """
    root = argv[1] if len(argv) > 1 else os.environ.get("T06_ENCODINGS", DEFAULT_ROOT)
    data = collect(root)
    print(
        f"{'n':>4} {'graphs':>7} {'L_pruned':>9} {'L_exhaust':>10} "
        f"{'longer %':>9} {'shorter %':>10} {'median excess':>14}"
    )
    all_pruned: list[int] = []
    all_exhaustive: list[int] = []
    violations = 0
    for n in sorted(data):
        pruned = np.array([a for a, _ in data[n]])
        exhaustive = np.array([b for _, b in data[n]])
        all_pruned += list(pruned)
        all_exhaustive += list(exhaustive)
        delta = pruned - exhaustive
        violations += int(np.sum(delta < 0))
        print(
            f"{n:>4} {len(pruned):>7} {np.median(pruned):>9.1f} {np.median(exhaustive):>10.1f} "
            f"{100 * np.mean(delta > 0):>8.1f}% {100 * np.mean(delta < 0):>9.1f}% "
            f"{np.median(delta):>+14.1f}"
        )
    pruned = np.array(all_pruned, dtype=float)
    exhaustive = np.array(all_exhaustive, dtype=float)
    excess = float((pruned - exhaustive).mean())
    print(
        f"\npooled n <= 12: {len(pruned)} graphs; "
        f"mean L_pruned = {pruned.mean():.3f}, mean L_exhaustive = {exhaustive.mean():.3f}; "
        f"mean excess = {excess:+.3f} symbols "
        f"({100 * (pruned.mean() / exhaustive.mean() - 1):+.2f} %), "
        f"= {excess * math.log2(ALPHABET_SIZE):+.3f} entropy bits"
    )
    print(f"pruned shorter than exhaustive on {violations} graphs (must be 0)")
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
