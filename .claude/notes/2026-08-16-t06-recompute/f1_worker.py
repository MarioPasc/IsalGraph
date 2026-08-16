"""F-1 re-verification worker: encode a contiguous band of graphs, streaming.

Calls ``isalgraph.core.backends`` DIRECTLY, bypassing the competitors registry.
That is the whole point: the registry refuses ``isalgraph_canonical`` above
n = 12 with ``SuiteScopeError`` before attempting any encode, so a completion
rate measured through it is a property of the scope guard, not of the encoder.

Prints ``BEGIN <idx>`` before each graph and ``DONE <idx> <seconds> <length>``
after, so the parent can enforce a per-graph deadline by killing the process
when a BEGIN is not followed by its DONE in time. One process per band means
engine warm-up is paid once per band rather than once per graph -- the defect
that inflated a single COIL-DEL encode from 4.95 ms to 578.95 ms.

Usage: f1_worker.py <npz> <encoder> <start> <stop>
"""

from __future__ import annotations

import sys
import time

import numpy as np

from isalgraph.core.backends import canonical_string, pruned_canonical_string
from isalgraph.core.sparse_graph import SparseGraph

ENCODERS = {"canonical": canonical_string, "pruned": pruned_canonical_string}


def build(off: np.ndarray, edges: np.ndarray, n_nodes: np.ndarray, k: int) -> SparseGraph:
    n = int(n_nodes[k])
    sg = SparseGraph(max_nodes=n, directed_graph=False)
    for _ in range(n):
        sg.add_node()
    seen: set[tuple[int, int]] = set()
    for u, v in zip(edges[0, off[k] : off[k + 1]].tolist(), edges[1, off[k] : off[k + 1]].tolist()):
        a, b = (u, v) if u < v else (v, u)
        if a != b and (a, b) not in seen:
            seen.add((a, b))
            sg.add_edge(a, b)
    return sg


def main() -> None:
    npz_path, encoder, start, stop = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with np.load(npz_path, allow_pickle=False) as z:
        off, edges, n_nodes = z["edge_offsets"], z["edges"], z["n_nodes"]

    fn = ENCODERS[encoder]
    # Warm the engine once, off the clock, so the first measured graph is not
    # the one that pays for dynamic symbol resolution.
    warm = SparseGraph(max_nodes=3, directed_graph=False)
    for _ in range(3):
        warm.add_node()
    warm.add_edge(0, 1)
    warm.add_edge(1, 2)
    fn(warm)

    for k in range(start, stop):
        print(f"BEGIN {k}", flush=True)
        t0 = time.process_time()
        w = fn(build(off, edges, n_nodes, k))
        print(f"DONE {k} {time.process_time() - t0:.6f} {len(w)}", flush=True)


if __name__ == "__main__":
    main()
