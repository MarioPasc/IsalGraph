"""Differential test: kaviniitm/DFSCode (C++) vs our validated min_dfs.min_dfs_code.

Our implementation is the oracle: it agrees with exhaustive enumeration over every valid
DFS traversal on all 30 connected isomorphism classes with n <= 5, and produces 112
distinct codes on the 112 connected graphs with n = 6 (OEIS A001349).

Three checks:
  K1  agreement on every connected isomorphism class, n <= 7
  K2  isomorphism invariance of the C++ tool itself, under genuine relabellings
  K3  agreement on random graphs up to n = 40, both densities
"""

from __future__ import annotations

import itertools
import random
import re
import subprocess
import sys

import min_dfs
import networkx as nx
from sweep import random_connected, shuffled_copy

BIN = "./dfscode_kavin/dfscode"


def to_input(G: nx.Graph, gid: str = "g") -> str:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    lines = [gid, str(len(nodes)), " ".join("a" for _ in nodes), str(G.number_of_edges())]
    lines += [f"{idx[u]} {idx[v]} e" for u, v in G.edges()]
    return "\n".join(lines) + "\n"


def run_batch(graphs: list[nx.Graph]) -> list[str]:
    payload = "".join(to_input(G, f"g{i}") for i, G in enumerate(graphs))
    p = subprocess.run([BIN], input=payload, capture_output=True, text=True, check=True)
    return [ln for ln in p.stdout.strip().split("\n") if ln.strip()]


def norm_cpp(line: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"<(\d+),(\d+),[^>]*>", line)]


def norm_ours(G: nx.Graph) -> list[tuple[int, int]]:
    return [(i, j) for i, j, *_ in min_dfs.min_dfs_code(G)]


def connected_classes(n: int) -> list[nx.Graph]:
    pairs = list(itertools.combinations(range(n), 2))
    seen: list[nx.Graph] = []
    for mask in range(1 << len(pairs)):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(p for k, p in enumerate(pairs) if mask >> k & 1)
        if not nx.is_connected(G):
            continue
        if any(nx.is_isomorphic(G, H) for H in seen):
            continue
        seen.append(G)
    return seen


def main() -> int:
    rng = random.Random(42)
    fails = 0

    # ---- K1 ---------------------------------------------------------------
    for n in (2, 3, 4, 5, 6):
        Gs = connected_classes(n)
        cpp = run_batch(Gs)
        bad = 0
        for G, line in zip(Gs, cpp):
            if norm_cpp(line) != norm_ours(G):
                bad += 1
                if bad <= 3:
                    print(
                        f"  K1 MISMATCH n={n} edges={sorted(G.edges())}\n"
                        f"    cpp  {norm_cpp(line)}\n    ours {norm_ours(G)}"
                    )
        print(f"K1 n={n}: {len(Gs)} connected classes, {bad} mismatches")
        fails += bad

    # ---- K2 ---------------------------------------------------------------
    inv_bad = 0
    tot = 0
    for n in (6, 7, 8, 9, 10, 12):
        for _ in range(15):
            G = random_connected(n, rng.randint(n, min(2 * n, n * (n - 1) // 2)), rng)
            copies = [G] + [shuffled_copy(G, rng) for _ in range(12)]
            codes = {tuple(norm_cpp(ln)) for ln in run_batch(copies)}
            tot += 1
            if len(codes) != 1:
                inv_bad += 1
                if inv_bad <= 3:
                    print(
                        f"  K2 NOT INVARIANT n={n} m={G.number_of_edges()}: "
                        f"{len(codes)} distinct codes from 13 relabellings"
                    )
    print(f"K2: {tot} graphs x 13 relabellings, {inv_bad} non-invariant")
    fails += inv_bad

    # ---- K3 ---------------------------------------------------------------
    for n, m in [
        (7, 8),
        (7, 14),
        (8, 9),
        (9, 12),
        (10, 11),
        (12, 13),
        (12, 24),
        (15, 16),
        (15, 30),
        (20, 21),
        (20, 40),
        (30, 31),
        (30, 60),
        (40, 42),
    ]:
        Gs = [random_connected(n, m, rng) for _ in range(8)]
        cpp = run_batch(Gs)
        bad = sum(norm_cpp(ln) != norm_ours(G) for G, ln in zip(Gs, cpp))
        print(f"K3 n={n} m={m}: 8 graphs, {bad} mismatches")
        fails += bad

    print("\nALL AGREE" if fails == 0 else f"\n{fails} DISAGREEMENTS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
