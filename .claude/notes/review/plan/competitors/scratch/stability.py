"""Edit-distance compatibility, measured: how far does ONE unit edit move each code?

competitors.md §4 pre-commits the claim that a canonical form can be *unique* without
being *stable* -- two similar graphs may receive very different canonical labellings.
That is an empirical claim and this measures it.

Protocol
--------
Draw a connected graph G, apply exactly one unit edit under the D6 cost model (an edge
insertion or a connectivity-preserving edge deletion), **relabel the result at random**,
and encode both.  The relabelling matters: without it the edited copy inherits G's
vertex numbering and every order-dependent format differs in exactly one bit, which
would flatter graph6, sparse6 and the raw adjacency matrix by an artefact.

Two quantities per representation:

    d1     median distance over 120 (G, G+1 edit) pairs      -- exact unit GED = 1
    drand  median distance over 120 random pairs matched on n -- the noise floor

`sep = d1 / drand` is the signal-to-noise ratio a GED proxy actually lives on.  A small
`d1` means nothing if `drand` is equally small; Spearman rho tracks separation, not
absolute displacement.
"""

from __future__ import annotations

import json
import random
import statistics

import agm_cam
import backends as B
import min_dfs
import networkx as nx
from sweep import random_connected, shuffled_copy

N_PAIRS = 120
SEED = 42


def one_edit(G: nx.Graph, rng: random.Random) -> nx.Graph | None:
    for _ in range(60):
        if rng.random() < 0.5 and G.number_of_edges() > G.number_of_nodes() - 1:
            u, v = rng.choice(list(G.edges()))
            H = G.copy()
            H.remove_edge(u, v)
            if nx.is_connected(H):
                return H
        else:
            non = [(u, v) for u in G for v in G if u < v and not G.has_edge(u, v)]
            if not non:
                continue
            u, v = rng.choice(non)
            H = G.copy()
            H.add_edge(u, v)
            return H
    return None


def tuple_string(G: nx.Graph) -> str:
    """min-DFS code with one Unicode symbol per DFS edge: an edit costs 1, not 4."""
    return "".join(chr(0xE000 + i * 100 + j) for i, j, *_ in min_dfs.min_dfs_code(G))


def encode(G: nx.Graph, isal: dict) -> dict[str, str | None]:
    row: dict[str, str | None] = {
        "graph6": B.graph6(G),
        "sparse6": B.sparse6(G),
        "nauty->graph6": B.nauty_canon_graph6(G),
        "adjacency": B.adjacency_bits(G),
        "min-DFS (chars)": B.mdfsc(G),
        "min-DFS (tuples)": tuple_string(G),
        "IsalGraph canonical": isal["canonical"],
        "IsalGraph pruned": isal["pruned"],
    }
    try:
        row["AGM CAM"] = B.agm_code(G, budget=300_000)[0]
    except agm_cam.AGMBudgetExceeded:
        row["AGM CAM"] = None
    return row


REPS = [
    "graph6",
    "sparse6",
    "nauty->graph6",
    "adjacency",
    "AGM CAM",
    "min-DFS (chars)",
    "min-DFS (tuples)",
    "IsalGraph canonical",
    "IsalGraph pruned",
]


def main() -> None:
    rng = random.Random(SEED)

    edit_pairs: list[tuple[nx.Graph, nx.Graph]] = []
    while len(edit_pairs) < N_PAIRS:
        n = rng.choice([6, 7, 8, 9, 10, 11, 12])
        G = random_connected(n, rng.randint(n, min(2 * n, n * (n - 1) // 2)), rng)
        H = one_edit(G, rng)
        if H is not None:
            edit_pairs.append((G, shuffled_copy(H, rng)))  # <-- the fix

    rand_pairs: list[tuple[nx.Graph, nx.Graph]] = []
    while len(rand_pairs) < N_PAIRS:
        n = rng.choice([6, 7, 8, 9, 10, 11, 12])
        a = random_connected(n, rng.randint(n, min(2 * n, n * (n - 1) // 2)), rng)
        b = random_connected(n, rng.randint(n, min(2 * n, n * (n - 1) // 2)), rng)
        if not nx.is_isomorphic(a, b):
            rand_pairs.append((a, b))

    flat = [g for p in edit_pairs + rand_pairs for g in p]
    isal = B.isalgraph_strings(flat, budget=20.0)
    codes = [encode(g, s) for g, s in zip(flat, isal)]

    def stats(offset: int, count: int) -> dict[str, dict]:
        out = {}
        for r in REPS:
            lev, ham, lens = [], [], []
            for k in range(count):
                a = codes[offset + 2 * k][r]
                b = codes[offset + 2 * k + 1][r]
                if a is None or b is None:
                    continue
                lev.append(B.levenshtein(a, b))
                ham.append(B.hamming(a, b))
                lens.append(len(a))
            defined = [h for h in ham if h is not None]
            out[r] = {
                "n": len(lev),
                "median_len": statistics.median(lens),
                "lev_median": statistics.median(lev),
                "lev_max": max(lev),
                "ham_defined_pct": round(100 * len(defined) / max(len(ham), 1), 1),
                "ham_median": statistics.median(defined) if defined else None,
            }
        return out

    e = stats(0, N_PAIRS)
    r = stats(2 * N_PAIRS, N_PAIRS)

    print(
        f"{N_PAIRS} one-edit pairs (unit GED = 1, edited copy randomly relabelled) "
        f"and {N_PAIRS} random same-n pairs; n in [6, 12]\n"
    )
    print(
        f"{'representation':<22}{'N':>4}{'med L':>7}"
        f"{'Lev d1':>8}{'Lev rand':>10}{'sep':>7}{'d1 max':>8}"
        f"{'Ham def%':>10}{'Ham d1':>8}{'Ham rand':>10}"
    )
    out = {}
    for k in REPS:
        sep = e[k]["lev_median"] / r[k]["lev_median"] if r[k]["lev_median"] else float("nan")
        out[k] = {"edit": e[k], "random": r[k], "separation": round(sep, 3)}
        print(
            f"{k:<22}{e[k]['n']:>4}{e[k]['median_len']:>7.0f}"
            f"{e[k]['lev_median']:>8.1f}{r[k]['lev_median']:>10.1f}{sep:>7.2f}"
            f"{e[k]['lev_max']:>8}{e[k]['ham_defined_pct']:>10.1f}"
            f"{str(e[k]['ham_median']):>8}{str(r[k]['ham_median']):>10}"
        )

    print("\nsep = median Levenshtein on one-edit pairs / median on random pairs.")
    print("Lower is better: it is the fraction of the noise floor a single unit edit costs.")

    with open("stability.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote stability.json")


if __name__ == "__main__":
    main()
