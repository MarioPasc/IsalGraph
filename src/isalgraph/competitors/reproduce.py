"""``python -m isalgraph.competitors.reproduce`` -- the reproduction gate.

The ``competitors/`` folder is the evidence base for five plan-level
findings that reach a printed number, and it was produced by scratch code.
**Until the shipped module reproduces it, the plan rests on unreproduced
measurements.**  That is why this is acceptance criterion 1 and not a
nice-to-have.

Wave 0 established that the gate has two halves with different obligations,
and the design note was amended accordingly on 2026-08-15 (PI-signed).

**``--mode artefacts`` -- provenance.**  Replays each scout script's own
``Random(42)`` stream and asserts against **that script's raw artefact**,
never against the README table.  Replaying ``real_suite1.py`` means
consuming the 50-graph F3 draw and all 5 x 50 x 20 ``shuffled_copy(rng)``
calls before the rho draw: the stream, not just the seed.  A mismatch here
is a behaviour change and **stops the ticket**.

**``--mode table`` -- the corrected measurement.**  Recomputes rho once,
from one script, on one draw per dataset, under the frozen conventions:
**column-wise** adjacency and the **shared-vocabulary** WL at ``h = 2``.
That output is what T-06, T-17 and T-20 quote.

**README §4.1 as printed is neither of those.**  It is a composite of three
draws -- most rows from ``real_size_null.py``, AGM from ``real_suite1.py``,
WL from ``real_wl.py`` -- and it differs from ``real_suite1.out`` by up to
0.074.  It is superseded by ``--mode table``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.base import VectorBackend, table_scope_error
from isalgraph.competitors.ged_reference import load_ged
from isalgraph.competitors.registry import (
    get_backend,
    get_metric,
    get_repr_backend,
    get_vector_backend,
)

#: Where the scout's raw artefacts live.  A repo path, so `src/` stays
#: importable when they are absent -- the gate skips rather than crashing.
DEFAULT_ARTEFACT_DIR = os.path.join(".claude", "notes", "review", "plan", "competitors", "scratch")

#: Suite-1 dataset order, which is also the order the scout's scripts
#: advanced their shared rng through.  **Changing it desynchronises the
#: replay**, which is why it is frozen here rather than derived.
REPLAY_ORDER: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids",
)

SEED = 42
RHO_SAMPLE = 200
F3_GRAPHS = 50
F3_RELABELLINGS = 20

#: Tolerance for the provenance half.  These are deterministic given the
#: same stream; 1e-9 leaves room for float summation order and nothing else.
ARTEFACT_TOL = 1e-9

#: Expected running-example strings.  Column-wise adjacency, corrected
#: 2026-08-15 -- the previously quoted '101001000100111' was row-major.
RUNNING_EXAMPLE_EXPECTED: dict[str, tuple[str, str]] = {
    "adjacency": ("101101000100011", "101001000100011"),
    "graph6": ("ElCW", "EhCW"),
    # The ':' is framing, not payload, so it is EXCLUDED from `symbols` and
    # RETAINED in `wire` (CONTRACTS §9, design §4.1). Both halves are asserted:
    # see RUNNING_EXAMPLE_WIRE_EXPECTED.
    "sparse6": ("EaWIzR", "EaYms"),
    "nauty_graph6": ("E@ro", "E@po"),
    "agm_cam": ("000001110011110", "000001011111000"),
}

#: The emitted bytes, where they differ from the symbol sequence.  sparse6 is
#: the only case: its ``':'`` prefix is counted in ``realised_bits`` and not in
#: ``entropy_bits``, and asserting only one of the two would let the convention
#: drift in whichever direction nobody checked.
RUNNING_EXAMPLE_WIRE_EXPECTED: dict[str, tuple[bytes, bytes]] = {
    "sparse6": (b":EaWIzR", b":EaYms"),
}

#: K33 vs the triangular prism.  Every canonical backend separates them;
#: WL does not, at any h.
WITNESS_EXPECTED: dict[str, tuple[str, str]] = {
    "nauty_graph6": ("Es\\o", "E{Sw"),
    "agm_cam": ("000111111011100", "001101110111100"),
}

#: The scout's ENCODERS dict, in insertion order.  The F3 block iterates it,
#: so the count of shuffled_copy calls -- and therefore the rng position at
#: the rho draw -- depends on this having five entries.
_REPLAY_ENCODER_COUNT = 5


# --------------------------------------------------------------------------- #
# Legacy reproductions -- for the provenance half only
# --------------------------------------------------------------------------- #


def _legacy_wl_features(graph: nx.Graph, h: int) -> dict[str, int]:
    """``scratch/backends.py::wl_features``, verbatim.  **Reproduction only.**

    This implementation compresses colours to small integers **per graph,
    per round**, and builds the next round's signature from those compressed
    labels.  The table is built from one graph's own signature set, so
    **features from rounds >= 2 are not comparable across graphs** -- which
    is trap 3 of ``competitors/README.md`` §6, committed by the file that
    documents it.  It produced README §4.1's WL row.

    It is deliberately private, deliberately not a registered backend, and
    exists for exactly one purpose: showing that the difference between the
    scout's WL numbers and ours is the *convention*, not our code.  **Do not
    use it for anything a paper prints.**
    """
    colour = {v: "0" for v in graph}
    feats: collections.Counter[str] = collections.Counter()
    feats.update(f"0:{c}" for c in colour.values())
    for it in range(1, h + 1):
        new = {}
        for v in graph:
            new[v] = colour[v] + "|" + ",".join(sorted(colour[w] for w in graph.neighbors(v)))
        table = {s: str(i) for i, s in enumerate(sorted(set(new.values())))}
        colour = {v: table[s] for v, s in new.items()}
        feats.update(f"{it}:{new[v]}" for v in graph)
    return dict(feats)


def _legacy_adjacency_rowmajor(graph: nx.Graph) -> str:
    """``scratch/backends.py::adjacency_bits``, row-major.  **Reproduction only.**

    The frozen convention is **column-wise**, because that is what graph6's
    payload and AGM's own code use and what criterion 3's family identity
    requires.  The scout measured row-major; this reproduces their number so
    the discrepancy is attributable to the reading order rather than left
    open.  **Not what the shipped ``adjacency`` backend emits.**
    """
    nodes = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    matrix = [[0] * n for _ in range(n)]
    for u, v in graph.edges():
        matrix[index[u]][index[v]] = matrix[index[v]][index[u]] = 1
    return "".join(str(matrix[i][j]) for i in range(n) for j in range(i + 1, n))


# --------------------------------------------------------------------------- #
# The running example and the witness
# --------------------------------------------------------------------------- #


def check_running_example() -> dict[str, Any]:
    """Every backend's expected string on ``G`` and ``H = G - (0,3)``."""
    graph_g = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    graph_h = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE)
    out: dict[str, Any] = {}
    for name, (want_g, want_h) in RUNNING_EXAMPLE_EXPECTED.items():
        try:
            backend = get_repr_backend(name)
            got_g = "".join(backend.encode(graph_g).symbols)
            got_h = "".join(backend.encode(graph_h).symbols)
            wire_g = backend.encode(graph_g).wire
            wire_h = backend.encode(graph_h).wire
        except Exception as exc:  # noqa: BLE001 - an absent backend is a reportable state
            out[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
            continue
        record: dict[str, Any] = {
            "status": "pass" if (got_g, got_h) == (want_g, want_h) else "FAIL",
            "expected": [want_g, want_h],
            "got": [got_g, got_h],
        }
        want_wire = RUNNING_EXAMPLE_WIRE_EXPECTED.get(name)
        if want_wire is not None:
            record["wire_expected"] = [w.decode() for w in want_wire]
            record["wire_got"] = [w.decode() if w else None for w in (wire_g, wire_h)]
            if (wire_g, wire_h) != want_wire:
                record["status"] = "FAIL"
        out[name] = record
    return out


def check_witness() -> dict[str, Any]:
    """K33 vs the prism: WL gives 0.0, every canonical backend separates them."""
    k33 = fixtures.to_networkx(fixtures.K33)
    prism = fixtures.to_networkx(fixtures.PRISM)
    out: dict[str, Any] = {}
    for name, (want_a, want_b) in WITNESS_EXPECTED.items():
        try:
            backend = get_repr_backend(name)
            got = (
                "".join(backend.encode(k33).symbols),
                "".join(backend.encode(prism).symbols),
            )
        except Exception as exc:  # noqa: BLE001
            out[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
            continue
        out[name] = {
            "status": "pass" if got == (want_a, want_b) else "FAIL",
            "expected": [want_a, want_b],
            "got": list(got),
            "separates": got[0] != got[1],
        }
    try:
        wl = get_vector_backend("wl_subtree")
        wl.fit([k33, prism])
        kernel = get_metric("kernel")
        distance = kernel.distance(wl.features(k33), wl.features(prism))
        out["wl_subtree"] = {
            "status": "pass" if distance == 0.0 else "FAIL",
            "distance": distance,
            "expected": 0.0,
            "note": "1-WL cannot separate two 3-regular graphs on six vertices at any h",
        }
    except Exception as exc:  # noqa: BLE001
        out["wl_subtree"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return out


def check_wl_identity() -> dict[str, Any]:
    """``grakel(n_iter=k) == ours(h=k)``, and the 5.830952 value.

    Corrected 2026-08-15: there is **no** off-by-one.  ``grakel(n_iter=2)``
    is 5.830952 and ``grakel(n_iter=3)`` is 7.211103.
    """
    graph_g = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    graph_h = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE)
    expected = {1: 2.0, 2: 5.830951894845301, 3: 7.211102550927978}
    out: dict[str, Any] = {"expected": expected, "measured": {}}
    try:
        kernel = get_metric("kernel")
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    for h, want in expected.items():
        # A fresh instance per h, not a mutated attribute: the frozen value is
        # h = 2 and nothing outside this check may reach an instance at another
        # value. Tuning h on rho is the error competitors.md 3.4 forbids.
        try:
            backend = get_vector_backend("wl_subtree", h=h)
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        backend.fit([graph_g, graph_h])
        got = kernel.distance(backend.features(graph_g), backend.features(graph_h))
        out["measured"][h] = got
        if abs(got - want) > 1e-9:
            out["status"] = "FAIL"
    out.setdefault("status", "pass")
    return out


# --------------------------------------------------------------------------- #
# 1a -- provenance: replay each scout script's stream
# --------------------------------------------------------------------------- #


def _replay_stream_to_rho_draw(
    rng: random.Random, graphs: list[nx.Graph], n: int
) -> tuple[int, ...]:
    """Advance *rng* exactly as ``real_suite1.py``'s F3 block does, then draw.

    The F3 block calls ``shuffled_copy(rng)`` five times per graph per
    relabelling -- once for each entry of the ``ENCODERS`` dict -- and each
    call consumes three shuffles whose lengths depend on ``n`` and ``m``.
    Only the draw at the end matters for rho, but the consumption must match
    exactly to reach it.
    """
    f3_idx = rng.sample(range(n), min(F3_GRAPHS, n))
    for _ in range(_REPLAY_ENCODER_COUNT):
        for i in f3_idx:
            for _ in range(F3_RELABELLINGS):
                fixtures.shuffled_copy(graphs[i], rng)
    return tuple(sorted(rng.sample(range(n), min(RHO_SAMPLE, n))))


def _spearman(xs: list[float], ys: list[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(xs, ys).statistic)


#: Backend name in the shipped registry -> key in ``real_suite1.json``.
_SUITE1_KEYS = {
    "graph6": "graph6",
    "sparse6": "sparse6",
    "nauty_graph6": "nauty->graph6",
    "agm_cam": "AGM CAM",
    "min_dfs": "min-DFS (tuples)",
    "isalgraph_pruned": "IsalGraph pruned",
    "isalgraph_canonical": "IsalGraph canonical",
}


def replay_suite1(artefact_dir: str) -> dict[str, Any]:
    """Replay ``real_suite1.py`` and compare against ``real_suite1.json``.

    ``adjacency`` is checked under the **scout's row-major** reading, via
    :func:`_legacy_adjacency_rowmajor`, and the shipped column-wise value is
    reported beside it.  Both numbers are real; they answer different
    questions, and printing only one is how the discrepancy stayed hidden.
    """
    path = os.path.join(artefact_dir, "real_suite1.json")
    if not os.path.exists(path):
        return {"status": "skipped", "reason": f"{path} not found"}
    with open(path, encoding="utf-8") as handle:
        reference = json.load(handle)

    rng = random.Random(SEED)
    out: dict[str, Any] = {"status": "pass", "datasets": {}}
    for dataset in REPLAY_ORDER:
        cohort = datasets.load(dataset)
        graphs = list(cohort.graphs)
        indices = _replay_stream_to_rho_draw(rng, graphs, len(graphs))
        ged = load_ged(dataset)
        pairs = ged.certified_pairs(indices)
        geds = [float(ged.ged[a, b]) for a, b in pairs]
        want = reference[dataset]["rho"] if "rho" in reference[dataset] else reference[dataset]

        row: dict[str, Any] = {"n_pairs": len(pairs)}
        levenshtein = get_metric("levenshtein")

        from rapidfuzz.distance import Levenshtein as _lev  # noqa: N813

        legacy = {i: _legacy_adjacency_rowmajor(graphs[i]) for i in indices}
        got = _spearman([float(_lev.distance(legacy[a], legacy[b])) for a, b in pairs], geds)
        row["adjacency_rowmajor_scout"] = _compare(got, want.get("adjacency"))

        for name, key in _SUITE1_KEYS.items():
            expected = want.get(key)
            if expected is None:
                continue
            try:
                backend = get_repr_backend(name)
                encoded = {i: backend.encode(graphs[i]) for i in indices}
            except Exception as exc:  # noqa: BLE001
                row[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                continue
            distances = [levenshtein.distance(encoded[a], encoded[b]) for a, b in pairs]
            row[name] = _compare(_spearman(distances, geds), expected)

        try:
            shipped = get_repr_backend("adjacency")
            enc = {i: shipped.encode(graphs[i]) for i in indices}
            row["adjacency_columnwise_shipped"] = {
                "status": "reported",
                "rho": _spearman([levenshtein.distance(enc[a], enc[b]) for a, b in pairs], geds),
                "note": "the frozen convention; no prior value to match",
            }
        except Exception as exc:  # noqa: BLE001
            row["adjacency_columnwise_shipped"] = {"status": "unavailable", "error": str(exc)}

        if any(isinstance(v, dict) and v.get("status") == "FAIL" for v in row.values()):
            out["status"] = "FAIL"
        out["datasets"][dataset] = row
    return out


def _compare(got: float, expected: float | dict[str, Any] | None) -> dict[str, Any]:
    if expected is None:
        return {"status": "no reference", "got": got}
    want = expected["spearman"] if isinstance(expected, dict) else float(expected)
    delta = got - want
    return {
        "status": "pass" if abs(delta) <= ARTEFACT_TOL else "FAIL",
        "expected": want,
        "got": got,
        "delta": delta,
    }


# --------------------------------------------------------------------------- #
# 1b -- the corrected table
# --------------------------------------------------------------------------- #


def corrected_table(
    dataset_names: tuple[str, ...] = REPLAY_ORDER, *, n_graphs: int = RHO_SAMPLE
) -> dict[str, Any]:
    """Recompute rho once, one draw per dataset, under the frozen conventions.

    One script, one seed, one convention for every cell.  Emits the
    all-pairs view (§4.1) and the **equal-``n`` restriction** (§4.2), which
    removes the size channel entirely and is the comparison the paper should
    lead with: there the canonical/non-canonical gap on Letter LOW is 0.42,
    which the all-pairs view hides because the size channel floats everyone.

    **The size null is emitted in the same record**, not by a later script.
    """
    from rapidfuzz.distance import Levenshtein as _lev  # noqa: N813

    out: dict[str, Any] = {
        "convention": {
            "adjacency": "strict upper triangle, COLUMN-WISE",
            "wl": "shared vocabulary, h = 2 (grakel n_iter = 2)",
            "distance": "symbol-level Levenshtein; WL uses its kernel distance",
            "sample": f"one seed-{SEED} {n_graphs}-graph draw per dataset",
        },
        "supersedes": "competitors/README.md 4.1 and 4.2, a three-draw composite",
        "datasets": {},
    }
    for dataset in dataset_names:
        cohort = datasets.load(dataset)
        indices = cohort.sample(n_graphs, seed=SEED)
        ged = load_ged(dataset)
        pairs = ged.certified_pairs(indices)
        if not pairs:
            continue
        geds = [float(ged.ged[a, b]) for a, b in pairs]
        orders = {i: cohort.graphs[i].number_of_nodes() for i in indices}
        equal_n = [k for k, (a, b) in enumerate(pairs) if orders[a] == orders[b]]

        row: dict[str, Any] = {
            "n_pairs": len(pairs),
            "n_equal_n_pairs": len(equal_n),
            "size_null": {
                "all_pairs": _spearman([float(abs(orders[a] - orders[b])) for a, b in pairs], geds),
                "equal_n": None,  # constant by construction; reported as such
            },
        }
        from isalgraph.competitors.registry import available_backends

        for name in available_backends():
            try:
                backend = get_backend(name)
                scope = table_scope_error(backend.capabilities, cohort.suite, name)
                if scope is not None:
                    row[name] = {"status": "refused", "reason": scope}
                    continue
                if isinstance(backend, VectorBackend):
                    backend.fit([cohort.graphs[i] for i in indices])
                    feats = {i: dict(backend.features(cohort.graphs[i])) for i in indices}
                    metric = get_metric("kernel")
                    distances = [metric.distance(feats[a], feats[b]) for a, b in pairs]
                else:
                    codes = {i: backend.encode(cohort.graphs[i]).symbols for i in indices}
                    distances = [float(_lev.distance(codes[a], codes[b])) for a, b in pairs]
            except Exception as exc:  # noqa: BLE001 - a ceiling is a reported result
                row[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                continue
            row[name] = {
                "all_pairs": _spearman(distances, geds),
                "equal_n": (
                    _spearman([distances[k] for k in equal_n], [geds[k] for k in equal_n])
                    if len(equal_n) >= 3
                    else None
                ),
            }
        out["datasets"][dataset] = row
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 1 if any provenance check failed."""
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.reproduce")
    parser.add_argument("--mode", choices=("artefacts", "table", "all"), default="all")
    parser.add_argument("--artefact-dir", default=DEFAULT_ARTEFACT_DIR)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {"mode": args.mode}
    failed = False

    if args.mode in ("artefacts", "all"):
        payload["running_example"] = check_running_example()
        payload["witness"] = check_witness()
        payload["wl_identity"] = check_wl_identity()
        payload["suite1_replay"] = replay_suite1(args.artefact_dir)
        for section in ("running_example", "witness", "wl_identity", "suite1_replay"):
            blob = json.dumps(payload[section])
            if '"FAIL"' in blob:
                failed = True

    if args.mode in ("table", "all"):
        payload["corrected_table"] = corrected_table()

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)

    print(f"mode={args.mode}  ->  {args.out}")
    for section in ("running_example", "witness", "wl_identity"):
        if section in payload:
            print(f"  {section}: {json.dumps(payload[section], default=str)[:400]}")
    if failed:
        print("\nAT LEAST ONE PROVENANCE CHECK FAILED. This stops the ticket (design note 9.1).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
