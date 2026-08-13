"""GEDLIB bound bake-off harness -- T-27.

Evaluates every candidate graph-edit-distance bound method against the certified
exact-GED census produced by T-03, so that the manuscript's primary lower and
upper bounds are chosen by measurement rather than by citation. ``BRANCH_FAST``
was licensed on 400 LINUX pairs at ``n_bar = 8.71``; ``IPFP`` was never measured
against exact GED at all. This module produces the measurement.

Contract
--------
Outputs follow ``.claude/notes/2026-08-13-t27-bakeoff/CONTRACTS.md`` exactly:
one index file per dataset, one cell file per ``(dataset, method)``, plus timing
and determinism JSON. Every array is in the canonical pair order
``numpy.triu_indices(n, k=1)`` and is never reordered or compacted.

The three traps this module exists to survive
---------------------------------------------
1. **Import order.** ``gklearn.gedlib.libraries_import`` ``dlopen()``s
   libdoublefann/libsvm/libnomad and must load before ``gedlibpy_gxl``. It is
   loaded through :func:`importlib.import_module` because ruff and isort reorder
   plain ``from ... import`` lines alphabetically and break the ``dlopen``.

2. **The wrong accessor returns garbage, not an error.** ``get_lower_bound()``
   on an upper-bound method returns ``0.00`` and ``HED`` returns
   ``get_upper_bound() = inf``; neither raises, so a whole matrix fills silently
   with zeros. Three independent guards catch this:
   :func:`read_bound` rejects ``NaN``, ``inf`` and negatives per read;
   :func:`capability_probe` requires a strictly positive value on a pair of
   known non-zero distance *before* the pair loop runs; and
   :func:`all_zero_guard` rejects a finished cell that is identically zero while
   some exact distance is positive.

   **What is deliberately not a guard: a single ``0.0`` where ``exact > 0``.**
   A valid lower bound legitimately returns zero whenever two graphs share a
   node count and a degree sequence but are not isomorphic -- ``C6`` against two
   disjoint triangles has exact GED 4 and BRANCH, BRANCH_FAST, BRANCH_TIGHT and
   STAR all return 0. Measured on a seeded 600-pair LINUX sample this is 1.0 %
   of certified pairs, with zero validity violations. Raising on it would halt
   the harness on a correct, merely loose bound. The backstop for a misread
   accessor is validity (:func:`validity_refuted`): a zero read as an upper
   bound is refuted on essentially every pair with positive distance.

3. **Direction dependence.** Every GEDLIB upper bound builds its edit path from
   a directed assignment, so ``UB(i, j) != UB(j, i)`` in general. Upper-bound
   cells are therefore evaluated in both orientations and report
   ``min(fwd, rev)``, which is the value a production distance matrix carries.
   Lower bounds depend only on ``|deg(u) - deg(v)|`` and are run once.

Ground truth is ``networkx.graph_edit_distance`` under the D6 unit cost model,
never GEDLIB -- comparing GEDLIB against GEDLIB would measure nothing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

End: TypeAlias = Literal["lower", "upper"]

SCHEMA_VERSION = 1
WAVE = "2026-08-13-t27-bakeoff"

#: The D6 cost model, in GEDLIB's ``CONSTANT`` argument order:
#: ``[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]``.
COST_MODEL: tuple[float, ...] = (1.0, 1.0, 0.0, 1.0, 1.0, 0.0)

INIT_OPTION = "EAGER_WITHOUT_SHUFFLED_COPIES"

#: A constant string label. ``add_nx_graph`` rejects non-string attributes, and
#: a constant label is what makes the comparison topology-only.
DUMMY_LABEL = "1"

TOL = 1e-9

DATASETS: tuple[str, ...] = (
    "linux",
    "aids",
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
)

LEV_VARIANTS: tuple[str, ...] = ("exhaustive", "greedy", "greedy_single")


class BakeoffError(Exception):
    """Raised when a measurement cannot be trusted.

    Every raise in this module means a number would otherwise have been
    recorded that cannot be justified. Nothing here degrades to a warning.
    """


# --------------------------------------------------------------------------
# Method registry
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One GEDLIB method as this bake-off configures it.

    Attributes
    ----------
    name : str
        GEDLIB method name, verbatim and upper case.
    end : {'lower', 'upper'}
        Which accessor carries this method's bound. ``'lower'`` reads
        ``get_lower_bound()``; ``'upper'`` reads ``get_upper_bound()``. Never
        both -- reading the other end returns garbage without raising.
    default_options : str
        Option string used for the primary campaign and the defaults arm of the
        determinism probe.
    pinned_options : str
        Option string for the pinned arm of the determinism probe. Empty when
        the method exposes nothing to pin.
    randomised : bool
        Whether the method performs randomised local search, and is therefore
        expected to vary across repetitions at defaults.
    """

    name: str
    end: End
    default_options: str = ""
    pinned_options: str = ""
    randomised: bool = False


#: The eight bake-off methods plus ``HED``.
#:
#: ``HED`` carries ``--edge-set-distances OPTIMAL`` because its default,
#: ``HED``, scores incident-edge sets by a row/column-minimum sum that is
#: identically zero when edge substitution is free, as it is under D6. The
#: default therefore yields a valid but vacuous bound of 0.0 on every pair.
#: ``OPTIMAL`` replaces that with an optimal LSAPE and the bound becomes
#: non-degenerate. ``HED`` sets only ``result.set_lower_bound`` in
#: ``hed.ipp``, so ``get_upper_bound() = inf`` is by design, not a defect.
#:
#: The pinned option strings carry no ``--seed``: GEDLIB exposes no seed
#: option, and passing one raises ``Invalid option "seed"``. Determinism for the
#: local-search methods is pinned through single-threading, pseudo-randomness,
#: a single initial solution and no randpost loops.
_LS_PINNED = "--threads 1 --randomness PSEUDO --initial-solutions 1 --num-randpost-loops 0"

METHODS: dict[str, MethodSpec] = {
    "BRANCH": MethodSpec("BRANCH", "lower", "", "--threads 1"),
    "BRANCH_FAST": MethodSpec("BRANCH_FAST", "lower", "", "--threads 1"),
    "BRANCH_TIGHT": MethodSpec("BRANCH_TIGHT", "lower", "", "--threads 1"),
    "STAR": MethodSpec("STAR", "lower", "", "--threads 1"),
    "BIPARTITE": MethodSpec("BIPARTITE", "upper", "", "--threads 1"),
    "IPFP": MethodSpec("IPFP", "upper", "", _LS_PINNED, randomised=True),
    "REFINE": MethodSpec("REFINE", "upper", "", _LS_PINNED, randomised=True),
    "BP_BEAM": MethodSpec("BP_BEAM", "upper", "", _LS_PINNED, randomised=True),
    "HED": MethodSpec(
        "HED",
        "lower",
        "--edge-set-distances OPTIMAL",
        "--edge-set-distances OPTIMAL --threads 1",
    ),
}

#: The eight methods the frozen selection rule ranks. ``HED`` is a ninth cell,
#: reported but outside the four-versus-four comparison.
BAKEOFF_METHODS: tuple[str, ...] = (
    "BRANCH",
    "BRANCH_FAST",
    "BRANCH_TIGHT",
    "STAR",
    "BIPARTITE",
    "IPFP",
    "REFINE",
    "BP_BEAM",
)


# --------------------------------------------------------------------------
# GEDLIB loading -- import order is load-bearing
# --------------------------------------------------------------------------

_GEDLIB: Any = None


def load_gedlib() -> Any:
    """Import the GEDLIB bindings in the only order that works.

    Returns
    -------
    module
        ``gklearn.gedlib.gedlibpy_gxl``.

    Raises
    ------
    BakeoffError
        If the bindings cannot be imported, most often because the C++ side was
        never built or ``PYTHONPATH`` does not point at the in-place checkout.

    Notes
    -----
    ``libraries_import`` ``dlopen()``s the shared objects the bindings link
    against and must run first. Both imports go through
    :func:`importlib.import_module` because a formatter cannot reorder a
    function call, and a reordered plain import produces
    ``libdoublefann.so.2: cannot open shared object file``.
    """
    global _GEDLIB
    if _GEDLIB is None:
        try:
            importlib.import_module("gklearn.gedlib.libraries_import")
            _GEDLIB = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
        except ImportError as exc:
            raise BakeoffError(
                "cannot import the GEDLIB bindings; export PYTHONPATH to the "
                f"in-place graphkit-learn checkout and rerun build_ext ({exc})"
            ) from exc
    return _GEDLIB


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Corpus:
    """One dataset's graphs and the ground truth aligned to them.

    Attributes
    ----------
    dataset : str
        Dataset key.
    graphs : list of networkx.Graph
        Topology only, every node and edge carrying :data:`DUMMY_LABEL`.
    graph_ids : numpy.ndarray
        Graph identifiers, in the order the graphs are listed.
    node_counts, edge_counts : numpy.ndarray
        Per-graph counts, int32.
    exact : numpy.ndarray
        Ground-truth GED in canonical pair order, ``inf`` where censored.
    exact_lb, exact_ub : numpy.ndarray
        Solver bracket in canonical pair order. Equal to ``exact`` where
        certified.
    certified : numpy.ndarray
        Boolean mask in canonical pair order.
    pair_i, pair_j : numpy.ndarray
        Canonical pair order, int32.
    """

    dataset: str
    graphs: list[nx.Graph]
    graph_ids: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    exact: np.ndarray
    exact_lb: np.ndarray
    exact_ub: np.ndarray
    certified: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray

    @property
    def n_graphs(self) -> int:
        """Number of graphs."""
        return len(self.graphs)

    @property
    def n_pairs(self) -> int:
        """Number of upper-triangular pairs."""
        return int(self.pair_i.size)


def exported_path(data_root: Path, dataset: str) -> Path:
    """Return the path of a dataset's exported topology file."""
    return data_root / "exported" / f"{dataset}.npz"


def ground_truth_path(data_root: Path, dataset: str) -> Path:
    """Return the path of a dataset's exact-GED ground-truth file."""
    return (
        data_root
        / "source"
        / "GED_PRECOMPUTED"
        / "extended_merged_exact_ged"
        / "computed"
        / f"{dataset}.npz"
    )


def levenshtein_path(data_root: Path, dataset: str, variant: str) -> Path:
    """Return the path of a dataset's Levenshtein matrix for one encoder variant."""
    return data_root / "eval" / "levenshtein_matrices" / f"{dataset}_{variant}.npz"


def build_graphs(exported: Any) -> list[nx.Graph]:
    """Reconstruct networkx graphs from the CSR-style exported arrays.

    Parameters
    ----------
    exported : numpy.lib.npyio.NpzFile
        Loaded ``exported/{ds}.npz``, carrying ``n_nodes``, ``edge_offsets`` and
        ``edges`` of shape ``(2, sum n_edges)``.

    Returns
    -------
    list of networkx.Graph
        One graph per row, every node and edge carrying :data:`DUMMY_LABEL`.
        ``add_nx_graph`` rejects non-string attributes and a constant label is
        what makes the comparison topology-only.

    Raises
    ------
    BakeoffError
        If the reconstructed edge count disagrees with the recorded one, which
        would mean the CSR offsets and the edge array have drifted apart.
    """
    n_nodes = exported["n_nodes"]
    n_edges = exported["n_edges"]
    offsets = exported["edge_offsets"]
    edges = exported["edges"]

    graphs: list[nx.Graph] = []
    for k in range(int(n_nodes.size)):
        graph = nx.Graph()
        graph.add_nodes_from((i, {"label": DUMMY_LABEL}) for i in range(int(n_nodes[k])))
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        graph.add_edges_from(
            (int(edges[0, c]), int(edges[1, c]), {"label": DUMMY_LABEL}) for c in range(lo, hi)
        )
        if graph.number_of_edges() != int(n_edges[k]):
            raise BakeoffError(
                f"graph {k}: reconstructed {graph.number_of_edges()} edges but the export "
                f"records {int(n_edges[k])}; the CSR offsets and the edge array disagree"
            )
        graphs.append(graph)
    return graphs


def load_corpus(data_root: Path, dataset: str) -> Corpus:
    """Load one dataset's graphs and ground truth, asserting they are aligned.

    Parameters
    ----------
    data_root : pathlib.Path
        The ``data`` directory holding ``exported/`` and ``source/``.
    dataset : str
        Dataset key.

    Returns
    -------
    Corpus
        Graphs and ground truth in canonical pair order.

    Raises
    ------
    BakeoffError
        If ``graph_ids`` differ element-wise or in order between the export and
        the ground truth, or if the per-graph node or edge counts disagree.
        The orchestrator has verified this alignment already; it is re-asserted
        here because a silent misalignment would produce a complete, plausible
        and entirely wrong bake-off.
    """
    exported = np.load(exported_path(data_root, dataset), allow_pickle=True)
    truth = np.load(ground_truth_path(data_root, dataset), allow_pickle=True)

    exported_ids = exported["graph_ids"]
    truth_ids = truth["graph_ids"]
    assert_aligned(exported_ids, truth_ids, dataset=dataset, what="graph_ids")

    node_counts = np.asarray(exported["n_nodes"], dtype=np.int32)
    edge_counts = np.asarray(exported["n_edges"], dtype=np.int32)
    truth_nodes = np.asarray(truth["node_counts"], dtype=np.int32)
    truth_edges = np.asarray(truth["edge_counts"], dtype=np.int32)
    if not np.array_equal(node_counts, truth_nodes):
        raise BakeoffError(f"{dataset}: n_nodes != node_counts between export and ground truth")
    if not np.array_equal(edge_counts, truth_edges):
        raise BakeoffError(f"{dataset}: n_edges != edge_counts between export and ground truth")

    graphs = build_graphs(exported)
    for k, graph in enumerate(graphs):
        if graph.number_of_nodes() != int(node_counts[k]):
            raise BakeoffError(f"{dataset}: graph {k} has the wrong node count after rebuild")

    n = len(graphs)
    pair_i, pair_j = np.triu_indices(n, k=1)
    pair_i = pair_i.astype(np.int32)
    pair_j = pair_j.astype(np.int32)

    exact = np.asarray(truth["ged_matrix"], dtype=np.float64)[pair_i, pair_j]
    exact_lb = np.asarray(truth["lb_matrix"], dtype=np.float64)[pair_i, pair_j]
    exact_ub = np.asarray(truth["ub_matrix"], dtype=np.float64)[pair_i, pair_j]
    certified = np.asarray(truth["certified_mask"], dtype=bool)[pair_i, pair_j]

    if not np.all(np.isfinite(exact[certified])):
        raise BakeoffError(f"{dataset}: a certified pair carries a non-finite exact GED")
    if not np.all(exact_lb[certified] == exact[certified]):
        raise BakeoffError(f"{dataset}: certified exact_lb != exact")
    if not np.all(exact_ub[certified] == exact[certified]):
        raise BakeoffError(f"{dataset}: certified exact_ub != exact")
    if not np.all(np.isfinite(exact_lb)) or not np.all(np.isfinite(exact_ub)):
        raise BakeoffError(f"{dataset}: the solver bracket is not finite on every pair")

    return Corpus(
        dataset=dataset,
        graphs=graphs,
        graph_ids=exported_ids,
        node_counts=node_counts,
        edge_counts=edge_counts,
        exact=exact,
        exact_lb=exact_lb,
        exact_ub=exact_ub,
        certified=certified,
        pair_i=pair_i,
        pair_j=pair_j,
    )


def assert_aligned(left: np.ndarray, right: np.ndarray, *, dataset: str, what: str) -> None:
    """Assert two identifier arrays are element-wise identical and in one order.

    Parameters
    ----------
    left, right : numpy.ndarray
        Identifier arrays.
    dataset : str
        Dataset key, for the message.
    what : str
        What is being compared, for the message.

    Raises
    ------
    BakeoffError
        If the lengths differ or any element differs. Sorted-equal but
        differently ordered arrays are a failure, not a pass: every array in
        this wave indexes by position.
    """
    if left.shape != right.shape:
        raise BakeoffError(f"{dataset}: {what} shapes differ, {left.shape} against {right.shape}")
    if not np.array_equal(left, right):
        mismatch = int(np.argmax(left != right))
        same_set = sorted(left.tolist()) == sorted(right.tolist())
        detail = "same set in a different order" if same_set else "different sets"
        raise BakeoffError(
            f"{dataset}: {what} are not aligned ({detail}); first mismatch at index "
            f"{mismatch}: {left[mismatch]!r} against {right[mismatch]!r}"
        )


def load_levenshtein(data_root: Path, corpus: Corpus, variant: str) -> np.ndarray:
    """Load one Levenshtein matrix and flatten it into canonical pair order.

    Parameters
    ----------
    data_root : pathlib.Path
        The ``data`` directory.
    corpus : Corpus
        The dataset whose order the vector must follow.
    variant : str
        Encoder variant.

    Returns
    -------
    numpy.ndarray
        int32 vector of length ``P``.

    Raises
    ------
    BakeoffError
        If the matrix's ``graph_ids`` are not aligned with the corpus.
    """
    payload = np.load(levenshtein_path(data_root, corpus.dataset, variant), allow_pickle=True)
    assert_aligned(
        payload["graph_ids"],
        corpus.graph_ids,
        dataset=corpus.dataset,
        what=f"levenshtein[{variant}] graph_ids",
    )
    matrix = np.asarray(payload["levenshtein_matrix"])
    return matrix[corpus.pair_i, corpus.pair_j].astype(np.int32)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git_head(path: Path) -> str:
    """Return ``git rev-parse HEAD`` for a checkout, or ``'unknown'``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return out.stdout.strip()


def gedlib_checkout() -> Path:
    """Return the graphkit-learn checkout the bindings were imported from."""
    module = load_gedlib()
    return Path(module.__file__).resolve().parents[3]


def build_meta(
    *,
    dataset: str,
    n_graphs: int,
    n_pairs: int,
    method: str | None,
    end: str | None,
    options: str | None,
    deterministic: bool | None,
    wall_seconds: float,
    code_root: Path,
) -> str:
    """Build the ``meta`` JSON string shared by index and cell files.

    Returns
    -------
    str
        JSON, matching CONTRACTS section 4. Index files pass ``None`` for
        ``method``, ``end``, ``options`` and ``deterministic``.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "wave": WAVE,
        "dataset": dataset,
        "n_graphs": n_graphs,
        "n_pairs": n_pairs,
        "method": method,
        "end": end,
        "options": options,
        "deterministic": deterministic,
        "cost_model": list(COST_MODEL),
        "gedlib_commit": _git_head(gedlib_checkout()),
        "code_commit": _git_head(code_root),
        "host": platform.node(),
        "wall_seconds": round(wall_seconds, 3),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(payload, sort_keys=True)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


def write_index(
    data_root: Path, out_root: Path, dataset: str, corpus: Corpus | None = None
) -> Path:
    """Build and write ``$OUT/data/index/{ds}.npz``.

    Parameters
    ----------
    data_root : pathlib.Path
        The ``data`` directory.
    out_root : pathlib.Path
        The report directory. ``data/index/`` is created under it.
    dataset : str
        Dataset key.
    corpus : Corpus, optional
        A corpus already loaded, to avoid loading it twice.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    started = time.time()
    if corpus is None:
        corpus = load_corpus(data_root, dataset)

    lev = {v: load_levenshtein(data_root, corpus, v) for v in LEV_VARIANTS}
    n_max = np.maximum(corpus.node_counts[corpus.pair_i], corpus.node_counts[corpus.pair_j]).astype(
        np.int32
    )

    destination = out_root / "data" / "index" / f"{dataset}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        pair_i=corpus.pair_i,
        pair_j=corpus.pair_j,
        exact=corpus.exact,
        exact_lb=corpus.exact_lb,
        exact_ub=corpus.exact_ub,
        certified=corpus.certified,
        n_max=n_max,
        lev_exhaustive=lev["exhaustive"],
        lev_greedy=lev["greedy"],
        lev_greedy_single=lev["greedy_single"],
        graph_ids=corpus.graph_ids,
        node_counts=corpus.node_counts,
        edge_counts=corpus.edge_counts,
        meta=np.asarray(
            build_meta(
                dataset=dataset,
                n_graphs=corpus.n_graphs,
                n_pairs=corpus.n_pairs,
                method=None,
                end=None,
                options=None,
                deterministic=None,
                wall_seconds=time.time() - started,
                code_root=repo_root(),
            )
        ),
    )
    logger.info(
        "index %s: %d graphs, %d pairs -> %s", dataset, corpus.n_graphs, corpus.n_pairs, destination
    )
    return destination


# --------------------------------------------------------------------------
# Reading a bound -- every guard lives here
# --------------------------------------------------------------------------


def read_bound(env: Any, gid_a: int, gid_b: int, end: End, *, context: str) -> float:
    """Read one bound through the accessor that matches the method's end.

    Parameters
    ----------
    env : GEDEnvGXL
        An environment on which ``run_method`` has just been called for this
        pair.
    gid_a, gid_b : int
        GEDLIB graph ids, in the orientation ``run_method`` was called with.
    end : {'lower', 'upper'}
        Which accessor carries this method's bound.
    context : str
        Free text identifying the cell and pair, used in the error message.

    Returns
    -------
    float
        The bound.

    Raises
    ------
    BakeoffError
        If the value is ``NaN``, infinite or negative. ``inf`` is the signature
        of reading an upper bound off a method that only sets a lower one, and
        a negative value cannot arise from a non-negative cost model; both mean
        the number would be uninterpretable rather than merely loose.
    """
    if end == "lower":
        raw = env.get_lower_bound(gid_a, gid_b)
    elif end == "upper":
        raw = env.get_upper_bound(gid_a, gid_b)
    else:
        raise BakeoffError(f"{context}: unknown end {end!r}")

    value = float(raw)
    if math.isnan(value):
        raise BakeoffError(f"{context}: {end} bound is NaN")
    if math.isinf(value):
        raise BakeoffError(
            f"{context}: {end} bound is infinite; this method does not set that end "
            "and the accessor is wrong"
        )
    if value < 0.0:
        raise BakeoffError(f"{context}: {end} bound is negative ({value})")
    return value


#: Probe pairs with known, strictly positive exact distances under D6. Used to
#: prove an accessor is live before a cell's pair loop runs. Measured values on
#: the local build: every one of the nine methods returns a strictly positive
#: finite bound on both pairs.
def probe_pairs() -> tuple[tuple[str, nx.Graph, nx.Graph, float], ...]:
    """Return the capability-probe pairs and their exact distances.

    Returns
    -------
    tuple
        ``(name, g1, g2, exact_ged)`` per probe.

    Notes
    -----
    Every probe pair is **synthetic and fixed**, so the expectation is
    hard-coded and the probe does not depend on which dataset is being run.
    All three differ in degree sequence, which is what forbids a valid bound
    from returning zero: under D6 a zero can only arise from a degree-preserving
    assignment, and free node *and* edge substitution is precisely what makes
    such an assignment cost nothing.

    ``K1,4`` against ``P5`` is the primary probe -- five nodes and four edges on
    both sides, so no node operation is forced and the whole distance comes from
    the degree mismatch. ``P4`` against ``C4`` is the smoke pair recorded in
    ``gedlib.md`` section 5. ``K1`` against ``C5`` adds a pair whose distance is
    large enough that a rounding-scale bound cannot land on zero by accident.
    """
    return (
        ("K14_vs_P5", nx.star_graph(4), nx.path_graph(5), 4.0),
        ("P4_vs_C4", nx.path_graph(4), nx.cycle_graph(4), 1.0),
        ("K1_vs_C5", nx.empty_graph(1), nx.cycle_graph(5), 9.0),
    )


def label_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy of a graph with constant string node and edge labels.

    ``add_nx_graph`` rejects non-string attributes. A constant label is what
    makes the comparison topology-only, which is what the D6 cost model wants.
    """
    labelled = nx.Graph()
    labelled.add_nodes_from((n, {"label": DUMMY_LABEL}) for n in graph.nodes())
    labelled.add_edges_from((u, v, {"label": DUMMY_LABEL}) for u, v in graph.edges())
    return labelled


def capability_probe(spec: MethodSpec, options: str | None = None) -> dict[str, float]:
    """Prove a method's configured accessor is live before measuring with it.

    Runs the method on :func:`probe_pairs`, whose exact distances are known and
    strictly positive, and requires a strictly positive finite value from the
    accessor the spec names.

    Parameters
    ----------
    spec : MethodSpec
        The method.
    options : str, optional
        Option string. Defaults to ``spec.default_options``.

    Returns
    -------
    dict
        Probe name to value.

    Raises
    ------
    BakeoffError
        If any probe returns zero. That is the wrong-accessor signature: an
        upper-bound method read through ``get_lower_bound()`` returns ``0.00``
        without raising, and a whole matrix would fill with zeros. A *loose*
        bound cannot produce zero here, because both probe pairs differ in
        degree sequence and in node count respectively.
    """
    module = load_gedlib()
    opts = spec.default_options if options is None else options
    values: dict[str, float] = {}
    for name, g1, g2, exact in probe_pairs():
        env = module.GEDEnvGXL()
        a = env.add_nx_graph(label_graph(g1), "")
        b = env.add_nx_graph(label_graph(g2), "")
        env.set_edit_cost("CONSTANT", edit_cost_constant=list(COST_MODEL))
        env.init(init_option=INIT_OPTION)
        env.set_method(spec.name, opts)
        env.init_method()
        env.run_method(a, b)
        value = read_bound(env, a, b, spec.end, context=f"probe {name} / {spec.name}")
        if value <= 0.0:
            raise BakeoffError(
                f"capability probe {name} failed for {spec.name} [{opts!r}]: the "
                f"{spec.end} accessor returned {value} where the exact distance is "
                f"{exact}. This is the wrong-accessor signature, not a loose bound."
            )
        if spec.end == "lower" and value > exact + TOL:
            raise BakeoffError(
                f"capability probe {name}: {spec.name} lower bound {value} exceeds the "
                f"exact distance {exact}"
            )
        if spec.end == "upper" and value < exact - TOL:
            raise BakeoffError(
                f"capability probe {name}: {spec.name} upper bound {value} falls below "
                f"the exact distance {exact}"
            )
        values[name] = value
    return values


def all_zero_guard(values: np.ndarray, exact: np.ndarray, *, context: str) -> None:
    """Reject a finished cell that is identically zero against positive truth.

    Parameters
    ----------
    values : numpy.ndarray
        The cell's reported bounds.
    exact : numpy.ndarray
        Ground-truth distances, ``inf`` where censored.
    context : str
        Cell identifier for the message.

    Raises
    ------
    BakeoffError
        If every value is zero while some exact distance is positive. Single
        zeros are legitimate for a loose lower bound; an entire column of them
        is the signature of a misread accessor.
    """
    positive = np.isfinite(exact) & (exact > 0.0)
    if positive.any() and np.all(values == 0.0):
        raise BakeoffError(
            f"{context}: every value is 0.0 while {int(positive.sum())} pairs have a "
            "positive exact distance; the accessor is wrong"
        )


def validity_refuted(
    values: np.ndarray, exact_lb: np.ndarray, exact_ub: np.ndarray, end: End
) -> np.ndarray:
    """Return the M4 refutation mask over all pairs, certified and censored.

    A lower bound is refuted iff it exceeds the solver's upper bracket; an upper
    bound is refuted iff it falls below the solver's lower bracket. On a
    certified pair ``exact_lb == exact_ub == exact``, so the same expression is
    the two-sided test; on a censored pair it is the one-sided test that design
    section 3.5 buys for free.

    Parameters
    ----------
    values : numpy.ndarray
        The cell's reported bounds.
    exact_lb, exact_ub : numpy.ndarray
        Solver bracket, finite on every pair.
    end : {'lower', 'upper'}
        Which end the values are.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the bound is refuted.
    """
    if end == "lower":
        return values > exact_ub + TOL
    return values < exact_lb - TOL


# --------------------------------------------------------------------------
# Cell evaluation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Environment:
    """A GEDLIB environment holding one dataset's graphs.

    Built once and reconfigured per method, which is what makes a worker
    process amortise the ``init`` cost across a whole cell.
    """

    env: Any
    ids: list[int]
    dataset: str
    method: str | None = None
    options: str | None = None


def build_environment(corpus: Corpus) -> Environment:
    """Add a corpus's graphs to a fresh environment and initialise it.

    Parameters
    ----------
    corpus : Corpus
        The dataset.

    Returns
    -------
    Environment
        Initialised, with no method configured yet.
    """
    module = load_gedlib()
    env = module.GEDEnvGXL()
    ids = [env.add_nx_graph(graph, "") for graph in corpus.graphs]
    env.set_edit_cost("CONSTANT", edit_cost_constant=list(COST_MODEL))
    env.init(init_option=INIT_OPTION)
    return Environment(env=env, ids=ids, dataset=corpus.dataset)


def configure(environment: Environment, spec: MethodSpec, options: str) -> None:
    """Configure a method on an environment.

    Parameters
    ----------
    environment : Environment
        The environment.
    spec : MethodSpec
        The method.
    options : str
        Option string.

    Raises
    ------
    BakeoffError
        If GEDLIB rejects the option string. GEDLIB raises on an unknown
        option rather than dropping it, so a rejection is always reported.
    """
    try:
        environment.env.set_method(spec.name, options)
        environment.env.init_method()
    except Exception as exc:
        raise BakeoffError(f"{spec.name}: GEDLIB rejected options {options!r} -- {exc}") from exc
    environment.method = spec.name
    environment.options = options


def run_range(
    environment: Environment,
    spec: MethodSpec,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    lo: int,
    hi: int,
    both_orientations: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate a contiguous slice of the canonical pair order.

    Parameters
    ----------
    environment : Environment
        A configured environment.
    spec : MethodSpec
        The method.
    pair_i, pair_j : numpy.ndarray
        Canonical pair order.
    lo, hi : int
        Slice bounds, ``[lo, hi)``.
    both_orientations : bool
        Evaluate ``(j, i)`` as well. Upper bounds are direction-dependent.

    Returns
    -------
    tuple
        ``(forward, reverse)``; ``reverse`` is ``None`` when
        ``both_orientations`` is false.
    """
    env = environment.env
    ids = environment.ids
    end = spec.end
    n = hi - lo
    forward = np.empty(n, dtype=np.float64)
    reverse = np.empty(n, dtype=np.float64) if both_orientations else None

    for k in range(n):
        i = int(pair_i[lo + k])
        j = int(pair_j[lo + k])
        gi, gj = ids[i], ids[j]
        context = f"{environment.dataset}/{spec.name} pair ({i},{j})"
        env.run_method(gi, gj)
        forward[k] = read_bound(env, gi, gj, end, context=f"{context} fwd")
        if reverse is not None:
            env.run_method(gj, gi)
            reverse[k] = read_bound(env, gj, gi, end, context=f"{context} rev")
    return forward, reverse


_WORKER: dict[str, Any] = {}


def _worker_init(data_root: str, dataset: str, method: str, options: str) -> None:
    """Build one persistent environment per worker process.

    The environment survives across pair ranges, so ``init`` is paid once per
    worker rather than once per range.
    """
    corpus = load_corpus(Path(data_root), dataset)
    spec = METHODS[method]
    environment = build_environment(corpus)
    configure(environment, spec, options)
    _WORKER["corpus"] = corpus
    _WORKER["environment"] = environment
    _WORKER["spec"] = spec


def _worker_run(task: tuple[int, int, bool]) -> tuple[int, int, np.ndarray, np.ndarray | None]:
    """Evaluate one pair range inside a worker."""
    lo, hi, both = task
    corpus: Corpus = _WORKER["corpus"]
    forward, reverse = run_range(
        _WORKER["environment"],
        _WORKER["spec"],
        corpus.pair_i,
        corpus.pair_j,
        lo=lo,
        hi=hi,
        both_orientations=both,
    )
    return lo, hi, forward, reverse


def evaluate_cell(
    data_root: Path,
    out_root: Path,
    dataset: str,
    method: str,
    *,
    corpus: Corpus | None = None,
    jobs: int = 1,
    chunk: int = 20_000,
    options: str | None = None,
) -> dict[str, Any]:
    """Run one ``(dataset, method)`` cell and write its ``.npz``.

    Parameters
    ----------
    data_root : pathlib.Path
        The ``data`` directory.
    out_root : pathlib.Path
        The report directory.
    dataset : str
        Dataset key.
    method : str
        GEDLIB method name.
    corpus : Corpus, optional
        A corpus already loaded.
    jobs : int
        Worker processes. ``1`` runs in-process. **A timing measurement must
        never come from ``jobs > 1``** -- contended process time is not a rate.
    chunk : int
        Pairs per task in the parallel path.
    options : str, optional
        Option string. Defaults to the spec's.

    Returns
    -------
    dict
        Summary: value statistics, mean relative error and M4 violations.

    Raises
    ------
    BakeoffError
        Propagated from any guard. The caller writes ``.failed.json`` and no
        ``.npz``; a partial cell is never left on disk.
    """
    spec = METHODS[method]
    opts = spec.default_options if options is None else options
    started = time.time()

    if corpus is None:
        corpus = load_corpus(data_root, dataset)

    probes = capability_probe(spec, opts)
    both = spec.end == "upper"
    n_pairs = corpus.n_pairs

    if jobs <= 1:
        environment = build_environment(corpus)
        configure(environment, spec, opts)
        forward, reverse = run_range(
            environment,
            spec,
            corpus.pair_i,
            corpus.pair_j,
            lo=0,
            hi=n_pairs,
            both_orientations=both,
        )
    else:
        forward = np.empty(n_pairs, dtype=np.float64)
        reverse = np.empty(n_pairs, dtype=np.float64) if both else None
        tasks = [(lo, min(lo + chunk, n_pairs), both) for lo in range(0, n_pairs, chunk)]
        import multiprocessing as mp

        context = mp.get_context("fork")
        with context.Pool(
            processes=jobs,
            initializer=_worker_init,
            initargs=(str(data_root), dataset, method, opts),
        ) as pool:
            for lo, hi, part_f, part_r in pool.imap_unordered(_worker_run, tasks):
                forward[lo:hi] = part_f
                if reverse is not None and part_r is not None:
                    reverse[lo:hi] = part_r

    values = forward if reverse is None else np.minimum(forward, reverse)
    context = f"{dataset}/{method}"
    all_zero_guard(values, corpus.exact, context=context)

    refuted = validity_refuted(values, corpus.exact_lb, corpus.exact_ub, spec.end)
    n_violations = int(refuted.sum())

    wall = time.time() - started
    payload: dict[str, Any] = {
        "value": values,
        "value_fwd": forward,
        "meta": np.asarray(
            build_meta(
                dataset=dataset,
                n_graphs=corpus.n_graphs,
                n_pairs=n_pairs,
                method=method,
                end=spec.end,
                options=opts,
                deterministic=not spec.randomised,
                wall_seconds=wall,
                code_root=repo_root(),
            )
        ),
    }
    if reverse is not None:
        payload["value_rev"] = reverse

    destination = out_root / "data" / "cells" / f"{dataset}__{method}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)

    summary = summarise_cell(values, corpus, spec, refuted)
    summary.update(
        {
            "dataset": dataset,
            "method": method,
            "end": spec.end,
            "options": opts,
            "probes": probes,
            "wall_seconds": round(wall, 3),
            "pairs_per_second": round(n_pairs / wall, 1) if wall > 0 else None,
            "n_violations": n_violations,
            "path": str(destination),
        }
    )
    if n_violations:
        logger.error("%s: %d M4 violations", context, n_violations)
    logger.info(
        "%s: %d pairs in %.1f s, mean rel err %s, M4 violations %d",
        context,
        n_pairs,
        wall,
        summary["mean_relative_error"],
        n_violations,
    )
    return summary


def summarise_cell(
    values: np.ndarray, corpus: Corpus, spec: MethodSpec, refuted: np.ndarray
) -> dict[str, Any]:
    """Summarise a cell for the work log.

    Relative error is undefined where ``exact == 0``, so those pairs are
    excluded from the mean -- design section 3.1. The count of excluded pairs is
    reported beside it.
    """
    certified = corpus.certified
    exact = corpus.exact
    eligible = certified & np.isfinite(exact) & (exact > 0.0)
    rel = np.abs(values[eligible] - exact[eligible]) / exact[eligible]
    abs_err = np.abs(values[certified] - exact[certified])
    return {
        "n_pairs": int(values.size),
        "n_certified": int(certified.sum()),
        "n_m1_eligible": int(eligible.sum()),
        "mean_relative_error": round(float(rel.mean()), 6) if rel.size else None,
        "mean_absolute_error": round(float(abs_err.mean()), 6) if abs_err.size else None,
        "n_exact_hits": int(np.sum(np.abs(values[certified] - exact[certified]) <= TOL)),
        "n_zero_values": int(np.sum(values == 0.0)),
        "n_zero_with_positive_exact": int(
            np.sum((values == 0.0) & np.isfinite(exact) & (exact > 0.0))
        ),
        "n_refuted": int(refuted.sum()),
        "n_refuted_certified": int(np.sum(refuted & certified)),
        "n_refuted_censored": int(np.sum(refuted & ~certified)),
    }


def write_failure(
    out_root: Path, dataset: str, method: str, exc: BaseException, options: str
) -> Path:
    """Write ``{ds}__{METHOD}.failed.json`` and no ``.npz``.

    A failed cell is reported, never omitted and never left as a partial array.
    """
    destination = out_root / "data" / "cells" / f"{dataset}__{method}.failed.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "method": method,
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "options": options,
            },
            indent=2,
        )
    )
    logger.error("%s/%s failed: %s", dataset, method, exc)
    return destination


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_pairs(n_pairs: int, size: int, seed: int) -> np.ndarray:
    """Draw a seeded sample of canonical pair indices, sorted.

    Sorting keeps the sampled subset in canonical order, so a sampled array is
    still positionally comparable with the index file.
    """
    rng = np.random.default_rng(seed)
    if size >= n_pairs:
        return np.arange(n_pairs, dtype=np.int64)
    return np.sort(rng.choice(n_pairs, size=size, replace=False))


# --------------------------------------------------------------------------
# Determinism probe -- design section 3.11
# --------------------------------------------------------------------------


def determinism_arm(
    corpus: Corpus,
    spec: MethodSpec,
    options: str,
    *,
    sample: np.ndarray,
    repetitions: int,
) -> dict[str, Any]:
    """Run one arm of the determinism probe.

    Parameters
    ----------
    corpus : Corpus
        The dataset.
    spec : MethodSpec
        The method.
    options : str
        Option string for this arm.
    sample : numpy.ndarray
        Canonical pair indices to evaluate.
    repetitions : int
        Independent repetitions.

    Returns
    -------
    dict
        ``options``, ``frac_varying``, ``max_spread``, and ``rejected`` with the
        GEDLIB error when the option string is refused. An option GEDLIB rejects
        is recorded, never silently dropped.
    """
    try:
        environment = build_environment(corpus)
        configure(environment, spec, options)
    except BakeoffError as exc:
        return {
            "options": options,
            "frac_varying": None,
            "max_spread": None,
            "rejected": str(exc),
        }

    pair_i = corpus.pair_i[sample]
    pair_j = corpus.pair_j[sample]
    runs = np.empty((repetitions, sample.size), dtype=np.float64)
    for r in range(repetitions):
        # A fresh environment per repetition: an environment that caches a
        # result would make every repetition identical by construction and the
        # probe would report determinism it never measured.
        environment = build_environment(corpus)
        configure(environment, spec, options)
        forward, _ = run_range(
            environment, spec, pair_i, pair_j, lo=0, hi=sample.size, both_orientations=False
        )
        runs[r] = forward

    spread = runs.max(axis=0) - runs.min(axis=0)
    return {
        "options": options,
        "frac_varying": round(float(np.mean(spread > TOL)), 6),
        "max_spread": round(float(spread.max()), 6) if spread.size else 0.0,
        "rejected": None,
    }


def run_determinism(
    data_root: Path,
    out_root: Path,
    dataset: str,
    method: str,
    *,
    corpus: Corpus | None = None,
    n_pairs: int = 5_000,
    seed: int = 42,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Run the determinism probe for one cell and write its JSON.

    Returns
    -------
    dict
        The written payload, matching CONTRACTS section 6.
    """
    spec = METHODS[method]
    if corpus is None:
        corpus = load_corpus(data_root, dataset)
    sample = sample_pairs(corpus.n_pairs, n_pairs, seed)

    defaults = determinism_arm(
        corpus, spec, spec.default_options, sample=sample, repetitions=repetitions
    )
    pinned = determinism_arm(
        corpus, spec, spec.pinned_options, sample=sample, repetitions=repetitions
    )

    payload = {
        "dataset": dataset,
        "method": method,
        "end": spec.end,
        "n_pairs": int(sample.size),
        "seed": seed,
        "repetitions": repetitions,
        "defaults": defaults,
        "pinned": pinned,
        "deterministic_under_pinned": pinned["frac_varying"] == 0.0,
    }
    destination = out_root / "data" / "determinism" / f"{dataset}__{method}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    logger.info(
        "determinism %s/%s: defaults frac_varying=%s pinned frac_varying=%s",
        dataset,
        method,
        defaults["frac_varying"],
        pinned["frac_varying"],
    )
    return payload


# --------------------------------------------------------------------------
# Timing -- design section 3.4a, single process only
# --------------------------------------------------------------------------


def time_method(
    environment: Environment,
    spec: MethodSpec,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    """Time ``run_method`` per pair with :func:`time.process_time`.

    Only ``run_method`` is inside the clock: the accessors, the guards and the
    array writes are excluded, because the quantity the M7 gate names is the
    solver's own cost. One orientation, whatever the method's end -- a rate is
    per method evaluation, and section 3.6's ``min`` over two orientations is a
    property of the reported value rather than of the solver.

    Returns
    -------
    numpy.ndarray
        Seconds per pair.
    """
    env = environment.env
    ids = environment.ids
    out = np.empty(pair_i.size, dtype=np.float64)
    for k in range(pair_i.size):
        gi, gj = ids[int(pair_i[k])], ids[int(pair_j[k])]
        t0 = time.process_time()
        env.run_method(gi, gj)
        out[k] = time.process_time() - t0
    return out


def _timing_payload(
    seconds: np.ndarray, *, dataset: str, method: str, options: str, seed: int, n_bar: float
) -> dict[str, Any]:
    """Build the timing JSON payload from a per-pair seconds array."""
    micros = seconds * 1e6
    return {
        "dataset": dataset,
        "method": method,
        "options": options,
        "n_pairs_timed": int(micros.size),
        "seed": seed,
        "n_bar": round(float(n_bar), 3),
        "us_per_pair_mean": round(float(micros.mean()), 3),
        "us_per_pair_median": round(float(np.median(micros)), 3),
        "us_per_pair_p95": round(float(np.percentile(micros, 95)), 3),
        "clock": "time.process_time",
        "parallel": False,
    }


def run_timing(
    data_root: Path,
    out_root: Path,
    dataset: str,
    method: str,
    *,
    corpus: Corpus | None = None,
    n_pairs: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the serial timing pass for one cell and write its JSON.

    Raises
    ------
    BakeoffError
        From the capability probe, so that a timing is never recorded for a
        method whose accessor is misconfigured.
    """
    spec = METHODS[method]
    if corpus is None:
        corpus = load_corpus(data_root, dataset)
    capability_probe(spec)

    sample = sample_pairs(corpus.n_pairs, n_pairs, seed)
    pair_i, pair_j = corpus.pair_i[sample], corpus.pair_j[sample]
    environment = build_environment(corpus)
    configure(environment, spec, spec.default_options)
    seconds = time_method(environment, spec, pair_i, pair_j)

    n_bar = float(np.mean(np.concatenate([corpus.node_counts[pair_i], corpus.node_counts[pair_j]])))
    payload = _timing_payload(
        seconds,
        dataset=dataset,
        method=method,
        options=spec.default_options,
        seed=seed,
        n_bar=n_bar,
    )
    destination = out_root / "data" / "timing" / f"{dataset}__{method}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    logger.info(
        "timing %s/%s: %.1f us/pair mean at n_bar %.2f",
        dataset,
        method,
        payload["us_per_pair_mean"],
        n_bar,
    )
    return payload


# --------------------------------------------------------------------------
# The n_bar = 30 cost probe -- design section 3.4b
# --------------------------------------------------------------------------


def load_n30_graphs(
    iam_root: Path, *, lo: int = 25, hi: int = 35
) -> tuple[list[nx.Graph], list[str]]:
    """Load IAM GREC and Protein graphs within a node-count window.

    The frozen M7 gate is stated at ``n_bar = 30`` and no such pair exists in
    the ``n <= 12`` bake-off corpus, so the gate is unevaluable there. This
    probe supplies the missing regime from the tracked IAM loader.

    Parameters
    ----------
    iam_root : pathlib.Path
        ``.../IAM_Database/extracted``.
    lo, hi : int
        Inclusive node-count window.

    Returns
    -------
    tuple
        Labelled graphs and their identifiers.

    Raises
    ------
    BakeoffError
        If fewer than two graphs fall in the window, which would make the probe
        impossible rather than merely small. The gate is then reported
        unevaluated, never passed.
    """
    loader_dir = str(Path(__file__).resolve().parent)
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    # importlib rather than a plain import: a formatter would hoist the plain
    # form above the sys.path bootstrap, the same trap the GEDLIB bindings hit.
    loader = importlib.import_module("iam_gxl_loader")

    graphs: list[nx.Graph] = []
    ids: list[str] = []
    for key in ("grec", "protein"):
        dataset = loader.load_iam_gxl(str(iam_root), key)
        for graph, gid in zip(dataset.graphs, dataset.graph_ids, strict=True):
            if lo <= graph.number_of_nodes() <= hi:
                graphs.append(label_graph(graph))
                ids.append(f"{key}:{gid}")
    if len(graphs) < 2:
        raise BakeoffError(
            f"n=30 probe: only {len(graphs)} IAM graphs fall in [{lo}, {hi}]; the M7 "
            "gate is unevaluable and must be reported so, not passed"
        )
    logger.info("n=30 probe: %d GREC/Protein graphs in [%d, %d]", len(graphs), lo, hi)
    return graphs, ids


def run_n30_probe(
    iam_root: Path,
    out_root: Path,
    method: str,
    *,
    n_pairs: int = 2_000,
    seed: int = 42,
    graphs: list[nx.Graph] | None = None,
) -> dict[str, Any]:
    """Time one method at ``n_bar ~ 30`` and write ``probe_n30__{METHOD}.json``.

    Returns
    -------
    dict
        The written payload, matching CONTRACTS section 5 plus ``source`` and
        ``n_range``.
    """
    spec = METHODS[method]
    capability_probe(spec)
    if graphs is None:
        graphs, _ = load_n30_graphs(iam_root)

    module = load_gedlib()
    env = module.GEDEnvGXL()
    ids = [env.add_nx_graph(graph, "") for graph in graphs]
    env.set_edit_cost("CONSTANT", edit_cost_constant=list(COST_MODEL))
    env.init(init_option=INIT_OPTION)
    environment = Environment(env=env, ids=ids, dataset="iam_gxl_n30")
    configure(environment, spec, spec.default_options)

    n = len(graphs)
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2
    take = min(n_pairs, total)
    flat = np.sort(rng.choice(total, size=take, replace=False))
    tri_i, tri_j = np.triu_indices(n, k=1)
    pair_i, pair_j = tri_i[flat], tri_j[flat]

    seconds = time_method(environment, spec, pair_i, pair_j)
    sizes = np.array([g.number_of_nodes() for g in graphs], dtype=np.float64)
    n_bar = float(np.mean(np.concatenate([sizes[pair_i], sizes[pair_j]])))

    payload = _timing_payload(
        seconds,
        dataset="iam_gxl_n30",
        method=method,
        options=spec.default_options,
        seed=seed,
        n_bar=n_bar,
    )
    payload["source"] = "iam_gxl:GREC+Protein"
    payload["n_range"] = [25, 35]
    payload["n_graphs"] = n

    destination = out_root / "data" / "timing" / f"probe_n30__{method}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    logger.info(
        "n=30 probe %s: %.1f us/pair mean at n_bar %.2f",
        method,
        payload["us_per_pair_mean"],
        n_bar,
    )
    return payload


# --------------------------------------------------------------------------
# Cross-check against our own implementation -- design section 3.10
# --------------------------------------------------------------------------


def run_cross_check(
    data_root: Path,
    dataset: str = "linux",
    *,
    n_pairs: int = 400,
    seed: int = 42,
    corpus: Corpus | None = None,
) -> dict[str, Any]:
    """Compare GEDLIB ``BRANCH`` and ``BIPARTITE`` against ``ged_bounds.py``.

    ``ged_bounds.branch_lower_bound`` and ``ged_bounds.bipartite_upper_bound``
    are an independent implementation of the same two constructions. Agreement
    is what rules out a systematic misconfiguration of GEDLIB; disagreement is a
    defect in one of them and is reported, never smoothed over.

    ``bipartite_upper_bound`` is called with ``symmetrise=False`` because
    GEDLIB's ``BIPARTITE`` is evaluated in one orientation, and comparing a
    symmetrised value against a single-orientation one would compare two
    different quantities.

    Returns
    -------
    dict
        Per construction: pairs compared, agreements, and up to five
        disagreement examples.
    """
    bounds_dir = str(Path(__file__).resolve().parent)
    if bounds_dir not in sys.path:
        sys.path.insert(0, bounds_dir)
    ged_bounds = importlib.import_module("ged_bounds")

    if corpus is None:
        corpus = load_corpus(data_root, dataset)
    sample = sample_pairs(corpus.n_pairs, n_pairs, seed)

    module = load_gedlib()
    env = module.GEDEnvGXL()
    ids = [env.add_nx_graph(graph, "") for graph in corpus.graphs]
    env.set_edit_cost("CONSTANT", edit_cost_constant=list(COST_MODEL))
    env.init(init_option=INIT_OPTION)

    report: dict[str, Any] = {"dataset": dataset, "n_pairs": int(sample.size), "seed": seed}
    plans = (
        ("BRANCH", "lower", ged_bounds.branch_lower_bound, {}),
        ("BIPARTITE", "upper", ged_bounds.bipartite_upper_bound, {"symmetrise": False}),
    )
    for method, end, reference, kwargs in plans:
        spec = METHODS[method]
        env.set_method(method, spec.default_options)
        env.init_method()
        agree = 0
        examples: list[dict[str, Any]] = []
        for s in sample:
            i, j = int(corpus.pair_i[s]), int(corpus.pair_j[s])
            env.run_method(ids[i], ids[j])
            gedlib_value = read_bound(
                env, ids[i], ids[j], spec.end, context=f"cross-check {method} ({i},{j})"
            )
            ours = float(
                reference(corpus.graphs[i], corpus.graphs[j], ged_bounds.UNIT_COSTS, **kwargs)
            )
            if abs(gedlib_value - ours) <= 1e-6:
                agree += 1
            elif len(examples) < 5:
                examples.append({"i": i, "j": j, "gedlib": gedlib_value, "ours": ours})
        report[method] = {
            "end": end,
            "n_compared": int(sample.size),
            "n_agree": agree,
            "n_disagree": int(sample.size) - agree,
            "examples": examples,
        }
        logger.info("cross-check %s: %d/%d agree", method, agree, int(sample.size))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root, three levels above this file."""
    return Path(__file__).resolve().parents[3]


def _resolve_methods(raw: str | None) -> list[str]:
    """Resolve a comma-separated method list, defaulting to the eight."""
    if not raw or raw == "bakeoff":
        return list(BAKEOFF_METHODS)
    if raw == "all":
        return list(METHODS)
    names = [n.strip().upper() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        raise BakeoffError(f"unknown methods: {unknown}; known: {sorted(METHODS)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("index", "cells", "timing", "determinism", "probe-n30", "cross-check"),
    )
    parser.add_argument("--data", required=True, type=Path, help="the data directory")
    parser.add_argument("--out", required=True, type=Path, help="the report directory")
    parser.add_argument("--datasets", default="linux", help="comma-separated, or 'all'")
    parser.add_argument("--methods", default="bakeoff", help="comma-separated, 'bakeoff' or 'all'")
    parser.add_argument(
        "--jobs", type=int, default=1, help="worker processes; never used for timing"
    )
    parser.add_argument("--chunk", type=int, default=20_000, help="pairs per parallel task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="override the stage's sample size")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iam-root", type=Path, default=None, help="IAM_Database/extracted")
    parser.add_argument("--summary", type=Path, default=None, help="write a JSON run summary here")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _resolve_datasets(raw: str) -> list[str]:
    """Resolve a comma-separated dataset list, defaulting to LINUX."""
    if raw == "all":
        return list(DATASETS)
    names = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise BakeoffError(f"unknown datasets: {unknown}; known: {list(DATASETS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    """Run one stage from the command line.

    Returns
    -------
    int
        ``0`` when every cell succeeded, ``1`` when any failed. A failure
        writes ``.failed.json`` and the run continues, so one broken method
        cannot hide the rest.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.jobs > 4:
        raise BakeoffError(
            f"--jobs {args.jobs} exceeds the 4-process ceiling this workstation is shared under"
        )

    datasets = _resolve_datasets(args.datasets)
    methods = _resolve_methods(args.methods)
    results: list[dict[str, Any]] = []
    failed = False

    if args.stage == "probe-n30":
        iam_root = args.iam_root or (args.data / "source" / "IAM_Database" / "extracted")
        graphs, _ = load_n30_graphs(iam_root)
        for method in methods:
            results.append(
                run_n30_probe(
                    iam_root,
                    args.out,
                    method,
                    n_pairs=args.sample or 2_000,
                    seed=args.seed,
                    graphs=graphs,
                )
            )
    else:
        for dataset in datasets:
            corpus = load_corpus(args.data, dataset)
            if args.stage == "index":
                write_index(args.data, args.out, dataset, corpus=corpus)
                continue
            if args.stage == "cross-check":
                results.append(
                    run_cross_check(
                        args.data,
                        dataset,
                        n_pairs=args.sample or 400,
                        seed=args.seed,
                        corpus=corpus,
                    )
                )
                continue
            for method in methods:
                try:
                    if args.stage == "cells":
                        results.append(
                            evaluate_cell(
                                args.data,
                                args.out,
                                dataset,
                                method,
                                corpus=corpus,
                                jobs=args.jobs,
                                chunk=args.chunk,
                            )
                        )
                    elif args.stage == "timing":
                        results.append(
                            run_timing(
                                args.data,
                                args.out,
                                dataset,
                                method,
                                corpus=corpus,
                                n_pairs=args.sample or 2_000,
                                seed=args.seed,
                            )
                        )
                    elif args.stage == "determinism":
                        results.append(
                            run_determinism(
                                args.data,
                                args.out,
                                dataset,
                                method,
                                corpus=corpus,
                                n_pairs=args.sample or 5_000,
                                seed=args.seed,
                                repetitions=args.repetitions,
                            )
                        )
                except Exception as exc:  # noqa: BLE001 -- a failed cell is data
                    failed = True
                    write_failure(args.out, dataset, method, exc, METHODS[method].default_options)

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(results, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
