"""Chunked, resumable exact-GED compute driver (T-03, CONTRACT C).

Consumes a CONTRACT A ``.npz`` of graphs, computes graph edit distance for a
contiguous chunk of the linear upper-triangle index space (or for an explicit
sampled pair list), and writes a CONTRACT C shard.

The module is **backend-agnostic by construction**. It never imports
``ged_backends``; it types against the :class:`GedBackend` protocol of CONTRACTS §5
and resolves a concrete backend lazily inside :func:`make_backend`. GEDLIB exists
only on Picasso, so injecting a stub is the only way this driver can be tested at
all, and that constraint is designed for rather than worked around.

Three properties matter more than anything else here.

**Resume must be exact.** A shard is worth hundreds of core-hours. The checkpoint
stores explicit pair indices rather than a completion count, so it is correct
regardless of the order in which workers finish, and it carries a SHA-256 of the
chunk's target pair set so that resuming chunk 7's checkpoint into chunk 8 is a
hard error rather than a silently corrupt shard.

**The checkpoint is one file, overwritten in place.** fscratch limits file *count*,
not space. The write goes to a temporary file in the same directory and is moved
into place with :func:`os.replace`, which is atomic on POSIX, so a ``SIGKILL``
mid-write cannot leave a truncated checkpoint.

**Nothing is dropped.** A pair the solver cannot certify is interval-censored
(D11): ``ged = inf``, ``certified = False``, finite ``lb <= ub``. Per-pair wall
time is recorded for every pair because the D12 censoring analysis is reported per
stratum and needs it.

CLI, frozen in CONTRACTS §6::

    python -m benchmarks.real_data.eval_setup.ged_exact_runner \\
      --input <key>.npz --out <shards>/<key>_c0007.npz \\
      --backend {gedlib,networkx,stub} --cost-model {unit,graphedx} \\
      --chunk-index 7 --n-chunks 24 [--pair-list p.npz] [--seed-from prev.npz] \\
      --workers 64 --timeout-per-pair 300 --checkpoint-every 2000 \\
      --checkpoint <shards>/<key>_c0007.ckpt.npz

Exit codes: ``0`` success, ``1`` failure, ``143`` terminated by ``SIGTERM`` with the
checkpoint flushed and the chunk incomplete (the launcher should requeue).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import signal
import socket
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, runtime_checkable

import networkx as nx
import numpy as np

if __package__:
    from .ged_pair_index import (
        ChunkRange,
        GedPairIndexError,
        indices_of_pairs,
        n_pairs,
        pair_from_index,
        split_chunk,
        split_range,
    )
else:  # pragma: no cover - only when run as a bare script from eval_setup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ged_pair_index import (  # noqa: E402
        ChunkRange,
        GedPairIndexError,
        indices_of_pairs,
        n_pairs,
        pair_from_index,
        split_chunk,
        split_range,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "EXIT_FAILURE",
    "EXIT_OK",
    "EXIT_TERMINATED",
    "ContractA",
    "GedBackend",
    "PairOutcome",
    "RunnerError",
    "ShardData",
    "load_contract_a",
    "main",
    "make_backend",
    "reference_stub_backend",
    "run_chunk",
]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_TERMINATED = 143

SHARD_KEYS = ("pair_index", "ged", "lb", "ub", "certified", "seconds")


class RunnerError(Exception):
    """Raised on an inadmissible input, a corrupt checkpoint or a backend failure."""


# --------------------------------------------------------------------------- #
# CONTRACT B, consumed structurally -- never imported
# --------------------------------------------------------------------------- #


@runtime_checkable
class PairResultLike(Protocol):
    """The result object CONTRACT B backends return.

    Declared structurally so that this module has no import-time dependency on
    ``ged_backends``, which lives on a peer's branch and on the cluster.
    """

    lb: float
    ub: float
    exact: float | None
    certified: bool
    seconds: float
    timed_out: bool
    method: str


@runtime_checkable
class GedBackend(Protocol):
    """CONTRACT B §5. Anything with these two members is a usable backend."""

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResultLike:
        """Return the GED bracket for one pair of graphs."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend identity, recorded in the shard metadata."""
        ...


# --------------------------------------------------------------------------- #
# CONTRACT A, read directly from the file format
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ContractA:
    """A loaded CONTRACT A dataset.

    Attributes:
        key: Dataset key, from ``metadata['dataset']`` or the file stem.
        graphs: Undirected, unattributed graphs relabelled ``0..n-1``.
        graph_ids: Stable id per graph.
        labels: Class label per graph, ``""`` where the source has none.
        n_nodes: ``int32 (N,)`` node count per graph.
        n_edges: ``int32 (N,)`` edge count per graph.
        metadata: The parsed ``metadata`` JSON.
    """

    key: str
    graphs: list[nx.Graph]
    graph_ids: list[str]
    labels: list[str]
    n_nodes: np.ndarray
    n_edges: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        """Number of graphs in the cohort."""
        return len(self.graphs)


def load_contract_a(path: str | os.PathLike[str]) -> ContractA:
    """Read a CONTRACT A ``.npz`` into graphs.

    Deliberately reads the *format* described in CONTRACTS §4 rather than calling
    ``export_graphs.load_exported``: that module is owned by a peer and is not on
    this branch. Swapping this for ``load_exported`` once it merges is a one-line
    change and is listed as a follow-up.

    Args:
        path: Path to the ``.npz``.

    Returns:
        The loaded :class:`ContractA`.

    Raises:
        RunnerError: If a required key is missing or the CSR offsets are
            inconsistent with the declared edge counts.
    """
    required = ("graph_ids", "n_nodes", "n_edges", "edge_offsets", "edges")
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in required if k not in data]
        if missing:
            raise RunnerError(f"{path} is not a CONTRACT A file: missing {missing}")
        graph_ids = [str(x) for x in data["graph_ids"]]
        labels = [str(x) for x in data["labels"]] if "labels" in data else [""] * len(graph_ids)
        n_nodes = np.asarray(data["n_nodes"], dtype=np.int32)
        n_edges = np.asarray(data["n_edges"], dtype=np.int32)
        offsets = np.asarray(data["edge_offsets"], dtype=np.int64)
        edges = np.asarray(data["edges"], dtype=np.int32)
        meta: dict[str, Any] = {}
        if "metadata" in data:
            try:
                meta = dict(json.loads(str(data["metadata"])))
            except (ValueError, TypeError):
                meta = {}

    n = len(graph_ids)
    if offsets.shape != (n + 1,):
        raise RunnerError(f"edge_offsets must have shape ({n + 1},), got {offsets.shape}")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise RunnerError(f"edges must have shape (2, M), got {edges.shape}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(edges.shape[1]):
        raise RunnerError(
            f"edge_offsets must run 0..M; got {int(offsets[0])}..{int(offsets[-1])} "
            f"against M={int(edges.shape[1])}"
        )
    if not np.array_equal(np.diff(offsets), n_edges.astype(np.int64)):
        raise RunnerError("edge_offsets deltas disagree with n_edges")

    graphs: list[nx.Graph] = []
    for g_idx in range(n):
        g = nx.Graph()
        g.add_nodes_from(range(int(n_nodes[g_idx])))
        lo, hi = int(offsets[g_idx]), int(offsets[g_idx + 1])
        if hi > lo:
            g.add_edges_from(zip(edges[0, lo:hi].tolist(), edges[1, lo:hi].tolist()))
        graphs.append(g)

    return ContractA(
        key=str(meta.get("dataset", "") or Path(str(path)).stem),
        graphs=graphs,
        graph_ids=graph_ids,
        labels=labels,
        n_nodes=n_nodes,
        n_edges=n_edges,
        metadata=meta,
    )


# --------------------------------------------------------------------------- #
# Backend resolution -- lazy, explicit, never silently degrading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Everything a worker process needs to build its own backend.

    Kept to plain scalars so it pickles into a worker without dragging a solver
    handle across the process boundary.

    Attributes:
        name: One of ``gedlib``, ``networkx``, ``stub``.
        cost_model: ``unit`` or ``graphedx``.
        timeout_s: Per-pair solver budget in seconds.
        factory_ref: Optional ``module:callable`` override, used by the smoke and
            by any harness that has to run without ``ged_backends`` present.
        options: Extra keyword arguments forwarded verbatim to the backend
            constructor. Empty by default, which reproduces T-03's construction
            exactly. This is how the per-role method and options strings of
            CONTRACTS §3 reach the backend without a new positional argument on
            every function between here and it.
        probe_accessors: Run the backend's accessor probe once per worker before
            any pair. Defaults to ``True``: a wrong accessor returns ``0.00``
            from GEDLIB without raising, and the alternative to probing is
            discovering it from a finished matrix of zeros.
    """

    name: str
    cost_model: str = "unit"
    timeout_s: float = 300.0
    factory_ref: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    probe_accessors: bool = True


def _edit_costs(cost_model: str) -> Any:
    """Return the ``EditCosts`` object for a cost model name.

    Imports ``ged_bounds`` lazily and reuses its constants rather than restating
    the numbers, so the two implementations of the cost model cannot drift
    (CONTRACTS §1).

    Args:
        cost_model: ``unit`` or ``graphedx``.

    Returns:
        The corresponding ``EditCosts`` instance.

    Raises:
        RunnerError: If the name is unknown or ``ged_bounds`` cannot be imported.
    """
    try:
        if __package__:
            mod = importlib.import_module(f"{__package__}.ged_bounds")
        else:  # pragma: no cover - bare-script path
            mod = importlib.import_module("ged_bounds")
    except ImportError as exc:  # pragma: no cover - ged_bounds is committed
        raise RunnerError(f"cannot import ged_bounds for the cost model: {exc}") from exc
    if cost_model == "unit":
        return mod.UNIT_COSTS
    if cost_model == "graphedx":
        return mod.GRAPHEDX_COSTS
    raise RunnerError(f"unknown cost model {cost_model!r}; expected 'unit' or 'graphedx'")


def _resolve_factory_ref(ref: str) -> Any:
    """Import and return the callable named by a ``module:callable`` reference.

    Args:
        ref: e.g. ``benchmarks.real_data.eval_setup.ged_exact_runner:reference_stub_backend``.

    Returns:
        The resolved callable.

    Raises:
        RunnerError: If the reference is malformed or does not resolve.
    """
    if ":" not in ref:
        raise RunnerError(f"backend factory reference must be 'module:callable', got {ref!r}")
    mod_name, _, attr = ref.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise RunnerError(f"cannot import backend factory module {mod_name!r}: {exc}") from exc
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise RunnerError(f"{mod_name!r} has no callable {attr!r}")
    return fn


def make_backend(spec: BackendSpec) -> GedBackend:
    """Build a backend from a spec, resolving ``ged_backends`` lazily.

    Never falls back silently. If GEDLIB is unavailable the run stops with an
    explanatory error rather than quietly computing something else -- a whole GED
    matrix filled by the wrong solver is exactly the failure mode this programme
    is trying to eliminate.

    Args:
        spec: The backend specification.

    Returns:
        A concrete backend satisfying :class:`GedBackend`.

    Raises:
        RunnerError: If the backend name is unknown, or ``ged_backends`` (owned by
            ``task-gedlib-gates``) is not importable.
    """
    if spec.factory_ref:
        return _resolve_factory_ref(spec.factory_ref)(spec)  # type: ignore[no-any-return]

    costs = _edit_costs(spec.cost_model)
    mod = None
    for candidate in (f"{__package__}.ged_backends" if __package__ else "", "ged_backends"):
        if not candidate:
            continue
        try:
            mod = importlib.import_module(candidate)
            break
        except ImportError:
            continue
    if mod is None:
        raise RunnerError(
            "ged_backends is not importable, so no production backend can be built. "
            "It is owned by task-gedlib-gates and installed alongside GEDLIB on Picasso. "
            "For a solver-free harness pass --backend-factory module:callable."
        )

    classes = {"gedlib": "GedlibBackend", "networkx": "NetworkxBackend", "stub": "StubBackend"}
    if spec.name not in classes:
        raise RunnerError(f"unknown backend {spec.name!r}; expected one of {sorted(classes)}")
    cls = getattr(mod, classes[spec.name], None)
    if cls is None:
        raise RunnerError(f"ged_backends has no {classes[spec.name]}")
    try:
        return cls(costs, timeout_s=spec.timeout_s, **dict(spec.options))  # type: ignore[no-any-return]
    except TypeError as exc:
        raise RunnerError(
            f"{classes[spec.name]} rejected the backend options {dict(spec.options)!r}: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# A deterministic test double, reachable only by explicit --backend-factory
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _StubPairResult:
    """A CONTRACT B shaped result produced without a solver."""

    lb: float
    ub: float
    exact: float | None
    certified: bool
    seconds: float
    timed_out: bool
    method: str


class _ReferenceStubBackend:
    """Deterministic, solver-free backend for smoke tests and unit tests.

    **Not a GED solver.** It returns a genuine but very loose bracket -- ``lb`` is
    the unavoidable node/edge count difference and ``ub`` is the cost of deleting
    every edge of one graph and inserting every edge of the other, both valid under
    the unit cost model -- and then certifies a deterministic subset of pairs by
    collapsing the bracket. The values are arithmetic, not distances, and must
    never appear in a reported table. It exists because the real backend lives only
    on the cluster, so it is the only way the driver's resume and merge logic can be
    exercised at all.

    Determinism is the point: the same pair yields the same ``lb``, ``ub``,
    ``certified`` **and** ``seconds`` in every process and every run, which is what
    makes "the resumed shard is byte-identical to the uninterrupted one" a
    checkable claim rather than an approximate one.
    """

    __slots__ = ("_timeout_s",)

    def __init__(self, timeout_s: float = 300.0) -> None:
        """Store the nominal timeout, which the stub never actually reaches."""
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        """Backend identity, written into the shard metadata."""
        return "reference_stub"

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> _StubPairResult:
        """Return a deterministic bracket for one pair.

        Args:
            g1: First graph.
            g2: Second graph.

        Returns:
            A :class:`_StubPairResult` with a valid ``lb <= ub`` bracket.
        """
        n1, m1 = g1.number_of_nodes(), g1.number_of_edges()
        n2, m2 = g2.number_of_nodes(), g2.number_of_edges()
        lb = float(abs(n1 - n2) + abs(m1 - m2))
        ub = float(abs(n1 - n2) + m1 + m2)
        d1 = sorted(d for _, d in g1.degree())
        d2 = sorted(d for _, d in g2.degree())
        payload = f"{n1},{m1},{d1}|{n2},{m2},{d2}".encode()
        h = zlib.crc32(payload)
        seconds = 1e-4 * (1 + h % 97)
        if h % 3 == 0:
            return _StubPairResult(lb, ub, None, False, seconds, False, "reference_stub/bracket")
        span = int(ub - lb)
        exact = lb + float((h // 3) % (span + 1))
        if exact <= 0.0 < ub:
            exact = 1.0
        return _StubPairResult(
            exact, exact, exact, True, seconds, False, "reference_stub/certified"
        )


def reference_stub_backend(spec: BackendSpec) -> GedBackend:
    """Factory for :class:`_ReferenceStubBackend`, for ``--backend-factory``.

    Args:
        spec: The backend spec; only ``timeout_s`` is used.

    Returns:
        A fresh stub backend.
    """
    return _ReferenceStubBackend(timeout_s=spec.timeout_s)


# --------------------------------------------------------------------------- #
# Per-pair outcome and accumulation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PairOutcome:
    """One row of a shard.

    Attributes:
        k: Linear upper-triangle index.
        ged: Certified optimum, or ``inf`` when the pair is interval-censored.
        lb: Certified lower bound, finite.
        ub: Certified upper bound, finite.
        certified: Whether ``ged`` is a proven optimum.
        seconds: Per-pair wall time, required by the D12 censoring analysis.
        failed: Whether the backend raised on this pair.
    """

    k: int
    ged: float
    lb: float
    ub: float
    certified: bool
    seconds: float
    failed: bool = False


def _outcome_from_result(
    k: int, res: Any, measured_s: float, computed: str | None = None
) -> PairOutcome:
    """Convert a CONTRACT B ``PairResult`` into a shard row, enforcing D11.

    An uncertified value is never promoted to ``ged``: that is precisely the defect
    the design note found in ``ged_computer.py::compute_ged_pair``, where
    ``nx.graph_edit_distance``'s best-so-far cost was recorded as if exact.

    Args:
        k: Linear pair index.
        res: The backend's result object.
        measured_s: Wall time the runner itself observed, used only if the backend
            reports an unusable value.
        computed: The compute mode the *campaign* declares. When given it is the
            authority and the result's own encoding is cross-checked against it,
            so a backend that quietly returned a one-sided bracket in a two-sided
            run is caught rather than merged. Defaults to trusting the result.

    Returns:
        The :class:`PairOutcome`.

    Raises:
        RunnerError: If the bracket is inconsistent, or a pair claims certification
            without an exact value.
    """
    lb = float(res.lb)
    ub = float(res.ub)
    # A one-sided campaign (CONTRACTS §6) leaves the end it did not compute at
    # its vacuous sentinel: +inf for an absent upper bound, -inf for an absent
    # lower one. Only the end that was actually evaluated is required to be a
    # number; the sentinel must be exactly the sentinel, so that an unevaluated
    # end can never be mistaken for a measurement.
    declared = str(computed if computed is not None else getattr(res, "computed", "both"))
    if declared not in ("both", "lb", "ub"):
        raise RunnerError(f"pair {k}: unknown compute mode {declared!r}")
    observed = str(getattr(res, "computed", declared))
    if observed != declared:
        raise RunnerError(
            f"pair {k}: the campaign computes {declared!r} but the result encodes {observed!r}; "
            "the bracket does not match the run it came from"
        )
    computed = declared
    if computed != "ub" and not np.isfinite(lb):
        raise RunnerError(f"pair {k}: backend returned a non-finite lower bound {lb}")
    if computed == "ub" and lb != -np.inf:
        raise RunnerError(f"pair {k}: compute='ub' must leave lb at -inf, got {lb}")
    if computed != "lb" and not np.isfinite(ub):
        raise RunnerError(f"pair {k}: backend returned a non-finite upper bound {ub}")
    if computed == "lb" and ub != np.inf:
        raise RunnerError(f"pair {k}: compute='lb' must leave ub at +inf, got {ub}")
    if lb > ub + 1e-9:
        raise RunnerError(f"pair {k}: backend returned lb {lb} > ub {ub}")
    seconds = float(getattr(res, "seconds", 0.0))
    if not np.isfinite(seconds) or seconds < 0.0:
        seconds = measured_s

    if bool(res.certified):
        exact = res.exact
        if exact is None:
            raise RunnerError(f"pair {k}: certified=True but exact is None")
        val = float(exact)
        if not np.isfinite(val):
            raise RunnerError(f"pair {k}: certified value is not finite")
        if val < lb - 1e-9 or val > ub + 1e-9:
            raise RunnerError(f"pair {k}: certified {val} outside its own bracket [{lb}, {ub}]")
        return PairOutcome(k, val, lb, ub, True, seconds)
    # D11: interval-censored, never dropped.
    return PairOutcome(k, float("inf"), lb, ub, False, seconds)


@dataclass(slots=True)
class ShardData:
    """Accumulated shard rows, keyed by pair index.

    Attributes:
        rows: Mapping from linear pair index to its outcome.
    """

    rows: dict[int, PairOutcome] = field(default_factory=dict)

    def add(self, outcome: PairOutcome) -> None:
        """Insert or overwrite one row."""
        self.rows[outcome.k] = outcome

    def __len__(self) -> int:
        """Number of rows accumulated."""
        return len(self.rows)

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the shard arrays, sorted ascending by pair index.

        Returns:
            A dict with the six CONTRACT C data keys.
        """
        keys = sorted(self.rows)
        return {
            "pair_index": np.asarray(keys, dtype=np.int64),
            "ged": np.asarray([self.rows[k].ged for k in keys], dtype=np.float64),
            "lb": np.asarray([self.rows[k].lb for k in keys], dtype=np.float64),
            "ub": np.asarray([self.rows[k].ub for k in keys], dtype=np.float64),
            "certified": np.asarray([self.rows[k].certified for k in keys], dtype=np.bool_),
            "seconds": np.asarray([self.rows[k].seconds for k in keys], dtype=np.float32),
        }

    @property
    def n_failed(self) -> int:
        """Number of pairs on which the backend raised."""
        return sum(1 for r in self.rows.values() if r.failed)


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray | np.generic]) -> None:
    """Write an ``.npz`` atomically, overwriting any previous version in place.

    ``os.replace`` is atomic within a filesystem, so a kill between the write and
    the rename leaves the previous checkpoint intact rather than a truncated one.
    Exactly one durable file exists at any time, which is what the fscratch
    file-count quota requires.

    Args:
        path: Destination path.
        payload: Arrays to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_shard_like(path: Path) -> dict[int, PairOutcome]:
    """Read any CONTRACT C shaped ``.npz`` into rows.

    Args:
        path: Path to a shard or checkpoint.

    Returns:
        Mapping from pair index to outcome.

    Raises:
        RunnerError: If a required key is missing or the arrays disagree in length.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in SHARD_KEYS if k not in data]
        if missing:
            raise RunnerError(f"{path} is not a CONTRACT C file: missing {missing}")
        pi = np.asarray(data["pair_index"], dtype=np.int64)
        ged = np.asarray(data["ged"], dtype=np.float64)
        lb = np.asarray(data["lb"], dtype=np.float64)
        ub = np.asarray(data["ub"], dtype=np.float64)
        cert = np.asarray(data["certified"], dtype=np.bool_)
        sec = np.asarray(data["seconds"], dtype=np.float32)
    sizes = {pi.size, ged.size, lb.size, ub.size, cert.size, sec.size}
    if len(sizes) != 1:
        raise RunnerError(f"{path}: shard arrays disagree in length: {sorted(sizes)}")
    return {
        int(pi[t]): PairOutcome(
            int(pi[t]), float(ged[t]), float(lb[t]), float(ub[t]), bool(cert[t]), float(sec[t])
        )
        for t in range(pi.size)
    }


def _pairs_fingerprint(pairs: np.ndarray) -> str:
    """Return a SHA-256 over the target pair set, used as the checkpoint identity.

    Args:
        pairs: ``int64`` array of pair indices.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(pairs, dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Worker-process state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    """Picklable description of what a worker must construct for itself."""

    input_path: str
    backend: BackendSpec


_WORKER_GRAPHS: list[nx.Graph] | None = None
_WORKER_BACKEND: GedBackend | None = None
_WORKER_N: int = 0


def _probe_backend(backend: Any, spec: BackendSpec) -> None:
    """Verify the backend's accessors before it computes a single pair.

    Campaign init, once per worker process. A backend that exposes no
    ``probe_accessors`` is left alone, so the stub and the networkx backend are
    unaffected.

    Args:
        backend: The freshly constructed backend.
        spec: The spec it was built from.

    Raises:
        RunnerError: If the probe fails. This is deliberately fatal and
            deliberately *not* caught per pair: reading a bound through the
            wrong accessor returns ``0.00`` from GEDLIB without an exception, so
            a campaign that survives a failed probe writes a matrix of zeros
            that no downstream assertion is looking for.
    """
    if not spec.probe_accessors:
        return
    probe = getattr(backend, "probe_accessors", None)
    if probe is None:
        return
    try:
        values = probe()
    except Exception as exc:
        raise RunnerError(f"accessor probe failed for backend {spec.name!r}: {exc}") from exc
    logger.info("accessor probe passed: %s", values)


def _init_worker(spec: _WorkerSpec) -> None:
    """Build one backend and one copy of the graphs per worker process.

    GEDLIB environment construction is expensive and must not happen per pair
    (CONTRACT B), so it happens exactly once here. The worker also ignores
    ``SIGTERM``: SLURM signals the whole job step, and letting each child die
    independently would lose in-flight pairs the parent could otherwise have
    checkpointed.

    Args:
        spec: The worker specification.
    """
    global _WORKER_GRAPHS, _WORKER_BACKEND, _WORKER_N
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    dataset = load_contract_a(spec.input_path)
    _WORKER_GRAPHS = dataset.graphs
    _WORKER_N = dataset.n_graphs
    _WORKER_BACKEND = make_backend(spec.backend)
    _probe_backend(_WORKER_BACKEND, spec.backend)


def _compute_batch(batch: tuple[int, ...]) -> list[PairOutcome]:
    """Compute a batch of pairs inside a worker process.

    Args:
        batch: Linear pair indices.

    Returns:
        One :class:`PairOutcome` per index, in the same order.

    Raises:
        RunnerError: If the worker was not initialised.
    """
    if _WORKER_BACKEND is None or _WORKER_GRAPHS is None:
        raise RunnerError("worker process was not initialised")
    return [_compute_one(k, _WORKER_GRAPHS, _WORKER_BACKEND, _WORKER_N) for k in batch]


def _compute_one(k: int, graphs: list[nx.Graph], backend: GedBackend, n: int) -> PairOutcome:
    """Compute one pair, converting any backend exception into a censored row.

    A raising backend is a failure, not a missing value: the row is written with a
    degenerate ``[0, inf]`` bracket and flagged, the runner exits non-zero, and
    gate 4 at merge rejects it. The pair is still present, because D11 forbids
    dropping pairs and a silently absent pair would leave a hole no assertion
    downstream is looking for.

    Args:
        k: Linear pair index.
        graphs: The worker's graph list.
        backend: The worker's backend.
        n: Number of graphs.

    Returns:
        The outcome for this pair.
    """
    i, j = pair_from_index(k, n)
    t0 = time.perf_counter()
    try:
        res = backend.pair(graphs[i], graphs[j])
        return _outcome_from_result(
            k, res, time.perf_counter() - t0, getattr(backend, "compute", None)
        )
    except Exception:  # noqa: BLE001 - one bad pair must not kill a 100-hour shard
        logger.exception("backend failed on pair k=%d (i=%d, j=%d)", k, i, j)
        return PairOutcome(
            k, float("inf"), 0.0, float("inf"), False, time.perf_counter() - t0, True
        )


# --------------------------------------------------------------------------- #
# The chunk driver
# --------------------------------------------------------------------------- #


_TERMINATE = False


def _install_term_handler() -> None:
    """Trap ``SIGTERM``/``SIGINT`` so the checkpoint is flushed before exit.

    SLURM sends ``SIGTERM`` and waits 30 s before ``SIGKILL``. Without this, a
    wallclock kill discards everything since the last checkpoint.
    """

    def _handler(signum: int, _frame: FrameType | None) -> None:
        global _TERMINATE
        _TERMINATE = True
        logger.warning("received signal %d; will flush the checkpoint and stop", signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except ValueError:  # pragma: no cover - not the main thread
            logger.debug("cannot install a handler for signal %d here", sig)


def _target_pairs(
    dataset: ContractA,
    *,
    pair_list: str | None,
    chunk_index: int,
    n_chunks: int,
) -> tuple[np.ndarray, ChunkRange]:
    """Determine which pair indices this task owns.

    With ``--pair-list`` the *sampled list* is split; otherwise the full upper
    triangle is. Either way the split is even with the remainder on the low
    chunks, never a ragged tail.

    Args:
        dataset: The loaded cohort.
        pair_list: Optional path to a ``.npz`` holding either a ``pair_index``
            array of linear indices, or the ``dataset_key``/``pair_i``/``pair_j``
            columns the §5 subsample sampler writes. ``pair_index`` wins where
            both are present.
        chunk_index: This task's index.
        n_chunks: Total task count.

    Returns:
        ``(pairs, chunk_range)``, pairs ascending and deduplicated.

    Raises:
        RunnerError: If the pair list is malformed or holds an out-of-range index.
    """
    n = dataset.n_graphs
    total = n_pairs(n)
    if pair_list is None:
        rng = split_chunk(n, n_chunks, chunk_index)
        return np.arange(rng.start, rng.stop, dtype=np.int64), rng

    with np.load(pair_list, allow_pickle=False) as data:
        if "pair_index" in data:
            listed = np.unique(np.asarray(data["pair_index"], dtype=np.int64))
        elif "pair_i" in data and "pair_j" in data:
            # The §5 subsample list is pooled across all ten datasets and keyed
            # by (dataset_key, pair_i, pair_j) rather than by a linear index,
            # because a linear index is only meaningful against one cohort's
            # graph count. Select this dataset's rows, then map forward.
            pi = np.asarray(data["pair_i"], dtype=np.int64)
            pj = np.asarray(data["pair_j"], dtype=np.int64)
            if "dataset_key" in data:
                keys = np.asarray(data["dataset_key"]).astype(str)
                sel = keys == dataset.key
                if not bool(sel.any()):
                    raise RunnerError(
                        f"{pair_list} holds no rows for dataset {dataset.key!r}; "
                        f"it names {sorted(set(keys.tolist()))[:8]}"
                    )
                pi, pj = pi[sel], pj[sel]
            if pi.size and (int(min(pi.min(), pj.min())) < 0 or int(max(pi.max(), pj.max())) >= n):
                raise RunnerError(
                    f"{pair_list} holds a graph index outside [0, {n}) for {dataset.key!r}"
                )
            if bool(np.any(pi == pj)):
                raise RunnerError(f"{pair_list} holds a self-pair for {dataset.key!r}")
            lo = np.minimum(pi, pj)
            hi = np.maximum(pi, pj)
            listed = np.unique(indices_of_pairs(lo, hi, n))
        else:
            raise RunnerError(
                f"{pair_list} has neither a 'pair_index' key nor a 'pair_i'/'pair_j' pair"
            )
    if listed.size and (int(listed.min()) < 0 or int(listed.max()) >= total):
        raise RunnerError(f"{pair_list} holds an index outside [0, C({n}, 2) = {total})")
    rng = split_range(int(listed.size), n_chunks, chunk_index)
    return listed[rng.start : rng.stop], rng


def _load_checkpoint(
    path: Path, fingerprint: str, target: set[int]
) -> tuple[dict[int, PairOutcome], dict[str, Any]]:
    """Load a checkpoint, refusing one that belongs to a different chunk.

    Args:
        path: Checkpoint path.
        fingerprint: SHA-256 of this task's target pair set.
        target: This task's target pair indices.

    Returns:
        ``(rows, checkpoint_meta)``.

    Raises:
        RunnerError: If the checkpoint belongs to a different pair set, or holds an
            index this task does not own.
    """
    rows = _read_shard_like(path)
    meta: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        if "meta" in data:
            try:
                meta = dict(json.loads(str(data["meta"])))
            except (ValueError, TypeError):
                meta = {}
    stored = str(meta.get("pairs_sha256", ""))
    if stored and stored != fingerprint:
        raise RunnerError(
            f"checkpoint {path} was written for a different pair set "
            f"(sha256 {stored[:12]}... vs {fingerprint[:12]}...). Resuming it would corrupt "
            "the shard. Delete it or point --checkpoint at the right file."
        )
    stray = [k for k in rows if k not in target]
    if stray:
        raise RunnerError(
            f"checkpoint {path} holds {len(stray)} pair indices this chunk does not own "
            f"(first: {sorted(stray)[:5]})"
        )
    return rows, meta


def _cpu_model() -> str:
    """Return the CPU model string, for the shard metadata.

    Returns:
        The first ``model name`` in ``/proc/cpuinfo``, or a platform fallback.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover - non-Linux
        pass
    return platform.processor() or "unknown"


def run_chunk(
    *,
    dataset: ContractA,
    pairs: np.ndarray,
    backend_spec: BackendSpec,
    input_path: str,
    workers: int,
    checkpoint_path: Path | None,
    checkpoint_every: int,
    seed_from: str | None = None,
    batch_size: int = 1,
    backend_factory: Any | None = None,
) -> tuple[ShardData, bool, dict[str, Any]]:
    """Compute a chunk of pairs, resuming from a checkpoint if one is present.

    Args:
        dataset: The loaded cohort.
        pairs: Ascending, deduplicated pair indices this task owns.
        backend_spec: How workers build their backend.
        input_path: Path workers reload the cohort from.
        workers: Process count. ``1`` runs in-process, which is also the only mode
            that accepts an injected ``backend_factory``.
        checkpoint_path: Where to write the single, in-place checkpoint.
        checkpoint_every: Flush after this many newly completed pairs.
        seed_from: Optional earlier result file whose pairs are reused rather than
            recomputed.
        batch_size: Pairs per task. Defaults to 1: process-pool overhead is tens of
            microseconds against seconds of solver time, and a larger batch would
            multiply the worst-case tail by the batch size.
        backend_factory: A zero-argument-compatible callable taking the
            :class:`BackendSpec`, for tests. Requires ``workers == 1``.

    Returns:
        ``(shard, completed, stats)``. ``completed`` is ``False`` when a signal cut
        the run short, in which case the checkpoint holds the partial state.

    Raises:
        RunnerError: On a mismatched checkpoint, an injected factory with more than
            one worker, or an unrecoverable backend construction failure.
    """
    n = dataset.n_graphs
    target = {int(k) for k in pairs}
    fingerprint = _pairs_fingerprint(pairs)
    shard = ShardData()
    stats: dict[str, Any] = {"n_resumed": 0, "n_seeded": 0, "n_computed": 0}

    if checkpoint_path is not None and checkpoint_path.exists():
        rows, _ = _load_checkpoint(checkpoint_path, fingerprint, target)
        for row in rows.values():
            shard.add(row)
        stats["n_resumed"] = len(rows)
        logger.info("resumed %d pairs from %s", len(rows), checkpoint_path)

    if seed_from:
        seeded = _read_shard_like(Path(seed_from))
        reused = 0
        for k, row in seeded.items():
            if k in target and k not in shard.rows:
                shard.add(row)
                reused += 1
        stats["n_seeded"] = reused
        logger.info("seeded %d pairs from %s", reused, seed_from)

    todo = [int(k) for k in pairs if int(k) not in shard.rows]
    logger.info(
        "chunk: %d pairs owned, %d already known, %d to compute",
        len(target),
        len(shard),
        len(todo),
    )

    def _flush() -> None:
        if checkpoint_path is None:
            return
        payload: dict[str, np.ndarray | np.generic] = dict(shard.arrays())
        payload["meta"] = np.array(
            json.dumps(
                {
                    "pairs_sha256": fingerprint,
                    "n_graphs": n,
                    "n_target": len(target),
                    "n_done": len(shard),
                    "input": input_path,
                    "backend": backend_spec.name,
                    "cost_model": backend_spec.cost_model,
                    "written_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        _write_npz_atomic(checkpoint_path, payload)

    if not todo:
        _flush()
        return shard, True, stats

    since_ckpt = 0
    completed = True
    batches = [tuple(todo[t : t + batch_size]) for t in range(0, len(todo), max(1, batch_size))]

    def _collect(outcomes: list[PairOutcome]) -> None:
        nonlocal since_ckpt
        for outcome in outcomes:
            shard.add(outcome)
            stats["n_computed"] = int(stats["n_computed"]) + 1
            since_ckpt += 1

    def _checkpoint_due() -> bool:
        return since_ckpt >= checkpoint_every

    def _flush_and_reset() -> None:
        nonlocal since_ckpt
        _flush()
        since_ckpt = 0

    if workers <= 1:
        backend = (
            backend_factory(backend_spec)
            if backend_factory is not None
            else make_backend(backend_spec)
        )
        _probe_backend(backend, backend_spec)
        for batch in batches:
            if _TERMINATE:
                completed = False
                break
            _collect([_compute_one(k, dataset.graphs, backend, n) for k in batch])
            if _checkpoint_due():
                _flush_and_reset()
    else:
        if backend_factory is not None:
            raise RunnerError("an injected backend_factory requires --workers 1; it may not pickle")
        completed = _run_pool(
            batches=batches,
            spec=_WorkerSpec(input_path=input_path, backend=backend_spec),
            workers=workers,
            collect=_collect,
            should_flush=_checkpoint_due,
            flush=_flush_and_reset,
        )

    _flush()
    return shard, completed, stats


def _run_pool(
    *,
    batches: list[tuple[int, ...]],
    spec: _WorkerSpec,
    workers: int,
    collect: Any,
    should_flush: Any,
    flush: Any,
) -> bool:
    """Drive a bounded window of batches through a process pool.

    A bounded window rather than ``Executor.map`` over the whole chunk: a stage-2
    task owns of the order of 10**5 pairs, and submitting them all at once would
    hold that many futures in memory for no benefit. Completion is out of order,
    which is safe because the checkpoint stores explicit pair indices.

    Args:
        batches: Work units.
        spec: Worker specification.
        workers: Process count.
        collect: Callback receiving each batch's outcomes.
        should_flush: Predicate saying a checkpoint is due.
        flush: Callback writing the checkpoint.

    Returns:
        ``True`` if every batch finished, ``False`` if a signal cut the run short.
    """
    window = max(4 * workers, 16)
    pending: dict[Future[list[PairOutcome]], tuple[int, ...]] = {}
    it = iter(batches)
    exhausted = False
    completed = True

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(spec,)
    ) as pool:
        while True:
            while not exhausted and not _TERMINATE and len(pending) < window:
                try:
                    batch = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(_compute_batch, batch)] = batch
            if not pending:
                if exhausted or _TERMINATE:
                    completed = exhausted and not _TERMINATE
                    break
                continue
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED, timeout=1.0)
            for fut in done:
                pending.pop(fut, None)
                collect(fut.result())
            if should_flush():
                flush()
            if _TERMINATE:
                completed = False
                for fut in list(pending):
                    fut.cancel()
                # Collect whatever already finished so the checkpoint keeps it.
                for fut in list(pending):
                    if fut.done() and not fut.cancelled():
                        collect(fut.result())
                    pending.pop(fut, None)
                pool.shutdown(wait=False, cancel_futures=True)
                break
    return completed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser, frozen in CONTRACTS §6.

    Returns:
        The configured parser. ``--backend-factory`` and ``--batch-size`` are
        additions; every flag named in the contract keeps its name and meaning.
    """
    p = argparse.ArgumentParser(
        prog="ged_exact_runner",
        description="Compute one chunk of exact GED pairs and write a CONTRACT C shard.",
    )
    p.add_argument("--input", required=True, help="CONTRACT A .npz")
    p.add_argument("--out", required=True, help="output shard .npz")
    p.add_argument("--backend", required=True, choices=("gedlib", "networkx", "stub"))
    p.add_argument("--cost-model", default="unit", choices=("unit", "graphedx"))
    p.add_argument("--chunk-index", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=1)
    p.add_argument("--pair-list", default=None, help="stage 1: explicit linear indices")
    p.add_argument("--seed-from", default=None, help="stage 2: reuse already-computed pairs")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--timeout-per-pair", type=float, default=300.0)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--checkpoint", default=None, help="single checkpoint file, overwritten")
    p.add_argument("--batch-size", type=int, default=1, help="pairs per pool task (default 1)")
    p.add_argument(
        "--backend-factory",
        default=None,
        help=(
            "explicit 'module:callable' override that builds the backend; the only way to "
            "drive this runner where ged_backends is absent. Requires --backend stub."
        ),
    )
    # CONTRACTS §6. Every default below is T-03's behaviour, unchanged: the
    # methods it ran, the option string it built from --threads 1, and both
    # ends. A campaign that passes none of these computes exactly what the
    # exact-GED census computed.
    p.add_argument("--lb-method", default="BRANCH_FAST", help="GEDLIB lower-bound method")
    p.add_argument(
        "--lb-options",
        default="--threads 1",
        help="verbatim GEDLIB option string for the lower-bound method",
    )
    p.add_argument("--ub-method", default="IPFP", help="GEDLIB upper-bound method")
    p.add_argument(
        "--ub-options",
        default="--threads 1",
        help="verbatim GEDLIB option string for the upper-bound method",
    )
    p.add_argument(
        "--compute",
        default="both",
        choices=("lb", "ub", "both"),
        help="which ends of the bracket to evaluate (default both, T-03's behaviour)",
    )
    p.add_argument("--role", default=None, help="role label recorded in the shard metadata")
    p.add_argument(
        "--no-accessor-probe",
        action="store_true",
        help="skip the P4/C4 accessor probe; only for a backend that has no accessors",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _backend_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the backend constructor options the CLI describes.

    Only GEDLIB takes a method specification, so the options are empty for
    every other backend and its construction is bit-identical to before.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Keyword arguments for the backend constructor.
    """
    if args.backend != "gedlib":
        return {}
    return {
        "lb_method": str(args.lb_method),
        "lb_options": str(args.lb_options),
        "ub_method": str(args.ub_method),
        "ub_options": str(args.ub_options),
        "compute": str(args.compute),
    }


def _shard_meta(
    *,
    dataset: ContractA,
    args: argparse.Namespace,
    shard: ShardData,
    stats: dict[str, Any],
    backend_name: str,
    started: str,
) -> str:
    """Build the shard's ``meta`` JSON string.

    Args:
        dataset: The cohort.
        args: Parsed CLI arguments.
        shard: The accumulated rows.
        stats: Resume/seed/compute counts.
        backend_name: The backend's self-reported name.
        started: ISO-8601 UTC start time.

    Returns:
        A JSON string.
    """
    arrays = shard.arrays()
    n_cert = int(np.count_nonzero(arrays["certified"]))
    return json.dumps(
        {
            "dataset": dataset.key,
            "n_graphs": dataset.n_graphs,
            "backend": args.backend,
            "backend_name": backend_name,
            "backend_factory": args.backend_factory,
            "cost_model": args.cost_model,
            # CONTRACTS §3: the options string is part of the method name, so a
            # shard that does not record it verbatim does not identify what it
            # computed and is rejected at the gate.
            "role": getattr(args, "role", None),
            "compute": str(getattr(args, "compute", "both")),
            "lb_method": str(getattr(args, "lb_method", "BRANCH_FAST")),
            "lb_options": str(getattr(args, "lb_options", "--threads 1")),
            "ub_method": str(getattr(args, "ub_method", "IPFP")),
            "ub_options": str(getattr(args, "ub_options", "--threads 1")),
            "chunk_index": int(args.chunk_index),
            "n_chunks": int(args.n_chunks),
            "pair_list": args.pair_list,
            "seed_from": args.seed_from,
            "timeout_per_pair": float(args.timeout_per_pair),
            "workers": int(args.workers),
            "batch_size": int(args.batch_size),
            "n_pairs": len(shard),
            "n_certified": n_cert,
            "n_censored": len(shard) - n_cert,
            "n_failed": shard.n_failed,
            "n_resumed": int(stats["n_resumed"]),
            "n_seeded": int(stats["n_seeded"]),
            "n_computed": int(stats["n_computed"]),
            "hostname": socket.gethostname(),
            "cpu_model": _cpu_model(),
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
        }
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on failure, ``143`` when a signal stopped the run
        after the checkpoint was flushed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_term_handler()
    started = datetime.now(timezone.utc).isoformat()

    if args.backend_factory and args.backend != "stub":
        logger.error("--backend-factory requires --backend stub, got %r", args.backend)
        return EXIT_FAILURE

    try:
        dataset = load_contract_a(args.input)
        pairs, rng = _target_pairs(
            dataset,
            pair_list=args.pair_list,
            chunk_index=args.chunk_index,
            n_chunks=args.n_chunks,
        )
        logger.info(
            "%s: %d graphs, chunk %d/%d -> positions [%d, %d), %d pairs",
            dataset.key,
            dataset.n_graphs,
            args.chunk_index,
            args.n_chunks,
            rng.start,
            rng.stop,
            pairs.size,
        )
        spec = BackendSpec(
            name=args.backend,
            cost_model=args.cost_model,
            timeout_s=float(args.timeout_per_pair),
            factory_ref=args.backend_factory,
            options=_backend_options(args),
            probe_accessors=not bool(args.no_accessor_probe),
        )
        shard, completed, stats = run_chunk(
            dataset=dataset,
            pairs=pairs,
            backend_spec=spec,
            input_path=str(args.input),
            workers=int(args.workers),
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            checkpoint_every=int(args.checkpoint_every),
            seed_from=args.seed_from,
            batch_size=int(args.batch_size),
        )
    except (RunnerError, GedPairIndexError, OSError) as exc:
        logger.error("chunk failed: %s", exc)
        return EXIT_FAILURE

    if not completed:
        logger.warning(
            "stopped early on a signal with %d/%d pairs done; checkpoint flushed, no shard written",
            len(shard),
            pairs.size,
        )
        logging.shutdown()
        return EXIT_TERMINATED

    if len(shard) != pairs.size:
        logger.error("shard holds %d rows but the chunk owns %d pairs", len(shard), pairs.size)
        return EXIT_FAILURE

    backend_name = args.backend_factory or args.backend
    payload: dict[str, np.ndarray | np.generic] = dict(shard.arrays())
    payload["meta"] = np.array(
        _shard_meta(
            dataset=dataset,
            args=args,
            shard=shard,
            stats=stats,
            backend_name=backend_name,
            started=started,
        )
    )
    _write_npz_atomic(Path(args.out), payload)
    logger.info(
        "wrote %s: %d pairs (%d resumed, %d seeded, %d computed, %d failed)",
        args.out,
        len(shard),
        stats["n_resumed"],
        stats["n_seeded"],
        stats["n_computed"],
        shard.n_failed,
    )
    if shard.n_failed:
        logger.error("%d pairs failed in the backend; gate 4 will reject them", shard.n_failed)
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
