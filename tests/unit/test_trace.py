"""Tests for the stdlib-only trace schema and the two trace emitters."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.core.trace import (
    SCHEMA_VERSION,
    AlgorithmTrace,
    StepSnapshot,
    dump_trace,
    graph_edges,
    graph_to_dict,
    load_trace,
    normalise_edge,
)

ALPHABET = "NnPpVvCcW"

instruction_strings = st.text(alphabet=ALPHABET, min_size=0, max_size=24)


# ---------------------------------------------------------------------------
# The core layer must stay dependency-free
# ---------------------------------------------------------------------------

_CORE_DIR = Path(__file__).resolve().parents[2] / "src" / "isalgraph" / "core"
_STDLIB_OK = {
    "abc",
    "array",
    "collections",
    "contextlib",
    "copy",
    "ctypes",
    "dataclasses",
    "enum",
    "functools",
    "heapq",
    "importlib",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "platform",
    "random",
    "sys",
    "time",
    "typing",
    "warnings",
    "__future__",
}


def _module_scope_imports(path: Path) -> set[str]:
    """Return top-level package names imported at module scope in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:  # module scope only, not function bodies
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("path", sorted(_CORE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_core_has_no_module_scope_third_party_imports(path: Path) -> None:
    """No module in ``isalgraph.core`` may import a third-party library."""
    offenders = _module_scope_imports(path) - _STDLIB_OK - {"isalgraph"}
    assert not offenders, f"{path.name} imports non-stdlib at module scope: {sorted(offenders)}"


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


def test_normalise_edge_orients_undirected_but_not_directed() -> None:
    assert normalise_edge(3, 1, directed=False) == (1, 3)
    assert normalise_edge(1, 3, directed=False) == (1, 3)
    assert normalise_edge(3, 1, directed=True) == (3, 1)


def test_trace_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        AlgorithmTrace(direction="x2y", directed=False, final_graph={}, snapshots=())


def test_trace_rejects_unknown_schema() -> None:
    payload = {"schema": "isalgraph.trace.v99", "direction": "s2g"}
    with pytest.raises(ValueError, match="unsupported trace schema"):
        AlgorithmTrace.from_json(payload)


def test_envelope_carries_the_version_tag() -> None:
    _, trace = StringToGraph("VNV", directed_graph=False).run_with_trace()
    assert trace.to_json()["schema"] == SCHEMA_VERSION


@given(instruction_strings, st.booleans())
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_trace_round_trips_through_json(instructions: str, directed: bool) -> None:
    """``dump_trace`` then ``load_trace`` must reproduce the trace exactly."""
    _, trace = StringToGraph(instructions, directed_graph=directed).run_with_trace()
    restored = AlgorithmTrace.from_json(json.loads(json.dumps(trace.to_json())))
    assert restored == trace


def test_dump_and_load_are_byte_identical(tmp_path: Path) -> None:
    """Re-dumping a loaded trace must produce the same bytes."""
    _, trace = StringToGraph("VNVnVCPvNC", directed_graph=False).run_with_trace()
    first = tmp_path / "trace.json"
    second = tmp_path / "trace2.json"
    dump_trace(trace, first)
    dump_trace(load_trace(first), second)
    assert first.read_bytes() == second.read_bytes()
    assert load_trace(first) == trace


def test_snapshot_json_preserves_none_created_edge() -> None:
    snap = StepSnapshot(0, None, (0,), 0, 0, (0,), (), None, "")
    assert StepSnapshot.from_json(snap.to_json()) == snap


# ---------------------------------------------------------------------------
# S2G emission
# ---------------------------------------------------------------------------


@given(instruction_strings, st.booleans())
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_s2g_run_and_run_with_trace_agree(instructions: str, directed: bool) -> None:
    """The frozen ``run()`` and the new emitter must build the same graph."""
    plain, _ = StringToGraph(instructions, directed_graph=directed).run()
    traced, trace = StringToGraph(instructions, directed_graph=directed).run_with_trace()

    assert plain.node_count() == traced.node_count()
    assert graph_edges(plain) == graph_edges(traced)
    assert trace.direction == "s2g"
    assert trace.directed is directed
    assert trace.instruction_string == instructions


@given(instruction_strings, st.booleans())
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_s2g_snapshot_count_and_replay(instructions: str, directed: bool) -> None:
    """One snapshot per instruction plus the initial state, and masks replay."""
    graph, trace = StringToGraph(instructions, directed_graph=directed).run_with_trace()

    assert len(trace.snapshots) == len(instructions) + 1

    final = trace.snapshots[-1]
    assert set(final.active_nodes) == set(range(graph.node_count()))
    assert set(final.active_edges) == set(graph_edges(graph))
    assert trace.final_graph == graph_to_dict(graph)

    for i, snap in enumerate(trace.snapshots):
        assert snap.step_idx == i
        assert snap.partial_string == instructions[:i]
        assert snap.instruction == (None if i == 0 else instructions[i - 1])
        assert len(snap.cdll_node_order) == len(snap.active_nodes)
        assert snap.primary_node in snap.active_nodes
        assert snap.secondary_node in snap.active_nodes

    # Active sets grow monotonically.
    for before, after in zip(trace.snapshots, trace.snapshots[1:], strict=False):
        assert set(before.active_nodes) <= set(after.active_nodes)
        assert set(before.active_edges) <= set(after.active_edges)


@given(instruction_strings, st.booleans())
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_created_edges_partition_the_final_edge_set(instructions: str, directed: bool) -> None:
    """Every edge is attributed to exactly one step, with no duplicates.

    This is the property that re-deriving attribution from token counts
    cannot hold: a ``C``/``c`` between already-adjacent nodes creates
    nothing, so a counter would over-attribute.
    """
    graph, trace = StringToGraph(instructions, directed_graph=directed).run_with_trace()

    created = [s.created_edge for s in trace.snapshots if s.created_edge is not None]
    assert len(created) == len(set(created)), "an edge was attributed to two steps"
    assert set(created) == set(graph_edges(graph))

    for snap in trace.snapshots:
        if snap.instruction in (None, "N", "P", "n", "p", "W"):
            assert snap.created_edge is None


def test_noop_connect_records_no_created_edge() -> None:
    """A ``C`` over an already-adjacent pair is a no-op and is not attributed.

    ``V`` builds edge ``(0, 1)`` but moves neither pointer, so ``n`` is
    needed to walk the secondary onto node 1. The following ``C`` then
    targets a pair that is already adjacent and must create nothing --
    the exact case a token-counting attribution scheme gets wrong.
    """
    _, trace = StringToGraph("VnC", directed_graph=False).run_with_trace()
    assert [s.instruction for s in trace.snapshots] == [None, "V", "n", "C"]
    assert trace.snapshots[1].created_edge == (0, 1)
    assert trace.snapshots[2].created_edge is None  # movement
    assert trace.snapshots[3].created_edge is None  # edge already present


def test_connect_with_both_pointers_on_one_node_creates_a_self_loop() -> None:
    """``V`` moves no pointer, so a bare ``VC`` genuinely adds ``(0, 0)``."""
    _, trace = StringToGraph("VCC", directed_graph=False).run_with_trace()
    assert trace.snapshots[1].created_edge == (0, 1)
    assert trace.snapshots[2].created_edge == (0, 0)
    assert trace.snapshots[3].created_edge is None  # now a no-op


# ---------------------------------------------------------------------------
# G2S emission
# ---------------------------------------------------------------------------


def _graph_from_string(instructions: str, *, directed: bool = False) -> SparseGraph:
    graph, _ = StringToGraph(instructions, directed_graph=directed).run()
    return graph


@given(instruction_strings)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_g2s_run_and_run_with_trace_agree(instructions: str) -> None:
    """Both entry points must emit the identical string, and the trace must replay."""
    graph = _graph_from_string(instructions)
    if graph.node_count() < 2:
        return
    try:
        plain, _ = GraphToString(graph).run(0)
        encoded, trace = GraphToString(graph).run_with_trace(0)
    except (ValueError, RuntimeError):
        return  # disconnected input; not this test's concern

    assert plain == encoded
    assert trace.direction == "g2s"
    assert len(trace.snapshots) == len(encoded) + 1
    assert trace.instruction_string == encoded

    replayed, _ = StringToGraph(encoded, directed_graph=False).run()
    assert replayed.is_isomorphic(graph)
    assert trace.final_graph == graph_to_dict(replayed)


def test_g2s_replay_reproduces_the_encoder_output_graph_exactly() -> None:
    """The replay used to build the G2S trace is exact, not merely isomorphic.

    ``run_with_trace`` documents this; the docstring would be wrong if
    node ids ever diverged between encoder and replay.
    """
    graph = _graph_from_string("VNVnVCPvNC")
    converter = GraphToString(graph)
    encoded, _ = converter.run(0)
    replayed, _ = StringToGraph(encoded, directed_graph=False).run()
    assert graph_edges(replayed) == graph_edges(converter._output_graph)  # noqa: SLF001
