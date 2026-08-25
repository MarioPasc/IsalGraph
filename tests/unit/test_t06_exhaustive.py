"""Acceptance tests for the ``isalgraph_exhaustive`` arm.

The arm exists because the ``SUITE1_ONLY`` guard on ``isalgraph_canonical``
encodes a ceiling the C++ engine does not have: the ``n = 12`` refusal was
calibrated at a 2 s budget on the pure-Python path.  The exhaustive form is the
true ``w*_G`` and is measurably shorter than the pruned form -- 8-12 % at
``n = 13-20``, 12-22 % at ``n = 23-26``, and never longer on any of the 5,350
Suite-1 graphs.

Each test below exists because a specific silent failure is available here:

- **The fallback is exercised with a budget small enough that it actually
  fires.**  A cascade assertion over zero substituted graphs passes for the
  wrong reason, which is the same defect the D14 tests in
  ``test_t06_encoding.py`` were written against.
- **The cascade is asserted to end in an unbudgeted tier**, because "a censored
  graph is retained, never dropped" is false the moment every tier can fail.
- **The fallback is asserted to happen in the driver, not the backend.**  A
  backend that substituted internally would report ``status="ok",
  fallback_used=False`` and launder a censored graph into a completed one.
- **``graph6`` and ``nauty_graph6`` are asserted to carry identical bit
  counts**, because graph6's length is a function of ``n`` alone and the two
  columns must therefore agree by construction -- worth asserting once rather
  than rediscovering in a results table.

Absolute bit counts produced by a backend are not asserted: ``import
isalgraph`` resolves to the installed checkout, so a count can move under this
file with no error raised.  Invariants and relative orderings are asserted
instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.real_data.eval_encoding import t06_cohort
from benchmarks.real_data.eval_encoding.t06_encode import (
    REPRESENTATIONS,
    EncodeConfig,
    fallback_tier_counts,
    run_campaign,
)
from benchmarks.real_data.eval_encoding.t06_encode_worker import (
    DEFAULT_CASCADE,
    ISALGRAPH_ALPHABET_SIZE,
    ISALGRAPH_ARMS,
    fallback_cascade,
    symbol_sep,
)
from isalgraph.competitors import bits as bits_mod
from isalgraph.competitors import fixtures
from isalgraph.competitors.base import Budget, Capability
from isalgraph.competitors.registry import get_repr_backend, registered_backends
from isalgraph.errors import SuiteScopeError

ARM = "isalgraph_exhaustive"

#: Node counts spanning the band where the exhaustive form starts to win. Below
#: ``n = 13`` the two forms usually coincide, so a strict-inequality assertion
#: there would be flaky rather than wrong.
SAVING_NODE_COUNTS = (14, 16, 18, 20)

CENSORED_DATASET = "protein"
CENSORED_LIMIT = 8

needs_cohort = pytest.mark.skipif(
    not (t06_cohort.cohort_root() / t06_cohort.EXPORT_SUBDIR).is_dir(),
    reason="the frozen cohort export is not mounted",
)


def _connected(n: int, extra: int, seed: int) -> nx.Graph:
    """A connected graph on *n* nodes: a random tree plus *extra* chords.

    The encoder requires a node that reaches every other, so a plain
    ``gnm_random_graph`` raises ``DisconnectedGraphError`` a large fraction of
    the time and would make these tests flaky for a reason unrelated to the arm.
    """
    graph = nx.random_labeled_tree(n, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(extra):
        u, v = (int(x) for x in rng.integers(0, n, size=2))
        if u != v:
            graph.add_edge(u, v)
    return graph


# ----------------------------------------------------------------------
# Registration and declared properties
# ----------------------------------------------------------------------


def test_the_arm_is_registered() -> None:
    assert ARM in registered_backends()


def test_the_arm_declares_the_same_capabilities_as_pruned_minus_the_scope_guard() -> None:
    """Same claims about the encoding; only the suite refusal differs."""
    exhaustive = get_repr_backend(ARM)
    pruned = get_repr_backend("isalgraph_pruned")
    assert exhaustive.capabilities == pruned.capabilities
    assert Capability.CANONICAL in exhaustive.capabilities
    assert Capability.COMPLETE_INVARIANT in exhaustive.capabilities
    assert Capability.REVERSIBLE in exhaustive.capabilities
    assert Capability.SUITE1_ONLY not in exhaustive.capabilities


def test_the_arm_accepts_a_graph_the_suite1_only_arm_refuses() -> None:
    """The whole point of the third arm: n = 40 must encode, not raise."""
    big = nx.path_graph(40)
    with pytest.raises(SuiteScopeError):
        get_repr_backend("isalgraph_canonical").encode(big)
    encoding = get_repr_backend(ARM).encode(big, budget=Budget(timeout_s=None))
    assert encoding.length > 0


def test_the_arm_computes_the_same_string_as_the_suite1_only_arm() -> None:
    """Same function, different guard. If these diverge, one of them is wrong."""
    canonical = get_repr_backend("isalgraph_canonical")
    exhaustive = get_repr_backend(ARM)
    for name in fixtures.CONNECTED_FIXTURES:
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[name])
        if graph.number_of_nodes() > 12:
            continue
        budget = Budget(timeout_s=None)
        assert (
            exhaustive.encode(graph, budget=budget).text
            == canonical.encode(graph, budget=budget).text
        )


# ----------------------------------------------------------------------
# Bit conventions -- it must flow through competitors/bits.py unchanged
# ----------------------------------------------------------------------


def test_the_arm_uses_the_nine_symbol_alphabet() -> None:
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    encoding = get_repr_backend(ARM).encode(graph, budget=Budget(timeout_s=None))
    assert encoding.alphabet_size == ISALGRAPH_ALPHABET_SIZE == 9


def test_the_arm_flows_through_the_bit_module_unchanged() -> None:
    """Same bit row as the pruned arm: only the string length differs."""
    graph = _connected(16, 6, seed=42)
    budget = Budget(timeout_s=None)
    exhaustive = get_repr_backend(ARM).encode(graph, budget=budget)
    pruned = get_repr_backend("isalgraph_pruned").encode(graph, budget=budget)
    count_e = bits_mod.count(exhaustive)
    count_p = bits_mod.count(pruned)
    # entropy bits are L*log2(9) under both, so the ratio is the length ratio.
    assert count_e.entropy_bits / exhaustive.length == pytest.approx(
        count_p.entropy_bits / pruned.length
    )


@pytest.mark.parametrize("n", SAVING_NODE_COUNTS)
def test_the_exhaustive_form_is_never_longer_than_the_pruned_one(n: int) -> None:
    """Measured 0 of 5,350 on Suite 1. It is a property, not a tendency."""
    graph = _connected(n, n // 3, seed=n)
    budget = Budget(timeout_s=None)
    exhaustive = get_repr_backend(ARM).encode(graph, budget=budget)
    pruned = get_repr_backend("isalgraph_pruned").encode(graph, budget=budget)
    assert exhaustive.length <= pruned.length


def test_the_saving_is_real_somewhere_in_the_measured_band() -> None:
    """Guards the parametrised test above from passing on ties alone."""
    budget = Budget(timeout_s=None)
    strict = 0
    for n in SAVING_NODE_COUNTS:
        graph = _connected(n, n // 3, seed=n)
        if (
            get_repr_backend(ARM).encode(graph, budget=budget).length
            < get_repr_backend("isalgraph_pruned").encode(graph, budget=budget).length
        ):
            strict += 1
    assert strict > 0, "no graph in the band was strictly shorter; the band is wrong"


def test_graph6_and_nauty_graph6_carry_identical_bit_counts() -> None:
    """graph6's length is a function of n alone, so canonical relabelling
    cannot move it. The two columns agree by construction; asserting it here
    beats rediscovering it in a results table.
    """
    if "nauty_graph6" not in registered_backends():
        pytest.skip("nauty backend unavailable")
    plain = get_repr_backend("graph6")
    canon = get_repr_backend("nauty_graph6")
    for name in fixtures.CONNECTED_FIXTURES:
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[name])
        a = bits_mod.count(plain.encode(graph))
        b = bits_mod.count(canon.encode(graph))
        assert a.entropy_bits == b.entropy_bits, name
        assert a.realised_bits == b.realised_bits, name


# ----------------------------------------------------------------------
# Reversibility
# ----------------------------------------------------------------------


def test_the_arm_round_trips_up_to_isomorphism() -> None:
    backend = get_repr_backend(ARM)
    for name in fixtures.CONNECTED_FIXTURES:
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[name])
        rebuilt = backend.decode(backend.encode(graph, budget=Budget(timeout_s=None)))
        assert nx.is_isomorphic(graph, rebuilt), name


# ----------------------------------------------------------------------
# The fallback cascade
# ----------------------------------------------------------------------


def test_the_backend_declares_its_fallback_but_does_not_perform_it() -> None:
    """D14 lives in the driver. A backend that substituted internally would
    report ``ok``/``False`` and hide the censoring the arm exists to measure.
    """
    backend = get_repr_backend(ARM)
    assert backend.fallback_variant == "pruned"
    source = Path("src/isalgraph/competitors/backends/isalgraph_ref.py").resolve()
    if not source.is_file():  # pragma: no cover - layout guard
        source = (
            Path(__file__).resolve().parents[2]
            / "src/isalgraph/competitors/backends/isalgraph_ref.py"
        )
    text = source.read_text()
    assert "GreedyMinG2S" not in text, "the backend must not build a fallback string itself"
    assert "except CanonicalizationTimeoutError" not in text


def test_the_declared_variant_matches_the_worker_cascade() -> None:
    """Two tables, one policy. They must not drift apart silently."""
    backend = get_repr_backend(ARM)
    assert fallback_cascade(ARM)[0] == backend.fallback_variant


def test_the_cascade_ends_in_an_unbudgeted_tier() -> None:
    """'Never drop a graph' is false the moment every tier can fail."""
    for representation in ISALGRAPH_ARMS:
        assert fallback_cascade(representation)[-1] == "greedy"
    assert DEFAULT_CASCADE[-1] == "greedy"


def test_the_arm_is_wired_into_both_driver_tables() -> None:
    assert ARM in ISALGRAPH_ARMS
    assert ARM in REPRESENTATIONS


def test_the_arm_uses_the_empty_separator() -> None:
    """IsalGraph symbols are single characters, so ``length == len(encoding)``."""
    assert symbol_sep(ARM) == ""


def test_the_cascade_degrades_to_pruned_then_greedy_under_an_impossible_budget() -> None:
    """Exercises **both** tiers, which is the point: a cascade asserted only at
    its first tier is not a cascade.
    """
    from benchmarks.real_data.eval_encoding.t06_encode_worker import _fallback_text

    graph = _connected(30, 10, seed=7)
    text, tier = _fallback_text(ARM, graph, Budget(timeout_s=None))
    assert tier == "pruned"
    assert set(text) <= set("NnPpVvCcW")

    text, tier = _fallback_text(ARM, graph, Budget(timeout_s=1e-9))
    assert tier == "greedy", "the pruned tier did not time out; the test is vacuous"
    assert set(text) <= set("NnPpVvCcW")


def test_the_pruned_arm_cascade_is_unchanged() -> None:
    """The new arm must not move a published failure rate on the old one."""
    assert fallback_cascade("isalgraph_pruned") == ("greedy",)
    assert fallback_cascade("isalgraph_canonical") == ("greedy",)


def test_tier_counts_default_to_greedy_for_a_message_without_a_tier() -> None:
    """Rows written before the tier was recorded are greedy-tier by history."""
    records = [
        {"fallback_used": True, "message": "fallback_tier=pruned"},
        {"fallback_used": True, "message": "fallback_tier=greedy"},
        {"fallback_used": True, "message": ""},
        {"fallback_used": False, "message": "fallback_tier=pruned"},
    ]
    assert fallback_tier_counts(records) == {"pruned": 1, "greedy": 2}


# ----------------------------------------------------------------------
# End to end, on the real cohort, with censoring that actually fires
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def censored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign whose budget is small enough that the cascade actually runs."""
    out = tmp_path_factory.mktemp("t06_exhaustive_censored")
    return run_campaign(
        EncodeConfig(
            suite="suite2",
            dataset=CENSORED_DATASET,
            representation=ARM,
            out_dir=out,
            limit=CENSORED_LIMIT,
            budget_s=0.001,
        )
    )


@needs_cohort
def test_censoring_actually_fires(censored: Path) -> None:
    with np.load(censored, allow_pickle=False) as handle:
        status = handle["status"]
    assert int((status == "censored").sum()) > 0, "censoring never fired; the test is vacuous"


@needs_cohort
def test_censored_implies_fallback_and_non_empty_encoding(censored: Path) -> None:
    """D14, on the new arm: retained with a substitute string, never dropped."""
    with np.load(censored, allow_pickle=False) as handle:
        status, fallback, encoding = handle["status"], handle["fallback_used"], handle["encoding"]
    mask = status == "censored"
    assert bool(fallback[mask].all())
    assert all(str(text) != "" for text in encoding[mask])


@needs_cohort
def test_no_graph_is_dropped(censored: Path) -> None:
    cohort = t06_cohort.load_cohort("suite2", CENSORED_DATASET, limit=CENSORED_LIMIT)
    with np.load(censored, allow_pickle=False) as handle:
        assert np.array_equal(handle["graph_ids"], cohort.graph_ids)
        assert handle["graph_ids"].shape[0] == CENSORED_LIMIT


@needs_cohort
def test_the_substituted_string_is_an_instruction_string(censored: Path) -> None:
    with np.load(censored, allow_pickle=False) as handle:
        status, encoding, length = handle["status"], handle["encoding"], handle["length"]
    mask = status == "censored"
    for text, count in zip(encoding[mask], length[mask], strict=True):
        assert set(str(text)) <= set("NnPpVvCcW")
        assert int(count) == len(str(text))


@needs_cohort
def test_the_metadata_records_the_budget_and_the_tier_tally(censored: Path) -> None:
    """A censoring rate is a property of its budget, so the budget travels with
    the file; the tier tally travels in ``notes`` because the schema rejects a
    file whose key set is not exactly ``ENCODINGS_KEYS``.
    """
    with np.load(censored, allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata"]))
    assert metadata["encode_budget_s"] == 0.001
    assert metadata["representation"] == ARM
    assert "fallback_tiers=" in metadata["notes"]


@needs_cohort
def test_the_emitted_file_has_exactly_the_schema_keys(censored: Path) -> None:
    """An extra array would make every distance cell for this arm unreadable."""
    from benchmarks.real_data.eval_distance.schema import ENCODINGS_KEYS

    with np.load(censored, allow_pickle=False) as handle:
        assert set(handle.files) == set(ENCODINGS_KEYS)
