"""T-04a annex E2: completeness -- metric or pseudometric?

E1 asks whether ``d_R(G, π(G)) = 0``.  E2 asks the **other** direction,
``d_R(G,H) = 0 ⇒ G ≅ H``, and the two failures are opposite defects: the
first makes the distance not a function on isomorphism classes at all, the
second makes it a pseudometric rather than a metric.  Only the second decides
whether a representation can support a claim that *distance zero certifies
isomorphism*, which is the axis on which IsalGraph's complete-invariant
theorem is the contribution.

What each test defends:

- **The witness is a proof, not a sample.**  ``K₃,₃`` and the triangular prism
  are both connected, both 3-regular on six vertices, and not isomorphic.
  1-WL leaves the colouring constant after round 1, so refinement never
  starts.  ``test_the_witness_is_3_regular_non_isomorphic_on_six_nodes`` and
  ``test_wl_fails_the_witness_and_every_complete_invariant_separates_it``
  pin both halves.  No number of relabellings or datasets can overturn a
  single exhibited collision, which is why the class table treats it as
  decisive.
- **The converse direction is escalated, never reported.**  An invariant
  representation cannot separate two isomorphic graphs -- its encoding is a
  function of the isomorphism class -- so a violation is a defect in our code.
  ``test_converse_check_raises_rather_than_reporting`` is acceptance
  criterion 5.
- **VF2 is the only verdict.**  A nauty certificate keeps the VF2 cost linear
  in the draw, but it never decides.
  ``test_the_oracle_rejects_a_lying_certificate`` feeds the oracle a
  certificate that groups ``K₃,₃`` with the prism and asserts the oracle still
  answers "not isomorphic".
- **A zero is never printed bare.**  ``test_a_zero_count_reports_the_rule_of_three``
  -- ``0/N`` asserts impossibility from a finite sample; ``3/N`` is what the
  sample licenses.

Synthetic graphs wherever the assertion does not genuinely need the real
cohorts: those live on an external drive, and a slow test is a test that gets
skipped.  The cohort-backed tests carry ``@pytest.mark.integration``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.admissibility import common
from isalgraph.competitors.admissibility import e2_completeness as e2
from isalgraph.competitors.base import Capability
from isalgraph.competitors.registry import available_backends, get_backend, get_metric

if TYPE_CHECKING:
    import networkx as nx

nx = pytest.importorskip("networkx")

#: A grid view that needs no grid file: every serialisation falls back to
#: Levenshtein and the one feature-vector representation keeps its kernel.
SYNTHETIC_GRID = e2.GridView(
    path="synthetic",
    primary={"wl_subtree": "kernel"},
    f3={},
    class_i=frozenset(),
)


def _complete_invariants() -> list[str]:
    """Pool members declaring :attr:`Capability.COMPLETE_INVARIANT`."""
    return [
        name
        for name in available_backends()
        if Capability.COMPLETE_INVARIANT in get_backend(name).capabilities
    ]


def _grid_backends() -> frozenset[str]:
    """Backends covered by the **frozen** T-04a admissibility grid.

    Part C classifies a representation from its F3 record in that grid, so a
    backend added to the registry afterwards has no record and is reported
    ``class = None`` with a reason -- which is the correct answer, not a
    defect.  The grid is a pre-registered artifact and is not regenerated to
    absorb a new arm; the assertion is scoped to what the grid covers instead.
    """
    import json

    with open(e2.DEFAULT_GRID, encoding="utf-8") as handle:
        return frozenset(json.load(handle).get("backends", ()))


# --------------------------------------------------------------------------
# Part A -- the witness
# --------------------------------------------------------------------------


def test_the_witness_is_3_regular_non_isomorphic_on_six_nodes() -> None:
    """The fixture's premises, asserted rather than trusted.

    Every claim E2 makes about ``K₃,₃`` versus the prism rests on these four
    facts.  If the fixture ever drifts, the witness stops being a witness and
    the qualitative claim silently becomes an unsupported one.
    """
    left = fixtures.to_networkx(fixtures.K33)
    right = fixtures.to_networkx(fixtures.PRISM)

    assert left.number_of_nodes() == right.number_of_nodes() == 6
    assert left.number_of_edges() == right.number_of_edges() == 9
    assert {d for _, d in left.degree()} == {3}
    assert {d for _, d in right.degree()} == {3}
    assert nx.is_connected(left) and nx.is_connected(right)
    assert not nx.is_isomorphic(left, right)


def test_wl_fails_the_witness_and_every_complete_invariant_separates_it() -> None:
    """Acceptance criterion 4, in one assertion per direction.

    ``wl_subtree`` must give the pair distance **exactly** zero -- not merely
    a small distance -- because 1-WL's colour histogram is identical on two
    regular graphs of the same degree and order.  Every backend declaring
    :attr:`Capability.COMPLETE_INVARIANT` must give a strictly positive
    distance, since equal encodings would contradict the declaration.
    """
    record = e2.separation_witness(available_backends(), SYNTHETIC_GRID)

    assert record["both_3_regular"] is True
    assert record["non_isomorphic_vf2"] is True

    wl = record["representations"]["wl_subtree"]
    assert wl["metric"] == "kernel"
    assert wl["distance"] == 0.0
    assert wl["separates"] is False

    complete = _complete_invariants()
    assert complete, "the pool must contain at least one declared complete invariant"
    for name in complete:
        entry = record["representations"][name]
        assert entry.get("error") is None, f"{name} failed to encode the witness: {entry}"
        assert entry["distance"] > e2.ZERO_TOL, f"{name} does not separate K33 from the prism"
        assert entry["separates"] is True


# --------------------------------------------------------------------------
# The isomorphism oracle -- VF2 decides, the certificate only groups
# --------------------------------------------------------------------------


def test_the_oracle_groups_relabelled_copies_and_certifies_them_with_vf2() -> None:
    """Equal certificates are certified, not believed."""
    rng = random.Random(common.SEED)
    left = fixtures.to_networkx(fixtures.K33)
    right = fixtures.shuffled_copy(left, rng)

    oracle = e2.IsomorphismOracle(graphs=[left, right], keys=["same", "same"])

    assert oracle.certificate_defects == []
    assert oracle.n_classes == 1
    assert oracle.are_isomorphic(0, 1) is True
    assert oracle.iso_pairs() == [(0, 1)]
    assert oracle.vf2_calls >= 1, "the class was certified by VF2, not asserted"


def test_the_oracle_rejects_a_lying_certificate() -> None:
    """A certificate that groups two non-isomorphic graphs does not decide.

    This is the guard that keeps ``pynauty`` a pre-filter.  If the oracle ever
    returns the certificate's answer, this test flips to ``True`` and every
    collision count in E2 silently becomes a count of certificate agreements.
    """
    left = fixtures.to_networkx(fixtures.K33)
    right = fixtures.to_networkx(fixtures.PRISM)

    oracle = e2.IsomorphismOracle(graphs=[left, right], keys=["lie", "lie"])

    assert oracle.certificate_defects == [(0, 1)]
    assert oracle.n_classes == 2
    assert oracle.are_isomorphic(0, 1) is False
    assert oracle.iso_pairs() == []


def test_a_missing_certificate_becomes_a_singleton_and_still_gets_a_verdict() -> None:
    """A backend refusal must not silently merge or drop a graph."""
    rng = random.Random(common.SEED)
    left = fixtures.to_networkx(fixtures.K33)
    right = fixtures.shuffled_copy(left, rng)

    oracle = e2.IsomorphismOracle(graphs=[left, right], keys=[None, None])

    assert oracle.n_classes == 2
    assert oracle.iso_pairs() == []
    assert oracle.are_isomorphic(0, 1) is True


# --------------------------------------------------------------------------
# The converse direction -- escalated, never reported
# --------------------------------------------------------------------------


def test_converse_check_raises_rather_than_reporting() -> None:
    """Acceptance criterion 5.

    An invariant representation separating two isomorphic graphs is a defect
    in our code, not a property of the method, so protocol §7 escalates it.
    Returning a count instead would let the defect land in a results table.
    """
    with pytest.raises(common.AdmissibilityError, match="certified isomorphic"):
        e2.converse_check("saboteur", [(0, 1)], lambda i, j: 3.0)


def test_converse_check_counts_only_evaluable_pairs() -> None:
    """A pair the backend refused is skipped, never counted as a pass."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    checked = e2.converse_check("ok", pairs, lambda i, j: None if j == 2 else 0.0)
    assert checked == 1


def test_converse_check_tolerates_nothing_above_the_float_epsilon() -> None:
    """The tolerance exists for float noise, not for a real separation."""
    assert e2.converse_check("ok", [(0, 1)], lambda i, j: e2.ZERO_TOL) == 1
    with pytest.raises(common.AdmissibilityError):
        e2.converse_check("bad", [(0, 1)], lambda i, j: e2.ZERO_TOL * 10)


# --------------------------------------------------------------------------
# Zero pairs, rates and the class table
# --------------------------------------------------------------------------


def test_zero_pairs_finds_relabelled_copies_and_reports_its_denominator() -> None:
    """The zero set of a complete invariant is exactly its isomorphic pairs."""
    rng = random.Random(common.SEED)
    left = fixtures.to_networkx(fixtures.K33)
    graphs = [left, fixtures.shuffled_copy(left, rng), fixtures.to_networkx(fixtures.PRISM)]

    items, failures = e2.encode_all("nauty_graph6", graphs)
    assert failures == {}
    pairs, evaluated, undefined, near_zero = e2.zero_pairs(items, get_metric("levenshtein"))

    assert pairs == [(0, 1)]
    assert evaluated == 3
    assert undefined == 0
    assert near_zero == 0, "both admissible distances are exact; the tolerance must not bind"


def test_encode_all_reports_a_refusal_instead_of_dropping_the_graph() -> None:
    """``None`` keeps the encodings aligned with the draw.

    Dropping a refusal shifts every later index, which would silently
    misattribute a collision witness to the wrong pair of graphs.
    """
    small = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    big = nx.path_graph(40)
    items, failures = e2.encode_all("isalgraph_canonical", [small, big, small])

    assert len(items) == 3
    assert items[0] is not None and items[2] is not None
    assert items[1] is None, "a SUITE1_ONLY backend must refuse a 40-node graph"
    assert sum(failures.values()) == 1


def test_a_zero_count_reports_the_rule_of_three() -> None:
    """``0/N`` is an upper bound, never the rate ``0``."""
    record = e2._rate(0, 500)
    assert record["events"] == 0
    assert record["rule_of_three_upper"] == pytest.approx(common.rule_of_three(500))
    assert "rule of three" in record["reported"]
    assert record["reported"] != "0"

    observed = e2._rate(3, 500)
    assert "rule_of_three_upper" not in observed
    lo, hi = observed["ci95_clopper_pearson"]
    assert 0.0 < lo < 3 / 500 < hi < 1.0


def test_an_exhibited_collision_makes_a_representation_class_two() -> None:
    """One witness outranks a sampled rate of zero.

    A representation that fails ``K₃,₃`` versus the prism is a pseudometric on
    isomorphism classes as a matter of proof.  If the class table let a clean
    sample overturn that, D-A2's scope limit -- class II is barred from any
    ``d = 0`` certifies isomorphism claim -- would stop being enforced by the
    data.
    """
    witness = {"representations": {"pseudo": {"separates": False}, "real": {"separates": True}}}
    pooled = {
        "pseudo": {
            "collisions": 0,
            "zero_pairs": 10,
            "collision_rate_among_zero": e2._rate(0, 10),
            "collision_rate_among_pairs": e2._rate(0, 100),
        },
        "real": {
            "collisions": 0,
            "zero_pairs": 10,
            "collision_rate_among_zero": e2._rate(0, 10),
            "collision_rate_among_pairs": e2._rate(0, 100),
        },
    }
    grid = e2.GridView(
        path="synthetic",
        primary={"pseudo": "kernel", "real": "levenshtein", "ordered": None},
        f3={"ordered": {"levenshtein": "1/50"}},
        class_i=frozenset({"ordered"}),
    )

    table = e2.classify(["pseudo", "real", "ordered"], grid, witness, pooled)

    assert table["pseudo"]["class"] == "II"
    assert "PSEUDOMETRIC" in table["pseudo"]["reason"]
    assert table["real"]["class"] == "III"
    assert "rule-of-three" in table["real"]["reason"]
    assert table["ordered"]["class"] == "I"


def test_class_one_comes_from_the_grid_and_is_not_recomputed() -> None:
    """F3 is E1's measurement; E2 reads it and asks the other question."""
    grid = e2.GridView(
        path="synthetic",
        primary={"a": None, "b": "levenshtein"},
        f3={"a": {"levenshtein": "1/50", "hamming": "1/50"}, "b": {"levenshtein": "50/50"}},
        class_i=frozenset(),
    )
    table = e2.classify(["a"], grid, {"representations": {}}, {})

    assert table["a"]["class"] is None, "class I is decided by GridView.class_i, not by classify"
    assert table["a"]["grid_f3"] == {"levenshtein": "1/50", "hamming": "1/50"}


# --------------------------------------------------------------------------
# Cohort-backed
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_load_grid_marks_the_n2_family_class_one() -> None:
    """The grid's own F3 puts ``adjacency``/``graph6``/``sparse6`` in class I.

    The ``size_null`` *metric* is invariant for every representation, so a
    reader of raw F3 cells that forgets to drop the non-candidate ones
    concludes the order-dependent family is invariant.  ``load_grid`` keeps
    candidate cells only.
    """
    import os

    if not os.path.exists(e2.DEFAULT_GRID):
        pytest.skip(f"grid not present at {e2.DEFAULT_GRID}")
    grid = e2.load_grid(e2.DEFAULT_GRID)

    assert {"adjacency", "graph6", "sparse6"} <= grid.class_i
    assert not ({"nauty_graph6", "isalgraph_pruned", "wl_subtree"} & grid.class_i)
    assert grid.primary["isalgraph_pruned"] == "levenshtein"
    assert grid.primary["wl_subtree"] == "kernel"
    assert "size_null" not in grid.f3.get("adjacency", {})


@pytest.mark.integration
def test_quick_run_classifies_every_representation(tmp_path: Path) -> None:
    """End-to-end on two cohorts: the smoke path, never a result."""
    import json
    import os

    if not datasets.available_datasets():
        pytest.skip("cohorts not mounted")
    if not os.path.exists(e2.DEFAULT_GRID):
        pytest.skip(f"grid not present at {e2.DEFAULT_GRID}")

    out = str(tmp_path / "e2_quick.json")
    assert e2.main(["--out", out, "--quick", "--log-level", "WARNING"]) == 0

    with open(out, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["experiment"] == "E2"
    assert payload["seed"] == common.SEED
    assert payload["quick"] is True
    assert payload["escalations"] == []
    assert payload["part_a_witness"]["representations"]["wl_subtree"]["separates"] is False
    assert payload["part_c_classes"]["wl_subtree"]["class"] == "II"
    covered = _grid_backends()
    classified = [name for name in _complete_invariants() if name in covered]
    assert classified, "no complete invariant is covered by the frozen grid; the test is vacuous"
    for name in classified:
        assert payload["part_c_classes"][name]["class"] == "III"

    # A complete invariant registered after the grid was frozen is unclassified,
    # and must say so rather than defaulting into a class it has no evidence for.
    for name in _complete_invariants():
        if name in covered:
            continue
        entry = payload["part_c_classes"][name]
        assert entry["class"] is None
        assert "not classified" in entry["reason"]
