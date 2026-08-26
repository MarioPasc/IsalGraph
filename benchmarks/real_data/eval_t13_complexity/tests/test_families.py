"""Tests for :mod:`benchmarks.real_data.eval_t13_complexity.families`.

The load-bearing ones:

- :func:`test_every_grid_spec_builds_and_verifies` -- all 644 cells of the
  campaign grid, each checking its own closed-form ``|Aut|`` inside
  :func:`~...families.build`.
- :func:`test_symmetry_ladder_holds_n_and_m_exactly` -- the matched design's
  whole premise.
- :func:`test_prism_at_a4_is_the_cube` -- the ``CONTRACTS`` §3 defect, pinned so
  it cannot be "fixed" back.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from benchmarks.real_data.eval_t13_complexity import families, symmetry

_SEED = 13


@pytest.fixture(scope="module")
def grid() -> tuple[families.FamilySpec, ...]:
    """The campaign grid, frozen at the parameters `CONTRACTS` §3 fixes."""
    return families.enumerate_grid(sizes=families.SIZES, replicates=5, seed=_SEED)


# ----------------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------------


def test_families_tuple() -> None:
    """CONTRACTS §3's ten, plus ``spider_ladder``.

    The eleventh was added mid-wave on the orchestrator's instruction: the two
    original ladder bases with factorial groups are dense (``K_{a,a}``) or
    medium (``Q_d``), and a cost law demonstrated only there would not transfer
    to the sparse IAM cohort.
    """
    assert families.FAMILIES == (
        "path",
        "cycle",
        "star",
        "complete",
        "complete_bipartite",
        "hypercube",
        "prism",
        "caterpillar",
        "rigid_er",
        "symmetry_ladder",
        "spider_ladder",
    )


def test_enumerate_grid_is_deterministic(grid: tuple[families.FamilySpec, ...]) -> None:
    again = families.enumerate_grid(sizes=families.SIZES, replicates=5, seed=_SEED)
    assert again == grid


def test_enumerate_grid_is_deduplicated(grid: tuple[families.FamilySpec, ...]) -> None:
    """Twelve requested sizes collapse to four hypercubes, not eight duplicates."""
    assert len(set(grid)) == len(grid)
    assert sum(1 for s in grid if s.family == "hypercube") == 4
    assert {s.n for s in grid if s.family == "hypercube"} == {8, 16, 32, 64}


def test_enumerate_grid_covers_every_family(grid: tuple[families.FamilySpec, ...]) -> None:
    assert {s.family for s in grid} == set(families.FAMILIES)


def test_enumerate_grid_replicates_only_the_random_families(
    grid: tuple[families.FamilySpec, ...],
) -> None:
    """Deterministic families get one replicate; a ladder's rung 0 collapses too."""
    deterministic = {
        "path",
        "cycle",
        "star",
        "complete",
        "complete_bipartite",
        "hypercube",
        "prism",
        "caterpillar",
    }
    assert all(s.replicate == 0 for s in grid if s.family in deterministic)
    assert {s.replicate for s in grid if s.family == "rigid_er"} == {0, 1, 2, 3, 4}
    rung_zero = [s for s in grid if s.family == "symmetry_ladder" and dict(s.params)["swaps"] == 0]
    assert all(s.replicate == 0 for s in rung_zero)


def test_enumerate_grid_rejects_zero_replicates() -> None:
    with pytest.raises(ValueError, match="replicates"):
        families.enumerate_grid(sizes=(8,), replicates=0, seed=_SEED)


def test_family_specs_are_hashable_and_frozen(grid: tuple[families.FamilySpec, ...]) -> None:
    spec = grid[0]
    assert hash(spec) == hash(spec)
    with pytest.raises((AttributeError, TypeError)):
        spec.n = 99  # type: ignore[misc]


# ----------------------------------------------------------------------------
# build: every cell verifies its own closed form
# ----------------------------------------------------------------------------


def test_every_grid_spec_builds_and_verifies(grid: tuple[families.FamilySpec, ...]) -> None:
    """All 644 cells build, are connected, and match their closed form.

    ``build`` raises :class:`~...families.FamilyVerificationError` on a
    ``|Aut|`` mismatch, so a green run here *is* the closed-form verification;
    this test only adds the structural checks the formula does not cover.
    """
    for spec in grid:
        graph = families.build(spec, seed=_SEED)
        assert graph.number_of_nodes() == spec.n, spec
        assert nx.is_connected(graph), spec
        assert set(graph.nodes()) == set(range(spec.n)), spec


def test_build_is_a_pure_function_of_spec_and_seed(
    grid: tuple[families.FamilySpec, ...],
) -> None:
    """Same spec and seed must give the same edge set, or a shard cannot rebuild it."""
    for spec in grid:
        if spec.family not in {"rigid_er", "symmetry_ladder"}:
            continue
        first = sorted(map(sorted, families.build(spec, seed=_SEED).edges()))
        second = sorted(map(sorted, families.build(spec, seed=_SEED).edges()))
        assert first == second, spec


def test_build_rejects_an_unknown_family() -> None:
    spec = families.FamilySpec("not_a_family", 8, 0, (), None)
    with pytest.raises(families.FamilyVerificationError, match="unknown family"):
        families.build(spec, seed=_SEED)


def test_build_raises_when_the_closed_form_is_wrong() -> None:
    """A tampered expectation must abort, not be recorded.

    `T-13-design.md` §3 rule 6: a constructed graph whose measured ``|Aut|``
    disagrees with its formula is a construction bug.  Recording it would put a
    mislabelled point on the experiment's primary axis.
    """
    honest = families.FamilySpec("cycle", 12, 0, (), math.log10(24.0))
    families.build(honest, seed=_SEED)  # sanity: the honest one passes
    tampered = families.FamilySpec("cycle", 12, 0, (), math.log10(25.0))
    with pytest.raises(families.FamilyVerificationError, match="closed form"):
        families.build(tampered, seed=_SEED)


def test_build_raises_when_n_is_wrong() -> None:
    spec = families.FamilySpec("hypercube", 12, 0, (("dimension", 3),), None)
    with pytest.raises(families.FamilyVerificationError, match="vertices"):
        families.build(spec, seed=_SEED)


@pytest.mark.parametrize(
    ("family", "n", "expected"),
    [
        ("path", 16, math.log10(2.0)),
        ("cycle", 16, math.log10(32.0)),
        ("star", 16, math.log10(math.factorial(15))),
        ("complete", 16, math.log10(math.factorial(16))),
        ("complete_bipartite", 16, math.log10(2 * math.factorial(8) ** 2)),
    ],
)
def test_closed_forms_at_n16(family: str, n: int, expected: float) -> None:
    """The formulas as the design note tabulates them, measured by nauty."""
    spec = next(
        s
        for s in families.enumerate_grid(sizes=(n,), replicates=1, seed=_SEED)
        if s.family == family
    )
    assert spec.log10_aut_expected == pytest.approx(expected, abs=1e-9)
    assert symmetry.log10_aut(families.build(spec, seed=_SEED)) == pytest.approx(
        expected, abs=families.AUT_TOLERANCE
    )


def test_prism_at_a4_is_the_cube() -> None:
    """``CONTRACTS`` §3 names ``a = 3`` as the prism exception.  It is ``a = 4``.

    Measured: the 3-prism ``C_3 x K_2`` has ``|Aut| = 12 = 4 * 3``, exactly what
    the ``4a`` formula predicts.  The 4-prism ``C_4 x K_2`` **is** the cube
    ``Q_3`` and has ``|Aut| = 48``, not ``16``.  The contract's exclusion is
    therefore both unnecessary at ``a = 3`` and misdirected: excluding ``a = 3``
    while keeping ``a = 4`` under the ``4a`` formula would have aborted the
    campaign at ``n = 8``.
    """
    assert symmetry.log10_aut(nx.circular_ladder_graph(3)) == pytest.approx(
        math.log10(12.0), abs=1e-9
    )
    assert symmetry.log10_aut(nx.circular_ladder_graph(4)) == pytest.approx(
        math.log10(48.0), abs=1e-9
    )
    assert nx.is_isomorphic(nx.circular_ladder_graph(4), nx.hypercube_graph(3))

    spec = next(
        s
        for s in families.enumerate_grid(sizes=(8,), replicates=1, seed=_SEED)
        if s.family == "prism"
    )
    assert spec.n == 8
    assert spec.log10_aut_expected == pytest.approx(math.log10(48.0), abs=1e-9)
    families.build(spec, seed=_SEED)


def test_caterpillar_is_two_to_the_k() -> None:
    """``|Aut| = 2^k`` with no mirror factor, because the leaf run is not a palindrome."""
    for spec in families.enumerate_grid(sizes=families.SIZES, replicates=1, seed=_SEED):
        if spec.family != "caterpillar":
            continue
        doubles = dict(spec.params)["doubles"]
        spine = dict(spec.params)["spine"]
        assert 0 < doubles < spine
        assert spec.n == spine + 2 * doubles
        graph = families.build(spec, seed=_SEED)
        assert nx.is_tree(graph)
        assert symmetry.log10_aut(graph) == pytest.approx(
            doubles * math.log10(2.0), abs=families.AUT_TOLERANCE
        )


def test_rigid_er_is_rigid_and_connected(grid: tuple[families.FamilySpec, ...]) -> None:
    for spec in grid:
        if spec.family != "rigid_er":
            continue
        graph = families.build(spec, seed=_SEED)
        assert nx.is_connected(graph)
        assert symmetry.log10_aut(graph) == pytest.approx(0.0, abs=families.AUT_TOLERANCE)


def test_rigid_er_aborts_rather_than_returning_a_symmetric_graph() -> None:
    """At a density where no rigid draw exists, the run stops.

    ``G(4, 0.01)`` is almost surely empty or a single edge, neither connected
    nor rigid, so the resampling budget is exhausted and
    :class:`~...families.FamilyVerificationError` is raised.  The alternative --
    returning the last draw -- would put a symmetric graph in the rigid control.
    """
    spec = families.FamilySpec("rigid_er", 4, 0, (("p_percent", 1),), None)
    with pytest.raises(families.FamilyVerificationError, match="rigid"):
        families.build(spec, seed=_SEED)


# ----------------------------------------------------------------------------
# symmetry_ladder -- the matched design
# ----------------------------------------------------------------------------


def _ladder(grid: tuple[families.FamilySpec, ...], n: int, base: str) -> list[nx.Graph]:
    """Every rung of one ladder, in swap order, one replicate."""
    base_index = families.LADDER_BASES.index(base)
    specs = [
        s
        for s in grid
        if s.family == "symmetry_ladder"
        and s.n == n
        and dict(s.params)["base"] == base_index
        and s.replicate in (0,)
    ]
    specs.sort(key=lambda s: dict(s.params)["swaps"])
    return [families.build(s, seed=_SEED) for s in specs]


@pytest.mark.parametrize(("n", "base"), [(16, "complete_bipartite"), (32, "hypercube")])
def test_symmetry_ladder_holds_n_and_m_exactly(
    grid: tuple[families.FamilySpec, ...], n: int, base: str
) -> None:
    """The premise of the matched design, asserted as `CONTRACTS` §3 words it."""
    ladder = _ladder(grid, n, base)
    assert len(ladder) == len(families.LADDER_SWAPS)
    assert len({(g.number_of_nodes(), g.number_of_edges()) for g in ladder}) == 1
    assert all(nx.is_connected(g) for g in ladder)


def test_symmetry_ladder_holds_the_whole_degree_sequence(
    grid: tuple[families.FamilySpec, ...],
) -> None:
    """Stronger than ``(n, m)``: the swap is degree-preserving, so the maximum
    degree is held fixed too and cannot be the hidden cause of any time trend."""
    for n, base in ((16, "complete_bipartite"), (64, "hypercube")):
        ladder = _ladder(grid, n, base)
        sequences = {tuple(sorted(d for _v, d in g.degree())) for g in ladder}
        assert len(sequences) == 1, (n, base)


def test_symmetry_ladder_never_uses_a_two_regular_base() -> None:
    """A cycle base has no rungs: every connectivity-preserving degree-preserving
    swap on a 2-regular graph either disconnects it or returns a cycle."""
    assert "cycle" not in families.LADDER_BASES
    for base in families.LADDER_BASES:
        graph = families._ladder_base(base, 16)
        assert min(d for _v, d in graph.degree()) >= 3


def test_symmetry_ladder_keeps_non_monotone_rungs(
    grid: tuple[families.FamilySpec, ...],
) -> None:
    """Every requested rung is present, whatever its ``|Aut|`` did.

    Dropping a rung whose ``|Aut|`` failed to fall would select on the outcome
    and make the ladder's own trend an artefact of the filter.  The check is
    structural -- all seven swap counts survive to the grid at every ladder --
    because a value-based check would itself be outcome-dependent.
    """
    by_ladder: dict[tuple[int, int], set[int]] = {}
    for spec in grid:
        if spec.family != "symmetry_ladder":
            continue
        params = dict(spec.params)
        by_ladder.setdefault((spec.n, params["base"]), set()).add(params["swaps"])
    assert by_ladder
    for key, swaps in by_ladder.items():
        assert swaps == set(families.LADDER_SWAPS), key

    # And at least one ladder is in fact non-monotone across consecutive rungs,
    # so the rule is not vacuous.
    ladder = _ladder(grid, 8, "complete_bipartite")
    values = [symmetry.log10_aut(g) for g in ladder]
    assert any(b >= a for a, b in zip(values, values[1:], strict=True))


def test_symmetry_ladder_spans_are_measured(grid: tuple[families.FamilySpec, ...]) -> None:
    """Most ladders fall at least three orders of magnitude; three do not.

    The short ones are ``K_{4,4}`` at ``n = 8`` (2.459), ``Q_3`` at ``n = 8``
    (1.079) and ``Q_4`` at ``n = 16`` (2.584), and the cause is arithmetic
    rather than a defect: their rung-0 groups are only ``1152``, ``48`` and
    ``384``, so three orders were never available.  They are kept because the
    low-``|Aut|`` end of the ladder is still a valid contrast at fixed
    ``(n, m)``; the analysis must not read a slope off them alone.
    """
    spans = families.ladder_spans(grid, seed=_SEED)
    assert len(spans) == 16 + len(families.SPIDER_CELLS)
    short = {key for key, span in spans.items() if span < 3.0}
    assert short == {
        (8, "complete_bipartite"),
        (8, "hypercube"),
        (16, "hypercube"),
        (10, "spider_k3"),  # the n <= 12 consistency gate; span log10(3!) = 0.778
    }
    assert spans[(64, "complete_bipartite")] > 60.0
    assert spans[(61, "spider_k10")] == pytest.approx(math.log10(math.factorial(10)), abs=1e-9)


def test_symmetry_ladder_rung_zero_is_the_closed_form_base(
    grid: tuple[families.FamilySpec, ...],
) -> None:
    """Rung 0 must be the untouched base, so the ladder starts from a known ``|Aut|``."""
    for n, base, expected in (
        (16, "complete_bipartite", math.log10(2 * math.factorial(8) ** 2)),
        (32, "hypercube", math.log10(2**5 * math.factorial(5))),
    ):
        base_index = families.LADDER_BASES.index(base)
        spec = next(
            s
            for s in grid
            if s.family == "symmetry_ladder"
            and s.n == n
            and dict(s.params)["base"] == base_index
            and dict(s.params)["swaps"] == 0
        )
        graph = families.build(spec, seed=_SEED)
        assert symmetry.log10_aut(graph) == pytest.approx(expected, abs=families.AUT_TOLERANCE)


# ----------------------------------------------------------------------------
# spider_ladder -- the sparse matched design
# ----------------------------------------------------------------------------


def _spider(grid: tuple[families.FamilySpec, ...], k: int, leg: int) -> list[families.FamilySpec]:
    """Every rung of one spider ladder, in rung order."""
    specs = [
        s
        for s in grid
        if s.family == "spider_ladder"
        and dict(s.params)["legs"] == k
        and dict(s.params)["leg"] == leg
    ]
    specs.sort(key=lambda s: dict(s.params)["rung"])
    return specs


@pytest.mark.parametrize(("k", "leg"), families.SPIDER_CELLS)
def test_spider_ladder_holds_n_m_and_the_degree_sequence(
    grid: tuple[families.FamilySpec, ...], k: int, leg: int
) -> None:
    """The three confounds, fixed by construction rather than by a swap.

    ``n = 1 + sum L_i`` is invariant because :func:`~...families.spider_legs`
    displaces lengths antisymmetrically; ``m = n - 1`` because every spider is a
    tree; and the degree sequence is ``(k, 2^(n-1-k), 1^k)`` whatever the leg
    partition, because the hub always has degree ``k``, there are always ``k``
    leaves, and ``sum_i (L_i - 1) = (n - 1) - k`` does not depend on how the
    lengths are distributed.
    """
    ladder = [families.build(s, seed=_SEED) for s in _spider(grid, k, leg)]
    assert len(ladder) == len(families.spider_rungs(k, leg)) >= 2

    assert len({(g.number_of_nodes(), g.number_of_edges()) for g in ladder}) == 1
    assert all(nx.is_connected(g) for g in ladder)
    assert all(nx.is_tree(g) for g in ladder)

    sequences = {tuple(sorted((d for _v, d in g.degree()), reverse=True)) for g in ladder}
    assert len(sequences) == 1
    n = ladder[0].number_of_nodes()
    assert sequences.pop() == tuple([k] + [2] * (n - 1 - k) + [1] * k)


@pytest.mark.parametrize(("k", "leg"), families.SPIDER_CELLS)
def test_spider_ladder_matches_its_closed_form(
    grid: tuple[families.FamilySpec, ...], k: int, leg: int
) -> None:
    """``|Aut| = prod_d (m_d)!``, and rung ``j`` leaves ``k - 2j`` legs equal.

    Unlike ``symmetry_ladder``, whose ``|Aut|`` can only be measured, every
    spider rung is predicted exactly -- so ``build`` verifies this ladder the
    same way it verifies ``complete``.
    """
    for spec in _spider(grid, k, leg):
        rung = dict(spec.params)["rung"]
        lengths = families.spider_legs(k, leg, rung)
        assert sum(lengths) == k * leg
        assert min(lengths) >= 1
        assert spec.log10_aut_expected == pytest.approx(
            math.log10(math.factorial(k - 2 * rung)), abs=1e-9
        )
        graph = families.build(spec, seed=_SEED)
        assert symmetry.log10_aut(graph) == pytest.approx(
            spec.log10_aut_expected, abs=families.AUT_TOLERANCE
        )


def test_spider_ladder_is_the_sparse_arm(grid: tuple[families.FamilySpec, ...]) -> None:
    """Its density is far below both regular ladders', which is why it exists.

    A tree is the sparsest connected graph, so the three ladders give the
    matched design three separated densities and let the analysis answer "is it
    really density?" instead of conceding it.
    """

    def density(spec: families.FamilySpec) -> float:
        return 2.0 * (spec.n - 1) / (spec.n * (spec.n - 1))

    spiders = [s for s in grid if s.family == "spider_ladder"]
    assert spiders
    assert max(density(s) for s in spiders) <= 0.20
    # K_{32,32} at n = 64 for contrast: m = 1024, density ~0.508.
    assert 2.0 * 1024 / (64 * 63) > 0.5


def test_spider_legs_rejects_a_two_legged_spider() -> None:
    """At ``k = 2`` the hub has degree 2, the spider is a path, and ``|Aut| = 2``
    however unequal the legs are -- so the ladder would have no rungs."""
    with pytest.raises(families.FamilyVerificationError, match="k >= 3"):
        families.spider_legs(2, 4, 0)
    path = nx.path_graph(9)
    assert symmetry.log10_aut(path) == pytest.approx(math.log10(2.0), abs=1e-9)


def test_spider_legs_rejects_a_rung_that_would_empty_a_leg() -> None:
    assert families.spider_rungs(8, 4) == (0, 1, 2, 3)
    assert families.spider_rungs(8, 8) == (0, 1, 2, 3, 4)
    assert families.spider_rungs(10, 3) == (0, 1, 2)
    with pytest.raises(families.FamilyVerificationError, match="out of range"):
        families.spider_legs(8, 4, 4)  # would need a leg of length 0


def test_spider_ladder_reaches_rigid_where_the_legs_allow_it(
    grid: tuple[families.FamilySpec, ...],
) -> None:
    """When ``leg > k // 2`` the last rung makes every leg length distinct."""
    for k, leg in ((10, 6), (8, 8)):
        last = _spider(grid, k, leg)[-1]
        assert last.log10_aut_expected == pytest.approx(0.0, abs=1e-12)
        lengths = families.spider_legs(k, leg, dict(last.params)["rung"])
        assert len(set(lengths)) == k


def test_ladder_base_rejects_an_unrealisable_order() -> None:
    with pytest.raises(families.FamilyVerificationError, match="power of two"):
        families._ladder_base("hypercube", 12)
    with pytest.raises(families.FamilyVerificationError, match="even"):
        families._ladder_base("complete_bipartite", 9)
    with pytest.raises(families.FamilyVerificationError, match="unknown ladder base"):
        families._ladder_base("cycle", 16)
