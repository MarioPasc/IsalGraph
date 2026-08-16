"""The frozen confirmatory family --- F0, F1, F2 --- and ``N_actual``.

``preregistration.md`` fixes ``N_max = 197`` across three families run in fixed
sequence (Dmitrienko, Tamhane & Bretz, *Multiple Testing Problems in
Pharmaceutical Statistics*, CRC Press, 2009, ch. 5), with BH-FDR at
``q = 0.05`` applied **within** each. Two of the protocol's pre-declared rules
are gates: their outcome decides which downstream tests are admissible, so
placing them inside the family they gate would make its cardinality a function
of a test inside it.

======  ==================================  =====
Family  Content                             Tests
======  ==================================  =====
F0      calibration gate, per Suite-1 set       5
F1      bracket gate (D13), per Suite-2 set    10
F2      primary: Claims A and B                182
======  ==================================  =====

``N_actual(F2)`` is **defined by enumerating the admissible cell set**, with a
closed form printed beside it as a check. Where the two disagree the enumeration
wins and the discrepancy is reported (``preregistration.md`` section 5).

The precedence is **F0-demotion (stage 0) -> k -> d -> c**, and no cell removed
by an earlier stage is ever charged again by a later one::

    ordinary branch:        N_actual = 182 - 15 k - 8 d + k d - c
    F0 majority fired:      N_actual = 101 -  5 k        - c

The ``+ k d`` term corrects an overlap the freeze originally missed: ``k``
removes ``(B1a, R, D)`` for all ten Suite-2 datasets and ``d`` removes
``(B1a, R', D)`` for all seven comparators, so the ``k d`` cells with ``R``
excluded and ``D`` uninformative sit in both removal sets. That overlap is
complete --- ``A1`` is untouched by either term, ``B1e`` is indexed by Suite-1
datasets so ``d`` cannot reach it, and ``B3a`` carries no representation index
so ``k`` cannot. Charging those cells twice reports an ``N_actual`` **below**
the admissible count, which is the anti-conservative direction: it lowers the
BH burden on every surviving test.

The enumeration, the printed closed form,
:attr:`FamilyCardinality.discrepancy` and :attr:`FamilyCardinality.double_charged`
are kept even though the closed form is now correct. That machinery is what
caught the defect, and it must survive to catch the next one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_stats.multiplicity import (
    BenjaminiHochbergResult,
    FriedmanResult,
    Regime,
    benjamini_hochberg,
    fcr_adjusted_level,
    friedman_omnibus,
)
from benchmarks.real_data.eval_stats.resampling import (
    FDR_Q,
    FloatArray,
    PercentileInterval,
    bootstrap_p_value,
    percentile_interval,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CLAIM_A_REPRESENTATIONS",
    "CLAIM_B_REPRESENTATIONS",
    "GATE_THRESHOLD",
    "N_AFTER_F0_DEMOTION",
    "N_MAX_F2",
    "N_MAX_TOTAL",
    "SUITE1",
    "SUITE2",
    "Cell",
    "FamilyCardinality",
    "FamilyError",
    "GateFamilyResult",
    "GateInput",
    "GateOutcome",
    "ReductionInputs",
    "admissible_cells",
    "cardinality",
    "enumerate_f2_cells",
    "evaluate_gate",
    "run_f0",
    "run_f1",
    "run_f2",
]


class FamilyError(Exception):
    """Raised when a family declaration is inconsistent with the freeze."""


#: CONTRACTS.md section 2, in this exact spelling and order.
SUITE2: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids_graphedx",
    "grec",
    "aids_iam",
    "coil_del",
    "mutagenicity",
    "protein",
)

#: Suite 1 applies ``n_max = 12``; ``aids`` here is a different cohort from
#: ``aids_graphedx`` in Suite 2 (769 graphs against 819).
SUITE1: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids",
)

#: ``preregistration.md`` section 4.1, the frozen six.
CLAIM_A_REPRESENTATIONS: tuple[str, ...] = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
)

#: The frozen seven: the six above plus the WL subtree kernel distance. WL
#: enters Claim B and not Claim A --- it is not reversible and emits a feature
#: vector, so it has no bit count, but it does yield a distance.
CLAIM_B_REPRESENTATIONS: tuple[str, ...] = (*CLAIM_A_REPRESENTATIONS, "wl_subtree")

#: The pre-declared effect-size threshold in F0's and F1's branch rules.
GATE_THRESHOLD: float = 0.05

#: ``preregistration.md`` section 4.2.
N_MAX_F2: int = 182

#: F0 (5) + F1 (10) + F2 (182).
N_MAX_TOTAL: int = 197

#: Rows charged to ``c``: a cell is ``(row, representation, dataset)`` for
#: these rows only. A2, B2, B3e and B3a are never charged
#: (``preregistration.md`` section 5.1, consequence 3).
_C_CHARGEABLE_ROWS: frozenset[str] = frozenset({"A1", "B1e", "B1a"})

#: Rows a ``k`` exclusion removes: a representation with no admissible distance
#: loses its Claim B rows and keeps its Claim A rows, because a bit count needs
#: no distance.
_K_REMOVABLE_ROWS: frozenset[str] = frozenset({"B1e", "B1a"})

#: Rows a ``d`` exclusion removes, per uninformative Suite-2 dataset.
_D_REMOVABLE_ROWS: frozenset[str] = frozenset({"B1a", "B3a"})

#: Rows demoted to descriptive if F0 fails on a majority of its five tests.
_APPROXIMATE_ROWS: frozenset[str] = frozenset({"B1a", "B2", "B3a"})


# ---------------------------------------------------------------------------
# The cell set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Cell:
    """One member of F2.

    Attributes:
        row: ``A1``, ``A2``, ``B1e``, ``B1a``, ``B2``, ``B3e`` or ``B3a``.
        suite: ``"suite1"``, ``"suite2"``, or ``None`` for an omnibus.
        dataset: Dataset key, or ``None`` for an omnibus.
        representation: Comparator name, or ``None`` for an omnibus and for the
            MRM rows, which use the IsalGraph reference arm alone.
    """

    row: str
    suite: str | None
    dataset: str | None
    representation: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "row": self.row,
            "suite": self.suite,
            "dataset": self.dataset,
            "representation": self.representation,
        }


def enumerate_f2_cells() -> tuple[Cell, ...]:
    """Enumerate F2 explicitly, all 182 cells.

    ====  =====================================================  ======  =====
    Row   Test                                                   Layout  Tests
    ====  =====================================================  ======  =====
    A1    Wilcoxon signed-rank on bits, per dataset               6 x 10     60
    A2    Friedman omnibus on bits, across the ten datasets            1      1
    B1e   bootstrap CI on a rho difference, per Suite-1 dataset    7 x 5     35
    B1a   bootstrap CI on a rho difference, per Suite-2 dataset   7 x 10     70
    B2    Friedman omnibus on rho, approximate regime only             1      1
    B3e   MRM standardised beta1, per Suite-1 dataset                  5      5
    B3a   MRM standardised beta1, per Suite-2 dataset                 10     10
    ====  =====================================================  ======  =====

    Returns:
        Every F2 cell, in row order then dataset order then representation
        order.

    Raises:
        FamilyError: If the enumeration does not total :data:`N_MAX_F2`, which
            would mean a frozen constant above had been edited.
    """
    cells: list[Cell] = []
    cells.extend(
        Cell("A1", "suite2", dataset, rep) for dataset in SUITE2 for rep in CLAIM_A_REPRESENTATIONS
    )
    cells.append(Cell("A2", None, None, None))
    cells.extend(
        Cell("B1e", "suite1", dataset, rep) for dataset in SUITE1 for rep in CLAIM_B_REPRESENTATIONS
    )
    cells.extend(
        Cell("B1a", "suite2", dataset, rep) for dataset in SUITE2 for rep in CLAIM_B_REPRESENTATIONS
    )
    cells.append(Cell("B2", None, None, None))
    cells.extend(Cell("B3e", "suite1", dataset, None) for dataset in SUITE1)
    cells.extend(Cell("B3a", "suite2", dataset, None) for dataset in SUITE2)

    if len(cells) != N_MAX_F2:
        raise FamilyError(f"F2 enumerated {len(cells)} cells, expected {N_MAX_F2}")
    return tuple(cells)


@dataclass(frozen=True)
class ReductionInputs:
    """The three pre-declared reduction terms, plus F0's global branch.

    ``k`` and ``c`` arrive from other tickets; ``d`` is produced by this
    module's own F1 run.

    Attributes:
        excluded_representations: ``k``. Representations excluded by
            ``competitors.md`` section 3.4 --- no candidate distance passes F1
            at 100 %, F2, F3 and F4.
        uninformative_datasets: ``d``. Suite-2 datasets whose bracket F1
            declares uninformative.
        noncomputable: ``c``'s source. ``(suite, dataset, representation)``
            triples failing the >= 99 % completion criterion at the 300 s
            per-graph budget. Suite is part of the key because Suite 1 and
            Suite 2 are different cohorts even where the name matches.
        f0_demotes_approximate: ``True`` when F0 fails on a majority of its
            five tests. Section 5.3, frozen 2026-08-16: the 81 approximate-regime
            cells (B1a 70 + B2 1 + B3a 10) are demoted to descriptive, leaving
            the 101 exact-regime and Claim-A cells; ``d`` is then **not applied
            at all**, and ``k`` removes only its 5 B1e cells per representation.
    """

    excluded_representations: frozenset[str] = frozenset()
    uninformative_datasets: frozenset[str] = frozenset()
    noncomputable: frozenset[tuple[str, str, str]] = frozenset()
    f0_demotes_approximate: bool = False

    def validate(self) -> None:
        """Check every term against the frozen vocabularies.

        Raises:
            FamilyError: If a name is not a declared representation or dataset.
        """
        unknown_reps = set(self.excluded_representations) - set(CLAIM_B_REPRESENTATIONS)
        if unknown_reps:
            raise FamilyError(f"k names undeclared representations: {sorted(unknown_reps)}")
        unknown_sets = set(self.uninformative_datasets) - set(SUITE2)
        if unknown_sets:
            raise FamilyError(f"d names non-Suite-2 datasets: {sorted(unknown_sets)}")
        for suite, dataset, rep in self.noncomputable:
            pool = SUITE1 if suite == "suite1" else SUITE2
            if suite not in {"suite1", "suite2"} or dataset not in pool:
                raise FamilyError(f"c names an unknown ({suite}, {dataset}) cohort")
            if rep not in CLAIM_B_REPRESENTATIONS:
                raise FamilyError(f"c names an undeclared representation {rep!r}")


@dataclass(frozen=True)
class FamilyCardinality:
    """``N_actual`` by enumeration, with the closed form as a printed check.

    Attributes:
        n_max: :data:`N_MAX_F2`.
        n_actual: The enumerated admissible cell count. **This is the
            definition.**
        closed_form: ``182 - 15 k - 8 d + k d - c``, or ``101 - 5 k - c`` when
            F0's majority branch fired, with ``c`` the net stage-3 count.
        discrepancy: ``n_actual - closed_form``. Non-zero means the closed form
            mis-charged; the enumeration wins and this is reported.
        k: ``len(excluded_representations)``.
        d: ``len(uninformative_datasets)``.
        c: Net cells removed for non-computability, after ``k`` and ``d``.
        removed_by_k: Cells removed at stage 1.
        removed_by_d: Cells removed at stage 2, net of stage 1.
        removed_by_c: Cells removed at stage 3, net of stages 1 and 2.
        removed_by_f0: Cells demoted to descriptive by F0's majority branch,
            net of stages 1 to 3.
        admissible: The surviving cells.
        double_charged: Cells that stages 1 and 2 would both have removed. The
            closed form charges each of these twice.
    """

    n_max: int
    n_actual: int
    closed_form: int
    k: int
    d: int
    c: int
    removed_by_k: tuple[Cell, ...]
    removed_by_d: tuple[Cell, ...]
    removed_by_c: tuple[Cell, ...]
    removed_by_f0: tuple[Cell, ...]
    admissible: tuple[Cell, ...]
    double_charged: tuple[Cell, ...]
    f0_demoted: bool = False

    @property
    def discrepancy(self) -> int:
        """``n_actual - closed_form``; non-zero means the closed form is wrong."""
        return self.n_actual - self.closed_form

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, for the printed family."""
        return {
            "n_max": self.n_max,
            "n_actual": self.n_actual,
            "closed_form": self.closed_form,
            "closed_form_expression": (
                f"101 - 5*{self.k} - {self.c}"
                if self.f0_demoted
                else f"182 - 15*{self.k} - 8*{self.d} + {self.k}*{self.d} - {self.c}"
            ),
            "f0_demoted": self.f0_demoted,
            "d_applied": not self.f0_demoted,
            "discrepancy": self.discrepancy,
            "k": self.k,
            "d": self.d,
            "c": self.c,
            "n_removed_by_k": len(self.removed_by_k),
            "n_removed_by_d": len(self.removed_by_d),
            "n_removed_by_c": len(self.removed_by_c),
            "n_removed_by_f0": len(self.removed_by_f0),
            "n_double_charged_by_closed_form": len(self.double_charged),
            "double_charged": [cell.as_dict() for cell in self.double_charged],
            "removed_cells": {
                "k": [cell.as_dict() for cell in self.removed_by_k],
                "d": [cell.as_dict() for cell in self.removed_by_d],
                "c": [cell.as_dict() for cell in self.removed_by_c],
                "f0": [cell.as_dict() for cell in self.removed_by_f0],
            },
        }


#: F2 minus the 81 approximate-regime cells: A1 (60) + A2 (1) + B1e (35) + B3e (5).
N_AFTER_F0_DEMOTION: int = 101


def _closed_form(k: int, d: int, c: int, f0_demoted: bool) -> int:
    """Return the printed closed-form check for ``N_actual``.

    ``preregistration.md`` sections 5 and 5.3, both frozen 2026-08-16::

        ordinary branch:    182 - 15 k - 8 d + k d - c
        F0 majority fired:  101 -  5 k             - c

    Args:
        k: Representations excluded by ``competitors.md`` section 3.4.
        d: Suite-2 datasets whose bracket F1 declares uninformative.
        c: Net cells removed for non-computability, after the earlier stages.
        f0_demoted: Whether F0's majority branch fired.

    Returns:
        The closed-form value. It is a **check**; the enumeration is the
        definition.
    """
    if f0_demoted:
        return N_AFTER_F0_DEMOTION - 5 * k - c
    return N_MAX_F2 - 15 * k - 8 * d + k * d - c


def _charged_to_c(cell: Cell, noncomputable: frozenset[tuple[str, str, str]]) -> bool:
    """Whether *cell* fails the computability criterion on its own dataset."""
    if cell.row not in _C_CHARGEABLE_ROWS:
        return False
    if cell.suite is None or cell.dataset is None or cell.representation is None:
        return False
    return (cell.suite, cell.dataset, cell.representation) in noncomputable


def admissible_cells(inputs: ReductionInputs) -> FamilyCardinality:
    """Apply the three reduction terms in their frozen precedence.

    ``preregistration.md`` sections 5.2 and 5.3::

        0. F0  demotes the 81 approximate-regime cells (B1a 70 + B2 1 + B3a 10)
               when its majority branch fires, leaving 101. d is then SKIPPED.
        1. k   removes a representation's 15 Claim-B cells (5 B1e + 10 B1a),
               or only its 5 B1e cells when stage 0 fired
        2. d   removes 8 cells per uninformative dataset   (7 B1a + 1 B3a)
        3. c   removes, from what REMAINS, each (row, representation, dataset)
               cell whose representation fails the >= 99 % criterion

    No cell removed by an earlier stage is ever charged again. Stages 1 and 2
    genuinely overlap on the ``k * d`` B1a cells belonging to an excluded
    representation on an uninformative dataset; those are charged to ``k``,
    reported in :attr:`FamilyCardinality.double_charged`, and accounted for by
    the closed form's ``+ k d`` term.

    Args:
        inputs: The three reduction terms and F0's branch.

    Returns:
        The cardinality, the surviving cells and the per-stage removal lists.

    Raises:
        FamilyError: If a reduction term names something outside the freeze.
    """
    inputs.validate()
    cells = enumerate_f2_cells()

    removed: set[Cell] = set()
    by_k: list[Cell] = []
    by_d: list[Cell] = []
    by_c: list[Cell] = []
    by_f0: list[Cell] = []
    double: list[Cell] = []

    if inputs.f0_demotes_approximate:
        for cell in cells:
            if cell.row in _APPROXIMATE_ROWS:
                by_f0.append(cell)
                removed.add(cell)

    for cell in cells:
        hit = (
            cell.row in _K_REMOVABLE_ROWS and cell.representation in inputs.excluded_representations
        )
        if hit and cell not in removed:
            by_k.append(cell)
            removed.add(cell)

    if not inputs.f0_demotes_approximate:
        for cell in cells:
            if (
                cell.row not in _D_REMOVABLE_ROWS
                or cell.dataset not in inputs.uninformative_datasets
            ):
                continue
            if cell in removed:
                double.append(cell)
                continue
            by_d.append(cell)
            removed.add(cell)
    elif inputs.uninformative_datasets:
        LOGGER.info(
            "F0's majority branch fired, so d = %d is not applied: F1 tests the bracket within "
            "the approximate regime, which is now descriptive (preregistration.md section 5.3)",
            len(inputs.uninformative_datasets),
        )

    for cell in cells:
        if cell in removed or not _charged_to_c(cell, inputs.noncomputable):
            continue
        by_c.append(cell)
        removed.add(cell)

    admissible = tuple(cell for cell in cells if cell not in removed)
    k = len(inputs.excluded_representations)
    d = len(inputs.uninformative_datasets)
    closed_form = _closed_form(k, d, len(by_c), inputs.f0_demotes_approximate)

    result = FamilyCardinality(
        n_max=N_MAX_F2,
        n_actual=len(admissible),
        closed_form=closed_form,
        k=k,
        d=d,
        c=len(by_c),
        removed_by_k=tuple(by_k),
        removed_by_d=tuple(by_d),
        removed_by_c=tuple(by_c),
        removed_by_f0=tuple(by_f0),
        admissible=admissible,
        double_charged=tuple(double),
        f0_demoted=inputs.f0_demotes_approximate,
    )
    if result.discrepancy:
        LOGGER.warning(
            "N_actual enumeration (%d) disagrees with the closed form (%d) by %+d; "
            "the enumeration wins (preregistration.md section 5). %d cells sit in both the k "
            "and d removal sets.",
            result.n_actual,
            result.closed_form,
            result.discrepancy,
            len(double),
        )
    return result


def cardinality(
    *,
    excluded_representations: Iterable[str] = (),
    uninformative_datasets: Iterable[str] = (),
    noncomputable: Iterable[tuple[str, str, str]] = (),
    f0_demotes_approximate: bool = False,
) -> FamilyCardinality:
    """Convenience wrapper over :func:`admissible_cells`.

    Args:
        excluded_representations: ``k``.
        uninformative_datasets: ``d``.
        noncomputable: ``c``'s source triples.
        f0_demotes_approximate: F0's majority branch.

    Returns:
        The cardinality of F2 under those reductions.
    """
    return admissible_cells(
        ReductionInputs(
            excluded_representations=frozenset(excluded_representations),
            uninformative_datasets=frozenset(uninformative_datasets),
            noncomputable=frozenset(noncomputable),
            f0_demotes_approximate=f0_demotes_approximate,
        )
    )


# ---------------------------------------------------------------------------
# F0 and F1 --- the gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateOutcome:
    """One gate test and the branch its pre-declared rule takes.

    The rule requires **both** conditions. A CI excluding 0 at an effect size
    of 0.04 does **not** fire the gate: a difference can be reliably non-zero
    and still too small to matter, and the pre-registration says so.

    Attributes:
        test_id: e.g. ``"F0.4"``.
        dataset: Dataset key.
        point: The point estimate of the rho difference.
        interval: The BH-adjusted (FCR) percentile interval.
        p_value: Two-sided bootstrap p-value on the difference.
        p_adjusted: BH-adjusted p-value within the gate family.
        ci_excludes_zero: Whether *interval* lies strictly on one side of 0.
        exceeds_threshold: Whether ``|point| > 0.05``.
        fails: Both of the above. ``True`` means the gate's negative branch.
    """

    test_id: str
    dataset: str
    point: float
    interval: PercentileInterval
    p_value: float
    p_adjusted: float
    ci_excludes_zero: bool
    exceeds_threshold: bool
    fails: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "test_id": self.test_id,
            "dataset": self.dataset,
            "point": self.point,
            "interval": self.interval.as_dict(),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "ci_excludes_zero": self.ci_excludes_zero,
            "exceeds_threshold": self.exceeds_threshold,
            "fails": self.fails,
        }


def evaluate_gate(
    point: float,
    ci_low: float,
    ci_high: float,
    threshold: float = GATE_THRESHOLD,
) -> tuple[bool, bool, bool]:
    """Apply the pre-declared gate rule to one test.

    The rule, identical in F0 and F1: the gate fires if the BH-adjusted CI
    excludes 0 **and** ``|point estimate| > threshold``.

    Args:
        point: The point estimate.
        ci_low: Lower bound of the BH-adjusted interval.
        ci_high: Upper bound of the BH-adjusted interval.
        threshold: The pre-declared effect size; 0.05.

    Returns:
        ``(ci_excludes_zero, exceeds_threshold, fails)``.
    """
    finite = bool(np.isfinite(ci_low) and np.isfinite(ci_high))
    excludes = finite and (ci_low > 0.0 or ci_high < 0.0)
    exceeds = bool(np.isfinite(point) and abs(point) > threshold)
    return excludes, exceeds, bool(excludes and exceeds)


@dataclass(frozen=True)
class GateInput:
    """One dataset's contribution to a gate family.

    Attributes:
        dataset: Dataset key.
        point: Full-sample rho difference.
        samples: Bootstrap replicate differences from **one shared resample**
            per dataset (D7).
    """

    dataset: str
    point: float
    samples: FloatArray


@dataclass(frozen=True)
class GateFamilyResult:
    """The outcome of F0 or F1.

    Attributes:
        family: ``"F0"`` or ``"F1"``.
        outcomes: One per dataset, in input order.
        bh: The BH result over the gate family.
        ci_level: The FCR-adjusted coverage the intervals were built at.
        failing_datasets: Datasets whose gate fired.
        note: The consequence of the branch, in words.
    """

    family: str
    outcomes: tuple[GateOutcome, ...]
    bh: BenjaminiHochbergResult
    ci_level: float
    failing_datasets: tuple[str, ...]
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "family": self.family,
            "outcomes": [o.as_dict() for o in self.outcomes],
            "benjamini_hochberg": self.bh.as_dict(),
            "fcr_adjusted_ci_level": self.ci_level,
            "failing_datasets": list(self.failing_datasets),
            "note": self.note,
        }


def _run_gate_family(
    family: str,
    inputs: Sequence[GateInput],
    expected: int,
    q: float,
    threshold: float,
) -> tuple[tuple[GateOutcome, ...], BenjaminiHochbergResult, float, tuple[str, ...]]:
    """Run a gate family: bootstrap p-values, BH, FCR intervals, branch."""
    if len(inputs) != expected:
        raise FamilyError(
            f"{family} is frozen at {expected} tests, one per dataset; got {len(inputs)}"
        )
    p_values = [bootstrap_p_value(item.samples) for item in inputs]
    bh = benjamini_hochberg(p_values, family=family, m=expected, q=q)
    level = fcr_adjusted_level(bh.n_rejected, bh.m, q)

    outcomes: list[GateOutcome] = []
    failing: list[str] = []
    for position, item in enumerate(inputs):
        interval = percentile_interval(item.samples, item.point, level)
        excludes, exceeds, fails = evaluate_gate(
            item.point, interval.ci_low, interval.ci_high, threshold
        )
        outcomes.append(
            GateOutcome(
                test_id=f"{family}.{position + 1}",
                dataset=item.dataset,
                point=float(item.point),
                interval=interval,
                p_value=p_values[position],
                p_adjusted=bh.adjusted[position],
                ci_excludes_zero=excludes,
                exceeds_threshold=exceeds,
                fails=fails,
            )
        )
        if fails:
            failing.append(item.dataset)
    return tuple(outcomes), bh, level, tuple(failing)


def run_f0(
    inputs: Sequence[GateInput],
    *,
    q: float = FDR_Q,
    threshold: float = GATE_THRESHOLD,
) -> GateFamilyResult:
    """Run F0, the calibration gate: ``rho(Lev, exact) - rho(Lev, approx)``.

    One test per Suite-1 dataset, on the pairs where both quantities exist and
    on the same graph-level resamples (D7). The approximation is **not** a
    validated stand-in at a dataset if its BH-adjusted CI excludes 0 and
    ``|point| > 0.05``. If it fails on a **majority (>= 3) of the five**, the
    exact-GED results become primary and the approximate regime is reported
    descriptively.

    Args:
        inputs: Exactly five gate inputs, one per Suite-1 dataset.
        q: The false discovery rate; 0.05.
        threshold: The pre-declared effect size; 0.05.

    Returns:
        The gate outcome, with ``note`` recording the branch taken.

    Raises:
        FamilyError: If the input count is not five.
    """
    outcomes, bh, level, failing = _run_gate_family("F0", inputs, len(SUITE1), q, threshold)
    demotes = len(failing) >= 3
    note = (
        "F0 fails on a majority of five; exact-GED results become primary and F1 and F2's "
        "approximate-regime rows (B1a, B2, B3a) are descriptive only."
        if demotes
        else "F0 admits the approximate regime as confirmatory."
    )
    return GateFamilyResult("F0", outcomes, bh, level, failing, note)


def run_f1(
    inputs: Sequence[GateInput],
    *,
    q: float = FDR_Q,
    threshold: float = GATE_THRESHOLD,
) -> GateFamilyResult:
    """Run F1, the bracket gate (D13): ``rho(Lev, LB) - rho(Lev, UB)``.

    One test per Suite-2 dataset. A dataset's bracket is **uninformative** if
    its BH-adjusted CI excludes 0 and ``|point| > 0.05``; its rho is then
    reported as an interval, descriptively, and its 8 F2 cells are removed
    (7 x B1a + 1 x B3a). The failing set is ``d``.

    Args:
        inputs: Exactly ten gate inputs, one per Suite-2 dataset.
        q: The false discovery rate; 0.05.
        threshold: The pre-declared effect size; 0.05.

    Returns:
        The gate outcome; ``failing_datasets`` is ``d``.

    Raises:
        FamilyError: If the input count is not ten.
    """
    outcomes, bh, level, failing = _run_gate_family("F1", inputs, len(SUITE2), q, threshold)
    note = (
        f"d = {len(failing)}; each uninformative dataset removes 8 F2 cells (7 B1a + 1 B3a)."
        if failing
        else "d = 0; every Suite-2 bracket is informative and F2 keeps all its cells."
    )
    return GateFamilyResult("F1", outcomes, bh, level, failing, note)


# ---------------------------------------------------------------------------
# F2 --- the primary family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F2Result:
    """The primary family: its cardinality, its BH columns and its omnibuses.

    Attributes:
        cardinality: ``N_actual`` by enumeration with the closed-form check.
        cells: The admissible cells carrying a p-value, in order.
        bh_primary: BH over ``N_actual``. The reported column.
        bh_sensitivity: BH over ``N_max``. A re-threshold of the same stored
            p-values, printed beside the primary column (decision 24).
        omnibuses: A2 and B2, both on the ten-dataset approximate regime.
        exact_regime_omnibus: Always ``None``. The exact regime gets no
            omnibus; see :attr:`exact_regime_reason`.
        exact_regime_reason: Why, in words, for the manuscript.
    """

    cardinality: FamilyCardinality
    cells: tuple[Cell, ...]
    bh_primary: BenjaminiHochbergResult
    bh_sensitivity: BenjaminiHochbergResult
    omnibuses: dict[str, FriedmanResult] = field(default_factory=dict)
    exact_regime_omnibus: None = None
    exact_regime_reason: str = (
        "statistics.md section 4 and preregistration.md section 6: the exact regime has five "
        "datasets; Friedman at N = 5 separates almost nothing and an underpowered figure "
        "dressed as a result is worse than no figure. The exact regime is reported "
        "descriptively -- per-dataset rho with graph-level bootstrap CIs and D7 paired "
        "differences -- and the reason is stated in the text."
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "cardinality": self.cardinality.as_dict(),
            "cells": [cell.as_dict() for cell in self.cells],
            "bh_primary": self.bh_primary.as_dict(),
            "bh_over_n_max_sensitivity": self.bh_sensitivity.as_dict(),
            "omnibuses": {name: result.as_dict() for name, result in self.omnibuses.items()},
            "exact_regime_omnibus": None,
            "exact_regime_reason": self.exact_regime_reason,
        }


def run_f2(
    p_values: Mapping[Cell, float],
    inputs: ReductionInputs,
    *,
    q: float = FDR_Q,
    omnibus_scores: Mapping[str, tuple[npt.NDArray[Any], Sequence[str], bool]] | None = None,
) -> F2Result:
    """Apply D9 to the primary family and run its two omnibuses.

    BH is computed over ``N_actual``, which is the **enumerated** admissible
    cell count. The BH-over-``N_max`` sensitivity column is a re-threshold of
    the same stored p-values and costs nothing.

    Args:
        p_values: A p-value per admissible cell. Cells absent from the mapping
            are reported as missing rather than silently dropped from the
            denominator, because shrinking ``N_actual`` further than the data
            forces is the anti-conservative direction.
        inputs: The three reduction terms and F0's branch.
        q: The false discovery rate; 0.05.
        omnibus_scores: Optional ``{"A2": (scores, methods, lower_is_better),
            "B2": ...}``. Both run on :attr:`Regime.APPROXIMATE`; no exact-regime
            omnibus is ever produced.

    Returns:
        The family result.

    Raises:
        FamilyError: If *p_values* names a cell that is not admissible.
    """
    card = admissible_cells(inputs)
    admissible = set(card.admissible)
    unknown = [cell for cell in p_values if cell not in admissible]
    if unknown:
        raise FamilyError(f"{len(unknown)} p-values name inadmissible cells, first: {unknown[0]}")

    ordered = tuple(cell for cell in card.admissible if cell in p_values)
    missing = len(card.admissible) - len(ordered)
    if missing:
        LOGGER.warning(
            "%d of %d admissible F2 cells carry no p-value; they stay in the BH denominator",
            missing,
            len(card.admissible),
        )
    values = [float(p_values[cell]) for cell in ordered]

    bh_primary = benjamini_hochberg(values, family="F2", m=card.n_actual, q=q)
    bh_sensitivity = benjamini_hochberg(values, family="F2 (N_max)", m=N_MAX_F2, q=q)

    omnibuses: dict[str, FriedmanResult] = {}
    for name, (scores, methods, lower_is_better) in (omnibus_scores or {}).items():
        omnibuses[name] = friedman_omnibus(
            scores, methods, Regime.APPROXIMATE, lower_is_better=lower_is_better
        )
    return F2Result(
        cardinality=card,
        cells=ordered,
        bh_primary=bh_primary,
        bh_sensitivity=bh_sensitivity,
        omnibuses=omnibuses,
    )
