"""The immutable T-13 measurement record: one row per ``(graph, representation, arm)``.

T-13 replaces the manuscript's unqualified "exponential worst case" with a
characterised one (R3.7d), and the characterisation is a *statistical* claim
fitted over these rows.  A row that is silently malformed -- a field renamed,
a field dropped, a field added by a later edit -- does not raise; it produces a
column of nulls in the analysis and a plausible wrong exponent.  So the field
set is frozen here, in one place, and :func:`validate_mapping` rejects both
directions: a missing field **and** an extra one.

**Why an extra field is an error and not a courtesy.**  The analysis builds its
frame from :data:`FIELDS`.  A row carrying an undeclared key comes from code
that disagrees with this module about what a measurement is, and the
disagreements that matter -- a second ``wall_seconds`` beside ``seconds``, a
second ``length`` -- are exactly the ones a permissive reader would average
together.

**Why censoring carries a mechanism and not just a flag.**  Four different
things stop a measurement, and they are not interchangeable.  A wall-clock kill
at 300 s and a min-DFS projection cap that fires in 40 ms are both "the budget
ran out", but pooling them puts a fabricated 300 s into a timing distribution
whose whole purpose is to be fitted.  :data:`TIME_CENSORING_KINDS` and
:data:`CAP_CENSORING_KINDS` keep them separable, and the validator enforces
that only the former may claim ``seconds == budget_s``.

Nothing here imports ``isalgraph``, ``networkx`` or ``numpy``: the schema must
be readable by an analysis process with no C++ engine and by a test with
neither optional dependency installed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Final, Literal, get_args

#: Bumped on any change to :data:`FIELDS`.  The analysis refuses a file whose
#: header declares a version it does not know, rather than reading it anyway.
SCHEMA_VERSION: Final = "t13.1"

#: ``source`` domain.
Source = Literal["constructed", "cohort"]

#: ``arm`` domain.  ``default`` is the shipped engine; the other three toggle
#: ``isalgraph.core._native.set_pairs_memo`` / ``set_branch_and_bound`` to price
#: the two implementation optimisations separately from the algorithm.
Arm = Literal["default", "no_pairs_memo", "no_bnb", "no_pairs_memo_no_bnb"]

#: ``status`` domain.
#:
#: - ``ok``          the representation was computed and timed.
#: - ``censored``    a declared budget stopped it.  ``length_chars is None``,
#:   and ``error_kind`` names *which* budget.  **Right-censored, never
#:   dropped**: the censored cells are the high-``|Aut|`` cells, which is where
#:   the cost law lives, so dropping them would delete the result.
#: - ``unsupported`` the backend declined the graph, e.g. a ``SUITE1_ONLY``
#:   scope guard.  A property *of the representation*, recorded, never dropped.
#: - ``error``       anything else.  ``error_kind`` names the exception class.
Status = Literal["ok", "censored", "error", "unsupported"]

#: Every arm name, in the order the CLI documents them.
ARMS: Final[tuple[str, ...]] = get_args(Arm)

#: Every status name.
STATUSES: Final[tuple[str, ...]] = get_args(Status)

#: Every source name.
SOURCES: Final[tuple[str, ...]] = get_args(Source)

#: The parent killed the subprocess at the wall clock.  The only mechanism that
#: covers every backend, and the only enforcement ``SIGALRM`` cannot provide:
#: T-05 finding 5 established that ``SIGALRM`` does not interrupt the C++
#: engine, so a signal-based timeout silently fails to fire.
KIND_WALLCLOCK: Final = "wallclock_kill"

#: The C++ engine's own ``timeout_s`` fired inside the child, raising
#: ``CanonicalizationTimeoutError``.  Same budget as :data:`KIND_WALLCLOCK`,
#: reached cleanly instead of by a kill.
KIND_TIMEOUT: Final = "timeout_s"

#: AGM exhausted its branch-and-bound search-node budget.
KIND_SEARCH_NODES: Final = "search_nodes"

#: min-DFS exhausted its retained-embedding cap.  A **memory** guard, not a
#: speed knob: the first Suite-2 run was OOM-killed rather than slow.
KIND_MAX_PROJECTIONS: Final = "max_projections"

#: Censoring mechanisms that consumed the whole wall clock, and therefore the
#: only ones whose ``seconds`` may be reported as ``budget_s``.
TIME_CENSORING_KINDS: Final[tuple[str, ...]] = (KIND_WALLCLOCK, KIND_TIMEOUT)

#: Censoring mechanisms that fire on an internal cap, typically fast.  Their
#: ``seconds`` is the measured time to the cap and must **not** be inflated to
#: ``budget_s``, which would inject a fabricated 300 s into the timing
#: distribution the cost law is fitted on.
CAP_CENSORING_KINDS: Final[tuple[str, ...]] = (KIND_SEARCH_NODES, KIND_MAX_PROJECTIONS)

#: Every legal ``error_kind`` on a ``censored`` row.
CENSORING_KINDS: Final[tuple[str, ...]] = TIME_CENSORING_KINDS + CAP_CENSORING_KINDS


class SchemaError(ValueError):
    """A record does not match the frozen ``t13.1`` field set or domains."""


@dataclass(frozen=True, slots=True)
class Record:
    """One measurement of one representation on one graph under one arm.

    Frozen and slotted: a record is written once and never patched.  A
    correction is a re-run, so that every number in the report traces to a line
    of a shard file rather than to an in-memory edit nobody logged.

    Attributes:
        schema_version: always :data:`SCHEMA_VERSION`.
        run_id: campaign identifier, shared by every shard of one launch.
        host: ``platform.node()`` of the machine that produced the row.
        engine: ``isalgraph.engine()``, which must read ``"cpp"``.
        build_hash: ``isalgraph.build_info()["build_hash"]``.  A timing whose
            build hash is unknown is unprovenanced, which is why T-06's
            headline rates were retracted.
        isalgraph_version: package version.
        timestamp_utc: ISO-8601 UTC, second resolution.
        source: ``"constructed"`` for the controlled grid, ``"cohort"`` for the
            real-data external-validity arm.
        family: constructed family name, ``None`` for cohort rows.
        n_target: requested order of the constructed graph, which may differ
            from the realised ``n`` (a hypercube snaps to a power of two).
        replicate: replicate index for a random family, ``0`` for a
            deterministic one, ``None`` for cohort rows.
        params: the family's construction parameters, rendered as
            ``"swaps=3,base=hypercube"``.  ``None`` for cohort rows.

            **This field is what makes the primary analysis possible.**  Design
            note rule 7 establishes the ``|Aut|`` law from the
            ``symmetry_ladder`` within-``(n, m)`` contrast, which means
            ordering the rungs by their swap count ``k``.  Every rung of one
            ladder shares ``family``, ``n_target``, ``replicate``, ``n`` and
            ``m`` -- holding ``(n, m)`` exactly constant is the entire point of
            the design -- so without ``params`` the rungs are indistinguishable
            in the record and the contrast cannot be computed at all.  Integer
            parameter values that index a table (the ladder's ``base``) are
            resolved to their names at write time, so a shard file stays
            readable without ``families.py``.
        dataset: cohort name, ``None`` for constructed rows.
        graph_index: index into ``datasets.load(dataset).graphs``, ``None`` for
            constructed rows.
        graph_id: the exporter's own identifier, carried verbatim as a string.
        n: realised order.
        m: realised size.
        density: ``2m / (n(n-1))``, ``0.0`` for ``n < 2``.
        max_degree: largest degree, ``0`` for an edgeless graph.
        connected: whether the graph is connected.
        log10_aut: ``log10|Aut(G)|``.  Never ``|Aut(G)|`` itself, which
            overflows a float above ~1e308 -- ``K_n`` passes that at ``n=171``.
        n_orbits: number of automorphism orbits.
        max_orbit_size: size of the largest orbit.
        n_wl_classes: number of stable 1-WL colour classes.
        n_triplet_classes: number of classes of the incumbent pruning key.
        wl_refines_triplet: exact partition containment, never a class count.
        triplet_refines_wl: likewise.
        wl_equals_orbits: whether 1-WL resolves exactly the orbits.
        triplet_equals_orbits: likewise for the triplet key.
        representation: registry key, one of ``measure.REPRESENTATIONS``.
        arm: see :data:`Arm`.
        status: see :data:`Status`.
        error_kind: for a ``censored`` row, one of :data:`CENSORING_KINDS`;
            for ``error``/``unsupported``, the exception class name; ``None``
            for ``ok``.
        seconds: ``time.process_time`` seconds -- the median of three repeats
            when the warm-up ran under 1 s, otherwise the warm-up itself.
            Equal to ``budget_s`` only on a :data:`TIME_CENSORING_KINDS` row.
        repeats: timed runs the median was taken over: ``3`` or ``1``.  ``0``
            when nothing completed.
        budget_s: the wall clock this unit was allowed.
        budget_spec: the fully resolved :class:`~isalgraph.competitors.base.Budget`
            rendered as ``"search_nodes=...,max_projections=...,timeout_s=..."``.

            **A censoring rate is meaningless without the caps that produced
            it**, and a cap named only by its default is not a specification --
            the T-27 lesson, applied here.
        length_chars: symbols in the produced encoding, ``None`` when nothing
            was produced.
        fallback_used: ``None`` where the backend declares no
            ``fallback_variant``; ``False`` where it declares one.

            **It is never ``True``, by construction.**  A declared
            ``fallback_variant`` is advisory metadata: no backend performs its
            own substitution (``isalgraph_ref.py`` module docstring; ``encode``
            has no ``except`` and never reads the attribute), and this runner
            does not apply D14 either.  A substituted row would report
            ``status="ok"`` with a ``seconds`` that is exhaustive-time plus
            pruned-time and a ``length_chars`` from a different algorithm -- in
            precisely the high-``|Aut|`` cells the cost law is fitted on.  A
            ``True`` in a future file means someone added a substitution, and
            the analysis must refuse the file rather than average it in.
    """

    schema_version: str
    run_id: str
    host: str
    engine: str
    build_hash: str
    isalgraph_version: str
    timestamp_utc: str

    source: str
    family: str | None
    n_target: int | None
    replicate: int | None
    params: str | None
    dataset: str | None
    graph_index: int | None
    graph_id: str | None

    n: int
    m: int
    density: float
    max_degree: int
    connected: bool

    log10_aut: float | None
    n_orbits: int | None
    max_orbit_size: int | None
    n_wl_classes: int | None
    n_triplet_classes: int | None
    wl_refines_triplet: bool | None
    triplet_refines_wl: bool | None
    wl_equals_orbits: bool | None
    triplet_equals_orbits: bool | None

    representation: str
    arm: str
    status: str
    error_kind: str | None
    seconds: float
    repeats: int
    budget_s: float
    budget_spec: str
    length_chars: int | None
    fallback_used: bool | None

    def to_mapping(self) -> dict[str, Any]:
        """Return the record as a plain ``dict`` in :data:`FIELDS` order."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Return one JSON Lines row, terminated by a newline.

        ``allow_nan=False``: a ``NaN`` in ``seconds`` would round-trip through
        ``json.loads`` as a float and be silently averaged.  Better to raise
        here, where the offending record is still in hand.
        """
        return json.dumps(self.to_mapping(), ensure_ascii=False, allow_nan=False) + "\n"


#: The frozen field set, in declaration order.  The analysis builds its frame
#: from this tuple, so a change here is a schema-version bump.
FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in dataclass_fields(Record))

#: The nine fields ``symmetry.resolution_record`` supplies verbatim.
SYMMETRY_FIELDS: Final[tuple[str, ...]] = (
    "log10_aut",
    "n_orbits",
    "max_orbit_size",
    "n_wl_classes",
    "n_triplet_classes",
    "wl_refines_triplet",
    "triplet_refines_wl",
    "wl_equals_orbits",
    "triplet_equals_orbits",
)


def validate_mapping(mapping: Mapping[str, Any]) -> None:
    """Raise unless *mapping* carries exactly :data:`FIELDS` with legal domains.

    Both directions are errors.  A missing field is an incomplete row; an extra
    field is a row produced by code that disagrees with this module about what
    a measurement is.

    Args:
        mapping: candidate record, e.g. ``json.loads`` of one shard line.

    Raises:
        SchemaError: on a missing field, an extra field, an unknown
            ``schema_version``, a value outside the ``source``/``arm``/
            ``status`` domains, or a status/field combination the frozen timing
            rule cannot produce.
    """
    keys = set(mapping)
    expected = set(FIELDS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise SchemaError(
            f"record does not match schema {SCHEMA_VERSION}: missing={missing} extra={extra}"
        )

    version = mapping["schema_version"]
    if version != SCHEMA_VERSION:
        raise SchemaError(f"schema_version is {version!r}, expected {SCHEMA_VERSION!r}")

    for field, domain in (("source", SOURCES), ("arm", ARMS), ("status", STATUSES)):
        value = mapping[field]
        if value not in domain:
            raise SchemaError(f"{field}={value!r} is outside {list(domain)}")

    _validate_status_consistency(mapping)


def _validate_status_consistency(mapping: Mapping[str, Any]) -> None:
    """Reject the status/field combinations the frozen timing rule cannot emit.

    These are not stylistic checks.  ``status="censored"`` with a non-null
    ``length_chars`` is the signature of a fallback substitution laundering a
    censored graph into a completed one; ``status="ok"`` with ``repeats=0`` is
    the signature of a row whose ``seconds`` was never measured.  Both read as
    valid data downstream.

    Args:
        mapping: candidate record, already known to carry exactly
            :data:`FIELDS`.

    Raises:
        SchemaError: on any such combination.
    """
    status = mapping["status"]
    if status == "ok":
        if mapping["repeats"] not in (1, 3):
            raise SchemaError(
                f"status='ok' with repeats={mapping['repeats']!r}: the frozen timing "
                f"rule emits 1 (warm-up >= 1 s) or 3 (median of three)"
            )
        if mapping["length_chars"] is None:
            raise SchemaError("status='ok' with length_chars=None: nothing was produced")
        if mapping["error_kind"] is not None:
            raise SchemaError(f"status='ok' carries error_kind={mapping['error_kind']!r}")
    elif mapping["error_kind"] is None:
        raise SchemaError(f"status={status!r} must name an error_kind")

    if status == "censored":
        _validate_censored(mapping)

    if mapping["fallback_used"] is True:
        raise SchemaError(
            "fallback_used=True: no backend performs its own substitution and this "
            "runner does not apply D14, so a True here means a substituted encoding "
            "was timed as if it were the requested one. Refuse the file"
        )


def _validate_censored(mapping: Mapping[str, Any]) -> None:
    """Enforce the censoring rules that keep the two mechanisms separable.

    Args:
        mapping: a record whose ``status`` is ``"censored"``.

    Raises:
        SchemaError: when the mechanism is unnamed, when a completed encoding
            is attached, or when a fast internal cap claims the full budget.
    """
    kind = mapping["error_kind"]
    if kind not in CENSORING_KINDS:
        raise SchemaError(
            f"status='censored' with error_kind={kind!r}: name the mechanism, one of "
            f"{list(CENSORING_KINDS)}. A wall-clock kill and a projection cap are "
            f"different censoring processes and the analysis must not pool them"
        )
    if mapping["length_chars"] is not None:
        raise SchemaError(
            "status='censored' with a non-null length_chars: a censored graph was "
            "laundered into a completed one, which is the D14 bias T-13 measures"
        )
    if kind in TIME_CENSORING_KINDS and mapping["seconds"] != mapping["budget_s"]:
        raise SchemaError(
            f"error_kind={kind!r} with seconds={mapping['seconds']!r} != "
            f"budget_s={mapping['budget_s']!r}"
        )
    if kind in CAP_CENSORING_KINDS and mapping["seconds"] == mapping["budget_s"]:
        raise SchemaError(
            f"error_kind={kind!r} reports seconds == budget_s. An internal cap fires "
            f"on a count, not on the clock; reporting it as a full budget injects a "
            f"fabricated {mapping['budget_s']} s into the timing distribution"
        )


def record_from_mapping(mapping: Mapping[str, Any]) -> Record:
    """Validate *mapping* and build a :class:`Record` from it.

    Args:
        mapping: candidate record.

    Returns:
        The record.

    Raises:
        SchemaError: exactly as :func:`validate_mapping`.
    """
    validate_mapping(mapping)
    return Record(**{name: mapping[name] for name in FIELDS})


def run_header(
    *,
    run_id: str,
    host: str,
    engine: str,
    build_info: Mapping[str, Any],
    isalgraph_version: str,
    timestamp_utc: str,
    source: str,
    shard: int,
    n_shards: int,
    arms: tuple[str, ...],
    representations: tuple[str, ...],
    budget_s: float,
    seed: int,
    symmetry_available: bool,
) -> dict[str, Any]:
    """Build the one header line each shard file opens with.

    The header carries the whole ``build_info()`` mapping, not just the hash,
    because the compiler and ISA level are what let a later reader decide
    whether two shards are comparable at all.

    Args:
        run_id: campaign identifier.
        host: machine name.
        engine: ``isalgraph.engine()``.
        build_info: ``isalgraph.build_info()``.
        isalgraph_version: package version.
        timestamp_utc: ISO-8601 UTC.
        source: ``"constructed"`` or ``"cohort"``.
        shard: this shard's index.
        n_shards: total shards.
        arms: arms this shard ran.
        representations: representations this shard ran.
        budget_s: the frozen wall clock.
        seed: the campaign seed.
        symmetry_available: whether ``symmetry.resolution_record`` was
            importable.  ``False`` means the nine symmetry fields are null and
            no row of this shard may enter the ``|Aut|`` regression.

    Returns:
        A JSON-serialisable mapping tagged ``record_kind = "header"``.
    """
    return {
        "record_kind": "header",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "host": host,
        "engine": engine,
        "build_info": dict(build_info),
        "isalgraph_version": isalgraph_version,
        "timestamp_utc": timestamp_utc,
        "source": source,
        "shard": shard,
        "n_shards": n_shards,
        "arms": list(arms),
        "representations": list(representations),
        "budget_s": budget_s,
        "seed": seed,
        "symmetry_available": symmetry_available,
    }


__all__ = [
    "ARMS",
    "CAP_CENSORING_KINDS",
    "CENSORING_KINDS",
    "FIELDS",
    "KIND_MAX_PROJECTIONS",
    "KIND_SEARCH_NODES",
    "KIND_TIMEOUT",
    "KIND_WALLCLOCK",
    "SCHEMA_VERSION",
    "SOURCES",
    "STATUSES",
    "SYMMETRY_FIELDS",
    "TIME_CENSORING_KINDS",
    "Arm",
    "Record",
    "SchemaError",
    "Source",
    "Status",
    "record_from_mapping",
    "run_header",
    "validate_mapping",
]
