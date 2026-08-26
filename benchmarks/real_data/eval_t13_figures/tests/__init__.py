"""Tests for the T-13 figure package, and the synthetic fixture they run on.

Not collected by ``pytest`` at the repository root: ``pyproject.toml`` sets
``testpaths = ["tests"]``, so these run only when the path is given
explicitly::

    $PY -m pytest benchmarks/real_data/eval_t13_figures/tests/ -q

That is deliberate.  It keeps the repository's reference test count -- the
figure ``CLAUDE.md`` tracks -- a property of ``tests/`` alone, so work under
``benchmarks/`` can neither inflate nor deflate it.

**Why the fixture lives in code and is also committed as data.**  The T-13
campaign has not run, so no real ``records_*.jsonl`` exists.  Every acceptance
criterion of this package is nonetheless checkable against a file that
conforms to the frozen ``t13.1`` schema, so this module builds one.  It is
built rather than hand-written because ``schema.validate_mapping`` rejects a
missing field, an extra field and several status/field combinations, and a
hand-written 400-row file drifts from all three the first time the schema
moves.  :func:`write_records` materialises the same rows to disk, and the
materialised copy is committed beside these tests so a human can read it and so
the CLI can be exercised end to end.

**The fixture is structurally realistic, and each feature is there for a
reason:**

* the ``spider_ladder`` reproduces the ``T-13-design.md`` 6.3 pilot exactly --
  ``n = 33``, ``m = 32``, four rungs, ``|Aut|`` from ``10^4.61`` to ``10^0.30``
  -- so a figure drawn from it shows the contrast the paper reports;
* ``min_dfs`` at the most symmetric rung is **cap-censored at 4 ms**, which is
  the specific trap the censoring rules exist for: pooled with completions it
  reads as the fastest measurement in the panel;
* ``isalgraph_canonical`` is **wall-clock censored at the full budget** on
  every spider rung, so the Kaplan--Meier median is legitimately not reached;
* one ``symmetry_ladder`` is deliberately **non-monotone in** ``log10_aut``
  across its rungs, so a rung ordering that used ``log10_aut`` would visibly
  differ from one that used the ``params`` index;
* ``agm_cam`` is ``unsupported`` on one ladder and one row is an ``error``,
  because both statuses must survive every summary without being counted as
  fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from benchmarks.real_data.eval_t13_complexity import measure, schema

#: Committed fixture file names, matching ``measure.shard_filename``'s shape so
#: a ``--records`` glob written for the real campaign also matches these.
RECORDS_FIXTURE: Final[str] = "records_constructed_0of1.jsonl"
COUNTERS_FIXTURE: Final[str] = "counters_fixture.jsonl"

#: The one build every fixture shard declares.
BUILD_HASH: Final[str] = "fixture0000000000000000000000000000000000"

#: A second, deliberately different build, used only by the mixed-build test.
OTHER_BUILD_HASH: Final[str] = "fixture1111111111111111111111111111111111"

#: The frozen wall clock the fixture campaign declares.
BUDGET_S: Final[float] = 300.0

#: The resolved budget object, rendered exactly as ``measure`` renders it.
BUDGET_SPEC: Final[str] = "search_nodes=200000,max_projections=500000,timeout_s=300.0"

_HEADER_COMMON: Final[dict[str, Any]] = {
    "run_id": "t13-fixture",
    "host": "fixture-host",
    "engine": "cpp",
    "isalgraph_version": "0.0.0-fixture",
    "timestamp_utc": "2026-08-26T00:00:00",
    "seed": 20260826,
}


def _row(
    *,
    family: str,
    n_target: int,
    replicate: int,
    params: str,
    n: int,
    m: int,
    max_degree: int,
    log10_aut: float | None,
    n_orbits: int | None,
    n_wl_classes: int | None,
    n_triplet_classes: int | None,
    representation: str,
    status: str,
    seconds: float,
    length_chars: int | None,
    error_kind: str | None = None,
    arm: str = "default",
) -> dict[str, Any]:
    """Build one ``t13.1`` row, validated before it is returned.

    Args:
        family: Constructed family name.
        n_target: Requested order.
        replicate: Replicate index.
        params: Rendered ``params`` string, exactly as ``measure`` writes it.
        n: Realised order.
        m: Realised size.
        max_degree: Largest degree.
        log10_aut: ``log10|Aut(G)|``.
        n_orbits: Automorphism orbits.
        n_wl_classes: Stable 1-WL classes.
        n_triplet_classes: Triplet-key classes.
        representation: Registry key.
        status: One of ``schema.STATUSES``.
        seconds: Observation time.
        length_chars: Encoding length, ``None`` unless ``status == "ok"``.
        error_kind: Censoring mechanism or exception class name.
        arm: Engine arm.

    Returns:
        The row.

    Raises:
        schema.SchemaError: If the row does not satisfy the frozen schema --
            the fixture is validated at construction so a broken fixture is a
            fixture bug and not a mysterious test failure.
    """
    row: dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": _HEADER_COMMON["run_id"],
        "host": _HEADER_COMMON["host"],
        "engine": "cpp",
        "build_hash": BUILD_HASH,
        "isalgraph_version": _HEADER_COMMON["isalgraph_version"],
        "timestamp_utc": _HEADER_COMMON["timestamp_utc"],
        "source": "constructed",
        "family": family,
        "n_target": n_target,
        "replicate": replicate,
        "params": params,
        "dataset": None,
        "graph_index": None,
        "graph_id": f"{family}|{params}|{replicate}",
        "n": n,
        "m": m,
        "density": (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0,
        "max_degree": max_degree,
        "connected": True,
        "log10_aut": log10_aut,
        "n_orbits": n_orbits,
        "max_orbit_size": None if n_orbits is None else max(1, n // max(n_orbits, 1)),
        "n_wl_classes": n_wl_classes,
        "n_triplet_classes": n_triplet_classes,
        "wl_refines_triplet": None if n_wl_classes is None else True,
        "triplet_refines_wl": None if n_wl_classes is None else False,
        "wl_equals_orbits": None if n_orbits is None else n_wl_classes == n_orbits,
        "triplet_equals_orbits": None if n_orbits is None else n_triplet_classes == n_orbits,
        "representation": representation,
        "arm": arm,
        "status": status,
        "error_kind": error_kind,
        "seconds": seconds,
        "repeats": 3 if status == "ok" and seconds < 1.0 else (1 if status == "ok" else 0),
        "budget_s": BUDGET_S,
        "budget_spec": BUDGET_SPEC,
        "length_chars": length_chars,
        "fallback_used": False,
    }
    schema.validate_mapping(row)
    return row


#: ``spider_ladder`` at ``n = 33``: the 6.3 pilot, rung by rung.
#: ``(rung, log10_aut, n_orbits, n_wl, n_triplet)``.
SPIDER_RUNGS: Final[tuple[tuple[int, float, int, int, int], ...]] = (
    (0, 4.6055, 5, 5, 4),
    (1, 2.8573, 9, 9, 8),
    (2, 1.3802, 13, 13, 11),
    (3, 0.3010, 17, 17, 14),
)

#: Per-representation seconds on each spider rung, from the 6.3 pilot table
#: where it reports one, and structurally plausible elsewhere.  ``None`` marks
#: a censored rung, whose seconds come from :data:`SPIDER_CENSORED`.
SPIDER_SECONDS: Final[dict[str, tuple[float | None, ...]]] = {
    "isalgraph_exhaustive": (0.4767, 0.4873, 0.5151, 0.4682),
    "isalgraph_pruned": (0.3807, 0.0741, 0.0147, 0.0030),
    "isalgraph_canonical": (None, None, None, None),
    "isalgraph_greedy": (0.0005, 0.0005, 0.0005, 0.0005),
    "min_dfs": (None, 0.0759, 0.0109, 0.0086),
    "agm_cam": (0.0210, 0.0180, 0.0160, 0.0150),
    "nauty_graph6": (0.0013, 0.0010, 0.0013, 0.0012),
    "sparse6_nauty": (0.0011, 0.0010, 0.0012, 0.0011),
    "graph6": (0.0004, 0.0004, 0.0004, 0.0004),
    "sparse6": (0.0004, 0.0004, 0.0004, 0.0004),
    "adjacency": (0.0003, 0.0003, 0.0003, 0.0003),
    "wl_subtree": (0.0012, 0.0012, 0.0012, 0.0012),
    "size_null": (0.0001, 0.0001, 0.0001, 0.0001),
}

#: Censoring on the spider ladder: ``(representation, rung) -> (kind, seconds)``.
#: ``isalgraph_canonical`` burns the whole wall clock on every rung -- an
#: exhaustive search with no fallback at ``n = 33`` -- while ``min_dfs`` trips
#: its projection cap in four milliseconds at the most symmetric rung, which is
#: the pooling trap in its purest form.
SPIDER_CENSORED: Final[dict[tuple[str, int], tuple[str, float]]] = {
    ("isalgraph_canonical", 0): (schema.KIND_WALLCLOCK, BUDGET_S),
    ("isalgraph_canonical", 1): (schema.KIND_WALLCLOCK, BUDGET_S),
    ("isalgraph_canonical", 2): (schema.KIND_WALLCLOCK, BUDGET_S),
    ("isalgraph_canonical", 3): (schema.KIND_WALLCLOCK, BUDGET_S),
    ("min_dfs", 0): (schema.KIND_MAX_PROJECTIONS, 0.0041),
}

#: ``symmetry_ladder`` on the 4-cube.  ``log10_aut`` is **non-monotone in the
#: rung index on purpose**: rung 2 sits above rung 1.  A grouping that ordered
#: rungs by ``log10_aut`` would put them in a different order from one that
#: ordered by the ``swaps`` parameter, which is what the ordering test detects.
HYPERCUBE_RUNGS: Final[tuple[tuple[int, float, int, int, int], ...]] = (
    (0, 2.9823, 1, 1, 1),
    (1, 0.9031, 6, 6, 5),
    (2, 1.2041, 5, 5, 4),
    (4, 0.3010, 9, 9, 8),
)

#: ``symmetry_ladder`` on ``K_{4,4}``, monotone, and the ladder that carries
#: the ``unsupported`` and ``error`` statuses.
BIPARTITE_RUNGS: Final[tuple[tuple[int, float, int, int, int], ...]] = (
    (0, 3.1584, 2, 2, 2),
    (1, 1.0792, 4, 4, 3),
    (2, 0.3010, 6, 6, 5),
)


#: Per-representation cost scale on the two ladders that do not transcribe the
#: pilot.  Without it every arm lands on the same value and the panel is a
#: single overplotted line, which is a fixture artefact a reader would have to
#: diagnose before trusting the figure.  The ordering is the pilot's --
#: exhaustive above pruned above the serialisations -- not a measurement.
FLAT_SCALE: Final[dict[str, float]] = {
    "isalgraph_canonical": 40.0,
    "isalgraph_exhaustive": 14.0,
    "isalgraph_pruned": 6.0,
    "isalgraph_greedy": 0.12,
    "min_dfs": 4.5,
    "agm_cam": 2.0,
    "nauty_graph6": 0.55,
    "sparse6_nauty": 0.45,
    "graph6": 0.10,
    "sparse6": 0.11,
    "adjacency": 0.08,
    "wl_subtree": 0.30,
    "size_null": 0.02,
}

#: Arms whose fixture cost rises with ``|Aut|`` -- the two the characterisation
#: says are governed by the automorphism group.  Everything else is drawn flat,
#: because that is what the null predicts.
AUT_SENSITIVE: Final[frozenset[str]] = frozenset({"isalgraph_pruned", "min_dfs"})


def _flat_seconds(key: str, base_seconds: float, aut: float, position: int) -> float:
    """Return the fixture's cost for one arm on a non-pilot ladder.

    Args:
        key: Registry key.
        base_seconds: The ladder's scale.
        aut: ``log10|Aut(G)|`` at this rung.
        position: Index of the rung within the ladder.

    Returns:
        Seconds, rising with ``aut`` for the two automorphism-governed arms and
        near-flat for the rest.
    """
    scale = base_seconds * FLAT_SCALE[key]
    if key in AUT_SENSITIVE:
        return float(scale * 10.0 ** (aut / 1.5))
    return float(scale * (1.0 + 0.04 * position))


def _ladder_rows(
    *,
    family: str,
    base_params: str,
    rung_key: str,
    n: int,
    m: int,
    max_degree: int,
    rungs: tuple[tuple[int, float, int, int, int], ...],
    seconds_at: dict[str, tuple[float | None, ...]] | None,
    censored: dict[tuple[str, int], tuple[str, float]],
    unsupported: frozenset[str] = frozenset(),
    error_at: tuple[str, int] | None = None,
    base_seconds: float = 0.01,
    replicate: int = 0,
) -> list[dict[str, Any]]:
    """Build every row of one ladder.

    Args:
        family: Ladder family name.
        base_params: The ``params`` fragment shared by every rung.
        rung_key: ``swaps`` or ``rung``.
        n: Order, constant across the ladder.
        m: Size, constant across the ladder.
        max_degree: Largest degree, constant across the ladder.
        rungs: ``(index, log10_aut, n_orbits, n_wl, n_triplet)`` per rung.
        seconds_at: Per-representation seconds per rung, or ``None`` to derive
            a flat series from *base_seconds*.
        censored: ``(representation, rung) -> (kind, seconds)``.
        unsupported: Representations that decline this ladder outright.
        error_at: One ``(representation, rung)`` to record as an error.
        base_seconds: Seconds used when *seconds_at* is ``None``.
        replicate: Replicate index carried by every row.

    Returns:
        The rows.
    """
    rows: list[dict[str, Any]] = []
    for position, (index, aut, orbits, wl, triplet) in enumerate(rungs):
        params = (
            ",".join(sorted([base_params, f"{rung_key}={index}"]))
            if base_params
            else (f"{rung_key}={index}")
        )
        for key in measure.REPRESENTATIONS:
            common: dict[str, Any] = {
                "family": family,
                "n_target": n,
                "replicate": replicate,
                "params": params,
                "n": n,
                "m": m,
                "max_degree": max_degree,
                "log10_aut": aut,
                "n_orbits": orbits,
                "n_wl_classes": wl,
                "n_triplet_classes": triplet,
                "representation": key,
            }
            if key in unsupported:
                rows.append(
                    _row(
                        **common,
                        status="unsupported",
                        seconds=0.0,
                        length_chars=None,
                        error_kind="ScopeError",
                    )
                )
                continue
            if error_at == (key, index):
                rows.append(
                    _row(
                        **common,
                        status="error",
                        seconds=0.0,
                        length_chars=None,
                        error_kind="RuntimeError",
                    )
                )
                continue
            censor = censored.get((key, index))
            if censor is not None:
                rows.append(
                    _row(
                        **common,
                        status="censored",
                        seconds=censor[1],
                        length_chars=None,
                        error_kind=censor[0],
                    )
                )
                continue
            if seconds_at is not None:
                seconds = seconds_at[key][position]
                assert seconds is not None, f"{key} rung {index} needs a censoring entry"
            else:
                seconds = _flat_seconds(key, base_seconds, aut, position)
            rows.append(
                _row(**common, status="ok", seconds=seconds, length_chars=n + m, error_kind=None)
            )
    return rows


def build_rows() -> list[dict[str, Any]]:
    """Build every measurement row of the fixture campaign.

    Returns:
        Three ladders' worth of validated ``t13.1`` rows.
    """
    rows = _ladder_rows(
        family="spider_ladder",
        base_params="leg=4,legs=8",
        rung_key="rung",
        n=33,
        m=32,
        max_degree=8,
        rungs=SPIDER_RUNGS,
        seconds_at=SPIDER_SECONDS,
        censored=SPIDER_CENSORED,
    )
    rows += _ladder_rows(
        family="symmetry_ladder",
        base_params="base=hypercube",
        rung_key="swaps",
        n=16,
        m=32,
        max_degree=4,
        rungs=HYPERCUBE_RUNGS,
        seconds_at=None,
        censored={("min_dfs", 0): (schema.KIND_MAX_PROJECTIONS, 0.0035)},
        base_seconds=0.02,
    )
    rows += _ladder_rows(
        family="symmetry_ladder",
        base_params="base=complete_bipartite",
        rung_key="swaps",
        n=8,
        m=16,
        max_degree=4,
        rungs=BIPARTITE_RUNGS,
        seconds_at=None,
        censored={("isalgraph_canonical", 0): (schema.KIND_TIMEOUT, BUDGET_S)},
        unsupported=frozenset({"agm_cam"}),
        error_at=("sparse6_nauty", 2),
        base_seconds=0.005,
    )
    return rows


def build_header(*, build_hash: str = BUILD_HASH, run_id: str | None = None) -> dict[str, Any]:
    """Build the fixture shard header.

    Args:
        build_hash: The engine build to declare.
        run_id: Campaign identifier, defaulting to the fixture's own.

    Returns:
        A ``record_kind = "header"`` mapping.
    """
    return schema.run_header(
        run_id=run_id or str(_HEADER_COMMON["run_id"]),
        host=str(_HEADER_COMMON["host"]),
        engine="cpp",
        build_info={
            "build_hash": build_hash,
            "compiler": "gcc 12.2.0",
            "isa": "x86-64-v3",
        },
        isalgraph_version=str(_HEADER_COMMON["isalgraph_version"]),
        timestamp_utc=str(_HEADER_COMMON["timestamp_utc"]),
        source="constructed",
        shard=0,
        n_shards=1,
        arms=("default",),
        representations=measure.REPRESENTATIONS,
        budget_s=BUDGET_S,
        seed=int(_HEADER_COMMON["seed"]),
        symmetry_available=True,
    )


def write_records(
    directory: Path,
    *,
    name: str = RECORDS_FIXTURE,
    build_hash: str = BUILD_HASH,
    rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a fixture shard file into *directory*.

    Args:
        directory: Destination directory, created if absent.
        name: File name.
        build_hash: The engine build the header and every row declares.
        rows: Rows to write, defaulting to :func:`build_rows`.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    payload = build_rows() if rows is None else rows
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(build_header(build_hash=build_hash), ensure_ascii=False) + "\n")
        for row in payload:
            stamped = dict(row)
            stamped["build_hash"] = build_hash
            handle.write(json.dumps(stamped, ensure_ascii=False, allow_nan=False) + "\n")
    return path


#: Encoders the counter fixture emits, and the fraction of the Section 2.1
#: bound each one's counts realise.
#:
#: Counts are expressed as a **fraction of the bound** rather than as an
#: independent growth law, because the fixture must satisfy the property real
#: counter data satisfies: the realised count sits under the derived bound.  A
#: fixture whose measured curve crossed its own bound would make the
#: supplementary figure look like a refutation of the derivation, which is a
#: worse defect in a figure fixture than an unrealistic magnitude.
#:
#: ``(encoder, fraction_of_bound, search_fraction)``.  Plausible, not measured:
#: the counter campaign has not run either, and these rows exist to exercise
#: the reader and the panel layout, never to be quoted.
COUNTER_ENCODERS: Final[tuple[tuple[str, float, float], ...]] = (
    ("greedy_single", 0.006, 0.0),
    ("greedy_min", 0.09, 0.0),
    ("canonical", 0.62, 1.0),
    ("pruned", 0.21, 0.22),
)

#: Node counts the counter fixture covers.
COUNTER_SIZES: Final[tuple[int, ...]] = (6, 8, 10, 12)


def build_counter_rows() -> list[dict[str, Any]]:
    """Build the ``t13c.1`` counter fixture rows.

    Returns:
        One row per ``(n, encoder)``, every one with ``parity_ok = True`` and
        every count under its Section 2.1 bound.
    """
    rows: list[dict[str, Any]] = []
    for n in COUNTER_SIZES:
        m = (n * (n - 1)) // 3
        delta = n - 1
        for encoder, fraction, search_fraction in COUNTER_ENCODERS:
            trials = int(fraction * m * n**2)
            rows.append(
                {
                    "schema_version": "t13c.1",
                    "source": "constructed",
                    "family": "rigid_er",
                    "n_target": n,
                    "replicate": 0,
                    "dataset": None,
                    "graph_index": None,
                    "n": n,
                    "m": m,
                    "encoder": encoder,
                    "frames": m if encoder == "greedy_single" else n * m,
                    "pair_trials": trials,
                    "scan_depth_total": trials,
                    "scan_depth_max": int(fraction * n**2) + 1,
                    "pointer_steps": int(fraction * m * n**3 / 2),
                    "neighbour_checks": int(fraction * m * delta),
                    "backtrack_nodes": int(search_fraction * n**2.2) if search_fraction else 0,
                    "search_leaves": int(search_fraction * n**1.7) if search_fraction else 0,
                    "string_length": m + n,
                    "parity_ok": True,
                }
            )
    return rows


def write_counters(directory: Path, *, name: str = COUNTERS_FIXTURE) -> Path:
    """Write the counter fixture into *directory*.

    Args:
        directory: Destination directory, created if absent.
        name: File name.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8") as handle:
        for row in build_counter_rows():
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path


#: Where the committed copies live.
FIXTURE_DIR: Final[Path] = Path(__file__).resolve().parent


def committed_records() -> Path:
    """Return the committed records fixture, writing it if it is absent."""
    path = FIXTURE_DIR / RECORDS_FIXTURE
    if not path.exists():
        write_records(FIXTURE_DIR)
    return path


def committed_counters() -> Path:
    """Return the committed counters fixture, writing it if it is absent."""
    path = FIXTURE_DIR / COUNTERS_FIXTURE
    if not path.exists():
        write_counters(FIXTURE_DIR)
    return path


if __name__ == "__main__":  # pragma: no cover - regeneration helper
    print(write_records(FIXTURE_DIR))
    print(write_counters(FIXTURE_DIR))
