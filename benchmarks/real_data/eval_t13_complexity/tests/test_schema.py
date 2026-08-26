"""Tests for the frozen ``t13.1`` record.

The point of these tests is that a schema drift must **fail here** rather than
surface as a null column three weeks later in the analysis, when the shards
have been produced and the cluster time is spent.  So the field list is spelled
out literally rather than derived from the dataclass: a test that reads
``FIELDS`` to check ``FIELDS`` cannot catch a rename.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.real_data.eval_t13_complexity import schema

#: The frozen field list, written out by hand.  If this diverges from
#: ``schema.FIELDS`` one of the two is wrong and a human must decide which.
EXPECTED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "host",
    "engine",
    "build_hash",
    "isalgraph_version",
    "timestamp_utc",
    "source",
    "family",
    "n_target",
    "replicate",
    "params",
    "dataset",
    "graph_index",
    "graph_id",
    "n",
    "m",
    "density",
    "max_degree",
    "connected",
    "log10_aut",
    "n_orbits",
    "max_orbit_size",
    "n_wl_classes",
    "n_triplet_classes",
    "wl_refines_triplet",
    "triplet_refines_wl",
    "wl_equals_orbits",
    "triplet_equals_orbits",
    "representation",
    "arm",
    "status",
    "error_kind",
    "seconds",
    "repeats",
    "budget_s",
    "budget_spec",
    "length_chars",
    "fallback_used",
)


def valid_mapping(**overrides: Any) -> dict[str, Any]:
    """Return a schema-valid record mapping, with *overrides* applied.

    Args:
        **overrides: fields to replace.

    Returns:
        The mapping.
    """
    base: dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": "t13_20260826T120000Z",
        "host": "picasso-sd01",
        "engine": "cpp",
        "build_hash": "298fc1188bf1b051",
        "isalgraph_version": "0.1.0",
        "timestamp_utc": "2026-08-26T12:00:00+00:00",
        "source": "constructed",
        "family": "cycle",
        "n_target": 12,
        "replicate": 0,
        "params": "swaps=0,base=cycle",
        "dataset": None,
        "graph_index": None,
        "graph_id": None,
        "n": 12,
        "m": 12,
        "density": 0.18181818181818182,
        "max_degree": 2,
        "connected": True,
        "log10_aut": 1.380211241711606,
        "n_orbits": 1,
        "max_orbit_size": 12,
        "n_wl_classes": 1,
        "n_triplet_classes": 1,
        "wl_refines_triplet": True,
        "triplet_refines_wl": True,
        "wl_equals_orbits": True,
        "triplet_equals_orbits": True,
        "representation": "isalgraph_pruned",
        "arm": "default",
        "status": "ok",
        "error_kind": None,
        "seconds": 0.000123,
        "repeats": 3,
        "budget_s": 300.0,
        "budget_spec": "search_nodes=200000,max_projections=50000,timeout_s=300.0",
        "length_chars": 23,
        "fallback_used": None,
    }
    base.update(overrides)
    return base


class TestFieldSet:
    """Criterion 1: exactly those field names, at ``schema_version t13.1``."""

    def test_schema_version_is_t13_1(self) -> None:
        assert schema.SCHEMA_VERSION == "t13.1"

    def test_fields_match_the_hand_written_list(self) -> None:
        assert schema.FIELDS == EXPECTED_FIELDS

    def test_record_is_frozen(self) -> None:
        record = schema.record_from_mapping(valid_mapping())
        with pytest.raises((AttributeError, TypeError)):
            record.seconds = 1.0  # type: ignore[misc]

    def test_symmetry_fields_are_a_subset_of_fields(self) -> None:
        assert set(schema.SYMMETRY_FIELDS) <= set(schema.FIELDS)
        assert len(schema.SYMMETRY_FIELDS) == 9

    def test_valid_mapping_round_trips(self) -> None:
        record = schema.record_from_mapping(valid_mapping())
        line = record.to_json_line()
        assert line.endswith("\n")
        assert schema.record_from_mapping(json.loads(line)) == record


class TestValidatorRejectsShapeDrift:
    """Criterion 1: a missing field **and** an extra field are both errors."""

    @pytest.mark.parametrize("dropped", EXPECTED_FIELDS)
    def test_missing_any_field_is_rejected(self, dropped: str) -> None:
        mapping = valid_mapping()
        del mapping[dropped]
        with pytest.raises(schema.SchemaError, match="missing"):
            schema.validate_mapping(mapping)

    def test_extra_field_is_rejected(self) -> None:
        mapping = valid_mapping(wall_seconds=1.5)
        with pytest.raises(schema.SchemaError, match="extra"):
            schema.validate_mapping(mapping)

    def test_wrong_schema_version_is_rejected(self) -> None:
        with pytest.raises(schema.SchemaError, match="schema_version"):
            schema.validate_mapping(valid_mapping(schema_version="t13.0"))

    @pytest.mark.parametrize(
        ("field", "bad"),
        [("source", "synthetic"), ("arm", "no_memo"), ("status", "timeout")],
    )
    def test_domains_are_closed(self, field: str, bad: str) -> None:
        with pytest.raises(schema.SchemaError, match=field):
            schema.validate_mapping(valid_mapping(**{field: bad}))


class TestStatusConsistency:
    """The status/field combinations the frozen timing rule cannot emit."""

    def test_ok_requires_one_or_three_repeats(self) -> None:
        with pytest.raises(schema.SchemaError, match="repeats"):
            schema.validate_mapping(valid_mapping(repeats=0))
        with pytest.raises(schema.SchemaError, match="repeats"):
            schema.validate_mapping(valid_mapping(repeats=2))

    def test_ok_requires_a_length(self) -> None:
        with pytest.raises(schema.SchemaError, match="length_chars"):
            schema.validate_mapping(valid_mapping(length_chars=None))

    def test_ok_may_not_carry_an_error_kind(self) -> None:
        with pytest.raises(schema.SchemaError, match="error_kind"):
            schema.validate_mapping(valid_mapping(error_kind="SuiteScopeError"))

    def test_non_ok_must_name_an_error_kind(self) -> None:
        with pytest.raises(schema.SchemaError, match="error_kind"):
            schema.validate_mapping(
                valid_mapping(status="unsupported", error_kind=None, repeats=0, length_chars=None)
            )

    def test_unsupported_row_is_valid(self) -> None:
        schema.validate_mapping(
            valid_mapping(
                status="unsupported",
                error_kind="SuiteScopeError",
                repeats=0,
                seconds=0.0,
                length_chars=None,
            )
        )


class TestCensoring:
    """Censoring carries its mechanism, and the two kinds stay separable."""

    def test_wallclock_censoring_is_valid(self) -> None:
        schema.validate_mapping(
            valid_mapping(
                status="censored",
                error_kind=schema.KIND_WALLCLOCK,
                seconds=300.0,
                repeats=0,
                length_chars=None,
            )
        )

    def test_censored_must_name_a_known_mechanism(self) -> None:
        with pytest.raises(schema.SchemaError, match="name the mechanism"):
            schema.validate_mapping(
                valid_mapping(
                    status="censored",
                    error_kind="TimeoutError",
                    seconds=300.0,
                    repeats=0,
                    length_chars=None,
                )
            )

    def test_censored_may_not_carry_a_length(self) -> None:
        with pytest.raises(schema.SchemaError, match="laundered"):
            schema.validate_mapping(
                valid_mapping(
                    status="censored",
                    error_kind=schema.KIND_WALLCLOCK,
                    seconds=300.0,
                    repeats=0,
                    length_chars=41,
                )
            )

    def test_time_censoring_must_report_the_full_budget(self) -> None:
        with pytest.raises(schema.SchemaError, match="!="):
            schema.validate_mapping(
                valid_mapping(
                    status="censored",
                    error_kind=schema.KIND_TIMEOUT,
                    seconds=287.4,
                    repeats=0,
                    length_chars=None,
                )
            )

    def test_cap_censoring_records_its_own_measured_time(self) -> None:
        schema.validate_mapping(
            valid_mapping(
                status="censored",
                error_kind=schema.KIND_MAX_PROJECTIONS,
                seconds=0.041,
                repeats=0,
                length_chars=None,
            )
        )

    def test_cap_censoring_may_not_claim_the_full_budget(self) -> None:
        with pytest.raises(schema.SchemaError, match="fabricated"):
            schema.validate_mapping(
                valid_mapping(
                    status="censored",
                    error_kind=schema.KIND_SEARCH_NODES,
                    seconds=300.0,
                    repeats=0,
                    length_chars=None,
                )
            )

    def test_the_two_mechanism_families_are_disjoint(self) -> None:
        assert not set(schema.TIME_CENSORING_KINDS) & set(schema.CAP_CENSORING_KINDS)
        assert set(schema.CENSORING_KINDS) == set(schema.TIME_CENSORING_KINDS) | set(
            schema.CAP_CENSORING_KINDS
        )


class TestFallbackIsNeverTrue:
    """A substituted encoding timed as the requested one must refuse the file."""

    def test_fallback_used_true_is_rejected(self) -> None:
        with pytest.raises(schema.SchemaError, match="fallback_used=True"):
            schema.validate_mapping(valid_mapping(fallback_used=True))

    @pytest.mark.parametrize("value", [None, False])
    def test_none_and_false_are_accepted(self, value: bool | None) -> None:
        schema.validate_mapping(valid_mapping(fallback_used=value))


class TestRunHeader:
    """The header carries the whole build, not just its hash."""

    def test_header_is_json_serialisable_and_tagged(self) -> None:
        header = schema.run_header(
            run_id="t13_x",
            host="node",
            engine="cpp",
            build_info={"build_hash": "298fc1188bf1b051", "compiler": "gcc 12.2.0"},
            isalgraph_version="0.1.0",
            timestamp_utc="2026-08-26T12:00:00+00:00",
            source="cohort",
            shard=3,
            n_shards=64,
            arms=("default",),
            representations=("graph6",),
            budget_s=300.0,
            seed=13,
            symmetry_available=True,
        )
        assert header["record_kind"] == "header"
        assert header["build_info"]["compiler"] == "gcc 12.2.0"
        assert json.loads(json.dumps(header))["shard"] == 3

    def test_header_is_not_a_record(self) -> None:
        header = schema.run_header(
            run_id="t13_x",
            host="node",
            engine="cpp",
            build_info={},
            isalgraph_version="0.1.0",
            timestamp_utc="2026-08-26T12:00:00+00:00",
            source="cohort",
            shard=0,
            n_shards=1,
            arms=(),
            representations=(),
            budget_s=300.0,
            seed=13,
            symmetry_available=False,
        )
        with pytest.raises(schema.SchemaError):
            schema.validate_mapping(header)
