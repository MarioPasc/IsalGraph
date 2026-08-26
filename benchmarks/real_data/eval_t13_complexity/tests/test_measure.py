"""Tests for the T-13 runner.

Each class pins one of the runner's load-bearing rules.  Three of them exist
because the rule has already been got wrong once in this project's history:

- the budget must be a **killed subprocess**, because ``SIGALRM`` does not
  interrupt the C++ engine and therefore fails silently (T-05 finding 5);
- the engine gate must **abort**, because T-06's headline rates were retracted
  as unprovenanced;
- an ablation arm must **restore** the native toggles even when the timed call
  raises, which on a censored encode is the normal exit path.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest

from benchmarks.real_data.eval_t13_complexity import measure, schema


class FakeClock:
    """A clock returning a fixed sequence, so the timing rule can be pinned.

    ``timed_call`` reads the clock twice per run, so a sequence of ``2(1 + k)``
    values drives a warm-up plus *k* repeats.
    """

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        value = self.values[self.calls]
        self.calls += 1
        return value


class CountingCall:
    """A timed callable that records how many times the rule invoked it."""

    def __init__(self, value: int = 7) -> None:
        self.value = value
        self.count = 0

    def __call__(self) -> int:
        self.count += 1
        return self.value


class FakeNative:
    """Records every toggle call, so restoration can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def set_pairs_memo(self, on: bool, /) -> None:
        self.calls.append(("pairs_memo", on))

    def set_branch_and_bound(self, on: bool, /) -> None:
        self.calls.append(("branch_and_bound", on))


# ---------------------------------------------------------------------------


class TestRepresentationsResolve:
    """Criterion 2: the frozen thirteen all resolve through the registry."""

    def test_there_are_thirteen(self) -> None:
        assert len(measure.REPRESENTATIONS) == 13
        assert len(set(measure.REPRESENTATIONS)) == 13

    def test_all_resolve(self) -> None:
        resolved = measure.resolve_representations()
        unresolved = {k: v for k, v in resolved.items() if not v.startswith("ok")}
        assert unresolved == {}, f"did not resolve: {unresolved}"

    def test_size_null_is_named_not_discovered(self) -> None:
        """``size_null`` carries BASELINE, so ``available_backends()`` omits it.

        Discovering the list instead of naming it would drop the null arm and
        the figure would regenerate successfully with it absent.
        """
        from isalgraph.competitors.registry import available_backends

        assert "size_null" not in available_backends()
        assert "size_null" in measure.REPRESENTATIONS

    def test_unknown_name_is_reported_not_raised(self) -> None:
        resolved = measure.resolve_representations(["graph6", "not_a_backend"])
        assert resolved["graph6"].startswith("ok")
        assert resolved["not_a_backend"].startswith("UNRESOLVED")


class TestTimingRule:
    """Criterion 3: both branches of CONTRACTS §5.3.2, pinned with a fake clock."""

    def test_slow_warmup_is_reported_alone(self) -> None:
        calls = CountingCall()
        clock = FakeClock([0.0, 1.5])
        timing = measure.timed_call(calls, clock=clock)
        assert timing.repeats == 1
        assert timing.seconds == pytest.approx(1.5)
        assert timing.warmup_seconds == pytest.approx(1.5)
        assert calls.count == 1, "a warm-up at or above 1 s must not be repeated"
        assert clock.calls == 2

    def test_exactly_one_second_takes_the_slow_branch(self) -> None:
        """The threshold is ``>=``, so 1.0 s must not trigger three repeats."""
        clock = FakeClock([0.0, 1.0])
        assert measure.timed_call(lambda: 1, clock=clock).repeats == 1

    def test_fast_warmup_takes_the_median_of_three(self) -> None:
        calls = CountingCall()
        clock = FakeClock([0.0, 0.5, 0.0, 0.2, 0.0, 0.9, 0.0, 0.3])
        timing = measure.timed_call(calls, clock=clock)
        assert timing.repeats == 3
        assert timing.seconds == pytest.approx(0.3), "median of 0.2, 0.9, 0.3"
        assert timing.warmup_seconds == pytest.approx(0.5)
        assert calls.count == 4, "one warm-up plus three timed runs"
        assert clock.calls == 8

    def test_median_not_mean(self) -> None:
        """A right-tail outlier must not enter the reported time."""
        clock = FakeClock([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 90.0])
        timing = measure.timed_call(lambda: 1, clock=clock)
        assert timing.seconds == pytest.approx(0.1)

    def test_frozen_constants(self) -> None:
        assert measure.WARMUP_THRESHOLD_S == 1.0
        assert measure.REPEATS == 3
        assert measure.DEFAULT_BUDGET_S == 300.0


class TestBudgetIsAKilledSubprocess:
    """Criterion 4: the budget is enforced by killing a child, never by SIGALRM."""

    def test_slow_unit_is_censored_and_the_parent_survives(self) -> None:
        started = time.monotonic()
        result = measure.run_unit(
            {"unused": True},
            budget_s=0.2,
            argv=[sys.executable, "-c", "import time; time.sleep(60)"],
            grace_s=0.4,
        )
        elapsed = time.monotonic() - started

        assert result["status"] == "censored"
        assert result["error_kind"] == schema.KIND_WALLCLOCK
        assert result["seconds"] == 0.2, "a censored row reports the budget, not the kill"
        assert result["length_chars"] is None
        assert result["repeats"] == 0
        assert elapsed < 30.0, "the parent must not wait for the child's own 60 s"

        # The parent is still usable: it runs another unit to completion.
        alive = measure.run_unit(
            {"unused": True},
            budget_s=5.0,
            argv=[sys.executable, "-c", 'print(\'{"status": "ok"}\')'],
            grace_s=5.0,
        )
        assert alive["status"] == "ok"

    def test_a_censored_record_validates(self) -> None:
        """The kill's output must be a legal record, not just a dict."""
        result = measure.run_unit(
            {"unused": True},
            budget_s=0.2,
            argv=[sys.executable, "-c", "import time; time.sleep(60)"],
            grace_s=0.3,
        )
        record = measure.assemble_record(
            provenance=measure.Provenance(
                run_id="t",
                host="h",
                engine="cpp",
                build_hash=measure.EXPECTED_BUILD_HASH,
                isalgraph_version="0.1.0",
                timestamp_utc="2026-08-26T12:00:00+00:00",
            ),
            spec=measure.GraphSpec(source="constructed", family="star", n_target=64),
            properties={"n": 64, "m": 63, "density": 0.03, "max_degree": 63, "connected": True},
            symmetry=dict.fromkeys(schema.SYMMETRY_FIELDS),
            representation="isalgraph_pruned",
            arm="default",
            measurement=result,
            budget_s=0.2,
            spec_string="search_nodes=100000,max_projections=50000,timeout_s=0.2",
        )
        assert record.status == "censored"
        assert record.seconds == record.budget_s

    def test_a_crashing_child_is_an_error_row_not_a_hang(self) -> None:
        result = measure.run_unit(
            {"unused": True},
            budget_s=5.0,
            argv=[sys.executable, "-c", "raise SystemExit(3)"],
            grace_s=5.0,
        )
        assert result["status"] == "error"
        assert result["error_kind"] == "ChildExit3"

    def test_no_signal_based_timeout_is_installed(self) -> None:
        """T-05 finding 5: ``SIGALRM`` does not interrupt the C++ engine.

        A signal-based timeout therefore never fires, the encode runs to the
        job's wallclock, and the shard produces nothing -- with no error.  The
        prohibition is on the *mechanism*, so the check is that the module
        installs no alarm and imports no ``signal``; the string itself appears
        in the docstrings that explain why.
        """
        from pathlib import Path

        source = Path(measure.__file__).read_text(encoding="utf-8")
        assert "signal.alarm(" not in source
        assert "signal.setitimer(" not in source
        assert "import signal" not in source


class TestEngineGate:
    """Criterion 5: a wrong engine or a wrong build aborts, never warns."""

    def test_wrong_build_hash_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import isalgraph

        monkeypatch.setattr(isalgraph, "build_info", lambda: {"build_hash": "deadbeefdeadbeef"})
        with pytest.raises(measure.EngineMismatchError, match="build_hash"):
            measure.assert_engine()

    def test_python_engine_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import isalgraph

        monkeypatch.setattr(isalgraph, "engine", lambda: "python")
        with pytest.raises(measure.EngineMismatchError, match="pure-Python"):
            measure.assert_engine()

    def test_the_real_environment_passes(self) -> None:
        info = measure.assert_engine()
        assert info["build_hash"] == measure.EXPECTED_BUILD_HASH


class TestAblationArmsRestoreState:
    """Criterion 6: both toggles return to ``True``, including on an exception."""

    @pytest.mark.parametrize(
        ("arm", "expected"),
        [
            ("default", (True, True)),
            ("no_pairs_memo", (False, True)),
            ("no_bnb", (True, False)),
            ("no_pairs_memo_no_bnb", (False, False)),
        ],
    )
    def test_arm_settings(self, arm: str, expected: tuple[bool, bool]) -> None:
        assert measure.arm_settings(arm) == expected

    def test_unknown_arm_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown arm"):
            measure.arm_settings("no_memo")

    def test_state_is_applied_then_restored(self) -> None:
        native = FakeNative()
        with measure.engine_arm("no_pairs_memo_no_bnb", native=native):
            pass
        assert native.calls == [
            ("pairs_memo", False),
            ("branch_and_bound", False),
            ("pairs_memo", True),
            ("branch_and_bound", True),
        ]

    def test_state_is_restored_when_the_timed_call_raises(self) -> None:
        """The normal exit path for a censored encode is an exception."""
        native = FakeNative()
        with pytest.raises(RuntimeError, match="boom"), measure.engine_arm("no_bnb", native=native):
            raise RuntimeError("boom")
        assert native.calls[-2:] == [("pairs_memo", True), ("branch_and_bound", True)]

    def test_the_real_engine_is_left_on_after_an_ablation(self) -> None:
        import importlib

        _native = importlib.import_module("isalgraph.core._native")

        with pytest.raises(RuntimeError), measure.engine_arm("no_pairs_memo_no_bnb"):
            assert _native.pairs_memo() is False
            assert _native.branch_and_bound() is False
            raise RuntimeError
        assert _native.pairs_memo() is True
        assert _native.branch_and_bound() is True

    def test_ablation_arms_only_reach_representations_the_toggles_change(self) -> None:
        spec = measure.GraphSpec(source="constructed", family="cycle", n_target=12, replicate=0)
        units = measure.units_for_graph(
            spec,
            representations=measure.REPRESENTATIONS,
            arms=("default", "no_bnb"),
            ablation_keys=frozenset({spec.key}),
        )
        ablated = {u.representation for u in units if u.arm == "no_bnb"}
        assert ablated == set(measure.ABLATABLE_REPRESENTATIONS)

    def test_ablation_subsample_is_stratified_and_order_independent(self) -> None:
        specs = [
            (measure.GraphSpec(source="cohort", dataset="d", graph_index=i), 4 + (i % 3))
            for i in range(60)
        ]
        chosen = measure.select_ablation_graphs(specs)
        reversed_choice = measure.select_ablation_graphs(list(reversed(specs)))
        assert chosen == reversed_choice
        # Three strata (n = 4, 5, 6), two graphs each.
        assert len(chosen) == 3 * measure.ABLATION_PER_STRATUM


class TestDecliningBackendsAreRecorded:
    """Criterion 7: a declined graph is ``unsupported``, never dropped."""

    def test_agm_cam_declines_above_suite_one(self) -> None:
        import networkx as nx

        graph = nx.cycle_graph(13)
        result = measure.execute_unit(
            graph=graph,
            representation="agm_cam",
            arm="default",
            budget_s=30.0,
            budget=measure.budget_fields(n_nodes=13, budget_s=30.0),
        )
        assert result["status"] == "unsupported"
        assert result["error_kind"] == "SuiteScopeError"
        assert result["length_chars"] is None

    def test_isalgraph_canonical_declines_above_suite_one(self) -> None:
        import networkx as nx

        result = measure.execute_unit(
            graph=nx.cycle_graph(13),
            representation="isalgraph_canonical",
            arm="default",
            budget_s=30.0,
            budget=measure.budget_fields(n_nodes=13, budget_s=30.0),
        )
        assert result["status"] == "unsupported"
        assert result["error_kind"] == "SuiteScopeError"

    def test_the_same_graph_is_supported_at_n_twelve(self) -> None:
        import networkx as nx

        result = measure.execute_unit(
            graph=nx.cycle_graph(12),
            representation="agm_cam",
            arm="default",
            budget_s=30.0,
            budget=measure.budget_fields(n_nodes=12, budget_s=30.0),
        )
        assert result["status"] == "ok"
        assert result["length_chars"] == 12 * 11 // 2

    def test_an_unsupported_row_validates(self) -> None:
        import networkx as nx

        result = measure.execute_unit(
            graph=nx.cycle_graph(13),
            representation="agm_cam",
            arm="default",
            budget_s=30.0,
            budget=measure.budget_fields(n_nodes=13, budget_s=30.0),
        )
        record = measure.assemble_record(
            provenance=measure.Provenance(
                run_id="t",
                host="h",
                engine="cpp",
                build_hash=measure.EXPECTED_BUILD_HASH,
                isalgraph_version="0.1.0",
                timestamp_utc="2026-08-26T12:00:00+00:00",
            ),
            spec=measure.GraphSpec(source="constructed", family="cycle", n_target=13),
            properties=measure.graph_properties(nx.cycle_graph(13)),
            symmetry=dict.fromkeys(schema.SYMMETRY_FIELDS),
            representation="agm_cam",
            arm="default",
            measurement=result,
            budget_s=30.0,
            spec_string="search_nodes=100000,max_projections=50000,timeout_s=30.0",
        )
        assert record.status == "unsupported"


class TestSharding:
    """Criterion 8: a deterministic hash partitions the grid exactly."""

    @staticmethod
    def grid() -> tuple[measure.WorkUnit, ...]:
        units: list[measure.WorkUnit] = []
        for dataset in ("iam_letter_low", "linux"):
            for index in range(40):
                spec = measure.GraphSpec(source="cohort", dataset=dataset, graph_index=index)
                units.extend(
                    measure.units_for_graph(
                        spec,
                        representations=measure.REPRESENTATIONS,
                        arms=("default",),
                        ablation_keys=frozenset(),
                    )
                )
        return tuple(units)

    @pytest.mark.parametrize("n_shards", [1, 7, 64])
    def test_partition_is_exact(self, n_shards: int) -> None:
        units = self.grid()
        keys = {u.key for u in units}
        assert len(keys) == len(units), "work-unit keys must be unique"

        buckets: dict[int, set[str]] = {k: set() for k in range(n_shards)}
        for unit in units:
            buckets[measure.shard_of(unit.key, n_shards)].add(unit.key)

        union: set[str] = set()
        total = 0
        for members in buckets.values():
            assert not (union & members), "shards must not overlap"
            union |= members
            total += len(members)
        assert union == keys, "no unit may be lost"
        assert total == len(keys)

    def test_membership_does_not_depend_on_order(self) -> None:
        units = self.grid()
        forward = [measure.shard_of(u.key, 64) for u in units]
        backward = [measure.shard_of(u.key, 64) for u in reversed(units)]
        assert forward == list(reversed(backward))

    def test_digest_is_stable_across_processes(self) -> None:
        """``hash()`` is salted per process; ``blake2b`` is not."""
        assert measure.unit_digest("cohort|linux|3|graph6|default") == measure.unit_digest(
            "cohort|linux|3|graph6|default"
        )
        assert measure.shard_of("a", 64) == measure.unit_digest("a") % 64

    def test_n_shards_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_shards"):
            measure.shard_of("a", 0)

    @pytest.mark.parametrize("n_shards", [1, 7, 64])
    def test_every_shard_is_reachable(self, n_shards: int) -> None:
        occupied = {measure.shard_of(u.key, n_shards) for u in self.grid()}
        assert occupied == set(range(n_shards))


class TestBudgets:
    """One fully populated budget, threaded through every backend."""

    def test_min_dfs_cap_is_always_set(self) -> None:
        """A ``None`` here runs min-DFS unbounded and re-opens the OOM kill."""
        for n in (4, 12, 13, 64):
            fields = measure.budget_fields(n_nodes=n, budget_s=300.0)
            assert fields["max_projections"] == measure.MIN_DFS_MAX_PROJECTIONS
            assert fields["timeout_s"] == 300.0

    def test_agm_node_budget_is_suite_conditional(self) -> None:
        assert measure.budget_fields(n_nodes=12, budget_s=1.0)["search_nodes"] == 200_000
        assert measure.budget_fields(n_nodes=13, budget_s=1.0)["search_nodes"] == 100_000

    def test_frozen_values_match_the_backends(self) -> None:
        from isalgraph.competitors.backends import agm, min_dfs

        assert measure.AGM_SEARCH_NODES_SUITE1 == agm.SUITE1_NODE_BUDGET
        assert measure.AGM_SEARCH_NODES_SUITE2 == agm.SUITE2_NODE_BUDGET
        assert measure.MIN_DFS_MAX_PROJECTIONS == min_dfs.MAX_PROJECTIONS
        assert measure.SUITE1_MAX_NODES == agm.SUITE1_MAX_NODES

    def test_spec_string_is_complete(self) -> None:
        fields = measure.budget_fields(n_nodes=8, budget_s=300.0)
        rendered = measure.budget_spec(fields)
        for key in ("search_nodes", "max_projections", "timeout_s"):
            assert key in rendered


class TestCanonicalIdentityGate:
    """The two exhaustive-canonical arms must agree wherever both may run."""

    @staticmethod
    def record(representation: str, *, status: str, length: int | None, n: int = 10) -> Any:
        return measure.assemble_record(
            provenance=measure.Provenance(
                run_id="t",
                host="h",
                engine="cpp",
                build_hash=measure.EXPECTED_BUILD_HASH,
                isalgraph_version="0.1.0",
                timestamp_utc="2026-08-26T12:00:00+00:00",
            ),
            spec=measure.GraphSpec(source="constructed", family="cycle", n_target=n, replicate=0),
            properties={
                "n": n,
                "m": n,
                "density": 0.2,
                "max_degree": 2,
                "connected": True,
            },
            symmetry=dict.fromkeys(schema.SYMMETRY_FIELDS),
            representation=representation,
            arm="default",
            measurement={
                "status": status,
                "error_kind": None if status == "ok" else "SuiteScopeError",
                "seconds": 0.001,
                "repeats": 3 if status == "ok" else 0,
                "length_chars": length,
                "fallback_used": False,
            },
            budget_s=300.0,
            spec_string="search_nodes=200000,max_projections=50000,timeout_s=300.0",
        )

    def test_agreement_reports_nothing(self) -> None:
        records = [
            self.record("isalgraph_canonical", status="ok", length=17),
            self.record("isalgraph_exhaustive", status="ok", length=17),
        ]
        assert measure.canonical_identity_violations(records) == ()

    def test_a_length_divergence_is_reported(self) -> None:
        records = [
            self.record("isalgraph_canonical", status="ok", length=17),
            self.record("isalgraph_exhaustive", status="ok", length=19),
        ]
        violations = measure.canonical_identity_violations(records)
        assert len(violations) == 1
        assert "17" in violations[0] and "19" in violations[0]

    def test_above_the_guard_the_gate_is_silent(self) -> None:
        """Above n = 12 only one arm may run, so there is nothing to compare."""
        records = [
            self.record("isalgraph_canonical", status="unsupported", length=None, n=20),
            self.record("isalgraph_exhaustive", status="ok", length=41, n=20),
        ]
        assert measure.canonical_identity_violations(records) == ()

    def test_the_two_arms_really_do_agree_on_a_real_graph(self) -> None:
        import networkx as nx

        graph = nx.petersen_graph()
        budget = measure.budget_fields(n_nodes=10, budget_s=30.0)
        left = measure.execute_unit(
            graph=graph,
            representation="isalgraph_canonical",
            arm="default",
            budget_s=30.0,
            budget=budget,
        )
        right = measure.execute_unit(
            graph=graph,
            representation="isalgraph_exhaustive",
            arm="default",
            budget_s=30.0,
            budget=budget,
        )
        assert left["status"] == right["status"] == "ok"
        assert left["length_chars"] == right["length_chars"]


class TestGraphProperties:
    """The five structural fields."""

    def test_cycle(self) -> None:
        import networkx as nx

        props = measure.graph_properties(nx.cycle_graph(6))
        assert props == {
            "n": 6,
            "m": 6,
            "density": pytest.approx(0.4),
            "max_degree": 2,
            "connected": True,
        }

    def test_single_node_has_zero_density(self) -> None:
        import networkx as nx

        props = measure.graph_properties(nx.empty_graph(1))
        assert props["density"] == 0.0
        assert props["max_degree"] == 0

    def test_disconnected(self) -> None:
        import networkx as nx

        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (2, 3)])
        assert measure.graph_properties(graph)["connected"] is False


class TestTrackADegradation:
    """The runner degrades explicitly where track A's modules are absent."""

    def test_missing_symmetry_nulls_the_nine_fields(self) -> None:
        import networkx as nx

        fields = measure.symmetry_fields(nx.cycle_graph(5), available=False)
        assert set(fields) == set(schema.SYMMETRY_FIELDS)
        assert all(value is None for value in fields.values())

    def test_constructed_source_refuses_by_name_when_families_is_absent(self) -> None:
        import importlib

        pytest.importorskip("networkx")
        try:
            importlib.import_module("benchmarks.real_data.eval_t13_complexity.families")
        except ImportError:
            with pytest.raises(measure.TrackAMissingError, match="families.py"):
                measure.build_grid("constructed", datasets=None, seed=13)
        else:  # pragma: no cover - after track A merges
            pytest.skip("families.py is present; the degradation path cannot be exercised")

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown source"):
            measure.build_grid("synthetic", datasets=None, seed=13)


class TestOutputPaths:
    """One append-safe file per shard."""

    def test_directory_gets_the_canonical_name(self, tmp_path: Any) -> None:
        path = measure.resolve_out_path(tmp_path, source="cohort", shard=3, n_shards=64)
        assert path.name == "records_cohort_3of64.jsonl"

    def test_explicit_jsonl_is_used_verbatim(self, tmp_path: Any) -> None:
        target = tmp_path / "smoke.jsonl"
        assert measure.resolve_out_path(target, source="cohort", shard=0, n_shards=1) == target

    def test_resume_skips_recorded_units(self, tmp_path: Any) -> None:
        path = tmp_path / "shard.jsonl"
        record = measure.assemble_record(
            provenance=measure.Provenance(
                run_id="t",
                host="h",
                engine="cpp",
                build_hash=measure.EXPECTED_BUILD_HASH,
                isalgraph_version="0.1.0",
                timestamp_utc="2026-08-26T12:00:00+00:00",
            ),
            spec=measure.GraphSpec(source="cohort", dataset="linux", graph_index=4),
            properties={"n": 5, "m": 4, "density": 0.4, "max_degree": 2, "connected": True},
            symmetry=dict.fromkeys(schema.SYMMETRY_FIELDS),
            representation="graph6",
            arm="default",
            measurement={
                "status": "ok",
                "error_kind": None,
                "seconds": 1e-5,
                "repeats": 3,
                "length_chars": 3,
                "fallback_used": None,
            },
            budget_s=300.0,
            spec_string="search_nodes=200000,max_projections=50000,timeout_s=300.0",
        )
        path.write_text(record.to_json_line(), encoding="utf-8")
        keys = measure.existing_unit_keys(path)
        spec = measure.GraphSpec(source="cohort", dataset="linux", graph_index=4)
        assert measure.WorkUnit(spec, "graph6", "default").record_key in keys
        assert measure.WorkUnit(spec, "graph6", "no_bnb").record_key not in keys

    def test_a_truncated_trailing_line_is_skipped_not_fatal(self, tmp_path: Any) -> None:
        """The signature of a task killed mid-write."""
        path = tmp_path / "shard.jsonl"
        path.write_text('{"record_kind": "header"}\n{"source": "coh', encoding="utf-8")
        assert measure.existing_unit_keys(path) == set()

    def test_absent_file_resumes_from_empty(self, tmp_path: Any) -> None:
        assert measure.existing_unit_keys(tmp_path / "nope.jsonl") == set()
