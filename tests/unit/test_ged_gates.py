"""Unit tests for the four T-03 validation gates.

None of these tests touch GEDLIB, ``torch`` or the source data tree. The gates
are driven with :class:`StubBackend` and with fabricated Contract A files, so
the pass/fail logic, the cost-model guard and the report schema are all covered
on a machine that has neither the library nor the cluster.

What is deliberately *not* covered here is whether GEDLIB itself returns the
right numbers. That is what gates 2 and 3 measure, and they can only be
measured where the library exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

from benchmarks.eval_setup.ged_backends import BackendSpec, PairResult, StubBackend
from benchmarks.eval_setup.ged_bounds import GRAPHEDX_COSTS, UNIT_COSTS
from benchmarks.eval_setup.ged_gates import (
    COHORT_SIZES,
    GATE2_ARCHIVE,
    GateError,
    GateResult,
    LoadedDataset,
    _json_default,
    _merge_stats,
    _quantiles,
    environment_record,
    evaluate_pairs,
    gate0_graphedx_agreement,
    gate1_bracket,
    gate2_replay,
    gate3_exact_benchmark,
    load_contract_a,
    load_gate2_archive,
    main,
    sample_pairs,
    write_report,
)

STUB = BackendSpec(kind="stub", costs=UNIT_COSTS)
STUB_GRAPHEDX = BackendSpec(kind="stub", costs=GRAPHEDX_COSTS)
NX_GRAPHEDX = BackendSpec(kind="networkx", costs=GRAPHEDX_COSTS, options={"timeout_s": 10.0})
NX_SPEC = BackendSpec(kind="networkx", costs=UNIT_COSTS, options={"timeout_s": 5.0})


def _toy_graphs(n: int, seed: int = 0) -> list[nx.Graph]:
    """Return ``n`` small connected graphs, deterministically."""
    rng = np.random.default_rng(seed)
    out: list[nx.Graph] = []
    while len(out) < n:
        size = int(rng.integers(3, 7))
        g = nx.gnp_random_graph(size, 0.55, seed=int(rng.integers(0, 10**6)))
        if nx.is_connected(g):
            out.append(g)
    return out


def write_contract_a(path: Path, graphs: list[nx.Graph], key: str = "toy") -> Path:
    """Fabricate a Contract A ``.npz`` from the documented key table.

    ``task-export`` owns the real writer; consumers fabricate their own until
    it merges, which is what CONTRACTS §4 instructs.
    """
    n_nodes = np.array([g.number_of_nodes() for g in graphs], dtype=np.int32)
    n_edges = np.array([g.number_of_edges() for g in graphs], dtype=np.int32)
    offsets = np.zeros(len(graphs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_edges)
    cols = []
    for g in graphs:
        for u, v in g.edges():
            cols.append((min(u, v), max(u, v)))
    edges = np.array(cols, dtype=np.int32).T if cols else np.zeros((2, 0), dtype=np.int32)
    metadata = json.dumps(
        {
            "dataset": key,
            "source": "fabricated",
            "n_kept": len(graphs),
            "n_pairs": len(graphs) * (len(graphs) - 1) // 2,
            "schema_version": 1,
        }
    )
    np.savez_compressed(
        path,
        graph_ids=np.array([f"{key}_{i:04d}" for i in range(len(graphs))]),
        n_nodes=n_nodes,
        n_edges=n_edges,
        edge_offsets=offsets,
        edges=edges,
        splits=np.array(["train"] * len(graphs)),
        labels=np.array([""] * len(graphs)),
        metadata=np.array(metadata),
    )
    return path


class TestContractALoader:
    """Reading the exported file without importing export_graphs."""

    def test_round_trip_preserves_every_edge(self, tmp_path: Path) -> None:
        graphs = _toy_graphs(8, seed=3)
        path = write_contract_a(tmp_path / "toy.npz", graphs)
        loaded = load_contract_a(path)
        assert len(loaded.graphs) == 8
        for original, restored in zip(graphs, loaded.graphs, strict=True):
            assert restored.number_of_nodes() == original.number_of_nodes()
            assert set(map(frozenset, restored.edges())) == set(map(frozenset, original.edges()))

    def test_a_missing_key_is_reported_not_guessed(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.npz"
        np.savez_compressed(path, graph_ids=np.array(["a"]))
        with pytest.raises(GateError, match="not a Contract A file"):
            load_contract_a(path)

    def test_inconsistent_offsets_are_caught(self, tmp_path: Path) -> None:
        graphs = _toy_graphs(3, seed=1)
        path = write_contract_a(tmp_path / "toy.npz", graphs)
        with np.load(path, allow_pickle=False) as z:
            data = dict(z)
        data["edge_offsets"] = data["edge_offsets"] + 1
        np.savez_compressed(path, **data)
        with pytest.raises(GateError, match="edge_offsets"):
            load_contract_a(path)

    def test_the_cohort_table_is_the_frozen_one(self) -> None:
        assert COHORT_SIZES == {
            "iam_letter_low": 1180,
            "iam_letter_med": 1253,
            "iam_letter_high": 2059,
            "linux": 89,
            "aids": 769,
        }
        assert sum(n * (n - 1) // 2 for n in COHORT_SIZES.values()) == 3_897_911


class TestSampling:
    """Pair sampling must be seeded, distinct and filterable."""

    def test_the_sample_is_reproducible(self) -> None:
        assert sample_pairs(50, 20, 42) == sample_pairs(50, 20, 42)

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        assert sample_pairs(50, 20, 42) != sample_pairs(50, 20, 7)

    def test_pairs_are_distinct_and_unordered(self) -> None:
        pairs = sample_pairs(30, 25, 1)
        assert len(set(pairs)) == len(pairs)
        assert all(i < j for i, j in pairs)

    def test_the_predicate_is_respected(self) -> None:
        pairs = sample_pairs(40, 20, 5, allowed=lambda i, j: (i + j) % 2 == 0)
        assert all((i + j) % 2 == 0 for i, j in pairs)

    def test_it_terminates_when_the_population_is_exhausted(self) -> None:
        assert len(sample_pairs(3, 100, 0)) <= 3


class TestEvaluatePairs:
    """The serial and pooled paths must agree."""

    def test_serial_and_pooled_results_match(self) -> None:
        graphs = _toy_graphs(6, seed=9)
        pairs = [(graphs[i], graphs[j]) for i, j in sample_pairs(6, 8, 0)]
        serial, _ = evaluate_pairs(STUB, pairs, workers=1)
        pooled, _ = evaluate_pairs(STUB, pairs, workers=2)
        assert [r["idx"] for r in pooled] == list(range(len(pairs)))
        for a, b in zip(serial, pooled, strict=True):
            assert (a["lb"], a["ub"], a["exact"]) == (b["lb"], b["ub"], b["exact"])

    def test_the_independent_bracket_is_attached_on_request(self) -> None:
        graphs = _toy_graphs(4, seed=2)
        records, _ = evaluate_pairs(STUB, [(graphs[0], graphs[1])], workers=1, independent=True)
        rec = records[0]
        assert rec["ind_lb"] <= rec["ind_ub"]
        assert rec["ind_ub"] == min(rec["ind_ub_fwd"], rec["ind_ub_rev"])

    def test_stats_merge_additively_and_rederive_rates(self) -> None:
        merged = _merge_stats(
            [
                {
                    "n_pairs": 2,
                    "n_certified": 1,
                    "certification_rate": 0.5,
                    "n_ub_orientations_compared": 2,
                    "n_ub_asymmetric": 1,
                    "ub_asymmetry_rate": 0.5,
                    "max_ub_gap": 3.0,
                    "total_seconds": 1.0,
                },
                {
                    "n_pairs": 2,
                    "n_certified": 2,
                    "certification_rate": 1.0,
                    "n_ub_orientations_compared": 2,
                    "n_ub_asymmetric": 0,
                    "ub_asymmetry_rate": 0.0,
                    "max_ub_gap": 7.0,
                    "total_seconds": 3.0,
                },
            ]
        )
        assert merged["n_pairs"] == 4
        assert merged["certification_rate"] == 0.75
        assert merged["ub_asymmetry_rate"] == 0.25
        assert merged["max_ub_gap"] == 7.0
        assert merged["mean_seconds"] == 1.0


class TestGate0:
    """Agreement with GraphEdX, under GraphEdX's own cost model."""

    @staticmethod
    def _fixture(n: int = 6) -> tuple[LoadedDataset, np.ndarray]:
        """Build a dataset whose published matrix is exactly reproducible.

        The published values come from a backend that certifies. The stub
        almost never closes its bracket, so a stub-built matrix would leave
        every delta undefined and the sign tests would pass vacuously.
        """
        from benchmarks.eval_setup.ged_backends import NetworkxBackend

        graphs = _toy_graphs(n, seed=11)
        ds = LoadedDataset(
            key="aids",
            graphs=graphs,
            graph_ids=[f"aids_train_{i:04d}" for i in range(n)],
            splits=["train"] * n,
        )
        backend = NetworkxBackend(GRAPHEDX_COSTS, timeout_s=10.0)
        published = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = backend.pair(graphs[i], graphs[j])
                assert r.exact is not None, "fixture needs certified reference values"
                published[i, j] = published[j, i] = r.exact
        return ds, published

    def test_the_production_cost_model_is_refused(self) -> None:
        """Running gate 0 under D6 guarantees a mismatch that looks like a bug."""
        ds, published = self._fixture()
        with pytest.raises(GateError, match="GRAPHEDX_COSTS"):
            gate0_graphedx_agreement(STUB, ds, published, n_pairs=5)

    def test_the_cost_vector_is_taken_from_graphedx_costs(self) -> None:
        assert GRAPHEDX_COSTS.as_gedlib_constant() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]

    def test_a_disagreement_is_recorded_with_its_sign(self) -> None:
        ds, published = self._fixture()
        published[0, 1] = published[1, 0] = published[0, 1] + 3.0
        result = gate0_graphedx_agreement(NX_GRAPHEDX, ds, published, n_pairs=15, seed=0)
        deltas = [r["delta"] for r in result.details["records"] if r["delta"] is not None]
        assert any(d < 0 for d in deltas), "ours below theirs must be visible"
        assert result.details["n_ours_lower"] >= 1
        assert set(result.details["signed_delta"]) >= {"min", "median", "max", "mean"}

    def test_a_systematic_offset_is_interpreted(self) -> None:
        ds, published = self._fixture()
        published[np.triu_indices_from(published, 1)] += 5.0
        published = np.triu(published) + np.triu(published, 1).T
        result = gate0_graphedx_agreement(NX_GRAPHEDX, ds, published, n_pairs=15, seed=0)
        assert not result.passed
        assert "systematically below" in result.details.get("interpretation", "")

    def test_only_within_split_pairs_are_drawn(self) -> None:
        ds, published = self._fixture(6)
        ds.splits = ["train", "train", "train", "test", "test", "test"]
        published[0:3, 3:6] = math.inf
        published[3:6, 0:3] = math.inf
        result = gate0_graphedx_agreement(NX_GRAPHEDX, ds, published, n_pairs=20, seed=0)
        for rec in result.details["records"]:
            i = int(rec["graph_i"].split("_")[-1])
            j = int(rec["graph_j"].split("_")[-1])
            assert ds.splits[i] == ds.splits[j]


class TestGate1:
    """The bracket gate must be non-vacuous and must catch a violation."""

    def test_it_passes_on_a_consistent_backend(self) -> None:
        graphs = _toy_graphs(6, seed=4)
        pairs = [(graphs[i], graphs[j]) for i, j in sample_pairs(6, 10, 0)]
        result = gate1_bracket(NX_SPEC, pairs)
        assert result.passed
        assert result.n_pairs > 0, "a gate that certifies nothing proves nothing"
        assert result.details["n_violations"] == 0

    def test_it_catches_an_exact_value_below_the_lower_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import benchmarks.eval_setup.ged_gates as mod

        def liar(
            backend: object,
            costs: object,
            idx: int,
            g1: nx.Graph,
            g2: nx.Graph,
            independent: bool,
        ) -> dict[str, Any]:
            return {
                "idx": idx,
                "n1": 3,
                "n2": 3,
                "m1": 2,
                "m2": 2,
                "lb": 0.0,
                "ub": 0.0,
                "exact": 0.0,
                "certified": True,
                "seconds": 0.0,
                "timed_out": False,
                "method": "liar",
                "ind_lb": 4.0,
                "ind_ub": 9.0,
            }

        monkeypatch.setattr(mod, "_pair_payload", liar)
        graphs = _toy_graphs(3, seed=6)
        result = mod.gate1_bracket(STUB, [(graphs[0], graphs[1])])
        assert not result.passed
        assert result.details["violations"][0]["reason"] == ("exact below independent lower bound")

    def test_slack_distributions_are_reported(self) -> None:
        graphs = _toy_graphs(5, seed=8)
        pairs = [(graphs[i], graphs[j]) for i, j in sample_pairs(5, 6, 0)]
        result = gate1_bracket(NX_SPEC, pairs)
        assert set(result.details["independent_lb_slack"]) >= {"median", "max"}
        assert set(result.details["independent_ub_slack"]) >= {"median", "max"}


class TestGate2:
    """Replay of the archived LINUX sample."""

    def test_the_archive_is_present_and_well_formed(self) -> None:
        archive = load_gate2_archive(GATE2_ARCHIVE)
        assert len(archive["pairs"]) == 400
        assert archive["report"]["n_violations"] == 0
        assert archive["report"]["dataset"] == "LINUX"

    def test_a_missing_archive_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(GateError, match="not found"):
            load_gate2_archive(tmp_path / "absent.json")

    @staticmethod
    def _archive_from(pairs: list[tuple[nx.Graph, nx.Graph]]) -> dict[str, Any]:
        from benchmarks.eval_setup.ged_bounds import (
            bipartite_upper_bound,
            branch_lower_bound,
            exact_ged,
        )

        records = []
        for k, (g1, g2) in enumerate(pairs):
            records.append(
                {
                    "idx": k,
                    "n1": g1.number_of_nodes(),
                    "n2": g2.number_of_nodes(),
                    "m1": g1.number_of_edges(),
                    "m2": g2.number_of_edges(),
                    "lb": branch_lower_bound(g1, g2, UNIT_COSTS),
                    # The real archive predates symmetrisation, so a faithful
                    # stand-in must record the forward orientation alone.
                    "ub": bipartite_upper_bound(g1, g2, UNIT_COSTS, symmetrise=False),
                    "exact": exact_ged(g1, g2, UNIT_COSTS),
                }
            )
        return {"report": {"dataset": "TOY"}, "pairs": records}

    def test_an_identical_replay_passes(self) -> None:
        graphs = _toy_graphs(5, seed=12)
        pairs = [(graphs[i], graphs[j]) for i, j in sample_pairs(5, 6, 0)]
        result = gate2_replay(NX_SPEC, pairs, self._archive_from(pairs))
        assert result.passed
        assert result.details["n_identity_failures"] == 0
        assert result.details["n_bound_mismatch"] == 0

    def test_a_different_sample_fails_identity(self) -> None:
        graphs = _toy_graphs(6, seed=13)
        pairs = [(graphs[i], graphs[j]) for i, j in sample_pairs(6, 6, 0)]
        archive = self._archive_from(pairs)
        archive["pairs"][0]["n1"] += 3
        result = gate2_replay(NX_SPEC, pairs, archive)
        assert not result.passed
        assert result.details["n_identity_failures"] == 1

    def test_an_archived_value_above_a_new_optimum_is_evidence_not_failure(self) -> None:
        """A timed-out archive entry is the defect this gate exists to expose."""
        graphs = _toy_graphs(4, seed=14)
        pairs = [(graphs[0], graphs[1])]
        archive = self._archive_from(pairs)
        archive["pairs"][0]["exact"] += 4.0
        archive["pairs"][0]["ub"] = max(archive["pairs"][0]["ub"], archive["pairs"][0]["exact"])
        result = gate2_replay(NX_SPEC, pairs, archive)
        assert result.details["n_archive_suboptimal"] == 1
        assert "best-so-far" in result.details["interpretation"]

    def test_a_new_value_above_the_archive_is_a_hard_failure(self) -> None:
        graphs = _toy_graphs(4, seed=15)
        pairs = [(graphs[0], graphs[1])]
        archive = self._archive_from(pairs)
        archive["pairs"][0]["exact"] -= 1.0
        result = gate2_replay(NX_SPEC, pairs, archive)
        assert not result.passed
        assert result.details["n_exact_regression"] == 1


class TestGate3:
    """Solver agreement and the timing table that sizes the production run."""

    def test_it_agrees_with_itself_and_stratifies_the_timings(self) -> None:
        graphs = _toy_graphs(5, seed=16)
        triples = [("toy", graphs[i], graphs[j]) for i, j in sample_pairs(5, 6, 0)]
        result = gate3_exact_benchmark(NX_SPEC, triples, timeout=10.0)
        assert result.passed
        assert result.details["n_disagreements"] == 0
        for stratum in result.details["strata"].values():
            assert stratum["n_pairs"] >= 1
            assert set(stratum["backend_seconds"]) >= {"median", "mean"}
            assert stratum["speedup_median"] is None or stratum["speedup_median"] > 0

    def test_a_networkx_run_is_flagged_as_not_the_real_benchmark(self) -> None:
        graphs = _toy_graphs(3, seed=17)
        triples = [("toy", graphs[0], graphs[1])]
        result = gate3_exact_benchmark(NX_SPEC, triples, timeout=10.0)
        assert result.details["benchmark_meaningful"] is False
        assert "not the ANCHOR_AWARE_GED benchmark" in result.details["benchmark_note"]

    def test_a_disagreement_fails_the_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import benchmarks.eval_setup.ged_gates as mod

        monkeypatch.setattr(mod, "_astar_reference", lambda *a, **k: (999.0, 0.01, False))
        graphs = _toy_graphs(3, seed=18)
        result = mod.gate3_exact_benchmark(NX_SPEC, [("toy", graphs[0], graphs[1])], timeout=5.0)
        assert not result.passed
        assert result.details["n_disagreements"] == 1

    def test_a_cut_off_reference_yields_no_value(self) -> None:
        import benchmarks.eval_setup.ged_gates as mod

        value, seconds, timed_out = mod._astar_reference(
            nx.star_graph(5), nx.cycle_graph(6), UNIT_COSTS, 1e-6
        )
        assert seconds >= 0.0
        assert value is None or not timed_out


class TestReporting:
    """The report schema and the exit code."""

    def test_a_report_is_written_and_reloadable(self, tmp_path: Path) -> None:
        result = GateResult(gate="9", passed=True, n_pairs=3, seconds=1.5, details={"a": 1})
        path = write_report(result, tmp_path, {"host": "x"})
        payload = json.loads(path.read_text())
        assert payload["gate"] == "9" and payload["passed"] is True
        assert payload["environment"]["host"] == "x"

    def test_infinities_survive_serialisation(self, tmp_path: Path) -> None:
        result = GateResult(
            gate="9",
            passed=True,
            n_pairs=1,
            seconds=0.0,
            details={"v": np.float64(math.inf), "arr": np.arange(3)},
        )
        payload = json.loads(write_report(result, tmp_path, {}).read_text())
        assert payload["details"]["v"] == "inf"
        assert payload["details"]["arr"] == [0, 1, 2]

    def test_json_default_handles_numpy_scalars(self) -> None:
        assert _json_default(np.int32(4)) == 4
        assert _json_default(np.float64(1.5)) == 1.5

    def test_environment_names_the_thread_settings(self) -> None:
        env = environment_record(workers=3)
        assert env["workers"] == 3
        assert "OMP_NUM_THREADS" in env
        assert env["networkx"] == nx.__version__

    def test_quantiles_of_an_empty_sample_are_empty(self) -> None:
        assert _quantiles([]) == {}

    def test_a_gate_that_cannot_run_fails_rather_than_crashing(self, tmp_path: Path) -> None:
        """A missing input must produce a failing report, not a traceback."""
        code = main(
            [
                "--gate",
                "0",
                "--backend",
                "stub",
                "--input-dir",
                str(tmp_path),
                "--source-dir",
                str(tmp_path / "absent"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert code == 1
        payload = json.loads((tmp_path / "out" / "gate0.json").read_text())
        assert payload["passed"] is False
        assert "error" in payload["details"]

    def test_the_summary_records_every_gate(self, tmp_path: Path) -> None:
        main(
            [
                "--gate",
                "2",
                "--backend",
                "stub",
                "--source-dir",
                str(tmp_path / "absent"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        summary = json.loads((tmp_path / "out" / "gates_summary.json").read_text())
        assert "2" in summary["gates"]
        assert summary["all_passed"] is False


class TestPairResultIsWhatGatesConsume:
    """The gates read exactly the seven Contract B fields."""

    def test_the_payload_carries_every_field(self) -> None:
        import benchmarks.eval_setup.ged_gates as mod

        graphs = _toy_graphs(2, seed=19)
        rec = mod._pair_payload(StubBackend(), UNIT_COSTS, 0, graphs[0], graphs[1], False)
        for name in PairResult.__slots__:
            assert name in rec
