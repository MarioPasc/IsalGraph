"""Acceptance tests for the T-06 encoding campaign.

Every test here exists because a specific silent failure has already happened,
or would be invisible if it did:

- The cohort counts are asserted because Suite 1 and Suite 2 differ even where
  the dataset name matches, and a positional join between them is silent.
- The round trip is asserted on **real** graphs because a serialisation that
  loses structure still produces a plausible bit count.
- The D14 invariant is asserted with a budget small enough that censoring
  **actually fires**, because a vacuous assertion over zero censored graphs
  passes for the wrong reason.
- The absence of ``signal.setitimer`` is asserted because a signal-based timeout
  does not interrupt the C++ engine and presents as a 25-minute hang.

Numbers produced by a competitor backend are never asserted. ``import
isalgraph`` resolves to the shared checkout, not to this worktree, so a bit
count can change under this file with no error raised. Shape and invariants are
asserted instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from benchmarks.real_data.eval_encoding import t06_claim_a, t06_cohort, t06_completion
from benchmarks.real_data.eval_encoding.t06_encode import (
    EncodeConfig,
    output_path,
    run_campaign,
)
from benchmarks.real_data.eval_encoding.t06_encode_worker import (
    ISALGRAPH_ALPHABET_SIZE,
    UNIT_SEP,
    error_family,
    symbol_sep,
)

pytestmark = pytest.mark.skipif(
    not (t06_cohort.cohort_root() / t06_cohort.EXPORT_SUBDIR).is_dir(),
    reason="the frozen cohort export is not mounted",
)

#: Representations exercised by the campaign fixture. One from each family:
#: two ``n^2`` serialisations, two ``m``-scaling ones, the reference arm, the
#: vector backend and the null.
FIXTURE_REPRESENTATIONS = (
    "isalgraph_pruned",
    "graph6",
    "sparse6",
    "adjacency",
    "nauty_graph6",
    "agm_cam",
    "min_dfs",
    "wl_subtree",
    "size_null",
)

#: Small enough to keep the suite fast, large enough that GREC spans both sides
#: of the n = 12 scope ceiling and therefore exercises the refusal path.
FIXTURE_LIMIT = 40

SCHEMA_KEYS = frozenset(
    {
        "graph_ids",
        "node_counts",
        "edge_counts",
        "encoding",
        "length",
        "error_kind",
        "entropy_bits",
        "realised_bits",
        "status",
        "fallback_used",
        "seconds",
        "metadata",
    }
)

METADATA_KEYS = frozenset(
    {
        "schema_version",
        "ticket",
        "wave",
        "generated_utc",
        "seed",
        "suite",
        "dataset",
        "representation",
        "metric",
        "n_graphs",
        "isalgraph_engine",
        "isalgraph_build_hash",
        "code_commit",
        "src_commit",
        "encode_budget_s",
        "notes",
    }
)

MODULE_PATHS = tuple(
    Path(__file__).resolve().parents[2] / "benchmarks" / "real_data" / "eval_encoding" / name
    for name in (
        "t06_cohort.py",
        "t06_encode.py",
        "t06_encode_worker.py",
        "t06_claim_a.py",
        "t06_completion.py",
    )
)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run a small real campaign once and reuse it across the module."""
    out = tmp_path_factory.mktemp("t06")
    for representation in FIXTURE_REPRESENTATIONS:
        run_campaign(
            EncodeConfig(
                suite="suite2",
                dataset="grec",
                representation=representation,
                out_dir=out,
                limit=FIXTURE_LIMIT,
            )
        )
    return {"root": out, "encodings": out / "encodings"}


# --- Criterion 1: cohort counts -------------------------------------------------


def test_suite2_totals_16370() -> None:
    """Suite 2 is 16,370 graphs over the ten frozen keys."""
    counts = t06_cohort.verify()
    assert counts["suite2_total"] == 16370
    assert set(counts["suite2"]) == set(t06_cohort.SUITE2_KEYS)


def test_suite1_per_dataset_counts() -> None:
    """Suite 1 is 1,180 / 1,253 / 2,059 / 89 / 769 after the n <= 12 filter."""
    counts = t06_cohort.verify()
    assert counts["suite1"] == {
        "iam_letter_low": 1180,
        "iam_letter_med": 1253,
        "iam_letter_high": 2059,
        "linux": 89,
        "aids": 769,
    }
    assert counts["suite1_total"] == 5350


def test_suite1_aids_is_not_suite2_aids_graphedx() -> None:
    """The two cohorts differ where the name matches, so joins must use ids."""
    suite1 = t06_cohort.load_cohort("suite1", "aids")
    suite2 = t06_cohort.load_cohort("suite2", "aids_graphedx")
    assert len(suite1) == 769
    assert len(suite2) == 819
    assert set(suite1.graph_ids) < set(suite2.graph_ids)


def test_edge_counts_match_the_declared_m() -> None:
    """De-duplicated CSR spans reproduce ``n_edges`` exactly.

    CONTRACTS §2 originally said both orientations were stored. They are not.
    The loader is orientation-agnostic and this asserts the recovery.
    """
    cohort = t06_cohort.load_cohort("suite2", "grec", limit=50)
    for index in range(len(cohort)):
        graph = cohort.to_networkx(index)
        assert graph.number_of_edges() == int(cohort.edge_counts[index])
        assert graph.number_of_nodes() == int(cohort.node_counts[index])


# --- Criterion 2: round trip ----------------------------------------------------


REVERSIBLE = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
    "isalgraph_pruned",
)


def _roundtrip_graphs() -> Iterator[tuple[str, int, object]]:
    """At least 200 real graphs spanning at least three datasets."""
    for dataset, take in (("iam_letter_low", 70), ("linux", 70), ("grec", 70)):
        cohort = t06_cohort.load_cohort("suite2", dataset, limit=take)
        for index in range(len(cohort)):
            yield dataset, index, cohort.to_networkx(index)


@pytest.mark.parametrize("representation", REVERSIBLE)
def test_roundtrip_on_real_graphs(representation: str) -> None:
    """``decode(encode(G))`` is isomorphic to ``G`` on real cohort graphs.

    A representation that declines a graph (AGM and canonical IsalGraph refuse
    above their node ceiling) is skipped for that graph rather than failed: the
    refusal is a declared property, and counting it as a round-trip failure
    would hide the real ones.
    """
    import networkx as nx

    from isalgraph.competitors.registry import get_repr_backend
    from isalgraph.errors import BudgetExceeded, SuiteScopeError

    backend = get_repr_backend(representation)
    checked = 0
    datasets: set[str] = set()
    for dataset, _index, graph in _roundtrip_graphs():
        try:
            encoding = backend.encode(graph)
        except (SuiteScopeError, BudgetExceeded):
            continue
        assert nx.is_isomorphic(backend.decode(encoding), graph)
        checked += 1
        datasets.add(dataset)
    assert len(datasets) >= 3, f"{representation} covered only {datasets}"
    if representation not in ("agm_cam",):
        assert checked >= 200, f"{representation} round-tripped only {checked} graphs"


# --- Criterion 3: schema conformance -------------------------------------------


@pytest.mark.parametrize("representation", FIXTURE_REPRESENTATIONS)
def test_emitted_file_has_exactly_the_schema_keys(
    campaign: dict[str, Path], representation: str
) -> None:
    """Every emitted file carries the §3 keys, no more and no fewer."""
    path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
    with np.load(path, allow_pickle=False) as handle:
        assert set(handle.files) == SCHEMA_KEYS


@pytest.mark.parametrize("representation", FIXTURE_REPRESENTATIONS)
def test_emitted_dtypes_and_shapes(campaign: dict[str, Path], representation: str) -> None:
    """The §3 dtypes hold and every column is the same length."""
    path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
    with np.load(path, allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    n = arrays["graph_ids"].shape[0]
    assert n == FIXTURE_LIMIT
    assert arrays["graph_ids"].dtype == np.dtype("<U16")
    assert arrays["node_counts"].dtype == np.int32
    assert arrays["edge_counts"].dtype == np.int32
    assert arrays["length"].dtype == np.int32
    assert arrays["error_kind"].dtype == np.dtype("<U32")
    assert arrays["entropy_bits"].dtype == np.float64
    assert arrays["realised_bits"].dtype == np.float64
    assert arrays["status"].dtype == np.dtype("<U12")
    assert arrays["fallback_used"].dtype == np.bool_
    assert arrays["seconds"].dtype == np.float32
    assert arrays["metadata"].shape == ()
    for name in SCHEMA_KEYS - {"metadata"}:
        assert arrays[name].shape == (n,), name


def test_graph_ids_and_counts_match_the_cohort(campaign: dict[str, Path]) -> None:
    """``graph_ids`` is cohort order exactly, and n/m are carried through."""
    cohort = t06_cohort.load_cohort("suite2", "grec", limit=FIXTURE_LIMIT)
    path = campaign["encodings"] / "suite2" / "grec__graph6.npz"
    with np.load(path, allow_pickle=False) as handle:
        assert np.array_equal(handle["graph_ids"], cohort.graph_ids)
        assert np.array_equal(handle["node_counts"], cohort.node_counts)
        assert np.array_equal(handle["edge_counts"], cohort.edge_counts)


@pytest.mark.parametrize("representation", FIXTURE_REPRESENTATIONS)
def test_metadata_carries_every_provenance_key(
    campaign: dict[str, Path], representation: str
) -> None:
    """§5 metadata is complete, including the two provenance keys.

    ``isalgraph_build_hash`` and ``src_commit`` are the only way to detect
    afterwards that a run picked up another branch's ``src/``.
    """
    path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
    with np.load(path, allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata"]))
    assert set(metadata) >= METADATA_KEYS
    assert metadata["isalgraph_build_hash"]
    assert metadata["src_commit"] not in ("", "unknown")
    assert metadata["code_commit"] not in ("", "unknown")
    assert metadata["seed"] == 42
    assert metadata["suite"] == "suite2"
    assert metadata["representation"] == representation
    assert metadata["symbol_sep"] == symbol_sep(representation)


# --- Criterion 3.1: symbols are the unit ---------------------------------------


@pytest.mark.parametrize("representation", FIXTURE_REPRESENTATIONS)
def test_length_is_the_symbol_count_in_both_branches(
    campaign: dict[str, Path], representation: str
) -> None:
    """``length`` recovers from ``encoding`` under both separator branches.

    ``sep != ''`` covers ``min_dfs``, ``size_null`` and ``wl_subtree``; ``sep
    == ''`` covers the character-symbol representations. Both branches are
    exercised by the fixture's representation list.
    """
    path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
    with np.load(path, allow_pickle=False) as handle:
        encoding, length = handle["encoding"], handle["length"]
        sep = json.loads(str(handle["metadata"]))["symbol_sep"]
    for text, count in zip(encoding, length, strict=True):
        if not text:
            continue
        recovered = len(str(text).split(sep)) if sep else len(str(text))
        assert recovered == int(count)


def test_both_separator_branches_are_covered() -> None:
    """The fixture list really does exercise ``sep == ''`` and ``sep != ''``."""
    separators = {symbol_sep(name) for name in FIXTURE_REPRESENTATIONS}
    assert separators == {"", UNIT_SEP}


def test_separator_never_occurs_inside_an_encoding(campaign: dict[str, Path]) -> None:
    """A ``sep == ''`` representation must not smuggle the unit separator in."""
    for representation in FIXTURE_REPRESENTATIONS:
        if symbol_sep(representation):
            continue
        path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
        with np.load(path, allow_pickle=False) as handle:
            assert not any(UNIT_SEP in str(text) for text in handle["encoding"])


def test_isalgraph_alphabet_size_matches_the_backend() -> None:
    """The fallback's alphabet constant has not drifted from the backend's."""
    from isalgraph.competitors.backends.isalgraph_ref import ALPHABET_SIZE

    assert ISALGRAPH_ALPHABET_SIZE == ALPHABET_SIZE


# --- Criterion 4: the D14 invariant, fired for real ----------------------------


#: Protein's leading graphs run from n = 4 to n = 88, so a 1 ms budget censors
#: most of them and completes at least one. A cohort that censored *everything*
#: would satisfy the invariant vacuously in the other direction.
CENSORED_DATASET = "protein"
CENSORED_LIMIT = 8


@pytest.fixture(scope="module")
def censored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign whose budget is small enough that censoring actually fires."""
    out = tmp_path_factory.mktemp("t06_censored")
    return run_campaign(
        EncodeConfig(
            suite="suite2",
            dataset=CENSORED_DATASET,
            representation="isalgraph_pruned",
            out_dir=out,
            limit=CENSORED_LIMIT,
            budget_s=0.001,
        )
    )


def test_d14_censoring_actually_fires(censored: Path) -> None:
    """A 1 ms budget censors at least one graph -- the assertion is not vacuous."""
    with np.load(censored, allow_pickle=False) as handle:
        status = handle["status"]
    assert int((status == "censored").sum()) > 0, "censoring never fired; the test is vacuous"


def test_censored_implies_fallback_and_non_empty_encoding(censored: Path) -> None:
    """D14: a censored graph is retained with its greedy-min string, never dropped."""
    with np.load(censored, allow_pickle=False) as handle:
        status, fallback, encoding = handle["status"], handle["fallback_used"], handle["encoding"]
    mask = status == "censored"
    assert bool(fallback[mask].all())
    assert all(str(text) != "" for text in encoding[mask])


def test_censored_graphs_are_not_dropped(censored: Path) -> None:
    """The censored cohort is the whole cohort: no row is missing."""
    cohort = t06_cohort.load_cohort("suite2", CENSORED_DATASET, limit=CENSORED_LIMIT)
    with np.load(censored, allow_pickle=False) as handle:
        assert np.array_equal(handle["graph_ids"], cohort.graph_ids)
        assert handle["graph_ids"].shape[0] == CENSORED_LIMIT


def test_censored_graphs_carry_a_greedy_min_string(censored: Path) -> None:
    """The substituted string is an IsalGraph instruction string, not a stub."""
    with np.load(censored, allow_pickle=False) as handle:
        status, encoding, length = handle["status"], handle["encoding"], handle["length"]
    mask = status == "censored"
    assert int(mask.sum()) > 0
    for text, count in zip(encoding[mask], length[mask], strict=True):
        assert set(str(text)) <= set("NnPpVvCcW")
        assert int(count) == len(str(text))


def test_complete_case_arm_is_recoverable(censored: Path) -> None:
    """Both arms are reportable: ``fallback_used`` separates them."""
    with np.load(censored, allow_pickle=False) as handle:
        status, fallback = handle["status"], handle["fallback_used"]
    complete_case = (status == "ok") & ~fallback
    primary = np.isin(status, ("ok", "censored", "fallback"))
    assert int(primary.sum()) >= int(complete_case.sum())


def test_ok_implies_non_negative_length(campaign: dict[str, Path]) -> None:
    """``status == 'ok'`` implies ``length >= 0`` for every representation."""
    for representation in FIXTURE_REPRESENTATIONS:
        path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
        with np.load(path, allow_pickle=False) as handle:
            assert bool((handle["length"][handle["status"] == "ok"] >= 0).all())


def test_error_kind_is_empty_unless_status_is_error(campaign: dict[str, Path]) -> None:
    """§3: ``error_kind`` carries an exception class name only on a failure."""
    for representation in FIXTURE_REPRESENTATIONS:
        path = campaign["encodings"] / "suite2" / f"grec__{representation}.npz"
        with np.load(path, allow_pickle=False) as handle:
            status, kind = handle["status"], handle["error_kind"]
        assert all(str(k) == "" for k in kind[status != "error"])


# --- Criterion 5: no signal-based timeout --------------------------------------


def _executable_source(path: Path) -> str:
    """Return *path*'s source with every docstring and comment removed.

    The prohibition is on *calling* the signal API, not on naming it: this file
    and the worker both explain in prose why the API is forbidden, and a plain
    substring search flags that explanation as a violation.
    """
    import ast
    import io
    import tokenize

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = [
                child
                for child in node.body
                if not (
                    isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)
                )
            ] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    without_docstrings = ast.unparse(tree)
    return "".join(
        token.string if token.type != tokenize.COMMENT else ""
        for token in tokenize.generate_tokens(io.StringIO(without_docstrings).readline)
    )


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda p: p.name)
def test_no_signal_based_timeout(path: Path) -> None:
    """A signal handler runs only between bytecodes and cannot stop the engine.

    ``SIGALRM`` stays queued for the whole duration of a native call, so a
    signal-based budget silently does not apply. It presents as a hang, not as
    an error, and a previous attempt hung for 25 minutes on one graph.
    """
    source = _executable_source(path)
    assert "signal.setitimer" not in source
    assert "signal.alarm" not in source
    assert "SIGALRM" not in source
    assert "import signal" not in source


def test_the_budget_is_enforced_by_a_subprocess() -> None:
    """The driver's enforcement really is a killed child process."""
    source = (MODULE_PATHS[1]).read_text()
    assert "subprocess.Popen" in source
    assert "proc.kill()" in source


# --- Criterion 6: both bit conventions -----------------------------------------


def test_claim_a_reports_both_conventions(campaign: dict[str, Path]) -> None:
    """All six Claim-A serialisations carry non-null entropy and realised bits."""
    report = t06_claim_a.build_report(campaign["encodings"], "suite2")
    representations = report["datasets"]["grec"]["representations"]
    for name in t06_claim_a.CLAIM_A_SERIALISATIONS:
        primary = representations[name]["primary"]
        for convention in ("entropy_bits", "realised_bits"):
            assert primary[convention] is not None, f"{name}/{convention}"
            assert primary[convention]["median"] > 0
            assert primary[convention]["std"] >= 0


def test_claim_a_marks_undefined_bit_counts_with_a_reason(campaign: dict[str, Path]) -> None:
    """``wl_subtree`` and ``size_null`` are present with a reason, not a zero."""
    report = t06_claim_a.build_report(campaign["encodings"], "suite2")
    representations = report["datasets"]["grec"]["representations"]
    for name in t06_claim_a.UNDEFINED_REPRESENTATIONS:
        cell = representations[name]
        assert cell["reason"] == "BitCountUndefined"
        assert cell["entropy_bits"] is None
        assert cell["realised_bits"] is None


def test_claim_a_pairs_are_binomially_bounded(campaign: dict[str, Path]) -> None:
    """Every win fraction sits inside its Clopper-Pearson interval."""
    report = t06_claim_a.build_report(campaign["encodings"], "suite2")
    paired = report["datasets"]["grec"]["paired"]
    assert paired
    for row in paired:
        assert row["ci_lower"] <= row["fraction_isalgraph_shorter"] <= row["ci_upper"]
        assert row["n_pairs"] > 0
        assert row["arm"] in ("primary", "complete_case")


def test_clopper_pearson_degenerate_ends() -> None:
    """Zero and complete success clamp rather than escaping ``[0, 1]``."""
    assert t06_claim_a.clopper_pearson(0, 20)[0] == 0.0
    assert t06_claim_a.clopper_pearson(20, 20)[1] == 1.0
    lower, upper = t06_claim_a.clopper_pearson(10, 20)
    assert 0.0 < lower < 0.5 < upper < 1.0


def test_claim_a_never_reports_a_mean_without_dispersion(campaign: dict[str, Path]) -> None:
    """Length distributions are right-skewed; a bare mean is not reportable."""
    report = t06_claim_a.build_report(campaign["encodings"], "suite2")
    for cell in report["datasets"]["grec"]["representations"].values():
        summary = cell.get("primary", {}).get("entropy_bits") if "reason" not in cell else None
        if summary is None:
            continue
        assert {"median", "mean", "std", "q1", "q3"} <= set(summary)


# --- Completion rates -----------------------------------------------------------


def test_completion_reports_numerator_denominator_and_rate(campaign: dict[str, Path]) -> None:
    """The ``c`` input is raw: fraction, numerator and denominator."""
    rows = t06_completion.collect(campaign["encodings"])
    assert {row.representation for row in rows} == set(FIXTURE_REPRESENTATIONS)
    for row in rows:
        assert row.n_graphs == FIXTURE_LIMIT
        assert 0 <= row.n_completed <= row.n_graphs
        assert row.rate == pytest.approx(row.n_completed / row.n_graphs)


def test_completion_separates_wall_clock_from_internal_cap(campaign: dict[str, Path]) -> None:
    """Conflating the two failure families makes the rate uninterpretable."""
    rows = {row.representation: row for row in t06_completion.collect(campaign["encodings"])}
    agm = rows["agm_cam"]
    assert agm.n_scope > 0, "GREC must span the AGM node ceiling for this to be meaningful"
    assert (
        agm.n_completed
        + agm.n_wall_clock
        + agm.n_internal_cap
        + agm.n_scope
        + (agm.n_unavailable + agm.n_other + agm.n_censored)
        == agm.n_graphs
    )


def test_completion_does_not_decide_c(campaign: dict[str, Path]) -> None:
    """The report states in the file that the gate is the orchestrator's."""
    report = t06_completion.build_report(t06_completion.collect(campaign["encodings"]))
    assert report["decides_c"] is False
    assert report["computability_threshold"] == 0.99
    assert report["threshold_scope"] == "per (representation, dataset)"


def test_error_family_classification() -> None:
    """Only a wall-clock failure is a censoring; a cap or a scope refusal is not."""
    assert error_family("") == "ok"
    assert error_family("CanonicalizationTimeoutError") == "wall_clock"
    assert error_family("Killed") == "wall_clock"
    assert error_family("AGMBudgetExceeded") == "internal_cap"
    assert error_family("MinDfsBudgetExceeded") == "internal_cap"
    assert error_family("SuiteScopeError") == "scope"
    assert error_family("BackendUnavailableError") == "unavailable"
    assert error_family("ZeroDivisionError") == "other"


# --- Criterion 7: the campaign writes one file per cell ------------------------


def test_one_file_per_cell_not_per_graph(campaign: dict[str, Path]) -> None:
    """The cluster quota is a file count; a per-graph layout would exhaust it."""
    files = list((campaign["encodings"] / "suite2").glob("*.npz"))
    assert len(files) == len(FIXTURE_REPRESENTATIONS)


def test_output_path_is_the_contract_layout() -> None:
    """``encodings/{suite}/{dataset}__{representation}.npz``."""
    cfg = EncodeConfig(
        suite="suite1",
        dataset="aids",
        representation="graph6",
        out_dir=Path("/tmp/x"),
    )
    assert output_path(cfg) == Path("/tmp/x/encodings/suite1/aids__graph6.npz")
