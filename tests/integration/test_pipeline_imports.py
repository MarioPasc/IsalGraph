# ruff: noqa: N802, E402
"""Integration test: verify all pipeline modules import successfully.

Catches missing imports, undefined names, and circular dependencies
that would only surface at runtime (as happened with the
compute_pruned_exhaustive NameError in eval_setup.py).

These tests do NOT require source data or GPU — only that the Python
modules can be imported without error.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Every module that the paper pipeline imports at the top level.
# If a module fails to import here, the pipeline will crash at runtime.
# ---------------------------------------------------------------------------

PIPELINE_MODULES = [
    # Step 1: eval_setup
    "benchmarks.eval_setup.eval_setup",
    "benchmarks.eval_setup.canonical_computer",
    "benchmarks.eval_setup.dataset_filter",
    "benchmarks.eval_setup.ged_computer",
    "benchmarks.eval_setup.greedy_single_computer",
    "benchmarks.eval_setup.pruned_exhaustive_computer",
    "benchmarks.eval_setup.levenshtein_computer",
    "benchmarks.eval_setup.method_comparator",
    "benchmarks.eval_setup.validator",
    "benchmarks.eval_setup.wl_kernel_computer",
    # Step 2a: eval_correlation
    "benchmarks.eval_correlation.eval_correlation",
    # Step 2b: eval_computational
    "benchmarks.eval_computational.eval_computational",
    "benchmarks.eval_computational.timing_utils",
    # Step 2c: eval_encoding
    "benchmarks.eval_encoding.eval_encoding",
    "benchmarks.eval_encoding.synthetic_generator",
    # Step 2d: eval_message_length
    "benchmarks.eval_message_length.eval_message_length",
    "benchmarks.eval_message_length.message_length_computer",
    # Visualizations
    "benchmarks.eval_visualizations.result_loader",
    "benchmarks.eval_visualizations.fig_message_length",
    "benchmarks.eval_visualizations.fig_empirical_complexity",
    "benchmarks.eval_visualizations.table_performance_summary",
    # Plotting
    "benchmarks.plotting_styles",
]


@pytest.mark.parametrize("module_name", PIPELINE_MODULES)
def test_pipeline_module_imports(module_name: str) -> None:
    """Each pipeline module must import without NameError or ImportError."""
    try:
        importlib.import_module(module_name)
    except NameError as e:
        pytest.fail(
            f"NameError importing {module_name}: {e}. "
            f"This likely means a function is used but never imported."
        )
    except ImportError as e:
        pytest.fail(
            f"ImportError importing {module_name}: {e}. "
            f"Check that the module exists and its dependencies are installed."
        )


# ---------------------------------------------------------------------------
# Verify that eval_setup.py can resolve all algorithm methods
# ---------------------------------------------------------------------------


def test_eval_setup_algorithm_mapping_complete() -> None:
    """All config algorithm names must map to internal method names."""
    from benchmarks.eval_setup.eval_setup import ALGORITHM_TO_METHOD

    expected_algorithms = ["canonical", "canonical_pruned", "greedy_min", "greedy_single"]
    expected_methods = {"exhaustive", "pruned_exhaustive", "greedy", "greedy_single"}

    for algo in expected_algorithms:
        assert algo in ALGORITHM_TO_METHOD, f"Algorithm {algo!r} missing from ALGORITHM_TO_METHOD"

    actual_methods = set(ALGORITHM_TO_METHOD.values())
    assert actual_methods == expected_methods, (
        f"Method set mismatch: {actual_methods} != {expected_methods}"
    )


def test_eval_setup_all_methods_have_compute_functions() -> None:
    """Each method referenced in eval_setup must have callable compute/save functions."""
    import benchmarks.eval_setup.eval_setup as es

    # These names must be resolvable in the eval_setup module's namespace
    required_names = [
        "compute_all_canonical",
        "save_canonical_strings",
        "compute_greedy_single",
        "save_greedy_single_strings",
        "compute_pruned_exhaustive",
        "save_pruned_exhaustive_strings",
        "compute_levenshtein_matrix",
        "save_levenshtein_matrix",
        "compute_wl_kernel_distance",
        "save_wl_kernel_matrix",
    ]

    for name in required_names:
        assert hasattr(es, name), f"{name!r} not found in eval_setup module. Missing import?"
        assert callable(getattr(es, name)), f"{name!r} is not callable"


# ---------------------------------------------------------------------------
# Verify validator knows about all methods
# ---------------------------------------------------------------------------


def test_validator_methods_match_algorithms() -> None:
    """Validator's METHODS list must include all pipeline method names."""
    from benchmarks.eval_setup.eval_setup import ALGORITHM_TO_METHOD
    from benchmarks.eval_setup.validator import METHODS as VALIDATOR_METHODS

    pipeline_methods = set(ALGORITHM_TO_METHOD.values())
    validator_methods = set(VALIDATOR_METHODS)

    missing = pipeline_methods - validator_methods
    assert not missing, (
        f"Validator METHODS missing pipeline methods: {missing}. Validator has: {validator_methods}"
    )


# ---------------------------------------------------------------------------
# Verify message_length_computer is stdlib-only
# ---------------------------------------------------------------------------


def test_message_length_computer_stdlib_only() -> None:
    """message_length_computer must not import any external packages."""
    import benchmarks.eval_message_length.message_length_computer as mlc

    # Check that the module only uses stdlib
    module_file = mlc.__file__
    assert module_file is not None

    with open(module_file) as f:
        source = f.read()

    # These external packages must NOT appear as top-level imports
    forbidden = ["numpy", "pandas", "scipy", "matplotlib", "networkx", "torch"]
    for pkg in forbidden:
        # Check for "import <pkg>" or "from <pkg>"
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith(f"import {pkg}"), (
                f"message_length_computer.py imports {pkg} (must be stdlib-only)"
            )
            assert not stripped.startswith(f"from {pkg}"), (
                f"message_length_computer.py imports from {pkg} (must be stdlib-only)"
            )
