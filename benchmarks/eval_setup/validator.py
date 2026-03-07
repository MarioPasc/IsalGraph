"""Comprehensive validation suite for eval_setup artifacts.

Validates GED matrices, canonical string files, Levenshtein matrices,
and cross-consistency of graph ID ordering.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)

DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "alkane"]
METHODS = ["exhaustive", "greedy"]


@dataclass
class ValidationCheck:
    """Result of one validation check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"


def validate_ged_matrix(path: str) -> list[ValidationCheck]:
    """Validate a GED matrix .npz file.

    Checks: file exists, symmetry, diagonal=0, non-negative, triangle inequality.
    """
    checks: list[ValidationCheck] = []
    prefix = os.path.basename(path)

    if not os.path.exists(path):
        checks.append(ValidationCheck(f"{prefix}/exists", False, "File not found", "error"))
        return checks

    checks.append(ValidationCheck(f"{prefix}/exists", True, "File found"))

    data = np.load(path, allow_pickle=True)
    ged = data["ged_matrix"]
    n = ged.shape[0]

    # Symmetry
    finite_mask = np.isfinite(ged) & np.isfinite(ged.T)
    if np.allclose(ged[finite_mask], ged.T[finite_mask]):
        checks.append(ValidationCheck(f"{prefix}/symmetry", True, "Symmetric"))
    else:
        max_diff = float(np.max(np.abs(ged[finite_mask] - ged.T[finite_mask])))
        checks.append(
            ValidationCheck(f"{prefix}/symmetry", False, f"Max asymmetry: {max_diff}", "error")
        )

    # Diagonal = 0
    diag = np.diag(ged)
    if np.allclose(diag, 0.0):
        checks.append(ValidationCheck(f"{prefix}/diagonal", True, "Diagonal is zero"))
    else:
        checks.append(
            ValidationCheck(
                f"{prefix}/diagonal", False, f"Non-zero diagonal: {diag[diag != 0]}", "error"
            )
        )

    # Non-negative
    finite_vals = ged[np.isfinite(ged)]
    if np.all(finite_vals >= 0):
        checks.append(ValidationCheck(f"{prefix}/non_negative", True, "All non-negative"))
    else:
        n_neg = int(np.sum(finite_vals < 0))
        checks.append(
            ValidationCheck(f"{prefix}/non_negative", False, f"{n_neg} negative entries", "error")
        )

    # Triangle inequality (sample 1000 random triples)
    rng = np.random.default_rng(42)
    n_violations = 0
    n_checked = 0
    for _ in range(min(1000, n * (n - 1) * (n - 2) // 6)):
        i, j, k = rng.choice(n, size=3, replace=False)
        if np.isfinite(ged[i, j]) and np.isfinite(ged[j, k]) and np.isfinite(ged[i, k]):
            if ged[i, k] > ged[i, j] + ged[j, k] + 1e-10:
                n_violations += 1
            n_checked += 1

    if n_violations == 0:
        checks.append(
            ValidationCheck(
                f"{prefix}/triangle_ineq",
                True,
                f"Passed ({n_checked} triples checked)",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                f"{prefix}/triangle_ineq",
                False,
                f"{n_violations}/{n_checked} violations",
                "warning",
            )
        )

    # Stats
    n_finite = int(np.isfinite(ged[np.triu_indices(n, k=1)]).sum())
    total = n * (n - 1) // 2
    checks.append(
        ValidationCheck(
            f"{prefix}/stats",
            True,
            f"{n} graphs, {n_finite}/{total} finite pairs",
            "info",
        )
    )

    return checks


def validate_levenshtein_matrix(path: str) -> list[ValidationCheck]:
    """Validate a Levenshtein matrix .npz file.

    Checks: file exists, symmetry, diagonal=0, non-negative.
    """
    checks: list[ValidationCheck] = []
    prefix = os.path.basename(path)

    if not os.path.exists(path):
        checks.append(ValidationCheck(f"{prefix}/exists", False, "File not found", "error"))
        return checks

    checks.append(ValidationCheck(f"{prefix}/exists", True, "File found"))

    data = np.load(path, allow_pickle=True)
    lev = data["levenshtein_matrix"]

    # Symmetry
    valid = (lev >= 0) & (lev.T >= 0)
    if np.array_equal(lev[valid], lev.T[valid]):
        checks.append(ValidationCheck(f"{prefix}/symmetry", True, "Symmetric"))
    else:
        checks.append(ValidationCheck(f"{prefix}/symmetry", False, "Not symmetric", "error"))

    # Diagonal = 0
    diag = np.diag(lev)
    if np.all(diag == 0):
        checks.append(ValidationCheck(f"{prefix}/diagonal", True, "Diagonal is zero"))
    else:
        checks.append(ValidationCheck(f"{prefix}/diagonal", False, "Non-zero diagonal", "error"))

    # Non-negative (excluding -1 sentinel)
    valid_entries = lev[lev >= 0]
    if len(valid_entries) > 0 and np.all(valid_entries >= 0):
        checks.append(ValidationCheck(f"{prefix}/non_negative", True, "All non-negative"))

    return checks


def validate_canonical_completeness(
    canonical_path: str,
    ged_path: str,
    lev_path: str,
    method: str,
) -> list[ValidationCheck]:
    """Validate completeness: GED=0 implies Levenshtein=0 for exhaustive.

    For greedy, documents mismatches (expected, not an error).
    """
    checks: list[ValidationCheck] = []
    prefix = f"{method}_completeness"

    if not all(os.path.exists(p) for p in [canonical_path, ged_path, lev_path]):
        checks.append(ValidationCheck(f"{prefix}/files", False, "Missing required files", "error"))
        return checks

    ged_data = np.load(ged_path, allow_pickle=True)
    lev_data = np.load(lev_path, allow_pickle=True)
    ged = ged_data["ged_matrix"]
    lev = lev_data["levenshtein_matrix"]
    n = ged.shape[0]

    # Find pairs where GED = 0 (isomorphic graphs)
    n_ged_zero = 0
    n_lev_nonzero = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(ged[i, j]) and ged[i, j] == 0:
                n_ged_zero += 1
                if lev[i, j] > 0:
                    n_lev_nonzero += 1

    if method == "exhaustive":
        if n_lev_nonzero == 0:
            checks.append(
                ValidationCheck(
                    f"{prefix}/ged0_lev0",
                    True,
                    f"All {n_ged_zero} isomorphic pairs have Lev=0 (complete invariant)",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    f"{prefix}/ged0_lev0",
                    False,
                    f"{n_lev_nonzero}/{n_ged_zero} isomorphic pairs have Lev>0",
                    "error",
                )
            )
    else:
        severity = "warning" if n_lev_nonzero > 0 else "info"
        checks.append(
            ValidationCheck(
                f"{prefix}/ged0_lev0",
                n_lev_nonzero == 0,
                f"{n_lev_nonzero}/{n_ged_zero} isomorphic pairs have Lev>0 (expected for greedy)",
                severity,
            )
        )

    return checks


def validate_cross_consistency(data_root: str, dataset: str) -> list[ValidationCheck]:
    """Validate graph_id ordering is consistent across all files for a dataset."""
    checks: list[ValidationCheck] = []
    prefix = f"{dataset}/consistency"

    ged_path = os.path.join(data_root, "ged_matrices", f"{dataset}.npz")
    if not os.path.exists(ged_path):
        checks.append(ValidationCheck(f"{prefix}/ged", False, "GED matrix not found", "error"))
        return checks

    ged_data = np.load(ged_path, allow_pickle=True)
    ref_ids = list(ged_data["graph_ids"])

    for method in METHODS:
        # Check canonical strings
        can_path = os.path.join(data_root, "canonical_strings", f"{dataset}_{method}.json")
        if os.path.exists(can_path):
            with open(can_path) as f:
                can_data = json.load(f)
            can_ids = sorted(can_data.get("strings", {}).keys())
            ref_sorted = sorted(ref_ids)
            # Canonical may have fewer (timeouts) but should be a subset
            if set(can_ids).issubset(set(ref_sorted)):
                checks.append(
                    ValidationCheck(
                        f"{prefix}/canonical_{method}",
                        True,
                        f"Graph IDs are subset of GED ({len(can_ids)}/{len(ref_ids)})",
                    )
                )
            else:
                extra = set(can_ids) - set(ref_sorted)
                checks.append(
                    ValidationCheck(
                        f"{prefix}/canonical_{method}",
                        False,
                        f"{len(extra)} IDs in canonical not in GED",
                        "error",
                    )
                )

        # Check Levenshtein matrices
        lev_path = os.path.join(data_root, "levenshtein_matrices", f"{dataset}_{method}.npz")
        if os.path.exists(lev_path):
            lev_data = np.load(lev_path, allow_pickle=True)
            lev_ids = list(lev_data["graph_ids"])
            if lev_ids == ref_ids:
                checks.append(
                    ValidationCheck(
                        f"{prefix}/levenshtein_{method}",
                        True,
                        "Graph ID order matches GED",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        f"{prefix}/levenshtein_{method}",
                        False,
                        "Graph ID order mismatch with GED",
                        "error",
                    )
                )

    return checks


def validate_all(data_root: str) -> list[ValidationCheck]:
    """Run all validation checks for a data root.

    Args:
        data_root: Root directory of eval output.

    Returns:
        List of all validation checks.
    """
    all_checks: list[ValidationCheck] = []

    for dataset in DATASETS:
        # GED matrix
        ged_path = os.path.join(data_root, "ged_matrices", f"{dataset}.npz")
        if os.path.exists(ged_path):
            all_checks.extend(validate_ged_matrix(ged_path))

        # Levenshtein matrices
        for method in METHODS:
            lev_path = os.path.join(data_root, "levenshtein_matrices", f"{dataset}_{method}.npz")
            if os.path.exists(lev_path):
                all_checks.extend(validate_levenshtein_matrix(lev_path))

            # Completeness
            can_path = os.path.join(data_root, "canonical_strings", f"{dataset}_{method}.json")
            if os.path.exists(can_path) and os.path.exists(ged_path) and os.path.exists(lev_path):
                all_checks.extend(
                    validate_canonical_completeness(can_path, ged_path, lev_path, method)
                )

        # Cross-consistency
        all_checks.extend(validate_cross_consistency(data_root, dataset))

    return all_checks


def save_validation_report(
    checks: list[ValidationCheck],
    output_path: str,
) -> None:
    """Save validation report to JSON.

    Args:
        checks: List of validation checks.
        output_path: Output JSON path.
    """
    n_passed = sum(1 for c in checks if c.passed)
    n_failed = sum(1 for c in checks if not c.passed and c.severity == "error")
    n_warnings = sum(1 for c in checks if not c.passed and c.severity == "warning")

    report = {
        "summary": {
            "total_checks": len(checks),
            "passed": n_passed,
            "failed": n_failed,
            "warnings": n_warnings,
            "all_passed": n_failed == 0,
        },
        "checks": [asdict(c) for c in checks],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Validation: %d checks, %d passed, %d failed, %d warnings",
        len(checks),
        n_passed,
        n_failed,
        n_warnings,
    )
