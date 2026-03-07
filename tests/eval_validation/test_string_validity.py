"""Validate that all generated strings use the correct alphabet and have consistent lengths."""

from __future__ import annotations

import pytest

from tests.eval_validation.conftest import DATASETS, METHODS, EvalData

pytestmark = pytest.mark.eval_validation

VALID_ALPHABET = set("NnPpVvCcW")


class TestStringAlphabet:
    """Every character in every generated string must be in Sigma."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_valid_characters(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        invalid = []
        for gid, string in ds.strings.items():
            bad_chars = set(string) - VALID_ALPHABET
            if bad_chars:
                invalid.append(f"{gid}: invalid chars {bad_chars} in '{string[:30]}'")

        assert not invalid, (
            f"{dataset}/{method}: {len(invalid)} strings with invalid characters:\n"
            + "\n".join(invalid[:10])
        )


class TestStringLengthConsistency:
    """Reported length must match actual string length."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_length_matches(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        mismatches = []
        for gid, string in ds.strings.items():
            reported = ds.lengths[gid]
            actual = len(string)
            if reported != actual:
                mismatches.append(f"{gid}: reported len={reported}, actual len={actual}")

        assert not mismatches, (
            f"{dataset}/{method}: {len(mismatches)} length mismatches:\n"
            + "\n".join(mismatches[:10])
        )


class TestStringNonEmpty:
    """All graphs with >= 2 nodes must produce non-empty strings."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_nonempty(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        empty = []
        for gid, string in ds.strings.items():
            n_nodes = ds.node_counts.get(gid, 0)
            if n_nodes >= 2 and len(string) == 0:
                empty.append(f"{gid} ({n_nodes} nodes)")

        assert not empty, (
            f"{dataset}/{method}: {len(empty)} empty strings for graphs with >= 2 nodes:\n"
            + "\n".join(empty[:10])
        )


class TestStringMinimumLength:
    """String length must be >= (n_nodes - 1) since we need at least n-1 V/v instructions."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_minimum_length(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        violations = []
        for gid, string in ds.strings.items():
            n_nodes = ds.node_counts.get(gid, 0)
            if n_nodes >= 2 and len(string) < n_nodes - 1:
                violations.append(f"{gid}: len={len(string)} < n_nodes-1={n_nodes - 1}")

        assert not violations, (
            f"{dataset}/{method}: {len(violations)} strings shorter than n-1:\n"
            + "\n".join(violations[:10])
        )
