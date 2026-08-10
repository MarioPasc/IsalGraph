"""Build probe and dispatch-contract tests for the native engine.

The importorskip call is at MODULE TOP LEVEL, before the imports, and that
placement is mandatory rather than stylistic: ``@pytest.mark.parametrize``
decorators are evaluated at collection time, so a ``pytestmark = skipif``
would arrive too late to prevent an ImportError when the .so is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isalgraph.core._native", reason="C++ extension not built")

import os

from isalgraph.core import _native as ext
from isalgraph.core import backends
from isalgraph.errors import BackendError

# ----------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------


def test_engine_name_is_cpp() -> None:
    assert ext.engine_name() == "cpp"


def test_build_info_has_every_key() -> None:
    info = ext.build_info()
    for key in (
        "compiler",
        "cplusplus",
        "isa_level",
        "avx2",
        "fma",
        "avx512f",
        "ndebug",
        "build_hash",
    ):
        assert key in info, key
        assert isinstance(info[key], str)


def test_build_is_release_cxx17_and_portable_isa() -> None:
    """A debug build, a pre-C++17 build, or an AVX-512 build is a build fault.

    -march=native on this workstation would report avx512f on some machines
    and not others; the pinned x86-64-v3 target is what keeps one build hash
    valid across the heterogeneous Picasso fleet.
    """
    info = ext.build_info()
    assert info["ndebug"] == "1", "extension was not built with -DNDEBUG"
    assert int(info["cplusplus"]) >= 201703, info["cplusplus"]
    assert info["isa_level"] == "x86-64-v3", (
        f"expected the portable x86-64-v3 target, got {info['isa_level']!r}; "
        "was ISALGRAPH_NATIVE_MARCH left ON?"
    )
    assert info["avx2"] == "1"
    assert info["fma"] == "1"
    assert info["avx512f"] == "0"


def test_build_hash_is_stable_across_calls() -> None:
    assert ext.build_info()["build_hash"] == ext.build_info()["build_hash"]
    assert len(ext.build_info()["build_hash"]) == 16


# ----------------------------------------------------------------------
# FNV-1a: the C++ header against an independent Python implementation
# ----------------------------------------------------------------------


def _fnv1a64_reference(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"a",
        b"IsalGraph",
        b"\x00\x01\x02\xfe\xff",
        bytes(range(256)),
        b"NVVCnpPVvW" * 37,
    ],
)
def test_fnv1a64_matches_python_reference(payload: bytes) -> None:
    assert ext.fnv1a64(payload) == _fnv1a64_reference(payload)


# ----------------------------------------------------------------------
# Dispatch contract
# ----------------------------------------------------------------------


def test_default_backend_is_cpp_when_extension_present() -> None:
    assert backends.DEFAULT_BACKEND == "cpp"


def test_engine_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISALGRAPH_ENGINE", "python")
    assert backends.engine() == "python"
    monkeypatch.setenv("ISALGRAPH_ENGINE", "cpp")
    assert backends.engine() == "cpp"


def test_engine_rejects_unknown_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISALGRAPH_ENGINE", "fortran")
    with pytest.raises(BackendError, match="not valid"):
        backends.engine()


def test_explicit_backend_beats_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """IsalSR shipped a bug where the env var was read before the kwarg.

    The symptom was silent: engine() reported "python" while every call still
    dispatched to C++. This probes the actual dispatch, not the report.
    """
    monkeypatch.setenv("ISALGRAPH_ENGINE", "python")
    registry = {"python": lambda: "python", "cpp": lambda: "cpp"}
    assert backends.resolve("cpp", registry)() == "cpp"
    assert backends.resolve(None, registry)() == "python"

    monkeypatch.setenv("ISALGRAPH_ENGINE", "cpp")
    assert backends.resolve("python", registry)() == "python"


def test_resolve_rejects_unknown_backend() -> None:
    with pytest.raises(BackendError, match="Unknown backend"):
        backends.resolve("julia", {"python": 1, "cpp": 2})


def test_build_info_reports_active_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISALGRAPH_ENGINE", "cpp")
    assert backends.build_info()["engine"] == "cpp"
    assert backends.build_info()["isa_level"] == "x86-64-v3"
    monkeypatch.setenv("ISALGRAPH_ENGINE", "python")
    assert backends.build_info()["engine"] == "python"
    assert backends.build_info()["isa_level"] == ""


def test_explicit_cpp_raises_when_extension_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """backend='cpp' must never silently degrade to Python."""
    monkeypatch.setattr(backends, "_CPP_AVAILABLE", False)
    with pytest.raises(BackendError, match="not available"):
        backends.resolve("cpp", {"python": 1, "cpp": 2})


def test_env_cpp_raises_when_extension_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_CPP_AVAILABLE", False)
    monkeypatch.setenv("ISALGRAPH_ENGINE", "cpp")
    with pytest.raises(BackendError, match="could not be"):
        backends.engine()


def test_python_backend_still_works_without_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no .so and no explicit request, everything must still run."""
    monkeypatch.setattr(backends, "_CPP_AVAILABLE", False)
    monkeypatch.setattr(backends, "DEFAULT_BACKEND", "python")
    monkeypatch.delenv("ISALGRAPH_ENGINE", raising=False)
    assert backends.engine() == "python"
    assert backends.levenshtein("abc", "abd") == 1


def test_no_stray_engine_env_var_in_this_session() -> None:
    """Guard against a leaked override making the whole suite meaningless."""
    assert os.environ.get("ISALGRAPH_ENGINE", "") in ("", "cpp", "python")
