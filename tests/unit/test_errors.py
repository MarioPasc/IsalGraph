"""Unit tests for custom exception hierarchy."""

from __future__ import annotations

import pytest

from isalgraph.errors import (
    CapacityError,
    InvalidNodeError,
    InvalidStringError,
    IsalGraphError,
)


class TestExceptionHierarchy:
    """Verify exception classes can be instantiated, raised, and caught."""

    def test_isalgraph_error_is_exception(self) -> None:
        err = IsalGraphError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    def test_capacity_error_inherits(self) -> None:
        err = CapacityError("full")
        assert isinstance(err, IsalGraphError)
        assert isinstance(err, Exception)
        with pytest.raises(CapacityError, match="full"):
            raise err

    def test_invalid_node_error_inherits(self) -> None:
        err = InvalidNodeError("bad node 99")
        assert isinstance(err, IsalGraphError)
        with pytest.raises(InvalidNodeError, match="bad node 99"):
            raise err

    def test_invalid_string_error_inherits(self) -> None:
        err = InvalidStringError("invalid char X")
        assert isinstance(err, IsalGraphError)
        with pytest.raises(InvalidStringError, match="invalid char X"):
            raise err

    def test_catch_all_via_base(self) -> None:
        """All sub-exceptions can be caught via IsalGraphError."""
        for exc_class in (CapacityError, InvalidNodeError, InvalidStringError):
            with pytest.raises(IsalGraphError):
                raise exc_class("test")
