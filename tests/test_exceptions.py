"""Tests all functions in exceptions.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.exceptions import BiggerThanError, SpaceError

__all__ = ()


def test_raise_bigger_than_error() -> None:
    """Test that BiggerThanError is raised correctly."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 10.",
    ):
        raise BiggerThanError(10)


def test_raise_space_error() -> None:
    """Test that SpaceError is raised correctly."""
    name = "test"
    with pytest.raises(
        expected_exception=SpaceError,
        match="The `test` container does not have enough room.",
    ):
        raise SpaceError(name)
