"""Holds fixtures and test data used by all tests."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.generation.primitives import Point

__all__ = ()


@pytest.fixture
def valid_point_one() -> Point:
    """Initialise the first valid point for use in testing.

    Returns
    -------
    Point
        The first valid point used for testing.
    """
    return Point(3, 5)


@pytest.fixture
def valid_point_two() -> Point:
    """Initialise the second valid point for use in testing.

    Returns
    -------
    Point
        The second valid point used for testing.
    """
    return Point(5, 7)


@pytest.fixture
def boundary_point() -> Point:
    """Initialise a boundary point for use in testing.

    Returns
    -------
    Point
        The boundary point used for testing.
    """
    return Point(0, 0)


@pytest.fixture
def invalid_point() -> Point:
    """Initialise an invalid point for use in testing.

    Returns
    -------
    Point
        The invalid point used for testing.
    """
    return Point("test", "test")  # type: ignore
