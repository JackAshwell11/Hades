"""Tests all functions in generation/primitives.py."""
from __future__ import annotations

# Custom
from game.generation.primitives import Point, Rect

__all__ = ()


def test_point() -> None:
    """Tests the Point class in primitives.py."""
    temp_point = Point(3, 5)
    assert isinstance(temp_point, Point) and temp_point == (3, 5)


def test_rect() -> None:
    """Tests the Rect class in primitives.py."""
