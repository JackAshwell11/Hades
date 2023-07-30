"""Tests all functions in game_objects/base.py."""
from __future__ import annotations

# Builtin
import math

# Pip
import pytest

# Custom
from hades.game_objects.base import Vec2d

__all__ = ()


def test_vec2d_init() -> None:
    """Test if the Vec2d class is initialised correctly."""
    assert repr(Vec2d(0, 0)) == "<Vec2d (X=0) (Y=0)>"


def test_vec2d_addition() -> None:
    """Test that adding two vectors produce the correct result."""
    assert Vec2d(0, 0) + Vec2d(1, 1) == Vec2d(1, 1)
    assert Vec2d(-3, -2) + Vec2d(-1, -1) == Vec2d(-4, -3)
    assert Vec2d(6, 3) + Vec2d(5, 5) == Vec2d(11, 8)
    assert Vec2d(1, 1) + Vec2d(1, 1) == Vec2d(2, 2)
    assert Vec2d(-5, 4) + Vec2d(7, -1) == Vec2d(2, 3)


def test_vec2d_subtraction() -> None:
    """Test that subtracting two vectors produce the correct result."""
    assert Vec2d(0, 0) - Vec2d(1, 1) == Vec2d(-1, -1)
    assert Vec2d(-3, -2) - Vec2d(-1, -1) == Vec2d(-2, -1)
    assert Vec2d(6, 3) - Vec2d(5, 5) == Vec2d(1, -2)
    assert Vec2d(1, 1) - Vec2d(1, 1) == Vec2d(0, 0)
    assert Vec2d(-5, 4) - Vec2d(7, -1) == Vec2d(-12, 5)


def test_vec2d_abs() -> None:
    """Test that the absolute value of a vector is calculated correctly."""
    assert abs(Vec2d(0, 0)) == 0
    assert abs(Vec2d(-3, -2)) == 3.605551275463989
    assert abs(Vec2d(6, 3)) == 6.708203932499369
    assert abs(Vec2d(1, 1)) == 1.4142135623730951
    assert abs(Vec2d(-5, 4)) == 6.4031242374328485


def test_vec2d_multiplication() -> None:
    """Test that multiplying a vector by a scalar produces the correct result."""
    assert Vec2d(0, 0) * 1 == Vec2d(0, 0)
    assert Vec2d(-3, -2) * 2 == Vec2d(-6, -4)
    assert Vec2d(6, 3) * 3 == Vec2d(18, 9)
    assert Vec2d(1, 1) * 4 == Vec2d(4, 4)
    assert Vec2d(-5, 4) * 5 == Vec2d(-25, 20)


def test_vec2d_division() -> None:
    """Test that dividing a vector by a scalar produces the correct result."""
    assert Vec2d(0, 0) // 1 == Vec2d(0, 0)
    assert Vec2d(-3, -2) // 2 == Vec2d(-2, -1)
    assert Vec2d(6, 3) // 3 == Vec2d(2, 1)
    assert Vec2d(1, 1) // 4 == Vec2d(0, 0)
    assert Vec2d(-5, 4) // 5 == Vec2d(-1, 0)


def test_vec2d_normalised() -> None:
    """Test that normalising a vector produces the correct result."""
    assert Vec2d(0, 0).normalised() == Vec2d(0, 0)
    assert Vec2d(-3, -2).normalised() == Vec2d(-0.8320502943378437, -0.5547001962252291)
    assert Vec2d(6, 3).normalised() == Vec2d(0.8944271909999159, 0.4472135954999579)
    assert Vec2d(1, 1).normalised() == Vec2d(0.7071067811865475, 0.7071067811865475)
    assert Vec2d(-5, 4).normalised() == Vec2d(-0.7808688094430304, 0.6246950475544243)


def test_vec2d_rotated() -> None:
    """Test that rotating a vector produces the correct result."""
    assert Vec2d(0, 0).rotated(math.radians(360)) == Vec2d(0, 0)
    assert Vec2d(-3, -2).rotated(math.radians(270)) == pytest.approx(Vec2d(-2, 3))
    assert Vec2d(6, 3).rotated(math.radians(180)) == pytest.approx(Vec2d(-6, -3))
    assert Vec2d(1, 1).rotated(math.radians(90)) == pytest.approx(Vec2d(-1, 1))
    assert Vec2d(-5, 4).rotated(math.radians(0)) == Vec2d(-5, 4)


def test_vec2d_get_angle_between() -> None:
    """Test that getting the angle between two vectors produces the correct result."""
    assert Vec2d(0, 0).get_angle_between(Vec2d(1, 1)) == 0
    assert Vec2d(-3, -2).get_angle_between(Vec2d(-1, -1)) == 0.19739555984988075
    assert Vec2d(6, 3).get_angle_between(Vec2d(5, 5)) == 0.3217505543966422
    assert Vec2d(1, 1).get_angle_between(Vec2d(1, 1)) == 0
    assert Vec2d(-5, 4).get_angle_between(Vec2d(7, -1)) == 3.674436541209182


def test_vec2d_get_distance_to() -> None:
    """Test that getting the distance of two vectors produces the correct result."""
    assert Vec2d(0, 0).get_distance_to(Vec2d(1, 1)) == 1.4142135623730951
    assert Vec2d(-3, -2).get_distance_to(Vec2d(-1, -1)) == 2.23606797749979
    assert Vec2d(6, 3).get_distance_to(Vec2d(5, 5)) == 2.23606797749979
    assert Vec2d(1, 1).get_distance_to(Vec2d(1, 1)) == 0
    assert Vec2d(-5, 4).get_distance_to(Vec2d(7, -1)) == 13
