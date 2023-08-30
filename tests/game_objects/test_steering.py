"""Tests all classes and functions in game_objects/steering.py."""
from __future__ import annotations

# Builtin
import math

# Pip
import pytest

# Custom
from hades.game_objects.steering import (
    Vec2d,
    arrive,
    evade,
    flee,
    follow_path,
    obstacle_avoidance,
    pursuit,
    seek,
    wander,
)

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
    assert Vec2d(-3, -2).rotated(math.radians(270)) == Vec2d(
        pytest.approx(-2),  # type: ignore[arg-type]
        pytest.approx(3),  # type: ignore[arg-type]
    )
    assert Vec2d(6, 3).rotated(math.radians(180)) == Vec2d(
        -6,
        pytest.approx(-3),  # type: ignore[arg-type]
    )
    assert Vec2d(1, 1).rotated(math.radians(90)) == Vec2d(
        pytest.approx(-1),  # type: ignore[arg-type]
        1,
    )
    assert Vec2d(-5, 4).rotated(math.radians(0)) == Vec2d(-5, 4)


def test_vec2d_get_angle_between() -> None:
    """Test that getting the angle between two vectors produces the correct result."""
    assert Vec2d(0, 0).get_angle_between(Vec2d(1, 1)) == 0
    assert Vec2d(-3, -2).get_angle_between(Vec2d(-1, -1)) == 0.19739555984988044
    assert Vec2d(6, 3).get_angle_between(Vec2d(5, 5)) == 0.32175055439664213
    assert Vec2d(1, 1).get_angle_between(Vec2d(1, 1)) == 0
    assert Vec2d(-5, 4).get_angle_between(Vec2d(7, -1)) == 3.674436541209182


def test_vec2d_get_distance_to() -> None:
    """Test that getting the distance of two vectors produces the correct result."""
    assert Vec2d(0, 0).get_distance_to(Vec2d(1, 1)) == 1.4142135623730951
    assert Vec2d(-3, -2).get_distance_to(Vec2d(-1, -1)) == 2.23606797749979
    assert Vec2d(6, 3).get_distance_to(Vec2d(5, 5)) == 2.23606797749979
    assert Vec2d(1, 1).get_distance_to(Vec2d(1, 1)) == 0
    assert Vec2d(-5, 4).get_distance_to(Vec2d(7, -1)) == 13


def test_arrive_outside_slowing_radius() -> None:
    """Test if a position outside the radius produces the correct arrive force."""
    assert arrive(Vec2d(500, 500), Vec2d(0, 0)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_arrive_on_slowing_range() -> None:
    """Test if a position on the radius produces the correct arrive force."""
    assert arrive(Vec2d(135, 135), Vec2d(0, 0)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_arrive_inside_slowing_range() -> None:
    """Test if a position inside the radius produces the correct arrive force."""
    assert arrive(Vec2d(100, 100), Vec2d(0, 0)) == Vec2d(
        -0.7071067811865476,
        -0.7071067811865476,
    )


def test_arrive_near_target() -> None:
    """Test if a position near the target produces the correct arrive force."""
    assert arrive(Vec2d(50, 50), Vec2d(0, 0)) == Vec2d(
        -0.7071067811865476,
        -0.7071067811865476,
    )


def test_arrive_on_target() -> None:
    """Test if a position on the target produces the correct arrive force."""
    assert arrive(Vec2d(0, 0), Vec2d(0, 0)) == Vec2d(0, 0)


def test_evade_non_moving_target() -> None:
    """Test if a non-moving target produces the correct evade force."""
    assert evade(Vec2d(0, 0), Vec2d(100, 100), Vec2d(0, 0)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_evade_moving_target() -> None:
    """Test if a moving target produces the correct evade force."""
    assert evade(Vec2d(0, 0), Vec2d(100, 100), Vec2d(-50, 0)) == Vec2d(
        -0.5428888213891885,
        -0.8398045770360255,
    )


def test_evade_same_positions() -> None:
    """Test if having the same position produces the correct evade force."""
    assert evade(Vec2d(0, 0), Vec2d(0, 0), Vec2d(0, 0)) == Vec2d(0, 0)
    assert evade(Vec2d(0, 0), Vec2d(0, 0), Vec2d(-50, 0)) == Vec2d(0, 0)


def test_flee_higher_current() -> None:
    """Test if a higher current position produces the correct flee force."""
    assert flee(Vec2d(100, 100), Vec2d(50, 50)) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_flee_higher_target() -> None:
    """Test if a higher target position produces the correct flee force."""
    assert flee(Vec2d(50, 50), Vec2d(100, 100)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_flee_equal() -> None:
    """Test if two equal positions produce the correct flee force."""
    assert flee(Vec2d(100, 100), Vec2d(100, 100)) == Vec2d(0, 0)


def test_flee_negative_current() -> None:
    """Test if a negative current position produces the correct flee force."""
    assert flee(Vec2d(-50, -50), Vec2d(100, 100)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_flee_negative_target() -> None:
    """Test if a negative target position produces the correct flee force."""
    assert flee(Vec2d(100, 100), Vec2d(-50, -50)) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_flee_negative_positions() -> None:
    """Test if two negative positions produce the correct flee force."""
    assert flee(Vec2d(-50, -50), Vec2d(-50, -50)) == Vec2d(0, 0)


def test_follow_path_single_point() -> None:
    """Test if a single point produces the correct follow path force."""
    assert follow_path(Vec2d(100, 100), [Vec2d(250, 250)]) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_follow_path_single_point_reached() -> None:
    """Test if reaching a single-point list produces the correct follow path force."""
    path_list = [Vec2d(100, 100)]
    assert follow_path(Vec2d(100, 100), path_list) == Vec2d(0, 0)
    assert path_list == [Vec2d(100, 100)]


def test_follow_path_multiple_points() -> None:
    """Test if multiple points produces the correct follow path force."""
    assert follow_path(Vec2d(200, 200), [Vec2d(350, 350), Vec2d(500, 500)]) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_follow_path_multiple_points_reached() -> None:
    """Test if reaching a multiple point list produces the correct follow path force."""
    path_list = [Vec2d(100, 100), Vec2d(250, 250)]
    assert follow_path(Vec2d(100, 100), path_list) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )
    assert path_list == [Vec2d(250, 250), Vec2d(100, 100)]


def test_follow_path_empty_list() -> None:
    """Test if an empty list raises the correct exception."""
    with pytest.raises(expected_exception=IndexError):
        follow_path(Vec2d(100, 100), [])


def test_obstacle_avoidance_no_obstacles() -> None:
    """Test if no obstacles produce the correct avoidance force."""
    assert obstacle_avoidance(Vec2d(100, 100), Vec2d(0, 100), set()) == Vec2d(0, 0)


def test_obstacle_avoidance_obstacle_out_of_range() -> None:
    """Test if an out of range obstacle produces the correct avoidance force."""
    assert obstacle_avoidance(Vec2d(100, 100), Vec2d(0, 100), {Vec2d(10, 10)}) == Vec2d(
        0,
        0,
    )


def test_obstacle_avoidance_angled_velocity() -> None:
    """Test if an angled velocity produces the correct avoidance force."""
    assert obstacle_avoidance(Vec2d(100, 100), Vec2d(100, 100), {Vec2d(1, 2)}) == Vec2d(
        0.2588190451025206,
        -0.9659258262890683,
    )


def test_obstacle_avoidance_non_moving() -> None:
    """Test if a non-moving game object produces the correct avoidance force."""
    assert obstacle_avoidance(Vec2d(100, 100), Vec2d(0, 100), {Vec2d(1, 2)}) == Vec2d(
        0,
        0,
    )


def test_obstacle_avoidance_single_forward() -> None:
    """Test if a single forward obstacle produces the correct avoidance force."""
    assert obstacle_avoidance(Vec2d(100, 100), Vec2d(0, 100), {Vec2d(1, 2)}) == Vec2d(
        0,
        0,
    )


def test_obstacle_avoidance_single_left() -> None:
    """Test if a single left obstacle produces the correct avoidance force."""
    assert obstacle_avoidance(
        Vec2d(100, 100),
        Vec2d(0, 100),
        {Vec2d(0, 2)},
    ) == Vec2d(
        0.8660254037844387,
        pytest.approx(-0.5),  # type: ignore[arg-type]
    )


def test_obstacle_avoidance_single_right() -> None:
    """Test if a single right obstacle produces the correct avoidance force."""
    assert obstacle_avoidance(
        Vec2d(100, 100),
        Vec2d(0, 100),
        {Vec2d(2, 2)},
    ) == Vec2d(
        -0.8660254037844386,
        pytest.approx(-0.5),  # type: ignore[arg-type]
    )


def test_obstacle_avoidance_left_forward() -> None:
    """Test if a left and forward obstacle produces the correct avoidance force."""
    assert obstacle_avoidance(
        Vec2d(100, 100),
        Vec2d(0, 100),
        {Vec2d(0, 2), Vec2d(1, 2)},
    ) == Vec2d(
        0.8660254037844387,
        pytest.approx(-0.5),  # type: ignore[arg-type]
    )


def test_obstacle_avoidance_right_forward() -> None:
    """Test if a right and forward obstacle produces the correct avoidance force."""
    assert obstacle_avoidance(
        Vec2d(100, 100),
        Vec2d(0, 100),
        {Vec2d(1, 2), Vec2d(2, 2)},
    ) == Vec2d(
        -0.8660254037844386,
        pytest.approx(-0.5),  # type: ignore[arg-type]
    )


def test_obstacle_avoidance_left_right_forward() -> None:
    """Test if all three obstacles produce the correct avoidance force."""
    assert obstacle_avoidance(
        Vec2d(100, 100),
        Vec2d(0, 100),
        {Vec2d(0, 2), Vec2d(1, 2), Vec2d(2, 2)},
    ) == Vec2d(0, -1)


def test_pursuit_non_moving_target() -> None:
    """Test if a non-moving target produces the correct pursuit force."""
    assert pursuit(Vec2d(0, 0), Vec2d(100, 100), Vec2d(0, 0)) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_pursuit_moving_target() -> None:
    """Test if a moving target produces the correct pursuit force."""
    assert pursuit(Vec2d(0, 0), Vec2d(100, 100), Vec2d(-50, 0)) == Vec2d(
        0.5428888213891885,
        0.8398045770360255,
    )


def test_pursuit_same_positions() -> None:
    """Test if having the same position produces the correct pursuit force."""
    assert pursuit(Vec2d(0, 0), Vec2d(0, 0), Vec2d(0, 0)) == Vec2d(0, 0)
    assert pursuit(Vec2d(0, 0), Vec2d(0, 0), Vec2d(-50, 0)) == Vec2d(0, 0)


def test_seek_higher_current() -> None:
    """Test if a higher current position produces the correct seek force."""
    assert seek(Vec2d(100, 100), Vec2d(50, 50)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_seek_higher_target() -> None:
    """Test if a higher target position produces the correct seek force."""
    assert seek(Vec2d(50, 50), Vec2d(100, 100)) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_seek_equal() -> None:
    """Test if two equal positions produce the correct seek force."""
    assert seek(Vec2d(100, 100), Vec2d(100, 100)) == Vec2d(0, 0)


def test_seek_negative_current() -> None:
    """Test if a negative current position produces the correct seek force."""
    assert seek(Vec2d(-50, -50), Vec2d(100, 100)) == Vec2d(
        0.7071067811865475,
        0.7071067811865475,
    )


def test_seek_negative_target() -> None:
    """Test if a negative target position produces the correct seek force."""
    assert seek(Vec2d(100, 100), Vec2d(-50, -50)) == Vec2d(
        -0.7071067811865475,
        -0.7071067811865475,
    )


def test_seek_negative_positions() -> None:
    """Test if two negative positions produce the correct seek force."""
    assert seek(Vec2d(-50, -50), Vec2d(-50, -50)) == Vec2d(0, 0)


def test_wander_non_moving() -> None:
    """Test if a non-moving game object produces the correct wander force."""
    assert wander(Vec2d(0, 0), 60) == Vec2d(
        0.8660254037844385,
        pytest.approx(-0.5),  # type: ignore[arg-type]
    )


def test_wander_moving() -> None:
    """Test if a moving game object produces the correct wander force."""
    assert wander(Vec2d(100, -100), 60) == Vec2d(
        0.7659012135559103,
        -0.6429582654213131,
    )
