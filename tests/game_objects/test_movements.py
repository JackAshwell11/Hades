"""Tests all functions in game_objects/movements.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest
from pymunk import Vec2d

# Custom
from hades.game_objects.attributes import MovementForce
from hades.game_objects.base import (
    ComponentType,
    SteeringBehaviours,
    SteeringMovementState,
)
from hades.game_objects.components import Footprint
from hades.game_objects.movements import (
    KeyboardMovement,
    SteeringMovement,
    arrive,
    evade,
    flee,
    follow_path,
    obstacle_avoidance,
    pursuit,
    seek,
    wander,
)
from hades.game_objects.system import ECS

__all__ = ()


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def keyboard_movement(ecs: ECS) -> KeyboardMovement:
    """Create a keyboard movement component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The keyboard movement component for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.MOVEMENT_FORCE: (100, 5)}},
        MovementForce,
        KeyboardMovement,
    )
    return cast(
        KeyboardMovement,
        ecs.get_component_for_game_object(0, ComponentType.MOVEMENTS),
    )


@pytest.fixture()
def steering_movement(ecs: ECS) -> SteeringMovement:
    """Create a steering movement component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The steering movement component for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.MOVEMENT_FORCE: (200, 10)}},
        Footprint,
        MovementForce,
        KeyboardMovement,
        steering=True,
    )
    ecs.add_game_object(
        {
            "attributes": {ComponentType.MOVEMENT_FORCE: (100, 5)},
            "steering_behaviours": {},
        },
        MovementForce,
        SteeringMovement,
        steering=True,
    )
    cast(
        SteeringMovement,
        ecs.get_component_for_game_object(1, ComponentType.MOVEMENTS),
    ).target_id = 0
    return cast(
        SteeringMovement,
        ecs.get_component_for_game_object(1, ComponentType.MOVEMENTS),
    )


@pytest.fixture()
def initialised_target(ecs: ECS) -> ECS:
    """Create a target game object for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.MOVEMENT_FORCE: (200, 10)}},
        Footprint,
        MovementForce,
        KeyboardMovement,
        steering=True,
    )
    return ecs


def initialise_steering_movement(
    ecs: ECS,
    steering_behaviours: dict[SteeringMovementState, list[SteeringBehaviours]],
) -> SteeringMovement:
    """Initialise a steering movement component for use in testing.

    Args:
        ecs: The entity component system for use in testing.
        steering_behaviours: The steering behaviours to initialise the component with.

    Returns:
        The steering movement component for use in testing.
    """
    ecs.add_game_object(
        {
            "attributes": {ComponentType.MOVEMENT_FORCE: (100, 5)},
            "steering_behaviours": steering_behaviours,
        },
        MovementForce,
        SteeringMovement,
        steering=True,
    )
    cast(
        SteeringMovement,
        ecs.get_component_for_game_object(1, ComponentType.MOVEMENTS),
    ).target_id = 0
    return cast(
        SteeringMovement,
        ecs.get_component_for_game_object(1, ComponentType.MOVEMENTS),
    )


def test_arrive_outside_slowing_radius() -> None:
    """Test if a position outside the radius produces the correct arrive force."""
    assert arrive(Vec2d(500, 500), Vec2d(0, 0)) == Vec2d(-500, -500)


def test_arrive_on_slowing_range() -> None:
    """Test if a position on the radius produces the correct arrive force."""
    assert arrive(Vec2d(135, 135), Vec2d(0, 0)) == Vec2d(
        -134.23980299088365,
        -134.23980299088365,
    )


def test_arrive_inside_slowing_range() -> None:
    """Test if a position inside the radius produces the correct arrive force."""
    assert arrive(Vec2d(100, 100), Vec2d(0, 0)) == Vec2d(
        -73.6569563735987,
        -73.6569563735987,
    )


def test_arrive_near_target() -> None:
    """Test if a position near the target produces the correct arrive force."""
    assert arrive(Vec2d(50, 50), Vec2d(0, 0)) == Vec2d(
        -18.414239093399676,
        -18.414239093399676,
    )


def test_arrive_on_target() -> None:
    """Test if a position on the target produces the correct arrive force."""
    assert arrive(Vec2d(0, 0), Vec2d(0, 0)) == Vec2d(0, 0)


def test_evade_non_moving_target() -> None:
    """Test if a non-moving target produces the correct evade force."""
    assert evade(Vec2d(0, 0), Vec2d(100, 100), Vec2d(0, 0)) == Vec2d(-100.0, -100.0)


def test_evade_moving_target() -> None:
    """Test if a moving target produces the correct evade force."""
    assert evade(Vec2d(0, 0), Vec2d(100, 100), Vec2d(-50, 0)) == Vec2d(
        -64.64466094067262,
        -100.0,
    )


def test_evade_same_positions() -> None:
    """Test if having the same position produces the correct evade force."""
    assert evade(Vec2d(0, 0), Vec2d(0, 0), Vec2d(0, 0)) == Vec2d(0, 0)
    assert evade(Vec2d(0, 0), Vec2d(0, 0), Vec2d(-50, 0)) == Vec2d(0, 0)


def test_flee_higher_current() -> None:
    """Test if a higher current position produces the correct flee force."""
    assert flee(Vec2d(100, 100), Vec2d(50, 50)) == Vec2d(50, 50)


def test_flee_higher_target() -> None:
    """Test if a higher target position produces the correct flee force."""
    assert flee(Vec2d(50, 50), Vec2d(100, 100)) == Vec2d(-50, -50)


def test_flee_equal() -> None:
    """Test if two equal positions produce the correct flee force."""
    assert flee(Vec2d(100, 100), Vec2d(100, 100)) == Vec2d(0, 0)


def test_flee_negative_current() -> None:
    """Test if a negative current position produces the correct flee force."""
    assert flee(Vec2d(-50, -50), Vec2d(100, 100)) == Vec2d(-150, -150)


def test_flee_negative_target() -> None:
    """Test if a negative target position produces the correct flee force."""
    assert flee(Vec2d(100, 100), Vec2d(-50, -50)) == Vec2d(150, 150)


def test_flee_negative_positions() -> None:
    """Test if two negative positions produce the correct flee force."""
    assert flee(Vec2d(-50, -50), Vec2d(-50, -50)) == Vec2d(0, 0)


def test_follow_path_single_point() -> None:
    """Test if a single point produces the correct follow path force."""
    assert follow_path(Vec2d(100, 100), [Vec2d(250, 250)]) == Vec2d(150, 150)


def test_follow_path_single_point_reached() -> None:
    """Test if reaching a single-point list produces the correct follow path force."""
    path_list = [Vec2d(100, 100)]
    assert follow_path(Vec2d(100, 100), path_list) == Vec2d(0, 0)
    assert path_list == [Vec2d(100, 100)]


def test_follow_path_multiple_points() -> None:
    """Test if multiple points produces the correct follow path force."""
    assert follow_path(Vec2d(200, 200), [Vec2d(350, 350), Vec2d(500, 500)]) == Vec2d(
        150,
        150,
    )


def test_follow_path_multiple_points_reached() -> None:
    """Test if reaching a multiple point list produces the correct follow path force."""
    path_list = [Vec2d(100, 100), Vec2d(250, 250)]
    assert follow_path(Vec2d(100, 100), path_list) == Vec2d(150, 150)
    assert path_list == [Vec2d(250, 250), Vec2d(100, 100)]


def test_follow_path_empty_list() -> None:
    """Test if an empty list raises the correct exception."""
    with pytest.raises(expected_exception=IndexError):
        follow_path(Vec2d(100, 100), [])


def test_obstacle_avoidance() -> None:
    """Test if the obstacle avoidance behaviour works correctly."""
    raise AssertionError


def test_pursuit_non_moving_target() -> None:
    """Test if a non-moving target produces the correct pursuit force."""
    assert pursuit(Vec2d(0, 0), Vec2d(100, 100), Vec2d(0, 0)) == Vec2d(100.0, 100.0)


def test_pursuit_moving_target() -> None:
    """Test if a moving target produces the correct pursuit force."""
    assert pursuit(Vec2d(0, 0), Vec2d(100, 100), Vec2d(-50, 0)) == Vec2d(
        64.64466094067262,
        100.0,
    )


def test_pursuit_same_positions() -> None:
    """Test if having the same position produces the correct pursuit force."""
    assert pursuit(Vec2d(0, 0), Vec2d(0, 0), Vec2d(0, 0)) == Vec2d(0, 0)
    assert pursuit(Vec2d(0, 0), Vec2d(0, 0), Vec2d(-50, 0)) == Vec2d(0, 0)


def test_seek_higher_current() -> None:
    """Test if a higher current position produces the correct seek force."""
    assert seek(Vec2d(100, 100), Vec2d(50, 50)) == Vec2d(-50, -50)


def test_seek_higher_target() -> None:
    """Test if a higher target position produces the correct seek force."""
    assert seek(Vec2d(50, 50), Vec2d(100, 100)) == Vec2d(50, 50)


def test_seek_equal() -> None:
    """Test if two equal positions produce the correct seek force."""
    assert seek(Vec2d(100, 100), Vec2d(100, 100)) == Vec2d(0, 0)


def test_seek_negative_current() -> None:
    """Test if a negative current position produces the correct seek force."""
    assert seek(Vec2d(-50, -50), Vec2d(100, 100)) == Vec2d(150, 150)


def test_seek_negative_target() -> None:
    """Test if a negative target position produces the correct seek force."""
    assert seek(Vec2d(100, 100), Vec2d(-50, -50)) == Vec2d(-150, -150)


def test_seek_negative_positions() -> None:
    """Test if two negative positions produce the correct seek force."""
    assert seek(Vec2d(-50, -50), Vec2d(-50, -50)) == Vec2d(0, 0)


def test_wander_non_moving() -> None:
    """Test if a non-moving game object produces the correct wander force."""
    assert wander(Vec2d(0, 0), 60) == Vec2d(21.650635094610966, -12.500000000000004)


def test_wander_moving() -> None:
    """Test if a moving game object produces the correct wander force."""
    assert wander(Vec2d(100, -100), 60) == Vec2d(57.00597415393834, -47.85533905932738)


def test_keyboard_movement_init(keyboard_movement: KeyboardMovement) -> None:
    """Test if the keyboard movement component is initialised correctly.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    assert (
        repr(keyboard_movement)
        == "<KeyboardMovement (North pressed=False) (South pressed=False) (East"
        " pressed=False) (West pressed=False)>"
    )


def test_keyboard_movement_calculate_force_none(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if no keys are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    assert keyboard_movement.calculate_force() == (0, 0)


def test_keyboard_movement_calculate_force_north(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move north.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.north_pressed = True
    assert keyboard_movement.calculate_force() == (0, 100)


def test_keyboard_movement_calculate_force_south(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move south.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.south_pressed = True
    assert keyboard_movement.calculate_force() == (0, -100)


def test_keyboard_movement_calculate_force_east(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move east.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.east_pressed = True
    assert keyboard_movement.calculate_force() == (100, 0)


def test_keyboard_movement_calculate_force_west(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move west.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (-100, 0)


def test_keyboard_movement_calculate_force_east_west(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if east and west are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.east_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (0, 0)


def test_keyboard_movement_calculate_force_north_south(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if north and south are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.east_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (0, 0)


def test_keyboard_movement_calculate_force_north_west(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if north and west are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.north_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (-100, 100)


def test_keyboard_movement_calculate_force_south_east(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if south and east are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.south_pressed = True
    keyboard_movement.east_pressed = True
    assert keyboard_movement.calculate_force() == (100, -100)


def test_steering_movement_init(steering_movement: SteeringMovement) -> None:
    """Test if the steering movement component is initialised correctly.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    assert (
        repr(steering_movement)
        == "<SteeringMovement (Behaviour count=0) (Target game object ID=0)>"
    )


def test_steering_movement_calculate_force_within_target_distance_empty_path_list(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the state is correctly changed to the target state.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    steering_movement.calculate_force()
    assert steering_movement.movement_state == SteeringMovementState.TARGET


def test_steering_movement_calculate_force_within_target_distance_non_empty_path_list(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the state is correctly changed to the target state.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    steering_movement.path_list = [Vec2d(300, 300), Vec2d(400, 400)]
    steering_movement.calculate_force()
    assert steering_movement.movement_state == SteeringMovementState.TARGET


def test_steering_movement_calculate_force_outside_target_distance_empty_path_list(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the state is correctly changed to the default state.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        500,
        500,
    )
    steering_movement.calculate_force()
    assert steering_movement.movement_state == SteeringMovementState.DEFAULT


def test_steering_movement_calculate_force_outside_target_distance_non_empty_path_list(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the state is correctly changed to the footprint state.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        500,
        500,
    )
    steering_movement.path_list = [Vec2d(300, 300), Vec2d(400, 400)]
    steering_movement.calculate_force()
    assert steering_movement.movement_state == SteeringMovementState.FOOTPRINT


def test_steering_movement_calculate_force_missing_state(
    initialised_target: ECS,
) -> None:
    """Test if a zero force is calculated if the state is missing.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(initialised_target, {})
    assert steering_movement.calculate_force() == Vec2d(0, 0)


def test_steering_movement_calculate_force_arrive(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the arrive behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.ARRIVE]},
    )
    steering_movement.system.get_steering_object_for_game_object(0).position = Vec2d(
        0,
        0,
    )
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert (
        steering_movement.calculate_force()
        == arrive(Vec2d(100, 100), Vec2d(0, 0)).normalized() * 100
    )


def test_steering_movement_calculate_force_evade(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the evade behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.EVADE]},
    )
    steering_movement.system.get_steering_object_for_game_object(0).position = Vec2d(
        100,
        100,
    )
    steering_movement.system.get_steering_object_for_game_object(0).velocity = Vec2d(
        -50,
        0,
    )
    assert (
        steering_movement.calculate_force()
        == evade(
            Vec2d(0, 0),
            Vec2d(100, 100),
            Vec2d(-50, 0),
        ).normalized()
        * 100
    )


def test_steering_movement_calculate_force_flee(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the flee behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.FLEE]},
    )
    steering_movement.system.get_steering_object_for_game_object(0).position = Vec2d(
        50,
        50,
    )
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert (
        steering_movement.calculate_force()
        == flee(Vec2d(100, 100), Vec2d(50, 50)).normalized() * 100
    )


def test_steering_movement_calculate_force_follow_path(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the follow path behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.FOOTPRINT: [SteeringBehaviours.FOLLOW_PATH]},
    )
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        200,
        200,
    )
    steering_movement.path_list = [Vec2d(350, 350), Vec2d(500, 500)]
    assert (
        steering_movement.calculate_force()
        == follow_path(
            Vec2d(200, 200),
            [Vec2d(350, 350), Vec2d(500, 500)],
        ).normalized()
        * 100
    )


def test_steering_movement_calculate_force_obstacle_avoidance(
    initialised_target: ECS,
) -> None:
    """Test if the correct force is calculated for the obstacle avoidance behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.OBSTACLE_AVOIDANCE]},
    )
    assert (
        steering_movement.calculate_force() == obstacle_avoidance().normalized() * 100
    )


def test_steering_movement_calculate_force_pursuit(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the pursuit behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT]},
    )
    steering_movement.system.get_steering_object_for_game_object(0).position = Vec2d(
        100,
        100,
    )
    steering_movement.system.get_steering_object_for_game_object(0).velocity = Vec2d(
        -50,
        0,
    )
    assert (
        steering_movement.calculate_force()
        == pursuit(
            Vec2d(0, 0),
            Vec2d(100, 100),
            Vec2d(-50, 0),
        ).normalized()
        * 100
    )


def test_steering_movement_calculate_force_seek(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the seek behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.SEEK]},
    )
    steering_movement.system.get_steering_object_for_game_object(0).position = Vec2d(
        50,
        50,
    )
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert (
        steering_movement.calculate_force()
        == seek(Vec2d(100, 100), Vec2d(50, 50)).normalized() * 100
    )


def test_steering_movement_calculate_force_wander(initialised_target: ECS) -> None:
    """Test if the correct force is calculated for the wander behaviour.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {SteeringMovementState.TARGET: [SteeringBehaviours.WANDER]},
    )
    steering_movement.system.get_steering_object_for_game_object(1).velocity = Vec2d(
        100,
        -100,
    )
    steering_force = steering_movement.calculate_force()
    assert steering_force != steering_movement.calculate_force()
    assert round(steering_force.length) == 100


def test_steering_movement_calculate_force_multiple_behaviours(
    initialised_target: ECS,
) -> None:
    """Test if the correct force is calculated when multiple behaviours are selected.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    steering_movement = initialise_steering_movement(
        initialised_target,
        {
            SteeringMovementState.FOOTPRINT: [
                SteeringBehaviours.FOLLOW_PATH,
                SteeringBehaviours.SEEK,
            ],
        },
    )
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        300,
        300,
    )
    steering_movement.path_list = [Vec2d(100, 200), Vec2d(-100, 0)]
    assert steering_movement.calculate_force() == Vec2d(
        -78.08688094430303,
        -62.46950475544243,
    )


def test_steering_movement_calculate_force_multiple_states(
    initialised_target: ECS,
) -> None:
    """Test if the correct force is calculated when multiple states are initialised.

    Args:
        initialised_target: The entity component system for use in testing.
    """
    # Initialise the steering movement component with multiple states
    steering_movement = initialise_steering_movement(
        initialised_target,
        {
            SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT],
            SteeringMovementState.DEFAULT: [SteeringBehaviours.SEEK],
        },
    )

    # Test the target state
    steering_movement.system.get_steering_object_for_game_object(0).velocity = Vec2d(
        -50,
        100,
    )
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert steering_movement.calculate_force() == Vec2d(
        -97.73793955511094,
        -21.14935392681019,
    )

    # Test the default state
    steering_movement.system.get_steering_object_for_game_object(1).position = Vec2d(
        300,
        300,
    )
    assert steering_movement.calculate_force() == Vec2d(
        -70.71067811865474,
        -70.71067811865474,
    )


def test_steering_movement_update_path_list_within_distance(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the position is within the distance.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list([Vec2d(300, 300), Vec2d(100, 100)])
    assert steering_movement.path_list == [(100, 100)]


def test_steering_movement_update_path_list_outside_distance(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the position is outside the distance.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list([Vec2d(300, 300), Vec2d(500, 500)])
    assert steering_movement.path_list == []


def test_steering_movement_update_path_list_equal_distance(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the position is equal to the distance.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list(
        [Vec2d(300, 300), Vec2d(135.764501987, 135.764501987)],
    )
    assert steering_movement.path_list == [(135.764501987, 135.764501987)]


def test_steering_movement_update_path_list_slice(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated with the array slice.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list([Vec2d(100, 100), Vec2d(300, 300)])
    assert steering_movement.path_list == [(100, 100), (300, 300)]


def test_steering_movement_update_path_list_empty_list(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the footprints list is empty.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list([])
    assert steering_movement.path_list == []


def test_steering_movement_update_path_list_multiple_points(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if multiple footprints are within distance.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list(
        [Vec2d(100, 100), Vec2d(300, 300), Vec2d(50, 100), Vec2d(500, 500)],
    )
    assert steering_movement.path_list == [(50, 100), (500, 500)]


def test_steering_movement_update_path_list_footprint_on_update(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated correctly if the Footprint component updates it.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    cast(
        Footprint,
        steering_movement.system.get_component_for_game_object(
            0,
            ComponentType.FOOTPRINT,
        ),
    ).on_update(0.5)
    assert steering_movement.path_list == [(0, 0)]
