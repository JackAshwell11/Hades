"""Tests all classes and functions in game_objects/systems/movements.py."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import pytest

# Custom
from hades.game_objects.base import SteeringBehaviours, SteeringMovementState
from hades.game_objects.components import (
    Footprint,
    KeyboardMovement,
    MovementForce,
    SteeringMovement,
)
from hades.game_objects.registry import Registry, RegistryError
from hades.game_objects.steering import Vec2d
from hades.game_objects.systems.movements import (
    FootprintSystem,
    KeyboardMovementSystem,
    SteeringMovementSystem,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ()


@pytest.fixture()
def registry() -> Registry:
    """Create a registry for use in testing.

    Returns:
        The registry for use in testing.
    """
    return Registry()


@pytest.fixture()
def footprint_system(registry: Registry) -> FootprintSystem:
    """Create a footprint system for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The footprint system for use in testing.
    """
    registry.create_game_object(Footprint(), kinematic=True)
    footprint_system = FootprintSystem(registry)
    registry.add_system(footprint_system)
    registry.add_system(SteeringMovementSystem(registry))
    return footprint_system


@pytest.fixture()
def keyboard_movement_system(registry: Registry) -> KeyboardMovementSystem:
    """Create a keyboard movement system for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The keyboard movement system for use in testing.
    """
    registry.create_game_object(
        Footprint(),
        Footprint(),
        MovementForce(100, 5),
        KeyboardMovement(),
        kinematic=True,
    )
    keyboard_movement_system = KeyboardMovementSystem(registry)
    registry.add_system(keyboard_movement_system)
    registry.add_system(FootprintSystem(registry))
    return keyboard_movement_system


@pytest.fixture()
def steering_movement_system_factory(
    keyboard_movement_system: KeyboardMovementSystem,
) -> Callable[
    [dict[SteeringMovementState, list[SteeringBehaviours]]],
    SteeringMovementSystem,
]:
    """Create a steering movement system factory for use in testing.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.

    Returns:
        The steering movement system factory for use in testing.
    """

    def wrap(
        steering_behaviours: dict[SteeringMovementState, list[SteeringBehaviours]],
    ) -> SteeringMovementSystem:
        steering_movement_system = SteeringMovementSystem(
            keyboard_movement_system.registry,
        )
        game_object_id = steering_movement_system.registry.create_game_object(
            MovementForce(100, 5),
            SteeringMovement(steering_behaviours),
            kinematic=True,
        )
        steering_movement_system.registry.add_system(steering_movement_system)
        steering_movement_system.registry.get_component_for_game_object(
            game_object_id,
            SteeringMovement,
        ).target_id = 0
        return steering_movement_system

    return wrap


@pytest.fixture()
def steering_movement_system(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> SteeringMovementSystem:
    """Create a steering movement system for use in testing.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.

    Returns:
        The steering movement system for use in testing.
    """
    return steering_movement_system_factory({})


def test_footprint_system_init(footprint_system: FootprintSystem) -> None:
    """Test that the footprint system is initialised correctly."""
    assert (
        repr(footprint_system)
        == "<FootprintSystem (Description=`Provides facilities to manipulate footprint"
        " components.`)>"
    )


def test_footprint_system_update_small_deltatime(
    footprint_system: FootprintSystem,
) -> None:
    """Test that the footprint system is updated with a small delta time.

    Args:
        footprint_system: The footprint system for use in testing.
    """
    footprint_system.update(0.1)
    footprint = footprint_system.registry.get_component_for_game_object(0, Footprint)
    assert footprint.footprints == []
    assert footprint.time_since_last_footprint == 0.1


def test_footprint_system_update_large_deltatime_empty_list(
    footprint_system: FootprintSystem,
) -> None:
    """Test that the footprint system creates a footprint in an empty list.

    Args:
        footprint_system: The footprint system for use in testing.
    """
    footprint_system.update(1)
    footprint = footprint_system.registry.get_component_for_game_object(0, Footprint)
    assert footprint.footprints == [Vec2d(0, 0)]
    assert footprint.time_since_last_footprint == 0


def test_footprint_system_update_large_deltatime_non_empty_list(
    footprint_system: FootprintSystem,
) -> None:
    """Test that the footprint system creates a footprint in a non-empty list.

    Args:
        footprint_system: The footprint system for use in testing.
    """
    footprint = footprint_system.registry.get_component_for_game_object(0, Footprint)
    footprint.footprints = [Vec2d(1, 1), Vec2d(2, 2), Vec2d(3, 3)]
    footprint_system.update(0.5)
    assert footprint.footprints == [Vec2d(1, 1), Vec2d(2, 2), Vec2d(3, 3), Vec2d(0, 0)]


def test_footprint_system_update_large_deltatime_full_list(
    footprint_system: FootprintSystem,
) -> None:
    """Test that the footprint system creates a footprint and removes the oldest one.

    Args:
        footprint_system: The footprint system for use in testing.
    """
    footprint = footprint_system.registry.get_component_for_game_object(0, Footprint)
    footprint.footprints = [
        Vec2d(0, 0),
        Vec2d(1, 1),
        Vec2d(2, 2),
        Vec2d(3, 3),
        Vec2d(4, 4),
        Vec2d(5, 5),
        Vec2d(6, 6),
        Vec2d(7, 7),
        Vec2d(8, 8),
        Vec2d(9, 9),
    ]
    footprint_system.registry.get_kinematic_object_for_game_object(0).position = Vec2d(
        10,
        10,
    )
    footprint_system.update(0.5)
    assert footprint.footprints == [
        Vec2d(1, 1),
        Vec2d(2, 2),
        Vec2d(3, 3),
        Vec2d(4, 4),
        Vec2d(5, 5),
        Vec2d(6, 6),
        Vec2d(7, 7),
        Vec2d(8, 8),
        Vec2d(9, 9),
        Vec2d(10, 10),
    ]


def test_footprint_system_update_multiple_updates(
    footprint_system: FootprintSystem,
) -> None:
    """Test that the footprint system is updated correctly multiple times.

    Args:
        footprint_system: The footprint system for use in testing.
    """
    footprint = footprint_system.registry.get_component_for_game_object(0, Footprint)
    footprint_system.update(0.6)
    assert footprint.footprints == [Vec2d(0, 0)]
    assert footprint.time_since_last_footprint == 0
    footprint_system.registry.get_kinematic_object_for_game_object(0).position = Vec2d(
        1,
        1,
    )
    footprint_system.update(0.7)
    assert footprint.footprints == [Vec2d(0, 0), Vec2d(1, 1)]
    assert footprint.time_since_last_footprint == 0


def test_keyboard_movement_system_init(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the keyboard movement system is initialised correctly.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    assert (
        repr(keyboard_movement_system)
        == "<KeyboardMovementSystem (Description=`Provides facilities to manipulate"
        " keyboard movement components.`)>"
    )


def test_keyboard_movement_system_calculate_force_none(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated if no keys are pressed.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    assert keyboard_movement_system.calculate_force(0) == Vec2d(0, 0)


def test_keyboard_movement_system_calculate_force_north(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated for a move north.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    ).north_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(0, 100)


def test_keyboard_movement_system_calculate_force_south(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated for a move south.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    ).south_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(0, -100)


def test_keyboard_movement_system_calculate_force_east(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated for a move east.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    ).east_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(100, 0)


def test_keyboard_movement_system_calculate_force_west(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated for a move west.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    ).west_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(-100, 0)


def test_keyboard_movement_system_calculate_force_east_west(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated if east and west are pressed.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement = keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    )
    keyboard_movement.east_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(0, 0)


def test_keyboard_movement_system_calculate_force_north_south(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated if north and south are pressed.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement = keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    )
    keyboard_movement.east_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(0, 0)


def test_keyboard_movement_system_calculate_force_north_west(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated if north and west are pressed.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement = keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    )
    keyboard_movement.north_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(-100, 100)


def test_keyboard_movement_system_calculate_force_south_east(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test if the correct force is calculated if south and east are pressed.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    keyboard_movement = keyboard_movement_system.registry.get_component_for_game_object(
        0,
        KeyboardMovement,
    )
    keyboard_movement.south_pressed = True
    keyboard_movement.east_pressed = True
    assert keyboard_movement_system.calculate_force(0) == Vec2d(100, -100)


def test_keyboard_movement_system_calculate_force_invalid_game_object_id(
    keyboard_movement_system: KeyboardMovementSystem,
) -> None:
    """Test that an exception is raised if an invalid game object ID is provided.

    Args:
        keyboard_movement_system: The keyboard movement system for use in testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        keyboard_movement_system.calculate_force(-1)


def test_steering_movement_system_init(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the steering movement system is initialised correctly.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    assert (
        repr(steering_movement_system)
        == "<SteeringMovementSystem (Description=`Provides facilities to manipulate"
        " steering movement components.`)>"
    )


def test_steering_movement_system_calculate_force_within_distance_empty_path_list(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the state is correctly changed to the target state.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(100, 100)
    steering_movement_system.calculate_force(1)
    assert (
        steering_movement_system.registry.get_component_for_game_object(
            1,
            SteeringMovement,
        ).movement_state
        == SteeringMovementState.TARGET
    )


def test_steering_movement_system_calculate_force_within_distance_non_empty_path_list(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the state is correctly changed to the target state.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(100, 100)
    steering_movement = steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    )
    steering_movement.path_list = [Vec2d(300, 300), Vec2d(400, 400)]
    steering_movement_system.calculate_force(1)
    assert steering_movement.movement_state == SteeringMovementState.TARGET


def test_steering_movement_system_calculate_force_outside_distance_empty_path_list(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the state is correctly changed to the default state.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(500, 500)
    steering_movement_system.calculate_force(1)
    assert (
        steering_movement_system.registry.get_component_for_game_object(
            1,
            SteeringMovement,
        ).movement_state
        == SteeringMovementState.DEFAULT
    )


def test_steering_movement_system_calculate_force_outside_distance_non_empty_path_list(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the state is correctly changed to the footprint state.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(500, 500)
    steering_movement = steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    )
    steering_movement.path_list = [Vec2d(300, 300), Vec2d(400, 400)]
    steering_movement_system.calculate_force(1)
    assert steering_movement.movement_state == SteeringMovementState.FOOTPRINT


def test_steering_movement_system_calculate_force_missing_state(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if a zero force is calculated if the state is missing.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    assert steering_movement_system_factory({}).calculate_force(1) == Vec2d(0, 0)


def test_steering_movement_system_calculate_force_arrive(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the arrive behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.ARRIVE]},
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        0,
    ).position = Vec2d(0, 0)
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(100, 100)
    assert steering_movement_system.calculate_force(1) == Vec2d(
        -70.71067811865476,
        -70.71067811865476,
    )


def test_steering_movement_system_calculate_force_evade(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the evade behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.EVADE]},
    )
    kinematic_object = (
        steering_movement_system.registry.get_kinematic_object_for_game_object(0)
    )
    kinematic_object.position = Vec2d(100, 100)
    kinematic_object.velocity = Vec2d(-50, 0)
    assert steering_movement_system.calculate_force(1) == Vec2d(
        -54.28888213891886,
        -83.98045770360257,
    )


def test_steering_movement_system_calculate_force_flee(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the flee behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.FLEE]},
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        0,
    ).position = Vec2d(50, 50)
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(
        100,
        100,
    )
    assert steering_movement_system.calculate_force(1) == Vec2d(
        70.71067811865475,
        70.71067811865475,
    )


def test_steering_movement_system_calculate_force_follow_path(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the follow path behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.FOOTPRINT: [SteeringBehaviours.FOLLOW_PATH]},
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(
        200,
        200,
    )
    steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list = [Vec2d(350, 350), Vec2d(500, 500)]
    assert steering_movement_system.calculate_force(1) == Vec2d(
        70.71067811865475,
        70.71067811865475,
    )


def test_steering_movement_system_calculate_force_obstacle_avoidance(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the obstacle avoidance behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.OBSTACLE_AVOIDANCE]},
    )
    kinematic_object = (
        steering_movement_system.registry.get_kinematic_object_for_game_object(1)
    )
    kinematic_object.position = Vec2d(100, 100)
    kinematic_object.velocity = Vec2d(100, 100)
    steering_movement_system.registry.walls = {Vec2d(1, 2)}
    assert steering_movement_system.calculate_force(1) == Vec2d(
        25.881904510252056,
        -96.59258262890683,
    )


def test_steering_movement_system_calculate_force_pursuit(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the pursuit behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT]},
    )
    kinematic_object = (
        steering_movement_system.registry.get_kinematic_object_for_game_object(0)
    )
    kinematic_object.position = Vec2d(100, 100)
    kinematic_object.velocity = Vec2d(-50, 0)
    assert steering_movement_system.calculate_force(1) == Vec2d(
        54.28888213891886,
        83.98045770360257,
    )


def test_steering_movement_system_calculate_force_seek(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the seek behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.SEEK]},
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        0,
    ).position = Vec2d(
        50,
        50,
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(
        100,
        100,
    )
    assert steering_movement_system.calculate_force(1) == Vec2d(
        -70.71067811865475,
        -70.71067811865475,
    )


def test_steering_movement_system_calculate_force_wander(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated for the wander behaviour.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.WANDER]},
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).velocity = Vec2d(
        100,
        -100,
    )
    steering_force = steering_movement_system.calculate_force(1)
    assert steering_force != steering_movement_system.calculate_force(1)
    assert round(abs(steering_force)) == 100


def test_steering_movement_system_calculate_force_multiple_behaviours(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated when multiple behaviours are selected.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    steering_movement_system = steering_movement_system_factory(
        {
            SteeringMovementState.FOOTPRINT: [
                SteeringBehaviours.FOLLOW_PATH,
                SteeringBehaviours.SEEK,
            ],
        },
    )
    steering_movement_system.registry.get_kinematic_object_for_game_object(
        1,
    ).position = Vec2d(300, 300)
    steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list = [Vec2d(100, 200), Vec2d(-100, 0)]
    assert steering_movement_system.calculate_force(1) == Vec2d(
        -81.12421851755609,
        -58.47102846637651,
    )


def test_steering_movement_system_calculate_force_multiple_states(
    steering_movement_system_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovementSystem,
    ],
) -> None:
    """Test if the correct force is calculated when multiple states are initialised.

    Args:
        steering_movement_system_factory: The steering movement system factory for use
            in testing.
    """
    # Initialise the steering movement component with multiple states
    steering_movement_system = steering_movement_system_factory(
        {
            SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT],
            SteeringMovementState.DEFAULT: [SteeringBehaviours.SEEK],
        },
    )
    kinematic_owner, kinematic_target = (
        steering_movement_system.registry.get_kinematic_object_for_game_object(1),
        steering_movement_system.registry.get_kinematic_object_for_game_object(0),
    )

    # Test the target state
    kinematic_target.velocity = Vec2d(-50, 100)
    kinematic_owner.position = Vec2d(100, 100)
    assert steering_movement_system.calculate_force(1) == Vec2d(
        -97.73793955511094,
        -21.14935392681019,
    )

    # Test the default state
    kinematic_owner.position = Vec2d(300, 300)
    assert steering_movement_system.calculate_force(1) == Vec2d(
        -70.71067811865476,
        -70.71067811865476,
    )


def test_steering_movement_system_calculate_force_invalid_game_object_id(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test that an exception is raised if an invalid game object ID is provided.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        steering_movement_system.calculate_force(-1)


def test_steering_movement_system_update_path_list_within_distance(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated if the position is within the view distance.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.update_path_list(0, [Vec2d(300, 300), Vec2d(100, 100)])
    assert steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list == [Vec2d(100, 100)]


def test_steering_movement_system_update_path_list_outside_distance(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated if the position is outside the view distance.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.update_path_list(0, [Vec2d(300, 300), Vec2d(500, 500)])
    assert (
        steering_movement_system.registry.get_component_for_game_object(
            1,
            SteeringMovement,
        ).path_list
        == []
    )


def test_steering_movement_system_update_path_list_equal_distance(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated if the position is equal to the view distance.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.update_path_list(
        0,
        [Vec2d(300, 300), Vec2d(135.764501987, 135.764501987)],
    )
    assert steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list == [Vec2d(135.764501987, 135.764501987)]


def test_steering_movement_system_update_path_list_slice(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated with the array slice.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.update_path_list(0, [Vec2d(100, 100), Vec2d(300, 300)])
    assert steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list == [Vec2d(100, 100), Vec2d(300, 300)]


def test_steering_movement_system_update_path_list_empty_list(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated if the footprints list is empty.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.update_path_list(0, [])
    assert (
        steering_movement_system.registry.get_component_for_game_object(
            1,
            SteeringMovement,
        ).path_list
        == []
    )


def test_steering_movement_system_update_path_list_multiple_points(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated if multiple footprints are within view distance.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.update_path_list(
        0,
        [Vec2d(100, 100), Vec2d(300, 300), Vec2d(50, 100), Vec2d(500, 500)],
    )
    assert steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list == [Vec2d(50, 100), Vec2d(500, 500)]


def test_steering_movement_update_path_list_different_target_id(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is not updated if the target ID doesn't match.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).target_id = -1
    steering_movement_system.update_path_list(0, [Vec2d(100, 100)])
    assert (
        steering_movement_system.registry.get_component_for_game_object(
            1,
            SteeringMovement,
        ).path_list
        == []
    )


def test_steering_movement_system_update_path_list_footprint_update(
    steering_movement_system: SteeringMovementSystem,
) -> None:
    """Test if the path list is updated correctly if the Footprint component updates it.

    Args:
        steering_movement_system: The steering movement system for use in testing.
    """
    steering_movement_system.registry.get_system(FootprintSystem).update(0.5)
    assert steering_movement_system.registry.get_component_for_game_object(
        1,
        SteeringMovement,
    ).path_list == [Vec2d(0, 0)]
