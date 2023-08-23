"""Tests all classes and functions in game_objects/systems/attacks.py."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import pytest

# Custom
from hades.game_objects.base import AttackAlgorithms
from hades.game_objects.components import Armour, Attacks, Health
from hades.game_objects.registry import Registry, RegistryError
from hades.game_objects.steering import Vec2d
from hades.game_objects.systems.attacks import AttackSystem
from hades.game_objects.systems.attributes import GameObjectAttributeSystem

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
def attack_system_factory(
    registry: Registry,
) -> Callable[[list[AttackAlgorithms]], AttackSystem]:
    """Create an attack system factory for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The attack system factory for use in testing.
    """

    def wrap(enabled_attacks: list[AttackAlgorithms]) -> AttackSystem:
        attack_system = AttackSystem(registry)
        game_object_id = registry.create_game_object(
            Attacks(enabled_attacks),
            kinematic=True,
        )
        registry.add_system(attack_system)
        registry.add_system(GameObjectAttributeSystem(registry))
        registry.get_kinematic_object_for_game_object(game_object_id).rotation = 180
        return attack_system

    return wrap


@pytest.fixture()
def targets(registry: Registry) -> list[int]:
    """Create a list of targets for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The list of targets for use in testing.
    """

    def create_target(position: Vec2d) -> int:
        target = registry.create_game_object(
            Health(50, -1),
            Armour(0, -1),
            kinematic=True,
        )
        registry.get_kinematic_object_for_game_object(target).position = position
        return target

    return [
        create_target(Vec2d(-20, -100)),
        create_target(Vec2d(20, 60)),
        create_target(Vec2d(-200, 100)),
        create_target(Vec2d(100, -100)),
        create_target(Vec2d(-100, -99)),
        create_target(Vec2d(0, -200)),
        create_target(Vec2d(0, -192)),
        create_target(Vec2d(0, 0)),
    ]


def test_attacks_init(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that the attacks component is initialised correctly.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    assert (
        repr(attack_system_factory([AttackAlgorithms.AREA_OF_EFFECT_ATTACK]))
        == "<AttackSystem (Description=`Provides facilities to manipulate attack"
        " components.`)>"
    )


def test_attacks_do_attack_area_of_effect_attack(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
    targets: list[int],
) -> None:
    """Test that performing an area of effect attack works correctly.

    Args:
        attack_system_factory: The attack system factory for use in testing.
        targets: The list of targets for use in testing.
    """
    attack_system = attack_system_factory([AttackAlgorithms.AREA_OF_EFFECT_ATTACK])
    assert attack_system.do_attack(8, targets) == {}
    assert (
        attack_system.registry.get_component_for_game_object(targets[0], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[1], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[2], Health).value
        == 50
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[3], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[4], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[5], Health).value
        == 50
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[6], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[7], Health).value
        == 40
    )


def test_attacks_do_attack_melee_attack(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
    targets: list[int],
) -> None:
    """Test that performing a melee attack works correctly.

    Args:
        attack_system_factory: The attack system factory for use in testing.
        targets: The list of targets for use in testing.
    """
    attack_system = attack_system_factory([AttackAlgorithms.MELEE_ATTACK])
    assert attack_system.do_attack(8, targets) == {}
    assert (
        attack_system.registry.get_component_for_game_object(targets[0], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[1], Health).value
        == 50
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[2], Health).value
        == 50
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[3], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[4], Health).value
        == 50
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[5], Health).value
        == 50
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[6], Health).value
        == 40
    )
    assert (
        attack_system.registry.get_component_for_game_object(targets[7], Health).value
        == 40
    )


def test_attacks_do_attack_ranged_attack(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that performing a ranged attack works correctly.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    attack_system = attack_system_factory([AttackAlgorithms.RANGED_ATTACK])
    assert attack_system.do_attack(0, []) == {
        "ranged_attack": (
            Vec2d(0, 0),
            -300.0,
            pytest.approx(0),  # This is due to floating point errors
        ),
    }


def test_attacks_do_attack_invalid_game_object_id(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that an exception is raised if an invalid game object ID is provided.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        attack_system_factory([]).do_attack(-1, [])


def test_attacks_previous_next_attack_single(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that switching between attacks once works correctly.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    attack_system = attack_system_factory(
        [
            AttackAlgorithms.AREA_OF_EFFECT_ATTACK,
            AttackAlgorithms.MELEE_ATTACK,
            AttackAlgorithms.RANGED_ATTACK,
        ],
    )
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )
    attack_system.next_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 1
    )
    attack_system.previous_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )


def test_attacks_previous_attack_multiple(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that switching between attacks multiple times works correctly.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    attack_system = attack_system_factory(
        [
            AttackAlgorithms.AREA_OF_EFFECT_ATTACK,
            AttackAlgorithms.MELEE_ATTACK,
            AttackAlgorithms.RANGED_ATTACK,
        ],
    )
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )
    attack_system.next_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 1
    )
    attack_system.next_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 2
    )
    attack_system.next_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 2
    )
    attack_system.previous_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 1
    )
    attack_system.previous_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )
    attack_system.previous_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )


def test_attacks_previous_next_attack_empty_attacks(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that changing the attack state works correctly when there are no attacks.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    attack_system = attack_system_factory([])
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )
    attack_system.next_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == -1
    )
    attack_system.previous_attack(0)
    assert (
        attack_system.registry.get_component_for_game_object(0, Attacks).attack_state
        == 0
    )


def test_attacks_previous_next_attack_invalid_game_object_id(
    attack_system_factory: Callable[[list[AttackAlgorithms]], AttackSystem],
) -> None:
    """Test that an exception is raised if an invalid game object ID is provided.

    Args:
        attack_system_factory: The attack system factory for use in testing.
    """
    attack_system = attack_system_factory([])
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        attack_system.previous_attack(-1)
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        attack_system.next_attack(-1)
