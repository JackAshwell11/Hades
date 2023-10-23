"""Tests all functions in game_objects/attacks.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
import pytest

# Custom
from hades.game_objects.attacks import Attacks
from hades.game_objects.attributes import Armour, Health
from hades.game_objects.base import AttackAlgorithms, ComponentType, Vec2d
from hades.game_objects.system import ECS

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.base import ComponentData

__all__ = ()


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def attacks_factory(ecs: ECS) -> Callable[[list[AttackAlgorithms]], Attacks]:
    """Create an attacks component factory for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The attacks component factory for use in testing.
    """

    def wrap(enabled_attacks: list[AttackAlgorithms]) -> Attacks:
        game_object_id = ecs.add_game_object(
            {"enabled_attacks": enabled_attacks},
            Attacks,
            physics=True,
        )
        ecs.get_physics_object_for_game_object(game_object_id).rotation = 180
        return cast(
            Attacks,
            ecs.get_component_for_game_object(game_object_id, ComponentType.ATTACKS),
        )

    return wrap


@pytest.fixture()
def targets(ecs: ECS) -> list[int]:
    """Create a list of targets for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The list of targets for use in testing.
    """

    def create_target(position: Vec2d) -> int:
        target = ecs.add_game_object(component_data, Health, Armour, physics=True)
        ecs.get_physics_object_for_game_object(target).position = position
        return target

    component_data: ComponentData = {
        "attributes": {
            ComponentType.HEALTH: (50, -1),
            ComponentType.ARMOUR: (0, -1),
        },
    }
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
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
) -> None:
    """Test that the attacks component is initialised correctly.

    Args:
        attacks_factory: The attacks component factory for use in testing.
    """
    assert (
        repr(attacks_factory([AttackAlgorithms.AREA_OF_EFFECT_ATTACK]))
        == "<Attacks (Attack algorithm count=1)>"
    )


def test_attacks_do_attack_area_of_effect_attack(
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
    targets: list[int],
) -> None:
    """Test that performing an area of effect attack works correctly.

    Args:
        attacks_factory: The attacks component factory for use in testing.
        targets: The list of targets for use in testing.
    """
    attacks_obj = attacks_factory([AttackAlgorithms.AREA_OF_EFFECT_ATTACK])
    assert attacks_obj.do_attack(targets) == {}
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[0],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[1],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[2],
                ComponentType.HEALTH,
            ),
        ).value
        == 50
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[3],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[4],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[5],
                ComponentType.HEALTH,
            ),
        ).value
        == 50
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[6],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[7],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )


def test_attacks_do_attack_melee_attack(
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
    targets: list[int],
) -> None:
    """Test that performing a melee attack works correctly.

    Args:
        attacks_factory: The attacks component factory for use in testing.
        targets: The list of targets for use in testing.
    """
    attacks_obj = attacks_factory([AttackAlgorithms.MELEE_ATTACK])
    assert attacks_obj.do_attack(targets) == {}
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[0],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[1],
                ComponentType.HEALTH,
            ),
        ).value
        == 50
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[2],
                ComponentType.HEALTH,
            ),
        ).value
        == 50
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[3],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[4],
                ComponentType.HEALTH,
            ),
        ).value
        == 50
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[5],
                ComponentType.HEALTH,
            ),
        ).value
        == 50
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[6],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )
    assert (
        cast(
            Health,
            attacks_obj.system.get_component_for_game_object(
                targets[7],
                ComponentType.HEALTH,
            ),
        ).value
        == 40
    )


def test_attacks_do_attack_ranged_attack(
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
) -> None:
    """Test that performing a ranged attack works correctly.

    Args:
        attacks_factory: The attacks component factory for use in testing.
    """
    attacks_obj = attacks_factory([AttackAlgorithms.RANGED_ATTACK])
    assert attacks_obj.do_attack([]) == {
        "ranged_attack": (
            Vec2d(0, 0),
            -300.0,
            pytest.approx(0),  # This is due to floating point errors
        ),
    }


def test_attacks_previous_next_attack_single(
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
) -> None:
    """Test that switching between attacks once works correctly.

    Args:
        attacks_factory: The attacks component factory for use in testing.
    """
    attacks_obj = attacks_factory(
        [
            AttackAlgorithms.AREA_OF_EFFECT_ATTACK,
            AttackAlgorithms.MELEE_ATTACK,
            AttackAlgorithms.RANGED_ATTACK,
        ],
    )
    attacks_obj.next_attack()
    assert attacks_obj.attack_state == 1
    attacks_obj.previous_attack()
    assert attacks_obj.attack_state == 0


def test_attacks_previous_attack_multiple(
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
) -> None:
    """Test that switching between attacks multiple times works correctly.

    Args:
        attacks_factory: The attacks component factory for use in testing.
    """
    attacks_obj = attacks_factory(
        [
            AttackAlgorithms.AREA_OF_EFFECT_ATTACK,
            AttackAlgorithms.MELEE_ATTACK,
            AttackAlgorithms.RANGED_ATTACK,
        ],
    )
    assert attacks_obj.attack_state == 0
    attacks_obj.next_attack()
    assert attacks_obj.attack_state == 1
    attacks_obj.next_attack()
    assert attacks_obj.attack_state == 2
    attacks_obj.next_attack()
    assert attacks_obj.attack_state == 2
    attacks_obj.previous_attack()
    assert attacks_obj.attack_state == 1
    attacks_obj.previous_attack()
    assert attacks_obj.attack_state == 0
    attacks_obj.previous_attack()
    assert attacks_obj.attack_state == 0


def test_attacks_previous_next_attack_empty_attacks(
    attacks_factory: Callable[[list[AttackAlgorithms]], Attacks],
) -> None:
    """Test that changing the attack state works correctly when there are no attacks.

    Args:
        attacks_factory: The attacks component factory for use in testing.
    """
    attacks_obj = attacks_factory([])
    assert attacks_obj.attack_state == 0
    attacks_obj.next_attack()
    assert attacks_obj.attack_state == -1
    attacks_obj.previous_attack()
    assert attacks_obj.attack_state == 0
