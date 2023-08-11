"""Tests all functions in game_objects/attacks.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest

# Custom
from hades.game_objects.attacks import Attacks
from hades.game_objects.base import AttackAlgorithms, ComponentType
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
def attacks(ecs: ECS) -> Attacks:
    """Create an attacks component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The attacks component for use in testing.
    """
    ecs.add_game_object(
        {
            "enabled_attacks": [
                AttackAlgorithms.AREA_OF_EFFECT_ATTACK,
                AttackAlgorithms.MELEE_ATTACK,
                AttackAlgorithms.RANGED_ATTACK,
            ],
        },
        Attacks,
    )
    return cast(
        Attacks,
        ecs.get_component_for_game_object(0, ComponentType.ATTACKS),
    )


def test_attacks_init(attacks: Attacks) -> None:
    """Test that the attacks component is initialised correctly.

    Args:
        attacks: The attacks component for use in testing.
    """
    assert repr(attacks) == "<Attacks (Attack algorithm count=3)>"
    assert attacks.attacks == [area_of_effect_attack, melee_attack, ranged_attack]


def test_attacks_do_attack(attacks: Attacks) -> None:
    """Test that performing an attack works correctly.

    Args:
        attacks: The attacks component for use in testing.
    """
    with pytest.raises(expected_exception=NotImplementedError):
        attacks.do_attack()


def test_attacks_previous_next_attack_single(attacks: Attacks) -> None:
    """Test that switching between attacks once works correctly.

    Args:
        attacks: The attacks component for use in testing.
    """
    attacks.next_attack()
    assert attacks.current_attack == 1
    attacks.previous_attack()
    assert attacks.current_attack == 0


def test_attacks_previous_attack_multiple(attacks: Attacks) -> None:
    """Test that switching between attacks multiple times works correctly.

    Args:
        attacks: The attacks component for use in testing.
    """
    attacks.next_attack()
    attacks.next_attack()
    attacks.next_attack()
    assert attacks.current_attack == 2
    attacks.previous_attack()
    attacks.previous_attack()
    attacks.previous_attack()
    assert attacks.current_attack == 0
