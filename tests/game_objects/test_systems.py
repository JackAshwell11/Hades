"""Tests all functions in game_objects/systems.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest

from hades.game_objects.base import ComponentType
from hades.game_objects.components import (
    ArmourRegen,
    Footprint,
    InstantEffects,
    Inventory,
    InventorySpaceError,
    StatusEffects,
)
from hades.game_objects.registry import ECS
from hades.game_objects.steering import Vec2d

# Custom
from hades.game_objects.systems import Armour, ArmourRegenCooldown

__all__ = ()


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def armour_regen(ecs: ECS) -> ArmourRegen:
    """Create an armour regen component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The armour regen component for use in testing.
    """
    ecs.add_game_object(
        {
            "attributes": {
                ComponentType.ARMOUR: (50, 3),
                ComponentType.ARMOUR_REGEN_COOLDOWN: (4, 5),
            },
        },
        Armour,
        ArmourRegenCooldown,
        ArmourRegen,
    )
    return cast(
        ArmourRegen,
        ecs.get_component_for_game_object(0, ComponentType.ARMOUR_REGEN),
    )


@pytest.fixture()
def footprint(ecs: ECS) -> Footprint:
    """Create a footprint component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The instant effects component for use in testing.
    """
    ecs.add_game_object({}, Footprint, physics=True)
    return cast(
        Footprint,
        ecs.get_component_for_game_object(0, ComponentType.FOOTPRINT),
    )


@pytest.fixture()
def instant_effects(ecs: ECS) -> InstantEffects:
    """Create an instant effects component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The instant effects component for use in testing.
    """
    ecs.add_game_object(
        {"instant_effects": (5, {ComponentType.HEALTH: lambda level: 2**level + 5})},
        InstantEffects,
    )
    return cast(
        InstantEffects,
        ecs.get_component_for_game_object(0, ComponentType.INSTANT_EFFECTS),
    )


@pytest.fixture()
def inventory(ecs: ECS) -> Inventory[int]:
    """Create an inventory component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The inventory component for use in testing.
    """
    ecs.add_game_object({"inventory_size": (3, 6)}, Inventory)
    return cast(
        Inventory[int],
        ecs.get_component_for_game_object(0, ComponentType.INVENTORY),
    )


@pytest.fixture()
def status_effects(ecs: ECS) -> StatusEffects:
    """Create a status effects component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The status effects component for use in testing.
    """
    ecs.add_game_object(
        {
            "status_effects": (
                3,
                {
                    ComponentType.ARMOUR: (
                        lambda level: 3**level + 10,
                        lambda level: 3 * level + 5,
                    ),
                },
            ),
        },
        StatusEffects,
    )
    return cast(
        StatusEffects,
        ecs.get_component_for_game_object(0, ComponentType.STATUS_EFFECTS),
    )


def test_raise_inventory_space_error_full() -> None:
    """Test that InventorySpaceError is raised correctly when full."""
    with pytest.raises(
        expected_exception=InventorySpaceError,
        match="The inventory is full.",
    ):
        raise InventorySpaceError(full=True)


def test_raise_inventory_space_error_empty() -> None:
    """Test that InventorySpaceError is raised correctly when empty."""
    with pytest.raises(
        expected_exception=InventorySpaceError,
        match="The inventory is empty.",
    ):
        raise InventorySpaceError(full=False)


def test_armour_regen_init(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is initialised correctly.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    assert repr(armour_regen) == "<ArmourRegen (Time since armour regen=0)>"


def test_armour_regen_on_update_small_deltatime(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated with a small deltatime.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.armour.value -= 10
    armour_regen.on_update(2)
    assert armour_regen.armour.value == 40
    assert armour_regen.time_since_armour_regen == 2


def test_armour_regen_on_update_large_deltatime(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated with a large deltatime.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.armour.value -= 10
    armour_regen.on_update(6)
    assert armour_regen.armour.value == 41
    assert armour_regen.time_since_armour_regen == 0


def test_armour_regen_on_update_multiple_updates(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated correctly multiple times.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.armour.value -= 10
    armour_regen.on_update(1)
    assert armour_regen.armour.value == 40
    assert armour_regen.time_since_armour_regen == 1
    armour_regen.on_update(2)
    assert armour_regen.armour.value == 40
    assert armour_regen.time_since_armour_regen == 3


def test_armour_regen_on_update_full_armour(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated even when armour is already full.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.on_update(5)
    assert armour_regen.armour.value == 50
    assert armour_regen.time_since_armour_regen == 0


def test_footprint_init(footprint: Footprint) -> None:
    """Test that the footprint component is initialised correctly."""
    assert (
        repr(footprint)
        == "<Footprint (Footprint count=0) (Time since last footprint=0)>"
    )


def test_footprint_on_update_small_deltatime(footprint: Footprint) -> None:
    """Test that the footprint component is updated with a small deltatime.

    Args:
        footprint: The footprint component for use in testing.
    """
    footprint.on_update(0.1)
    assert footprint.footprints == []
    assert footprint.time_since_last_footprint == 0.1


def test_footprint_on_update_large_deltatime_empty_list(footprint: Footprint) -> None:
    """Test that the footprint component creates a footprint in an empty list.

    Args:
        footprint: The footprint component for use in testing.
    """
    footprint.on_update(1)
    assert footprint.footprints == [Vec2d(0, 0)]
    assert footprint.time_since_last_footprint == 0


def test_footprint_on_update_large_deltatime_non_empty_list(
    footprint: Footprint,
) -> None:
    """Test that the footprint component creates a footprint in a non-empty list.

    Args:
        footprint: The footprint component for use in testing.
    """
    footprint.footprints = [Vec2d(1, 1), Vec2d(2, 2), Vec2d(3, 3)]
    footprint.on_update(0.5)
    assert footprint.footprints == [Vec2d(1, 1), Vec2d(2, 2), Vec2d(3, 3), Vec2d(0, 0)]


def test_footprint_on_update_large_deltatime_full_list(footprint: Footprint) -> None:
    """Test that the footprint component creates a footprint and removes the oldest one.

    Args:
        footprint: The footprint component for use in testing.
    """
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
    footprint.system.get_physics_object_for_game_object(0).position = Vec2d(10, 10)
    footprint.on_update(0.5)
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


def test_footprint_on_update_multiple_updates(footprint: Footprint) -> None:
    """Test that the footprint component is updated correctly multiple times.

    Args:
        footprint: The footprint component for use in testing.
    """
    footprint.on_update(0.6)
    assert footprint.footprints == [Vec2d(0, 0)]
    assert footprint.time_since_last_footprint == 0
    footprint.system.get_physics_object_for_game_object(0).position = Vec2d(1, 1)
    footprint.on_update(0.7)
    assert footprint.footprints == [Vec2d(0, 0), Vec2d(1, 1)]
    assert footprint.time_since_last_footprint == 0


def test_instant_effects_init(instant_effects: InstantEffects) -> None:
    """Test that the instant effects component is initialised correctly.

    Args:
        instant_effects: The instant effects component for use in testing.
    """
    assert repr(instant_effects) == "<InstantEffects (Level limit=5)>"


def test_inventory_init(inventory: Inventory[int]) -> None:
    """Test that the inventory component is initialised correctly.

    Args:
        inventory: The inventory component for use in testing.
    """
    assert repr(inventory) == "<Inventory (Width=3) (Height=6)>"
    assert not inventory.inventory


def test_inventory_add_item_to_inventory_valid(inventory: Inventory[int]) -> None:
    """Test that a valid item is added to the inventory correctly.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.add_item_to_inventory(50)
    assert inventory.inventory == [50]


def test_inventory_add_item_to_inventory_zero_size(inventory: Inventory[int]) -> None:
    """Test that a valid item is not added to a zero size inventory.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.width = 0
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory.add_item_to_inventory(5)


def test_inventory_remove_item_from_inventory_valid(inventory: Inventory[int]) -> None:
    """Test that a valid item is removed from the inventory correctly.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.add_item_to_inventory(1)
    inventory.add_item_to_inventory(7)
    inventory.add_item_to_inventory(4)
    assert inventory.remove_item_from_inventory(1) == 7
    assert inventory.inventory == [1, 4]


def test_inventory_remove_item_from_inventory_large_index(
    inventory: Inventory[int],
) -> None:
    """Test that an exception is raised if a larger index is provided.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.add_item_to_inventory(5)
    inventory.add_item_to_inventory(10)
    inventory.add_item_to_inventory(50)
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory.remove_item_from_inventory(10)


def test_status_effects_init(status_effects: StatusEffects) -> None:
    """Test that the status effects component is initialised correctly.

    Args:
        status_effects: The status effects component for use in testing.
    """
    assert repr(status_effects) == "<StatusEffects (Level limit=3)>"


"""Tests all functions in game_objects/attacks.py."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
import pytest

# Custom
from hades.game_objects.attacks import Attacks
from hades.game_objects.base import AttackAlgorithms, ComponentType
from hades.game_objects.registry import ECS
from hades.game_objects.steering import Vec2d
from hades.game_objects.systems import Armour, Health

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


"""Tests all functions in game_objects/systems.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest

from hades.game_objects.base import ComponentType
from hades.game_objects.registry import ECS

# Custom
from hades.game_objects.systems import (
    GameObjectAttributeBase,
    GameObjectAttributeError,
    deal_damage,
)

__all__ = ()


class FullGameObjectAttribute(GameObjectAttributeBase):
    """Represents a full game object attribute useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.HEALTH


class EmptyGameObjectAttribute(GameObjectAttributeBase):
    """Represents an empty game object attribute useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR
    instant_effect: bool = False
    maximum: bool = False
    status_effect: bool = False
    upgradable: bool = False


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def full_game_object_attribute(ecs: ECS) -> FullGameObjectAttribute:
    """Create a full game object attribute for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The full game object attribute for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.HEALTH: (150, 3)}},
        FullGameObjectAttribute,
    )
    return cast(
        FullGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.HEALTH),
    )


@pytest.fixture()
def empty_game_object_attribute(ecs: ECS) -> EmptyGameObjectAttribute:
    """Create an empty game object attribute for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The empty game object attribute for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.ARMOUR: (100, 5)}},
        EmptyGameObjectAttribute,
    )
    return cast(
        EmptyGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.ARMOUR),
    )


@pytest.fixture()
def both_game_object_attributes(
    ecs: ECS,
) -> tuple[FullGameObjectAttribute, EmptyGameObjectAttribute]:
    """Create two game object attributes for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        Both game object attributes for use in testing.
    """
    ecs.add_game_object(
        {
            "attributes": {
                ComponentType.HEALTH: (200, 4),
                ComponentType.ARMOUR: (150, 6),
            },
        },
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    )
    return cast(
        FullGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.HEALTH),
    ), cast(
        EmptyGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.ARMOUR),
    )


def test_raise_game_object_attribute_error() -> None:
    """Test that GameObjectAttributeError is raised correctly."""
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match="The game object attribute `test` cannot be upgraded.",
    ):
        raise GameObjectAttributeError(name="test", error="be upgraded")


def test_full_game_object_attribute_init(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is initialised correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert (
        repr(full_game_object_attribute)
        == "<FullGameObjectAttribute (Value=150) (Max value=150) (Level=0/3)>"
    )


def test_empty_game_object_attribute_init(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an empty game object attribute is initialised correctly.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    assert (
        repr(empty_game_object_attribute)
        == "<EmptyGameObjectAttribute (Value=100) (Max value=inf) (Level=0/5)>"
    )


def test_full_game_object_attribute_setter_lower(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is set with a lower value correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value = 100
    assert full_game_object_attribute.value == 100


def test_full_game_object_attribute_setter_higher(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is set with a higher value correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value = 200
    assert full_game_object_attribute.value == 150


def test_full_game_object_attribute_setter_isub(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that subtracting a value from the full game object attribute is correct.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 200
    assert full_game_object_attribute.value == 0


def test_full_game_object_attribute_setter_iadd(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that adding a value to the full game object attribute is correct.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value += 100
    assert full_game_object_attribute.value == 150


def test_full_game_object_attribute_upgrade_value_equal_max(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is upgraded correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert full_game_object_attribute.upgrade(lambda level: 150 * (level + 1))
    assert full_game_object_attribute.value == 300
    assert full_game_object_attribute.max_value == 300
    assert full_game_object_attribute.current_level == 1


def test_full_game_object_attribute_upgrade_value_lower_max(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is upgraded if value is lower than max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 50
    assert full_game_object_attribute.upgrade(lambda level: 150 + 2 ^ level)
    assert full_game_object_attribute.value == 101
    assert full_game_object_attribute.max_value == 151
    assert full_game_object_attribute.current_level == 1


def test_full_game_object_attribute_upgrade_max_limit(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is not upgraded if level limit is reached.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.upgrade(lambda _: 0)
    full_game_object_attribute.upgrade(lambda _: 0)
    full_game_object_attribute.upgrade(lambda _: 0)
    assert not full_game_object_attribute.upgrade(lambda _: 0)


def test_full_game_object_attribute_upgrade_invalid_increase(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is not upgraded given an invalid increase.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    with pytest.raises(expected_exception=TypeError):
        full_game_object_attribute.upgrade(
            lambda _: "str",  # type: ignore[arg-type,return-value]
        )


def test_full_game_object_attribute_apply_instant_effect_lower(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that an instant effect is applied if the value is lower than the max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 50
    assert full_game_object_attribute.apply_instant_effect(lambda level: 10 * level, 2)
    assert full_game_object_attribute.value == 120


def test_full_game_object_attribute_apply_instant_effect_equal(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that an instant effect is not applied if the value is equal to the max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert not full_game_object_attribute.apply_instant_effect(lambda _: 50, 3)
    assert full_game_object_attribute.value == 150


def test_full_game_object_attribute_apply_status_effect_no_applied_effect(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is applied if no status effect is currently applied.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert full_game_object_attribute.apply_status_effect(
        lambda level: 150 + 3**level,
        lambda level: 20 + 10 * level,
        2,
    )
    assert (
        repr(full_game_object_attribute.applied_status_effect)
        == "StatusEffect(value=159, duration=40, original_value=150,"
        " original_max_value=150, time_counter=0)"
    )


def test_full_game_object_attribute_apply_status_effect_value_lower_max(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is applied if the value is lower than the max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 20
    assert full_game_object_attribute.apply_status_effect(
        lambda level: 20 * level,
        lambda level: 10 - 2**level,
        3,
    )
    assert (
        repr(full_game_object_attribute.applied_status_effect)
        == "StatusEffect(value=60, duration=2, original_value=130,"
        " original_max_value=150, time_counter=0)"
    )


def test_full_game_object_attribute_apply_status_effect_existing_status_effect(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is not applied if a status effect is already applied.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(lambda _: 50, lambda _: 20, 3)
    assert not full_game_object_attribute.apply_status_effect(
        lambda _: 60,
        lambda _: 30,
        2,
    )


def test_full_game_object_attribute_on_update_no_deltatime(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a game object is updated when no time has passed.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(
        lambda level: level * 2,
        lambda level: level + 100,
        2,
    )
    full_game_object_attribute.on_update(0)
    assert full_game_object_attribute.applied_status_effect
    assert full_game_object_attribute.applied_status_effect.time_counter == 0


def test_full_game_object_attribute_on_update_larger_deltatime(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is removed if deltatime is larger than the duration.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(
        lambda level: 2**level,
        lambda _: 20,
        2,
    )
    full_game_object_attribute.on_update(30)
    assert full_game_object_attribute.value == 150
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_full_game_object_attribute_on_update_multiple_deltatimes(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is removed after multiple updates.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    # Apply the status effect and check the attribute's state after one update
    full_game_object_attribute.apply_status_effect(
        lambda level: 50 + level / 2,
        lambda level: 3**level + 50,
        3,
    )
    full_game_object_attribute.on_update(40)
    assert full_game_object_attribute.applied_status_effect
    assert full_game_object_attribute.applied_status_effect.time_counter == 40
    assert full_game_object_attribute.value == 201.5
    assert full_game_object_attribute.max_value == 201.5

    # Check the attribute's state after the final update
    full_game_object_attribute.on_update(40)
    assert full_game_object_attribute.value == 150
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_full_game_object_attribute_on_update_less_than_original(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is removed when value is less than the original.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(lambda _: 100, lambda _: 20, 4)
    full_game_object_attribute.value -= 200
    full_game_object_attribute.on_update(30)
    assert full_game_object_attribute.value == 50
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_full_game_object_attribute_on_update_no_status_effect(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a game object is updated if a status effect does not exist.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert not full_game_object_attribute.applied_status_effect
    full_game_object_attribute.on_update(5)
    assert not full_game_object_attribute.applied_status_effect


def test_empty_game_object_attribute_setter(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an empty game object attribute's value can be changed.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    empty_game_object_attribute.value -= 10
    assert empty_game_object_attribute.value == 90


def test_empty_game_object_attribute_upgrade(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot be upgraded."
        ),
    ):
        empty_game_object_attribute.upgrade(lambda _: 0)


def test_empty_game_object_attribute_instant_effect(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an instant effect raises an error on an empty game object attribute.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot have an"
            " instant effect."
        ),
    ):
        empty_game_object_attribute.apply_instant_effect(lambda _: 0, 5)


def test_empty_game_object_attribute_apply_status_effect(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that a status effect raises an error on an empty game object attribute.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot have a status"
            " effect."
        ),
    ):
        empty_game_object_attribute.apply_status_effect(lambda _: 0, lambda _: 0, 6)


def test_empty_game_object_attribute_on_update(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that updating a game object fails on an empty game object attribute.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    assert not empty_game_object_attribute.applied_status_effect
    empty_game_object_attribute.on_update(1)
    assert not empty_game_object_attribute.applied_status_effect


def test_deal_damage_low_health_armour(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that damage is dealt when health and armour are lower than damage.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    both_game_object_attributes[0].value = 150
    both_game_object_attributes[1].value = 100
    deal_damage(0, ecs, 200)
    assert both_game_object_attributes[0].value == 50
    assert both_game_object_attributes[1].value == 0


def test_deal_damage_large_armour(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that no damage is dealt when armour is larger than damage.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    deal_damage(0, ecs, 100)
    assert both_game_object_attributes[0].value == 200
    assert both_game_object_attributes[1].value == 50


def test_deal_damage_zero_damage(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that no damage is dealt when damage is zero.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    deal_damage(0, ecs, 0)
    assert both_game_object_attributes[0].value == 200
    assert both_game_object_attributes[1].value == 150


def test_deal_damage_zero_armour(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that damage is dealt when armour is zero.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    both_game_object_attributes[1].value = 0
    deal_damage(0, ecs, 100)
    assert both_game_object_attributes[0].value == 100
    assert both_game_object_attributes[1].value == 0


def test_deal_damage_zero_health(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that damage is dealt when health is zero.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    both_game_object_attributes[0].value = 0
    deal_damage(0, ecs, 200)
    assert both_game_object_attributes[0].value == 0
    assert both_game_object_attributes[1].value == 0


def test_deal_damage_nonexistent_attributes(ecs: ECS) -> None:
    """Test that no damage is dealt when the attributes are not initialised.

    Args:
        ecs: The entity component system for use in testing.
    """
    ecs.add_game_object({})
    with pytest.raises(expected_exception=KeyError):
        deal_damage(0, ecs, 100)


"""Tests all functions in game_objects/movements.py."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
import pytest

from hades.game_objects.base import (
    ComponentType,
    SteeringBehaviours,
    SteeringMovementState,
)
from hades.game_objects.components import Footprint
from hades.game_objects.registry import ECS
from hades.game_objects.steering import Vec2d

# Custom
from hades.game_objects.systems import MovementForce

if TYPE_CHECKING:
    from collections.abc import Callable

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
        Footprint,
        MovementForce,
        KeyboardMovement,
        physics=True,
    )
    return cast(
        KeyboardMovement,
        ecs.get_component_for_game_object(0, ComponentType.MOVEMENTS),
    )


@pytest.fixture()
def steering_movement_factory(
    keyboard_movement: KeyboardMovement,
) -> Callable[
    [dict[SteeringMovementState, list[SteeringBehaviours]]],
    SteeringMovement,
]:
    """Create a steering movement component factory for use in testing.

    Args:
        keyboard_movement: The keyboard movement component to use in testing.

    Returns:
        The steering movement component factory for use in testing.
    """

    def wrap(
        steering_behaviours: dict[SteeringMovementState, list[SteeringBehaviours]],
    ) -> SteeringMovement:
        game_object_id = keyboard_movement.system.add_game_object(
            {
                "attributes": {ComponentType.MOVEMENT_FORCE: (100, 5)},
                "steering_behaviours": steering_behaviours,
            },
            MovementForce,
            SteeringMovement,
            physics=True,
        )
        steering_movement = cast(
            SteeringMovement,
            keyboard_movement.system.get_component_for_game_object(
                game_object_id,
                ComponentType.MOVEMENTS,
            ),
        )
        steering_movement.target_id = 0
        return steering_movement

    return wrap


@pytest.fixture()
def steering_movement(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> SteeringMovement:
    """Create a steering movement component for use in testing.

    Args:
        steering_movement_factory: The steering movement component factory to use in
            testing.

    Returns:
        The steering movement component for use in testing.
    """
    return steering_movement_factory({})


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
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
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
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
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
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
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
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        500,
        500,
    )
    steering_movement.path_list = [Vec2d(300, 300), Vec2d(400, 400)]
    steering_movement.calculate_force()
    assert steering_movement.movement_state == SteeringMovementState.FOOTPRINT


def test_steering_movement_calculate_force_missing_state(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if a zero force is calculated if the state is missing.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    assert steering_movement_factory({}).calculate_force() == Vec2d(0, 0)


def test_steering_movement_calculate_force_arrive(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the arrive behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.ARRIVE]},
    )
    steering_movement.system.get_physics_object_for_game_object(0).position = Vec2d(
        0,
        0,
    )
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert steering_movement.calculate_force() == Vec2d(
        -70.71067811865476,
        -70.71067811865476,
    )


def test_steering_movement_calculate_force_evade(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the evade behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.EVADE]},
    )
    physics_object = steering_movement.system.get_physics_object_for_game_object(0)
    physics_object.position = Vec2d(100, 100)
    physics_object.velocity = Vec2d(-50, 0)
    assert steering_movement.calculate_force() == Vec2d(
        -54.28888213891886,
        -83.98045770360257,
    )


def test_steering_movement_calculate_force_flee(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the flee behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.FLEE]},
    )
    steering_movement.system.get_physics_object_for_game_object(0).position = Vec2d(
        50,
        50,
    )
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert steering_movement.calculate_force() == Vec2d(
        70.71067811865475,
        70.71067811865475,
    )


def test_steering_movement_calculate_force_follow_path(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the follow path behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.FOOTPRINT: [SteeringBehaviours.FOLLOW_PATH]},
    )
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        200,
        200,
    )
    steering_movement.path_list = [Vec2d(350, 350), Vec2d(500, 500)]
    assert steering_movement.calculate_force() == Vec2d(
        70.71067811865475,
        70.71067811865475,
    )


def test_steering_movement_calculate_force_obstacle_avoidance(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the obstacle avoidance behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.OBSTACLE_AVOIDANCE]},
    )
    physics_object = steering_movement.system.get_physics_object_for_game_object(1)
    physics_object.position = Vec2d(100, 100)
    physics_object.velocity = Vec2d(100, 100)
    steering_movement.walls = {(1, 2)}
    assert steering_movement.calculate_force() == Vec2d(
        25.881904510252056,
        -96.59258262890683,
    )


def test_steering_movement_calculate_force_pursuit(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the pursuit behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT]},
    )
    physics_object = steering_movement.system.get_physics_object_for_game_object(0)
    physics_object.position = Vec2d(100, 100)
    physics_object.velocity = Vec2d(-50, 0)
    assert steering_movement.calculate_force() == Vec2d(
        54.28888213891886,
        83.98045770360257,
    )


def test_steering_movement_calculate_force_seek(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the seek behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.SEEK]},
    )
    steering_movement.system.get_physics_object_for_game_object(0).position = Vec2d(
        50,
        50,
    )
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert steering_movement.calculate_force() == Vec2d(
        -70.71067811865475,
        -70.71067811865475,
    )


def test_steering_movement_calculate_force_wander(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated for the wander behaviour.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {SteeringMovementState.TARGET: [SteeringBehaviours.WANDER]},
    )
    steering_movement.system.get_physics_object_for_game_object(1).velocity = Vec2d(
        100,
        -100,
    )
    steering_force = steering_movement.calculate_force()
    assert steering_force != steering_movement.calculate_force()
    assert round(abs(steering_force)) == 100


def test_steering_movement_calculate_force_multiple_behaviours(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated when multiple behaviours are selected.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    steering_movement = steering_movement_factory(
        {
            SteeringMovementState.FOOTPRINT: [
                SteeringBehaviours.FOLLOW_PATH,
                SteeringBehaviours.SEEK,
            ],
        },
    )
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        300,
        300,
    )
    steering_movement.path_list = [Vec2d(100, 200), Vec2d(-100, 0)]
    assert steering_movement.calculate_force() == Vec2d(
        -81.12421851755609,
        -58.47102846637651,
    )


def test_steering_movement_calculate_force_multiple_states(
    steering_movement_factory: Callable[
        [dict[SteeringMovementState, list[SteeringBehaviours]]],
        SteeringMovement,
    ],
) -> None:
    """Test if the correct force is calculated when multiple states are initialised.

    Args:
        steering_movement_factory: The steering movement component factory for use in
            testing.
    """
    # Initialise the steering movement component with multiple states
    steering_movement = steering_movement_factory(
        {
            SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT],
            SteeringMovementState.DEFAULT: [SteeringBehaviours.SEEK],
        },
    )

    # Test the target state
    steering_movement.system.get_physics_object_for_game_object(0).velocity = Vec2d(
        -50,
        100,
    )
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        100,
        100,
    )
    assert steering_movement.calculate_force() == Vec2d(
        -97.73793955511094,
        -21.14935392681019,
    )

    # Test the default state
    steering_movement.system.get_physics_object_for_game_object(1).position = Vec2d(
        300,
        300,
    )
    assert steering_movement.calculate_force() == Vec2d(
        -70.71067811865476,
        -70.71067811865476,
    )


def test_steering_movement_update_path_list_within_distance(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the position is within the view distance.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list([Vec2d(300, 300), Vec2d(100, 100)])
    assert steering_movement.path_list == [(100, 100)]


def test_steering_movement_update_path_list_outside_distance(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the position is outside the view distance.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    steering_movement.update_path_list([Vec2d(300, 300), Vec2d(500, 500)])
    assert steering_movement.path_list == []


def test_steering_movement_update_path_list_equal_distance(
    steering_movement: SteeringMovement,
) -> None:
    """Test if the path list is updated if the position is equal to the view distance.

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
    """Test if the path list is updated if multiple footprints are within view distance.

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
