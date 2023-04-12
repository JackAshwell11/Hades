"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Custom
from hades.game_objects.enums import GameObjectData


__all__ = ()

PLAYER = GameObjectData(
    name="Player",
    component_data={

    },
    #         EntityAttributeType.HEALTH: EntityAttributeData(
    #         ),
    #         EntityAttributeType.SPEED: EntityAttributeData(
    #         ),
    #         EntityAttributeType.ARMOUR: EntityAttributeData(
    #         ),
    #         EntityAttributeType.REGEN_COOLDOWN: EntityAttributeData(
    #         ),
    #         EntityAttributeType.FIRE_RATE_PENALTY: EntityAttributeData(
    #         ),
    #         EntityAttributeType.MONEY: EntityAttributeData(
    #         ),
    #     },
    # ),
    #         EntityAttributeSectionType.ENDURANCE: lambda level: 1 * 3**level,
    #         EntityAttributeSectionType.DEFENCE: lambda level: 1 * 3**level,
    #     },
    # ),
)
