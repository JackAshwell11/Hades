"""Stores all the constructors used to make the entities and consumables."""
from __future__ import annotations

# Custom
from game.constants.consumable import (
    ConsumableData,
    InstantData,
    InstantEffectType,
    StatusEffectData,
    StatusEffectType,
)
from game.constants.entity import (
    AreaOfEffectAttackData,
    AttributeUpgradeData,
    BaseData,
    EnemyData,
    EntityData,
    EntityUpgradeData,
    MeleeAttackData,
    PlayerData,
    RangedAttackData,
    UpgradeAttribute,
    UpgradeSection,
)
from game.textures import moving_textures, non_moving_textures

__all__ = (
    "PLAYER",
    "ENEMY1",
    "HEALTH_POTION",
    "ARMOUR_POTION",
    "HEALTH_BOOST_POTION",
    "ARMOUR_BOOST_POTION",
    "SPEED_BOOST_POTION",
    "FIRE_RATE_BOOST_POTION",
)

# Player characters
PLAYER = BaseData(
    entity_data=EntityData(
        name="player",
        textures=moving_textures["player"],
        armour_regen=True,
        upgrade_level_limit=5,
        upgrade_data=[
            EntityUpgradeData(
                section_type=UpgradeSection.ENDURANCE,
                cost=lambda current_level: 1 * 3**current_level,
                upgrades=[
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.HEALTH,
                        increase=lambda current_level: 100 * 1.4**current_level,
                    ),
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.SPEED,
                        increase=lambda current_level: 150 * 1.4**current_level,
                    ),
                ],
            ),
            EntityUpgradeData(
                section_type=UpgradeSection.DEFENCE,
                cost=lambda current_level: 1 * 3**current_level,
                upgrades=[
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.ARMOUR,
                        increase=lambda current_level: 20 * 1.4**current_level,
                    ),
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.REGEN_COOLDOWN,
                        increase=lambda current_level: 2 * 0.5**current_level,
                    ),
                ],
            ),
        ],
    ),
    player_data=PlayerData(
        melee_degree=60,
    ),
    ranged_attack_data=RangedAttackData(
        damage=10, attack_cooldown=3, attack_range=0, max_range=10
    ),
    melee_attack_data=MeleeAttackData(damage=10, attack_cooldown=1, attack_range=3),
    area_of_effect_attack_data=AreaOfEffectAttackData(
        damage=10, attack_cooldown=10, attack_range=3
    ),
)

# Enemy characters
ENEMY1 = BaseData(
    entity_data=EntityData(
        name="enemy1",
        textures=moving_textures["enemy"],
        armour_regen=True,
        upgrade_level_limit=5,
        upgrade_data=[
            EntityUpgradeData(
                section_type=UpgradeSection.ENDURANCE,
                cost=lambda current_level: -1,
                upgrades=[
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.HEALTH,
                        increase=lambda current_level: 10 * 1.4**current_level,
                    ),
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.SPEED,
                        increase=lambda current_level: 50 * 1.4**current_level,
                    ),
                ],
            ),
            EntityUpgradeData(
                section_type=UpgradeSection.DEFENCE,
                cost=lambda current_level: -1,
                upgrades=[
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.ARMOUR,
                        increase=lambda current_level: 10 * 1.4**current_level,
                    ),
                    AttributeUpgradeData(
                        attribute_type=UpgradeAttribute.REGEN_COOLDOWN,
                        increase=lambda current_level: 3 * 0.6**current_level,
                    ),
                ],
            ),
        ],
    ),
    enemy_data=EnemyData(view_distance=5),
    ranged_attack_data=RangedAttackData(
        damage=5, attack_cooldown=5, attack_range=5, max_range=10
    ),
)

# Base instant consumables
HEALTH_POTION = ConsumableData(
    name="health potion",
    texture=non_moving_textures["items"][0],
    level_limit=5,
    instant=[
        InstantData(
            instant_type=InstantEffectType.HEALTH,
            increase=lambda current_level: 10 * 1.5**current_level,
        ),
    ],
)

ARMOUR_POTION = ConsumableData(
    name="armour potion",
    texture=non_moving_textures["items"][1],
    level_limit=5,
    instant=[
        InstantData(
            instant_type=InstantEffectType.ARMOUR,
            increase=lambda current_level: 10 * 1.5**current_level,
        ),
    ],
)

# Base status effect consumables
HEALTH_BOOST_POTION = ConsumableData(
    name="health boost potion",
    texture=non_moving_textures["items"][2],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.HEALTH,
            increase=lambda current_level: 25 * 1.3**current_level,
            duration=lambda current_level: 5 * 1.3**current_level,
        )
    ],
)

ARMOUR_BOOST_POTION = ConsumableData(
    name="armour boost potion",
    texture=non_moving_textures["items"][3],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.ARMOUR,
            increase=lambda current_level: 10 * 1.3**current_level,
            duration=lambda current_level: 5 * 1.3**current_level,
        )
    ],
)

SPEED_BOOST_POTION = ConsumableData(
    name="speed boost potion",
    texture=non_moving_textures["items"][4],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.HEALTH,
            increase=lambda current_level: 25 * 1.3**current_level,
            duration=lambda current_level: 2 * 1.3**current_level,
        )
    ],
)

FIRE_RATE_BOOST_POTION = ConsumableData(
    name="fire rate boost potion",
    texture=non_moving_textures["items"][5],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.HEALTH,
            increase=lambda current_level: -0.5,
            duration=lambda current_level: 2 * 1.3**current_level,
        )
    ],
)
