"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Custom
from hades.constants.game_object import (
    AttackAlgorithmType,
    AttackData,
    BaseData,
    ConsumableData,
    EnemyData,
    EntityAttributeData,
    EntityAttributeSectionType,
    EntityAttributeType,
    EntityData,
    InstantData,
    InstantEffectType,
    PlayerData,
    RangedAttackData,
    StatusEffectData,
    StatusEffectType,
)
from hades.constants.generation import TileType
from hades.textures import moving_textures, non_moving_textures

__all__ = (
    "CONSUMABLES",
    "PLAYERS",
)

# Player characters
PLAYER = BaseData(
    entity_data=EntityData(
        name="player",
        textures=moving_textures["player"],
        armour_regen=True,
        level_limit=5,
        attribute_data={
            EntityAttributeType.HEALTH: EntityAttributeData(
                increase=lambda current_level: 100 * 1.4**current_level,
                status_effect=True,
                variable=True,
            ),
            EntityAttributeType.SPEED: EntityAttributeData(
                increase=lambda current_level: 150 * 1.4**current_level,
                status_effect=True,
            ),
            EntityAttributeType.ARMOUR: EntityAttributeData(
                increase=lambda current_level: 20 * 1.4**current_level,
                status_effect=True,
                variable=True,
            ),
            EntityAttributeType.REGEN_COOLDOWN: EntityAttributeData(
                increase=lambda current_level: 2 * 0.5**current_level,
                status_effect=True,
            ),
            EntityAttributeType.FIRE_RATE_PENALTY: EntityAttributeData(
                increase=lambda current_level: 1 * 0.9**current_level,
                status_effect=True,
            ),
            EntityAttributeType.MONEY: EntityAttributeData(
                increase=lambda _: 0,
                maximum=False,
                variable=True,
            ),
        },
    ),
    attacks={
        AttackAlgorithmType.RANGED: AttackData(
            damage=10,
            attack_cooldown=3,
            attack_range=0,
            extra=RangedAttackData(max_bullet_range=10),
        ),
        AttackAlgorithmType.MELEE: AttackData(
            damage=10, attack_cooldown=1, attack_range=3
        ),
        AttackAlgorithmType.AREA_OF_EFFECT: AttackData(
            damage=10, attack_cooldown=10, attack_range=3
        ),
    },
    player_data=PlayerData(
        melee_degree=60,
        section_upgrade_data={
            EntityAttributeSectionType.ENDURANCE: lambda current_level: 1
            * 3**current_level,
            EntityAttributeSectionType.DEFENCE: lambda current_level: 1
            * 3**current_level,
        },
    ),
)

# Enemy characters
ENEMY1 = BaseData(
    entity_data=EntityData(
        name="enemy1",
        textures=moving_textures["enemy"],
        armour_regen=True,
        level_limit=5,
        attribute_data={
            EntityAttributeType.HEALTH: EntityAttributeData(
                increase=lambda current_level: 10 * 1.4**current_level,
                status_effect=True,
                variable=True,
            ),
            EntityAttributeType.SPEED: EntityAttributeData(
                increase=lambda current_level: 50 * 1.4**current_level,
                status_effect=True,
            ),
            EntityAttributeType.ARMOUR: EntityAttributeData(
                increase=lambda current_level: 10 * 1.4**current_level,
                status_effect=True,
                variable=True,
            ),
            EntityAttributeType.REGEN_COOLDOWN: EntityAttributeData(
                increase=lambda current_level: 3 * 0.6**current_level,
                status_effect=True,
            ),
            EntityAttributeType.FIRE_RATE_PENALTY: EntityAttributeData(
                increase=lambda current_level: 1 * 0.95**current_level,
                status_effect=True,
            ),
        },
    ),
    attacks={
        AttackAlgorithmType.RANGED: AttackData(
            damage=5,
            attack_cooldown=5,
            attack_range=5,
            extra=RangedAttackData(max_bullet_range=10),
        )
    },
    enemy_data=EnemyData(view_distance=5),
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
    status_effects=[],
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
    status_effects=[],
)

# Base status effect consumables
HEALTH_BOOST_POTION = ConsumableData(
    name="health boost potion",
    texture=non_moving_textures["items"][2],
    level_limit=5,
    instant=[],
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
    instant=[],
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
    instant=[],
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.SPEED,
            increase=lambda current_level: 25 * 1.3**current_level,
            duration=lambda current_level: 2 * 1.3**current_level,
        )
    ],
)

FIRE_RATE_BOOST_POTION = ConsumableData(
    name="fire rate boost potion",
    texture=non_moving_textures["items"][5],
    level_limit=5,
    instant=[],
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.FIRE_RATE,
            increase=lambda current_level: -0.05 * 1.1**current_level,
            duration=lambda current_level: 2 * 1.3**current_level,
        )
    ],
)


# Constructor mappings
PLAYERS = {TileType.PLAYER: PLAYER}
CONSUMABLES = {
    TileType.HEALTH_POTION: HEALTH_POTION,
    TileType.ARMOUR_POTION: ARMOUR_POTION,
    TileType.HEALTH_BOOST_POTION: HEALTH_BOOST_POTION,
    TileType.ARMOUR_BOOST_POTION: ARMOUR_POTION,
    TileType.SPEED_BOOST_POTION: SPEED_BOOST_POTION,
    TileType.FIRE_RATE_BOOST_POTION: FIRE_RATE_BOOST_POTION,
}
