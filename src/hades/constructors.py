"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Pip
from arcade import load_texture

# Custom
from hades_extensions.game_objects import (
    SPRITE_SCALE,
    AttackAlgorithm,
    GameObjectType,
    SteeringBehaviours,
    SteeringMovementState,
    Vec2d,
)
from hades_extensions.game_objects.components import (
    Armour,
    Attack,
    EffectApplier,
    Footprints,
    Health,
    Inventory,
    KeyboardMovement,
    KinematicComponent,
    MovementForce,
    StatusEffect,
    SteeringMovement,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades_extensions.game_objects import ComponentBase

__all__ = ("GameObjectConstructor", "game_object_constructors")


@dataclass()
class GameObjectConstructor:
    """Represents a constructor for a game object.

    Args:
        name: The game object's name.
        description: The game object's description.
        game_object_type: The game object's type.
        texture_paths: The paths to the game object's textures.
        components: The game object's components.
        kinematic: Whether the game object can move or not.
        static: Whether the game object blocks movement or not.
    """

    name: str
    description: str
    game_object_type: GameObjectType
    texture_paths: list[str]
    components: list[ComponentBase] = field(default_factory=list)
    kinematic: bool = False
    static: bool = False

    def __post_init__(self: GameObjectConstructor) -> None:
        """Post-initialise the object."""
        if self.kinematic:
            self.components.append(
                KinematicComponent(
                    [
                        Vec2d(*hit_box_point) * SPRITE_SCALE
                        for hit_box_point in load_texture(
                            self.texture_paths[0],
                        ).hit_box_points
                    ],
                ),
            )
        elif self.static:
            self.components.append(KinematicComponent(is_static=True))


def wall_factory() -> GameObjectConstructor:
    """Create a wall game object constructor."""
    return GameObjectConstructor(
        "Wall",
        "A wall that blocks movement.",
        GameObjectType.Wall,
        [":resources:wall.png"],
        static=True,
    )


def floor_factory() -> GameObjectConstructor:
    """Create a floor game object constructor."""
    return GameObjectConstructor(
        "Floor",
        "A floor that allows movement.",
        GameObjectType.Floor,
        [":resources:floor.png"],
    )


def player_factory() -> GameObjectConstructor:
    """Create a player game object constructor."""
    return GameObjectConstructor(
        "Player",
        "The player character.",
        GameObjectType.Player,
        [":resources:player_idle.png"],
        [
            Health(200, 5),
            Armour(100, 5),
            Inventory(6, 5),
            Attack(
                [
                    AttackAlgorithm.Ranged,
                    AttackAlgorithm.Melee,
                    AttackAlgorithm.AreaOfEffect,
                ],
            ),
            MovementForce(5000, 5),
            KeyboardMovement(),
            Footprints(),
            StatusEffect(),
        ],
        kinematic=True,
    )


def enemy_factory() -> GameObjectConstructor:
    """Create an enemy game object constructor."""
    return GameObjectConstructor(
        "Enemy",
        "An enemy character.",
        GameObjectType.Enemy,
        [":resources:enemy_idle.png"],
        [
            Health(100, 5),
            Armour(50, 5),
            Attack([AttackAlgorithm.Ranged]),
            MovementForce(1000, 5),
            SteeringMovement(
                {
                    SteeringMovementState.Default: [
                        SteeringBehaviours.ObstacleAvoidance,
                        SteeringBehaviours.Wander,
                    ],
                    SteeringMovementState.Footprint: [
                        SteeringBehaviours.FollowPath,
                    ],
                    SteeringMovementState.Target: [SteeringBehaviours.Pursue],
                },
            ),
        ],
        kinematic=True,
    )


def health_potion_factory() -> GameObjectConstructor:
    """Create a health potion game object constructor."""
    return GameObjectConstructor(
        "Health Potion",
        "A potion that restores health.",
        GameObjectType.Potion,
        [":resources:health_potion.png"],
        [
            EffectApplier(
                {
                    Health: lambda level: 2**level + 10,
                },
                {},
            ),
        ],
    )


def bullet_factory() -> GameObjectConstructor:
    """Create a bullet game object constructor."""
    return GameObjectConstructor(
        "Bullet",
        "A bullet that damages other game objects.",
        GameObjectType.Bullet,
        [":resources:bullet.png"],
    )


game_object_constructors: dict[GameObjectType, Callable[[], GameObjectConstructor]] = {
    GameObjectType.Bullet: bullet_factory,
    GameObjectType.Enemy: enemy_factory,
    GameObjectType.Floor: floor_factory,
    GameObjectType.Player: player_factory,
    GameObjectType.Potion: health_potion_factory,
    GameObjectType.Wall: wall_factory,
}

# TODO: Attack should be modified to accept multiple categories (e.g. ranged,
#  close, special), so `Default` will change
