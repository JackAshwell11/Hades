from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants.entity import (
    BULLET_OFFSET,
    BULLET_VELOCITY,
    EnemyType,
    EntityID,
    PlayerType,
)
from constants.general import SPRITE_SCALE
from constants.generation import TileType
from textures import pos_to_pixel

if TYPE_CHECKING:
    from entities.player import Player
    from physics import PhysicsEngine
    from views.game import Game


class Bullet(arcade.SpriteSolidColor):
    """
    Represents a bullet in the game.

    Parameters
    ----------
    x: float
        The starting x position of the bullet.
    y: float
        The starting y position of the bullet.
    width: int
        Width of the bullet.
    height: int
        Height of the bullet.
    color: tuple[int, int, int]
        The color of the bullet.
    owner: Entity
        The entity which shot the bullet.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        color: tuple[int, int, int],
        owner: Entity,
    ) -> None:
        super().__init__(width, height, color)
        self.center_x: float = x
        self.center_y: float = y
        self.owner: Entity = owner

    def __repr__(self) -> str:
        return f"<Bullet (Position=({self.center_x}, {self.center_y}))>"


class Entity(arcade.Sprite):
    """
    Represents an entity in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the entity can have a reference to it.
    x: int
        The x position of the entity in the game map.
    y: int
        The y position of the entity in the game map.
    entity_type: PlayerType | EnemyType
        The constant data about this specific entity. This allows me to support multiple
        player types later on when I add different characters.

    Attributes
    ----------
    health: int
        The health of this entity.
    armour: int
        The armour of this entity.
    direction: float
        The angle the entity is facing.
    facing: int
        The direction the entity is facing. 0 is right and 1 is left.
    time_since_last_attack: float
        The time since the last attack.
    """

    # Class variables
    ID: EntityID = EntityID.ENTITY

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
        entity_type: PlayerType | EnemyType,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.game: Game = game
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.entity_type: PlayerType | EnemyType = entity_type
        self.texture: arcade.Texture = self.entity_type.textures["idle"][0][0]
        self.health: int = self.entity_type.health
        self.armour: int = self.entity_type.armour
        self.direction: float = 0
        self.facing: int = 0
        self.time_since_last_attack: float = 0

    def __repr__(self) -> str:
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes movement and game logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        raise NotImplementedError

    def deal_damage(self, damage: int) -> None:
        """
        Deals damage to an entity.

        Parameters
        ----------
        damage: int
            The amount of health to take away from the entity.
        """
        self.health -= damage

    def ranged_attack(self, bullet_list: arcade.SpriteList) -> None:
        """
        Performs a ranged attack in the direction the entity is facing.

        Parameters
        ----------
        bullet_list: arcade.SpriteList
            The sprite list to add the bullet to.
        """
        # Reset the time counter
        self.time_since_last_attack = 0

        # Create and add the new bullet to the physics engine
        new_bullet = Bullet(self.center_x, self.center_y, 25, 5, arcade.color.RED, self)
        new_bullet.angle = self.direction
        physics: PhysicsEngine = self.physics_engines[0]
        physics.add_bullet(new_bullet)
        bullet_list.append(new_bullet)

        # Move the bullet away from the entity a bit to stop its colliding with them
        angle_radians = self.direction * math.pi / 180
        new_x, new_y = (
            new_bullet.center_x + math.cos(angle_radians) * BULLET_OFFSET,
            new_bullet.center_y + math.sin(angle_radians) * BULLET_OFFSET,
        )
        physics.set_position(new_bullet, (new_x, new_y))

        # Calculate its velocity
        change_x, change_y = (
            math.cos(angle_radians) * BULLET_VELOCITY,
            math.sin(angle_radians) * BULLET_VELOCITY,
        )
        physics.set_velocity(new_bullet, (change_x, change_y))

    def melee_attack(self, target: list[Entity]) -> None:
        """
        Performs a melee attack in the direction the entity is facing.

        Parameters
        ----------
        target: list[Entity]
            The entities to deal damage to.
        """
        # Deal damage to every entity in the target list
        print(target)


class Tile(arcade.Sprite):
    """
    Represents a tile in the game.

    Parameters
    ----------
    x: int
        The x position of the tile in the game map.
    y: int
        The y position of the tile in the game map.

    Attributes
    ----------
    center_x: float
        The x position of the tile on the screen.
    center_y: float
        The y position of the tile on the screen.
    texture: arcade.Texture
        The sprite which represents this tile.
    """

    # Class variables
    raw_texture: arcade.Texture | None = None
    is_static: bool = False

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.texture: arcade.Texture = self.raw_texture

    def __repr__(self) -> str:
        return f"<Tile (Position=({self.center_x}, {self.center_y}))>"


class Item(Tile):
    """
    Represents an item in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the item can have a reference to it.
    x: int
        The x position of the item in the game map.
    y: int
        The y position of the item in the game map.
    """

    # Class variables
    item_id: TileType = TileType.NONE

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(x, y)
        self.game: Game = game

    def __repr__(self) -> str:
        return f"<Item (Position=({self.center_x}, {self.center_y}))>"

    @property
    def player(self) -> Player:
        """Returns the player object for ease of access."""
        return self.game.player

    def item_activate(self) -> None:
        """Called when the item is activated by the player. Override this to add item
        activate functionality."""
        return

    def item_use(self) -> None:
        """Called when the item is ued by the player. Override this to add item use
        functionality."""
        return
