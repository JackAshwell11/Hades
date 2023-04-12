"""Manages the different attack algorithms available."""
from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING

# Pip
import arcade

if TYPE_CHECKING:
    from hades.game_objects.enums import AttackerData
    from hades.game_objects.objects import GameObject


__all__ = (
    "Attacker",
    "RangedAttackMixin",
    "MeleeAttackMixin",
    "AreaOfEffectAttackMixin",
)


class Bullet(arcade.SpriteSolidColor):
    """Represents a bullet in the game.

    Attributes
    ----------
    start_position: tuple[float, float]
        The starting position of the bullet. This is used to kill the bullet after a
        certain amount of tiles if it hasn't hit anything.
    """

    def __init__(
        self: Bullet,
        width: int,
        height: int,
        color: tuple[int, int, int],
        max_range: float,
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        width: int
            Width of the bullet.
        height: int
            Height of the bullet.
        color: tuple[int, int, int]
            The color of the bullet.
        max_range: float
            The max range of the bullet.
        """
        super().__init__(width=width, height=height, color=color)
        self.max_range: float = max_range
        self.start_position: tuple[float, float] = self.center_x, self.center_y

    def on_update(self: Bullet, _: float = 1 / 60) -> None:
        """Process bullet logic."""
        # Check if the bullet is pass the max range
        if math.dist(self.position, self.start_position) >= self.max_range:
            self.remove_from_sprite_lists()

    def __repr__(self: Bullet) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Bullet (Position={self.position})>"


class Attacker:
    """Allows a game object to attack another game object."""

    __slots__ = ("attacker_data", "time_since_last_attack", "time_out_of_combat")

    def __init__(self: type[Attacker], attacker_data: AttackerData) -> None:
        """Initialise the object.

        Parameters
        ----------
        attacker_data: AttackerData
            The data for the attacker component.
        """
        self.attacker_data: AttackerData = attacker_data


class RangedAttackMixin:
    """Performs a ranged attack in the direction the entity is facing."""

    def do_ranged_attack(self: type[GameObject]) -> None:
        """Perform a ranged attack in the direction the entity is facing."""
        raise NotImplementedError


class MeleeAttackMixin:
    """Performs a melee attack dealing damage to any entity in front of the owner."""

    def do_melee_attack(self: type[GameObject]) -> None:
        """Perform a melee attack in the direction the entity is facing."""
        raise NotImplementedError


class AreaOfEffectAttackMixin:
    """Performs an area of effect attack around the entity."""

    def do_area_of_effect_attack(self: type[GameObject]) -> None:
        """Perform an area of effect attack around the entity."""
        raise NotImplementedError
