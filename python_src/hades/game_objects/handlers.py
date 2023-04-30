"""Stores all the event handlers which glue the different components together."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.game_objects.events import add_event_handler

if TYPE_CHECKING:
    from hades.game_objects.attributes import Armour, Health

__all__ = ()


@add_event_handler()
def on_damage(damage: int, health_obj: Health, armour_obj: Armour) -> None:
    """Handle a damage event for a game object.

    Args:
        damage: The damage to deal to the game object.
        health_obj: The health component for the game object.
        armour_obj: The armour component for the game object.
    """
    # Damage the armour and carry over the extra damage to the health
    health_obj.value -= max(damage - armour_obj.value, 0)
    armour_obj.value -= damage
