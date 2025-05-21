"""Contains the functionality that manages the player menu."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import View

# Custom
from hades.player.controller import PlayerController
from hades.player.model import PlayerModel
from hades.player.view import PlayerView

if TYPE_CHECKING:
    from hades_extensions.ecs import Registry

__all__ = ("Player",)


class Player(View):
    """Manages the game and its events.

    Attributes:
        model: The model managing the player state.
        view: The renderer for the player.
        controller: The controller managing the player logic.
    """

    __slots__ = ("controller", "model", "view")

    def __init__(self: Player) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: PlayerModel = PlayerModel()
        self.view: PlayerView = PlayerView(self.window)
        self.controller: PlayerController = PlayerController(self.model, self.view)

    def setup(self: Player, registry: Registry, player_id: int) -> None:
        """Set up the player view.

        Args:
            registry: The registry which manages the game objects, components, and
                systems.
            player_id: The ID of the player game object.
        """
        self.model.setup(registry, player_id)
        self.view.setup()
        self.controller.setup()

    def on_show_view(self: Player) -> None:
        """Process show view functionality."""
        self.controller.show_view()

    def on_hide_view(self: Player) -> None:
        """Process hide view functionality."""
        self.controller.hide_view()

    def on_draw(self: Player) -> None:
        """Render the screen."""
        self.view.draw()
