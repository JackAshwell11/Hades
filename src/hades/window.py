"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade import Texture, Window, get_default_texture, get_image
from arcade.gui import UIImage
from arcade.resources import resolve
from PIL.ImageFilter import GaussianBlur

# Custom
from hades import SceneType
from hades.model import HadesModel
from hades.scenes.game import GameScene
from hades.scenes.game_options import GameOptionsScene
from hades.scenes.inventory import InventoryScene
from hades.scenes.shop import ShopScene
from hades.scenes.start_menu import StartMenuScene

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ("HadesWindow", "main")

# The Gaussian blur filter to apply to the background image
BACKGROUND_BLUR: Final[GaussianBlur] = GaussianBlur(5)

# The path to the shop offerings JSON file
SHOP_OFFERINGS: Final[Path] = resolve(":resources:shop_offerings.json")

# The event types to register for the window
EVENT_TYPES: Final[list[str]] = [
    "on_texture_button_callback",
    "on_use_button_callback",
    "on_start_game",
    "on_optioned_start_game",
    "on_quit_game",
]


class HadesWindow(Window):
    """Manages the window and allows switching between scenes.

    Attributes:
        scenes: Holds all the scenes used by the game.
        background_image: The background image of the window.
        model: The model providing access to the game engine and its functionality.
    """

    __slots__ = ("background_image", "model", "scenes")

    def __init__(self: HadesWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: HadesModel = HadesModel()
        self.background_image: UIImage = UIImage(
            texture=get_default_texture(),
            width=self.width,
            height=self.height,
        )
        self.scenes: dict[
            SceneType,
            StartMenuScene | GameScene | GameOptionsScene | InventoryScene | ShopScene,
        ] = {
            SceneType.START_MENU: StartMenuScene(),
            SceneType.GAME: GameScene(),
            SceneType.GAME_OPTIONS: GameOptionsScene(),
            SceneType.INVENTORY: InventoryScene(),
            SceneType.SHOP: ShopScene(),
        }
        for event_type in EVENT_TYPES:
            self.register_event_type(event_type)

    def setup(self: HadesWindow) -> None:
        """Set up the window and its scenes."""
        self.center_window()
        self.model.game_engine.setup(str(SHOP_OFFERINGS))
        self.show_view(self.scenes[SceneType.START_MENU])

    def save_background(self: HadesWindow) -> None:
        """Save the current background image to a texture."""
        self.background_image.texture = Texture(get_image().filter(BACKGROUND_BLUR))


def main() -> None:
    """Initialise the game and run it."""
    window = HadesWindow()
    window.setup()
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
