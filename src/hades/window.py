"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from pathlib import Path
from typing import Final

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
from hades.scenes.load_game import LoadGameMenuScene
from hades.scenes.shop import ShopScene
from hades.scenes.start_menu import StartMenuScene

__all__ = ("HadesWindow", "main")

# The Gaussian blur filter to apply to the background image
BACKGROUND_BLUR: Final[GaussianBlur] = GaussianBlur(5)

# The path to the save directory
SAVE_DIRECTORY: Final[Path] = Path().home() / ".hades" / "saves"
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

# The path to the shop offerings JSON file
SHOP_OFFERINGS: Final[Path] = resolve(":resources:shop_offerings.json")

# The event types to register for the window
EVENT_TYPES: Final[list[str]] = [
    "on_delete_save",
    "on_load_game",
    "on_load_save",
    "on_new_game",
    "on_optioned_start_level",
    "on_previous_view",
    "on_quit_game",
    "on_texture_button_callback",
    "on_use_button_callback",
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
            GameScene
            | GameOptionsScene
            | InventoryScene
            | LoadGameMenuScene
            | ShopScene
            | StartMenuScene,
        ] = {
            SceneType.GAME: GameScene(),
            SceneType.GAME_OPTIONS: GameOptionsScene(),
            SceneType.INVENTORY: InventoryScene(),
            SceneType.LOAD_GAME: LoadGameMenuScene(),
            SceneType.SHOP: ShopScene(),
            SceneType.START_MENU: StartMenuScene(),
        }
        for event_type in EVENT_TYPES:
            self.register_event_type(event_type)

    def setup(self: HadesWindow) -> None:
        """Set up the window and its scenes."""
        self.center_window()
        self.model.game_engine.setup(str(SHOP_OFFERINGS), str(SAVE_DIRECTORY))
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
