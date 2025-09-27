"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

# Pip
from arcade import Texture, View, Window, get_default_texture, get_image
from arcade.gui import UIImage, UIOnClickEvent
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

if TYPE_CHECKING:
    from hades.scenes.base import BaseScene
    from hades.scenes.base.view import BaseView

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
    "on_difficulty_change",
    "on_load_game",
    "on_load_save",
    "on_new_game",
    "on_previous_view",
    "on_quit_game",
    "on_start_level",
    "on_texture_button_callback",
    "on_use_button_callback",
]


class HadesWindow(Window):
    """Manages the window and allows switching between scenes.

    Attributes:
        model: The model providing access to the game engine and its functionality.
        last_scenes: Holds the last scenes shown in the game.
        background_image: The background image of the window.
        scenes: Holds all the scenes used by the game.
    """

    __slots__ = ("background_image", "last_scenes", "model", "scenes")

    def __init__(self: HadesWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: HadesModel = HadesModel()
        self.last_scenes: list[BaseScene[BaseView]] = []
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

    def on_previous_view(self: HadesWindow, _: UIOnClickEvent) -> None:
        """Process the previous view event."""
        if self.last_scenes:
            self.show_view(self.last_scenes.pop())

    def show_view(self: HadesWindow, new_view: View) -> None:
        """Set the currently active view.

        Args:
            new_view: The view to set as the active view.
        """
        if self.current_view is not None:
            self.last_scenes.append(cast("BaseScene[BaseView]", self.current_view))
        self.background_image.texture = Texture(get_image().filter(BACKGROUND_BLUR))
        if new_view is self.scenes[SceneType.GAME]:
            self.model.save_manager.save_game()
        super().show_view(new_view)

    def setup(self: HadesWindow) -> None:
        """Set up the window and its scenes."""
        self.center_window()
        self.model.game_state.set_window_size(self.width, self.height)
        for event_type in EVENT_TYPES:
            self.register_event_type(event_type)
        self.model.game_engine.setup(str(SHOP_OFFERINGS), str(SAVE_DIRECTORY))
        self.show_view(self.scenes[SceneType.START_MENU])


def main() -> None:
    """Initialise the game and run it."""
    window = HadesWindow()
    window.setup()
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
