"""Manages the rendering of load menu elements on the screen."""

from __future__ import annotations

# Builtin
from functools import partial
from typing import TYPE_CHECKING

# Pip
from arcade import color, get_window
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton, UILabel, UISpace
from arcade.gui.experimental import UIScrollArea
from arcade.gui.experimental.scroll_area import (  # type: ignore[attr-defined]
    UIScrollBar,
)
from dateutil.parser import parse

# Custom
from hades import UI_BACKGROUND_COLOUR, UI_PADDING, BackButton
from hades.scenes.base.view import BaseView

if TYPE_CHECKING:
    from hades.window import HadesWindow
    from hades_engine import SaveFileInfo

__all__ = (
    "LoadGameMenuView",
    "SaveEntry",
    "SaveEntryInfo",
)


class SaveEntryInfo(UIBoxLayout):
    """Represents the information of a save entry."""

    def __init__(
        self: SaveEntryInfo,
        name: str,
        last_modified: str,
        player_level: int,
    ) -> None:
        """Initialise the object.

        Args:
            name: The name of the save file.
            last_modified: The last modified date of the save file.
            player_level: The level of the player in the save file.
        """
        super().__init__(vertical=False, space_between=UI_PADDING)
        left_layout = UIBoxLayout(align="left", space_between=UI_PADDING)
        left_layout.add(UILabel(name, text_color=color.BLACK))
        left_layout.add(
            UILabel(f"Player Level: {player_level}", text_color=color.BLACK),
        )
        self.add(left_layout)
        self.add(UISpace(size_hint=(1, None)))
        self.add(
            UILabel(
                f"Last Modified: {parse(last_modified).strftime("%d %B %Y, %H:%M")}",
                text_color=color.BLACK,
            ),
        )


class SaveEntry(UIBoxLayout):
    """Represents a single save entry in the load game menu."""

    def __init__(self: SaveEntry, save_info: SaveFileInfo, save_index: int) -> None:
        """Initialise the object.

        Args:
            save_info: The save file information to display.
            save_index: The index of the save entry.
        """
        super().__init__(vertical=False, space_between=UI_PADDING)
        play_button = UIFlatButton(text="Play")
        play_button.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_load_save",
            save_index,
        )
        self.add(play_button)
        self.add(
            SaveEntryInfo(
                save_info.name,
                save_info.last_modified,
                save_info.player_level,
            ),
        )
        delete_button = UIFlatButton(text="Delete")
        delete_button.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_delete_save",
            save_index,
        )
        self.add(delete_button)
        self.with_background(color=UI_BACKGROUND_COLOUR)


class LoadGameMenuView(BaseView):
    """Manages the rendering of load menu elements on the screen.

    Attributes:
        save_layout: The layout that contains the save entries.
    """

    __slots__ = ("save_layout",)

    def _setup_layout(self: LoadGameMenuView) -> None:
        """Set up the layout for the view."""
        vertical_box = UIBoxLayout(space_between=UI_PADDING, size_hint=(1, 0.5))
        scroll_area = UIScrollArea(size_hint=(1, 1))
        scroll_area.add(self.save_layout)
        scroll_container = UIBoxLayout(vertical=False, size_hint=(0.5, 1))
        scroll_container.add(scroll_area)
        scroll_container.add(UIScrollBar(scroll_area))
        vertical_box.add(scroll_container)
        vertical_box.add(BackButton())
        self.ui.add(UIAnchorLayout(children=(vertical_box,)))

    def __init__(self: LoadGameMenuView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the game.
        """
        self.save_layout: UIBoxLayout = UIBoxLayout(space_between=UI_PADDING)
        super().__init__(window)
