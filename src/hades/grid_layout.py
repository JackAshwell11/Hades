"""Manages the rendering of player elements on the screen."""

from __future__ import annotations

# Builtin
from abc import ABC, abstractmethod
from typing import Final, cast

# Pip
from arcade import Texture, color, get_default_texture, get_window
from arcade.gui import (
    UIBoxLayout,
    UIButtonRow,
    UIFlatButton,
    UIGridLayout,
    UIImage,
    UILabel,
    UIOnActionEvent,
    UIOnClickEvent,
    UISpace,
    UITextureButton,
)
from arcade.types import Color

# Custom
from hades import UI_BACKGROUND_COLOUR, UI_PADDING
from hades_extensions.ecs import SPRITE_SIZE

__all__ = (
    "ItemButton",
    "PaginatedGridLayout",
    "StatsLayout",
    "create_default_layout",
    "create_divider_line",
)

# The height of the item button
ITEM_BUTTON_HEIGHT: Final[int] = 90

# The colour of the tab separator
TAB_SEPARATOR_COLOUR: Final[Color] = Color(128, 128, 128)


def create_divider_line(*, vertical: bool = False) -> UISpace:
    """Create a divider line.

    Args:
        vertical: Whether the divider line should be vertical or horizontal.

    Returns:
        The divider line.
    """
    return UISpace(
        width=2 if vertical else 0,
        height=2 if not vertical else 0,
        color=TAB_SEPARATOR_COLOUR,
        size_hint=(None, 1) if vertical else (1, None),
    )


def create_default_layout() -> UISpace:
    """Create a default layout for the paginated grid layout.

    Returns:
        The default layout.
    """
    return UISpace(
        color=UI_BACKGROUND_COLOUR,
        width=SPRITE_SIZE,
        height=ITEM_BUTTON_HEIGHT,
    ).with_border(color=color.BLACK)


class ItemButton(UIBoxLayout, ABC):
    """Represents an item button.

    Attributes:
        texture_button: The button for displaying the item texture.
    """

    __slots__ = ("texture_button", "use_button")

    def __init__(self: ItemButton, use_text: str) -> None:
        """Initialise the object.

        Args:
            use_text: The text to display on the use button.
        """
        super().__init__()
        self.texture_button: UITextureButton = UITextureButton(
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT * 2 // 3,
        )
        self.texture_button.on_click = (  # type: ignore[method-assign]
            self.on_texture_button_click
        )
        self.add(self.texture_button.with_border(color=color.BLACK))
        self.use_button: UIFlatButton = UIFlatButton(
            text=use_text,
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT // 3,
        )
        self.use_button.on_click = (  # type: ignore[method-assign]
            self.on_use_button_click
        )
        self.add(self.use_button.with_border(color=color.BLACK))

    @abstractmethod
    def get_info(self: ItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            The name, description, and texture of the item.
        """
        raise NotImplementedError

    @property
    def texture(self: ItemButton) -> Texture:
        """Get the texture of the item button.

        Returns:
            The texture of the item button.
        """
        return self.texture_button.texture  # type: ignore[no-any-return]

    @texture.setter
    def texture(self: ItemButton, value: Texture) -> None:
        """Set the texture of the item button.

        Args:
            value: The texture to set.
        """
        self.texture_button.texture = value
        self.texture_button.texture_hovered = value
        self.texture_button.texture_pressed = value

    @staticmethod
    def on_texture_button_click(event: UIOnClickEvent) -> None:
        """Process texture button click logic.

        Args:
            event: The event that occurred.
        """
        get_window().dispatch_event("on_texture_button_callback", event)

    @staticmethod
    def on_use_button_click(event: UIOnClickEvent) -> None:
        """Process use button click logic.

        Args:
            event: The event that occurred.
        """
        get_window().dispatch_event("on_use_button_callback", event)


class StatsLayout(UIBoxLayout):
    """Represents a layout for displaying stats about an item."""

    __slots__ = ()

    def __init__(self: StatsLayout) -> None:
        """Initialise the object."""
        super().__init__(space_between=UI_PADDING)
        self.add(UILabel("", text_color=(0, 0, 0)))
        self.add(UIImage(texture=get_default_texture()).with_border(color=color.BLACK))
        self.add(UILabel(text="", text_color=(0, 0, 0), multiline=True))
        self.with_background(color=UI_BACKGROUND_COLOUR).with_padding(all=UI_PADDING)
        self.reset()

    def set_info(
        self: StatsLayout,
        title: str,
        description: str,
        texture: Texture,
    ) -> None:
        """Set the information for the stats layout.

        Args:
            title: The title of the stats layout.
            description: The description of the stats layout.
            texture: The texture to display in the stats layout.
        """
        title_obj = cast("UILabel", self.children[0])
        title_obj.text = title
        title_obj.fit_content()
        texture_obj = cast("UIImage", self.children[1])
        texture_obj.texture = texture
        description_obj = cast("UILabel", self.children[2])
        description_obj.text = description
        description_obj.fit_content()

    def reset(self: StatsLayout) -> None:
        """Reset the stats layout to its default state."""
        self.set_info("No Item Selected", "No Item Selected", get_default_texture())


class PaginatedGridLayout[T: ItemButton](UIBoxLayout):
    """Represents a paginated grid layout for displaying items.

    Attributes:
        current_row: The current row to display at the top of the grid layout.
        grid_layout: The grid layout to display the items.
    """

    __slots__ = ("_items", "current_row", "grid_layout", "stats_layout")

    def _update_grid(self: PaginatedGridLayout[T]) -> None:
        """Update the grid adding the next row of items to display."""
        # Update the grid layout with the current items
        self.grid_layout.clear()
        start = self.current_row * self.grid_layout.column_count
        for i, item in enumerate(
            self._items[
                start : start
                + self.grid_layout.column_count * self.grid_layout.row_count
            ],
        ):
            row, column = divmod(i, self.grid_layout.column_count)
            self.grid_layout.add(item, column=column, row=row)

        # Fill the rest of the grid with default layouts
        for i in range(
            len(self._items),
            self.grid_layout.column_count * self.grid_layout.row_count,
        ):
            row, column = divmod(i, self.grid_layout.column_count)
            self.grid_layout.add(create_default_layout(), column=column, row=row)

    def __init__(self: PaginatedGridLayout[T]) -> None:
        """Initialise the object."""
        super().__init__(vertical=False, space_between=UI_PADDING * 5)
        self.current_row: int = 0
        self._items: list[T] = []
        self.grid_layout: UIGridLayout = UIGridLayout(
            horizontal_spacing=UI_PADDING,
            vertical_spacing=UI_PADDING,
            column_count=round(
                (get_window().width * 0.25) / (SPRITE_SIZE + UI_PADDING),
            ),
            row_count=round(
                (get_window().height * 0.4) / (ITEM_BUTTON_HEIGHT + UI_PADDING),
            ),
        )
        self._update_grid()
        self.add(self.grid_layout)
        vertical_layout = UIBoxLayout(space_between=UI_PADDING)
        self.stats_layout: StatsLayout = StatsLayout()
        vertical_layout.add(self.stats_layout)
        button_layout: UIButtonRow = UIButtonRow(space_between=UI_PADDING)
        button_layout.on_action = self.on_action  # type: ignore[method-assign]
        button_layout.add_button("Up")
        button_layout.add_button("Down")
        vertical_layout.add(button_layout)
        self.add(vertical_layout)
        self.with_background(color=UI_BACKGROUND_COLOUR).with_padding(
            all=UI_PADDING * 2,
        )

    @property
    def items(self: PaginatedGridLayout[T]) -> list[T]:
        """Get the items in the grid layout.

        Returns:
            The list of items in the grid layout.
        """
        return self._items

    @items.setter
    def items(self: PaginatedGridLayout[T], value: list[T]) -> None:
        """Set the items in the grid layout.

        Args:
            value: The list of items to set.
        """
        self._items = value
        self.current_row = 0
        self._update_grid()

    def add_item(self: PaginatedGridLayout[T], item: T) -> None:
        """Add an item to the grid layout.

        Args:
            item: The item to add.
        """
        self._items.append(item)
        self._update_grid()

    def on_action(self: PaginatedGridLayout[T], event: UIOnActionEvent) -> None:
        """Process paginated grid layout click logic.

        Args:
            event: The event that occurred.
        """
        if event.action == "Up":
            self.navigate_rows(-1)
        elif event.action == "Down":
            self.navigate_rows(1)

    def navigate_rows(self: PaginatedGridLayout[T], diff: int) -> None:
        """Navigate the rows.

        Args:
            diff: The difference to navigate by.
        """
        new_row = self.current_row + diff
        start_index = new_row * self.grid_layout.column_count
        if (
            new_row >= 0
            and len(self._items[start_index:]) > self.grid_layout.column_count
        ):
            self.current_row = new_row
            self._update_grid()
