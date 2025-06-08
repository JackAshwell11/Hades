"""Manages the rendering of player elements on the screen."""

from __future__ import annotations

# Builtin
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final, cast

# Pip
from arcade import Texture, color, get_default_texture, get_window
from arcade.gui import (
    UIAnchorLayout,
    UIBoxLayout,
    UIButtonRow,
    UIFlatButton,
    UIGridLayout,
    UILabel,
    UIManager,
    UIOnActionEvent,
    UIOnClickEvent,
    UISpace,
    UITextureButton,
    UIWidget,
)
from arcade.types import Color

# Custom
from hades import ViewType
from hades.constructors import IconType
from hades_extensions.ecs import SPRITE_SIZE

if TYPE_CHECKING:
    from hades.sprite import HadesSprite
    from hades.window import HadesWindow

__all__ = (
    "InventoryItemButton",
    "ItemButton",
    "PaginatedGridLayout",
    "PlayerAttributesLayout",
    "PlayerView",
    "ShopItemButton",
    "StatsLayout",
    "create_default_layout",
    "create_divider_line",
)

# The spacing between the player widgets
PLAYER_WIDGET_SPACING: Final[int] = 5

# The height of the item button
ITEM_BUTTON_HEIGHT: Final[int] = 90

# The colour of the tab separator
TAB_SEPARATOR_COLOUR: Final[Color] = Color(128, 128, 128)

# The background colour of the UI
PLAYER_BACKGROUND_COLOUR: Final[Color] = Color(198, 198, 198)


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
        color=PLAYER_BACKGROUND_COLOUR,
        width=SPRITE_SIZE,
        height=ITEM_BUTTON_HEIGHT,
    ).with_border(color=color.BLACK)


class ItemButton(UIBoxLayout, ABC):
    """Represents an item button.

    Attributes:
        texture_button: The button for displaying the item texture.
    """

    __slots__ = ("texture_button", "use_button")

    def __init__(self: ItemButton) -> None:
        """Initialise the object."""
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
            text="Use",
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


class InventoryItemButton(ItemButton):
    """Represents an inventory item button."""

    __slots__ = ("sprite_object",)

    def __init__(self: InventoryItemButton, sprite_object: HadesSprite) -> None:
        """Initialise the object."""
        super().__init__()
        self.sprite_object: HadesSprite = sprite_object
        self.texture = sprite_object.texture

    def get_info(self: InventoryItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Returns:
            The name, description, and texture of the item.
        """
        return self.sprite_object.name, self.sprite_object.description, self.texture


class ShopItemButton(ItemButton):
    """Represents a shop item button."""

    __slots__ = ("cost", "description", "name", "shop_index")

    def __init__(
        self: ShopItemButton,
        index: int,
        data: tuple[str, str, str],
        cost: int,
    ) -> None:
        """Initialise the object.

        Args:
            index: The index of the item in the shop.
            data: A tuple containing the name, description, and icon type of the item.
            cost: The cost of the item.
        """
        super().__init__()
        self.shop_index: int = index
        self.name: str = data[0]
        self.description: str = data[1]
        self.cost: int = cost
        self.texture = IconType[data[2].upper()].get_texture()
        self.use_button.text = "Buy"

    def get_info(self: ShopItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Returns:
            The name, description, and texture of the item.
        """
        return self.name, f"{self.description}\nCost: {self.cost}", self.texture


class PaginatedGridLayout[T: ItemButton](UIBoxLayout):
    """Represents a paginated grid layout for displaying items.

    Attributes:
        button_layout: The button layout to navigate the grid.
        current_row: The current row to display at the top of the grid layout.
        grid_layout: The grid layout to display the items.
    """

    __slots__ = (
        "_items",
        "button_layout",
        "current_row",
        "grid_layout",
    )

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
        super().__init__(
            vertical=False,
            space_between=PLAYER_WIDGET_SPACING,
        )
        self.grid_layout: UIGridLayout = UIGridLayout(
            horizontal_spacing=PLAYER_WIDGET_SPACING,
            vertical_spacing=PLAYER_WIDGET_SPACING,
            column_count=round(
                (get_window().width * 0.4) / (SPRITE_SIZE + PLAYER_WIDGET_SPACING),
            ),
            row_count=round(
                (get_window().height * 0.3)
                / (ITEM_BUTTON_HEIGHT + PLAYER_WIDGET_SPACING),
            ),
        )
        self.add(self.grid_layout)
        self.button_layout: UIButtonRow = UIButtonRow(
            vertical=True,
            space_between=PLAYER_WIDGET_SPACING,
        )
        self.button_layout.on_action = self.on_action  # type: ignore[method-assign]
        self.button_layout.add_button("Up")
        self.button_layout.add_button("Down")
        self.add(self.button_layout)
        self.current_row: int = 0
        self._items: list[T] = []
        self._update_grid()

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


class StatsLayout(UIBoxLayout):
    """Represents a layout for displaying stats about the player or an item."""

    __slots__ = ()

    def __init__(self: StatsLayout) -> None:
        """Initialise the object."""
        super().__init__(size_hint=(0.3, 0.4), space_between=PLAYER_WIDGET_SPACING)
        self.add(UILabel("", text_color=(0, 0, 0)))
        self.add(create_divider_line())
        self.add(UIWidget())
        self.add(create_divider_line())
        self.add(UILabel(text="", text_color=(0, 0, 0), multiline=True))
        self.with_background(color=PLAYER_BACKGROUND_COLOUR).with_padding(
            all=PLAYER_WIDGET_SPACING,
        )
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
        description_obj = cast("UILabel", self.children[4])
        description_obj.text = description
        description_obj.fit_content()
        self.children[2].with_background(texture=texture)

    def reset(self: StatsLayout) -> None:
        """Reset the stats layout to its default state."""
        self.set_info("Test", "Test description", get_default_texture())


class PlayerAttributesLayout(UIBoxLayout):
    """Represents a layout for displaying the player's attributes.

    Attributes:
        inventory_layout: The layout for displaying the player's inventory.
        shop_layout: The layout for displaying the shop items.
    """

    __slots__ = ("inventory_layout", "shop_layout")

    def __init__(self: PlayerAttributesLayout) -> None:
        """Initialise the object."""
        super().__init__(size_hint=(0.8, 0.5), space_between=PLAYER_WIDGET_SPACING)
        self.inventory_layout: PaginatedGridLayout[InventoryItemButton] = (
            PaginatedGridLayout()
        )
        self.shop_layout: PaginatedGridLayout[ShopItemButton] = PaginatedGridLayout()

        # Add the tab menu
        tab_menu = UIButtonRow(space_between=PLAYER_WIDGET_SPACING)
        tab_menu.on_action = self.on_action  # type: ignore[method-assign]
        tab_menu.add_button("Inventory")
        tab_menu.add_button("Shop")

        # Add all the widgets to the layout
        self.add(tab_menu)
        self.add(create_divider_line())
        self.add(self.inventory_layout)
        self.with_background(color=PLAYER_BACKGROUND_COLOUR).with_padding(
            all=PLAYER_WIDGET_SPACING,
        )

    def on_action(self: PlayerAttributesLayout, event: UIOnActionEvent) -> None:
        """Process player attributes layout click logic.

        Args:
            event: The event that occurred.
        """
        if event.action == "Inventory" and self.shop_layout in self.children:
            self.remove(self.shop_layout)
            self.add(self.inventory_layout)
        elif event.action == "Shop" and self.inventory_layout in self.children:
            self.remove(self.inventory_layout)
            self.add(self.shop_layout)


class PlayerView:
    """Manages the rendering of player elements on the screen.

    Attributes:
        ui: The UI manager for the player view.
        stats_layout: The layout for displaying the player's stats.
        player_attributes_layout: The layout for displaying the player's attributes.
    """

    __slots__ = (
        "inventory",
        "player_attributes_layout",
        "stats_layout",
        "ui",
        "window",
    )

    def __init__(self: PlayerView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the start menu.
        """
        self.window: HadesWindow = window
        self.ui: UIManager = UIManager()
        self.stats_layout: StatsLayout = StatsLayout()
        self.player_attributes_layout: PlayerAttributesLayout = PlayerAttributesLayout()

        self.ui.add(self.window.background_image)
        layout = UIBoxLayout(vertical=True, space_between=PLAYER_WIDGET_SPACING)
        layout.add(self.stats_layout)
        layout.add(self.player_attributes_layout)
        back_button = UIFlatButton(text="Back")
        back_button.on_click = (  # type: ignore[method-assign]
            lambda _: self.window.show_view(  # type: ignore[assignment]
                self.window.views[ViewType.GAME],
            )
        )
        layout.add(back_button)
        self.ui.add(UIAnchorLayout(children=(layout,)))

    def reset(self: PlayerView) -> None:
        """Reset the view to its initial state."""
        self.stats_layout.reset()
        self.player_attributes_layout.inventory_layout.items = []
        self.player_attributes_layout.shop_layout.items = []

    def draw(self: PlayerView) -> None:
        """Draw the player elements."""
        self.window.clear()
        self.ui.draw()
