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
from hades_extensions.ecs import SPRITE_SIZE
from hades_extensions.ecs.components import Money
from hades_extensions.ecs.systems import InventorySystem, UpgradeSystem

if TYPE_CHECKING:
    from hades.player import Player
    from hades.sprite import HadesSprite
    from hades.window import HadesWindow
    from hades_extensions.ecs.components import ActionFunction, Stat

__all__ = (
    "InventoryItemButton",
    "ItemButton",
    "PaginatedGridLayout",
    "PlayerAttributesLayout",
    "PlayerView",
    "StatsLayout",
    "UpgradesItemButton",
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


class ItemButton(UIBoxLayout, ABC):
    """Represents an item button.

    Attributes:
        default_layout: The default layout for the item button.
        sprite_layout: The layout for displaying the item sprite.
        texture_button: The button for displaying the item texture.
    """

    __slots__ = ("default_layout", "item_button", "sprite_layout", "texture_button")

    def __init__(self: ItemButton) -> None:
        """Initialise the object."""
        self.item_button: ItemButton | None = None
        self.default_layout: UISpace = UISpace(
            color=PLAYER_BACKGROUND_COLOUR,
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT,
        ).with_border(color=color.BLACK)
        self.sprite_layout: UIBoxLayout = UIBoxLayout()
        self.texture_button: UITextureButton = UITextureButton(
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT * 2 // 3,
        )
        self.texture_button.on_click = (  # type: ignore[method-assign]
            self.on_texture_button_click
        )
        self.sprite_layout.add(self.texture_button.with_border(color=color.BLACK))
        button = UIFlatButton(
            text="Use",
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT // 3,
        )
        button.on_click = self.on_use_button_click  # type: ignore[method-assign]
        self.sprite_layout.add(button.with_border(color=color.BLACK))
        super().__init__(children=(self.default_layout,))

    @abstractmethod
    def get_info(self: ItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            The name, description, and texture of the item.
        """
        raise NotImplementedError

    @abstractmethod
    def use(self: ItemButton) -> None:
        """Use the item."""
        raise NotImplementedError

    @property
    def texture(self: ItemButton) -> Texture | None:
        """Get the texture of the item button.

        Returns:
            The texture of the item button.
        """
        return self.texture_button.texture  # type: ignore[no-any-return]

    @texture.setter
    def texture(self: ItemButton, value: Texture | None) -> None:
        """Set the texture of the item button.

        Args:
            value: The texture to set.
        """
        self.texture_button.texture = value
        self.texture_button.texture_hovered = value
        self.texture_button.texture_pressed = value
        if value and self.default_layout in self.children:
            self.remove(self.default_layout)
            self.add(self.sprite_layout)
        elif self.sprite_layout in self.children:
            self.remove(self.sprite_layout)
            self.add(self.default_layout)

    @staticmethod
    def on_texture_button_click(event: UIOnClickEvent) -> None:
        """Handle the texture button click.

        Args:
            event: The event that occurred.
        """
        get_window().dispatch_event("on_texture_button_callback", event)

    @staticmethod
    def on_use_button_click(event: UIOnClickEvent) -> None:
        """Handle the use button click.

        Args:
            event: The event that occurred.
        """
        get_window().dispatch_event("on_use_button_callback", event)


class InventoryItemButton(ItemButton):
    """Represents an inventory item button."""

    __slots__ = ("_sprite_object",)

    def __init__(self: InventoryItemButton) -> None:
        """Initialise the object."""
        super().__init__()
        self._sprite_object: HadesSprite | None = None

    @property
    def sprite_object(self: InventoryItemButton) -> HadesSprite | None:
        """Get the sprite object of the item button.

        Returns:
            The sprite object of the item button.
        """
        return self._sprite_object

    @sprite_object.setter
    def sprite_object(self: InventoryItemButton, value: HadesSprite | None) -> None:
        """Set the sprite object of the item button.

        Args:
            value: The sprite object to set.
        """
        self._sprite_object = value
        self.texture = value.texture if value else None

    def get_info(self: InventoryItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Returns:
            The name, description, and texture of the item.
        """
        if self.sprite_object is None or self.texture is None:
            return "", "", get_default_texture()
        return self.sprite_object.name, self.sprite_object.description, self.texture

    def use(self: InventoryItemButton) -> None:
        """Use the item."""
        if self.sprite_object is None:
            return
        view = cast("Player", get_window().current_view)
        view.controller.model.registry.get_system(InventorySystem).use_item(
            view.controller.model.player_id,
            self.sprite_object.game_object_id,
        )


class UpgradesItemButton(ItemButton):
    """Represents an upgrades item button."""

    __slots__ = ("target_component", "target_functions")

    def __init__(self: UpgradesItemButton) -> None:
        """Initialise the object."""
        super().__init__()
        self.target_component: type[Stat] | None = None
        self.target_functions: tuple[ActionFunction, ActionFunction] | None = None

    def get_description(self: UpgradesItemButton) -> str:
        """Get the description of the item.

        Raises:
            ValueError: If no target component or functions are set for this item.

        Returns:
            The description of the item.
        """
        if not self.target_component or not self.target_functions:
            error = "No target component or functions set for this item."
            raise ValueError(error)

        # Get the required components
        view = cast("Player", get_window().current_view)
        component = view.controller.model.registry.get_component(
            view.controller.model.player_id,
            self.target_component,
        )
        money = view.controller.model.registry.get_component(
            view.controller.model.player_id,
            Money,
        )

        # Calculate the new values for the description
        new_component_value = component.get_max_value() + self.target_functions[0](
            component.get_current_level(),
        )
        new_money_value = money.money - self.target_functions[1](
            component.get_current_level(),
        )

        # Return the description
        return (
            f"Max {self.target_component.__name__}:\n"
            f"  {component.get_max_value()} -> {new_component_value}\n"
            f"{money.__class__.__name__}:\n"
            f"  {money.money} -> {new_money_value}"
        )

    def get_info(self: UpgradesItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Raises:
            ValueError: If no target component is set for this item.

        Returns:
            The name, description, and texture of the item.
        """
        if self.target_component is None or self.texture is None:
            error = "No target component or functions set for this item."
            raise ValueError(error)
        return self.target_component.__name__, self.get_description(), self.texture

    def use(self: UpgradesItemButton) -> None:
        """Use the item.

        Raises:
            ValueError: If no target component or functions are set for this item.
        """
        if not self.target_component:
            error = "No target component or functions set for this item."
            raise ValueError(error)
        view = cast("Player", get_window().current_view)
        view.controller.model.registry.get_system(UpgradeSystem).upgrade_component(
            view.controller.model.player_id,
            self.target_component,
        )
        view.view.stats_layout.set_info(*self.get_info())


class PaginatedGridLayout(UIBoxLayout):
    """Represents a paginated grid layout for displaying items.

    Attributes:
        button_layout: The button layout to navigate the grid.
        current_row: The current row to display at the top of the grid layout.
        grid_layout: The grid layout to display the items.
        items: The items to display in the grid layout.
    """

    __slots__ = (
        "button_layout",
        "button_type",
        "current_row",
        "grid_layout",
        "items",
    )

    def __init__(self: PaginatedGridLayout, button_type: type[ItemButton]) -> None:
        """Initialise the object.

        Args:
            button_type: The type of item button to use.
        """
        super().__init__(
            vertical=False,
            space_between=PLAYER_WIDGET_SPACING,
        )
        self.button_type: type[ItemButton] = button_type
        self.grid_layout: UIGridLayout = UIGridLayout(
            column_count=round(
                (get_window().width * 0.4) / (SPRITE_SIZE + PLAYER_WIDGET_SPACING),
            ),
            row_count=round(
                (get_window().height * 0.3)
                / (ITEM_BUTTON_HEIGHT + PLAYER_WIDGET_SPACING),
            ),
            horizontal_spacing=PLAYER_WIDGET_SPACING,
            vertical_spacing=PLAYER_WIDGET_SPACING,
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
        self.items: list[ItemButton] = []

    def _update_grid(self: PaginatedGridLayout) -> None:
        """Update the grid adding the next row of items to display."""
        self.grid_layout.clear()
        start = self.current_row * self.grid_layout.column_count
        for i, item in enumerate(
            self.items[
                start : start
                + self.grid_layout.column_count * self.grid_layout.row_count
            ],
        ):
            row, column = divmod(i, self.grid_layout.column_count)
            self.grid_layout.add(item, column, row)

    @property
    def total_size(self: PaginatedGridLayout) -> int:
        """Get the total size of the grid layout.

        Returns:
            The total size of the grid layout.
        """
        return len(self.items)

    @total_size.setter
    def total_size(self: PaginatedGridLayout, value: int) -> None:
        """Set the total size of the grid layout.

        Args:
            value: The total size of the grid layout.
        """
        self.items = [
            self.button_type()
            for _ in range(
                max(self.grid_layout.column_count * self.grid_layout.row_count, value),
            )
        ]
        self._update_grid()

    def on_action(self: PaginatedGridLayout, event: UIOnActionEvent) -> None:
        """Handle the button row actions.

        Args:
            event: The event that occurred.
        """
        if event.action == "Up":
            self.navigate_rows(-1)
        elif event.action == "Down":
            self.navigate_rows(1)

    def navigate_rows(self: PaginatedGridLayout, diff: int) -> None:
        """Navigate the rows.

        Args:
            diff: The difference to navigate by.
        """
        new_row = self.current_row + diff
        start_index = new_row * self.grid_layout.column_count
        if (
            new_row >= 0
            and len(self.items[start_index:]) > self.grid_layout.column_count
        ):
            self.current_row = new_row
            self._update_grid()


class StatsLayout(UIBoxLayout):
    """Represents a layout for displaying stats about the player or an item."""

    __slots__ = ()

    def __init__(self: StatsLayout) -> None:
        """Initialise the object."""
        super().__init__(size_hint=(0.3, 0.4), space_between=PLAYER_WIDGET_SPACING)
        self.add(UILabel("Test", text_color=(0, 0, 0)))
        self.add(create_divider_line())
        self.add(UIWidget().with_background(texture=get_default_texture()))
        self.add(create_divider_line())
        self.add(UILabel(text="Test description", text_color=(0, 0, 0), multiline=True))
        self.with_background(color=PLAYER_BACKGROUND_COLOUR).with_padding(
            all=PLAYER_WIDGET_SPACING,
        )

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


class PlayerAttributesLayout(UIBoxLayout):
    """Represents a layout for displaying the player's attributes.

    Attributes:
        inventory_layout: The layout for displaying the player's inventory.
        upgrades_layout: The layout for displaying the player's upgrades
    """

    __slots__ = ("inventory_layout", "upgrades_layout")

    def __init__(self: PlayerAttributesLayout) -> None:
        """Initialise the object."""
        super().__init__(size_hint=(0.8, 0.5), space_between=PLAYER_WIDGET_SPACING)
        self.inventory_layout: PaginatedGridLayout = PaginatedGridLayout(
            InventoryItemButton,
        )
        self.upgrades_layout: PaginatedGridLayout = PaginatedGridLayout(
            UpgradesItemButton,
        )

        # Add the tab menu
        tab_menu = UIButtonRow(space_between=PLAYER_WIDGET_SPACING)
        tab_menu.on_action = self.on_action  # type: ignore[method-assign]
        tab_menu.add_button("Inventory")
        tab_menu.add_button("Upgrades")

        # Add all the widgets to the layout
        self.add(tab_menu)
        self.add(create_divider_line())
        self.add(self.inventory_layout)
        self.with_background(color=PLAYER_BACKGROUND_COLOUR).with_padding(
            all=PLAYER_WIDGET_SPACING,
        )

    def on_action(self: PlayerAttributesLayout, event: UIOnActionEvent) -> None:
        """Handle the button row actions.

        Args:
            event: The event that occurred.
        """
        if event.action == "Inventory" and self.upgrades_layout in self.children:
            self.remove(self.upgrades_layout)
            self.add(self.inventory_layout)
        elif event.action == "Upgrades" and self.inventory_layout in self.children:
            self.remove(self.inventory_layout)
            self.add(self.upgrades_layout)


class PlayerView:
    """Manages the rendering of player elements on the screen.

    Attributes:
        ui: The UI manager for the game view.
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
            window: The window for the game.
        """
        self.window: HadesWindow = window
        self.ui: UIManager = UIManager()
        self.stats_layout: StatsLayout = StatsLayout()
        self.player_attributes_layout: PlayerAttributesLayout = PlayerAttributesLayout()

    def setup(self: PlayerView) -> None:
        """Set up the renderer."""
        self.ui.clear()
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

    def draw(self: PlayerView) -> None:
        """Draw the player elements."""
        self.window.clear()
        self.ui.draw()
