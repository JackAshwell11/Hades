"""Manages the player menu and its functionality."""

from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING, Final, cast

# Pip
from arcade import (
    SpriteList,
    Texture,
    View,
    draw_texture_rectangle,
    get_default_texture,
    get_image,
    get_window,
)
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
from PIL.ImageFilter import GaussianBlur

# Custom
from hades_extensions.game_objects import SPRITE_SIZE
from hades_extensions.game_objects.components import Inventory

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.sprite import HadesSprite
    from hades_extensions.game_objects import Registry

__all__ = ("PlayerView",)

# Get the logger
logger = logging.getLogger(__name__)

# Constants
WIDGET_SPACING: Final[int] = 5
PLAYER_VIEW_BACKGROUND_COLOUR: Final[Color] = Color(198, 198, 198)
BUTTON_BACKGROUND_COLOUR: Final[Color] = Color(68, 68, 68)
TAB_SEPARATOR_COLOUR: Final[Color] = Color(128, 128, 128)


def create_divider_line(*, vertical: bool = False) -> UIWidget:
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


class InventoryItemButton(UIBoxLayout):
    """Represents an inventory item button."""

    __slots__ = (
        "_sprite_object",
        "default_layout",
        "flat_button",
        "sprite_layout",
        "texture_button",
    )

    def __init__(
        self,
        callback: Callable[[UIOnClickEvent], None],
        sprite_object: HadesSprite | None = None,
    ) -> None:
        """Initialise the object.

        Args:
            callback: The callback to call when the button is clicked.
            sprite_object: The sprite object to display.
        """
        # Store the sprite object if it exists
        self._sprite_object: HadesSprite | None = sprite_object

        # The default layout which just has a black border
        self.default_layout: UISpace = UISpace(
            color=PLAYER_VIEW_BACKGROUND_COLOUR,
            width=SPRITE_SIZE,
            height=90,
        ).with_border(color=(0, 0, 0))

        # The sprite layout which displays the sprite and a "Use" button
        self.sprite_layout: UIBoxLayout = UIBoxLayout()
        self.texture_button: UITextureButton = UITextureButton(
            texture=get_default_texture(),
            width=SPRITE_SIZE,
            height=60,
        )
        self.texture_button.on_click = callback  # type: ignore[assignment]
        self.sprite_layout.add(self.texture_button.with_border(color=(0, 0, 0)))
        flat_button: UIFlatButton = UIFlatButton(
            text="Use",
            width=SPRITE_SIZE,
            height=30,
        )
        flat_button.on_click = callback  # type: ignore[assignment]
        self.sprite_layout.add(flat_button.with_border(color=(0, 0, 0)))
        super().__init__(children=(self.default_layout,))

    @property
    def sprite_object(self: InventoryItemButton) -> HadesSprite | None:
        """Get the sprite object.

        Returns:
            The sprite object.
        """
        return self._sprite_object

    @sprite_object.setter
    def sprite_object(self: InventoryItemButton, value: HadesSprite | None) -> None:
        """Set the sprite object.

        Args:
            value: The sprite object.
        """
        if value:
            self._sprite_object = value
            self.texture_button.texture = value.texture
            self.texture_button.texture_hovered = value.texture
            self.texture_button.texture_pressed = value.texture
            self.remove(self.default_layout)
            self.add(self.sprite_layout)
        else:
            self._sprite_object = None
            self.remove(self.sprite_layout)
            self.add(self.default_layout)


class PaginatedGridLayout(UIBoxLayout):
    """Create a paginated grid layout.

    Attributes:
        button_layout: The button layout to navigate the grid.
        current_row: The current row to display at the top of the grid layout.
        grid_layout: The grid layout to display the items.
        items: The items to display in the grid layout.
    """

    __slots__ = (
        "button_layout",
        "current_row",
        "grid_layout",
        "items",
        "total_count",
    )

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

    def __init__(
        self,
        column_count: int,
        row_count: int,
        total_count: int,
        callback: Callable[[UIOnClickEvent], None],
    ) -> None:
        """Initialise the object.

        Args:
            column_count: The number of columns to display.
            row_count: The number of rows to display.
            total_count: The total number of items to display.
            callback: The callback to call when an item is clicked.
        """
        super().__init__(
            vertical=False,
            space_between=WIDGET_SPACING,
        )

        # Create and add the layouts necessary for this object
        self.grid_layout = UIGridLayout(
            column_count=column_count,
            row_count=row_count,
            horizontal_spacing=WIDGET_SPACING,
            vertical_spacing=WIDGET_SPACING,
        )
        self.add(self.grid_layout)
        self.button_layout = UIButtonRow(
            vertical=True,
            space_between=WIDGET_SPACING,
        )
        self.button_layout.on_action = self.on_action  # type: ignore[method-assign]
        self.button_layout.add_button("Up")
        self.button_layout.add_button("Down")
        self.add(self.button_layout)

        # Initialise the rest of the object
        self.total_count: int = total_count
        self.current_row: int = 0
        self.items = [InventoryItemButton(callback) for _ in range(total_count)]
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
        if new_row >= 0 and new_row * self.grid_layout.column_count < self.total_count:
            self.current_row = new_row
            self._update_grid()


class PlayerView(View):
    """Creates a player view useful for managing the player and its attributes.

    Attributes:
        background_image: The background image to display.
        ui_manager: Manages all the different UI elements for this view.
        inventory_layout: The grid layout for the player's inventory.
    """

    __slots__ = (
        "background_image",
        "game_object_id",
        "inventory",
        "inventory_layout",
        "item_sprites",
        "registry",
        "stats_layout",
        "ui_manager",
    )

    def __init__(
        self: PlayerView,
        registry: Registry,
        game_object_id: int,
        item_sprites: SpriteList[HadesSprite],
    ) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
                systems.
            game_object_id: The ID of the player game object to manage.
            item_sprites: The list of item sprites that exist in the game.
        """
        super().__init__()
        self.registry: Registry = registry
        self.game_object_id: int = game_object_id
        self.item_sprites: SpriteList[HadesSprite] = item_sprites
        self.inventory: Inventory = self.registry.get_component(
            game_object_id,
            Inventory,
        )
        self.background_image: Texture = get_default_texture()
        self.ui_manager: UIManager = UIManager()

        # Create the UI widgets which will modify the state
        self.inventory_layout: PaginatedGridLayout = PaginatedGridLayout(
            7,
            2,
            self.inventory.get_capacity(),
            self.update_info_view,
        )
        self.stats_layout: UIBoxLayout = UIBoxLayout(
            width=get_window().width * 0.3,
            height=get_window().height * 0.4,
            children=(
                UILabel("Test", text_color=(0, 0, 0)),  # type: ignore[arg-type]
                create_divider_line(),
                UIWidget().with_background(texture=get_default_texture()),
                create_divider_line(),
                UILabel(text="Test description", text_color=(0, 0, 0)),
            ),
            space_between=WIDGET_SPACING,
        )

        # Make the player view UI
        root_layout = UIBoxLayout(vertical=True, space_between=WIDGET_SPACING)
        root_layout.add(
            self.stats_layout.with_background(
                color=PLAYER_VIEW_BACKGROUND_COLOUR,
            ).with_padding(all=WIDGET_SPACING),
        )
        self.make_player_attributes(root_layout)
        back_button = UIFlatButton(text="Back")
        back_button.on_click = (  # type: ignore[method-assign]
            lambda _: self.window.show_view(  # type: ignore[assignment]
                self.window.views["Game"],
            )
        )
        root_layout.add(back_button)
        self.ui_manager.add(UIAnchorLayout(children=(root_layout,)))

        # Update the inventory to show the player's items
        self.on_update_inventory(game_object_id)

    def on_draw(self: PlayerView) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the background colour and the UI elements
        if self.background_image:
            draw_texture_rectangle(
                self.window.width // 2,
                self.window.height // 2,
                self.window.width,
                self.window.height,
                self.background_image,
            )
        self.ui_manager.draw()

    def on_show_view(self: PlayerView) -> None:
        """Process show view functionality."""
        self.ui_manager.enable()
        self.background_image = Texture(get_image().filter(GaussianBlur(5)))

    def on_hide_view(self: PlayerView) -> None:
        """Process hide view functionality."""
        self.ui_manager.disable()

    def on_update_inventory(self: PlayerView, _: int) -> None:
        """Update the inventory view."""
        for i, item in enumerate(self.inventory.items):
            inventory_item = self.inventory_layout.items[i]
            for item_sprite in self.item_sprites:
                if item_sprite.game_object_id == item:
                    inventory_item.sprite_object = item_sprite
                    break

    def make_player_attributes(self: PlayerView, root_layout: UIBoxLayout) -> None:
        """Make the player attributes UI.

        Args:
            root_layout: The layout to add the player attributes to.
        """
        # Create the required layouts for this section
        base_layout = UIBoxLayout(
            width=get_window().width * 0.8,
            height=get_window().height * 0.5,
            space_between=WIDGET_SPACING,
        )
        upgrades_layout = UIBoxLayout(space_between=WIDGET_SPACING)

        # The event handler for the button rows
        def on_action(event: UIOnActionEvent) -> None:
            """Handle the button row actions.

            Args:
                event: The event that occurred.
            """
            if event.action == "Inventory" and upgrades_layout in base_layout.children:
                base_layout.remove(upgrades_layout)
                base_layout.add(self.inventory_layout)
            elif (
                event.action == "Upgrades"
                and self.inventory_layout in base_layout.children
            ):
                base_layout.remove(self.inventory_layout)
                base_layout.add(upgrades_layout)

        # Add the tab menu
        tab_menu = UIButtonRow(space_between=WIDGET_SPACING)
        tab_menu.on_action = on_action  # type: ignore[method-assign]
        tab_menu.add_button("Inventory")
        tab_menu.add_button("Upgrades")

        # Add all the widgets to their respective layouts
        upgrades_layout.add(UILabel(text="Test upgrades"))
        base_layout.add(tab_menu)
        base_layout.add(create_divider_line())
        base_layout.add(self.inventory_layout)
        root_layout.add(
            base_layout.with_background(
                color=PLAYER_VIEW_BACKGROUND_COLOUR,
            ).with_padding(
                all=WIDGET_SPACING,
            ),
        )

    def update_info_view(self: PlayerView, event: UIOnClickEvent) -> None:
        """Update the info view.

        Args:
            event: The event that occurred.
        """
        # Get the inventory item that was clicked
        inventory_item = cast(InventoryItemButton, event.source.parent.parent)
        if not inventory_item.sprite_object:
            return

        # Update the stats layout with the inventory item's information
        title = cast(UILabel, self.stats_layout.children[0])
        title.text = inventory_item.sprite_object.name
        title.fit_content()
        description = cast(UILabel, self.stats_layout.children[4])
        description.text = inventory_item.sprite_object.description
        description.fit_content()
        self.stats_layout.children[2].with_background(
            texture=inventory_item.sprite_object.texture,
        )

    def __repr__(self: PlayerView) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<PlayerView (Current window={self.window})>"
