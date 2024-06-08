"""Manages the player menu and its functionality."""

from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING, Final

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
from arcade.color import Color
from arcade.gui import (
    UIAnchorLayout,
    UIBoxLayout,
    UIButtonRow,
    UIFlatButton,
    UIGridLayout,
    UILabel,
    UIManager,
    UIOnActionEvent,
    UISpace,
    UIWidget,
)
from PIL.ImageFilter import GaussianBlur

if TYPE_CHECKING:
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


class PaginatedGridLayout(UIGridLayout):
    """Create a paginated grid layout.

    Attributes:
        current_page: The current page of the grid layout.
        items: The items to display in the grid layout.
        items_per_page: The number of items to display per page.
    """

    __slots__ = ("current_page", "items", "items_per_page", "total_count")

    def __init__(self, column_count: int, row_count: int, total_count: int) -> None:
        """Initialise the object.

        Args:
            column_count: The number of columns to display.
            row_count: The number of rows to display.
            total_count: The total number of items to display.
        """
        super().__init__(
            column_count=column_count,
            row_count=row_count,
            horizontal_spacing=WIDGET_SPACING,
            vertical_spacing=WIDGET_SPACING,
        )
        self.total_count: int = total_count
        self.current_page: int = 0
        self.items_per_page: int = column_count * row_count
        self.items = [
            UIBoxLayout(
                children=(
                    UIFlatButton(text=f"Slot {i + 1}", width=90, height=60),
                    UIFlatButton(text="Use", width=90, height=30),
                ),
            )
            for i in range(self.total_count)
        ]
        self.update_grid()

    def next_page(self: PaginatedGridLayout) -> None:
        """Go to the next page."""
        if self.current_page < len(self.items) // self.items_per_page:
            self.current_page += 1
            self.update_grid()

    def previous_page(self: PaginatedGridLayout) -> None:
        """Go to the previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_grid()

    def update_grid(self: PaginatedGridLayout) -> None:
        """Update the grid."""
        self.clear()
        start = self.current_page * self.items_per_page
        for i, item in enumerate(self.items[start : start + self.items_per_page]):
            row, column = divmod(i, self.column_count)
            self.add(item, column, row)


class PlayerView(View):
    """Creates a player view useful for managing the player and its attributes.

    Attributes:
        background_image: The background image to display.
        ui_manager: Manages all the different UI elements for this view.
        inventory_grid: The grid layout for the player's inventory.
    """

    __slots__ = (
        "background_image",
        "game_object_id",
        "inventory_grid",
        "item_sprites",
        "registry",
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
        self.background_image: Texture = get_default_texture()
        self.ui_manager: UIManager = UIManager()

        # Create the UI widgets which will modify the state
        self.inventory_grid = PaginatedGridLayout(7, 2, 18)

        # Make the player view UI
        root_layout = UIBoxLayout(vertical=True, space_between=WIDGET_SPACING)
        self.make_info_box(root_layout)
        self.make_player_attributes(root_layout)
        self.ui_manager.add(UIAnchorLayout(children=(root_layout,)))

        # Update the inventory to show the player's items
        # self.on_update_inventory(game_object_id)

    def make_info_box(self: PlayerView, layout: UIBoxLayout) -> None:
        """Make the info box UI.

        Args:
            layout: The layout to add the info box to.
        """
        # Add the stats and sprite image
        stats_layout = UIBoxLayout(
            vertical=False,
            space_between=WIDGET_SPACING,
            children=(
                UILabel(text="Test stats"),
                create_divider_line(vertical=True),
                UIWidget().with_background(texture=get_default_texture()),
            ),
        )

        # Add the root layout to the UI manager
        layout.add(
            UIBoxLayout(
                width=get_window().width * 0.3,
                height=get_window().height * 0.4,
                children=(
                    UILabel("Test"),
                    create_divider_line(),
                    stats_layout,
                    create_divider_line(),
                    UILabel(text="Test description"),
                ),
                space_between=WIDGET_SPACING,
            )
            .with_background(color=PLAYER_VIEW_BACKGROUND_COLOUR)
            .with_padding(
                all=WIDGET_SPACING,
            ),
        )

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
        inventory_layout = UIBoxLayout(vertical=False, space_between=WIDGET_SPACING)
        upgrades_layout = UIBoxLayout(space_between=WIDGET_SPACING)

        # The event handler for the button rows
        def on_action(event: UIOnActionEvent) -> None:
            """Handle the button row actions.

            Args:
                event: The event that occurred.
            """
            if event.action == "Inventory" and upgrades_layout in base_layout.children:
                base_layout.remove(upgrades_layout)
                base_layout.add(inventory_layout)
            elif (
                event.action == "Upgrades" and inventory_layout in base_layout.children
            ):
                base_layout.remove(inventory_layout)
                base_layout.add(upgrades_layout)
            elif event.action == "Up":
                self.inventory_grid.previous_page()
            elif event.action == "Down":
                self.inventory_grid.next_page()

        # Add the tab menu
        tab_menu = UIButtonRow(space_between=WIDGET_SPACING)
        tab_menu.on_action = on_action
        tab_menu.add_button("Inventory")
        tab_menu.add_button("Upgrades")

        # Add the up and down buttons
        up_down_buttons = UIButtonRow(vertical=True, space_between=WIDGET_SPACING)
        up_down_buttons.on_action = on_action
        up_down_buttons.add_button("Up")
        up_down_buttons.add_button("Down")

        # Add all the widgets to their respective layouts
        inventory_layout.add(self.inventory_grid)
        inventory_layout.add(up_down_buttons)
        upgrades_layout.add(UILabel(text="Test upgrades"))
        base_layout.add(tab_menu)
        base_layout.add(create_divider_line())
        base_layout.add(inventory_layout)
        root_layout.add(
            base_layout.with_background(
                color=PLAYER_VIEW_BACKGROUND_COLOUR,
            ).with_padding(
                all=WIDGET_SPACING,
            ),
        )

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

    def on_update_inventory(self: PlayerView, game_object_id: int) -> None:
        """Update the inventory view.

        Args:
            game_object_id: The ID of the game object whose inventory to update.
        """
        # Create a grid layout
        grid_layout = UIGridLayout(
            column_count=5,
            row_count=5,
            horizontal_spacing=5,
            vertical_spacing=5,
        )
        for y in range(5):
            for x in range(5):
                sprite_button = UIFlatButton(
                    text=f"Slot {y + x * 5}",
                    width=90,  # Increase the width
                    height=60,  # Increase the height
                )
                operation_button = UIFlatButton(
                    text="Use",
                    width=90,  # Increase the width
                    height=30,  # Increase the height
                )
                grid_layout.add(
                    UIBoxLayout(children=(sprite_button, operation_button)),
                    y,
                    x,
                )

        # Add the grid layout to the UI manager
        ui_anchor_layout = UIAnchorLayout()
        ui_anchor_layout.add(
            child=grid_layout,
            anchor_x="center_x",
            anchor_y="center_y",
        )
        self.ui_manager.add(ui_anchor_layout)

    def __repr__(self: PlayerView) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<PlayerView (Current window={self.window})>"
