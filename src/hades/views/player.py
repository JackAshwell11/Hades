"""Manages the player menu and its functionality."""

from __future__ import annotations

# Builtin
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final, TypedDict, Unpack, cast

# Pip
from arcade import (
    XYWH,
    SpriteList,
    Texture,
    View,
    draw_texture_rect,
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
from hades_extensions.game_objects.components import (
    Inventory,
    InventorySize,
    Money,
    PythonSprite,
    Stat,
    Upgrades,
)
from hades_extensions.game_objects.systems import InventorySystem, UpgradeSystem

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.sprite import HadesSprite
    from hades_extensions.game_objects import ActionFunction, Registry

__all__ = (
    "InventoryItemButton",
    "ItemButton",
    "ItemButtonKwargs",
    "PaginatedGridLayout",
    "PlayerAttributesLayout",
    "PlayerView",
    "StatsLayout",
    "UpgradesItemButton",
    "create_divider_line",
)

# Get the logger
logger = logging.getLogger(__name__)

# Constants
WIDGET_SPACING: Final[int] = 5
ITEM_BUTTON_HEIGHT: Final[int] = 90
PLAYER_VIEW_BACKGROUND_COLOUR: Final[Color] = Color(198, 198, 198)
BUTTON_BACKGROUND_COLOUR: Final[Color] = Color(68, 68, 68)
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


class ItemButtonKwargs(TypedDict, total=False):
    """The keyword arguments for the item button."""

    target_component: type[Stat]
    target_functions: tuple[ActionFunction, ActionFunction]


class ItemButton(UIBoxLayout, ABC):
    """Represents an item button."""

    __slots__ = ("default_layout", "sprite_layout", "texture_button")

    def __init__(
        self: InventoryItemButton,
        callback: Callable[[UIOnClickEvent], None],
        **_: Unpack[ItemButtonKwargs],
    ) -> None:
        """Initialise the object.

        Args:
            callback: The callback to call when an item is clicked or used.
        """
        # The sprite layout which displays the sprite and a "Use" button
        self.sprite_layout: UIBoxLayout = UIBoxLayout()
        self.texture_button: UITextureButton = UITextureButton(
            texture=get_default_texture(),
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT * 2 // 3,
        )
        self.texture_button.on_click = callback  # type: ignore[assignment]
        self.sprite_layout.add(self.texture_button.with_border(color=(0, 0, 0)))
        flat_button: UIFlatButton = UIFlatButton(
            text="Use",
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT // 3,
        )
        flat_button.on_click = callback  # type: ignore[assignment]
        self.sprite_layout.add(flat_button.with_border(color=(0, 0, 0)))

        # The default layout which just has a black border
        self.default_layout: UISpace = UISpace(
            color=PLAYER_VIEW_BACKGROUND_COLOUR,
            width=SPRITE_SIZE,
            height=ITEM_BUTTON_HEIGHT,
        ).with_border(color=(0, 0, 0))
        super().__init__(children=(self.default_layout,))

    @abstractmethod
    def get_info(self: ItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def use(self: ItemButton) -> None:
        """Use the item."""
        raise NotImplementedError

    def __repr__(self: InventoryItemButton) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<ItemButton (Children={self.children})>"


class InventoryItemButton(ItemButton):
    """Represents an inventory item button."""

    __slots__ = ("_sprite_object",)

    def __init__(
        self: InventoryItemButton,
        callback: Callable[[UIOnClickEvent], None],
        **_: Unpack[ItemButtonKwargs],
    ) -> None:
        """Initialise the object.

        Args:
            callback: The callback to call when an item is clicked or used.
        """
        super().__init__(callback, **_)
        self._sprite_object: HadesSprite | None = None

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
        self.clear()
        self._sprite_object = value
        if value:
            self.texture_button.texture = value.texture
            self.texture_button.texture_hovered = value.texture
            self.texture_button.texture_pressed = value.texture
            self.add(self.sprite_layout)
        else:
            self.add(self.default_layout)

    def get_info(self: InventoryItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Raises:
            ValueError: If no sprite object is set for this item.

        Returns:
            The name, description, and texture of the item.
        """
        if not self.sprite_object:
            error = "No sprite object set for this item."
            raise ValueError(error)
        return (
            self.sprite_object.name,
            self.sprite_object.description,
            self.sprite_object.texture,
        )

    def use(self: InventoryItemButton) -> None:
        """Use the item.

        Raises:
            ValueError: If no sprite object is set for this
        """
        if not self.sprite_object:
            error = "No sprite object set for this item."
            raise ValueError(error)
        view = cast(PlayerView, get_window().current_view)
        view.registry.get_system(InventorySystem).use_item(
            view.game_object_id,
            self.sprite_object.game_object_id,
        )

    def __repr__(self: InventoryItemButton) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<InventoryItemButton (Sprite object={self.sprite_object})>"


class UpgradesItemButton(ItemButton):
    """Represents an upgrades item button."""

    __slots__ = (
        "description",
        "target_component",
        "target_functions",
    )

    def __init__(
        self: UpgradesItemButton,
        callback: Callable[[UIOnClickEvent], None],
        **kwargs: Unpack[ItemButtonKwargs],
    ) -> None:
        """Initialise the object.

        Args:
            callback: The callback to call when an item is clicked or used.
            kwargs: The keyword arguments for the item button.
        """
        super().__init__(callback, **kwargs)
        self.target_component: type[Stat] | None = kwargs.get("target_component")
        self.target_functions: tuple[ActionFunction, ActionFunction] | None = (
            kwargs.get("target_functions")
        )
        self.description: str = ""

        # Enable the sprite layout if valid targets are given
        if self.target_component and self.target_functions:
            self.remove(self.default_layout)
            self.add(self.sprite_layout)

    def update_description(self: UpgradesItemButton) -> None:
        """Update the description of the item."""
        if not self.target_component or not self.target_functions:
            return

        # Get the required components
        view = cast(PlayerView, get_window().current_view)
        component = view.registry.get_component(
            view.game_object_id,
            self.target_component,
        )
        money = view.registry.get_component(view.game_object_id, Money)

        # Calculate the new values for the description
        new_component_value = component.get_max_value() + self.target_functions[0](
            component.get_current_level(),
        )
        new_money_value = money.money - self.target_functions[1](
            component.get_current_level(),
        )

        # Update the description
        self.description = (
            f"Max {self.target_component.__name__}:\n"
            f"  {component.get_max_value()} -> {new_component_value}\n"
            f"{money.__class__.__name__}:\n"
            f"  {money.money} -> {new_money_value}"
        )

    def get_info(self: UpgradesItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Raises:
            ValueError: If no target component or functions are set for this item.

        Returns:
            The name, description, and texture of the item.
        """
        if not self.target_component or not self.target_functions:
            error = "No target component or functions set for this item."
            raise ValueError(error)
        return self.target_component.__name__, self.description, get_default_texture()

    def use(self: UpgradesItemButton) -> None:
        """Use the item.

        Raises:
            ValueError: If no target component or functions are set for this item
        """
        if not self.target_component or not self.target_functions:
            error = "No target component or functions set for this item."
            raise ValueError(error)
        view = cast(PlayerView, get_window().current_view)
        view.registry.get_system(UpgradeSystem).upgrade_component(
            view.game_object_id,
            self.target_component,
        )
        view.stats_layout.set_info(*self.get_info())

    def __repr__(self: UpgradesItemButton) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<UpgradesItemButton (Target component={self.target_component})>"


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
        "current_row",
        "grid_layout",
        "items",
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
        total_size: int,
        button_type: type[ItemButton],
        button_params: list[ItemButtonKwargs],
    ) -> None:
        """Initialise the object.

        Args:
            total_size: The total number of items to display in the grid layout.
            button_type: The type of button to use for the items.
            button_params: The parameters to pass to the button type.
        """
        super().__init__(
            vertical=False,
            space_between=WIDGET_SPACING,
        )

        # Create and add the layouts necessary for this object
        self.grid_layout = UIGridLayout(
            column_count=round(
                (get_window().width * 0.4) / (SPRITE_SIZE + WIDGET_SPACING),
            ),
            row_count=round(
                (get_window().height * 0.3) / (ITEM_BUTTON_HEIGHT + WIDGET_SPACING),
            ),
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
        self.current_row: int = 0
        self.items = [
            button_type(
                self.item_clicked,
                **button_params[i] if i < len(button_params) else {},
            )
            for i in range(
                max(
                    self.grid_layout.column_count * self.grid_layout.row_count,
                    total_size,
                ),
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

    @staticmethod
    def item_clicked(event: UIOnClickEvent) -> None:
        """Handle the item click event.

        Args:
            event: The event that occurred.
        """
        # Get the item that was clicked and either show its stats or use it
        # depending on what button was clicked
        item = cast(ItemButton, event.source.parent.parent)
        if isinstance(event.source, UITextureButton):
            cast(PlayerView, get_window().current_view).stats_layout.set_info(
                *item.get_info(),
            )
        else:
            item.use()

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

    def __repr__(self: PaginatedGridLayout) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<PaginatedGridLayout (Column count={self.grid_layout.column_count}) (Row"
            f" count={self.grid_layout.row_count}) (Current row={self.current_row})>"
        )


class StatsLayout(UIBoxLayout):
    """Represents a layout for displaying stats about the player or an item."""

    __slots__ = ()

    def __init__(self: StatsLayout) -> None:
        """Initialise the object."""
        super().__init__(
            width=get_window().width * 0.3,
            height=get_window().height * 0.4,
            space_between=WIDGET_SPACING,
        )
        self.add(UILabel("Test", text_color=(0, 0, 0)))
        self.add(create_divider_line())
        self.add(UIWidget().with_background(texture=get_default_texture()))
        self.add(create_divider_line())
        self.add(UILabel(text="Test description", text_color=(0, 0, 0), multiline=True))

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
        title_obj = cast(UILabel, self.children[0])
        title_obj.text = title
        title_obj.fit_content()
        description_obj = cast(UILabel, self.children[4])
        description_obj.text = description
        description_obj.fit_content()
        self.children[2].with_background(texture=texture)

    def __repr__(self: StatsLayout) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StatsLayout (Current layout={self.children[0]})>"


class PlayerAttributesLayout(UIBoxLayout):
    """Represents a layout for displaying the player's attributes.

    Attributes:
        inventory_layout: The layout for displaying the player's inventory.
        upgrades_layout: The layout for displaying the player's upgrades
    """

    __slots__ = ("inventory_layout", "upgrades_layout")

    def __init__(
        self: PlayerAttributesLayout,
        registry: Registry,
        game_object_id: int,
    ) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
                systems.
            game_object_id: The ID of the player game object to manage.
        """
        super().__init__(
            width=get_window().width * 0.8,
            height=get_window().height * 0.5,
            space_between=WIDGET_SPACING,
        )
        upgrades = registry.get_component(game_object_id, Upgrades).upgrades
        self.inventory_layout: PaginatedGridLayout = PaginatedGridLayout(
            int(registry.get_component(game_object_id, InventorySize).get_value()),
            InventoryItemButton,
            [],
        )
        self.upgrades_layout: PaginatedGridLayout = PaginatedGridLayout(
            len(upgrades),
            UpgradesItemButton,
            [
                {
                    "target_component": component,
                    "target_functions": functions,
                }
                for component, functions in upgrades.items()
            ],
        )

        # Add the tab menu
        tab_menu = UIButtonRow(space_between=WIDGET_SPACING)
        tab_menu.on_action = self.on_action  # type: ignore[method-assign]
        tab_menu.add_button("Inventory")
        tab_menu.add_button("Upgrades")

        # Add all the widgets to the layout
        self.add(tab_menu)
        self.add(create_divider_line())
        self.add(self.inventory_layout)

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

    def __repr__(self: PlayerAttributesLayout) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<PlayerAttributesLayout (Current layout={self.children[0]})>"


class PlayerView(View):
    """Creates a player view useful for managing the player and its attributes.

    Attributes:
        background_image: The background image to display.
        inventory: The player's inventory.
        player_attributes_layout: The layout for displaying the player's attributes.
        stats_layout: The layout for displaying the player or item's stats.
        ui_manager: Manages all the different UI elements for this view.
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

        # Create the UI widgets for the player view
        self.stats_layout: StatsLayout = StatsLayout()
        self.player_attributes_layout: PlayerAttributesLayout = PlayerAttributesLayout(
            self.registry,
            self.game_object_id,
        )

        # Make the player view UI
        root_layout = UIBoxLayout(vertical=True, space_between=WIDGET_SPACING)
        root_layout.add(
            self.stats_layout.with_background(
                color=PLAYER_VIEW_BACKGROUND_COLOUR,
            ).with_padding(all=WIDGET_SPACING),
        )
        root_layout.add(
            self.player_attributes_layout.with_background(
                color=PLAYER_VIEW_BACKGROUND_COLOUR,
            ).with_padding(all=WIDGET_SPACING),
        )
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
        draw_texture_rect(
            self.background_image,
            XYWH(
                self.window.width // 2,
                self.window.height // 2,
                self.window.width,
                self.window.height,
            ),
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
        for index, button in enumerate(
            self.player_attributes_layout.inventory_layout.items,
        ):
            if index < len(self.inventory.items):
                button.sprite_object = self.registry.get_component(
                    self.inventory.items[index],
                    PythonSprite,
                ).sprite
            else:
                button.sprite_object = None

    def __repr__(self: PlayerView) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<PlayerView (Current window={self.window})>"
