# pylint: disable=redefined-outer-name
"""Tests all classes and functions in views/player.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Pip
import pytest
from arcade.gui import UIOnActionEvent, UIOnClickEvent, UIWidget

# Custom
from hades.sprite import HadesSprite
from hades.views.player import (
    InventoryItemButton,
    PaginatedGridLayout,
    PlayerView,
    create_divider_line,
)
from hades_extensions.game_objects import GameObjectType, Registry, RegistryError, Vec2d
from hades_extensions.game_objects.components import Inventory

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture()
def mock_callback() -> Callable[[UIOnClickEvent], None]:
    """Mock the callback function for testing.

    Returns:
        The mock callback function.
    """

    def callback(_: UIOnClickEvent) -> None:
        """The mock callback function that sets a flag when called."""
        callback.called = True  # type: ignore[attr-defined]

    # Store the called flag on the function to allow for mocking
    callback.called = False  # type: ignore[attr-defined]
    return callback


@pytest.fixture()
def paginated_grid_layout(
    mock_callback: Callable[[UIOnClickEvent], None],
) -> PaginatedGridLayout:
    """Create a PaginatedGridLayout for testing.

    Args:
        mock_callback: The mock callback function to use for testing.

    Returns:
        The PaginatedGridLayout object for testing.
    """
    return PaginatedGridLayout(3, 2, 8, mock_callback)


@pytest.fixture()
def player_view() -> PlayerView:
    """Create a PlayerView for testing.

    Returns:
        The PlayerView object for testing.
    """
    # Make sure to set up the required mocks so the initialisation doesn't fail
    mock_inventory = Mock(spec=Inventory)
    mock_inventory.get_capacity.return_value = 10
    mock_inventory.items = []
    mock_registry = Mock(spec=Registry)
    mock_registry.get_component.return_value = mock_inventory
    return PlayerView(mock_registry, 0, [])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expected_width", "expected_height", "expected_size_hint", "vertical"),
    [
        (0, 2, (1, None), False),
        (2, 0, (None, 1), True),
    ],
)
def test_create_divider_line(
    expected_width: int,
    expected_height: int,
    expected_size_hint: tuple[int, int],
    *,
    vertical: bool,
) -> None:
    """Test that create_divider_line() creates the correct UISpace widget.

    Args:
        expected_width: The expected width of the divider line.
        expected_height: The expected height of the divider line.
        expected_size_hint: The expected size hint of the divider line.
        vertical: Whether the divider line should be vertical or not.
    """
    space = create_divider_line(vertical=vertical)
    assert space.height == expected_height
    assert space.width == expected_width
    assert space.color == (128, 128, 128, 255)
    assert space.size_hint == expected_size_hint


@pytest.mark.usefixtures("window")
def test_inventory_item_button_init(
    mock_callback: Callable[[UIOnClickEvent], None],
) -> None:
    """Test that the InventoryItemButton initializes correctly.

    Args:
        mock_callback: The mock callback function to use for testing.
    """
    inventory_item = InventoryItemButton(mock_callback)
    assert inventory_item.sprite_object is None
    for child in inventory_item.sprite_layout.children:
        assert child.on_click == mock_callback
    assert inventory_item.children == [inventory_item.default_layout]


@pytest.mark.usefixtures("window")
def test_inventory_item_button_set_sprite(
    mock_callback: Callable[[UIOnClickEvent], None],
) -> None:
    """Test that the InventoryItemButton correctly sets the sprite object.

    Args:
        mock_callback: The mock callback function to use for testing.
    """
    # Create a mock sprite object
    mock_sprite_object = Mock(spec=HadesSprite)
    mock_sprite_object.texture = "test"

    # Test setting the sprite object changes to the sprite layout
    inventory_item = InventoryItemButton(mock_callback)
    inventory_item.sprite_object = mock_sprite_object
    assert inventory_item.sprite_object == mock_sprite_object
    assert inventory_item.texture_button.texture == "test"
    assert inventory_item.texture_button.texture_hovered == "test"
    assert inventory_item.texture_button.texture_pressed == "test"
    assert inventory_item.children == [inventory_item.sprite_layout]

    # Test setting the sprite object to None changes back to the default layout
    inventory_item.sprite_object = None
    assert inventory_item.sprite_object is None
    assert inventory_item.children == [inventory_item.default_layout]


@pytest.mark.usefixtures("window")
def test_inventory_item_button_on_click(
    mock_callback: Callable[[UIOnClickEvent], None],
) -> None:
    """Test that the InventoryItemButton correctly calls the callback function.

    Args:
        mock_callback: The mock callback function to use for testing.
    """
    inventory_item = InventoryItemButton(mock_callback)
    inventory_item.texture_button.on_click(None)  # type: ignore[arg-type]
    assert mock_callback.called  # type: ignore[attr-defined]


@pytest.mark.usefixtures("window")
def test_inventory_item_button_on_click_invalid_callback() -> None:
    """Test that the InventoryItemButton raises a TypeError for an invalid callback."""

    def invalid_callback(_: str, __: int) -> None:
        """An invalid callback function that takes too many arguments."""

    inventory_item = InventoryItemButton(invalid_callback)  # type: ignore[arg-type]
    with pytest.raises(
        expected_exception=TypeError,
        match=".*missing 1 required positional argument: '__'.*",
    ):
        inventory_item.texture_button.on_click(None)  # type: ignore[arg-type]


@pytest.mark.usefixtures("window")
@pytest.mark.parametrize(
    ("column_count", "row_count", "total_count"),
    [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 4),
        (3, 3, 9),
        (4, 4, 16),
        (5, 5, 25),
        (5, 5, 50),
    ],
)
def test_paginated_grid_layout_init(
    column_count: int,
    row_count: int,
    total_count: int,
    mock_callback: Callable[[UIOnClickEvent], None],
) -> None:
    """Test that the PaginatedGridLayout initializes correctly.

    Args:
        column_count: The number of columns in the grid layout.
        row_count: The number of rows in the grid layout.
        total_count: The total number of items in the grid layout.
        mock_callback: The mock callback function to use for testing.
    """
    paginated_grid_layout = PaginatedGridLayout(
        column_count,
        row_count,
        total_count,
        mock_callback,
    )
    assert paginated_grid_layout.grid_layout.column_count == column_count
    assert paginated_grid_layout.grid_layout.row_count == row_count
    assert paginated_grid_layout.total_count == total_count
    assert paginated_grid_layout.current_row == 0
    assert [button.text for button in paginated_grid_layout.button_layout.children] == [
        "Up",
        "Down",
    ]
    assert len(paginated_grid_layout.items) == total_count
    assert len(paginated_grid_layout.grid_layout.children) == column_count * row_count


@pytest.mark.usefixtures("window")
def test_paginated_grid_layout_on_action(
    paginated_grid_layout: PaginatedGridLayout,
) -> None:
    """Test that the PaginatedGridLayout correctly calls the on_action method.

    Args:
        paginated_grid_layout: The PaginatedGridLayout object for testing.
    """
    # Test the down button
    paginated_grid_layout.on_action(UIOnActionEvent(None, "Down"))
    assert paginated_grid_layout.current_row == 1

    # Test the up button
    paginated_grid_layout.on_action(UIOnActionEvent(None, "Up"))
    assert paginated_grid_layout.current_row == 0

    # Test an invalid button
    paginated_grid_layout.on_action(UIOnActionEvent(None, "Test"))
    assert paginated_grid_layout.current_row == 0


@pytest.mark.usefixtures("window")
@pytest.mark.parametrize(
    ("diffs", "expected_current_rows", "expected_row_counts"),
    [
        ([-1], [0], [6]),
        ([0], [0], [6]),
        ([1], [1], [5]),
        ([2], [2], [2]),
        ([3], [0], [6]),
        ([2, -2], [2, 0], [2, 6]),
        ([1, 1, 1, -1, -1, -1], [1, 2, 2, 1, 0, 0], [5, 2, 2, 5, 6, 6]),
    ],
)
def test_paginated_grid_layout_navigate_rows(
    diffs: list[int],
    expected_current_rows: list[int],
    expected_row_counts: list[int],
    paginated_grid_layout: PaginatedGridLayout,
) -> None:
    """Test that the PaginatedGridLayout correctly navigates rows.

    Args:
        diffs: The number of rows to navigate.
        expected_current_rows: The expected current rows after navigating.
        expected_row_counts: The expected row counts after navigating.
        paginated_grid_layout: The PaginatedGridLayout object for testing.
    """
    for diff, expected_current_row, expected_row_count in zip(
        diffs,
        expected_current_rows,
        expected_row_counts,
        strict=True,
    ):
        paginated_grid_layout.navigate_rows(diff)
        assert paginated_grid_layout.current_row == expected_current_row
        children = paginated_grid_layout.grid_layout.children
        assert len(children) == expected_row_count
        assert (
            children
            == paginated_grid_layout.items[
                expected_current_row * 3 : expected_current_row * 3 + 3 * 2
            ]
        )


@pytest.mark.usefixtures("window")
def test_player_view_init_no_inventory(registry: Registry) -> None:
    """Test that the PlayerView raises a RegistryError when no inventory is found.

    Args:
        registry: The mock registry object for testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object `0` is not registered with the registry or does not have"
        " the required component.",
    ):
        PlayerView(
            registry,
            registry.create_game_object(GameObjectType.Player, Vec2d(0, 0), []),
            Mock(),
        )


@pytest.mark.usefixtures("window")
def test_player_view_init_invalid_game_object(registry: Registry) -> None:
    """Test that the PlayerView raises a RegistryError when the game object is invalid.

    Args:
        registry: The mock registry object for testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object `0` is not registered with the registry or does not have"
        " the required component.",
    ):
        PlayerView(registry, 0, Mock())


@pytest.mark.usefixtures("window")
def test_player_view_on_draw(monkeypatch: MonkeyPatch, player_view: PlayerView) -> None:
    """Test that the PlayerView correctly draws the player view.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        player_view: The PlayerView object for testing.
    """
    # Set up the required monkeypatches
    mock_clear = Mock()
    monkeypatch.setattr(player_view, "clear", mock_clear)
    mock_ui_manager_draw = Mock()
    monkeypatch.setattr(player_view.ui_manager, "draw", mock_ui_manager_draw)

    # Make sure the correct methods are called
    assert player_view.background_image is not None
    player_view.on_draw()
    mock_clear.assert_called_once()
    mock_ui_manager_draw.assert_called_once()


@pytest.mark.usefixtures("window")
def test_player_view_on_show_view(
    monkeypatch: MonkeyPatch,
    player_view: PlayerView,
) -> None:
    """Test that the PlayerView correctly sets up the view when shown.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        player_view: The PlayerView object for testing.
    """
    # Set up the required monkeypatches
    mock_ui_manager_enable = Mock()
    monkeypatch.setattr(player_view.ui_manager, "enable", mock_ui_manager_enable)

    # Make sure the correct methods are called
    background_image = player_view.background_image
    player_view.on_show_view()
    mock_ui_manager_enable.assert_called_once()
    assert player_view.background_image is not background_image


@pytest.mark.usefixtures("window")
def test_player_view_on_hide_view(
    monkeypatch: MonkeyPatch,
    player_view: PlayerView,
) -> None:
    """Test that the PlayerView correctly cleans up the view when hidden.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        player_view: The PlayerView object for testing.
    """
    # Set up the required monkeypatches
    mock_ui_manager_disable = Mock()
    monkeypatch.setattr(player_view.ui_manager, "disable", mock_ui_manager_disable)

    # Make sure the correct methods are called
    player_view.on_hide_view()
    mock_ui_manager_disable.assert_called_once()


@pytest.mark.usefixtures("window")
@pytest.mark.parametrize(
    ("inventory_items", "item_sprites"),
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1, 2, 3]),  # This should never happen
        ([1, 2, 3], [1, 2]),  # This should never happen
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_player_view_on_update_inventory(
    player_view: PlayerView,
    inventory_items: list[int],
    item_sprites: list[int],
) -> None:
    """Test that the PlayerView correctly updates the inventory.

    Args:
        player_view: The PlayerView object for testing.
        inventory_items: The inventory items to add to the inventory.
        item_sprites: The item sprites to add to the inventory.
    """
    # Make sure all inventory items are empty beforehand
    assert all(
        inventory_item.sprite_object is None
        for inventory_item in player_view.inventory_layout.children[0]
    )

    # Add an item and sprite to the inventory
    for item in inventory_items:
        player_view.inventory.items.append(item)
    for sprite in item_sprites:
        mock_item_sprite = Mock(spec=HadesSprite)
        mock_item_sprite.game_object_id = sprite
        player_view.item_sprites.append(mock_item_sprite)

    # Make sure the inventory item buttons are updated correctly
    player_view.on_update_inventory(0)
    grid_layout = player_view.inventory_layout.children[0]
    for i, inventory_item in enumerate(
        grid_layout.children[: min(len(inventory_items), len(item_sprites))],
    ):
        assert inventory_item.sprite_object is not None
        assert inventory_item.sprite_object.game_object_id == item_sprites[i]
    assert all(
        inventory_item.sprite_object is None
        for inventory_item in grid_layout.children[len(inventory_items) :]
    )


@pytest.mark.usefixtures("window")
def test_player_view_update_info_view(
    monkeypatch: MonkeyPatch,
    player_view: PlayerView,
) -> None:
    """Test that the PlayerView correctly updates the info view.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        player_view: The PlayerView object for testing.
    """
    # Create a mock sprite object
    mock_sprite_object = Mock(spec=HadesSprite)
    mock_sprite_object.name = "Test Item"
    mock_sprite_object.description = "Test Description"
    mock_sprite_object.texture = "Test Texture"
    mock_with_background = Mock()
    monkeypatch.setattr(UIWidget, "with_background", mock_with_background)

    # Create a mock InventoryItemButton
    mock_inventory_item_button = Mock(spec=InventoryItemButton)
    mock_inventory_item_button.sprite_object = mock_sprite_object

    # Create a mock UIOnClickEvent
    mock_event = Mock()
    mock_event.source.parent.parent = mock_inventory_item_button

    # Check if the info view is updated correctly
    player_view.update_info_view(mock_event)
    assert player_view.stats_layout.children[0].text == "Test Item"
    assert player_view.stats_layout.children[4].text == "Test Description"
    mock_with_background.assert_called_once_with(texture="Test Texture")


@pytest.mark.usefixtures("window")
def test_player_view_update_info_view_no_inventory_item(
    player_view: PlayerView,
) -> None:
    """Test that the PlayerView doesn't update the info view if no sprite is given.

    Args:
        player_view: The PlayerView object for testing.
    """
    # Create a mock UIOnClickEvent
    mock_inventory_item = Mock(spec=InventoryItemButton)
    mock_inventory_item.sprite_object = None
    mock_event = Mock()
    mock_event.source.parent.parent = mock_inventory_item

    # Check if the info view is updated correctly
    player_view.update_info_view(mock_event)
    assert player_view.stats_layout.children[0].text == "Test"


@pytest.mark.usefixtures("window")
def test_player_view_tab_menu_buttons(player_view: PlayerView) -> None:
    """Test that the PlayerView correctly sets up the tab menu buttons.

    Args:
        player_view: The PlayerView object for testing.
    """
    # Get all the required layouts
    layout = player_view.ui_manager.children[0][0].children[0].children[1]
    tab_menu = layout.children[0]
    inventory_layout = layout.children[2]

    # Test the "Upgrades" button
    tab_menu.on_action(UIOnActionEvent(None, "Upgrades"))
    assert layout.children[2] != inventory_layout
    tab_menu.on_action(UIOnActionEvent(None, "Upgrades"))
    assert layout.children[2] != inventory_layout

    # Test the "Inventory" button
    tab_menu.on_action(UIOnActionEvent(None, "Inventory"))
    assert layout.children[2] == inventory_layout
    tab_menu.on_action(UIOnActionEvent(None, "Inventory"))
    assert layout.children[2] == inventory_layout


@pytest.mark.usefixtures("window")
def test_player_view_back_button(
    monkeypatch: MonkeyPatch,
    player_view: PlayerView,
) -> None:
    """Test that the PlayerView correctly sets up the back button.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        player_view: The PlayerView object for testing.
    """
    # Set up the required monkeypatches
    mock_views = {"Game": Mock()}
    player_view.window.views = mock_views
    mock_show_view = Mock()
    monkeypatch.setattr(player_view.window, "show_view", mock_show_view)

    # Make sure the correct methods are called
    back_button = player_view.ui_manager.children[0][0].children[0].children[2]
    back_button.on_click(Mock())
    mock_show_view.assert_called_once_with(mock_views["Game"])
