# pylint: disable=redefined-outer-name, unused-argument
"""Tests all classes and functions in views/player.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

# Pip
import pytest
from arcade import Texture, get_default_texture, get_window
from arcade.gui import UIOnActionEvent, UITextureButton

# Custom
from hades.sprite import HadesSprite
from hades.views.player import (
    InventoryItemButton,
    ItemButton,
    ItemButtonKwargs,
    PaginatedGridLayout,
    PlayerAttributesLayout,
    PlayerView,
    StatsLayout,
    UpgradesItemButton,
    create_divider_line,
)
from hades_extensions.game_objects import (
    ComponentBase,
    GameObjectType,
    Registry,
    RegistryError,
    Vec2d,
)
from hades_extensions.game_objects.components import (
    Health,
    Inventory,
    InventorySize,
    Money,
    PythonSprite,
    StatusEffect,
    Upgrades,
)
from hades_extensions.game_objects.systems import InventorySystem, UpgradeSystem

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch
    from arcade import Window
    from arcade.gui import UIOnClickEvent


@pytest.fixture
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


@pytest.fixture
def mock_sprite() -> HadesSprite:
    """Create a mock HadesSprite object for testing.

    Returns:
        The mock HadesSprite object for testing.
    """
    mock_sprite = Mock(spec=HadesSprite)
    mock_sprite.game_object_id = 1
    mock_sprite.name = "Test Item"
    mock_sprite.description = "Test Description"
    mock_sprite.texture = "Test Texture"
    return mock_sprite


@pytest.fixture
def component_sprite(registry: Registry) -> int:
    """Create a HadesSprite object with components for testing.

    Args:
        registry: The registry for testing.

    Returns:
        The game object ID of the HadesSprite object.
    """
    python_sprite = PythonSprite()
    game_object_id = registry.create_game_object(
        GameObjectType.Player,
        Vec2d(0, 0),
        [
            Health(100, -1),
            Inventory(),
            InventorySize(10, -1),
            Money(),
            StatusEffect(),
            Upgrades({}),
            python_sprite,
        ],
    )
    python_sprite.sprite = Mock(spec=HadesSprite)
    return game_object_id


@pytest.fixture
def item_button(
    window: Window,  # noqa: ARG001
    mock_callback: Callable[[UIOnClickEvent], None],
) -> ItemButton:
    """Create an ItemButton for testing.

    Args:
        window: The window for testing.
        mock_callback: The mock callback function to use for testing.

    Returns:
        The ItemButton object for testing.
    """

    class CompleteItemButton(ItemButton):
        """A subclass of ItemButton that mocks the texture button."""

        def get_info(self: CompleteItemButton) -> tuple[str, str, Texture]:
            """Get the information about the item.

            Returns:
                The name, description, and texture of the item.
            """
            return "", "", self.texture_button.texture

        def use(self: CompleteItemButton) -> None:
            """Use the item."""

    return CompleteItemButton(mock_callback)


@pytest.fixture
def inventory_item_button(
    window: Window,  # noqa: ARG001
    mock_callback: Callable[[UIOnClickEvent], None],
) -> InventoryItemButton:
    """Create an InventoryItemButton for testing.

    Args:
        window: The window for testing
        mock_callback: The mock callback function to use for testing.

    Returns:
        The InventoryItemButton object for testing.
    """
    return InventoryItemButton(mock_callback)


@pytest.fixture
def upgrades_item_button(
    window: Window,  # noqa: ARG001
    mock_callback: Callable[[UIOnClickEvent], None],
) -> UpgradesItemButton:
    """Create an UpgradesItemButton for testing.

    Args:
        window: The window for testing
        mock_callback: The mock callback function to use for testing.

    Returns:
        The UpgradesItemButton object for testing.
    """
    functions = (lambda level: level + 5, lambda level: level + 10)
    return UpgradesItemButton(
        mock_callback,
        target_component=Health,
        target_functions=functions,
    )


@pytest.fixture
def paginated_grid_layout(window: Window) -> PaginatedGridLayout:  # noqa: ARG001
    """Create a PaginatedGridLayout for testing.

    Args:
        window: The window for testing.

    Returns:
        The PaginatedGridLayout object for testing.
    """
    return PaginatedGridLayout(25, InventoryItemButton)


@pytest.fixture
def stats_layout(window: Window) -> StatsLayout:  # noqa: ARG001
    """Create a StatsLayout for testing.

    Args:
        window: The window for testing.

    Returns:
        The StatsLayout object for testing.
    """
    return StatsLayout()


@pytest.fixture
def player_attributes_layout(
    registry: Registry,
    window: Window,  # noqa: ARG001
    component_sprite: int,
) -> PlayerAttributesLayout:
    """Create a PlayerAttributesLayout for testing.

    Args:
        registry: The registry for testing.
        window: The window for testing.
        component_sprite: The game object ID of the HadesSprite object.

    Returns:
        The PlayerAttributesLayout object for testing.
    """
    return PlayerAttributesLayout(registry, component_sprite)


@pytest.fixture
def player_view(
    registry: Registry,
    window: Window,  # noqa: ARG001
    component_sprite: int,
) -> PlayerView:
    """Create a PlayerView for testing.

    Args:
        registry: The registry for testing.
        window: The window for testing.
        component_sprite: The game object ID of the HadesSprite object.

    Returns:
        The PlayerView object for testing.
    """
    player_view = PlayerView(
        registry,
        component_sprite,
        [],  # type: ignore[arg-type]
    )
    get_window().show_view(player_view)
    return player_view


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


def test_item_button_init(
    mock_callback: Callable[[UIOnClickEvent], None],
    item_button: ItemButton,
) -> None:
    """Test that the ItemButton initialises correctly.

    Args:
        mock_callback: The mock callback function to use for testing.
        item_button: The ItemButton object for testing.
    """
    assert item_button.texture_button is not None
    assert item_button.texture_button.on_click == mock_callback
    assert item_button.children == [item_button.sprite_layout]
    for child in item_button.sprite_layout.children:
        assert child.on_click == mock_callback


def test_item_button_on_click(
    mock_callback: Callable[[UIOnClickEvent], None],
    item_button: ItemButton,
) -> None:
    """Test that the ItemButton correctly calls the callback function.

    Args:
        mock_callback: The mock callback function to use for
        item_button: The ItemButton object for testing.
    """
    item_button.texture_button.on_click(None)  # type: ignore[arg-type]
    assert mock_callback.called  # type: ignore[attr-defined]


def test_item_button_on_click_invalid_callback(item_button: ItemButton) -> None:
    """Test that the InventoryItemButton raises a TypeError for an invalid callback.

    Args:
        item_button: The ItemButton object for testing.
    """

    def invalid_callback(_: str, __: int) -> None:
        """An invalid callback function that takes too many arguments."""

    item_button.texture_button.on_click = invalid_callback  # type: ignore[assignment]
    with pytest.raises(
        expected_exception=TypeError,
        match=".*missing 1 required positional argument: '__'.*",
    ):
        item_button.texture_button.on_click(None)  # type: ignore[arg-type]


def test_inventory_item_button_init(inventory_item_button: InventoryItemButton) -> None:
    """Test that the InventoryItemButton initialises correctly.

    Args:
        inventory_item_button: The InventoryItemButton object for testing.
    """
    assert inventory_item_button.sprite_object is None
    assert inventory_item_button.children == [inventory_item_button.default_layout]


def test_inventory_item_button_set_sprite(
    mock_sprite: HadesSprite,
    inventory_item_button: InventoryItemButton,
) -> None:
    """Test that the InventoryItemButton correctly sets the sprite object.

    Args:
        mock_sprite: The mock HadesSprite object for testing.
        inventory_item_button: The InventoryItemButton object for testing.
    """
    # Test setting the sprite object changes to the sprite layout
    for _ in range(2):
        inventory_item_button.sprite_object = mock_sprite
        assert inventory_item_button.sprite_object == mock_sprite
        assert inventory_item_button.texture_button.texture == "Test Texture"
        assert inventory_item_button.texture_button.texture_hovered == "Test Texture"
        assert inventory_item_button.texture_button.texture_pressed == "Test Texture"
        assert inventory_item_button.children == [inventory_item_button.sprite_layout]

    # Test setting the sprite object to None changes back to the default layout
    for _ in range(2):
        inventory_item_button.sprite_object = None
        assert inventory_item_button.sprite_object is None
        assert inventory_item_button.children == [inventory_item_button.default_layout]


def test_inventory_item_button_get_info_no_sprite_object(
    inventory_item_button: InventoryItemButton,
) -> None:
    """Test that getting info from an InventoryItemButton raises an error.

    Args:
        inventory_item_button: The InventoryItemButton object for testing.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="No sprite object set for this item.",
    ):
        inventory_item_button.get_info()


def test_inventory_item_button_get_info_set_sprite_object(
    mock_sprite: HadesSprite,
    inventory_item_button: InventoryItemButton,
) -> None:
    """Test that getting info from an InventoryItemButton returns the correct values.

    Args:
        mock_sprite: The mock HadesSprite object for testing.
        inventory_item_button: The InventoryItemButton object for
    """
    inventory_item_button.sprite_object = mock_sprite
    assert inventory_item_button.get_info() == (  # type: ignore[comparison-overlap]
        "Test Item",
        "Test Description",
        "Test Texture",
    )


def test_inventory_item_button_use_no_sprite_object(
    inventory_item_button: InventoryItemButton,
) -> None:
    """Test that using an InventoryItemButton raises an error.

    Args:
        inventory_item_button: The InventoryItemButton object for testing.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="No sprite object set for this item.",
    ):
        inventory_item_button.use()


def test_inventory_item_button_use_set_sprite_object(
    mock_sprite: HadesSprite,
    inventory_item_button: InventoryItemButton,
    player_view: PlayerView,
) -> None:
    """Test that using an InventoryItemButton correctly calls the use callback.

    Args:
        mock_sprite: The mock HadesSprite object for testing.
        inventory_item_button: The InventoryItemButton object for testing.
        player_view: The PlayerView object for testing.
    """
    # Create the mock objects
    mock_registry = Mock(spec=Registry)
    mock_inventory_system = Mock(spec=InventorySystem)

    # Set up the mock return values
    player_view.registry = mock_registry
    mock_registry.get_system.return_value = mock_inventory_system

    # Update the sprite object and use the item
    inventory_item_button.sprite_object = mock_sprite
    inventory_item_button.use()
    mock_inventory_system.use_item.assert_called_once_with(0, 1)


def test_upgrades_item_button_init(upgrades_item_button: UpgradesItemButton) -> None:
    """Test that the UpgradesItemButton initialises correctly.

    Args:
        upgrades_item_button: The UpgradesItemButton object for testing.
    """
    assert upgrades_item_button.target_component == Health
    assert len(upgrades_item_button.target_functions) == 2
    assert not upgrades_item_button.description


@pytest.mark.parametrize(
    ("new_values", "expected_description", "increment_level"),
    [
        ((0, 0), "Max Health:\n  100.0 -> 105.0\nMoney:\n  0 -> -10", False),
        ((0, 0), "Max Health:\n  100.0 -> 106.0\nMoney:\n  0 -> -11", True),
        ((100, 0), "Max Health:\n  200.0 -> 205.0\nMoney:\n  0 -> -10", False),
        ((100, 0), "Max Health:\n  200.0 -> 206.0\nMoney:\n  0 -> -11", True),
        ((0, 10), "Max Health:\n  100.0 -> 105.0\nMoney:\n  10 -> 0", False),
        ((0, 10), "Max Health:\n  100.0 -> 106.0\nMoney:\n  10 -> -1", True),
        ((100, 10), "Max Health:\n  200.0 -> 205.0\nMoney:\n  10 -> 0", False),
        ((100, 10), "Max Health:\n  200.0 -> 206.0\nMoney:\n  10 -> -1", True),
    ],
)
def test_upgrades_item_button_update_description(
    upgrades_item_button: UpgradesItemButton,
    player_view: PlayerView,
    new_values: tuple[float, float],
    expected_description: str,
    *,
    increment_level: bool,
) -> None:
    """Test that the UpgradesItemButton correctly updates the description.

    Args:
        upgrades_item_button: The UpgradesItemButton object for testing.
        player_view: The PlayerView object for testing.
        new_values: The new values to set for the components.
        increment_level: Whether to increment the current level of the component or not.
        expected_description: The expected description after updating.
    """
    assert not upgrades_item_button.description
    player_view.registry.get_component(0, Health).add_to_max_value(new_values[0])
    money = player_view.registry.get_component(0, Money)
    money.money = new_values[1]  # type: ignore[assignment]
    if increment_level:
        player_view.registry.get_component(0, Health).increment_current_level()
    upgrades_item_button.update_description()
    assert upgrades_item_button.description == expected_description


@pytest.mark.usefixtures("player_view")
def test_upgrades_item_button_get_info(
    upgrades_item_button: UpgradesItemButton,
) -> None:
    """Test that the UpgradesItemButton correctly returns the info.

    Args:
        upgrades_item_button: The UpgradesItemButton object for testing.
    """
    assert upgrades_item_button.get_info() == (
        "Health",
        "Max Health:\n  100.0 -> 105.0\nMoney:\n  0 -> -10",
        get_default_texture(),
    )


def test_upgrades_item_button_use(
    upgrades_item_button: UpgradesItemButton,
    player_view: PlayerView,
) -> None:
    """Test that the UpgradesItemButton correctly uses the item.

    Args:
        upgrades_item_button: The UpgradesItemButton object for testing.
        player_view: The PlayerView object for testing.
    """
    # Create the mock objects
    mock_registry = Mock(spec=Registry)
    mock_upgrade_system = Mock(spec=UpgradeSystem)
    mock_stats_layout = Mock(spec=StatsLayout)
    mock_get_info = Mock()

    # Set up the mock return values
    player_view.registry = mock_registry
    player_view.stats_layout = mock_stats_layout
    mock_registry.get_system.return_value = mock_upgrade_system
    upgrades_item_button.get_info = mock_get_info  # type: ignore[method-assign]
    mock_get_info.return_value = ("Test Info",)

    # Use the item
    upgrades_item_button.use()
    mock_upgrade_system.upgrade_component.assert_called_once_with(0, Health)
    mock_stats_layout.set_info.assert_called_once_with("Test Info")


@pytest.mark.usefixtures("window")
@pytest.mark.parametrize(
    ("total_size", "expected_size", "button_type", "button_params"),
    [
        (0, 14, InventoryItemButton, None),
        (1, 14, InventoryItemButton, None),
        (2, 14, InventoryItemButton, None),
        (5, 14, InventoryItemButton, None),
        (10, 14, InventoryItemButton, None),
        (20, 20, InventoryItemButton, None),
        (35, 35, InventoryItemButton, None),
        (50, 50, InventoryItemButton, None),
        (1, 14, InventoryItemButton, [{}]),
        (
            1,
            14,
            UpgradesItemButton,
            [
                {
                    "target_component": Health,
                    "target_functions": (
                        lambda level: level + 5,
                        lambda level: level + 10,
                    ),
                },
            ],
        ),
        (1, 14, UpgradesItemButton, None),
    ],
)
def test_paginated_grid_layout_init(
    total_size: int,
    expected_size: int,
    button_type: type[ItemButton],
    button_params: list[ItemButtonKwargs] | None,
) -> None:
    """Test that the PaginatedGridLayout initialises correctly.

    Args:
        total_size: The total size of the grid layout.
        expected_size: The expected size of the grid layout.
        button_type: The type of button to use for the grid layout.
        button_params: The parameters to pass to the button type.
    """
    if button_type is UpgradesItemButton and button_params is None:
        with pytest.raises(
            expected_exception=KeyError,
            match="'target_component'",
        ):
            PaginatedGridLayout(total_size, button_type, button_params)
        return
    paginated_grid_layout = PaginatedGridLayout(total_size, button_type, button_params)
    assert paginated_grid_layout.total_size == expected_size
    assert paginated_grid_layout.current_row == 0
    assert [button.text for button in paginated_grid_layout.button_layout.children] == [
        "Up",
        "Down",
    ]
    assert len(paginated_grid_layout.items) == total_size
    assert all(isinstance(item, button_type) for item in paginated_grid_layout.items)


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


def test_paginated_grid_layout_item_clicked_texture_button(
    paginated_grid_layout: PaginatedGridLayout,
    player_view: PlayerView,
) -> None:
    """Test that the PaginatedGridLayout correctly calls the item clicked method.

    Args:
        paginated_grid_layout: The PaginatedGridLayout object for testing.
        player_view: The PlayerView object for testing.
    """
    # Create the mock objects
    mock_item = Mock(spec=ItemButton)
    mock_parent = Mock()
    texture_button = UITextureButton(width=1, height=1)
    mock_event = Mock()
    mock_stats_layout = Mock(spec=StatsLayout)

    # Set up the mock return values
    mock_item.get_info.return_value = ("Test", "Test", get_default_texture())
    mock_parent.parent = mock_item
    texture_button.parent = mock_parent
    mock_event.source = texture_button
    player_view.stats_layout = mock_stats_layout

    # Test the item clicked method
    paginated_grid_layout.item_clicked(mock_event)
    mock_stats_layout.set_info.assert_called_once_with(
        "Test",
        "Test",
        get_default_texture(),
    )


@pytest.mark.usefixtures("player_view")
def test_paginated_grid_layout_item_clicked_use_button(
    paginated_grid_layout: PaginatedGridLayout,
) -> None:
    """Test that the PaginatedGridLayout correctly calls the item clicked method.

    Args:
        paginated_grid_layout: The PaginatedGridLayout object for testing.
    """
    # Create the mock objects
    mock_item = Mock(spec=ItemButton)
    mock_event = Mock()

    # Set up the mock return values
    mock_event.source.parent.parent = mock_item

    # Test the item clicked method
    paginated_grid_layout.item_clicked(mock_event)
    mock_item.use.assert_called_once_with()


@pytest.mark.parametrize(
    ("diffs", "expected_current_rows", "expected_row_counts"),
    [
        ([-1], [0], [14]),
        ([0], [0], [14]),
        ([1], [1], [14]),
        ([2], [2], [11]),
        ([3], [0], [14]),
        ([2, -2], [2, 0], [11, 14]),
        ([1, 1, 1, -1, -1, -1], [1, 2, 2, 1, 0, 0], [14, 11, 11, 14, 14, 14]),
    ],
)
def test_paginated_grid_layout_navigate_rows(
    paginated_grid_layout: PaginatedGridLayout,
    diffs: list[int],
    expected_current_rows: list[int],
    expected_row_counts: list[int],
) -> None:
    """Test that the PaginatedGridLayout correctly navigates rows.

    Args:
        paginated_grid_layout: The PaginatedGridLayout object for testing.
        diffs: The number of rows to navigate.
        expected_current_rows: The expected current rows after navigating.
        expected_row_counts: The expected row counts after navigating.
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
        grid_column_count, grid_row_count = (
            paginated_grid_layout.grid_layout.column_count,
            paginated_grid_layout.grid_layout.row_count,
        )
        assert (
            children
            == paginated_grid_layout.items[
                expected_current_row
                * grid_column_count : expected_current_row
                * grid_column_count
                + grid_column_count * grid_row_count
            ]
        )


def test_stats_layout_init(stats_layout: StatsLayout) -> None:
    """Test that the StatsLayout initialises correctly.

    Args:
        stats_layout: The StatsLayout object for testing.
    """
    assert stats_layout.width == 384
    assert stats_layout.height == 288
    assert stats_layout.children[0].text == "Test"
    description = stats_layout.children[4]
    assert description.text == "Test description"
    assert description.label.multiline


@pytest.mark.parametrize(
    ("title", "description", "texture"),
    [
        ("Test", "Test description", True),
        ("", "Test description", True),
        ("Test", "", True),
        ("Test", "Test description", False),
        ("", "", False),
    ],
)
def test_stats_layout_set_info_no_title(
    stats_layout: StatsLayout,
    title: str,
    description: str,
    *,
    texture: bool,
) -> None:
    """Test that the StatsLayout correctly sets the info without a title.

    Args:
        stats_layout: The StatsLayout object for testing.
        title: The title to set for the info.
        texture: Whether to give a valid texture or not.
        description: The description to set for the info.
    """
    # Create a mock texture
    mock_texture = Mock(spec=Texture)

    # Test setting the info
    resultant_texture = mock_texture if texture else None
    stats_layout.set_info(
        title,
        description,
        resultant_texture,  # type: ignore[arg-type]
    )
    assert stats_layout.children[0].text == title
    assert stats_layout.children[4].text == description


@pytest.mark.parametrize(
    "components",
    [
        [InventorySize(0, -1), Upgrades({})],
        [
            InventorySize(10, -1),
            Upgrades({Health: (lambda level: level * 5, lambda level: level * 10)}),
        ],
    ],
)
def test_player_attributes_layout_init(
    registry: Registry,
    components: list[ComponentBase],
) -> None:
    """Test that the PlayerAttributesLayout initialises correctly.

    Args:
        registry: The registry for testing.
        components: The components to use for testing.
    """
    # Make sure the width and height are correct
    player_attributes_layout = PlayerAttributesLayout(
        registry,
        registry.create_game_object(GameObjectType.Player, Vec2d(0, 0), components),
    )
    assert player_attributes_layout.width == 1024
    assert player_attributes_layout.height == 360

    # Make sure the layout is correct
    assert [child.text for child in player_attributes_layout.children[0].children] == [
        "Inventory",
        "Upgrades",
    ]
    assert (
        player_attributes_layout.inventory_layout in player_attributes_layout.children
    )
    assert (
        player_attributes_layout.upgrades_layout
        not in player_attributes_layout.children
    )

    # Make sure the layouts have the minimum size
    assert player_attributes_layout.inventory_layout.total_size == 14
    assert player_attributes_layout.upgrades_layout.total_size == 14

    # Make sure the upgrades layout is correct
    for upgrades_item_button, component in zip(
        player_attributes_layout.upgrades_layout.items,
        cast(Upgrades, components[1]).upgrades.keys(),
    ):
        assert upgrades_item_button.target_component == component


@pytest.mark.parametrize(
    ("components", "expected_error", "expected_error_message"),
    [
        (
            [Upgrades({})],
            RegistryError,
            "The component `InventorySize` for the game object ID `0` is not registered"
            " with the registry.",
        ),
        (
            [InventorySize(0, -1)],
            RegistryError,
            "The component `Upgrades` for the game object ID `0` is not registered with"
            " the registry.",
        ),
    ],
)
def test_player_attributes_layout_init_errors(
    registry: Registry,
    components: list[ComponentBase],
    expected_error: type[Exception],
    expected_error_message: str,
) -> None:
    """Test that the PlayerAttributesLayout raises the correct errors.

    Args:
        registry: The registry for testing.
        components: The components to use for testing.
        expected_error: The expected error to raise.
        expected_error_message: The expected error message to raise.
    """
    with pytest.raises(expected_error, match=expected_error_message):
        PlayerAttributesLayout(
            registry,
            registry.create_game_object(GameObjectType.Player, Vec2d(0, 0), components),
        )


def test_player_attributes_layout_on_action(
    player_attributes_layout: PlayerAttributesLayout,
) -> None:
    """Test that the PlayerAttributesLayout on_action method works correctly.

    Args:
        player_attributes_layout: The PlayerAttributesLayout object for testing.
    """
    # Check that the default layout is the stats layout
    assert (
        player_attributes_layout.children[2]
        == player_attributes_layout.inventory_layout
    )

    # Test the "Upgrades" button
    player_attributes_layout.on_action(UIOnActionEvent(None, "Upgrades"))
    assert (
        player_attributes_layout.children[2] == player_attributes_layout.upgrades_layout
    )
    player_attributes_layout.on_action(UIOnActionEvent(None, "Upgrades"))
    assert (
        player_attributes_layout.children[2] == player_attributes_layout.upgrades_layout
    )

    # Test the "Inventory" button
    player_attributes_layout.on_action(UIOnActionEvent(None, "Inventory"))
    assert (
        player_attributes_layout.children[2]
        == player_attributes_layout.inventory_layout
    )
    player_attributes_layout.on_action(UIOnActionEvent(None, "Inventory"))
    assert (
        player_attributes_layout.children[2]
        == player_attributes_layout.inventory_layout
    )


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


@pytest.mark.parametrize(
    "items",
    [
        [],
        [1],
        [1, 2],
        [1, 2, 3],
    ],
)
def test_player_view_on_update_inventory(
    player_view: PlayerView,
    items: list[int],
) -> None:
    """Test that the PlayerView correctly updates the inventory.

    Args:
        player_view: The PlayerView object for testing.
        items: The items to add to the inventory.
    """

    def get_inventory_count() -> int:
        """Get the number of items in the inventory.

        Returns:
            The number of items in the inventory.
        """
        inventory_items = player_view.player_attributes_layout.inventory_layout.items
        return sum(
            1
            for inventory_item in inventory_items
            if inventory_item.sprite_object is not None
        )

    # Add an item to the inventory
    sprites = []
    for item in items:
        player_view.registry.get_system(InventorySystem).add_item_to_inventory(
            0,
            item,
        )
        python_sprite = PythonSprite()
        player_view.registry.create_game_object(
            GameObjectType.Player,
            Vec2d(0, 0),
            [python_sprite],
        )
        mock_item_sprite = Mock(spec=HadesSprite)
        python_sprite.sprite = mock_item_sprite

        # Prevent garbage collection of the mock sprite due to the PythonSprite using a
        # pybind11::handle which is not reference counted as this would cause a segfault
        sprites.append(mock_item_sprite)

    # Make sure the inventory item buttons are updated correctly
    player_view.on_update_inventory(0)
    assert get_inventory_count() == len(items)

    # Remove an item from the inventory
    player_view.registry.get_system(InventorySystem).remove_item_from_inventory(0, 1)
    player_view.on_update_inventory(0)
    assert get_inventory_count() == max(len(items) - 1, 0)


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
