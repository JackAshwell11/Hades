"""Tests all classes and functions in player/view.py."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades.player.view import create_divider_line


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
    assert item_button.children == [item_button.default_layout]
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
        match=r".*missing 1 required positional argument: '__'.*",
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
        match=r"No sprite object set for this item.",
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
        match=r"No sprite object set for this item.",
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
    assert len(upgrades_item_button.target_functions) == 2  # type: ignore[arg-type]
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
        "",
        get_default_texture(),
    )
    upgrades_item_button.update_description()
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
def test_upgrades_item_button_no_targets(
    mock_callback: Callable[[UIOnClickEvent], None],
) -> None:
    """Test that the UpgradesItemButton raises errors for missing target components.

    Args:
        mock_callback: The mock callback function to use for testing.
    """
    upgrades_item_button = UpgradesItemButton(
        mock_callback,
        target_component=None,  # type: ignore[arg-type]
        target_functions=None,  # type: ignore[arg-type]
    )
    upgrades_item_button.update_description()
    assert not upgrades_item_button.description
    with pytest.raises(
        expected_exception=ValueError,
        match=r"No target component or functions set for this item.",
    ):
        upgrades_item_button.get_info()
    with pytest.raises(
        expected_exception=ValueError,
        match=r"No target component or functions set for this item.",
    ):
        upgrades_item_button.use()
