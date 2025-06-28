"""Tests all classes and functions in grid_layout.py."""

from __future__ import annotations

# Builtin
from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Pip
import pytest
from arcade import Texture, get_default_texture, get_window
from arcade.gui import UIOnActionEvent, UIOnClickEvent
from PIL import Image

# Custom
from hades import SceneType
from hades.grid_layout import (
    GridView,
    ItemButton,
    PaginatedGridLayout,
    StatsLayout,
    create_default_layout,
    create_divider_line,
)
from hades_extensions.ecs import SPRITE_SIZE

if TYPE_CHECKING:

    from hades.window import HadesWindow


class TestItemButton(ItemButton):
    """An implementation of ItemButton for testing purposes."""

    def __init__(self: TestItemButton) -> None:
        """Initialise the object."""
        super().__init__(use_text="Test")
        self.texture = Texture(Image.new("RGBA", (1, 1)))

    def get_info(self: TestItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Returns:
            The name, description, and texture of the item.
        """
        return "Test Name", "Test Description", self.texture


def set_items(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
    count: int,
) -> None:
    """Set the items in the paginated grid layout.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
        count: The number of items to set in the layout.
    """
    paginated_grid_layout.items = [TestItemButton() for _ in range(count)]


def assert_counts(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
    item_button_count: int,
    default_layout_count: int,
) -> None:
    """Assert the counts of item buttons and default layouts in the grid layout.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
        item_button_count: The expected number of item buttons.
        default_layout_count: The expected number of default layouts.
    """
    counts = Counter(
        isinstance(child, TestItemButton)
        for child in paginated_grid_layout.grid_layout.children
    )
    assert counts[True] == item_button_count
    assert counts[False] == default_layout_count


@pytest.fixture
def item_button(hades_window: HadesWindow) -> ItemButton:  # noqa: ARG001
    """Create an item button for testing.

    Args:
        hades_window: The hades window for testing.

    Returns:
        A item button for testing.
    """
    return TestItemButton()


@pytest.fixture
def stats_layout(hades_window: HadesWindow) -> StatsLayout:  # noqa: ARG001
    """Create a stats layout for testing.

    Args:
        hades_window: The hades window for testing.

    Returns:
        The stats layout object for testing.
    """
    return StatsLayout()


@pytest.fixture
def paginated_grid_layout(
    hades_window: HadesWindow,  # noqa: ARG001
) -> PaginatedGridLayout[TestItemButton]:
    """Create a paginated grid layout for testing.

    Args:
        hades_window: The hades window for testing.

    Returns:
        The paginated grid layout object for testing.
    """
    return PaginatedGridLayout()


@pytest.fixture
def grid_view(
    hades_window: HadesWindow,
) -> GridView[TestItemButton]:
    """Create a grid view for testing.

    Args:
        hades_window: The hades window for testing.

    Returns:
        The grid view object for testing.
    """
    return GridView(hades_window)


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


def test_create_default_layout() -> None:
    """Test that create_default_layout() creates the correct layout."""
    layout = create_default_layout()
    assert layout.color == (198, 198, 198, 255)
    assert layout.width == SPRITE_SIZE
    assert layout.height == 90
    assert layout._border_color == (0, 0, 0, 255)  # noqa: SLF001


def test_item_button_init(item_button: ItemButton) -> None:
    """Test that the item button object initialises correctly.

    Args:
        item_button: A item button for testing.
    """
    assert len(item_button.children) == 2
    assert item_button.use_button.text == "Test"


def test_item_button_get_info(item_button: ItemButton) -> None:
    """Test that the item button correctly returns the info.

    Args:
        item_button: The item button object for testing.
    """
    name, description, texture = item_button.get_info()
    assert name == "Test Name"
    assert description == "Test Description"
    assert isinstance(texture, Texture)
    assert texture.width == 1
    assert texture.height == 1


def test_item_button_texture(item_button: ItemButton) -> None:
    """Test that the item button's texture property works correctly.

    Args:
        item_button: The item button object for testing.
    """
    assert item_button.texture_button.texture == item_button.texture
    assert item_button.texture_button.texture_hovered == item_button.texture
    assert item_button.texture_button.texture_pressed == item_button.texture


def test_item_button_texture_button_on_click(item_button: ItemButton) -> None:
    """Test that the item button's texture button on_click method works correctly.

    Args:
        item_button: The item button object for testing.
    """
    called = []

    def on_texture_button_callback(_: UIOnClickEvent) -> None:
        """A mock callback function for the texture button."""
        called.append(True)

    get_window().push_handlers(on_texture_button_callback)
    item_button.texture_button.on_click(Mock())
    get_window().dispatch_pending_events()
    assert called


def test_item_button_use_button_on_click(item_button: ItemButton) -> None:
    """Test that the item button's use button on_click method works correctly.

    Args:
        item_button: The item button object for testing.
    """
    called = []

    def on_use_button_callback(_: UIOnClickEvent) -> None:
        """A mock callback function for the use button."""
        called.append(True)

    get_window().push_handlers(on_use_button_callback)
    item_button.use_button.on_click(Mock())
    get_window().dispatch_pending_events()
    assert called


def test_stats_layout_init(stats_layout: StatsLayout) -> None:
    """Test that the stats layout initialises correctly.

    Args:
        stats_layout: The stats layout object for testing.
    """
    assert stats_layout.children[0].text == "No Item Selected"
    assert stats_layout.children[1].texture == get_default_texture()
    assert stats_layout.children[2].text == "No Item Selected"


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
    """Test that the stats layout correctly sets the info without a title.

    Args:
        stats_layout: The stats layout object for testing.
        title: The title to set for the info.
        texture: Whether to give a valid texture or not.
        description: The description to set for the info.
    """
    mock_texture = Mock(spec=Texture)
    resultant_texture = mock_texture if texture else None
    stats_layout.set_info(
        title,
        description,
        resultant_texture,  # type: ignore[arg-type]
    )
    assert stats_layout.children[0].text == title
    assert stats_layout.children[1].texture == resultant_texture
    assert stats_layout.children[2].text == description


def test_stats_layout_reset(stats_layout: StatsLayout) -> None:
    """Test that the stats layout resets correctly.

    Args:
        stats_layout: The stats layout object for testing.
    """
    stats_layout.set_info("Test", "Test description", Mock(spec=Texture))
    stats_layout.reset()
    assert stats_layout.children[0].text == "No Item Selected"
    assert stats_layout.children[1].texture == get_default_texture()
    assert stats_layout.children[2].text == "No Item Selected"


def test_paginated_grid_layout_init(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
) -> None:
    """Test that the paginated grid layout initialises correctly.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
    """
    assert paginated_grid_layout.current_row == 0
    assert paginated_grid_layout.items == []
    assert len(paginated_grid_layout.children) == 2
    assert len(paginated_grid_layout.children[1].children) == 2


@pytest.mark.usefixtures("sized_window")
@pytest.mark.parametrize(
    ("sized_window", "counts"),
    [
        ((320, 240), (1, 1)),
        ((640, 480), (2, 2)),
        ((1280, 720), (5, 3)),
        ((1920, 1080), (7, 5)),
        ((2560, 1440), (9, 6)),
        ((3840, 2160), (14, 9)),
        ((7680, 4320), (28, 18)),
    ],
    indirect=["sized_window"],
)
def test_paginated_grid_layout_init_window_size(counts: tuple[int, int]) -> None:
    """Test that the paginated grid layout handles different window sizes correctly.

    Args:
        counts: The expected column and row counts.
    """
    paginated_grid_layout = PaginatedGridLayout[TestItemButton]()
    assert paginated_grid_layout.grid_layout.column_count == counts[0]
    assert paginated_grid_layout.grid_layout.row_count == counts[1]


def test_paginated_grid_layout_on_action(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
) -> None:
    """Test that the paginated grid layout correctly calls the on_action method.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
    """
    # Test the down button
    set_items(paginated_grid_layout, 50)
    paginated_grid_layout.on_action(UIOnActionEvent(None, "Down"))
    assert paginated_grid_layout.current_row == 1

    # Test the up button
    paginated_grid_layout.on_action(UIOnActionEvent(None, "Up"))
    assert paginated_grid_layout.current_row == 0

    # Test an invalid button
    paginated_grid_layout.on_action(UIOnActionEvent(None, "Test"))
    assert paginated_grid_layout.current_row == 0


@pytest.mark.parametrize(
    ("total_size", "item_button_count", "default_layout_count"),
    [
        (0, 0, 15),
        (1, 1, 14),
        (2, 2, 13),
        (5, 5, 10),
        (10, 10, 5),
        (14, 14, 1),
        (15, 15, 0),
        (25, 15, 0),
    ],
)
def test_paginated_grid_layout_set_items(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
    total_size: int,
    item_button_count: int,
    default_layout_count: int,
) -> None:
    """Test that the paginated grid layout correctly sets the items.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
        total_size: The total size of the grid layout.
        item_button_count: The expected number of item buttons.
        default_layout_count: The expected number of default layouts.
    """
    set_items(paginated_grid_layout, total_size)
    assert_counts(paginated_grid_layout, item_button_count, default_layout_count)


def test_paginated_grid_layout_add_items_reset_current_row(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
) -> None:
    """Test that the paginated grid layout resets the current row when adding items.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
    """
    paginated_grid_layout.items = [TestItemButton() for _ in range(25)]
    paginated_grid_layout.navigate_rows(1)
    assert paginated_grid_layout.current_row == 1
    set_items(paginated_grid_layout, 50)
    assert paginated_grid_layout.current_row == 0


def test_paginated_grid_layout_add_item(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
) -> None:
    """Test that the paginated grid layout correctly adds an item.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
    """
    paginated_grid_layout.items = [TestItemButton() for _ in range(10)]
    assert_counts(paginated_grid_layout, 10, 5)
    paginated_grid_layout.add_item(TestItemButton())
    assert_counts(paginated_grid_layout, 11, 4)


@pytest.mark.parametrize(
    ("diffs", "expected_current_rows", "expected_row_counts"),
    [
        ([-1], [0], [15]),
        ([0], [0], [15]),
        ([1], [1], [15]),
        ([2], [2], [15]),
        ([3], [3], [10]),
        ([2, -2], [2, 0], [15, 15]),
        (
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 2, 3, 3, 2, 1, 0, 0],
            [15, 15, 10, 10, 15, 15, 15, 15],
        ),
    ],
)
def test_paginated_grid_layout_navigate_rows(
    paginated_grid_layout: PaginatedGridLayout[TestItemButton],
    diffs: list[int],
    expected_current_rows: list[int],
    expected_row_counts: list[int],
) -> None:
    """Test that the paginated grid layout correctly navigates rows.

    Args:
        paginated_grid_layout: The paginated grid layout object for testing.
        diffs: The number of rows to navigate.
        expected_current_rows: The expected current rows after navigating.
        expected_row_counts: The expected row counts after navigating.
    """
    set_items(paginated_grid_layout, 25)
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


def test_grid_view_init(grid_view: GridView[TestItemButton]) -> None:
    """Test that the grid view initialises correctly.

    Args:
        grid_view: The grid view object for testing.
    """
    assert grid_view.grid_layout is not None
    assert len(grid_view.ui.children[0]) == 2


def test_grid_view_back_button_on_click(grid_view: GridView[TestItemButton]) -> None:
    """Test that the grid view back button works correctly.

    Args:
        grid_view: The grid view object for testing.
    """
    mock_show_view = Mock()
    grid_view.window.show_view = mock_show_view  # type: ignore[method-assign]
    back_button = grid_view.ui.children[0][1].children[0].children[1]
    back_button.on_click(UIOnClickEvent(Mock(), -1, -1, -1, -1))
    mock_show_view.assert_called_once_with(grid_view.window.scenes[SceneType.GAME])
