# pylint: disable=redefined-outer-name
"""Tests all classes and functions in progress_bar.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import pytest
from arcade import color

# Custom
from hades.progress_bar import ProgressBar

if TYPE_CHECKING:
    from arcade.types.color import RGBA255

__all__ = ()


@pytest.mark.parametrize(
    ("init_data", "expected_width", "expected_height"),
    [
        (((0.5, 0.5), color.GREEN, 0), 29, 9),
        (((1, 1), color.YELLOW, 0), 58, 18),
        (((1, 1), color.YELLOW, 1), 58, 18),
        (((2, 2), color.GREEN, 0), 116, 36),
    ],
)
def test_progress_bar_init(
    init_data: tuple[tuple[float, float], RGBA255, int],
    expected_width: int,
    expected_height: int,
) -> None:
    """Test that a progress bar initialises correctly.

    Args:
        init_data: The data to initialise the progress bar with.
        expected_width: The expected width of the progress bar
        expected_height: The expected height of the progress bar
    """
    progress_bar = ProgressBar(*init_data)
    assert progress_bar.width == expected_width
    assert progress_bar.height == expected_height
    assert progress_bar.order == init_data[2]
    assert progress_bar.size_hint is None
    assert progress_bar.actual_bar.color == init_data[1]
    assert progress_bar.actual_bar in progress_bar.children
    assert progress_bar.actual_bar.size_hint == (1, 1)


@pytest.mark.parametrize("scale", [(0, 0), (-1, -1)])
def test_progress_bar_init_invalid_scale(scale: tuple[float, float]) -> None:
    """Test that a progress bar with an invalid scale raises an error.

    Args:
        scale: The scale to initialise the progress bar with.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="Scale must be greater than 0",
    ):
        ProgressBar(scale, color.GREEN, 0)


def test_progress_bar_init_invalid_order() -> None:
    """Test that a progress bar with an invalid order raises an error."""
    with pytest.raises(
        expected_exception=ValueError,
        match="Order must be greater than or equal to 0",
    ):
        ProgressBar((1, 1), color.GREEN, -1)
