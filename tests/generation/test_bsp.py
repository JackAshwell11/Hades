"""Tests all functions in generation/bsp.py."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from game.generation.bsp import Leaf
from game.generation.primitives import Point

if TYPE_CHECKING:
    import numpy as np

__all__ = ()


def test_bsp_init(
    boundary_point: Point, invalid_point: Point, grid: np.ndarray
) -> None:
    """Test the initialisation of the Leaf class in bsp.py.

    Parameters
    ----------
    boundary_point: Point
        A boundary point used for testing.
    invalid_point: Point
        An invalid point used for testing.
    grid: np.ndarray
        The 2D grid used for testing.
    """
    assert (
        repr(Leaf(boundary_point, Point(grid.shape[1], grid.shape[0]), grid))
        == "<Leaf (Left=None) (Right=None) (Top-left position=Point(x=0, y=0))"
        " (Bottom-right position=Point(x=50, y=50))>"
    )
    assert (
        repr(Leaf(invalid_point, Point(grid.shape[1], grid.shape[0]), grid))
        == "<Leaf (Left=None) (Right=None) (Top-left position=Point(x='test',"
        " y='test')) (Bottom-right position=Point(x=50, y=50))>"
    )


def test_leaf_split(leaf: Leaf) -> None:
    """Test the split function in the Leaf class.

    Parameters
    ----------
    leaf: Leaf
        The leaf used for testing.
    """
    # Repeat until we have tested the vertical and horizontal split routes
    test_vertical = test_horizontal = False
    while not test_vertical or not test_horizontal:
        leaf.split()
        if leaf.split_vertical:
            test_vertical = True
        else:
            test_horizontal = True
        assert (
            leaf.left is not None
            and leaf.right is not None
            and isinstance(leaf.split_vertical, bool)
        )
        leaf.left = leaf.right = leaf.split_vertical = None


def test_leaf_create_room(leaf: Leaf) -> None:
    """Test the create_room function in the Leaf class.

    Parameters
    ----------
    leaf: Leaf
        The leaf used for testing.
    """
    # Repeat until a room is created since the ratio may be wrong sometimes
    result = leaf.create_room()
    while not result:
        result = leaf.create_room()
    assert result and leaf.room

    # Make sure we test what happens if the leaf's left and right nodes are not None
    leaf.left = leaf.right = "test"
    assert not leaf.create_room()
