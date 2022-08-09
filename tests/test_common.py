"""Tests all functions in common.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from game.common import grid_bfs
from game.generation.primitives import Point

__all__ = ()


def test_grid_bfs() -> None:
    """Test the grid_bfs function in common.py."""
    custom_offsets = [
        (-1, -1),
        (1, -1),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]
    assert list(grid_bfs((5, 5), 10, 10)) == [(5, 4), (4, 5), (6, 5), (5, 6)]
    assert list(grid_bfs((5, 5), 10, 10, offsets=custom_offsets)) == [
        (4, 4),
        (6, 4),
        (6, 5),
        (4, 6),
        (5, 6),
        (6, 6),
    ]
    result = list(grid_bfs((5, 5), 10, 10, return_point=True))
    assert result == [(5, 4), (4, 5), (6, 5), (5, 6)] and all(
        [isinstance(i, Point) for i in result]
    )
    assert list(grid_bfs((0, 0), 10, 10)) == [(1, 0), (0, 1)]
    assert list(grid_bfs((9, 9), 10, 10)) == [(9, 8), (8, 9)]
    with pytest.raises(TypeError):
        list(grid_bfs(("test", "test"), 10, 10))
