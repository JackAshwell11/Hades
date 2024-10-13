# pylint: disable=redefined-outer-name
"""Tests all classes and functions in sprite.py."""

from __future__ import annotations

# Builtin
from pathlib import Path
from unittest.mock import Mock

# Pip
import pytest

# Custom
from hades.constructors import GameObjectConstructor
from hades.sprite import AnimatedSprite, HadesSprite
from hades_extensions.ecs import GameObjectType, Registry
from hades_extensions.ecs.components import KinematicComponent

__all__ = ()

# Create the texture path
texture_path = (
    Path(__file__).resolve().parent.parent / "src" / "hades" / "resources" / "textures"
)


@pytest.fixture
def constructor(request: pytest.FixtureRequest) -> GameObjectConstructor:
    """Create a game object constructor for testing.

    Args:
        request: The fixture request object.

    Returns:
        The game object constructor.
    """
    return GameObjectConstructor(
        "Test",
        "Test description",
        GameObjectType.Player,
        [f":resources:{param}" for param in request.param],
    )


@pytest.mark.parametrize(
    ("constructor", "position", "expected_path"),
    [
        (
            ["floor.png"],
            (672.0, 1312.0),
            texture_path / "floor.png",
        ),
        (
            ["floor.png"],
            (352.0, 672.0),
            texture_path / "floor.png",
        ),
        (
            ["floor.png"],
            (32.0, 32.0),
            texture_path / "floor.png",
        ),
    ],
    indirect=["constructor"],
)
def test_hades_sprite_init(
    constructor: GameObjectConstructor,
    position: tuple[float, float],
    expected_path: Path,
) -> None:
    """Test that a HadesSprite object is initialised correctly.

    Args:
        constructor: The game object constructor for testing.
        position: The position of the sprite object.
        expected_path: The expected path of the texture.
    """
    sprite = HadesSprite(
        Mock(),
        0,
        position,
        constructor,
    )
    assert sprite.position == position
    assert sprite.texture.file_path == expected_path
    assert sprite.game_object_id == 0
    assert sprite.game_object_type == GameObjectType.Player
    assert sprite.name == "Test"
    assert sprite.description == "Test description"


def test_hades_sprite_update() -> None:
    """Test that a HadesSprite object is updated correctly."""
    # Set up the mocks for the test
    registry = Mock(spec=Registry)
    kinematic_component = Mock(spec=KinematicComponent)
    kinematic_component.get_position.return_value = (64.0, 64.0)
    registry.get_component.return_value = kinematic_component

    # Create the sprite object and check that the position is correct
    constructor = GameObjectConstructor(
        "Test",
        "Test description",
        GameObjectType.Player,
        [":resources:floor.png"],
    )
    sprite = HadesSprite(registry, -1, (32.0, 32.0), constructor)

    # Update the sprite object and check that the position is correct
    sprite.update()
    assert sprite.position == (64.0, 64.0)


@pytest.mark.parametrize(
    ("constructor", "position", "expected_paths"),
    [
        (
            ["floor.png"],
            (672.0, 1312.0),
            [texture_path / "floor.png"],
        ),
        (
            ["floor.png"],
            (352.0, 672.0),
            [texture_path / "floor.png"],
        ),
        (
            ["floor.png", "wall.png"],
            (32.0, 32.0),
            [texture_path / "floor.png", texture_path / "wall.png"],
        ),
    ],
    indirect=["constructor"],
)
def test_animated_sprite_init(
    constructor: GameObjectConstructor,
    position: tuple[float, float],
    expected_paths: list[Path],
) -> None:
    """Test that an AnimatedSprite object is initialised correctly.

    Args:
        constructor: The game object constructor for testing.
        position: The position of the sprite object.
        expected_paths: The expected path of the textures.
    """
    sprite = AnimatedSprite(
        Mock(),
        0,
        position,
        constructor,
    )
    assert sprite.position == position
    assert sprite.texture.file_path == expected_paths[0]
    assert sprite.game_object_id == 0
    assert sprite.game_object_type == GameObjectType.Player
    assert sprite.name == "Test"
    assert sprite.description == "Test description"
    assert len(sprite.sprite_textures) == len(expected_paths)
    for i, textures in enumerate(sprite.sprite_textures):
        assert textures[0].file_path == expected_paths[i]
        assert len(sprite.sprite_textures[0]) == 2


def test_hades_sprite_no_texture() -> None:
    """Test that a HadesSprite object raises an error when no textures are provided."""
    with pytest.raises(expected_exception=IndexError, match="list index out of range"):
        HadesSprite(
            Mock(),
            0,
            (0, 0),
            GameObjectConstructor("Test", "Test", GameObjectType.Player, []),
        )
