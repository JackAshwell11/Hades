"""Initialises and manages the main game."""
from __future__ import annotations

# Builtin
import logging
from typing import NamedTuple

# Pip
from arcade import (
    MOUSE_BUTTON_LEFT,
    Camera,
    SpriteList,
    Text,
    View,
    color,
    draw_points,
    get_sprites_at_point,
    key,
    schedule,
)
from hades_extensions import TileType, create_map

# Custom
from hades.constants import (
    DEBUG_ENEMY_SPAWN_COLOR,
    DEBUG_ENEMY_SPAWN_SIZE,
    DEBUG_GAME,
    ENEMY_GENERATE_INTERVAL,
    ENEMY_RETRY_COUNT,
    SPRITE_SIZE,
    TOTAL_ENEMY_COUNT,
    ComponentType,
    GameObjectType,
)
from hades.game_objects.constructors import (
    ENEMY,
    FLOOR,
    PLAYER,
    POTION,
    WALL,
    GameObjectConstructor,
)
from hades.game_objects.sprite import GameObject, HadesSprite
from hades.game_objects.system import ECS
from hades.physics import PhysicsEngine
from hades.textures import grid_pos_to_pixel

__all__ = ("Game",)

# Get the logger
logger = logging.getLogger(__name__)


class LevelConstants(NamedTuple):
    """Holds the constants for a specific level.

    Args:
        level: The level of this game.
        width: The width of the game map.
        height: The height of the game map.
    """

    level: int
    width: int
    height: int


class Game(View):
    """Manages the game and its actions.

    Attributes:
        level_constants: Holds the constants for the current level.
        system: The entity component system which manages the game objects.
        ids: The dictionary which stores the IDs and sprite objects for each game object
        type.
        tile_sprites: The sprite list for the tile game objects.
        entity_sprites: The sprite list for the entity game objects.
        physics_engine: The physics engine which processes wall collision.
        game_camera: The camera used for moving the viewport around the screen.
        gui_camera: The camera used for visualising the GUI elements.
        possible_enemy_spawns: A list of possible positions that enemies can spawn in.
        player_status_text: The text object used for displaying the player's health and
            armour.
    """

    def _initialise_game_object(
        self: Game,
        constructor: GameObjectConstructor,
        sprite_list: SpriteList[HadesSprite],
        position: tuple[int, int],
        *,
        single: bool = False,
    ) -> None:
        """Initialise a game object from a constructor into the ECS.

        Args:
            constructor: The game object constructor to initialise.
            sprite_list: The sprite list to add the sprite object too.
            position: The position of the game object in the grid.
            single: Whether the game object will be the only game object of its type or
            not.

        Returns:
            The initialised sprite object.
        """
        # Initialise a sprite object
        sprite_obj = HadesSprite(
            GameObject(
                self.system.add_game_object(
                    constructor.component_data,
                    *constructor.components,
                ),
                constructor.game_object_type,
                self.system,
            ),
            position,
            constructor.game_object_textures,
        )

        # Add it to the various collections and systems
        sprite_list.append(sprite_obj)
        self.physics_engine.add_game_object(sprite_obj, blocking=constructor.blocking)
        if single:
            self.ids[constructor.game_object_type] = sprite_obj
        else:
            self.ids.setdefault(constructor.game_object_type, []).append(sprite_obj)

    def __init__(self: Game, level: int) -> None:
        """Initialise the object.

        Args:
            level: The level to create a game for.
        """
        super().__init__()
        generation_result = create_map(level)
        self.level_constants: LevelConstants = LevelConstants(*generation_result[1])
        self.system: ECS = ECS()
        self.ids: dict[GameObjectType, HadesSprite | list[HadesSprite]] = {}
        self.tile_sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.entity_sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.physics_engine: PhysicsEngine = PhysicsEngine()
        self.game_camera: Camera = Camera()
        self.gui_camera: Camera = Camera()
        self.possible_enemy_spawns: set[tuple[int, int]] = set()
        self.upper_camera_x: float = -1
        self.upper_camera_y: float = -1
        self.player_status_text: Text = Text(
            "Money: 0",
            10,
            10,
            font_size=20,
        )

        # Initialise the game objects
        for count, tile in enumerate(generation_result[0]):
            # Get the screen position from the grid position
            position = (
                count % self.level_constants.width,
                count // self.level_constants.width,
            )

            # Determine the type of the tile
            if tile == TileType.Wall:
                self._initialise_game_object(WALL, self.tile_sprites, position)
            else:
                if tile == TileType.Player:
                    self._initialise_game_object(
                        PLAYER,
                        self.entity_sprites,
                        position,
                        single=True,
                    )
                else:
                    self._initialise_game_object(POTION, self.tile_sprites, position)

                # Make the game object's backdrop a floor
                self._initialise_game_object(FLOOR, self.tile_sprites, position)
                self.possible_enemy_spawns.add(position)
                # TODO: Properly get possible_enemy_spawns

        # Calculate upper_camera_x and upper_camera_y
        half_sprite_size = SPRITE_SIZE / 2
        screen_width, screen_height = grid_pos_to_pixel(
            self.level_constants.width,
            self.level_constants.height,
        )
        self.upper_camera_x, self.upper_camera_y = (
            screen_width - self.game_camera.viewport_width - half_sprite_size,
            screen_height - self.game_camera.viewport_height - half_sprite_size,
        )

        # Generate half of the total enemies allowed then schedule their generation
        for _ in range(TOTAL_ENEMY_COUNT // 2):
            self.generate_enemy()
        schedule(
            self.generate_enemy,
            ENEMY_GENERATE_INTERVAL,
        )

    def on_draw(self: Game) -> None:
        """Render the screen."""
        # Clear the screen and set the background color
        self.clear()
        self.window.background_color = color.BLACK

        # Activate our game camera
        self.game_camera.use()

        # Draw the various spritelists
        self.tile_sprites.draw(pixelated=True)
        self.entity_sprites.draw(pixelated=True)

        # Draw the stuff needed for the debug mode
        if DEBUG_GAME:
            # Draw the enemy spawn locations
            draw_points(
                [
                    grid_pos_to_pixel(*location)
                    for location in self.possible_enemy_spawns
                ],
                DEBUG_ENEMY_SPAWN_COLOR,
                DEBUG_ENEMY_SPAWN_SIZE,
            )

        # Draw the gui on the screen
        self.gui_camera.use()
        self.player_status_text.value = "Money: " + str(
            self.system.get_component_for_game_object(
                self.ids[GameObjectType.PLAYER].game_object_id,
                ComponentType.MONEY,
            ).value,
        )
        self.player_status_text.draw()

    def on_update(self: Game, delta_time: float) -> None:
        """Process movement and game logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        # Check if the game should end
        # if self.player.health.value <= 0 or not self.enemy_sprites:

        # Update the entities
        self.entity_sprites.on_update(delta_time)

        # Update the physics engine
        self.physics_engine.step()

        # Position the camera
        self.center_camera_on_player()

    def on_key_press(self: Game, symbol: int, modifiers: int) -> None:
        """Process key press functionality.

        Args:
            symbol: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "Received key press with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case key.W:
                self.player.up_pressed = True
            case key.S:
                self.player.down_pressed = True
            case key.A:
                self.player.left_pressed = True
            case key.D:
                self.player.right_pressed = True

    def on_key_release(self: Game, symbol: int, modifiers: int) -> None:
        """Process key release functionality.

        Args:
            symbol: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "Received key release with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case key.W:
                self.player.up_pressed = False
            case key.S:
                self.player.down_pressed = False
            case key.A:
                self.player.left_pressed = False
            case key.D:
                self.player.right_pressed = False

    def on_mouse_press(self: Game, x: int, y: int, button: int, modifiers: int) -> None:
        """Process mouse button functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
            button: Which button was hit.
            modifiers:Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "%r mouse button was pressed at position (%f, %f) with modifiers %r",
            button,
            x,
            y,
            modifiers,
        )
        if button is MOUSE_BUTTON_LEFT:
            self.player.attack()

    def generate_enemy(self: Game, _: float = 1 / 60) -> None:
        """Generate an enemy outside the player's fov."""
        if len(self.ids.get(GameObjectType.ENEMY, [])) >= TOTAL_ENEMY_COUNT:
            return

        # Enemy limit not reached so attempt to initialise a new enemy game object
        # ENEMY_RETRY_COUNT times
        for _ in range(ENEMY_RETRY_COUNT):
            position = self.possible_enemy_spawns.pop()
            if not get_sprites_at_point(
                grid_pos_to_pixel(*position),
                self.entity_sprites,
            ):
                self._initialise_game_object(ENEMY, self.entity_sprites, position)
                return

    def center_camera_on_player(self: Game) -> None:
        """Centers the camera on the player."""
        # Check if the camera is already centered on the player
        player_sprite = self.ids[GameObjectType.PLAYER]
        if self.game_camera.position == player_sprite.position:
            return

        # Make sure the camera doesn't extend beyond the boundaries
        screen_center = (
            min(
                max(player_sprite.center_x - (self.game_camera.viewport_width / 2), 0),
                self.upper_camera_x,
            ),
            min(
                max(player_sprite.center_y - (self.game_camera.viewport_height / 2), 0),
                self.upper_camera_y,
            ),
        )

        # Check if the camera position has changed. If so, move the camera to the new
        # position
        if self.game_camera.position != screen_center:
            self.game_camera.move_to(screen_center)

    def __repr__(self: Game) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Game (Current window={self.window})>"
