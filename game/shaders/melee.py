from __future__ import annotations

# Builtin
import pathlib
import struct
from typing import TYPE_CHECKING

# Custom
import arcade
from constants import PLAYER_ATTACK_RANGE, SPRITE_WIDTH

if TYPE_CHECKING:
    from arcade import ArcadeContext
    from arcade.gl import Buffer, Framebuffer, Program, Query
    from entities.enemy import Enemy
    from entities.player import Player
    from views.game import Game
    from window import Window

# Create the paths to the shader files
vertex_path = pathlib.Path(__file__).parent.joinpath("melee_vertex.glsl")
geometry_path = pathlib.Path(__file__).parent.joinpath("melee_geometry.glsl")


class MeleeShader:
    """
    A helper class which eases setting up the shader for the player's melee attack. This
    currently only works for the player but will probably change in the future.

    Parameters
    ----------
    view: Game
        The game view. This is used to access various parts of the game needed for the
        shader to work correctly.

    Attributes
    ----------
    program: Program | None
        The actual shader program which will run on the GPU to find the enemies.
    result_buffer: Buffer | None
        An OpenGL buffer object which holds the result of the shader program.
    query: Query | None
        An OpenGL query object used to query the shader, so we can get which enemies are
        within range.
    walls_framebuffer: Framebuffer | None
        An OpenGL framebuffer object used for holding the wall textures so the shader
        can check for collisions.
    """

    def __init__(self, view: Game) -> None:
        self.view: Game = view
        self.program: Program | None = None
        self.result_buffer: Buffer | None = None
        self.query: Query | None = None
        self.walls_framebuffer: Framebuffer | None = None

    def __repr__(self) -> str:
        return "<MeleeShader>"

    @property
    def window(self) -> Window:
        """Returns the window object for easy access."""
        return self.view.window

    @property
    def ctx(self) -> ArcadeContext:
        """Returns the arcade context object for running OpenGL programs."""
        return self.window.ctx

    @property
    def walls(self) -> arcade.SpriteList:
        """Returns the wall sprites for easy access."""
        return self.view.wall_sprites

    @property
    def enemies(self) -> arcade.SpriteList:
        """Returns the enemy sprites for easy access."""
        return self.view.enemies

    @property
    def player(self) -> Player:
        """Returns the player object for easy access."""
        return self.view.player

    def setup_shader(self) -> None:
        """Sets up the shader and it's needed attributes."""
        # Create the shader program. This draws lines from the player to each enemy
        # which is within a specific distance. It then checks if the player has line of
        # sight with each enemy that has a line drawn to them
        self.program = self.ctx.program(
            vertex_shader=open(vertex_path).read(),
            geometry_shader=open(geometry_path).read(),
        )
        # Configure program with maximum distance
        self.program["maxDistance"] = PLAYER_ATTACK_RANGE * SPRITE_WIDTH

        # We now need a buffer that can capture the result from the shader and process
        # it. But we need to make sure there is room for len(self.view.enemies) 32-bit
        # floats. To do this, since each enemy is 8-bit, we can multiply them by 4 to
        # get 32-bit and therefore multiply len(self.view.enemies) by 4 to get multiple
        # 32-bit floats
        self.enemies._write_sprite_buffers_to_gpu()
        self.result_buffer = self.ctx.buffer(reserve=len(self.enemies) * 4)

        # We also need a query to count how many sprites the shader gave us. To do this
        # we make a sampler that counts the number of primitives (the enemy locations in
        # this case) it emitted into the result buffer
        self.query = self.ctx.query()

        # Lastly, we need a framebuffer to hold the walls. To do this, we can render the
        # screen size as a texture then create a framebuffer from that texture. Then we
        # can load the wall textures into that framebuffer
        self.walls_framebuffer = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.texture((self.window.width, self.window.height))
            ]
        )
        with self.walls_framebuffer.activate() as fbo:
            fbo.clear()
            self.walls.draw()

    def run_shader(self) -> list[Enemy]:
        """
        Runs the shader program to find all enemies within range of the player based on
        the player's direction.

        Returns
        -------
        list[Enemy]
            A list of enemy objects that the player can attack.
        """
        # Make sure variables needed are valid
        assert self.program is not None
        assert self.result_buffer is not None
        assert self.query is not None
        assert self.walls_framebuffer is not None

        # Update the shader's origin point, so we can draw the rays from the correct
        # position
        self.program["origin"] = self.player.position

        # Ensure the internal sprite buffers are up-to-date
        self.enemies._write_sprite_buffers_to_gpu()

        # Make sure the wall positions are up-to-date
        with self.walls_framebuffer.activate() as fbo:
            fbo.clear()
            self.walls.draw()

        # Bind the wall textures to channel 0 so the shader can read them
        self.walls_framebuffer.color_attachments[0].use(0)

        # Query the shader to find enemies we can attack
        with self.query:
            # We already have a geometry instance in the sprite list that can be used to
            # run the shader. This only requires the correct input names (in_pos in this
            # case) which will automatically map the enemy position in the position
            # buffer to the vertex shader
            self.enemies._geometry.transform(
                self.program,
                self.result_buffer,
                vertices=len(self.enemies),
            )

        # Store the number of primitives/sprites found
        num_sprites_found = self.query.primitives_generated
        if num_sprites_found > 0:
            # Transfer the data from the shader into python and decode the value into
            # python objects. To do this, we unpack the result buffer from the VRAM and
            # convert each item into 32-bit floats which can then be searched for in the
            # enemies list
            return [
                self.enemies[int(i)]
                for i in struct.unpack(
                    f"{num_sprites_found}f",
                    self.result_buffer.read(size=num_sprites_found * 4),
                )
            ]
        # No sprites found
        return []
