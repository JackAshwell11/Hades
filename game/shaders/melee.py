from __future__ import annotations

# Builtin
import pathlib
import struct
from typing import TYPE_CHECKING

# Custom
from constants import PLAYER_ATTACK_RANGE, PLAYER_MELEE_DEGREE, SPRITE_WIDTH

if TYPE_CHECKING:
    from arcade import ArcadeContext
    from arcade.gl import Buffer, Framebuffer, Program, Query
    from entities.enemy import Enemy
    from views.game import Game

# Create the paths to the shader files
vertex_path = pathlib.Path(__file__).parent.joinpath("melee_vertex.glsl")
geometry_path = pathlib.Path(__file__).parent.joinpath("melee_geometry.glsl")


class MeleeShader:
    """
    A helper class which eases setting up the shader for the player's melee attack. This
    currently only works for the player but may change in the future.

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
    def ctx(self) -> ArcadeContext:
        """Returns the arcade context object for running OpenGL programs."""
        return self.view.window.ctx

    def setup_shader(self) -> None:
        """Sets up the shader and it's needed attributes."""
        # Create the shader program. This draws lines from the player to each enemy
        # which is within a specific distance. It then checks if the player has line of
        # sight with each enemy that has a line drawn to them
        self.program = self.ctx.program(
            vertex_shader=open(vertex_path).read(),
            geometry_shader=open(geometry_path).read(),
        )
        # Configure program with maximum distance and angle range
        self.program["maxDistance"] = PLAYER_ATTACK_RANGE * SPRITE_WIDTH
        self.program["half_angle_range"] = PLAYER_MELEE_DEGREE // 2

        # We now need a buffer that can capture the result from the shader and process
        # it. But we need to make sure there is room for len(self.view.enemies) 32-bit
        # floats. To do this, since each enemy is 8-bit, we can multiply them by 4 to
        # get 32-bit and therefore multiply len(self.view.enemies) by 4 to get multiple
        # 32-bit floats
        self.view.enemies.write_sprite_buffers_to_gpu()
        self.result_buffer = self.ctx.buffer(reserve=len(self.view.enemies) * 4)

        # We also need a query to count how many sprites the shader gave us. To do this
        # we make a sampler that counts the number of primitives (the enemy locations in
        # this case) it emitted into the result buffer
        self.query = self.ctx.query()

        # Lastly, we need a framebuffer to hold the walls. To do this, we can render the
        # screen size as a texture then create a framebuffer from that texture. Then we
        # can load the wall textures into that framebuffer
        self.walls_framebuffer = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.texture((self.view.window.width, self.view.window.height))
            ]
        )
        self.update_collision()

    def update_collision(self) -> None:
        """Updates the wall framebuffer to ensure collision detection is accurate."""
        # Make sure variables needed are valid
        assert self.walls_framebuffer is not None

        # Update the framebuffer
        with self.walls_framebuffer.activate() as fbo:
            fbo.clear()
            self.view.wall_sprites.draw()

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
        self.program["origin"] = self.view.player.position

        # Update the shader's relative origin point. This is the player's position
        # accounting for the camera scroll
        self.program["origin_relative"] = (
            self.view.player.center_x - self.view.camera.position[0],
            self.view.player.center_y - self.view.camera.position[1],
        )

        # Update the shader's direction, so we can attack in specific directions
        self.program["direction"] = self.view.player.direction

        # Ensure the internal sprite buffers are up-to-date
        self.view.enemies.write_sprite_buffers_to_gpu()

        # Bind the wall textures to channel 0 so the shader can read them
        self.walls_framebuffer.color_attachments[0].use(0)

        # Query the shader to find enemies we can attack
        with self.query:
            # We already have a geometry instance in the sprite list that can be used to
            # run the shader. This only requires the correct input names (in_pos in this
            # case) which will automatically map the enemy position in the position
            # buffer to the vertex shader
            self.view.enemies.geometry.transform(
                self.program,
                self.result_buffer,
                vertices=len(self.view.enemies),
            )

        # Store the number of primitives/sprites found
        num_sprites_found = self.query.primitives_generated
        if num_sprites_found > 0:
            # Transfer the data from the shader into python and decode the value into
            # python objects. To do this, we unpack the result buffer from the VRAM and
            # convert each item into 32-bit floats which can then be searched for in the
            # enemies list
            return [
                self.view.enemies[int(i)]
                for i in struct.unpack(
                    f"{num_sprites_found}f",
                    self.result_buffer.read(size=num_sprites_found * 4),
                )
            ]
        # No sprites found
        return []
