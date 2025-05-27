"""Creates a start menu so the player can change their settings or game mode."""

from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import pygame
from pygame import Color, Rect, Surface, font

# Custom
from hades import ViewType
from hades.scene import Scene

if TYPE_CHECKING:
    from pygame.event import Event

    from hades.window import HadesWindow

__all__ = ("StartMenu",)

# Get the logger
logger = logging.getLogger(__name__)

FONT = font.SysFont(None, 40)


class Button(Rect):
    """A simple button class that inherits from pygame's Rect."""

    def __init__(self: Button, text: str) -> None:
        """Initialise the button with text and a callback function.

        Args:
            text: The text to display on the button.
        """
        super().__init__(0, 0, 200, 50)
        self.text: Surface = FONT.render(
            text=text,
            antialias=True,
            color=Color(255, 255, 255, 255),
        )
        self.colour: Color = Color(44, 62, 80, 255)


class StartMenu(Scene):
    """Creates a start menu useful for picking the game mode and options."""

    def __init__(self: StartMenu, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window where the start menu is displayed.
        """
        super().__init__(window)
        screen_width, screen_height = window.size
        spacing = 20

        self.start_button: Button = Button("Start Game")
        self.start_button.center = (
            screen_width // 2,
            screen_height // 2 - spacing // 2 - self.start_button.height // 2,
        )
        self.quit_button: Button = Button("Quit Game")
        self.quit_button.center = (
            screen_width // 2,
            screen_height // 2 + spacing // 2 + self.quit_button.height // 2,
        )

    def handle_events(self: StartMenu, events: list[Event]) -> None:
        """Handle input events."""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.start_button.collidepoint(mouse_pos):
                    self.window.scenes[ViewType.GAME].setup(0)
                    self.window.change_scene(ViewType.GAME)
                elif self.quit_button.collidepoint(mouse_pos):
                    pygame.quit()

    def draw(self: StartMenu, surface: Surface) -> None:
        """Draw the start menu on the given surface.

        Args:
            surface: The surface to draw the start menu on.
        """
        surface.fill((0, 119, 190))
        pygame.draw.rect(surface, self.start_button.colour, self.start_button)
        surface.blit(
            self.start_button.text,
            self.start_button.text.get_rect(center=self.start_button.center),
        )
        pygame.draw.rect(surface, self.quit_button.colour, self.quit_button)
        surface.blit(
            self.quit_button.text,
            self.quit_button.text.get_rect(center=self.quit_button.center),
        )
