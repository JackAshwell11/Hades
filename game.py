# Builtin
from typing import Optional
# Pip
import pygame


class Game:
    """Manages the game and updates the display."""
    def __init__(self) -> None:
        self._running: bool = False
        self._width: int = 1280
        self._height: int = 720
        self._displaySurface: Optional[pygame.Surface] = None

    def doInitialisation(self) -> None:
        """Initialises pygame modules and sets up the game."""
        pygame.init()
        self._running = True
        self._displaySurface = pygame.display.set_mode(size=(self._width, self._height), flags=pygame.RESIZABLE)

    @staticmethod
    def doCleanup() -> None:
        """Cleans up pygame and its modules."""
        pygame.quit()

    def startLoop(self) -> None:
        """Calls the initialisation and starts the game loop."""
        # Initialise pygame
        self.doInitialisation()
        self._running = True
        # Start the game loop
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
        # Cleanup the game
        self.doCleanup()


# Runs the game
if __name__ == "__main__":
    game = Game()
    game.startLoop()
