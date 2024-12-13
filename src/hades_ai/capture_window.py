"""Manages the window for the AI model allowing periodic gameplay recordings."""

from __future__ import annotations

# Builtin
from pathlib import Path

# Pip
import cv2
import numpy as np
from arcade import get_image

# Custom
from hades.window import HadesWindow

__all__ = ("CaptureWindow",)


class CaptureWindow(HadesWindow):
    """Allow periodic gameplay recordings for the AI model.

    Attributes:
        path: The path to save the recorded gameplay.
        writer: The video writer object to save the recorded gameplay.
    """

    __slots__ = ("path", "writer")

    def __init__(self: CaptureWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.path: Path = Path()
        self.writer: cv2.VideoWriter | None = None

    def make_writer(self: CaptureWindow) -> None:
        """Create a video writer object."""
        self.writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (self.width, self.height),
        )

    def on_update(self: CaptureWindow, _: float) -> None:
        """Capture the current gameplay frame."""
        if self.writer:
            self.writer.write(cv2.cvtColor(np.array(get_image()), cv2.COLOR_RGB2BGR))

    def save_video(self: CaptureWindow) -> None:
        """Save the recorded gameplay as a video."""
        if not self.writer:
            return

        # Create a video writer object
        self.writer.release()
        self.writer = None
        self.path = Path()
