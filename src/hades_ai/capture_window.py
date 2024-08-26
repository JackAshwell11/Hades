"""Manages the window for the AI model allowing periodic gameplay recordings."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import cv2
import numpy as np
from arcade import get_image

# Custom
from hades.window import HadesWindow

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image


class CaptureWindow(HadesWindow):
    """Allow periodic gameplay recordings for the AI model.

    Attributes:
        record: Whether to record the gameplay or not.
    """

    __slots__ = ("frames", "record")

    def __init__(self: CaptureWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.record: bool = False
        self.frames: list[Image] = []

    def on_update(self: CaptureWindow, _: float) -> None:
        """Capture the current gameplay frame."""
        # Save the current image to the frames list if we're recording
        if self.record:
            self.frames.append(get_image())

    def save_video(self: CaptureWindow, path: Path) -> None:
        """Save the recorded gameplay as a video."""
        # Check if there is any gameplay to save or not
        if not self.record:
            return

        # Create a video writer object
        video = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (self.width, self.height),
        )

        # Write each frame to the video
        for frame in self.frames:
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # Release the video writer object and clear the frames list
        video.release()
        self.frames.clear()
