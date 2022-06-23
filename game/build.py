"""Builds the game into an executable format for easy distribution."""
from __future__ import annotations

# Builtin
import pathlib

# Pip
import PyInstaller.__main__  # noqa

# Get path to the resources folder
RESOURCES_FOLDER_NAME = "resources"
resources_path = (
    pathlib.Path(__file__).resolve().parent / "game" / RESOURCES_FOLDER_NAME
)

PyInstaller.__main__.run(
    [
        "game/window.py",
        "--noconsole",
        "--clean",
        "--noconfirm",
        f"--add-data={resources_path};{RESOURCES_FOLDER_NAME}",
    ]
)

# CHANGE TO NUITKA, COMMAND:
#   nuitka game/window.py --standalone --follow-imports
#   --include-data-dir=./game/resources=resources --enable-plugin=numpy
