from __future__ import annotations

# Builtin
import pathlib

# Pip
import PyInstaller.__main__  # noqa

# Get path to the resources folder
resources_folder_name = "resources"
resources_path = (
    pathlib.Path(__file__)
    .resolve()
    .parent.joinpath("game")
    .joinpath(resources_folder_name)
)

PyInstaller.__main__.run(
    [
        "game/window.py",
        "--noconsole",
        "--clean",
        "--noconfirm",
        f"--add-data={resources_path};{resources_folder_name}",
    ]
)
