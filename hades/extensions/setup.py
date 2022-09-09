"""Compiles the extensions used by the game."""
from __future__ import annotations

# Builtin
import shutil
from pathlib import Path

# Pip
import numpy as np
from setuptools import Extension, setup

if __name__ == "__main__":
    # Get the path to the extension
    extension_path = Path(__file__).parent
    build_path = extension_path / "build"

    # Build all the extensions
    print("Building extensions")
    setup(
        ext_modules=[
            Extension(
                "astar",
                [str(extension_path / "src" / "astar.cpp")],
                include_dirs=[np.get_include()],
            ),
            Extension(
                "vector_field",
                [str(extension_path / "src" / "vector_field.cpp")],
            ),
        ],
        script_args=["build_ext"],
    )

    # Move the built binary out of the build folder
    print(f"Extensions built. Moving result to {Path(__file__).parent}")
    dest_file = list(build_path.rglob("*.pyd"))[0]
    shutil.move(dest_file, extension_path / dest_file.name)

    # Delete the build folder
    print(f"Extensions moved. Deleting {build_path}")
    shutil.rmtree(build_path)
