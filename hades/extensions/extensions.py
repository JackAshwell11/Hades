"""Compiles the extensions used by the game."""
from __future__ import annotations

# Builtin
import shutil
from pathlib import Path

# Pip
import numpy as np
from setuptools import Extension, setup

if __name__ == "__main__":
    # Set up a few variables needed for the compiling
    NUMPY_EXTENSIONS = ["astar"]
    extension_path = Path(__file__).parent
    build_path = extension_path / "build"

    # Build all the extensions
    ext_modules = [
        Extension(
            path.stem,
            [str(path)],
            include_dirs=[np.get_include() if path.stem in NUMPY_EXTENSIONS else None],
        )
        for path in extension_path.joinpath("src").glob("*.cpp")
    ]
    print(f"Building {len(ext_modules)} extensions")
    print("****************************************")
    setup(
        ext_modules=ext_modules,
        script_args=["build_ext"],
    )
    print("****************************************")
    print(f"Successfully built {len(ext_modules)} extensions. Beginning moving process")

    # Move the built binaries out of the build folder
    for extension in ext_modules:
        print(f"Moving {extension.name} to module directory")
        dest_file = list(build_path.rglob("*.pyd"))[0]
        target_dir = extension_path / extension.name / dest_file.name
        shutil.move(dest_file, target_dir)
        print(f"Successfully moved {dest_file} to {target_dir}")

    # Delete the build folder
    print(f"{len(ext_modules)} extensions moved. Deleting {build_path}")
    shutil.rmtree(build_path)
