"""Compiles the C++ extensions used by the game using setuptools."""
from __future__ import annotations

# Builtin
import shutil
from pathlib import Path

# Pip
import numpy as np
from setuptools import Extension, setup
from setuptools.command.build import build

# The build directory path
EXTENSION_PATH = Path(__file__).parent
BUILD_DIR = EXTENSION_PATH / "build"


class OverrideBuildDir(build):
    """Allows the build directory path to be overridden."""

    def initialize_options(self) -> None:
        """Override the build_base argument of setuptools changing the build path."""
        build.initialize_options(self)
        self.build_base = str(BUILD_DIR)


if __name__ == "__main__":
    # Set up a few variables needed for the compiling
    NUMPY_EXTENSIONS = ["astar"]

    # Build all the extensions
    ext_modules = [
        Extension(
            path.stem,
            [str(path)],
            include_dirs=[np.get_include() if path.stem in NUMPY_EXTENSIONS else None],
        )
        for path in EXTENSION_PATH.joinpath("src").glob("*.cpp")
    ]
    print(f"Building {len(ext_modules)} extensions")
    print("****************************************")
    setup(
        ext_modules=ext_modules,
        script_args=["build_ext"],
        cmdclass={"build": OverrideBuildDir},
    )
    print("****************************************")
    print(f"Successfully built {len(ext_modules)} extensions. Beginning moving process")

    # Move the built binaries out of the build folder
    built_extensions = list(BUILD_DIR.rglob("*.pyd")) + list(BUILD_DIR.rglob("*.so"))
    for extension in built_extensions:
        target_dir = EXTENSION_PATH / extension.name.split(".")[0]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / extension.name
        target_file.unlink(True)
        shutil.move(extension, target_file)
        print(f"Successfully moved {extension} to {target_file}")

    # Delete the build folder
    print(f"{len(ext_modules)} extensions moved. Deleting {BUILD_DIR}")
    shutil.rmtree(BUILD_DIR)
