"""Compiles the C++ extensions used by the game using setuptools."""
from __future__ import annotations

# Builtin
import shutil
from pathlib import Path

# Pip
from setuptools import Extension, setup

# The build directory path
EXTENSION_PATH = Path(__file__).parent
BUILD_DIR = EXTENSION_PATH / "build"


if __name__ == "__main__":
    # Build all the extensions
    ext_modules = [
        Extension("map", [str(EXTENSION_PATH.joinpath("src").joinpath("map.cpp"))])
    ]
    print(f"Building {len(ext_modules)} extensions")
    print("****************************************")
    setup(
        ext_modules=ext_modules,
        script_args=["build_ext"],
        options={"build_ext": {"build_lib": str(BUILD_DIR)}},
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
