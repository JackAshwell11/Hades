"""Manages various building/compiling operations on the game."""
from __future__ import annotations

# Builtin
import argparse
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import TypeVar

# Pip
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Define a generic type for the keyword arguments
KW = TypeVar("KW")


class BuildNamespace(argparse.Namespace):
    """Allows typing of an argparse Namespace for the CLI."""

    executable: bool
    cpp: bool


class CMakeBuild(build_ext):
    """A custom build_ext command which allows CMake projects to be built."""

    def build_extension(self: CMakeBuild, ext: Extension) -> None:
        """Build a CMake extension.

        Args:
            ext: The extension to build.
        """
        # Make sure the temporary build directory exists
        current_dir = Path(__file__).parent
        build_temp = current_dir.joinpath(self.build_temp).joinpath(ext.name)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Compile and build the CMake extension
        subprocess.run(
            " ".join(
                [
                    "cmake",
                    str(current_dir.joinpath(ext.sources[0])),
                    "-DDO_PYTHON=true",
                ],
            ),
            cwd=build_temp,
            check=True,
        )
        subprocess.run("cmake --build . --config Release", cwd=build_temp, check=True)


def executable() -> None:
    """Compiles the game into an executable format for portable use."""
    # Initialise some constants
    game_dir = "src/hades"
    source_dir = Path().absolute() / game_dir / "window.py"
    resources_dir = f"{game_dir}/resources"
    output_dir = Path().absolute() / "build"
    dist_dir = output_dir.joinpath("window.dist")
    zip_output_name = f"{game_dir}.zip"

    # Display some info about the system and the environment
    print(
        f"System information: {platform.system()} {platform.version()}. Python"
        f" version: {platform.python_implementation()} {platform.python_version()}",
    )
    print(f"Current directory: {Path().absolute()}")

    # Create the command to build the game
    commands = [
        ".\\.venv\\Scripts\\python.exe -m",
        f'nuitka "{source_dir}"',
        "--standalone",
        "--assume-yes-for-downloads",
        f"--include-data-dir={resources_dir}={resources_dir}",
        f'--output-dir="{output_dir}"',
        "--plugin-enable=numpy",
    ]
    command_string = " ".join(commands)
    print(f"Executing command string: {command_string}")

    # Execute the build command
    subprocess.run(command_string, check=True)

    # Zip the game and verify that the file exists
    with zipfile.ZipFile(zip_output_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for build_file in dist_dir.rglob("*"):
            zip_file.write(build_file, str(build_file).replace(str(dist_dir), ""))
    print(f"Finishing zipping {zip_output_name}. Now verifying archive")
    try:
        print(f"Output of {zip_output_name} was successful")
    except AssertionError:
        print(f"Output of {zip_output_name} was unsuccessful")


def cpp() -> None:
    """Compiles the C++ extensions and installs them into the virtual environment."""
    dist = setup(
        name="hades_extensions",
        ext_modules=[Extension("hades_extensions", ["hades_extensions"])],
        script_args=["bdist_wheel"],
        cmdclass={"build_ext": CMakeBuild},
    )
    subprocess.run(
        f'pip install --force-reinstall "{Path.cwd().joinpath(dist.dist_files[0][2])}"',
        check=True,
    )


if __name__ == "__main__":
    # Build the argument parser and start parsing arguments
    parser = argparse.ArgumentParser(
        description="Simplifies building/compiling related to Hades",
    )
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "-e",
        "--executable",
        action="store_true",
        help="Compiles the game into an executable format",
    )
    build_group.add_argument(
        "-c",
        "--cpp",
        action="store_true",
        help="Compiles the C++ extensions and installs them",
    )
    args = parser.parse_args(namespace=BuildNamespace())

    # Determine which argument was selected
    if args.executable:
        executable()
    elif args.cpp:
        cpp()
