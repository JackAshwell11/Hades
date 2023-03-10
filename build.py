"""Builds the game into an executable form using Nuitka."""
from __future__ import annotations

# Builtin
import argparse
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import Any


def executable() -> None:
    """Compiles the game into an executable format for portable use."""
    # Initialise some constants
    game_dir = "python_src/hades"
    source_dir = Path().absolute() / game_dir / "window.py"
    resources_dir = f"{game_dir}/resources"
    output_dir = Path().absolute() / "build"
    dist_dir = output_dir.joinpath("window.dist")
    zip_output_name = f"{game_dir}.zip"

    # Display some info about the system and the environment
    print(
        f"System information: {platform.system()} {platform.version()}. Python version:"
        f" {platform.python_implementation()} {platform.python_version()}"
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
        assert Path(zip_output_name).exists()
        print(f"Output of {zip_output_name} was successful")
    except AssertionError:
        print(f"Output of {zip_output_name} was unsuccessful")


def rust() -> None:
    """Compiles the Rust extensions and installs them into the virtual environment."""
    # Compile the Rust extensions and install them
    subprocess.run("maturin develop", check=True)


def build(_: dict[str, Any]) -> None:
    """Allow Poetry to automatically build the Rust extensions upon installation."""
    rust()


if __name__ == "__main__":
    # Build the argument parser and start parsing arguments
    parser = argparse.ArgumentParser(
        description="Simplifies building/compiling related to Hades"
    )
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "-e",
        "--executable",
        action="store_true",
        help="compiles the game into an executable format",
    )
    build_group.add_argument(
        "-r",
        "--rust",
        action="store_true",
        help="compiles the Rust extensions and installs them",
    )
    args = parser.parse_args()

    # Determine which argument was selected
    if args.executable:
        executable()
    elif args.rust:
        rust()
