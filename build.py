"""Builds the game into an executable form using Nuitka."""
from __future__ import annotations

# Builtin
import platform
import subprocess
import zipfile
from pathlib import Path

# Initialise some constants
GAME_NAME = "hades"
SOURCE_DIR = Path().absolute() / GAME_NAME / "window.py"
RESOURCES_DIR = f"{GAME_NAME}/resources"
OUTPUT_DIR = Path().absolute() / "build"
DIST_DIR = OUTPUT_DIR.joinpath("window.dist")
ZIP_OUTPUT_NAME = f"{GAME_NAME}.zip"

# Display some info about the system and the environment
print(
    f"System information: {platform.system()} {platform.version()}. Python version:"
    f" {platform.python_implementation()} {platform.python_version()}"
)
print(f"Current directory: {Path().absolute()}")

# Create the command to build the game
COMMANDS = [
    ".\\.venv\\Scripts\\python.exe -m",
    f'nuitka "{SOURCE_DIR}"',
    "--standalone",
    "--assume-yes-for-downloads",
    "--follow-imports",
    f"--include-data-dir={RESOURCES_DIR}={RESOURCES_DIR}",
    f'--output-dir="{OUTPUT_DIR}"',
    "--plugin-enable=numpy",
]
COMMAND_STRING = " ".join(COMMANDS)
print(f"Executing command string: {COMMAND_STRING}")

# Execute the build command
subprocess.run(COMMAND_STRING, check=True)

# Zip the game and verify that the file exists
with zipfile.ZipFile(ZIP_OUTPUT_NAME, "w", zipfile.ZIP_DEFLATED) as zip_file:
    for build_file in DIST_DIR.rglob("*"):
        zip_file.write(build_file, str(build_file).replace(str(DIST_DIR), ""))
print(f"Finishing zipping {ZIP_OUTPUT_NAME}. Now verifying archive")
try:
    assert Path(ZIP_OUTPUT_NAME).exists()
    print(f"Output of {ZIP_OUTPUT_NAME} was successful")
except AssertionError:
    print(f"Output of {ZIP_OUTPUT_NAME} was unsuccessful")
