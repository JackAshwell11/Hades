"""Manages Nox sessions used for automated testing and building."""

from __future__ import annotations

# Builtin
import platform
import subprocess
import zipfile
from pathlib import Path

# Pip
from nox import Session, options, session

# Define which sessions should be run
options.sessions = ["tests"]


@session()  # type: ignore[misc]
def executable(executable_session: Session) -> None:  # type: ignore[misc]
    """Compiles the game into an executable format for portable use.

    Args:
        executable_session: The Nox session to compile the game into an executable with.
    """
    # Install the required dependencies
    executable_session.install(".", "nuitka", "ordered-set")

    # Display some info about the system and the environment
    print(
        f"System information: {platform.system()} {platform.version()}. Python version:"
        f" {platform.python_implementation()} {platform.python_version()}",
    )

    # Initialise some paths
    current_dir = Path(__file__).parent
    source_dir = current_dir / "src" / "hades" / "window.py"
    output_dir = current_dir / "build"
    dist_dir = output_dir / "window.dist"

    # Execute the build command
    subprocess.run(
        [
            f"{executable_session.virtualenv.bin}/python",
            "-m",
            "nuitka",
            str(source_dir),
            "--standalone",
            "--assume-yes-for-downloads",
            "--include-data-dir=src/hades/resources=hades/resources",
            f"--output-dir={output_dir}",
            "--remove-output",
        ],
        check=True,
    )

    # Zip the game and display the result
    with zipfile.ZipFile("hades.zip", "w", zipfile.ZIP_DEFLATED) as zip_file:
        for build_file in dist_dir.rglob("*"):
            zip_file.write(build_file, str(build_file).replace(str(dist_dir), ""))


@session(python=["3.12", "3.13"])  # type: ignore[misc]
def tests(test_session: Session) -> None:  # type: ignore[misc]
    """Run the tests with coverage.

    Args:
        test_session: The Nox session to run the tests with.
    """
    test_session.install(".", "pytest-cov")
    test_session.run("pytest", "--cov-append")
    test_session.run("coverage", "lcov")
