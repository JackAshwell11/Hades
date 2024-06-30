"""Manages Nox sessions used for automated testing and building."""

from __future__ import annotations

# Builtin
import platform
import shutil
import subprocess
import sys
import zipfile
from contextlib import suppress
from pathlib import Path

# Pip
from nox import Session, options, session

# Define which sessions should be run
options.sessions = ["tests"]


def build_cpp_extensions(
    python_path: str = sys.prefix,
    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}",
) -> None:
    """Compiles the C++ extensions and installs them into the virtual environment.

    Args:
        python_path: The Python environment to install the C++ extensions into.
        python_version: The version of Python to install the C++ extensions into.
    """
    # Make sure the build directory is empty
    build_dir = Path(__file__).parent.joinpath("dist")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Compile and build the CMake extension
    os_type = platform.system()
    subprocess.run(
        [
            "cmake",
            str(Path(__file__).parent.joinpath("src/hades_extensions")),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={build_dir}",
            "-DDO_TESTS=OFF",
            "--fresh",
            "--preset",
            "WinRelease" if os_type == "Windows" else "LinuxRelease",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "src/hades_extensions/build-release"],
        check=True,
    )

    # Move the compiled files to the site-packages directory
    site_packages_dir = Path(python_path).joinpath(
        (
            "Lib/site-packages"
            if os_type == "Windows"
            else f"lib/python{python_version}/site-packages"
        ),
    )
    for file in build_dir.glob("*"):
        with suppress(shutil.Error):
            shutil.move(file, site_packages_dir)


@session()  # type: ignore[misc]
def cpp(_: Session) -> None:
    """Compiles the C++ extensions and installs them into the virtual environment."""
    build_cpp_extensions()


@session()  # type: ignore[misc]
def executable(executable_session: Session) -> None:
    """Compiles the game into an executable format for portable use.

    Args:
        executable_session: The Nox session to compile the game into an executable with.
    """
    # Install the required dependencies
    executable_session.install("nuitka", "ordered-set")

    # Display some info about the system and the environment
    print(
        f"System information: {platform.system()} {platform.version()}. Python version:"
        f" {platform.python_implementation()} {platform.python_version()}",
    )

    # Initialise some paths
    current_dir = Path(__file__).parent
    source_dir = current_dir.joinpath("src").joinpath("hades").joinpath("window.py")
    output_dir = current_dir.joinpath("build")
    dist_dir = output_dir.joinpath("window.dist")

    # Execute the build command
    build_cpp_extensions(
        executable_session.virtualenv.location,
        executable_session.python,
    )
    subprocess.run(
        [
            f"{executable_session.virtualenv.bin}/python -m",
            f'nuitka "{source_dir}"',
            "--standalone",
            "--assume-yes-for-downloads",
            "--include-data-dir=src/hades/resources=hades/resources",
            "--include-module=hades_extensions",
            f'--output-dir="{output_dir}"',
            "--remove-output",
        ],
        check=True,
    )

    # Zip the game and display the result
    with zipfile.ZipFile("hades.zip", "w", zipfile.ZIP_DEFLATED) as zip_file:
        for build_file in dist_dir.rglob("*"):
            zip_file.write(build_file, str(build_file).replace(str(dist_dir), ""))


@session(python=["3.11", "3.12"])  # type: ignore[misc]
def tests(test_session: Session) -> None:
    """Run the tests with coverage.

    Args:
        test_session: The Nox session to run the tests with.
    """
    test_session.install(".", "pytest-cov")
    build_cpp_extensions(test_session.virtualenv.location, test_session.python)
    test_session.run("pytest", "--cov-append")
    test_session.run("coverage", "lcov")
