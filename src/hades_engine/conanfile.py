"""Manages the Conan configuration for the application."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from conan import ConanFile

if TYPE_CHECKING:
    from typing import ClassVar


class HadesExtensions(ConanFile):  # type: ignore[misc]
    """The Conan configuration for the application."""

    # The metadata for the package
    generators: ClassVar[list[str]] = ["CMakeDeps"]
    settings: ClassVar[list[str]] = ["build_type"]

    def requirements(self: ConanFile) -> None:
        """Add the required dependencies to the Conan configuration."""
        self.requires("chipmunk2d/7.0.3")
        self.requires("gtest/1.16.0")
        self.requires("nlohmann_json/3.12.0")
        self.requires("pybind11/2.13.6")
