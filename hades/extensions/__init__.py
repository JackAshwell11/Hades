"""Contains all the extensions used by the game to speed up various features."""
from __future__ import annotations


def dummy(*_) -> None:
    """Allow faking the import of un-compiled C extensions."""
    raise NotImplementedError(
        "Extensions not compiled. Compile them before running Hades"
    )


# Check to see if the extensions are compiled or not. If so, import them normally,
# however, if they're not, replace the imports with a dummy function to fake the imports
try:
    from hades.extensions.astar.astar import calculate_astar_path, heuristic
    from hades.extensions.vector_field.vector_field import VectorField
except ImportError:
    calculate_astar_path, heuristic, VectorField = dummy, dummy, dummy


__all__ = (
    "calculate_astar_path",
    "heuristic",
    "VectorField",
)
