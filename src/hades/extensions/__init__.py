"""Contains all the extensions used by the game to speed up various features."""
from __future__ import annotations


def dummy(*_) -> None:
    """Allow faking the import of un-compiled or non-functioning C extensions."""
    raise NotImplementedError(  # pragma: no cover
        "Extensions not compiled or not functioning. Compile/fix them before running"
        " Hades"
    )


# Check to see if the extensions are compiled or not. If so, import them normally,
# however, if they're not, replace the imports with a dummy function to fake the imports
try:
    from hades.extensions.astar.astar import calculate_astar_path, heuristic
    from hades.extensions.vector_field.vector_field import VectorField
except ImportError as error:  # pragma: no cover
    print(error)
    calculate_astar_path = dummy  # type: ignore
    heuristic = dummy  # type: ignore
    VectorField = dummy  # type: ignore


__all__ = (
    "calculate_astar_path",
    "heuristic",
    "VectorField",
)
