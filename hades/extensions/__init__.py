"""Contains all the extensions used by the game to speed up various features."""
from __future__ import annotations

# Builtin
import sys

# Custom
try:
    from hades.extensions.astar.astar import calculate_astar_path, heuristic
except ImportError:
    print("Extensions not compiled. Compile them before running Hades")
    sys.exit(1)


__all__ = (
    "calculate_astar_path",
    "heuristic",
)
