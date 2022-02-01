from __future__ import annotations

# Builtin
from typing import Optional

# Custom
from entities.ai import FollowLineOfSight


class Character:
    """
    Represents an enemy or playable character in the game.

    Parameters
    ----------
    ai: Optional[FollowLineOfSight]
        The AI algorithm which this character uses.
    """

    def __init__(self, ai: Optional[FollowLineOfSight] = None) -> None:
        self.ai: Optional[FollowLineOfSight] = ai

    def __repr__(self) -> str:
        return "<Character>"
