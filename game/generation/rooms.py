from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class RectInstance(Enum):
    """Stores all the possible instances a rect could represent in the game map."""

    SMALL_ROOM = "SMALL_ROOM"
    MEDIUM_ROOM = "MEDIUM_ROOM"
    LARGE_ROOM = "LARGE_ROOM"


@dataclass()
class Template:
    """
    Represents the base class for defining rooms/hallways.

    Parameters
    ----------
    min_width: int
        The minimum width this object can be.
    max_width: int
        The maximum width this object can be.
    min_height: int
        The minimum height this object can be.
    max_height: int
        The maximum height this object can be.
    probabilities: Dict[str, float]
        A dict containing the probabilities for spawning more rects after the current
        one.
    """

    min_width: int = -1
    max_width: int = -1
    min_height: int = -1
    max_height: int = -1
    probabilities: Dict[str, float] = field(default_factory=dict)


# These templates have to be the value of RectInstance otherwise we wouldn't be able to
# iterate over the dict to calculate the new probabilities

starting_room = Template(
    7,
    7,
    7,
    7,
    {
        RectInstance.SMALL_ROOM.value: 0.34,
        RectInstance.MEDIUM_ROOM.value: 0.33,
        RectInstance.LARGE_ROOM.value: 0.33,
    },
)

small_room = Template(
    5,
    7,
    5,
    7,
    {
        RectInstance.SMALL_ROOM.value: 0.2,
        RectInstance.MEDIUM_ROOM.value: 0.4,
        RectInstance.LARGE_ROOM.value: 0.4,
    },
)

medium_room = Template(
    7,
    9,
    7,
    9,
    {
        RectInstance.SMALL_ROOM.value: 0.4,
        RectInstance.MEDIUM_ROOM.value: 0.3,
        RectInstance.LARGE_ROOM.value: 0.2,
    },
)

large_room = Template(
    9,
    11,
    9,
    11,
    {
        RectInstance.SMALL_ROOM.value: 0.6,
        RectInstance.MEDIUM_ROOM.value: 0.3,
        RectInstance.LARGE_ROOM.value: 0.1,
    },
)
