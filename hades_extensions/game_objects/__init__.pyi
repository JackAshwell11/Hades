# Builtin
from abc import abstractmethod
from enum import Enum
from typing import Callable, Final, Self, TypeAlias, TypeVar

# Define some type vars for the registry
_C = TypeVar("_C")
_S = TypeVar("_S")

# Define the action function for callables
ActionFunction: TypeAlias = Callable[[int], float]

# Define the global variables
SPRITE_SCALE: Final[int] = ...
SPRITE_SIZE: Final[int] = ...

class RegistryError(Exception): ...
class InventorySpaceError(Exception): ...

class AttackAlgorithm(Enum):
    AreaOfEffect = ...
    Melee = ...
    Ranged = ...

class StatusEffectType(Enum):
    TEMP = ...
    TEMP2 = ...

class SteeringBehaviours(Enum):
    Arrive = ...
    Evade = ...
    Flee = ...
    FollowPath = ...
    ObstacleAvoidance = ...
    Pursue = ...
    Seek = ...
    Wander = ...

class SteeringMovementState(Enum):
    Default = ...
    Footprint = ...
    Target = ...

class ComponentBase: ...

class SystemBase:
    def __init__(self: SystemBase, registry: Registry) -> None: ...
    def get_registry(self: SystemBase) -> Registry: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...

class Vec2d:
    x: float
    y: float

    def __init__(self: Vec2d, x: float, y: float) -> None: ...
    def magnitude(self: Vec2d) -> float: ...
    def normalised(self: Vec2d) -> Vec2d: ...
    def rotated(self: Vec2d, angle: float) -> Vec2d: ...
    def angle_between(self: Vec2d, other: Vec2d) -> float: ...
    def distance_to(self: Vec2d, other: Vec2d) -> float: ...
    def __iter__(self: Vec2d) -> tuple[float, float]: ...
    def __add__(self: Vec2d, other: Vec2d) -> Vec2d: ...
    def __iadd__(self: Self, other: Vec2d) -> Self: ...
    def __sub__(self: Vec2d, other: Vec2d) -> Vec2d: ...
    def __mul__(self: Vec2d, other: Vec2d) -> Vec2d: ...
    def __truediv__(self: Vec2d, other: Vec2d) -> Vec2d: ...
    def __eq__(self: Vec2d, other: object) -> bool: ...
    def __ne__(self: Vec2d, other: object) -> bool: ...
    def __hash__(self: Vec2d) -> int: ...

class KinematicObject:
    position: Vec2d
    velocity: Vec2d
    rotation: float

class Registry:
    def __init__(self: Registry) -> None: ...
    def create_game_object(
        self: Registry,
        components: list[ComponentBase],
        *,
        kinematic: bool = False,
    ) -> int: ...
    def delete_game_object(self: Registry, game_object_id: int) -> None: ...
    def has_component(
        self: Registry,
        game_object_id: int,
        component: type[ComponentBase],
    ) -> bool: ...
    def get_component(
        self: Registry,
        game_object_id: int,
        component: type[_C],
    ) -> _C: ...
    def update(self: Registry, delta_time: float) -> None: ...
    def get_kinematic_object(
        self: Registry,
        game_object_id: int,
    ) -> KinematicObject: ...
    def add_wall(self: Registry, wall: Vec2d) -> None: ...
    def add_systems(self: Registry) -> None: ...
    def get_system(self: Registry, system: type[_S]) -> _S: ...