# Builtin
from collections.abc import Callable, Iterator
from enum import Enum
from typing import Final, SupportsFloat, TypeAlias, TypeVar

# Define some type vars for the registry
_C = TypeVar("_C")
_S = TypeVar("_S")

# Define the action function for callables
ActionFunction: TypeAlias = Callable[[int], float]

# Define the global variables
SPRITE_SCALE: Final[int] = ...
SPRITE_SIZE: Final[int] = ...

class Vec2d:
    def __init__(self: Vec2d, x: float, y: float) -> None: ...
    def __iter__(self: Vec2d) -> Iterator[SupportsFloat]: ...
    def __mul__(self, other: float) -> Vec2d: ...
    @property
    def x(self: Vec2d) -> float: ...
    @property
    def y(self: Vec2d) -> float: ...

class RegistryError(Exception): ...

class AttackAlgorithm(Enum):
    AreaOfEffect = ...
    Melee = ...
    Ranged = ...

class EventType(Enum):
    GameObjectCreation = ...
    GameObjectDeath = ...
    InventoryUpdate = ...
    SpriteRemoval = ...

class GameObjectType(Enum):
    Bullet = ...
    Enemy = ...
    Floor = ...
    Player = ...
    Wall = ...
    Goal = ...
    HealthPotion = ...
    Chest = ...

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
class SystemBase: ...

class Registry:
    def create_game_object(
        self: Registry,
        game_object_type: GameObjectType,
        position: Vec2d,
        components: list[ComponentBase],
    ) -> int: ...
    def delete_game_object(self: Registry, game_object_id: int) -> None: ...
    def has_game_object(self: Registry, game_object_id: int) -> bool: ...
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
    def get_game_object_type(self: Registry, game_object_id: int) -> GameObjectType: ...
    def get_game_object_ids(
        self: Registry,
        game_object_type: GameObjectType,
    ) -> list[int]: ...
    def get_system(self: Registry, system: type[_S]) -> _S: ...
    def update(self: Registry, delta_time: float) -> None: ...
    def add_callback(
        self: Registry,
        event_type: EventType,
        callback: Callable[[int], None],
    ) -> None: ...

def grid_pos_to_pixel(position: Vec2d) -> tuple[float, float]: ...
