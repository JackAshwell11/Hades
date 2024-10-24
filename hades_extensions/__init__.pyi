# Builtin
from collections.abc import Sequence

# Custom
from hades_extensions.ecs import GameObjectType, Registry

def load_hitbox(
    game_object_type: GameObjectType,
    hitbox: Sequence[tuple[float, float]],
) -> bool: ...

class GameEngine:
    def __init__(self: GameEngine, level: int, seed: int | None = None) -> None: ...
    @property
    def player_id(self: GameEngine) -> int: ...
    def get_registry(self: GameEngine) -> Registry: ...
    def create_game_objects(self: GameEngine) -> None: ...
    def generate_enemy(self: GameEngine, delta_time: float = ...) -> None: ...
