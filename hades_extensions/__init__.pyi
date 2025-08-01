# Builtin
from collections.abc import Sequence

# Custom
from hades_extensions.ecs import GameObjectType, Registry

def load_hitbox(
    game_object_type: GameObjectType,
    hitbox: Sequence[tuple[float, float]],
) -> bool: ...

class GameEngine:
    def __init__(self: GameEngine) -> None: ...
    @property
    def registry(self: GameEngine) -> Registry: ...
    @property
    def player_id(self: GameEngine) -> int: ...
    def set_seed(self: GameEngine, seed: str) -> None: ...
    def setup(self: GameEngine, file: str = "") -> None: ...
    def on_update(self: GameEngine, delta_time: float) -> None: ...
    def on_fixed_update(self: GameEngine, delta_time: float) -> None: ...
    def on_key_press(self: GameEngine, symbol: int, modifiers: int) -> None: ...
    def on_key_release(self: GameEngine, symbol: int, modifiers: int) -> None: ...
    def on_mouse_press(
        self: GameEngine,
        x: float,
        y: float,
        button: int,
        modifiers: int,
    ) -> bool: ...
    def use_item(self: GameEngine, target_id: int, item_id: int) -> None: ...
