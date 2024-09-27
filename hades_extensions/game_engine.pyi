from hades_extensions.ecs import Registry

class GameEngine:
    def get_registry(self: GameEngine) -> Registry: ...
    def set_player_id(self: GameEngine, player_id) -> None: ...
    def set_window_size(self: GameEngine, window_size: tuple[float, float]) -> None: ...
    def set_camera_position(
        self: GameEngine,
        camera_position: tuple[float, float],
    ) -> None: ...
    def on_key_press(self: GameEngine, symbol: int, modifiers: int) -> None: ...
    def on_key_release(self: GameEngine, symbol: int, modifiers: int) -> None: ...
    def on_mouse_motion(
        self: GameEngine,
        x: float,
        y: float,
        dx: float,
        dy: float,
    ) -> None: ...
