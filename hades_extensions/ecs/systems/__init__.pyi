# Custom
from hades_extensions.ecs import SystemBase

class PhysicsSystem(SystemBase):
    def get_wall_distances(
        self: PhysicsSystem,
        current_position: tuple[float, float],
    ) -> list[tuple[float, float]]: ...

class ShopSystem(SystemBase):
    def purchase(self: ShopSystem, buyer_id: int, offering_index: int) -> None: ...
