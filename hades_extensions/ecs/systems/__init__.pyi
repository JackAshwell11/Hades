# Custom
from hades_extensions.ecs import ComponentBase, SystemBase

class InventorySystem(SystemBase):
    def use_item(self: InventorySystem, target_id: int, item_id: int) -> bool: ...

class PhysicsSystem(SystemBase):
    def get_wall_distances(
        self: PhysicsSystem,
        current_position: tuple[float, float],
    ) -> list[tuple[float, float]]: ...

class UpgradeSystem(SystemBase):
    def upgrade_component(
        self: UpgradeSystem,
        game_object_id: int,
        target_component: type[ComponentBase],
    ) -> None: ...
