# Custom
from hades_extensions.ecs import AttackType, ComponentBase, SystemBase, Vec2d

class ArmourRegenSystem(SystemBase): ...

class AttackSystem(SystemBase):
    def do_attack(
        self: AttackSystem,
        game_object_id: int,
        attack_type: AttackType,
    ) -> bool: ...

class EffectSystem(SystemBase):
    def apply_effects(self: EffectSystem, source: int, target: int) -> bool: ...

class FootprintSystem(SystemBase): ...

class InventorySystem(SystemBase):
    def add_item_to_inventory(
        self: InventorySystem,
        game_object_id: int,
        item: int,
    ) -> bool: ...
    def remove_item_from_inventory(
        self: InventorySystem,
        game_object_id: int,
        item_id: int,
    ) -> int: ...
    def use_item(self: InventorySystem, target_id: int, item_id: int) -> bool: ...

class KeyboardMovementSystem(SystemBase): ...

class PhysicsSystem(SystemBase):
    def get_nearest_item(self: PhysicsSystem, game_object_id: int) -> int: ...
    def get_wall_distances(
        self: PhysicsSystem,
        current_position: Vec2d,
    ) -> list[Vec2d]: ...

class SteeringMovementSystem(SystemBase): ...

class UpgradeSystem(SystemBase):
    def upgrade_component(
        self: UpgradeSystem,
        game_object_id: int,
        target_component: type[ComponentBase],
    ) -> None: ...
