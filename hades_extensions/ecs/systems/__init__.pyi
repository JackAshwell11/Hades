# Custom
from hades_extensions.ecs import ComponentBase, GameObjectType, SystemBase

class ArmourRegenSystem(SystemBase): ...

class AttackSystem(SystemBase):
    def do_attack(
        self: AttackSystem,
        game_object: tuple[int, GameObjectType],
        targets: list[int],
    ) -> None: ...
    def previous_attack(self: AttackSystem, game_object_id: int) -> None: ...
    def next_attack(self: AttackSystem, game_object_id: int) -> None: ...

class DamageSystem(SystemBase):
    def deal_damage(
        self: DamageSystem,
        game_object_id: int,
        attacker_id: int,
    ) -> None: ...

class EffectSystem(SystemBase):
    def apply_effects(
        self: EffectSystem,
        game_object_id: int,
        target_game_object_id: int,
    ) -> bool: ...

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
class PhysicsSystem(SystemBase): ...
class SteeringMovementSystem(SystemBase): ...

class UpgradeSystem(SystemBase):
    def upgrade_component(
        self: UpgradeSystem,
        game_object_id: int,
        target_component: type[ComponentBase],
    ) -> None: ...
