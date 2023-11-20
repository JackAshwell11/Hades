# Builtin
from abc import abstractmethod

# Custom
from hades_extensions.game_objects import (
    ActionFunction,
    ComponentBase,
    SystemBase,
    Vec2d,
)
from hades_extensions.game_objects.components import StatusEffectData

class ArmourRegenSystem(SystemBase):
    def update(self: ArmourRegenSystem, delta_time: float) -> None: ...

class AttackSystem(SystemBase):
    def do_attack(
        self: AttackSystem,
        game_object_id: int,
        targets: list[int],
    ) -> None: ...
    def previous_attack(self: AttackSystem, game_object_id: int) -> None: ...
    def next_attack(self: AttackSystem, game_object_id: int) -> None: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...

class DamageSystem(SystemBase):
    def deal_damage(self: DamageSystem, game_object_id: int, damage: int) -> None: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...

class EffectSystem(SystemBase):
    def apply_instant_effect(
        self: EffectSystem,
        game_object_id: int,
        target_component: type[ComponentBase],
        increase: ActionFunction,
        level: int,
    ) -> None: ...
    def apply_status_effect(
        self: EffectSystem,
        game_object_id: int,
        target_component: type[ComponentBase],
        status_effect_data: StatusEffectData,
        level: int,
    ) -> None: ...
    def update(self: SystemBase, delta_time: float) -> None: ...

class InventorySystem(SystemBase):
    def add_item_to_inventory(
        self: InventorySystem,
        game_object_id: int,
        item: int,
    ) -> None: ...
    def remove_item_from_inventory(
        self: InventorySystem,
        game_object_id: int,
        index: int,
    ) -> int: ...
    def update(self: SystemBase, delta_time: float) -> None: ...

class FootprintSystem(SystemBase):
    def update(self: SystemBase, delta_time: float) -> None: ...

class KeyboardMovementSystem(SystemBase):
    def calculate_force(self: KeyboardMovementSystem, game_object_id: int) -> Vec2d: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...

class SteeringMovementSystem(SystemBase):
    def calculate_force(self: SteeringMovementSystem, game_object_id: int) -> Vec2d: ...
    def update_path_list(
        self: SteeringMovementSystem,
        game_object_id: int,
        path_list: list[Vec2d],
    ) -> None: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...

class UpgradeSystem(SystemBase):
    def upgrade_component(
        self: UpgradeSystem,
        game_object_id: int,
        target_component: type[ComponentBase],
    ) -> None: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...
