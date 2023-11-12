# Builtin
from abc import abstractmethod
from enum import Enum
from typing import Callable

# Custom
from hades_extensions.game_objects import ComponentBase, SystemBase, Vec2d

# TODO: see if everything here is used

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

class Attacks(ComponentBase):
    attack_algorithms: list[AttackAlgorithm]
    attack_state: int
    def __init__(self: Attacks, attack_algorithms: list[AttackAlgorithm]) -> None: ...

class StatusEffect(ComponentBase):
    value: float
    duration: float
    interval: float
    target_component: type
    def __init__(
        self: StatusEffect,
        value: float,
        duration: float,
        interval: float,
        target_component: type,
    ) -> None: ...

class StatusEffectData(ComponentBase):
    status_effect_type: StatusEffectType
    increase: Callable[[int], float]
    def __init__(
        self: StatusEffectData,
        status_effect_type: StatusEffectType,
        increase: Callable[[int], float],
        duration: Callable[[int], float],
        interval: Callable[[int], float],
    ) -> None: ...

class EffectApplier(ComponentBase):
    instant_effects: dict[type, Callable[[int], float]]
    status_effects: dict[type, StatusEffectData]
    def __init__(
        self: EffectApplier,
        instant_effects: dict[type, Callable[[int], float]],
        status_effects: dict[type, StatusEffectData],
    ) -> None: ...

class StatusEffects(ComponentBase):
    applied_effects: dict[type, StatusEffect]
    def __init__(self: StatusEffects) -> None: ...

class Inventory(ComponentBase):
    width: int
    height: int
    items: list[int]
    def __init__(self: Inventory, width: int, height: int) -> None: ...
    def get_capacity(self: Inventory) -> int: ...

class Footprints(ComponentBase):
    footprints: list[int]
    time_since_last_footprint: float
    def __init__(self: Footprints) -> None: ...

class KeyboardMovement(ComponentBase):
    moving_north: bool
    moving_east: bool
    moving_south: bool
    moving_west: bool
    def __init__(self: KeyboardMovement) -> None: ...

class SteeringMovement(ComponentBase):
    behaviours: dict[SteeringMovementState, list[SteeringBehaviours]]
    movement_state: SteeringMovementState
    target_id: int
    path_list: list[Vec2d]
    def __init__(
        self: SteeringMovement,
        behaviours: dict[SteeringMovementState, list[SteeringBehaviours]],
    ) -> None: ...

class Money(ComponentBase):
    money: int
    def __init__(self: Money, money: int) -> None: ...

class Upgrades(ComponentBase):
    upgrades: dict[type, Callable[[int], float]]
    def __init__(
        self: Upgrades,
        upgrades: dict[type, Callable[[int], float]],
    ) -> None: ...

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
        target_component: type,
        increase: Callable[[int], float],
        level: int,
    ) -> None: ...
    def apply_status_effect(
        self: EffectSystem,
        game_object_id: int,
        target_component: type,
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
    def calculate_keyboard_force(
        self: KeyboardMovementSystem,
        game_object_id: int,
    ) -> Vec2d: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...

class SteeringMovementSystem(SystemBase):
    def calculate_steering_force(
        self: SteeringMovementSystem,
        game_object_id: int,
    ) -> Vec2d: ...
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
        target_component: type,
    ) -> None: ...
    @abstractmethod
    def update(self: SystemBase, delta_time: float) -> None: ...
