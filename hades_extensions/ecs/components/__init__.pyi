# Built-in
from typing import overload

# Custom
from hades.sprite import HadesSprite
from hades_extensions.ecs import (
    ActionFunction,
    AttackAlgorithm,
    ComponentBase,
    StatusEffectType,
    SteeringBehaviours,
    SteeringMovementState,
    Vec2d,
)

class Stat(ComponentBase):
    def __init__(self: Stat, value: float, maximum_level: float) -> None: ...
    def get_value(self: Stat) -> float: ...
    def set_value(self: Stat, new_value: float) -> None: ...
    def get_max_value(self: Stat) -> float: ...
    def add_to_max_value(self: Stat, value: float) -> None: ...
    def get_current_level(self: Stat) -> int: ...
    def increment_current_level(self: Stat) -> None: ...
    def get_max_level(self: Stat) -> int: ...

class Armour(Stat): ...
class ArmourRegen(Stat): ...

class Attack(ComponentBase):
    def __init__(self: Attack, attack_algorithms: list[AttackAlgorithm]) -> None: ...
    @property
    def current_attack(self: Attack) -> AttackAlgorithm: ...

class AttackCooldown(Stat): ...
class AttackRange(Stat): ...

class StatusEffectData:
    def __init__(
        self: StatusEffectData,
        status_effect_type: StatusEffectType,
        increase: ActionFunction,
        duration: ActionFunction,
        interval: ActionFunction,
    ) -> None: ...

class Damage(Stat): ...

class EffectApplier(ComponentBase):
    def __init__(
        self: EffectApplier,
        instant_effects: dict[type[ComponentBase], ActionFunction],
        status_effects: dict[type[ComponentBase], StatusEffectData],
    ) -> None: ...

class EffectLevel(Stat): ...
class FootprintInterval(Stat): ...
class FootprintLimit(Stat): ...

class Footprints(ComponentBase):
    def __init__(self: Footprints) -> None: ...

class Health(Stat): ...

class Inventory(ComponentBase):
    def __init__(self: Inventory) -> None: ...
    @property
    def items(self: Inventory) -> list[int]: ...

class InventorySize(Stat): ...

class KeyboardMovement(ComponentBase):
    moving_north: bool
    moving_east: bool
    moving_south: bool
    moving_west: bool
    def __init__(self: KeyboardMovement) -> None: ...

class KinematicComponent(ComponentBase):
    @overload
    def __init__(self: KinematicComponent, vertices: list[Vec2d]) -> None: ...
    @overload
    def __init__(self: KinematicComponent, *, is_static: bool = False) -> None: ...
    def get_position(self: KinematicComponent) -> tuple[float, float]: ...
    def set_rotation(self: KinematicComponent, angle: float) -> None: ...

class MeleeAttackSize(Stat): ...

class Money(ComponentBase):
    money: int
    def __init__(self: Money) -> None: ...

class MovementForce(Stat): ...

class PythonSprite(ComponentBase):
    sprite: HadesSprite
    def __init__(self: PythonSprite) -> None: ...

class Effect:
    @property
    def duration(self: Effect) -> float: ...
    @property
    def target_component(self: Effect) -> type[ComponentBase]: ...

class StatusEffect(ComponentBase):
    def __init__(self: StatusEffect) -> None: ...
    @property
    def applied_effects(self: StatusEffect) -> dict[StatusEffectType, Effect]: ...

class SteeringMovement(ComponentBase):
    target_id: int
    def __init__(
        self: SteeringMovement,
        behaviours: dict[SteeringMovementState, list[SteeringBehaviours]],
    ) -> None: ...

class Upgrades(ComponentBase):
    def __init__(
        self: Upgrades,
        upgrades: dict[type[Stat], tuple[ActionFunction, ActionFunction]],
    ) -> None: ...
    @property
    def upgrades(
        self: Upgrades,
    ) -> dict[type[Stat], tuple[ActionFunction, ActionFunction]]: ...

class ViewDistance(Stat): ...
