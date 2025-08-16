# Custom
from hades_extensions.ecs import SystemBase

class InventorySystem(SystemBase):
    def use_item(self: InventorySystem, target_id: int, item_id: int) -> None: ...

class ShopSystem(SystemBase):
    def purchase(self: ShopSystem, buyer_id: int, offering_index: int) -> None: ...
