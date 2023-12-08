// Related header
#include "game_objects/systems/inventory.hpp"

// ----- FUNCTIONS ------------------------------
void InventorySystem::add_item_to_inventory(const GameObjectID game_object_id, const GameObjectID item) const {
  auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  if (static_cast<int>(inventory->items.size()) == inventory->get_capacity()) {
    throw InventorySpaceError(true);
  }
  inventory->items.push_back(item);
}

auto InventorySystem::remove_item_from_inventory(const GameObjectID game_object_id, const int index) const -> int {
  auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  if (inventory->items.empty()) {
    throw InventorySpaceError(false);
  }
  if (index < 0 || index >= static_cast<int>(inventory->items.size())) {
    throw InventorySpaceError("The index is out of range.");
  }
  const int item{inventory->items[index]};
  inventory->items.erase(inventory->items.begin() + index);
  return item;
}
