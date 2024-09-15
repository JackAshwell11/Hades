// Related header
#include "ecs/systems/inventory.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/effects.hpp"

auto InventorySystem::add_item_to_inventory(const GameObjectID game_object_id, const GameObjectID item) const -> bool {
  const auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  if (const auto inventory_size{get_registry()->get_component<InventorySize>(game_object_id)};
      static_cast<int>(inventory->items.size()) == inventory_size->get_value()) {
    throw std::runtime_error("The inventory is full.");
  }
  inventory->items.push_back(item);
  get_registry()->notify_callbacks(EventType::InventoryUpdate, game_object_id);
  return true;
}

auto InventorySystem::remove_item_from_inventory(const GameObjectID game_object_id, const GameObjectID item_id) const
    -> bool {
  const auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  const auto index{std::ranges::find(inventory->items.begin(), inventory->items.end(), item_id) -
                   inventory->items.begin()};
  if (index < 0 || index >= static_cast<int>(inventory->items.size())) {
    return false;
  }
  inventory->items.erase(inventory->items.begin() + index);
  get_registry()->delete_game_object(item_id);
  get_registry()->notify_callbacks(EventType::InventoryUpdate, game_object_id);
  return true;
}

auto InventorySystem::use_item(const GameObjectID target_id, const GameObjectID item_id) const -> bool {
  bool used{false};
  if (auto *const registry{get_registry()}; registry->has_component(item_id, typeid(EffectApplier))) {
    used = registry->get_system<EffectSystem>()->apply_effects(item_id, target_id);
  }
  if (used) {
    [[maybe_unused]] const auto item{remove_item_from_inventory(target_id, item_id)};
  }
  return used;
}
