// Related header
#include "ecs/systems/inventory.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/physics.hpp"

namespace {
/// The game objects that can be added to the inventory.
constexpr std::array COLLECTIBLE_TYPES{GameObjectType::HealthPotion};
}  // namespace

auto InventorySystem::has_item_in_inventory(const GameObjectID game_object_id, const GameObjectID item_id) const
    -> bool {
  // Check if the item is a valid game object or not
  if (!get_registry()->has_game_object(item_id)) {
    return false;
  }

  // Check if the item is in the inventory or not
  const auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  return std::ranges::find(inventory->items.begin(), inventory->items.end(), item_id) != inventory->items.end();
}

void InventorySystem::add_item_to_inventory(const GameObjectID game_object_id, const GameObjectID item) const {
  // Check if the item is a valid game object or not
  if (!get_registry()->has_game_object(item)) {
    return;
  }

  // Check if the item is a collectible item or not
  if (const auto item_type{get_registry()->get_game_object_type(item)};
      std::ranges::find(COLLECTIBLE_TYPES, item_type) == COLLECTIBLE_TYPES.end()) {
    return;
  }

  // Check if the inventory is full or not
  const auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  if (const auto inventory_size{get_registry()->get_component<InventorySize>(game_object_id)};
      static_cast<int>(inventory->items.size()) == inventory_size->get_value()) {
    throw std::runtime_error("The inventory is full.");
  }

  // Add the item to the inventory, set the collected flag to prevent collision detection, and notify the callbacks
  inventory->items.push_back(item);
  if (get_registry()->has_component(item, typeid(KinematicComponent))) {
    get_registry()->get_component<KinematicComponent>(item)->collected = true;
  }
  get_registry()->notify<EventType::InventoryUpdate>(inventory->items);
  get_registry()->notify<EventType::SpriteRemoval>(item);
}

void InventorySystem::remove_item_from_inventory(const GameObjectID game_object_id, const GameObjectID item_id) const {
  // Check if the item is a valid game object or not
  if (!get_registry()->has_game_object(item_id)) {
    return;
  }

  // Check if the inventory is empty or not
  const auto inventory{get_registry()->get_component<Inventory>(game_object_id)};
  const auto index{std::ranges::find(inventory->items.begin(), inventory->items.end(), item_id) -
                   inventory->items.begin()};
  if (index < 0 || index >= static_cast<int>(inventory->items.size())) {
    return;
  }

  // Remove the item from the inventory, delete the game object, and notify the callbacks
  inventory->items.erase(inventory->items.begin() + index);
  get_registry()->delete_game_object(item_id);
  get_registry()->notify<EventType::InventoryUpdate>(inventory->items);
}
