// Related header
#include "ecs/systems/inventory.hpp"

// Std headers
#include <utility>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/effects.hpp"
#include "events.hpp"

namespace {
/// The game objects that can be added to the inventory.
constexpr std::array COLLECTIBLE_TYPES{GameObjectType::HealthPotion};

/// The inventory size.
constexpr auto INVENTORY_SIZE{30};
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
  if (std::cmp_equal(inventory->items.size(), INVENTORY_SIZE)) {
    throw std::runtime_error("The inventory is full.");
  }

  // Add the item to the inventory, set the collected flag to prevent collision detection, and notify the callbacks
  inventory->items.push_back(item);
  if (get_registry()->has_component<KinematicComponent>(item)) {
    const auto kinematic_component{get_registry()->get_component<KinematicComponent>(item)};
    cpSpaceRemoveShape(get_registry()->get_space(), *kinematic_component->shape);
    cpSpaceRemoveBody(get_registry()->get_space(), *kinematic_component->body);
  }
  notify<EventType::InventoryUpdate>(inventory->items);
  notify<EventType::SpriteRemoval>(item);
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
  notify<EventType::InventoryUpdate>(inventory->items);
}

void InventorySystem::use_item(const GameObjectID target_id, const GameObjectID item_id) const {
  // Check if the item is a valid game object or not
  if (!get_registry()->has_game_object(target_id) || !get_registry()->has_game_object(item_id)) {
    return;
  }

  // Use the item if it can be used
  bool used{false};
  if (get_registry()->has_component<EffectApplier>(item_id)) {
    used = get_registry()->get_system<EffectSystem>()->apply_effects(item_id, target_id);
  }

  // If the item has been used, remove it from the inventory or the dungeon
  if (used) {
    if (has_item_in_inventory(target_id, item_id)) {
      remove_item_from_inventory(target_id, item_id);
    } else {
      get_registry()->delete_game_object(item_id);
    }
  }
}
