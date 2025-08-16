// Related header
#include "ecs/systems/inventory.hpp"

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/physics.hpp"
#include "events.hpp"
#include "factories.hpp"

namespace {
/// The game objects that can be added to the inventory.
constexpr std::array COLLECTIBLE_TYPES{GameObjectType::HealthPotion};
}  // namespace

void Inventory::reset() { items.clear(); }

void Inventory::to_file(nlohmann::json &json, const Registry *registry) const {
  json["items"] = nlohmann::json::array();
  for (const auto item_id : items) {
    json.at("items").push_back(registry->get_game_object_type(item_id));
  }
}

void Inventory::from_file(const nlohmann::json &json, Registry *registry) {
  const auto player_id{registry->get_game_object_ids(GameObjectType::Player)[0]};
  for (const auto &item_type : json.at("items").items()) {
    const auto game_object_type{static_cast<GameObjectType>(item_type.value())};
    const auto game_object_id{
        registry->create_game_object(game_object_type, cpvzero, get_game_object_components(game_object_type))};
    registry->get_system<InventorySystem>()->add_item_to_inventory(player_id, game_object_id);
  }
}

void InventorySize::to_file(nlohmann::json &json) const { to_file_base(json["inventory_size"]); }

void InventorySize::from_file(const nlohmann::json &json) { from_file_base(json.at("inventory_size")); }

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
  if (get_registry()->has_component(item_id, typeid(EffectApplier))) {
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
