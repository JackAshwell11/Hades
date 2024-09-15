// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/stats.hpp"

/// Allows a game object to have a fixed size inventory.
struct Inventory final : ComponentBase {
  /// The game object's inventory.
  std::vector<int> items;
};

/// Allows a game object to change the size of its inventory.
struct InventorySize final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the inventory size stat.
  /// @param maximum_level - The maximum level of the inventory size stat.
  InventorySize(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Provides facilities to manipulate inventory components.
struct InventorySystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit InventorySystem(Registry *registry) : SystemBase(registry) {}

  /// Add an item to the inventory of a game object.
  ///
  /// @param game_object_id - The ID of the game object to add the item to.
  /// @param item - The item to add to the inventory.
  /// @throws RegistryError - If the game object does not exist or does not have an inventory component.
  /// @throws runtime_error - If the inventory is full.
  /// @return Whether the item was added or not.
  [[nodiscard]] auto add_item_to_inventory(GameObjectID game_object_id, GameObjectID item) const -> bool;

  /// Remove an item from the inventory of a game object.
  ///
  /// @param game_object_id - The ID of the game object to remove the item from.
  /// @param item_id - The ID of the item to remove from the inventory.
  /// @throws RegistryError - If the game object does not exist or does not have an inventory component.
  /// @return Whether the item was removed or not.
  [[nodiscard]] auto remove_item_from_inventory(GameObjectID game_object_id, GameObjectID item_id) const -> bool;

  /// Use an item from the inventory.
  ///
  /// @param target_id - The game object ID of the game object to use the item on.
  /// @param item_id - The game object ID of the item to use.
  /// @throws RegistryError - If the game object does not exist or if the required systems is not registered.
  /// @return Whether the item was used or not.
  [[nodiscard]] auto use_item(GameObjectID target_id, GameObjectID item_id) const -> bool;
};
