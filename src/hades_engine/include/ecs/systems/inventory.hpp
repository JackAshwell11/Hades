// Ensure this file is only included once
#pragma once

// Std headers
#include <vector>

// Local headers
#include "ecs/bases.hpp"
#include "game_object.hpp"

/// Allows a game object to have a fixed size inventory.
struct Inventory final : ComponentBase {
  /// The game object's inventory.
  std::vector<GameObjectID> items;
};

/// Provides facilities to manipulate inventory components.
struct InventorySystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit InventorySystem(Registry* registry) : SystemBase(registry) {}

  /// Check if a game object has an item in its inventory.
  ///
  /// @param game_object_id - The ID of the game object to check.
  /// @param item_id - The ID of the item to check for.
  /// @return True if the item is in the inventory, false otherwise.
  [[nodiscard]] auto has_item_in_inventory(GameObjectID game_object_id, GameObjectID item_id) const -> bool;

  /// Add an item to the inventory of a game object.
  ///
  /// @param game_object_id - The ID of the game object to add the item to.
  /// @param item - The item to add to the inventory.
  /// @throws runtime_error - If the inventory is full.
  void add_item_to_inventory(GameObjectID game_object_id, GameObjectID item) const;

  /// Remove an item from the inventory of a game object.
  ///
  /// @param game_object_id - The ID of the game object to remove the item from.
  /// @param item_id - The ID of the item to remove from the inventory.
  void remove_item_from_inventory(GameObjectID game_object_id, GameObjectID item_id) const;

  /// Use an item on a target game object.
  ///
  /// @param target_id - The game object ID of the target.
  /// @param item_id - The game object ID of the item to use.
  void use_item(GameObjectID target_id, GameObjectID item_id) const;
};
