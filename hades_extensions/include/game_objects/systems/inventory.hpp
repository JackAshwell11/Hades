// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- EXCEPTIONS ------------------------------
/// Thrown when there is a space problem with the inventory.
class InventorySpaceException : public std::runtime_error {
 public:
  /// Initialise the exception.
  ///
  /// @param message - The message to display.
  explicit InventorySpaceException(const char *message) : std::runtime_error(message) {}

  /// Initialise the exception.
  ///
  /// @param full - Whether the inventory is full or not.
  explicit InventorySpaceException(const bool full) : std::runtime_error(
      std::string("The inventory is ") + (full ? "full" : "empty") + ".") {}
};

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate inventory components.
struct InventorySystem : public SystemBase {
  /// Add an item to the inventory of a game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID.
  /// @param item - The item to add to the inventory.
  /// @throws InventorySpaceException - If the inventory is full.
  static void add_item_to_inventory(Registry &registry, GameObjectID game_object_id, GameObjectID item);

  /// Remove an item from the inventory of a game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID.
  /// @param index - The index of the item to remove.
  /// @throws InventorySpaceException - If the inventory is empty or the index is out of range.
  static int remove_item_from_inventory(Registry &registry, GameObjectID game_object_id, int index);
};
