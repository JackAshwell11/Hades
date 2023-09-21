// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/registry.hpp"

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

// ----- COMPONENTS ------------------------------
/// Allows a game object to have a fixed size inventory.
struct Inventory : public ComponentBase {
  /// The width of the inventory.
  int width;

  /// The height of the inventory.
  int height;

  /// The game object's inventory.
  std::vector<int> items;

  /// Initialise the component.
  ///
  /// @param width - The width of the inventory.
  /// @param height - The height of the inventory.
  Inventory(int width, int height) : width(width), height(height) {}

  /// Get the capacity of the inventory.
  ///
  /// @return The capacity of the inventory.
  [[nodiscard]] inline int capacity() const {
    return width * height;
  }
};

// ----- SYSTEMS --------------------------------
/// Provides facilities to manipulate inventory components.
struct InventorySystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit InventorySystem(Registry &registry) : SystemBase(registry) {}

  /// Add an item to the inventory of a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @param item - The item to add to the inventory.
  /// @throws InventorySpaceException - If the inventory is full.
  /// @throws RegistryException - If the game object does not exist or does not have an inventory component.
  void add_item_to_inventory(GameObjectID game_object_id, GameObjectID item);

  /// Remove an item from the inventory of a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @param index - The index of the item to remove.
  /// @throws InventorySpaceException - If the inventory is empty or the index is out of range.
  /// @throws RegistryException - If the game object does not exist or does not have an inventory component.
  int remove_item_from_inventory(GameObjectID game_object_id, int index);
};
