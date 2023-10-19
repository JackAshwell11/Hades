// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/registry.hpp"

// ----- EXCEPTIONS ------------------------------
/// Thrown when there is a space problem with the inventory.
class InventorySpaceException : public std::runtime_error {
 public:
  /// Initialise the object.
  ///
  /// @param message - The message to display.
  explicit InventorySpaceException(const char *message) : std::runtime_error(message) {}

  /// Initialise the object.
  ///
  /// @param full - Whether the inventory is full or not.
  explicit InventorySpaceException(const bool full)
      : std::runtime_error(std::string("The inventory is ") + (full ? "full" : "empty") + ".") {}
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

  /// Initialise the object.
  ///
  /// @param width - The width of the inventory.
  /// @param height - The height of the inventory.
  Inventory(int width, int height) : width(width), height(height) {}

  /// Get the capacity of the inventory.
  ///
  /// @return The capacity of the inventory.
  [[nodiscard]] inline int get_capacity() const { return width * height; }
};

// ----- SYSTEMS --------------------------------
/// Provides facilities to manipulate inventory components.
struct InventorySystem : public SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit InventorySystem(Registry &registry) : SystemBase(registry) {}

  /// Add an item to the inventory of a game object.
  ///
  /// @param game_object_id - The ID of the game object to add the item to.
  /// @param item - The item to add to the inventory.
  /// @throws RegistryException - If the game object does not exist or does not have an inventory component.
  /// @throws InventorySpaceException - If the inventory is full.
  void add_item_to_inventory(GameObjectID game_object_id, GameObjectID item);

  /// Remove an item from the inventory of a game object.
  ///
  /// @param game_object_id - The ID of the game object to remove the item from.
  /// @param index - The index of the item to remove from the inventory.
  /// @throws RegistryException - If the game object does not exist or does not have an inventory component.
  /// @throws InventorySpaceException - If the inventory is empty or the index is out of bounds
  int remove_item_from_inventory(GameObjectID game_object_id, int index);
};
