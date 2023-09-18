// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate upgrades.
struct UpgradeSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit UpgradeSystem(Registry &registry) : SystemBase(registry) {}

  /// Upgrade a component to the next level if possible.
  ///
  /// @param game_object_id - The ID of the game object to upgrade.
  /// @param component_type - The type of component to upgrade.
  /// @return Whether the component upgrade was successful or not.
  bool upgrade_component(GameObjectID game_object_id, const std::type_index &component_type);
};
