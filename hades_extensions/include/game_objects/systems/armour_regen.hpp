// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate armour regen components.
struct ArmourRegenSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit ArmourRegenSystem(Registry &registry) : SystemBase(registry) {}

  /// Process update logic for an armour regeneration component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) final;
};
