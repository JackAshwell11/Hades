// Ensure this file is only included once
#pragma once

// Local headers
#include "game_objects/registry.hpp"

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate armour regen components.
struct ArmourRegenSystem : public SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit ArmourRegenSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for an armour regeneration component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const final;
};
