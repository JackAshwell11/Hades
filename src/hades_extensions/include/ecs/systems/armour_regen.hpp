// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/stats.hpp"

// ----- COMPONENTS ------------------------------
/// Allows a game object to regenerate armour.
struct ArmourRegen final : Stat {
  /// The time since the game object last regenerated armour.
  double time_since_armour_regen{0};

  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the armour regen stat.
  /// @param maximum_level - The maximum level of the armour regen stat.
  ArmourRegen(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate armour regen components.
struct ArmourRegenSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit ArmourRegenSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for an armour regeneration component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;
};
