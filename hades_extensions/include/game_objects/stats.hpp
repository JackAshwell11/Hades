// Ensure this file is only included once
#pragma once

// Std includes
#include <algorithm>
#include <limits>

// Custom includes
#include "game_objects/registry.hpp"

// ----- STRUCTURES ------------------------------
/// Represents a component that has a variable value and maximum value.
class Stat : public ComponentBase {
 public:
  /// The maximum value of the stat.
  double max_value;

  /// The current level of the stat.
  int current_level = 0;

  /// The maximum level of the stat.
  int maximum_level;

  /// Initialise the component.
  ///
  /// @param value - The initial and maximum value of the stat.
  /// @param maximum_level - The maximum level of the stat.
  /// @param max_value - Whether the stat has a maximum value or not.
  Stat(double value, int maximum_level, bool max_value = true)
      : value_(value),
        maximum_level(maximum_level),
        max_value(max_value ? value : std::numeric_limits<double>::infinity()) {}

  /// Get the value of the stat.
  ///
  /// @return The value of the stat.
  [[nodiscard]] inline double get_value() const {
    return value_;
  }

  /// Set the value of the stat.
  ///
  /// @param new_value - The new value of the stat.
  inline void set_value(double new_value) {
    value_ = std::max(std::min(new_value, max_value), 0.0);
  }

 protected:
  /// The current value of the variable.
  double value_;
};

// ----- COMPONENTS ------------------------------
/// Allows a game object to have an armour stat.
struct Armour : public Stat {
  /// Initialise the component.
  ///
  /// @param value - The initial and maximum value of the armour stat.
  /// @param maximum_level - The maximum level of the armour stat.
  Armour(double value, int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to regenerate armour.
struct ArmourRegen : public Stat {
  /// The time since the game object last regenerated armour.
  double time_since_armour_regen = 0;

  /// Initialise the component.
  ///
  /// @param value - The duration between armour regenerations.
  /// @param maximum_level - The maximum level of the armour regen stat.
  ArmourRegen(double value, int maximum_level) : Stat(value, maximum_level, false) {}
};

/// Allows a game object to have a health stat.
struct Health : public Stat {
  /// Initialise the component.
  ///
  /// @param value - The initial and maximum value of the health stat.
  /// @param maximum_level - The maximum level of the health stat.
  Health(double value, int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to determine how fast it can move.
struct MovementForce : public Stat {
  /// Initialise the component.
  ///
  /// @param force - The movement force of the game object.
  /// @param maximum_level - The maximum level of the movement force.
  MovementForce(double force, int maximum_level) : Stat(force, maximum_level, false) {}
};
