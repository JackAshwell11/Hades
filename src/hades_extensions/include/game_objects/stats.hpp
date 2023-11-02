// Ensure this file is only included once
#pragma once

// Std headers
#include <algorithm>
#include <limits>

// Local headers
#include "game_objects/registry.hpp"

// ----- STRUCTURES ------------------------------
/// Represents a component that has a variable value and maximum value.
class Stat : public ComponentBase {
 public:
  /// The maximum value of the stat.
  double max_value;

  /// The current level of the stat.
  int current_level{0};

  /// The maximum level of the stat.
  int maximum_level{};

  /// Initialise the object.
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
  [[nodiscard]] inline double get_value() const { return value_; }

  /// Set the value of the stat.
  ///
  /// @param new_value - The new value of the stat.
  inline void set_value(double new_value) { value_ = std::max(std::min(new_value, max_value), 0.0); }

 private:
  /// The current value of the variable.
  double value_;
};

// ----- COMPONENTS ------------------------------
/// Allows a game object to have an armour stat.
struct Armour : public Stat {};

/// Allows a game object to regenerate armour.
struct ArmourRegen : public Stat {
  /// The time since the game object last regenerated armour.
  double time_since_armour_regen = 0;
};

/// Allows a game object to have a health stat.
struct Health : public Stat {};

/// Allows a game object to determine how fast it can move.
struct MovementForce : public Stat {};
