// Ensure this file is only included once
#pragma once

// Std headers
#include <algorithm>
#include <limits>

// Local headers
#include "game_objects/registry.hpp"

// ----- COMPONENTS ------------------------------
/// Represents a component that has a variable value and maximum value.
class Stat : public ComponentBase {
 public:
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the stat.
  /// @param maximum_level - The maximum level of the stat.
  Stat(const double value, const int maximum_level) : value_(value), maximum_level(maximum_level), max_value_(value) {}

  /// Get the value of the stat.
  ///
  /// @return The value of the stat.
  [[nodiscard]] inline auto get_value() const -> double { return value_; }

  /// Set the value of the stat.
  ///
  /// @param new_value - The new value of the stat.
  inline void set_value(const double new_value) { value_ = std::max(std::min(new_value, max_value_), 0.0); }

  /// Get the maximum value of the stat.
  ///
  /// @return The maximum value of the stat.
  [[nodiscard]] inline auto get_max_value() const -> double { return max_value_; }

  /// Add a value to the maximum value of the stat.
  ///
  /// @param value - The value to add to the maximum value of the stat.
  inline void add_to_max_value(const double value) { max_value_ += value; }

  /// Get the current level of the stat.
  ///
  /// @return The current level of the stat.
  [[nodiscard]] inline auto get_current_level() const -> int { return current_level; }

  /// Increment the current level of the stat.
  inline void increment_current_level() { current_level++; }

  /// Get the maximum level of the stat.
  ///
  /// @return The maximum level of the stat.
  [[nodiscard]] inline auto get_maximum_level() const -> int { return maximum_level; }

 private:
  /// The current value of the variable.
  double value_;

  /// The maximum value of the stat.
  double max_value_;

  /// The current level of the stat.
  int current_level{0};

  /// The maximum level of the stat.
  int maximum_level;
};

/// Allows a game object to have an armour stat.
struct Armour : public Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the armour stat.
  /// @param maximum_level - The maximum level of the armour stat.
  Armour(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to regenerate armour.
struct ArmourRegen : public Stat {
  /// The time since the game object last regenerated armour.
  double time_since_armour_regen{0};

  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the armour regen stat.
  /// @param maximum_level - The maximum level of the armour regen stat.
  ArmourRegen(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to have a health stat.
struct Health : public Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the health stat.
  /// @param maximum_level - The maximum level of the health stat.
  Health(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to determine how fast it can move.
struct MovementForce : public Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the movement force stat.
  /// @param maximum_level - The maximum level of the movement force stat.
  MovementForce(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};
