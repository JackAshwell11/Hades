// Ensure this file is only included once
#pragma once

// Std headers
#include <algorithm>

// Local headers
#include "ecs/bases.hpp"

/// Represents a component that has a variable value and maximum value.
class Stat : public ComponentBase {
 public:
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the stat.
  explicit Stat(const double value) : value_(value), max_value_(value) {}

  /// Get the value of the stat.
  ///
  /// @return The value of the stat.
  [[nodiscard]] auto get_value() const -> double { return value_; }

  /// Set the value of the stat.
  ///
  /// @param new_value - The new value of the stat.
  void set_value(const double new_value) { value_ = std::clamp(new_value, 0.0, max_value_); }

  /// Get the maximum value of the stat.
  ///
  /// @return The maximum value of the stat.
  [[nodiscard]] auto get_max_value() const -> double { return max_value_; }

 private:
  /// The current value of the variable.
  double value_;

  /// The maximum value of the stat.
  double max_value_;
};

/// Allows a game object to have an armour stat.
struct Armour final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the armour stat.
  explicit Armour(const double value) : Stat(value) {}
};

/// Allows a game object to have a health stat.
struct Health final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the health stat.
  explicit Health(const double value) : Stat(value) {}
};
