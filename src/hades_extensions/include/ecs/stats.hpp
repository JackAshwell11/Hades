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
  /// @param max_level - The maximum level of the stat.
  Stat(const double value, const int max_level) : value_(value), max_value_(value), max_level_(max_level) {}

  /// Reset the component to its default state.
  void reset() final;

  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file(nlohmann::json &json) const override = 0;

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file(const nlohmann::json &json) override = 0;

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

  /// Add a value to the maximum value of the stat.
  ///
  /// @param value - The value to add to the maximum value of the stat.
  void add_to_max_value(const double value) { max_value_ += value; }

  /// Get the current level of the stat.
  ///
  /// @return The current level of the stat.
  [[nodiscard]] auto get_current_level() const -> int { return current_level_; }

  /// Increment the current level of the stat.
  void increment_current_level() { current_level_++; }

  /// Get the maximum level of the stat.
  ///
  /// @return The maximum level of the stat.
  [[nodiscard]] auto get_max_level() const -> int { return max_level_; }

 protected:
  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file_base(nlohmann::json &json) const;

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file_base(const nlohmann::json &json);

 private:
  /// The current value of the variable.
  double value_;

  /// The maximum value of the stat.
  double max_value_;

  /// The current level of the stat.
  int current_level_{0};

  /// The maximum level of the stat.
  int max_level_;
};

/// Allows a game object to have an armour stat.
struct Armour final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the armour stat.
  /// @param maximum_level - The maximum level of the armour stat.
  Armour(const double value, const int maximum_level) : Stat(value, maximum_level) {}

  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file(nlohmann::json &json) const override;

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file(const nlohmann::json &json) override;
};

/// Allows a game object to have a health stat.
struct Health final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the health stat.
  /// @param maximum_level - The maximum level of the health stat.
  Health(const double value, const int maximum_level) : Stat(value, maximum_level) {}

  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file(nlohmann::json &json) const override;

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file(const nlohmann::json &json) override;
};
