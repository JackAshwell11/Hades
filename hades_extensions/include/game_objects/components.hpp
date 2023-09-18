// Ensure this file is only included once
#pragma once

// Std includes
#include <algorithm>
#include <deque>
#include <functional>
#include <optional>
#include <typeindex>
#include <unordered_map>

// Custom includes
#include "registry.hpp"
#include "steering.hpp"

// ----- ENUMS ------------------------------
enum class AttackAlgorithms {
  /// Stores the different types of attack algorithms available.
  AreaOfEffect,
  Melee,
  Ranged,
};

enum class SteeringBehaviours {
  /// Stores the different types of steering behaviours available.
  Arrive,
  Evade,
  Flee,
  FollowPath,
  ObstacleAvoidance,
  Pursuit,
  Seek,
  Wander,
};

enum class SteeringMovementState {
  /// Stores the different states the steering movement component can be in.
  Default,
  Footprint,
  Target,
};

// TODO: Look at making all component (maybe all structs and classes) members
//  private

// ----- UTILITY STRUCTURES ------------------------------
/// Represents a status effect that can be applied to a game object.
struct StatusEffect {
  /// The value that should be applied to the game object temporarily.
  double value;

  /// The duration the status effect should be applied for.
  double duration;

  /// The original value of the game object component.
  double original_value;

  /// The original maximum value of the game object component.
  double original_max_value;

  /// The time counter for the status effect.
  double time_counter = 0;

  /// Initialise the status effect.
  ///
  /// @param value - The value that should be applied to the game object temporarily.
  /// @param duration - The duration the status effect should be applied for.
  /// @param original_value - The original value of the game object component.
  /// @param original_max_value - The original maximum value of the game object component.
  StatusEffect(double value, double duration, double original_value, double original_max_value)
      : value(value), duration(duration), original_value(original_value), original_max_value(original_max_value) {}
};

/// Holds the lambda functions for a single effect.
struct EffectFunction {
  /// The lambda function which calculates the value of the effect.
  std::function<double(int)> increase;

  /// The lambda function which calculates the duration of the effect.
  std::function<double(int)> duration;

  /// Initialise the object.
  ///
  /// @param increase - The lambda function which calculates the value of the effect.
  /// @param duration - The lambda function which calculates the duration of the effect.
  explicit EffectFunction(std::function<double(int)> increase,
                          std::function<double(int)> duration = [](int level) { return 0.0; }) : increase(std::move(
      increase)), duration(std::move(duration)) {}
};

// ----- COMPONENTS ------------------------------
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

/// Allows a game object to attack other game objects.
struct Attacks : public ComponentBase {
  /// The attack algorithms the game object can use.
  std::vector<AttackAlgorithms> attack_algorithms;

  /// The current state of the game object's attack.
  int attack_state = 0;

  /// Initialise the component.
  ///
  /// @param attack_algorithms - The attack algorithms the game object can use.
  explicit Attacks(std::vector<AttackAlgorithms> attack_algorithms) : attack_algorithms(std::move(attack_algorithms)) {}
};

/// Allows a game object to periodically leave footprints around the game map.
struct Footprints : public ComponentBase {
  /// The footprints the game object has left.
  std::deque<Vec2d> footprints = {};

  /// The time since the game object last left a footprint.
  double time_since_last_footprint = 0;
};

/// Allows a game object to have a health stat.
struct Health : public Stat {
  /// Initialise the component.
  ///
  /// @param value - The initial and maximum value of the health stat.
  /// @param maximum_level - The maximum level of the health stat.
  Health(double value, int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to provide instant or status effects.
struct EffectApplier : public ComponentBase {
  /// The instant effects the game object provides.
  std::unordered_map<std::type_index, EffectFunction> instant_effects;

  /// The status effects the game object provides.
  std::unordered_map<std::type_index, EffectFunction> status_effects;

  /// Initialise the component.
  ///
  /// @param instant_effects - The instant effects the game object provides.
  /// @param level_limit - The level limit of the instant effects.
  EffectApplier(std::unordered_map<std::type_index, EffectFunction> instant_effects,
                std::unordered_map<std::type_index, EffectFunction> status_effects) : instant_effects(std::move(
      instant_effects)), status_effects(std::move(status_effects)) {}
};

/// Allows a game object to have a fixed size inventory.
struct Inventory : public ComponentBase {
  /// The width of the inventory.
  int width;

  /// The height of the inventory.
  int height;

  /// The game object's inventory.
  std::vector<int> items;

  /// Initialise the component.
  ///
  /// @param width - The width of the inventory.
  /// @param height - The height of the inventory.
  Inventory(int width, int height) : width(width), height(height) {}

  /// Get the capacity of the inventory.
  ///
  /// @return The capacity of the inventory.
  [[nodiscard]] inline int capacity() const {
    return width * height;
  }
};

/// Allows a game object's movement to be controlled by the keyboard.
struct KeyboardMovement : public ComponentBase {
  /// Whether the game object is moving north or not.
  bool moving_north = false;

  /// Whether the game object is moving east or not.
  bool moving_east = false;

  /// Whether the game object is moving south or not.
  bool moving_south = false;

  /// Whether the game object is moving west or not.
  bool moving_west = false;
};

/// Allows a game object to record the amount of money it has.
struct Money : public ComponentBase {
  /// The amount of money the game object has.
  int money;

  /// Initialise the component.
  ///
  /// @param money - The amount of money the game object has.
  explicit Money(int money) : money(money) {}
};

/// Allows a game object to determine how fast it can move.
struct MovementForce : public Stat {
  /// Initialise the component.
  ///
  /// @param force - The movement force of the game object.
  /// @param maximum_level - The maximum level of the movement force.
  MovementForce(double force, int maximum_level) : Stat(force, maximum_level, false) {}
};

/// Allows a game object to have status effects applied to it.
struct StatusEffects : public ComponentBase {
  /// The status effects the game object provides.
  std::unordered_map<std::type_index, StatusEffect> status_effects = {};
};

/// Allows a game object's movement to be controlled by steering algorithms.
struct SteeringMovement : public ComponentBase {
  /// The steering behaviours used by the game object.
  std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> behaviours;

  /// The current movement state of the game object.
  SteeringMovementState movement_state = SteeringMovementState::Default;

  /// The game object ID of the target.
  int target_id = -1;

  /// The list of positions the game object should follow.
  std::vector<Vec2d> path_list;

  /// Initialise the component.
  ///
  /// @param behaviours - The steering behaviours used by the game object.
  explicit SteeringMovement(std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> behaviours)
      : behaviours(std::move(behaviours)) {}
};

/// Allows a game object to be upgraded.
struct Upgrades : public ComponentBase {
  /// The upgrades the game object has.
  std::unordered_map<std::type_index, std::function<double(int)>> upgrades;

  /// Initialise the component.
  ///
  /// @param upgrades - The upgrades the game object has.
  explicit Upgrades(std::unordered_map<std::type_index, std::function<double(int)>> upgrades) : upgrades(std::move(
      upgrades)) {}
};
