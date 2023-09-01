// Ensure this file is only included once
#pragma once

// Std includes
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

// ----- STRUCTURES ------------------------------
/// Represents a status effect that can be applied to a game object attribute.
struct StatusEffect {
  /// The value that should be applied to the game object temporarily.
  float value;

  /// The duration the status effect should be applied for.
  float duration;

  /// The original value of the game object attribute which is being changed.
  float original_value;

  /// The original maximum value of the game object attribute which is being changed.
  float original_max_value;

  /// The time counter for the status effect.
  float time_counter = 0;

  /// Initialise the object.
  ///
  /// @param value - The value that should be applied to the game object temporarily.
  /// @param duration - The duration the status effect should be applied for.
  /// @param original_value - The original value of the game object attribute which is being changed.
  /// @param original_max_value - The original maximum value of the game object attribute which is being changed.
  StatusEffect(float value, float duration, float original_value, float original_max_value)
      : value(value), duration(duration), original_value(original_value), original_max_value(original_max_value) {}
};

/// The base class for all game object attributes.
class GameObjectAttributeBase : ComponentBase {
 public:
  /// The level limit of the game object attribute.
  int level_limit;

  /// The maximum value of the game object attribute.
  float max_value;

  /// The current level of the game object attribute.
  int current_level = 0;

  /// The status effect currently applied to the game object.
  std::optional<StatusEffect> applied_status_effect;

  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the game object attribute.
  /// @param level_limit - The level limit of the game object attribute.
  GameObjectAttributeBase(float initial_value, int level_limit)
      : value_(initial_value),
        max_value(has_maximum() ? initial_value : std::numeric_limits<float>::infinity()),
        level_limit(level_limit) {}

  /// Get the game object attribute's value.
  ///
  /// @return The game object attribute's value.
  [[nodiscard]] float value() const;

  /// Set the game object attribute's value.
  ///
  /// @param new_value - The new game object attribute's value.
  void value(float new_value);

 private:
  /// The game object attribute's value.
  float value_;

  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] virtual bool has_instant_effect() const { return true; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] virtual bool has_maximum() const { return true; }

  /// Get if the game object attribute can have status effects or not.
  ///
  /// @return Whether the game object attribute can have status effects or not.
  [[nodiscard]] virtual bool has_status_effect() const { return true; }

  /// Get if the game object attribute can be upgraded or not.
  ///
  /// @return Whether the game object attribute can be upgraded or not.
  [[nodiscard]] virtual bool is_upgradable() const { return true; }
};

/// Allows a game object to have an armour attribute
class Armour : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the armour attribute.
  /// @param level_limit - The level limit of the armour attribute.
  Armour(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}
};

/// Allows a game object to regenerate armour.
struct ArmourRegen : ComponentBase {
  /// The time since the game object last regenerated armour.
  float time_since_armour_regen = 0;
};

/// Allows a game object to have an armour regen cooldown attribute.
class ArmourRegenCooldown : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the armour regen cooldown attribute.
  /// @param level_limit - The level limit of the armour regen cooldown attribute.
  ArmourRegenCooldown(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}

 private:
  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] bool has_instant_effect() const final { return false; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] bool has_maximum() const final { return false; }
};

/// Allows a game object to attack other game objects.
struct Attacks : ComponentBase {
  /// The attack algorithms the game object can use.
  std::vector<AttackAlgorithms> attack_algorithms;

  /// The current state of the game object's attack.
  int attack_state = 0;

  /// Initialise the object.
  ///
  /// @param attack_algorithms - The attack algorithms the game object can use.
  explicit Attacks(std::vector<AttackAlgorithms> attack_algorithms) : attack_algorithms(std::move(attack_algorithms)) {}
};

/// Allows a game object to have a fire rate penalty attribute.
class FireRatePenalty : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the fire rate penalty attribute.
  /// @param level_limit - The level limit of the fire rate penalty attribute.
  FireRatePenalty(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}

 private:
  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] bool has_instant_effect() const final { return false; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] bool has_maximum() const final { return false; }
};

/// Allows a game object to periodically leave footprints around the game map.
///
/// @param footprints - The footprints the game object has left.
struct Footprints : ComponentBase {
  /// The footprints the game object has left.
  std::vector<Vec2d> footprints;

  /// The time since the game object last left a footprint.
  float time_since_last_footprint = 0;

  /// Initialise the object.
  ///
  /// @param footprints - The footprints the game object has left.
  explicit Footprints(std::vector<Vec2d> footprints) : footprints(std::move(footprints)) {}
};

/// Allows a game object to have a health attribute.
class Health : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the health attribute.
  /// @param level_limit - The level limit of the health attribute.
  Health(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}
};

/// Allows a game object to provide instant effects.
struct InstantEffects : ComponentBase {
  /// The instant effects the game object provides.
  std::unordered_map<std::type_index, std::function<float(int)>> instant_effects;

  /// The level limit of the instant effects.
  int level_limit;

  /// Initialise the object.
  ///
  /// @param instant_effects - The instant effects the game object provides.
  /// @param level_limit - The level limit of the instant effects.
  InstantEffects(std::unordered_map<std::type_index, std::function<float(int)>> instant_effects, int level_limit)
      : instant_effects(std::move(instant_effects)), level_limit(level_limit) {}
};

/// Allows a game object to have a fixed size inventory.
struct Inventory : ComponentBase {
  /// The width of the inventory.
  int width;

  /// The height of the inventory.
  int height;

  /// The game object's inventory.
  std::vector<int> items;

  /// Initialise the object.
  ///
  /// @param width - The width of the inventory.
  /// @param height - The height of the inventory.
  Inventory(int width, int height) : width(width), height(height) {}
};

/// Allows a game object's movement to be controlled by the keyboard.
struct KeyboardMovement : ComponentBase {
  /// Whether the game object is moving north or not.
  bool moving_north = false;

  /// Whether the game object is moving east or not.
  bool moving_east = false;

  /// Whether the game object is moving south or not.
  bool moving_south = false;

  /// Whether the game object is moving west or not.
  bool moving_west = false;
};

/// Allows a game object to have a money attribute.
class Money : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the money attribute.
  /// @param level_limit - The level limit of the money attribute.
  Money(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}

 private:
  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] bool has_instant_effect() const final { return false; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] bool has_maximum() const final { return false; }

  /// Get if the game object attribute can have status effects or not.
  ///
  /// @return Whether the game object attribute can have status effects or not.
  [[nodiscard]] bool has_status_effect() const final { return false; }

  /// Get if the game object attribute can be upgraded or not.
  ///
  /// @return Whether the game object attribute can be upgraded or not.
  [[nodiscard]] bool is_upgradable() const final { return false; }
};

/// Allows a game object to have a movement force attribute.
class MovementForce : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the movement force attribute.
  /// @param level_limit - The level limit of the movement force attribute.
  MovementForce(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}

 private:
  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] bool has_instant_effect() const final { return false; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] bool has_maximum() const final { return false; }
};

/// Allows a game object to provide status effects.
struct StatusEffects : ComponentBase {
  /// The status effects the game object provides.
  std::unordered_map<std::type_index, std::function<StatusEffect(int)>> status_effects;

  /// The level limit of the status effects.
  int level_limit;

  /// Initialise the object.
  ///
  /// @param status_effects - The status effects the game object provides.
  /// @param level_limit - The level limit of the status effects.
  StatusEffects(std::unordered_map<std::type_index, std::function<StatusEffect(int)>> status_effects, int level_limit)
      : status_effects(std::move(status_effects)), level_limit(level_limit) {}
};

/// Allows a game object's movement to be controlled by steering algorithms.
struct SteeringMovement : ComponentBase {
  /// The steering behaviours used by the game object.
  std::vector<SteeringBehaviours> behaviours;

  /// The current movement state of the game object.
  SteeringMovementState movement_state = SteeringMovementState::Default;

  /// The game object ID of the target.
  int target_id = -1;

  /// The list of points the game object should follow.
  std::vector<Vec2d> path_list;

  /// Initialise the object.
  ///
  /// @param behaviours - The steering behaviours used by the game object.
  explicit SteeringMovement(std::vector<SteeringBehaviours> behaviours) : behaviours(std::move(behaviours)) {}
};

/// Allows a game object to have a view distance attribute.
class ViewDistance : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the view distance attribute.
  /// @param level_limit - The level limit of the view distance attribute.
  ViewDistance(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}

 private:
  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] bool has_instant_effect() const final { return false; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] bool has_maximum() const final { return false; }
};
