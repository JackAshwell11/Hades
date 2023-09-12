// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Holds the result of an attack.
struct AttackResult {
/// The result of a ranged attack.
  std::optional<std::tuple<Vec2d, double, double>> ranged_attack;

  /// The default constructor.
  AttackResult() = default;

  /// Initialise the object.
  ///
  /// @param ranged_attack_result - The result of a ranged attack.
  explicit AttackResult(Vec2d &current_position, double x_velocity, double y_velocity) : ranged_attack(std::make_tuple(
      current_position,
      x_velocity,
      y_velocity)) {}
};

/// Provides facilities to manipulate attack components.
class AttackSystem : public SystemBase {
 public:
  /// Performs the currently selected attack algorithm.
  ///
  /// @param registry - The registry to perform the attack for.
  /// @param game_object_id - The ID of the game object to perform the attack for.
  /// @param targets - The targets to attack.
  ///
  /// @return The result of the attack.
  static AttackResult do_attack(Registry &registry, int game_object_id, std::vector<int> &targets);

  /// Selects the previous attack algorithm.
  ///
  /// @param registry - The registry to select the previous attack for.
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  static inline void previous_attack(Registry &registry, int game_object_id) {
    auto *attacks = registry.get_component<Attacks>(game_object_id);
    if (attacks->attack_state > 0) {
      attacks->attack_state--;
    }
  }

  /// Selects the next attack algorithm.
  ///
  /// @param registry - The registry to select the next attack for.
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  static inline void next_attack(Registry &registry, int game_object_id) {
    auto *attacks = registry.get_component<Attacks>(game_object_id);
    if (attacks->attack_state < attacks->attack_algorithms.size() - 1) {
      attacks->attack_state++;
    }
  }

 private:
  /// Performs an area of effect attack around the game object.
  ///
  /// @param registry - The registry to perform the attack for.
  /// @param current_position - The current position of the game object.
  /// @param targets - The targets to attack.
  static void area_of_effect_attack(Registry &registry, Vec2d &current_position, std::vector<int> &targets);

  /// Performs a melee attack in the direction the game object is facing.
  ///
  /// @param registry - The registry to perform the attack for.
  /// @param current_position - The current position of the game object.
  /// @param current_rotation - The current rotation of the game object in radians.
  /// @param targets - The targets to attack.
  static void melee_attack(Registry &registry,
                           Vec2d &current_position,
                           double current_rotation,
                           std::vector<int> &targets);

  /// Performs a ranged attack in the direction the game object is facing.
  ///
  /// @param current_position - The current position of the game object.
  /// @param current_rotation - The current rotation of the game object in radians.
  ///
  /// @return The result of the attack.
  static AttackResult ranged_attack(Vec2d &current_position, double current_rotation);
};
