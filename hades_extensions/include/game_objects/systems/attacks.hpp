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
  explicit AttackResult(const Vec2d &current_position, double x_velocity, double y_velocity) : ranged_attack(std::make_tuple(current_position, x_velocity, y_velocity)) {}
};

/// Provides facilities to manipulate attack components.
class AttackSystem : public SystemBase {
 public:
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit AttackSystem(Registry &registry) : SystemBase(registry) {}

  /// Performs the currently selected attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to perform the attack for.
  /// @param targets - The targets to attack.
  ///
  /// @return The result of the attack.
  AttackResult do_attack(int game_object_id, std::vector<int> &targets);

  /// Selects the previous attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  inline void previous_attack(int game_object_id) {
    auto attacks = registry.get_component<Attacks>(game_object_id);
    if (attacks->attack_state > 0) {
      attacks->attack_state--;
    }
  }

  /// Selects the next attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  inline void next_attack(int game_object_id) {
    auto attacks = registry.get_component<Attacks>(game_object_id);
    if (!attacks->attack_algorithms.empty() && attacks->attack_state < attacks->attack_algorithms.size() - 1) {
      attacks->attack_state++;
    }
  }
};
