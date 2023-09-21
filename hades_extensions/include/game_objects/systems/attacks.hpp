// Ensure this file is only included once
#pragma once

// Std includes
#include <optional>

// Custom includes
#include "game_objects/registry.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of attack algorithms available.
enum class AttackAlgorithms {
  AreaOfEffect,
  Melee,
  Ranged,
};

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
  explicit AttackResult(const Vec2d &current_position, double x_velocity, double y_velocity)
      : ranged_attack(std::make_tuple(current_position, x_velocity, y_velocity)) {}
};

// ----- COMPONENTS ------------------------------
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

// ----- SYSTEMS ------------------------------
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
  /// @throws RegistryException - If the game object does not exist or does not have an attack component.
  /// @return The result of the attack.
  AttackResult do_attack(int game_object_id, std::vector<int> &targets);

  /// Selects the previous attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  /// @throws RegistryException - If the game object does not exist or does not have an attack component.
  inline void previous_attack(int game_object_id) {
    auto attacks = registry.get_component<Attacks>(game_object_id);
    if (attacks->attack_state > 0) {
      attacks->attack_state--;
    }
  }

  /// Selects the next attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  /// @throws RegistryException - If the game object does not exist or does not have an attack component.
  inline void next_attack(int game_object_id) {
    auto attacks = registry.get_component<Attacks>(game_object_id);
    if (!attacks->attack_algorithms.empty() && attacks->attack_state < attacks->attack_algorithms.size() - 1) {
      attacks->attack_state++;
    }
  }
};

/// Provides facilities to damage game objects.
struct DamageSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit DamageSystem(Registry &registry) : SystemBase(registry) {}

  /// Deal damage to a game object.
  ///
  /// @param game_object_id - The game object ID to deal damage to.
  /// @param damage - The amount of damage to deal to the game object.
  /// @throws RegistryException - If the game object does not exist or does not have health and armour components.
  void deal_damage(GameObjectID game_object_id, int damage);
};
