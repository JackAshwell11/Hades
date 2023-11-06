// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>

// Local headers
#include "game_objects/registry.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of attack algorithms available.
enum class AttackAlgorithms {
  AreaOfEffect,
  Melee,
  Ranged,
};

// ----- COMPONENTS ------------------------------
/// Allows a game object to attack other game objects.
struct Attacks : public ComponentBase {
  /// The attack algorithms the game object can use.
  std::vector<AttackAlgorithms> attack_algorithms;

  /// The current state of the game object's attack.
  int attack_state{0};

  /// Initialise the object.
  ///
  /// @param attack_algorithms - The attack algorithms the game object can use.
  explicit Attacks(const std::vector<AttackAlgorithms> &attack_algorithms) : attack_algorithms(attack_algorithms) {}
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate attack components.
struct AttackSystem : public SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit AttackSystem(Registry *registry) : SystemBase(registry) {}

  /// Perform the currently selected attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to perform the attack for.
  /// @param targets - The targets to attack.
  /// @throws RegistryException - If the game object does not exist or does not have an attack component.
  /// @return The result of the attack.
  [[nodiscard]] auto do_attack(int game_object_id, const std::vector<int> &targets) const
      -> std::optional<std::tuple<Vec2d, double, double>>;

  /// Select the previous attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  /// @throws RegistryException - If the game object does not exist or does not have an attack component.
  inline void previous_attack(const GameObjectID game_object_id) const {
    auto attacks{get_registry()->get_component<Attacks>(game_object_id)};
    if (attacks->attack_state > 0) {
      attacks->attack_state--;
    }
  }

  /// Select the next attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  /// @throws RegistryException - If the game object does not exist or does not have an attack component.
  inline void next_attack(const GameObjectID game_object_id) const {
    auto attacks{get_registry()->get_component<Attacks>(game_object_id)};
    if (!attacks->attack_algorithms.empty() && attacks->attack_state < attacks->attack_algorithms.size() - 1) {
      attacks->attack_state++;
    }
  }
};

/// Provides facilities to damage game objects.
struct DamageSystem : public SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit DamageSystem(Registry *registry) : SystemBase(registry) {}

  /// Deal damage to a game object.
  ///
  /// @param game_object_id - The game object ID to deal damage to.
  /// @param damage - The amount of damage to deal to the game object.
  /// @throws RegistryException - If the game object does not exist or does not have health and armour components.
  void deal_damage(GameObjectID game_object_id, int damage) const;
};
