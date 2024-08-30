// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of attack algorithms available.
enum class AttackAlgorithm : std::uint8_t {
  AreaOfEffect,
  Melee,
  Ranged,
};

// ----- COMPONENTS ------------------------------
/// Allows a game object to attack other game objects.
struct Attack final : ComponentBase {
  /// The attack algorithms the game object can use.
  std::vector<AttackAlgorithm> attack_algorithms;

  /// The current state of the game object's attack.
  int attack_state{0};

  /// The time since the game object last attacked.
  double time_since_last_attack{0};

  /// Initialise the object.
  ///
  /// @param attack_algorithms - The attack algorithms the game object can use.
  explicit Attack(const std::vector<AttackAlgorithm> &attack_algorithms) : attack_algorithms(attack_algorithms) {}
};

/// Allows a game object to have an attack cooldown.
struct AttackCooldown final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the attack cooldown stat.
  /// @param maximum_level - The maximum level of the attack cooldown stat.
  AttackCooldown(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to have an attack range.
struct AttackRange final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the attack range stat.
  /// @param maximum_level - The maximum level of the attack range stat.
  AttackRange(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to deal damage to other game objects.
struct Damage final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the damage stat.
  /// @param maximum_level - The maximum level of the damage stat.
  Damage(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to have a melee attack size.
struct MeleeAttackSize final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the melee attack size stat.
  /// @param maximum_level - The maximum level of the melee attack size stat.
  MeleeAttackSize(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate attack components.
struct AttackSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit AttackSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for an attack component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;

  /// Perform the currently selected attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to perform the attack for.
  /// @param targets - The targets to attack.
  /// @throws RegistryError - If the game object does not exist or does not have an attack or kinematic component.
  void do_attack(GameObjectID game_object_id, const std::vector<int> &targets) const;

  /// Select the previous attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  /// @throws RegistryError - If the game object does not exist or does not have an attack component.
  void previous_attack(const GameObjectID game_object_id) const {
    if (const auto attack{get_registry()->get_component<Attack>(game_object_id)}; attack->attack_state > 0) {
      attack->attack_state--;
    }
  }

  /// Select the next attack algorithm.
  ///
  /// @param game_object_id - The ID of the game object to select the previous attack for.
  /// @throws RegistryError - If the game object does not exist or does not have an attack component.
  void next_attack(const GameObjectID game_object_id) const {
    if (const auto attack{get_registry()->get_component<Attack>(game_object_id)};
        !attack->attack_algorithms.empty() &&
        attack->attack_state < static_cast<int>(attack->attack_algorithms.size() - 1)) {
      attack->attack_state++;
    }
  }
};

/// Provides facilities to damage game objects.
struct DamageSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit DamageSystem(Registry *registry) : SystemBase(registry) {}

  /// Deal damage to a game object.
  ///
  /// @param game_object_id - The game object ID to deal damage to.
  /// @param attacker_id - The game object ID of the attacker.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  void deal_damage(GameObjectID game_object_id, GameObjectID attacker_id) const;
};
