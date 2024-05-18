// Ensure this file is only included once
#pragma once

// Std headers
#include <cstdint>
#include <optional>

// Local headers
#include "game_objects/registry.hpp"

// ----- CONSTANTS ------------------------------
constexpr int DAMAGE{10};

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
  /// @return The result of the attack.
  [[nodiscard]] auto do_attack(GameObjectID game_object_id, const std::vector<int> &targets) const
      -> std::optional<GameObjectID>;

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
  /// @param damage - The amount of damage to deal to the game object.
  /// @throws RegistryError - If the game object does not exist or does not have health and armour components.
  void deal_damage(GameObjectID game_object_id, int damage) const;
};
