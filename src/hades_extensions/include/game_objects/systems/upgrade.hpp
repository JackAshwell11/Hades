// Ensure this file is only included once
#pragma once

// Local headers
#include "game_objects/registry.hpp"

// ----- COMPONENTS ------------------------------
/// Allows a game object to record the amount of money it has.
struct Money : public ComponentBase {
  /// The amount of money the game object has.
  int money;

  /// Initialise the object.
  ///
  /// @param money - The amount of money the game object has.
  explicit Money(const int money) : money(money) {}
};

/// Allows a game object to be upgraded.
struct Upgrades : public ComponentBase {
  /// The upgrades the game object has.
  std::unordered_map<std::string, ActionFunction> upgrades;

  /// Initialise the object.
  ///
  /// @param upgrades - The upgrades the game object has.
  explicit Upgrades(const std::unordered_map<std::string, ActionFunction> &upgrades) : upgrades(upgrades) {}
};

/// Stores the identifier for the money component.
template <>
struct ComponentIdentifier<Money> {
  /// The identifier for the money component.
  static constexpr auto identifier{"Money"};
};

/// Stores the identifier for the upgrades component.
template <>
struct ComponentIdentifier<Upgrades> {
  /// The identifier for the upgrades component.
  static constexpr auto identifier{"Upgrades"};
};

// ----- SYSTEMS --------------------------------
/// Provides facilities to manipulate game object upgrades.
struct UpgradeSystem : public SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit UpgradeSystem(Registry *registry) : SystemBase(registry) {}

  /// Upgrade a component to the next level if possible.
  ///
  /// @param game_object_id - The ID of the game object to upgrade the component for.
  /// @param target_component - The type of component to upgrade.
  /// @throws RegistryError if the game object does not exist or does not have the target component.
  /// @return Whether the component upgrade was successful or not.
  [[nodiscard]] auto upgrade_component(GameObjectID game_object_id, const std::string &target_component) const -> bool;
};

/// Stores the identifier for the upgrade system.
template <>
struct SystemIdentifier<UpgradeSystem> {
  /// The identifier for the upgrade system.
  static constexpr auto identifier{"UpgradeSystem"};
};
