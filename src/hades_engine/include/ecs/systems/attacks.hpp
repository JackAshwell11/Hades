// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>
#include <optional>
#include <vector>

// Local headers
#include "ecs/bases.hpp"
#include "game_object.hpp"

/// Stores the different types of attack effects.
enum class AttackEffect : std::uint8_t {
  Standard,
  Explosive,
  Poison,
};

/// Represents the base class for an attack
struct BaseAttack {
  /// The cooldown for this attack.
  double cooldown;

  /// The damage dealt by this attack.
  double damage;

  /// The range of this attack.
  double range;

  /// The time since this attack was last used.
  double time_since_last_use{0};

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  BaseAttack(const double cooldown, const double damage, const double range)
      : cooldown(cooldown), damage(damage), range(range) {}

  /// The virtual destructor.
  virtual ~BaseAttack() = default;

  /// The copy constructor.
  BaseAttack(const BaseAttack&) = default;

  /// The move constructor.
  BaseAttack(BaseAttack&&) = default;

  /// The copy assignment operator.
  auto operator=(const BaseAttack&) -> BaseAttack& = default;

  /// The move assignment operator.
  auto operator=(BaseAttack&&) -> BaseAttack& = default;

  /// Get the time until the attack can be used again.
  ///
  /// @return The time until the attack can be used again.
  [[nodiscard]] auto get_time_until_attack() const -> double { return std::max(0.0, cooldown - time_since_last_use); }

  /// Check if the attack is ready to be used or not.
  ///
  /// @return True if the attack is ready, false otherwise.
  [[nodiscard]] auto is_ready() const -> bool { return get_time_until_attack() <= 0.0; }

  /// Update the cooldown timer.
  ///
  /// @param delta_time - The time interval since the last update.
  void update(const double delta_time) { time_since_last_use += delta_time; }

  /// Perform the attack.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  virtual void perform_attack(const Registry* registry, GameObjectID game_object_id) const = 0;
};

/// Represents a ranged attack that can be performed.
struct RangedAttack : BaseAttack {
  /// The bullet velocity of this attack.
  double velocity;

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param velocity - The velocity of the bullet.
  RangedAttack(const double cooldown, const double damage, const double range, const double velocity)
      : BaseAttack(cooldown, damage, range), velocity(velocity) {}
};

/// Represents a ranged attack that only fires a single bullet.
struct SingleBulletAttack final : RangedAttack {
  /// Initialise the object
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param velocity - The velocity of the bullet.
  SingleBulletAttack(const double cooldown, const double damage, const double range, const double velocity)
      : RangedAttack(cooldown, damage, range, velocity) {}

  /// Perform the single bullet attack.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  void perform_attack(const Registry* registry, GameObjectID game_object_id) const override;
};

/// Represents a ranged attack that fires multiple bullets.
struct MultiBulletAttack final : RangedAttack {
  /// The number of bullets to fire.
  int bullet_count;

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param velocity - The velocity of the bullet.
  /// @param bullet_count - The number of bullets to fire.
  MultiBulletAttack(const double cooldown, const double damage, const double range, const double velocity,
                    const int bullet_count)
      : RangedAttack(cooldown, damage, range, velocity), bullet_count(bullet_count) {}

  /// Perform the multi bullet attack.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  void perform_attack(const Registry* registry, GameObjectID game_object_id) const override;
};

/// Allows a game object to act as a projectile that deals damage on impact.
struct Bullet final : ComponentBase {
  /// The damage dealt by this bullet.
  double damage;

  /// The type of the game object that created this bullet.
  GameObjectType source_type;
};

/// Allows a game object to attack other game objects.
struct Attack final : ComponentBase {
  /// The ranged attacks available to the game object.
  std::vector<std::unique_ptr<RangedAttack>> ranged_attacks;

  /// The index of the currently selected attack.
  int selected_ranged_attack{0};

  /// Initialise the object.
  ///
  /// @param ranged_attacks - The ranged attacks available to the game object.
  explicit Attack(std::vector<std::unique_ptr<RangedAttack>> ranged_attacks = {})
      : ranged_attacks(std::move(ranged_attacks)) {}

  /// Get the currently selected ranged attack.
  ///
  /// @return The currently selected ranged attack.
  [[nodiscard]] auto get_selected_ranged_attack() const -> RangedAttack* {
    return ranged_attacks[selected_ranged_attack].get();
  }
};

/// Provides facilities to manipulate attack components.
struct AttackSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit AttackSystem(Registry* registry) : SystemBase(registry) {}

  /// Process update logic for an attack component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;

  /// Switch to the previous ranged attack.
  ///
  /// @param game_object_id - The ID of the game object to switch to the previous ranged attack for.
  /// @throws RegistryError - If the game object does not exist or does not have an attack component.
  void previous_ranged_attack(GameObjectID game_object_id) const;

  /// Switch to the next ranged attack.
  ///
  /// @param game_object_id - The ID of the game object to switch to the next ranged attack for.
  /// @throws RegistryError - If the game object does not exist or does not have an attack component.
  void next_ranged_attack(GameObjectID game_object_id) const;

  /// Perform an attack if possible.
  ///
  /// @param game_object_id - The ID of the game object to perform the attack for.
  /// @throws RegistryError - If the game object does not exist or does not have an attack or kinematic component.
  [[nodiscard]] auto do_attack(GameObjectID game_object_id) const -> bool;
};

/// Provides facilities to damage game objects.
struct DamageSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit DamageSystem(Registry* registry) : SystemBase(registry) {}

  /// Deal damage to a game object.
  ///
  /// @param game_object_id - The game object ID to deal damage to.
  /// @param damage - The amount of damage to deal.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  void deal_damage(GameObjectID game_object_id, double damage) const;
};
