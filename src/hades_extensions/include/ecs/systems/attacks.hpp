// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>

// Local headers
#include "ecs/stats.hpp"

/// Stores the different types of attacks available in the game.
enum class AttackType : std::uint8_t { Ranged, Melee, Special };

/// Represents the base class for an attack
struct BaseAttack {
  /// The cooldown for this attack.
  Stat cooldown;

  /// The damage dealt by this attack.
  Stat damage;

  /// The range of this attack.
  Stat range;

  /// The time since this attack was last used.
  double time_since_last_use{0};

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  BaseAttack(Stat cooldown, Stat damage, Stat range)
      : cooldown(std::move(cooldown)), damage(std::move(damage)), range(std::move(range)) {}

  /// The virtual destructor.
  virtual ~BaseAttack() = default;

  /// The copy constructor.
  BaseAttack(const BaseAttack &) = default;

  /// The move constructor.
  BaseAttack(BaseAttack &&) = default;

  /// The copy assignment operator.
  auto operator=(const BaseAttack &) -> BaseAttack & = default;

  /// The move assignment operator.
  auto operator=(BaseAttack &&) -> BaseAttack & = default;

  /// Get the time until the attack can be used again.
  ///
  /// @return The time until the attack can be used again.
  [[nodiscard]] auto get_time_until_attack() const -> double {
    return std::max(0.0, cooldown.get_value() - time_since_last_use);
  }

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
  virtual void perform_attack(const Registry *registry, GameObjectID game_object_id) const = 0;
};

/// Represents a ranged attack that can be performed.
struct RangedAttack : BaseAttack {
  /// The bullet velocity of this attack.
  Stat velocity;

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param velocity - The velocity of the bullet.
  RangedAttack(Stat cooldown, Stat damage, Stat range, Stat velocity)
      : BaseAttack(std::move(cooldown), std::move(damage), std::move(range)), velocity(std::move(velocity)) {}
};

/// Represents a ranged attack that only fires a single bullet.
struct SingleBulletAttack final : RangedAttack {
  /// Initialise the object
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param velocity - The velocity of the bullet.
  SingleBulletAttack(Stat cooldown, Stat damage, Stat range, Stat velocity)
      : RangedAttack(std::move(cooldown), std::move(damage), std::move(range), std::move(velocity)) {}

  /// Perform the single bullet attack.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  void perform_attack(const Registry *registry, GameObjectID game_object_id) const override;
};

/// Represents a ranged attack that fires multiple bullets.
struct MultiBulletAttack final : RangedAttack {
  /// The number of bullets to fire.
  Stat bullet_count;

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param velocity - The velocity of the bullet.
  /// @param bullet_count - The number of bullets to fire.
  MultiBulletAttack(Stat cooldown, Stat damage, Stat range, Stat velocity, Stat bullet_count)
      : RangedAttack(std::move(cooldown), std::move(damage), std::move(range), std::move(velocity)),
        bullet_count(std::move(bullet_count)) {}

  /// Perform the multi bullet attack.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  void perform_attack(const Registry *registry, GameObjectID game_object_id) const override;
};

/// Represents a melee attack that attacks in a cone in front of the game object.
struct MeleeAttack final : BaseAttack {
  /// The arc size of the melee attack.
  Stat size;

  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  /// @param size - The arc size of the melee attack.
  MeleeAttack(Stat cooldown, Stat damage, Stat range, Stat size)
      : BaseAttack(std::move(cooldown), std::move(damage), std::move(range)), size(std::move(size)) {}

  /// Perform the melee attack.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  void perform_attack(const Registry *registry, GameObjectID game_object_id) const override;
};

/// Represents an area of effect attack that attacks all targets in a radius.
struct AreaOfEffectAttack final : BaseAttack {
  /// Initialise the object.
  ///
  /// @param cooldown - The cooldown for this attack.
  /// @param damage - The damage dealt by this attack.
  /// @param range - The range of this attack.
  AreaOfEffectAttack(Stat cooldown, Stat damage, Stat range)
      : BaseAttack(std::move(cooldown), std::move(damage), std::move(range)) {}

  /// Perform an area of effect attack
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The game object ID of the attacking game object.
  void perform_attack(const Registry *registry, GameObjectID game_object_id) const override;
};

/// Allows a game object to act as a projectile that deals damage on impact.
struct Bullet final : ComponentBase {
  /// The damage dealt by this bullet.
  double damage{0};

  /// The type of the game object that created this bullet.
  GameObjectType source_type{GameObjectType::Player};
};

/// Allows a game object to attack other game objects.
struct Attack final : ComponentBase {
  /// The ranged attacks available to the game object.
  std::vector<std::unique_ptr<RangedAttack>> ranged_attacks;

  /// The melee attack available to the game object.
  std::optional<MeleeAttack> melee_attack;

  /// The special attacks available to the game object.
  std::optional<AreaOfEffectAttack> special_attack;

  /// The index of the currently selected attack.
  int selected_ranged_attack{0};

  /// Add a ranged attack.
  ///
  /// @param attack - The ranged attack to add.
  void add_ranged_attack(std::unique_ptr<RangedAttack> attack) { ranged_attacks.emplace_back(std::move(attack)); }

  /// Set the melee attack.
  ///
  /// @param attack - The melee attack to set.
  void set_melee_attack(MeleeAttack attack) { melee_attack = std::move(attack); }

  /// Set the special attack.
  ///
  /// @param attack - The special attack to set.
  void set_special_attack(AreaOfEffectAttack attack) { special_attack = std::move(attack); }

  /// Get the currently selected ranged attack.
  ///
  /// @return The currently selected ranged attack.
  [[nodiscard]] auto get_selected_ranged_attack() const -> RangedAttack * {
    return ranged_attacks[selected_ranged_attack].get();
  }
};

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
  /// @param attack_type - The type of attack to perform.
  /// @throws RegistryError - If the game object does not exist or does not have an attack or kinematic component.
  [[nodiscard]] auto do_attack(GameObjectID game_object_id, AttackType attack_type) const -> bool;
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
  /// @param damage - The amount of damage to deal.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  void deal_damage(GameObjectID game_object_id, double damage) const;
};
