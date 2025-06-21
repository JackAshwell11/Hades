// Ensure this file is only included once
#pragma once

// Std headers
#include <typeindex>
#include <unordered_map>
#include <vector>

// Local headers
#include "ecs/bases.hpp"
#include "game_object.hpp"

/// Stores the different types of status effects available.
enum class StatusEffectType : std::uint8_t {
  Regeneration,
  Poison,
};

/// Represents the base class for an effect.
struct BaseEffect {
  /// The value that should be applied to the game object's stat.
  double value;

  /// The component the effect should be applied to.
  std::type_index target_component;

  /// Initialise the object.
  ///
  /// @param value - The value that should be applied to the game object's stat.
  /// @param target_component - The component the effect should be applied to.
  BaseEffect(const double value, const std::type_index target_component)
      : value(value), target_component(target_component) {}

  /// The virtual destructor.
  virtual ~BaseEffect() = default;

  /// The copy constructor.
  BaseEffect(const BaseEffect &) = default;

  /// The move constructor.
  BaseEffect(BaseEffect &&) = default;

  /// The copy assignment operator.
  auto operator=(const BaseEffect &) -> BaseEffect & = default;

  /// The move assignment operator.
  auto operator=(BaseEffect &&) -> BaseEffect & = default;

  /// Apply the effect to the game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The ID of the game object to apply the effect to.
  /// @return Whether the effect was applied or not.
  virtual auto apply(const Registry *registry, GameObjectID game_object_id) const -> bool;
};

/// Represents an effect that can be applied immediately to a game object.
struct InstantEffect final : BaseEffect {
  /// Initialise the object.
  ///
  /// @param value - The value that should be applied to the game object's stat.
  /// @param target_component - The component the effect should be applied to.
  InstantEffect(const double value, const std::type_index target_component) : BaseEffect(value, target_component) {}
};

/// Represents a status effect that can be applied to a game object over time.
struct StatusEffect final : BaseEffect {
  /// The duration the status effect should be applied for.
  double duration;

  /// The interval the status effect should be applied at.
  double interval;

  /// The time the status effect has been applied for.
  double time_elapsed{0.0};

  /// The time left over from the last interval.
  double interval_accumulator{0.0};

  /// Initialise the object.
  ///
  /// @param value - The value that should be applied to the game object's stat.
  /// @param duration - The duration the status effect should be applied for.
  /// @param interval - The interval the status effect should be applied at.
  /// @param target_component - The component the status effect should be applied to.
  StatusEffect(const double value, const double duration, const double interval, const std::type_index target_component)
      : BaseEffect(value, target_component), duration(duration), interval(interval) {}

  /// Apply the status effect to the game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param target - The ID of the game object to apply the status effect to.
  /// @param delta_time - The time interval since the last time the function was called.
  /// @return Whether the status effect has expired or not.
  auto update(const Registry *registry, GameObjectID target, double delta_time) -> bool;
};

/// Allows a game object to have status effects applied to it.
struct StatusEffects final : ComponentBase {
  /// The status effects currently applied to the game object.
  std::unordered_map<StatusEffectType, StatusEffect> active_effects;
};

/// Allows a game object to provide instant or status effects.
struct EffectApplier final : ComponentBase {
  /// The instant effects the game object provides.
  std::vector<InstantEffect> instant_effects;

  /// The status effects the game object provides.
  std::vector<std::pair<StatusEffectType, StatusEffect>> status_effects;

  /// Add an instant effect to the game object.
  ///
  /// @param value - The value that should be applied to the game object's stat.
  /// @param target_component - The component the effect should be applied to.
  void add_instant_effect(const double value, const std::type_index target_component) {
    instant_effects.emplace_back(value, target_component);
  }

  /// Add a status effect to the game object.
  ///
  /// @param type - The type of status effect.
  /// @param value - The value that should be applied to the game object's stat.
  /// @param duration - The duration the status effect should be applied for.
  /// @param interval - The interval the status effect should be applied at.
  /// @param target_component - The component the status effect should be applied to.
  void add_status_effect(const StatusEffectType type, const double value, const double duration, const double interval,
                         const std::type_index target_component) {
    status_effects.emplace_back(type, StatusEffect{value, duration, interval, target_component});
  }
};

/// Provides facilities to manipulate instant and status effects.
struct EffectSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit EffectSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for a status effect component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;

  /// Apply effects to a game object.
  ///
  /// @param source - The ID of the game object to get the effects from.
  /// @param target - The ID of the game object to apply the effects to.
  /// @throws RegistryError - If either game object does not exist or does not have the required components.
  /// @return Whether the effects were applied or not.
  [[nodiscard]] auto apply_effects(GameObjectID source, GameObjectID target) const -> bool;
};
