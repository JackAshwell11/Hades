// Ensure this file is only included once
#pragma once

// Std headers
#include <unordered_map>
#include <vector>

// Local headers
#include "ecs/bases.hpp"
#include "game_object.hpp"

/// Stores the different types of effects available.
enum class EffectType : std::uint8_t {
  Regeneration,
  Poison,
};

/// Represents the base class for an effect.
struct BaseEffect {
  /// The type of the effect.
  EffectType effect_type;

  /// The value that should be applied to the game object's stat.
  double value;

  /// Initialise the object.
  ///
  /// @param effect_type - The type of the effect.
  /// @param value - The value that should be applied to the game object's stat.
  BaseEffect(const EffectType effect_type, const double value) : effect_type(effect_type), value(value) {}

  /// The virtual destructor.
  virtual ~BaseEffect() = default;

  /// The copy constructor.
  BaseEffect(const BaseEffect&) = default;

  /// The move constructor.
  BaseEffect(BaseEffect&&) = default;

  /// The copy assignment operator.
  auto operator=(const BaseEffect&) -> BaseEffect& = default;

  /// The move assignment operator.
  auto operator=(BaseEffect&&) -> BaseEffect& = default;

  /// Apply the effect to the game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The ID of the game object to apply the effect to.
  /// @return Whether the effect was applied or not.
  virtual auto apply(const Registry* registry, GameObjectID game_object_id) const -> bool;
};

/// Represents an effect that can be applied immediately to a game object.
struct InstantEffect final : BaseEffect {
  /// Initialise the object.
  ///
  /// @param effect_type - The type of the status effect.
  /// @param value - The value that should be applied to the game object's stat.
  InstantEffect(const EffectType effect_type, const double value) : BaseEffect(effect_type, value) {}
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
  /// @param effect_type - The type of the status effect.
  /// @param value - The value that should be applied to the game object's stat.
  /// @param duration - The duration the status effect should be applied for.
  /// @param interval - The interval the status effect should be applied at.
  StatusEffect(const EffectType effect_type, const double value, const double duration, const double interval)
      : BaseEffect(effect_type, value), duration(duration), interval(interval) {}

  /// Apply the status effect to the game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param target - The ID of the game object to apply the status effect to.
  /// @param delta_time - The time interval since the last time the function was called.
  /// @return Whether the status effect has expired or not.
  auto update(const Registry* registry, GameObjectID target, double delta_time) -> bool;
};

/// Allows a game object to have status effects applied to it.
struct StatusEffects final : ComponentBase {
  /// The status effects currently applied to the game object.
  std::unordered_map<EffectType, StatusEffect> active_effects;
};

/// Allows a game object to provide instant or status effects.
struct EffectApplier final : ComponentBase {
  /// The instant effects the game object provides.
  std::vector<InstantEffect> instant_effects;

  /// The status effects the game object provides.
  std::vector<std::pair<EffectType, StatusEffect>> status_effects;

  /// Add an instant effect to the game object.
  ///
  /// @param effect_type - The type of instant effect.
  /// @param value - The value that should be applied to the game object's stat.
  void add_instant_effect(const EffectType effect_type, const double value) {
    instant_effects.emplace_back(effect_type, value);
  }

  /// Add a status effect to the game object.
  ///
  /// @param effect_type - The type of status effect.
  /// @param value - The value that should be applied to the game object's stat.
  /// @param duration - The duration the status effect should be applied for.
  /// @param interval - The interval the status effect should be applied at.
  void add_status_effect(const EffectType effect_type, const double value, const double duration,
                         const double interval) {
    status_effects.emplace_back(effect_type, StatusEffect{effect_type, value, duration, interval});
  }
};

/// Provides facilities to manipulate instant and status effects.
struct EffectSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit EffectSystem(Registry* registry) : SystemBase(registry) {}

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
