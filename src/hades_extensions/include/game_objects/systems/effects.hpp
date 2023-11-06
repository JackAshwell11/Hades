// Ensure this file is only included once
#pragma once

// Local headers
#include "game_objects/registry.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of status effects available.
enum class StatusEffectType {
  TEMP,
  TEMP2,
};

// ----- STRUCTURES ------------------------------
/// Represents a status effect that can be applied to a game object.
struct StatusEffect {
  /// The value that should be applied to the game object temporarily.
  double value;

  /// The duration the status effect should be applied for.
  double duration;

  /// The interval the status effect should be applied at.
  double interval;

  /// The component the status effect should be applied to.
  std::type_index target_component;

  /// Tracks the time the status effect has been applied for.
  double time_counter{0};

  /// Tracks the time left over from the last interval.
  double leftover_time{0};

  /// Initialise the object.
  ///
  /// @param value - The value that should be applied to the game object temporarily.
  /// @param duration - The duration the status effect should be applied for.
  /// @param interval - The interval the status effect should be applied at.
  /// @param target_component - The component the status effect should be applied to.
  StatusEffect(const double value, const double duration, const double interval, const std::type_index target_component)
      : value(value), duration(duration), interval(interval), target_component(target_component) {}
};

// ----- COMPONENTS ------------------------------
/// Represents the data required to apply a status effect.
struct StatusEffectData {
  /// The type of status effect.
  StatusEffectType status_effect_type;

  /// The increase function to apply.
  ActionFunction increase;

  /// The duration function to apply.
  ActionFunction duration;

  /// The interval function to apply.
  ActionFunction interval;
};

/// Allows a game object to provide instant or status effects.
struct EffectApplier : public ComponentBase {
  /// The instant effects the game object provides.
  std::unordered_map<std::type_index, ActionFunction> instant_effects;

  /// The status effects the game object provides.
  std::unordered_map<std::type_index, StatusEffectData> status_effects;

  /// Initialise the object.
  ///
  /// @param instant_effects - The instant effects the game object provides.
  /// @param status_effects - The status effects the game object provides.
  EffectApplier(const std::unordered_map<std::type_index, ActionFunction> &instant_effects,
                const std::unordered_map<std::type_index, StatusEffectData> &status_effects)
      : instant_effects(instant_effects), status_effects(status_effects) {}
};

/// Allows a game object to have status effects applied to it.
struct StatusEffects : public ComponentBase {
  /// The status effects currently applied to the game object.
  std::unordered_map<StatusEffectType, StatusEffect> applied_effects{};
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate instant and status effects.
struct EffectSystem : public SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit EffectSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for a status effect component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const final;

  /// Apply an instant effect to a game object.
  ///
  /// @param game_object_id - The ID of the game object to apply the effect to.
  /// @param target_component - The component to apply the effect to.
  /// @param increase_function - The increase function to apply.
  /// @param level - The level of the effect to apply.
  /// @throws RegistryException - If the game object does not exist or does not have the target component.
  /// @return Whether the instant effect was applied or not.
  auto apply_instant_effect(GameObjectID game_object_id, const std::type_index &target_component,
                            const ActionFunction &increase_function, int level) -> bool;

  /// Apply a status effect to a game object.
  ///
  /// @param game_object_id - The ID of the game object to apply the effect to.
  /// @param target_component - The component to apply the effect to.
  /// @param status_effect_data - The data required to apply the status effect.
  /// @param level - The level of the effect to apply.
  /// @throws RegistryException - If the game object does not exist or does not have the target component.
  /// @return Whether the status effect was applied or not.
  auto apply_status_effect(GameObjectID game_object_id, const std::type_index &target_component,
                           const StatusEffectData &status_effect_data, int level) -> bool;
};
