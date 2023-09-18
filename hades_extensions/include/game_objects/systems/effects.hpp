// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate instant and status effects.
struct EffectSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit EffectSystem(Registry &registry) : SystemBase(registry) {}

  /// Process update logic for an effect component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) final;

  /// Apply an instant effect to a game object.
  ///
  /// @param game_object_id - The ID of the game object to apply the effect to.
  /// @param target_component - The component to apply the effect to.
  /// @param effect_function - The effect function to apply.
  /// @param level - The level of the effect to apply.
  void apply_instant_effect(GameObjectID game_object_id,
                            const std::type_index &target_component,
                            const EffectFunction &effect_function,
                            int level);

  /// Apply a status effect to a game object.
  ///
  /// @param game_object_id - The ID of the game object to apply the effect to.
  /// @param target_component - The component to apply the effect to.
  /// @param effect_function - The effect function to apply.
  /// @param level - The level of the effect to apply.
  void apply_status_effect(GameObjectID game_object_id,
                           const std::type_index &target_component,
                           const EffectFunction &effect_function,
                           int level);
};
